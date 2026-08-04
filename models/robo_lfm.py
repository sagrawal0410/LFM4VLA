"""
LFM2.5-VL robotics policy adapter.

Preprocessing contract (matches BaseTrainer + RoboVLMBackbone pipeline):
  - Dataloader: use ``image_processor`` → uint8 CHW tensors in [0, 255], no normalization.
  - Trainer/dataset: call ``process_vision_info`` then ``build_processor_inputs`` to produce
    the processor dict consumed by ``forward_continuous`` as ``lang_x``.
  - Forward: placeholder-token fusion via ``masked_scatter`` (not PaliGemma-style concat).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from models.model_backbone import RoboVLMBackbone

ImageInput = Union[Image.Image, torch.Tensor, np.ndarray]


class RoboLFM25VL(RoboVLMBackbone):
    """LFM2.5-VL VLA adapter.

    Processor outputs expected in ``lang_x``:
      - input_ids
      - attention_mask
      - pixel_values
      - spatial_shapes
      - pixel_attention_mask
    """

    # LFM vision defaults (processor may override at runtime).
    MIN_IMAGE_TOKENS = 64
    MAX_IMAGE_TOKENS = 256
    DO_IMAGE_SPLITTING = True

    # ------------------------------------------------------------------
    # Model structure accessors (Lfm2VlForConditionalGeneration layout)
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        cfg = self.model.config
        if hasattr(cfg, "text_config"):
            return cfg.text_config.hidden_size
        return cfg.hidden_size

    @property
    def word_embedding(self):
        return self.model.get_input_embeddings()

    @property
    def text_tower(self):
        return self.model.model.language_model

    @property
    def vision_tower(self):
        return self.model.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.model.multi_modal_projector

    @property
    def model(self):
        return self.backbone

    @property
    def image_token_id(self) -> int:
        return self.model.config.image_token_id

    @property
    def start_image_token_id(self):
        raise NotImplementedError("LFM2.5-VL uses image_token_id placeholders, not start/end tokens.")

    @property
    def end_image_token_id(self):
        raise NotImplementedError("LFM2.5-VL uses image_token_id placeholders, not start/end tokens.")

    # ------------------------------------------------------------------
    # Preprocessing (dataloader + trainer)
    # ------------------------------------------------------------------

    @property
    def image_processor(self):
        """Convert PIL images to float CHW tensors at native resolution (no resize/normalize)."""

        def _pil_to_chw(img: Image.Image) -> torch.Tensor:
            img = img.convert("RGB")
            return torch.from_numpy(np.array(img, copy=False)).permute(2, 0, 1).float()

        return _pil_to_chw

    def process_vision_info(self, images: Sequence[ImageInput]) -> List[Image.Image]:
        """Convert dataloader tensors to PIL images for the LFM processor."""
        pil_images: List[Image.Image] = []
        for image in images:
            if isinstance(image, Image.Image):
                pil_images.append(image.convert("RGB"))
            elif isinstance(image, torch.Tensor):
                arr = image.detach().cpu()
                if arr.dtype.is_floating_point and arr.max() <= 1.0:
                    arr = (arr * 255.0).clamp(0, 255)
                arr = arr.to(torch.uint8)
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = arr.permute(1, 2, 0)
                pil_images.append(Image.fromarray(arr.numpy()).convert("RGB"))
            elif isinstance(image, np.ndarray):
                pil_images.append(Image.fromarray(image).convert("RGB"))
            else:
                raise TypeError(f"Unsupported vision input type: {type(image)}")
        return pil_images

    @staticmethod
    def build_conversation(image: Image.Image, instruction: str) -> List[Dict[str, Any]]:
        """Single user turn with one image + instruction (LFM chat format)."""
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }]

    def build_processor_inputs(
        self,
        texts: Sequence[str],
        images: Sequence[ImageInput],
        *,
        padding: bool = True,
        add_generation_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Build the processor batch dict passed to ``forward_continuous`` as ``lang_x``.

        Prefer this over calling ``self.processor(...)`` directly so LFM chat-template
        and vision token constraints stay centralized here.
        """
        pil_images = self.process_vision_info(images)
        if len(texts) != len(pil_images):
            raise ValueError(
                f"text/image count mismatch: {len(texts)} instructions vs {len(pil_images)} images")

        conversations = [
            self.build_conversation(img, text)
            for img, text in zip(pil_images, texts)
        ]

        # apply_chat_template is the canonical LFM2.5-VL preprocessing path.
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=padding,
        )
        return dict(inputs)

    def encode_images(self, images, image_sizes=None):
        raise NotImplementedError(
            "LFM2.5-VL fuses images via processor placeholders; use build_processor_inputs instead.")

    def model_encode_images(self, images):
        raise NotImplementedError("LFM2.5-VL does not expose a standalone model_encode_images path.")

    # ------------------------------------------------------------------
    # Multimodal fusion helpers
    # ------------------------------------------------------------------

    def _pop_processor_batch(self, lang_x: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """Extract and remove LFM processor tensors from ``lang_x`` (mutates dict)."""
        input_ids = lang_x.pop("input_ids")
        attention_mask = lang_x.pop("attention_mask")
        pixel_values = lang_x.pop("pixel_values")
        spatial_shapes = lang_x.pop("spatial_shapes")
        pixel_attention_mask = lang_x.pop("pixel_attention_mask")

        vision_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)

        return input_ids, attention_mask, pixel_values, spatial_shapes, pixel_attention_mask

    def _fuse_image_features(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        *,
        return_image_tokens: bool = False,
    ):
        """Scatter projected vision features into ``image_token_id`` placeholder positions.

        When ``return_image_tokens=True``, also returns per-sample image token features
        ``[B, Ni_max, D]`` (zero-padded) and a boolean mask ``[B, Ni_max]`` for the QFormer.
        """
        image_outputs = self.model.get_image_features(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            return_dict=True,
        )
        # pooler_output is typically a list/tuple of [Ni_i, D] (one per image).
        pooler = image_outputs.pooler_output
        if isinstance(pooler, (list, tuple)):
            per_image = [
                feat.to(device=input_embeds.device, dtype=input_embeds.dtype)
                for feat in pooler
            ]
        else:
            # Fallback: already concatenated [sum Ni, D] — treat as one block per batch row
            # only when lengths are uniform (split equally).
            flat = pooler.to(device=input_embeds.device, dtype=input_embeds.dtype)
            bs_flat = input_embeds.shape[0]
            if flat.shape[0] % bs_flat != 0:
                raise ValueError(
                    f"Cannot split flat image features {flat.shape[0]} across batch {bs_flat}"
                )
            n_each = flat.shape[0] // bs_flat
            per_image = list(flat.split(n_each, dim=0))
        image_features = torch.cat(per_image, dim=0)

        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        n_image_features = image_features.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={n_image_features}")

        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(input_embeds)
        fused = input_embeds.masked_scatter(image_mask.to(input_embeds.device), image_features)

        if not return_image_tokens:
            return fused

        # Pack variable-length per-image features into a padded batch for the QFormer.
        bs = input_embeds.shape[0]
        if len(per_image) != bs:
            # Multiple images per sample is unsupported for depth fusion currently.
            raise ValueError(
                f"Depth QFormer expects one image feature tensor per batch row "
                f"(got {len(per_image)} feature groups for batch {bs})."
            )
        max_n = max(t.shape[0] for t in per_image)
        dim = per_image[0].shape[-1]
        packed = image_features.new_zeros(bs, max_n, dim)
        packed_mask = torch.zeros(bs, max_n, dtype=torch.bool, device=image_features.device)
        for i, feat in enumerate(per_image):
            n = feat.shape[0]
            packed[i, :n] = feat
            packed_mask[i, :n] = True
        return fused, packed, packed_mask

    def _resolve_depth_maps(
        self,
        vision_x: torch.Tensor,
        depth: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return LIBERO GT depth maps ``[B*T, 1, H, W]`` from the batch."""
        bs, seq_len = vision_x.shape[:2]
        if depth is None:
            raise ValueError(
                "use_depth=True but no depth tensor was provided. "
                "Training must load LIBERO depth maps; eval must pass agentview_depth."
            )
        if depth.ndim == 5:
            depth = rearrange(depth, "b t c h w -> (b t) c h w")
        elif depth.ndim == 4 and depth.shape[1] != 1:
            # [B, T, H, W]
            depth = rearrange(depth, "b t h w -> (b t) 1 h w")
        elif depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[0] != bs * seq_len:
            raise ValueError(
                f"depth batch {depth.shape[0]} != bs*seq_len {bs * seq_len}"
            )
        return depth.to(device=vision_x.device, dtype=torch.float32)

    def _insert_depth_tokens(
        self,
        multimodal_embeds: torch.Tensor,
        multimodal_attention_mask: Optional[torch.Tensor],
        depth_tokens: torch.Tensor,
    ):
        """Append QFormer depth tokens just before the (soon-to-be-added) action tokens."""
        return self.merge_multi_modal_input(
            multimodal_embeds,
            depth_tokens,
            labels=None,
            attention_mask=multimodal_attention_mask,
            is_image=False,
            insert_idx=multimodal_embeds.shape[1],
            fill_zero=False,
        )[:3]  # embeds, labels, attn_mask (drop insert_mask)

    @staticmethod
    def _format_loss(loss: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        total = sum(v for k, v in loss.items() if "loss" in k and v is not None)
        loss["loss"] = total
        return loss

    # ------------------------------------------------------------------
    # Forward (processor dict in, action loss dict out)
    # ------------------------------------------------------------------

    def forward_continuous(
        self,
        vision_x: torch.Tensor,
        lang_x: Dict[str, Any],
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        action_labels: Tuple[torch.Tensor, Optional[torch.Tensor]] = None,
        action_mask: torch.Tensor = None,
        vision_gripper=None,
        raw_text=None,
        rel_state=None,
        depth=None,
        mode: str = "train",
        **kwargs,
    ):
        loss: Dict[str, Any] = {}
        assert vision_x is not None
        bs, seq_len = vision_x.shape[:2]

        action_space = self.act_head_configs.get("action_space", "continuous")
        history_type = self.act_head_configs.get("history_type", "post")
        assert history_type in ("post", "pre")

        if not isinstance(lang_x, dict):
            raise TypeError(
                "RoboLFM25VL expects lang_x to be a processor dict. "
                "Run build_processor_inputs() in the trainer or dataset.")

        input_ids, attention_mask, pixel_values, spatial_shapes, pixel_attention_mask = (
            self._pop_processor_batch(lang_x)
        )
        assert input_ids.shape[0] == bs * seq_len, (
            f"batch size mismatch: input_ids {input_ids.shape[0]} vs bs*seq_len {bs * seq_len}")

        input_embeds = self.word_embedding(input_ids)
        if self.use_depth and self.depth_conditioner is not None:
            input_embeds, image_tokens, image_tok_mask = self._fuse_image_features(
                input_ids,
                input_embeds,
                pixel_values,
                spatial_shapes,
                pixel_attention_mask,
                return_image_tokens=True,
            )
        else:
            input_embeds = self._fuse_image_features(
                input_ids,
                input_embeds,
                pixel_values,
                spatial_shapes,
                pixel_attention_mask,
            )
            image_tokens = None
            image_tok_mask = None

        multimodal_embeds = input_embeds
        multimodal_labels = None
        multimodal_attention_mask = attention_mask

        if vision_gripper is not None:
            raise NotImplementedError("hand_rgb / vision_gripper is not supported for LFM2.5-VL yet.")

        if rel_state is not None and self.use_state:
            raise NotImplementedError("rel_state conditioning is not implemented for LFM2.5-VL yet.")

        # Depth CNN → QFormer (cross-attn over depth / image / text) → insert tokens.
        if self.use_depth and self.depth_conditioner is not None:
            depth_maps = self._resolve_depth_maps(vision_x, depth)
            depth_maps = torch.nan_to_num(depth_maps.float(), nan=0.0, posinf=0.0, neginf=0.0)
            depth_maps = depth_maps.clamp(0.0, 1.0)
            # Text tokens = non-image embeddings already in the sequence (instruction).
            text_mask = (input_ids != self.image_token_id)
            if multimodal_attention_mask is not None:
                text_mask = text_mask & multimodal_attention_mask.bool()
            depth_tokens = self.depth_conditioner(
                depth=depth_maps.to(device=multimodal_embeds.device),
                image_tokens=image_tokens.to(dtype=multimodal_embeds.dtype),
                text_tokens=multimodal_embeds,
                text_mask=text_mask,
                image_mask=image_tok_mask,
            )
            depth_tokens = depth_tokens.to(dtype=multimodal_embeds.dtype)
            if not torch.isfinite(depth_tokens).all():
                depth_tokens = torch.nan_to_num(depth_tokens, nan=0.0, posinf=0.0, neginf=0.0)
            multimodal_embeds, multimodal_labels, multimodal_attention_mask = (
                self._insert_depth_tokens(
                    multimodal_embeds, multimodal_attention_mask, depth_tokens
                )
            )

        action_token_mask = None
        depth_pred_token_mask = None
        if action_space == "continuous":
            if mode not in ("train", "val"):
                # Keep parameter dtype in sync with the backbone for inference.
                model_dtype = next(self.model.parameters()).dtype
                if self.action_token.dtype != model_dtype:
                    self.action_token.data = self.action_token.data.to(dtype=model_dtype)
                if self.depth_pred_token is not None and self.depth_pred_token.dtype != model_dtype:
                    self.depth_pred_token.data = self.depth_pred_token.data.to(
                        dtype=model_dtype
                    )
            action_tokens = self._expand_action_tokens(multimodal_embeds.shape[0])
            (
                multimodal_embeds,
                multimodal_labels,
                multimodal_attention_mask,
                action_token_mask,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                action_tokens,
                multimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                insert_idx=multimodal_embeds.shape[1],
                fill_zero=self.act_head_configs.get("fill_zero", False),
            )
            # Append learnable depth-prediction queries after action queries.
            if self.predict_depth and self.depth_pred_token is not None:
                depth_pred_tokens = self._expand_depth_pred_tokens(
                    multimodal_embeds.shape[0]
                )
                (
                    multimodal_embeds,
                    multimodal_labels,
                    multimodal_attention_mask,
                    depth_pred_token_mask,
                ) = self.merge_multi_modal_input(
                    multimodal_embeds,
                    depth_pred_tokens,
                    multimodal_labels,
                    multimodal_attention_mask,
                    is_image=False,
                    insert_idx=multimodal_embeds.shape[1],
                    fill_zero=False,
                )
                # Depth tokens are appended at the end, so action indices are
                # unchanged — pad the action mask with False for the new slots.
                n_depth = depth_pred_tokens.shape[1]
                action_token_mask = torch.cat(
                    [
                        action_token_mask,
                        action_token_mask.new_zeros(
                            action_token_mask.shape[0], n_depth, dtype=torch.bool
                        ),
                    ],
                    dim=1,
                )

        if history_type == "pre":
            multimodal_embeds = rearrange(multimodal_embeds, "(b l) n d -> b (l n) d", l=seq_len)
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(
                    multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len)

        if mode not in ("train", "val"):
            model_dtype = next(self.model.parameters()).dtype
            if multimodal_embeds.dtype != model_dtype:
                multimodal_embeds = multimodal_embeds.to(dtype=model_dtype)

        output = self.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeds,
            use_cache=False,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()
        if history_type == "pre":
            output_hs = rearrange(output_hs, "b (l n) d -> (b l) n d", l=seq_len)

        depth_hs = None
        if action_space == "continuous":
            action_hs = output_hs[action_token_mask].reshape(bs, seq_len, self.latent_num, -1)
            if depth_pred_token_mask is not None:
                depth_hs = output_hs[depth_pred_token_mask].reshape(
                    bs, seq_len, self.depth_latent_num, -1
                )
        elif action_space == "down_sample":
            token_src = self.act_head_configs.get("token_source", "all")
            if token_src != "all":
                raise ValueError(f"Unsupported token source {token_src}")
            action_hs = output_hs.reshape(bs, seq_len, *output_hs.shape[1:])
        else:
            raise ValueError(f"Unsupported action space {action_space}")

        if self.use_clip_norm and mode == "train":
            clip_loss = self.clip_norm_head(action_hs, raw_text)
            self._update_loss(loss, clip_loss, "clip")

        if mode not in ("train", "val") and self.act_head is not None:
            head_dtype = next(self.act_head.parameters()).dtype
            action_hs = action_hs.to(dtype=head_dtype)
            if depth_hs is not None:
                depth_hs = depth_hs.to(dtype=head_dtype)

        head_kwargs = {}
        if depth_hs is not None:
            head_kwargs["depth_hs"] = depth_hs
        # GT depth for aux loss (also used as conditioner input when use_depth=True).
        if self.predict_depth and depth is not None:
            head_kwargs["depth_gt"] = depth
        elif self.predict_depth and mode in ("train", "val") and action_labels is not None:
            raise ValueError(
                "predict_depth=True requires GT depth in the batch. "
                "Set train_dataset.load_depth=true and point data_root_dir at "
                "modified_libero_rlds_depth."
            )

        action_logits, action_loss = self.forward_action_head(
            action_hs, action_labels, action_mask, **head_kwargs
        )

        if mode in ("train", "val") and action_labels is not None:
            if action_loss is not None and action_loss.get("loss_depth") is not None:
                ratio = float(self.configs.get("depth_loss_ratio", 0.1))
                action_loss = dict(action_loss)
                action_loss["loss_depth"] = action_loss["loss_depth"] * ratio
            self._update_loss(loss, action_loss, "act")
            loss = self._format_loss(loss)
            return loss

        return action_logits
