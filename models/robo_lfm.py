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
from einops import rearrange, repeat
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
    ) -> torch.Tensor:
        """Scatter projected vision features into ``image_token_id`` placeholder positions."""
        image_outputs = self.model.get_image_features(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            return_dict=True,
        )
        image_features = torch.cat(image_outputs.pooler_output, dim=0)
        image_features = image_features.to(device=input_embeds.device, dtype=input_embeds.dtype)

        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        n_image_features = image_features.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={n_image_features}")

        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(input_embeds)
        return input_embeds.masked_scatter(image_mask.to(input_embeds.device), image_features)

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
        input_embeds = self._fuse_image_features(
            input_ids,
            input_embeds,
            pixel_values,
            spatial_shapes,
            pixel_attention_mask,
        )

        multimodal_embeds = input_embeds
        multimodal_labels = None
        multimodal_attention_mask = attention_mask

        if vision_gripper is not None:
            raise NotImplementedError("hand_rgb / vision_gripper is not supported for LFM2.5-VL yet.")

        if rel_state is not None and self.use_state:
            raise NotImplementedError("rel_state conditioning is not implemented for LFM2.5-VL yet.")

        action_token_mask = None
        if action_space == "continuous":
            action_tokens = repeat(
                self.action_token,
                "d -> b n d",
                b=multimodal_embeds.shape[0],
                n=self.latent_num,
            )
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

        if history_type == "pre":
            multimodal_embeds = rearrange(multimodal_embeds, "(b l) n d -> b (l n) d", l=seq_len)
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(
                    multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len)

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

        if action_space == "continuous":
            action_hs = output_hs[action_token_mask].reshape(bs, seq_len, self.latent_num, -1)
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

        action_logits, action_loss = self.forward_action_head(
            action_hs, action_labels, action_mask)

        if mode == "train":
            self._update_loss(loss, action_loss, "act")
            loss = self._format_loss(loss)
            return loss

        return action_logits
