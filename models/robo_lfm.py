"""
LFM2.5-VL robotics policy adapter.

Preprocessing contract (matches BaseTrainer + RoboVLMBackbone pipeline):
  - Dataloader: use ``image_processor`` → uint8 CHW tensors in [0, 255], no normalization.
  - Trainer/dataset: call ``process_vision_info`` then ``build_processor_inputs`` to produce
    the processor dict consumed by ``forward_continuous`` as ``lang_x``.
  - Forward: placeholder-token fusion via ``masked_scatter`` (not PaliGemma-style concat).
"""

from __future__ import annotations

import math

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from models.model_backbone import RoboVLMBackbone

ImageInput = Union[Image.Image, torch.Tensor, np.ndarray]


def _sinusoid(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Scalar offsets -> [..., 2*n_freq] sinusoidal features."""
    half = torch.exp(
        torch.arange(n_freq, device=x.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(n_freq - 1, 1)))
    a = x.float().unsqueeze(-1) * half
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class FrameOffsetEmbedding(torch.nn.Module):
    """Temporal tags for ``history_type='pre'`` packing.

    The retained frames are NOT evenly spaced (uniform-spread vs latest-window
    sampling), so ordinal slot position is not enough. Each frame is tagged
    with two sinusoidally-encoded distances, in frames:

        d_cur  = index(current frame) - index(this frame)
        d_next = index(next retained frame) - index(this frame)

    d_cur says how stale the observation is; d_next says how big the jump to
    the following retained frame is, which is what makes an irregular stride
    legible (e.g. frames [1, 4] with current 6 -> frame 1: d_cur 5, d_next 3;
    frame 4: d_cur 2, d_next 2).

    Zero-initialised output projection, so the tag is a no-op at step 0 and
    existing checkpoints keep their behaviour until it is learned.
    """

    def __init__(self, dim: int, n_freq: int = 64):
        super().__init__()
        self.n_freq = n_freq
        self.proj = torch.nn.Linear(4 * n_freq, dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """offsets: [B, W, 2] (d_cur, d_next) -> [B, W, dim]."""
        f = torch.cat([_sinusoid(offsets[..., 0], self.n_freq),
                       _sinusoid(offsets[..., 1], self.n_freq)], dim=-1)
        return self.proj(f.to(self.proj.weight.dtype))


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

    # Set once if Bridge Attention has to drop task tokens (see _pack_vla_adapter_features).
    _warned_task_trunc = False

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
        """Convert PIL images to uint8 CHW tensors at native resolution.

        Kept uint8 on purpose (the documented dataloader contract): the trainer
        converts frames back to PIL before the LFM processor, so a float cast
        here only quadruples host-RAM churn — which, with variable native
        resolutions, fragments the allocator over long runs.
        """

        def _pil_to_chw(img: Image.Image) -> torch.Tensor:
            img = img.convert("RGB")
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()

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
    def build_conversation(
        images: Union[Image.Image, Sequence[Image.Image]],
        instruction: str,
    ) -> List[Dict[str, Any]]:
        """User turn with one or more images + instruction (LFM chat format)."""
        if isinstance(images, Image.Image):
            image_list: List[Image.Image] = [images]
        else:
            image_list = list(images)
        content: List[Dict[str, Any]] = [
            {"type": "image", "image": img} for img in image_list
        ]
        content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    def build_processor_inputs(
        self,
        texts: Sequence[str],
        images: Sequence[ImageInput],
        *,
        padding: bool = True,
        add_generation_prompt: bool = True,
        images_per_sample: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Build the processor batch dict passed to ``forward_continuous`` as ``lang_x``.

        Prefer this over calling ``self.processor(...)`` directly so LFM chat-template
        and vision token constraints stay centralized here.

        When ``images_per_sample > 1``, ``images`` is a flat list grouped as
        ``[img0_cam0, img0_cam1, ..., img1_cam0, ...]``.
        """
        pil_images = self.process_vision_info(images)
        n_img = images_per_sample
        if n_img < 1:
            raise ValueError(f"images_per_sample must be >= 1, got {n_img}")
        if len(pil_images) != len(texts) * n_img:
            raise ValueError(
                f"text/image count mismatch: {len(texts)} instructions × {n_img} cams "
                f"vs {len(pil_images)} images"
            )

        conversations = []
        for i, text in enumerate(texts):
            sample_images = pil_images[i * n_img : (i + 1) * n_img]
            conversations.append(self.build_conversation(sample_images, text))

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

    def _pack_vla_adapter_features(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        image_token_mask: torch.Tensor,
        action_token_mask: torch.Tensor,
        num_task_tokens: int,
        bs: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Pack per-layer Raw vision + ActionQuery features for Bridge Attention.

        Returns ``[B, T, L, N_task + N_aq, D]`` with vision tokens zero-padded/truncated
        to ``num_task_tokens``.
        """
        n_aq = int(self.latent_num)
        dim = hidden_states[0].shape[-1]
        n_layers = len(hidden_states)
        packed = hidden_states[0].new_zeros(
            bs * seq_len, n_layers, num_task_tokens + n_aq, dim
        )

        for layer_idx, hs in enumerate(hidden_states):
            # hs: [B*T, S, D]
            for row in range(bs * seq_len):
                img_idx = image_token_mask[row].nonzero(as_tuple=False).flatten()
                act_idx = action_token_mask[row].nonzero(as_tuple=False).flatten()
                if act_idx.numel() != n_aq:
                    raise ValueError(
                        f"Expected {n_aq} ActionQuery tokens, got {act_idx.numel()} "
                        f"at batch row {row}"
                    )
                if int(img_idx.numel()) > num_task_tokens and not self._warned_task_trunc:
                    self._warned_task_trunc = True
                    print(
                        f"[vla-adapter] WARNING: {int(img_idx.numel())} task tokens "
                        f"(vision + depth) exceed num_task_tokens={num_task_tokens}; "
                        "the excess is silently dropped. Raise act_head.num_task_tokens.",
                        flush=True,
                    )
                n_img = min(int(img_idx.numel()), num_task_tokens)
                if n_img > 0:
                    packed[row, layer_idx, :n_img] = hs[row, img_idx[:n_img]]
                packed[row, layer_idx, num_task_tokens : num_task_tokens + n_aq] = hs[
                    row, act_idx
                ]

        return packed.view(bs, seq_len, n_layers, num_task_tokens + n_aq, dim)

    def _final_text_norm(self):
        """Final norm module of the text tower (for pre-norm feature capture)."""
        import torch.nn as nn

        tower = self.text_tower
        for name in ("norm", "final_layernorm", "embedding_norm", "ln_f", "final_norm"):
            mod = getattr(tower, name, None)
            if isinstance(mod, nn.Module):
                return mod
        if not getattr(self, "_warned_no_final_norm", False):
            self._warned_no_final_norm = True
            print(
                "[robotnav] WARNING: could not locate the text tower's final norm; "
                "use_pre_norm_features falls back to hidden_states[-1].",
                flush=True,
            )
        return None

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

        # Pack variable-length per-row image features into a padded batch for the QFormer.
        # A row may carry several images (agentview + wrist), so group by each row's
        # image-token count instead of assuming one image per row. ``image_features`` is
        # concatenated in the same row-major order that ``masked_scatter`` consumed.
        bs = input_embeds.shape[0]
        tokens_per_row = (input_ids == self.image_token_id).sum(dim=1).tolist()
        if sum(tokens_per_row) != image_features.shape[0]:
            raise ValueError(
                f"Per-row image-token counts sum to {sum(tokens_per_row)} but got "
                f"{image_features.shape[0]} image features."
            )
        max_n = max(max(tokens_per_row), 1)
        dim = image_features.shape[-1]
        packed = image_features.new_zeros(bs, max_n, dim)
        packed_mask = torch.zeros(bs, max_n, dtype=torch.bool, device=image_features.device)
        for i, feat in enumerate(image_features.split(tokens_per_row, dim=0)):
            n = feat.shape[0]
            if n:
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
        frame_offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        loss: Dict[str, Any] = {}
        assert vision_x is not None
        if isinstance(vision_x, (list, tuple)):
            # RobotNav: list of per-sample [ws, C, H, W] tensors at native,
            # per-episode resolutions (pixels flow via lang_x processor dict).
            bs, seq_len = len(vision_x), len(vision_x[0])
        else:
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

        # Wrist / secondary RGB is fused upstream via build_processor_inputs
        # (multi-image chat). vision_gripper is unused here.
        if vision_gripper is not None and not self.is_vla_adapter:
            raise NotImplementedError(
                "hand_rgb / vision_gripper requires VLA-Adapter multi-image path "
                "(act_head.type=VLAAdapterL1Head) or build_processor_inputs(images_per_sample=2)."
            )

        # VLA-Adapter uses proprio on the policy side only (not injected into VLM).
        if rel_state is not None and self.use_state and not self.is_vla_adapter:
            raise NotImplementedError("rel_state conditioning is not implemented for LFM2.5-VL yet.")

        # Image-token mask before ActionQuery / depth-pred tokens are appended.
        image_token_mask = input_ids == self.image_token_id

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
            # Count depth QFormer tokens as task tokens so Bridge Attention reads them
            # directly (they sit after the image tokens, so ordering stays vision→depth).
            # ``num_task_tokens`` must leave room for both; only the adapter path reads
            # this mask, so other heads are unaffected.
            n_depth_cond = depth_tokens.shape[1]
            image_token_mask = torch.cat(
                [
                    image_token_mask,
                    image_token_mask.new_full(
                        (image_token_mask.shape[0], n_depth_cond),
                        self.is_vla_adapter,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
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
            # Keep image mask aligned with the longer sequence (False on new slots).
            n_aq = action_tokens.shape[1]
            image_token_mask = torch.cat(
                [
                    image_token_mask,
                    image_token_mask.new_zeros(
                        image_token_mask.shape[0], n_aq, dtype=torch.bool
                    ),
                ],
                dim=1,
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
                image_token_mask = torch.cat(
                    [
                        image_token_mask,
                        image_token_mask.new_zeros(
                            image_token_mask.shape[0], n_depth, dtype=torch.bool
                        ),
                    ],
                    dim=1,
                )

        if history_type == "pre":
            # Tag each frame's VISUAL tokens with its temporal offsets before
            # the slots are concatenated, so the LLM can tell how far back each
            # frame is and how uneven the stride was. Text tokens are left
            # alone (the instruction is identical in every slot).
            if getattr(self, "frame_offset_embed", None) is not None:
                if frame_offsets is None:                 # fall back: 8,7,...,0
                    ar = torch.arange(seq_len, device=multimodal_embeds.device)
                    d_cur = (seq_len - 1 - ar).float()
                    d_next = torch.ones_like(d_cur); d_next[-1] = 0.0
                    frame_offsets = torch.stack([d_cur, d_next], -1)[None].expand(
                        bs, seq_len, 2)
                temb = self.frame_offset_embed(
                    frame_offsets.to(multimodal_embeds.device))     # [b, l, d]
                temb = rearrange(temb, "b l d -> (b l) 1 d").to(multimodal_embeds.dtype)
                multimodal_embeds = multimodal_embeds + temb * image_token_mask[..., None]
            multimodal_embeds = rearrange(multimodal_embeds, "(b l) n d -> b (l n) d", l=seq_len)
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(
                    multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len)
            if action_token_mask is not None:
                action_token_mask = rearrange(
                    action_token_mask, "(b l) n -> b (l n)", l=seq_len
                )
            image_token_mask = rearrange(
                image_token_mask, "(b l) n -> b (l n)", l=seq_len
            )

        if mode not in ("train", "val"):
            model_dtype = next(self.model.parameters()).dtype
            if multimodal_embeds.dtype != model_dtype:
                multimodal_embeds = multimodal_embeds.to(dtype=model_dtype)

        # Optionally capture the final-layer features BEFORE the backbone's
        # final norm (GR00T-style heads set act_head.use_pre_norm_features).
        _pre_norm_capture: Dict[str, torch.Tensor] = {}
        _pre_norm_hook = None
        if (
            self.act_head_configs is not None
            and self.act_head_configs.get("use_pre_norm_features", False)
        ):
            _norm_mod = self._final_text_norm()
            if _norm_mod is not None:
                _pre_norm_hook = _norm_mod.register_forward_pre_hook(
                    lambda mod, args: _pre_norm_capture.__setitem__("hs", args[0])
                )

        output = self.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeds,
            use_cache=False,
            output_hidden_states=True,
        )

        if _pre_norm_hook is not None:
            _pre_norm_hook.remove()

        depth_hs = None
        head_kwargs: Dict[str, Any] = {}
        # head_kwargs is an explicit allow-list, so anything arriving in
        # **kwargs must be copied in by name or the head never sees it. The
        # stop head's BCE target is silently dropped otherwise: stop_loss()
        # returns None, loss_stop is never added, and the head trains with no
        # gradient at all while still looking alive in the checkpoint.
        if kwargs.get("stop_label") is not None:
            head_kwargs["stop_label"] = kwargs["stop_label"]
        if kwargs.get("lepig_w") is not None:
            head_kwargs["lepig_w"] = kwargs["lepig_w"]

        if self.is_vla_adapter:
            # Bridge Attention consumes every layer (embed + transformer blocks).
            num_task_tokens = int(self.act_head_configs.get("num_task_tokens", 512))
            hs_pack = output.hidden_states
            img_m, act_m = image_token_mask, action_token_mask
            if history_type == "pre":
                # The packer indexes ONE slot per row ([B*T, S, D] + [B*T, S]
                # masks); under "pre" the slots were concatenated into [B, T*S].
                # Slice them back — cross-frame mixing already happened inside
                # the LLM, so Bridge Attention now reads history-aware features
                # at every layer. (Purely a layout fix: VLA-Adapter's
                # contribution is depth-wise fusion, orthogonal to how
                # timesteps are packed.)
                hs_pack = tuple(
                    rearrange(h, "b (l n) d -> (b l) n d", l=seq_len)
                    for h in hs_pack)
                img_m = rearrange(img_m, "b (l n) -> (b l) n", l=seq_len)
                act_m = rearrange(act_m, "b (l n) -> (b l) n", l=seq_len)
            action_hs = self._pack_vla_adapter_features(
                hs_pack,
                image_token_mask=img_m,
                action_token_mask=act_m,
                num_task_tokens=num_task_tokens,
                bs=bs,
                seq_len=seq_len,
            )
            head_kwargs["proprio"] = rel_state
            head_kwargs["phase"] = "Training" if mode in ("train", "val") else "Inference"
        else:
            output_hs = _pre_norm_capture.get("hs", output.hidden_states[-1]).clone()
            if history_type == "pre":
                output_hs = rearrange(output_hs, "b (l n) d -> (b l) n d", l=seq_len)
                # The token masks were flattened alongside the embeddings; put
                # them back on the same per-slot layout or the boolean index
                # below mismatches ([b, l*n] vs [(b l), n, d]).
                # (depth_pred_token_mask is never flattened above, so it is
                # already on the per-slot layout and must be left alone.)
                # down_sample heads have no ActionQuery tokens -> mask is None.
                if action_token_mask is not None:
                    action_token_mask = rearrange(
                        action_token_mask, "b (l n) -> (b l) n", l=seq_len)

            if action_space == "continuous":
                action_hs = output_hs[action_token_mask].reshape(
                    bs, seq_len, self.latent_num, -1
                )
                if depth_pred_token_mask is not None:
                    depth_hs = output_hs[depth_pred_token_mask].reshape(
                        bs, seq_len, self.depth_latent_num, -1
                    )
            elif action_space == "down_sample":
                token_src = self.act_head_configs.get("token_source", "all")
                if token_src != "all":
                    raise ValueError(f"Unsupported token source {token_src}")
                action_hs = output_hs.reshape(bs, seq_len, *output_hs.shape[1:])
                # Full-sequence heads need to ignore padding when attending
                # over VLM features (RobotNav FM heads).
                head_kwargs["encoder_attention_mask"] = multimodal_attention_mask
                if self.act_head_configs.get("type") == "SmolVLAFlowMatchingHead":
                    hs_all = output.hidden_states
                    if history_type == "pre":
                        # Slice the fused sequence back per slot: cross-frame
                        # mixing already happened inside the LLM, and the
                        # expert expects [(b l), n, d] like the action tokens.
                        hs_all = tuple(
                            rearrange(h, "b (l n) d -> (b l) n d", l=seq_len)
                            for h in hs_all)
                    head_kwargs["per_layer_hs"] = hs_all
            else:
                raise ValueError(f"Unsupported action space {action_space}")

        if self.use_clip_norm and mode == "train" and not self.is_vla_adapter:
            clip_loss = self.clip_norm_head(action_hs, raw_text)
            self._update_loss(loss, clip_loss, "clip")

        if mode not in ("train", "val") and self.act_head is not None:
            head_dtype = next(self.act_head.parameters()).dtype
            action_hs = action_hs.to(dtype=head_dtype)
            if depth_hs is not None:
                depth_hs = depth_hs.to(dtype=head_dtype)

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

        # LEPIG gradient routing: identity forward, per-example scaling of the
        # gradient that reaches the shared backbone. The action head's own
        # parameter gradient is deliberately left unweighted -- weighting the
        # whole FM loss would distort q(a|c) and skew the policy toward rare
        # modes, which the plan document explicitly prohibits.
        _w = head_kwargs.pop("lepig_w", None)
        if _w is not None:
            from models.lepig.routing import grad_scale_identity
            action_hs = grad_scale_identity(action_hs, _w)
        action_logits, action_loss, depth_pred = self.forward_action_head(
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

        if depth_pred is not None:
            return {"action": action_logits, "depth": depth_pred}
        return action_logits
