from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPTanhHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPNohHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSigmoidHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPHead(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)

class BasePolicyHead(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        action_dim,
        action_space="continuous",
        down_sample="pooling",
        latent=1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.down_sample = down_sample
        self.latent = latent
        self.action_space = action_space
    
    def _get_target_modes(self, output_hs, tok_mask):
        index = tok_mask.nonzero(as_tuple=True)
        return output_hs[index]

    def _get_normal_modes(self, tok_seq, tok_dict, modal_name):
        assert modal_name in tok_dict, f"{modal_name} not in token sequence"
        return self._get_target_modes(tok_seq, tok_dict[modal_name])

    def loss(self, pred_action, labels, attention_mask=None):
        """
        pred_action_logits: [bs, seq_len, chunck_size, 7], 1-6 refers to ee pose, 7 refers to gripper open/close
        lables: (pose gt [bs, seq_len, chunck_size, 6], gripper gt [bs, seq_len, chunck_size])
        attention_mask: [bs, seq_len, chunck_size]
        """
        if labels is None or labels[0] is None:
            return {"loss": None}

        if isinstance(pred_action, tuple) or isinstance(pred_action, list):
            if pred_action[0].ndim == pred_action[1].ndim:
                pred_action = torch.cat(pred_action, dim=-1)
            elif pred_action[0].ndim == pred_action[1].ndim + 1:
                pred_action = torch.cat([pred_action[0], pred_action[1].unsqueeze(-1)], dim=-1)
            else:
                raise ValueError("Can not solve the gripper action dim")
        if attention_mask is None:
            pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0])
            # pose_loss = torch.nn.functional.mse_loss(pred_action[..., :6], labels[0])
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_action[..., -1], labels[1])
        else:
            pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0], reduction="none")
            # pose_loss = torch.nn.functional.mse_loss(pred_action[..., :6], labels[0], reduction='none')
            attention_mask = attention_mask.bool()
            pose_loss = pose_loss[attention_mask].mean()
            # gripper_loss = torch.nn.functional.binary_cross_entropy(pred_action[..., -1], labels[1], reduction='none')
            gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_action[..., -1], labels[1], reduction="none")
            gripper_loss = gripper_loss[attention_mask].mean()

        gripper_action_preds = (F.sigmoid(pred_action[..., -1]) > 0.5).float()
        acc_gripper_act = torch.eq(gripper_action_preds, labels[1]).float()
        if attention_mask is None:
            acc_gripper_act = acc_gripper_act.mean()
        else:
            # acc_gripper_act = (acc_gripper_act * attention_mask).sum() / attention_mask.sum()
            acc_gripper_act = acc_gripper_act[attention_mask].mean()

        return {
            "loss_arm": pose_loss,
            "loss_gripper": gripper_loss,
            "acc_gripper": acc_gripper_act,
        }

    def get_labels(self, pred_actions, labels, action_masks, **kwargs):
        return pred_actions, labels, action_masks


def initialize_param(model):
    with torch.no_grad():
        for m in model.children():
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias"):
                    m.bias.fill_(0)
            else:
                initialize_param(m)
                

class FCDecoder(BasePolicyHead):
    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        **kwargs,
    ):

        super().__init__(hidden_size, action_dim, **kwargs)
        self.in_features = in_features
        self.fwd_pred_next_n = fwd_pred_next_n
        self.down_sample = down_sample
        self.latent = latent
        self.action_dim = action_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features * latent, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, hidden_size * latent),
        )
        self.actions = MLPTanhHead(hidden_size * latent, fwd_pred_next_n * (action_dim - 1))
        self.gripper = MLPSigmoidHead(hidden_size * latent, fwd_pred_next_n)

        if self.down_sample == "pooling":
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif self.down_sample == "resampler":
            pass 
        elif self.down_sample == "none":
            pass
        else:
            NotImplementedError
        initialize_param(self)

    def forward(self, tok_seq, **kwargs):
        if len(tok_seq.shape) == 4:
            bs, seq_len, n_tok, tok_dim = tok_seq.shape
            tok_seq = rearrange(tok_seq,
                                "b l n d-> (b l) n d")  # reduce the seq_len dim (4, 8, 1, 1024)->(4*8, 1, 1024)
        elif tok_seq.dim() == 3:
            bs, n_tok, tok_dim = tok_seq.shape
            seq_len = None
        else:
            assert len(tok_seq.shape) == 2
            bs, tok_dim = tok_seq.shape
            seq_len = None
            n_tok = None
            tok_seq = tok_seq.unsqueeze(1)

        # here tok_seq is (bs*seq_len, n_tok, tok_dim)
        if self.down_sample == "pooling":
            tok_seq = self.pooling(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, "b d n -> b (n d)")
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b n d -> b (n d)")
        else:
            raise NotImplementedError

        tok_seq = self.mlp(tok_seq)
        actions = self.actions(tok_seq)
        gripper = self.gripper(tok_seq)
        if seq_len is not None:
            actions = rearrange(
                actions,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
            )
            gripper = rearrange(
                gripper,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
            )
        elif n_tok is not None:
            actions = rearrange(actions, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n)
            gripper = rearrange(gripper, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n)

        return actions, gripper


class DepthMapHead(torch.nn.Module):
    """Decode a feature vector into a single-channel depth map in ``[0, 1]``."""

    def __init__(self, in_dim: int, map_size: int = 56):
        super().__init__()
        if map_size not in (28, 56, 112, 224):
            raise ValueError(f"depth_map_size must be one of {{28,56,112,224}}, got {map_size}")
        self.map_size = map_size
        # Start from a 7x7 latent grid, then upsample by 2x until map_size.
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128 * 7 * 7),
        )
        ups = []
        channels = [128, 64, 32, 16, 8]
        cur_size = 7
        stage = 0
        while cur_size < map_size:
            ups.extend(
                [
                    torch.nn.ConvTranspose2d(
                        channels[stage], channels[stage + 1], kernel_size=4, stride=2, padding=1
                    ),
                    torch.nn.ReLU(),
                ]
            )
            cur_size *= 2
            stage += 1
        ups.extend(
            [
                torch.nn.Conv2d(channels[stage], 1, kernel_size=3, padding=1),
                torch.nn.Sigmoid(),
            ]
        )
        self.decoder = torch.nn.Sequential(*ups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, D]``
        Returns:
            depth: ``[B, 1, H, W]`` with ``H=W=map_size``
        """
        feat = self.fc(x).view(x.shape[0], 128, 7, 7)
        return self.decoder(feat)


class HierarchicalFCDecoder(BasePolicyHead):
    """Shared trunk over action+depth query HS, then action (arm/gripper) + depth map.

    Hierarchy
    ---------
    1. Concatenate action tokens and depth-pred tokens → shared MLP trunk
    2. Split into action branch / depth branch
    3. Action branch → arm (tanh) + gripper (logits), same as ``FCDecoder``
    4. Depth branch → ``DepthMapHead`` predicting a GT depth map (aux loss)
    """

    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        depth_latent: int = 1,
        depth_map_size: int = 56,
        **kwargs,
    ):
        super().__init__(hidden_size, action_dim, **kwargs)
        self.in_features = in_features
        self.fwd_pred_next_n = fwd_pred_next_n
        self.down_sample = down_sample
        self.action_latent = int(latent)
        self.depth_latent = int(depth_latent)
        self.latent = self.action_latent  # BasePolicyHead / callers use this for action slots
        self.total_latent = self.action_latent + self.depth_latent
        self.depth_map_size = int(depth_map_size)
        self.action_dim = action_dim

        shared_out = hidden_size * self.total_latent
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(in_features * self.total_latent, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, shared_out),
        )
        self.action_proj = torch.nn.Sequential(
            torch.nn.Linear(shared_out, hidden_size * self.action_latent),
            torch.nn.ReLU(),
        )
        self.depth_proj = torch.nn.Sequential(
            torch.nn.Linear(shared_out, hidden_size * self.depth_latent),
            torch.nn.ReLU(),
        )
        self.actions = MLPTanhHead(
            hidden_size * self.action_latent, fwd_pred_next_n * (action_dim - 1)
        )
        self.gripper = MLPSigmoidHead(hidden_size * self.action_latent, fwd_pred_next_n)
        self.depth_head = DepthMapHead(hidden_size * self.depth_latent, self.depth_map_size)

        if self.down_sample == "pooling":
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif self.down_sample in ("none", "resampler"):
            pass
        else:
            raise NotImplementedError(f"Unsupported down_sample={self.down_sample}")
        initialize_param(self)

    def _flatten_tokens(self, tok_seq: torch.Tensor) -> torch.Tensor:
        if self.down_sample == "pooling":
            tok_seq = self.pooling(tok_seq.permute(0, 2, 1))
            return rearrange(tok_seq, "b d n -> b (n d)")
        if self.down_sample == "none":
            return rearrange(tok_seq, "b n d -> b (n d)")
        raise NotImplementedError(f"Unsupported down_sample={self.down_sample}")

    def forward(self, tok_seq, depth_hs=None, **kwargs):
        """
        Args:
            tok_seq: action query HS ``[B, T, Na, D]`` (or flattened variants like FCDecoder)
            depth_hs: depth-pred query HS ``[B, T, Nd, D]`` (required)
        Returns:
            dict with ``actions``, ``gripper``, and ``depth`` (``[B, T, 1, H, W]``).
        """
        if depth_hs is None:
            raise ValueError("HierarchicalFCDecoder requires depth_hs from depth_pred tokens.")

        if tok_seq.ndim != 4 or depth_hs.ndim != 4:
            raise ValueError(
                f"Expected action/depth HS as [B,T,N,D], got {tuple(tok_seq.shape)} / "
                f"{tuple(depth_hs.shape)}"
            )
        if tok_seq.shape[2] != self.action_latent or depth_hs.shape[2] != self.depth_latent:
            raise ValueError(
                f"Token counts mismatch: action {tok_seq.shape[2]} vs latent={self.action_latent}, "
                f"depth {depth_hs.shape[2]} vs depth_latent={self.depth_latent}"
            )

        bs, seq_len = tok_seq.shape[:2]
        # [B, T, Na+Nd, D] → [(B*T), Na+Nd, D]
        combined = torch.cat([tok_seq, depth_hs], dim=2)
        combined = rearrange(combined, "b l n d -> (b l) n d")
        shared_feat = self.shared(self._flatten_tokens(combined))
        action_feat = self.action_proj(shared_feat)
        depth_feat = self.depth_proj(shared_feat)

        actions = self.actions(action_feat)
        gripper = self.gripper(action_feat)
        depth_map = self.depth_head(depth_feat)  # [(B*T), 1, H, W]

        actions = rearrange(
            actions, "(b l) (n d) -> b l n d", b=bs, l=seq_len, n=self.fwd_pred_next_n
        )
        gripper = rearrange(
            gripper, "(b l) (n d) -> b l n d", b=bs, l=seq_len, n=self.fwd_pred_next_n
        )
        depth_map = rearrange(depth_map, "(b l) c h w -> b l c h w", b=bs, l=seq_len)
        return {"actions": actions, "gripper": gripper, "depth": depth_map}

    def loss(self, pred_action, labels, attention_mask=None, depth_pred=None, depth_gt=None):
        """Arm/gripper losses plus optional auxiliary depth-map L1."""
        base = BasePolicyHead.loss(self, pred_action, labels, attention_mask)
        if depth_pred is None or depth_gt is None:
            return base

        # depth_pred: [B, T, 1, H, W]; depth_gt: [B, T, 1, Hg, Wg] in [0, 1]
        gt = depth_gt.float()
        if gt.ndim == 4:
            gt = gt.unsqueeze(2)  # [B, T, H, W] → [B, T, 1, H, W]
        if gt.shape[-2:] != depth_pred.shape[-2:]:
            b, t, c, _, _ = depth_pred.shape
            gt_flat = rearrange(gt, "b t c h w -> (b t) c h w")
            gt_flat = F.interpolate(
                gt_flat, size=depth_pred.shape[-2:], mode="bilinear", align_corners=False
            )
            gt = rearrange(gt_flat, "(b t) c h w -> b t c h w", b=b, t=t)
        pred = depth_pred.float()
        # Match conditioning path: clamp GT to [0, 1]
        gt = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        depth_loss = F.l1_loss(pred, gt)
        base["loss_depth"] = depth_loss
        return base


class FCDecoderDualArm(BasePolicyHead):
    def __init__(
        self,
        in_features,
        hidden_size,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        **kwargs,
    ):

        super().__init__(hidden_size, action_dim, **kwargs)
        self.in_features = in_features
        self.fwd_pred_next_n = fwd_pred_next_n
        self.down_sample = down_sample
        self.latent = latent
        assert action_dim == 14, "action_dim must be 14 for dual arm"
        self.mlp = (
            torch.nn.Linear(in_features * latent, 1024), #1024 for calvin, might be a good idea to change to in_features * latent // 2
            torch.nn.ReLU(),
            torch.nn.Linear(1024, hidden_size * latent),
        )
        self.actions = MLPTanhHead(hidden_size * latent, fwd_pred_next_n * (action_dim - 2))
        self.gripper = MLPSigmoidHead(hidden_size * latent, fwd_pred_next_n * 2)

        if self.down_sample == "pooling":
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        elif self.down_sample == "resampler":
            pass 
        elif self.down_sample == "none":
            pass
        else:
            NotImplementedError
        initialize_param(self)

    def forward(self, tok_seq, **kwargs):
        if len(tok_seq.shape) == 4:
            bs, seq_len, n_tok, tok_dim = tok_seq.shape
            tok_seq = rearrange(tok_seq,
                                "b l n d-> (b l) n d")  # reduce the seq_len dim (4, 8, 1, 1024)->(4*8, 1, 1024)
        elif tok_seq.dim() == 3:
            bs, n_tok, tok_dim = tok_seq.shape
            seq_len = None
        else:
            assert len(tok_seq.shape) == 2
            bs, tok_dim = tok_seq.shape
            seq_len = None
            n_tok = None
            tok_seq = tok_seq.unsqueeze(1)

        # here tok_seq is (bs*seq_len, n_tok, tok_dim)
        if self.down_sample == "pooling":
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, "b d n -> b (n d)")
        elif self.down_sample == "resampler":
            raise NotImplementedError
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b n d -> b (n d)")
        else:
            raise NotImplementedError

        tok_seq = self.mlp(tok_seq)
        actions = self.actions(tok_seq)
        gripper = self.gripper(tok_seq)
        if seq_len is not None:
            # input is 4-dim
            actions = rearrange(
                actions,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
                x = 2
            )
            gripper = rearrange(
                gripper,
                "(b l) (n d) -> b l n d",
                b=bs,
                l=seq_len,
                n=self.fwd_pred_next_n,
                x = 2
            )
        elif n_tok is not None:
            # input is 3-dim
            actions = rearrange(actions, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n)
            gripper = rearrange(gripper, "b (n d) -> b n d", b=bs, n=self.fwd_pred_next_n)

        return actions, gripper
