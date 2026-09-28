"""Multi-head factorised attention (MHFA) pooling head.

Basis: the MHFA speaker head of WeSpeaker, extended here with the LayerNorm and
projection parameters of the SpoofLLM dual-stream adapter. The same attention
weights and values feed the pooled global stream and the time-resolved frame
stream (Sec. 3.2 of the paper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch import nn


FRONTEND_PREFIX = "frontend.upstream."
MHFA_FULL_LAYER_COUNT = 25
MHFA_L20_LAYER_COUNT = 21
LEGACY_L20_DROPPED_MHFA_LAYER_INDICES = (21, 22, 23, 24)
NEGLIGIBLE_DROPPED_MASS_THRESHOLD = 0.01
EXPECTED_MHFA_BACKEND_KEYS = frozenset(
    {
        "weights_k",
        "weights_v",
        "cmp_linear_k.weight",
        "cmp_linear_k.bias",
        "cmp_linear_v.weight",
        "cmp_linear_v.bias",
        "norm_k.weight",
        "norm_k.bias",
        "norm_v.weight",
        "norm_v.bias",
        "att_head.weight",
        "att_head.bias",
        "pooling_fc.weight",
        "pooling_fc.bias",
        "projection.weight",
    }
)


class GradMultiply(torch.autograd.Function):
    """Gradient scaling helper copied from the wespeaker MHFA source."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.new(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SSL_BACKEND_MHFA_LayerNorm(nn.Module):
    """Base MHFA with compressed-key/value LayerNorm and no-bias projection."""

    supports_ssl_frame_mask = True

    def __init__(
        self,
        head_nb: int = 64,
        feat_dim: int = 1024,
        compression_dim: int = 128,
        embed_dim: int = 256,
        nb_layer: int = MHFA_FULL_LAYER_COUNT,
        feature_grad_mult: float = 1.0,
        projection_dim: int = 17982,
    ):
        super().__init__()

        self.feature_grad_mult = feature_grad_mult
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)

        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim
        self.nb_layer = nb_layer

        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.norm_k = nn.LayerNorm(self.cmp_dim)
        self.norm_v = nn.LayerNorm(self.cmp_dim)
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)
        self.projection = nn.Linear(self.ous_dim, projection_dim, bias=False)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        probs = nn.functional.softmax(masked_logits, dim=dim)
        probs = probs * mask.to(dtype=logits.dtype)
        denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-6)
        return probs / denom

    @staticmethod
    def _validate_frame_mask(x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        expected_shape = (x.shape[0], x.shape[2], x.shape[3])
        if frame_mask.shape[-1] > x.shape[-1]:
            frame_mask = frame_mask[..., : x.shape[-1]]
        if frame_mask.shape != expected_shape:
            raise ValueError(f"Expected frame_mask shape {expected_shape}, got {tuple(frame_mask.shape)}.")
        return frame_mask.to(device=x.device, dtype=torch.bool)

    @staticmethod
    def _select_layer_logits(layer_weights: torch.Tensor, nb_layers: int) -> torch.Tensor:
        if nb_layers > layer_weights.numel():
            raise ValueError(f"Input has {nb_layers} layers, but MHFA only has {layer_weights.numel()} layer logits.")
        return layer_weights[:nb_layers]

    def _layer_weighted_sum(
        self,
        x: torch.Tensor,
        layer_weights: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        layer_logits = self._select_layer_logits(layer_weights, x.shape[-1])
        soft_weights = nn.functional.softmax(layer_logits, dim=-1).to(dtype=x.dtype)

        if frame_mask is None:
            return torch.sum(x.mul(soft_weights.view(1, 1, 1, -1)), dim=-1).transpose(1, 2)

        x_layers = x.permute(0, 2, 1, 3)
        masked_weights = frame_mask.to(dtype=x.dtype) * soft_weights.view(1, 1, -1)
        masked_weights = masked_weights / masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(x_layers * masked_weights.unsqueeze(2), dim=-1)

    def get_frame_att_outputs(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        # Input x has shape [B, D, T, L]. L can be 25 full states or 21 kept L0-L20 states.
        x = GradMultiply.apply(x, self.feature_grad_mult)
        if frame_mask is not None:
            frame_mask = self._validate_frame_mask(x, frame_mask)
            valid_frames = frame_mask.any(dim=-1)
        else:
            valid_frames = None

        k = self._layer_weighted_sum(x, self.weights_k, frame_mask)
        v = self._layer_weighted_sum(x, self.weights_v, frame_mask)

        k = self.norm_k(self.cmp_linear_k(k))
        v = self.norm_v(self.cmp_linear_v(v))

        att_k = self.att_head(k)
        if valid_frames is not None:
            att_prob = self._masked_softmax(att_k, valid_frames.unsqueeze(-1), dim=1)
        else:
            att_prob = nn.functional.softmax(att_k, dim=1)

        att_out = v.unsqueeze(-2).mul(att_prob.unsqueeze(-1))
        if valid_frames is not None:
            att_out = att_out * valid_frames.unsqueeze(-1).unsqueeze(-1).to(dtype=att_out.dtype)
        return {"att_out": att_out, "frame_features": v, "att_prob": att_prob}

    def get_frame_att_emb(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.get_frame_att_outputs(x, frame_mask=frame_mask)["att_out"]

    def frame_features(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.get_frame_att_outputs(x, frame_mask=frame_mask)["frame_features"]

    def dual_readout(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        outputs = self.get_frame_att_outputs(x, frame_mask=frame_mask)
        pooling_outs = torch.sum(outputs["att_out"], dim=1)
        batch, heads, features = pooling_outs.shape
        outputs["embedding"] = self.pooling_fc(pooling_outs.reshape(batch, heads * features))
        outputs["pooling_tokens"] = pooling_outs
        return outputs

    def get_frame_emb(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.get_frame_att_emb(x, frame_mask=frame_mask).mean(dim=2)

    def pooling_tokens(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.sum(self.get_frame_att_emb(x, frame_mask=frame_mask), dim=1)

    def forward(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.dual_readout(x, frame_mask=frame_mask)["embedding"]

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)


def _backend_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = {key: value for key, value in checkpoint.items() if not key.startswith(FRONTEND_PREFIX)}
    actual = set(state_dict)
    if actual != EXPECTED_MHFA_BACKEND_KEYS:
        missing = sorted(EXPECTED_MHFA_BACKEND_KEYS - actual)
        unexpected = sorted(actual - EXPECTED_MHFA_BACKEND_KEYS)
        raise RuntimeError(f"MHFA backend key mismatch: missing={missing}, unexpected={unexpected}")
    return state_dict


def load_mhfa_backend(
    ckpt: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> SSL_BACKEND_MHFA_LayerNorm:
    checkpoint = torch.load(Path(ckpt), map_location="cpu")
    state_dict = _backend_state_dict(checkpoint)
    projection_dim = int(state_dict["projection.weight"].shape[0])
    backend = SSL_BACKEND_MHFA_LayerNorm(projection_dim=projection_dim)
    missing, unexpected = backend.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"MHFA backend load mismatch: missing={missing}, unexpected={unexpected}")
    if device is not None:
        device = torch.device(device)
        if dtype is not None:
            backend = backend.to(device=device, dtype=dtype)
        else:
            backend = backend.to(device=device)
    backend.eval()
    backend.load_info = {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "keys": sorted(state_dict),
        "projection_dim": projection_dim,
    }
    return backend


def mhfa_l20_mask_summary(backend: SSL_BACKEND_MHFA_LayerNorm) -> dict[str, dict[str, float] | float]:
    def summarize(weights: torch.Tensor) -> dict[str, float]:
        probs = torch.softmax(weights.detach().float().cpu(), dim=-1)
        kept_mass = probs[:MHFA_L20_LAYER_COUNT].sum()
        dropped_mass = probs[MHFA_L20_LAYER_COUNT:].sum()
        return {
            "kept_0_20_mass": float(kept_mass.item()),
            "dropped_21_24_mass": float(dropped_mass.item()),
            "implied_relative_increase_to_each_0_20_weight": float((dropped_mass / kept_mass).item()),
        }

    weights_k = summarize(backend.weights_k)
    weights_v = summarize(backend.weights_v)
    return {
        "weights_k": weights_k,
        "weights_v": weights_v,
        "max_dropped_21_24_mass": max(weights_k["dropped_21_24_mass"], weights_v["dropped_21_24_mass"]),
    }


def mhfa_input_from_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    kept_layers: int = MHFA_FULL_LAYER_COUNT,
) -> torch.Tensor:
    if len(hidden_states) < kept_layers:
        raise ValueError(f"Expected at least {kept_layers} hidden states, got {len(hidden_states)}.")
    stacked = torch.stack(tuple(hidden_states[:kept_layers]), dim=-1)
    return stacked.permute(0, 2, 1, 3).contiguous()


class W2VBertMHFAXVector(nn.Module):
    """Full W2V-BERT encoder plus the real 25-layer MHFA xvector backend."""

    def __init__(self, encoder: nn.Module, backend: SSL_BACKEND_MHFA_LayerNorm):
        super().__init__()
        self.encoder = encoder
        self.backend = backend

    @torch.no_grad()
    def xvector(self, wav: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.encoder(wav, lengths=lengths)
        mhfa_input = mhfa_input_from_hidden_states(outputs.hidden_states)
        return self.backend(mhfa_input)

    def forward(self, wav: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.xvector(wav, lengths=lengths)


def load_mhfa_xvector_model(
    ckpt: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[Union[str, Path]] = None,
    model_dir: Optional[Union[str, Path]] = None,
    dtype: Optional[torch.dtype] = None,
    allow_unsafe_l20_mask: bool = False,
) -> W2VBertMHFAXVector:
    from . import DEFAULT_CKPT, DEFAULT_CONFIG, DEFAULT_MODEL_DIR, load_w2vbert_encoder

    ckpt = DEFAULT_CKPT if ckpt is None else ckpt
    config = DEFAULT_CONFIG if config is None else config
    model_dir = DEFAULT_MODEL_DIR if model_dir is None else model_dir
    backend = load_mhfa_backend(ckpt=ckpt, device=device, dtype=dtype)
    mask_summary = mhfa_l20_mask_summary(backend)
    encoder = load_w2vbert_encoder(
        ckpt=ckpt,
        device=device,
        config=config,
        model_dir=model_dir,
        dtype=dtype,
        truncate_to_l20=False,
    )
    model = W2VBertMHFAXVector(encoder=encoder, backend=backend)
    model.load_info = {
        "mhfa_layers": MHFA_FULL_LAYER_COUNT,
        "encoder_truncate_to_l20": False,
        "mhfa_l20_mask_summary": mask_summary,
        "allow_unsafe_l20_mask_argument_ignored": bool(allow_unsafe_l20_mask),
    }
    model.eval()
    return model


@torch.no_grad()
def xvector(model: W2VBertMHFAXVector, wav: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    return model.xvector(wav, lengths=lengths)
