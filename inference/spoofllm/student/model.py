"""Qwen soft-token student model for Wave-1 spoof distillation."""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from spoofllm.student.mhfa import SSL_BACKEND_MHFA_LayerNorm, mhfa_input_from_hidden_states


# Env-gated forward-phase profiler (default OFF -> zero overhead, safe for prod/resume).
# Set UNI_A3_PROFILE=1 to accumulate per-phase CUDA-synced wall time and print every
# UNI_A3_PROFILE_EVERY (default 25) calls.
_PROFILE_ON = os.environ.get("UNI_A3_PROFILE", "0") == "1"
_PROFILE_EVERY = int(os.environ.get("UNI_A3_PROFILE_EVERY", "25"))
# Skip the first _PROFILE_WARMUP calls (one-time MIOpen/kernel compilation makes the
# first fwd+bwd ~100x slower); reset accumulators at that boundary for clean steady-state.
_PROFILE_WARMUP = int(os.environ.get("UNI_A3_PROFILE_WARMUP", "15"))
_PROFILE_ACC: dict[str, float] = {}
_PROFILE_N = {"n": 0, "measured": 0}


@contextlib.contextmanager
def _profile_phase(name: str):
    if not _PROFILE_ON or _PROFILE_N["n"] < _PROFILE_WARMUP:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _PROFILE_ACC[name] = _PROFILE_ACC.get(name, 0.0) + (time.time() - t0)


def _profile_tick() -> None:
    if not _PROFILE_ON:
        return
    _PROFILE_N["n"] += 1
    if _PROFILE_N["n"] < _PROFILE_WARMUP:
        return
    _PROFILE_N["measured"] += 1
    m = _PROFILE_N["measured"]
    if m % _PROFILE_EVERY == 0 and int(os.environ.get("RANK", "0")) == 0:
        parts = " ".join(f"{k}={v / m * 1000:.1f}ms" for k, v in sorted(_PROFILE_ACC.items()))
        print(f"[profile] steady_calls={m} per-call: {parts}", flush=True)


# Fallback only: callers pass the Qwen2.5-1.5B-Instruct directory explicitly
# (score_wavs.py --llm). Set SPOOFLLM_QWEN_DIR to change the fallback.
DEFAULT_LLM = Path(os.environ.get("SPOOFLLM_QWEN_DIR", "Qwen/Qwen2.5-1.5B-Instruct"))


class AttnPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, max(1, hidden // 4)), nn.Tanh(), nn.Linear(max(1, hidden // 4), 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class SpoofStudentReasoner(nn.Module):
    """DF-Arena pooled/window embeddings -> Qwen LoRA -> LR heads + trace LM."""

    def __init__(
        self,
        backbone: nn.Module,
        lm_head: nn.Module,
        emb_layer: nn.Module,
        hidden: int,
        *,
        in_dim: int = 1280,
        n_global_tokens: int = 8,
        max_windows: int = 30,
        n_speaker_tokens: int = 0,
        speaker_dim: int = 256,
        num_methods: int = 0,
        readout: str = "last",
        region_head: str = "global",
        enable_region: bool = True,
        enable_speaker: bool = True,
        input_embed_rms: float = 1.0,
        use_bf16_autocast: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.lm_head = lm_head
        self.emb_layer = emb_layer
        self.hidden = int(hidden)
        self.in_dim = int(in_dim)
        self.n_global_tokens = int(n_global_tokens)
        self.max_windows = int(max_windows)
        self.n_speaker_tokens = int(n_speaker_tokens)
        self.speaker_dim = int(speaker_dim)
        self.num_methods = int(num_methods)
        self.readout = str(readout)
        self.enable_region = bool(enable_region)
        self.enable_speaker = bool(enable_speaker)
        self.region_head_mode = str(region_head).lower()
        if self.region_head_mode not in {"global", "windowed"}:
            raise ValueError(f"unsupported region_head: {region_head}")
        self.use_bf16_autocast = bool(use_bf16_autocast)
        self.register_buffer("input_embed_rms", torch.tensor(float(input_embed_rms), dtype=torch.float32), persistent=False)

        self.global_adapter = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, self.n_global_tokens * hidden),
        )
        self.window_adapter = nn.Sequential(nn.Linear(in_dim, 1024), nn.GELU(), nn.Linear(1024, hidden))
        self.speaker_adapter = (
            nn.Sequential(nn.Linear(self.speaker_dim, 512), nn.GELU(), nn.Linear(512, self.n_speaker_tokens * hidden))
            if self.n_speaker_tokens > 0
            else None
        )
        self.ln = nn.LayerNorm(hidden)
        self.global_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.window_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.window_pos = nn.Parameter(torch.zeros(1, self.max_windows, hidden))
        self.speaker_type = nn.Parameter(torch.zeros(1, 1, hidden)) if self.n_speaker_tokens > 0 else None
        self.attn = AttnPool(hidden) if readout == "attnpool" else None
        self.lr_head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 4))
        self.verdict_head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 1))
        self.speaker_readout_head = (
            nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Linear(512, self.speaker_dim))
            if self.enable_speaker
            else None
        )
        self.method_head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, self.num_methods)) if self.num_methods > 0 else None
        self.region_head = (
            nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, self.max_windows))
            if self.enable_region
            else None
        )
        self.region_seq: nn.GRU | None = None
        self.region_win_cls: nn.Linear | None = None
        if self.enable_region and self.region_head_mode == "windowed":
            if self.hidden % 2 != 0:
                raise ValueError("--region-head windowed requires an even hidden size for bidirectional GRU output")
            self.region_seq = nn.GRU(
                input_size=self.hidden,
                hidden_size=self.hidden // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.region_win_cls = nn.Linear(self.hidden, 1)
        self.tokcal_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.tokcal_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def soft_token_count(self) -> int:
        return int(self.n_global_tokens + self.max_windows + self.n_speaker_tokens)

    def backbone_context(self) -> contextlib.AbstractContextManager:
        if self.use_bf16_autocast and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _normalize_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        tok = self.ln(tok.float())
        tok_rms = tok.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        tok = tok / tok_rms
        tok = tok * self.input_embed_rms.to(device=tok.device, dtype=tok.dtype)
        return tok.to(dtype=self.emb_layer.weight.dtype)

    def soft_tokens(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(global_emb.shape[0])
        global_tok = self.global_adapter(global_emb.float()).view(batch, self.n_global_tokens, self.hidden)
        global_tok = global_tok + self.global_type.to(device=global_tok.device, dtype=global_tok.dtype)
        global_tok = self._normalize_tokens(global_tok)

        n_win = min(int(window_emb.shape[1]), self.max_windows)
        win = self.window_adapter(window_emb[:, :n_win, :].float())
        win = win + self.window_type.to(device=win.device, dtype=win.dtype) + self.window_pos[:, :n_win, :].to(device=win.device, dtype=win.dtype)
        win = self._normalize_tokens(win)
        if n_win < self.max_windows:
            pad = win.new_zeros((batch, self.max_windows - n_win, self.hidden))
            win = torch.cat([win, pad], dim=1)
            mask_pad = torch.zeros((batch, self.max_windows - n_win), dtype=torch.bool, device=window_mask.device)
            window_mask = torch.cat([window_mask[:, :n_win].bool(), mask_pad], dim=1)
        else:
            window_mask = window_mask[:, : self.max_windows].bool()

        tokens = torch.cat([global_tok, win], dim=1)
        global_mask = torch.ones((batch, self.n_global_tokens), dtype=torch.bool, device=window_mask.device)
        soft_mask = torch.cat([global_mask, window_mask], dim=1)
        if self.n_speaker_tokens > 0:
            if self.speaker_adapter is None or self.speaker_type is None:
                raise RuntimeError("speaker token bank requested without speaker modules")
            if speaker_emb is None:
                speaker_emb = global_emb.new_zeros((batch, self.speaker_dim), dtype=torch.float32)
                speaker_mask = torch.zeros((batch,), dtype=torch.bool, device=global_emb.device)
            else:
                speaker_emb = speaker_emb.to(device=global_emb.device, dtype=torch.float32)
                speaker_mask = (
                    torch.ones((batch,), dtype=torch.bool, device=global_emb.device)
                    if speaker_mask is None
                    else speaker_mask.to(device=global_emb.device).bool()
                )
            spk = self.speaker_adapter(speaker_emb.float()).view(batch, self.n_speaker_tokens, self.hidden)
            spk = spk + self.speaker_type.to(device=spk.device, dtype=spk.dtype)
            spk = self._normalize_tokens(spk)
            spk_mask = speaker_mask[:, None].expand(batch, self.n_speaker_tokens)
            tokens = torch.cat([tokens, spk], dim=1)
            soft_mask = torch.cat([soft_mask, spk_mask], dim=1)
        return tokens, soft_mask

    def prompt_embeds(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch = int(global_emb.shape[0])
        soft, soft_mask = self.soft_tokens(global_emb, window_emb, window_mask, speaker_emb=speaker_emb, speaker_mask=speaker_mask)
        embed_dtype = self.emb_layer.weight.dtype
        pre = prefix.to(device=global_emb.device, dtype=embed_dtype).unsqueeze(0).expand(batch, -1, -1)
        suf = suffix.to(device=global_emb.device, dtype=embed_dtype).unsqueeze(0).expand(batch, -1, -1)
        seq = torch.cat([pre, soft, suf], dim=1)
        pre_mask = torch.ones((batch, prefix.shape[0]), dtype=torch.bool, device=global_emb.device)
        suf_mask = torch.ones((batch, suffix.shape[0]), dtype=torch.bool, device=global_emb.device)
        attn_mask = torch.cat([pre_mask, soft_mask, suf_mask], dim=1)
        return seq, attn_mask, int(prefix.shape[0])

    def pool_hidden(self, hidden_states: torch.Tensor, prefix_len: int, prompt_len: int, soft_mask: torch.Tensor) -> torch.Tensor:
        if self.readout == "last":
            return hidden_states[:, prompt_len - 1, :].float()
        soft_len = self.soft_token_count
        states = hidden_states[:, prefix_len : prefix_len + soft_len, :].float()
        if self.readout == "meanpool":
            weights = soft_mask.float().unsqueeze(-1)
            return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if self.readout == "attnpool":
            assert self.attn is not None
            return self.attn(states, soft_mask)
        raise ValueError(f"unsupported readout: {self.readout}")

    def heads_from_hidden(self, hidden_states: torch.Tensor, prefix_len: int, prompt_len: int, soft_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool_hidden(hidden_states, prefix_len, prompt_len, soft_mask)
        lr = self.lr_head(pooled).float()
        verdict = self.verdict_head(pooled).squeeze(-1).float()
        return lr, verdict

    def _project_region_windows(self, window_emb: torch.Tensor) -> torch.Tensor:
        return self.window_adapter(window_emb.float())

    def region_logits_from_windows(self, window_emb: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        if self.region_seq is None or self.region_win_cls is None:
            raise RuntimeError("windowed region head requested without region_seq/region_win_cls modules")
        batch = int(window_emb.shape[0])
        n_win = min(int(window_emb.shape[1]), self.max_windows)
        logits = window_emb.new_zeros((batch, self.max_windows), dtype=torch.float32)
        if n_win <= 0:
            return logits
        win = self._project_region_windows(window_emb[:, :n_win, :].float())
        win = win + self.window_type.to(device=win.device, dtype=win.dtype) + self.window_pos[:, :n_win, :].to(device=win.device, dtype=win.dtype)
        win = self._normalize_tokens(win).float()
        valid = window_mask[:, :n_win].to(device=win.device).bool()
        win = win * valid.unsqueeze(-1).to(dtype=win.dtype)
        # ROCm: fused MIOpen RNN crashes (SIGSEGV) under bf16 autocast — force the GRU to FP32 with autocast disabled.
        with torch.autocast(device_type=win.device.type, enabled=False):
            ctx, _ = self.region_seq(win.float())
            logits[:, :n_win] = self.region_win_cls(ctx.float()).squeeze(-1).float()
        return logits

    def uni_heads_from_hidden(
        self,
        hidden_states: torch.Tensor,
        prefix_len: int,
        prompt_len: int,
        soft_mask: torch.Tensor,
        window_emb: torch.Tensor | None = None,
        window_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self.pool_hidden(hidden_states, prefix_len, prompt_len, soft_mask)
        if not self.enable_region:
            region = pooled.new_zeros((pooled.shape[0], 0), dtype=torch.float32)
        elif self.region_head_mode == "windowed":
            if window_emb is None or window_mask is None:
                raise ValueError("windowed region head requires window_emb and window_mask")
            region = self.region_logits_from_windows(window_emb, window_mask)
        else:
            region = self.region_head(pooled).float()
        if self.speaker_readout_head is None:
            speaker = pooled.new_zeros((pooled.shape[0], 0), dtype=torch.float32)
        else:
            speaker = self.speaker_readout_head(pooled).float()
        out = {
            "lr": self.lr_head(pooled).float(),
            "verdict": self.verdict_head(pooled).squeeze(-1).float(),
            "speaker": speaker,
            "region": region,
        }
        if self.method_head is None:
            out["method"] = pooled.new_zeros((pooled.shape[0], 0), dtype=torch.float32)
        else:
            out["method"] = self.method_head(pooled).float()
        return out

    def forward_scores(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt, attn_mask, prefix_len = self.prompt_embeds(
            global_emb, window_emb, window_mask, prefix, suffix, speaker_emb=speaker_emb, speaker_mask=speaker_mask
        )
        soft_mask = attn_mask[:, prefix_len : prefix_len + self.soft_token_count]
        with self.backbone_context():
            out = self.backbone(inputs_embeds=prompt, attention_mask=attn_mask.long(), use_cache=False, return_dict=True)
        return self.heads_from_hidden(out.last_hidden_state, prefix_len, prompt.shape[1], soft_mask)

    def forward_joint(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        return_scores: bool = True,
        verdict_prefix_ids: torch.Tensor | None = None,
        verdict_prefix_mask: torch.Tensor | None = None,
        verdict_prefix_lens: torch.Tensor | None = None,
        pos_ids: list[int] | tuple[int, ...] | None = None,
        neg_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        prompt, prompt_mask, prefix_len = self.prompt_embeds(
            global_emb, window_emb, window_mask, prefix, suffix, speaker_emb=speaker_emb, speaker_mask=speaker_mask
        )
        prev_ids = target_ids[:, :-1]
        prev_mask = target_mask[:, :-1]
        prev_emb = self.emb_layer(prev_ids).to(dtype=prompt.dtype)
        seq = torch.cat([prompt, prev_emb], dim=1)
        attn_mask = torch.cat([prompt_mask, prev_mask.bool()], dim=1)
        prompt_len = prompt.shape[1]
        with self.backbone_context():
            out = self.backbone(inputs_embeds=seq, attention_mask=attn_mask.long(), use_cache=False, return_dict=True)
            hidden = out.last_hidden_state
        valid = target_mask.reshape(-1).bool()
        if torch.any(valid):
            lm_hidden = hidden[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[1], :].reshape(-1, hidden.shape[-1])[valid]
            head_dtype = next(self.lm_head.parameters()).dtype
            logits = self.lm_head(lm_hidden.to(dtype=head_dtype))
            lm_loss = F.cross_entropy(logits.float(), target_ids.reshape(-1)[valid])
        else:
            lm_loss = hidden.sum() * 0.0
        if return_scores:
            soft_mask = prompt_mask[:, prefix_len : prefix_len + self.soft_token_count]
            lr, verdict = self.heads_from_hidden(hidden, prefix_len, prompt_len, soft_mask)
        else:
            lr = hidden.new_zeros((global_emb.shape[0], 4), dtype=torch.float32)
            verdict = hidden.new_zeros((global_emb.shape[0],), dtype=torch.float32)
        if verdict_prefix_ids is not None:
            if verdict_prefix_mask is None or verdict_prefix_lens is None or pos_ids is None or neg_ids is None:
                raise ValueError("token calibration requested without complete verdict-prefix inputs")
            tok_margin, tok_llr = self.forward_verdict_token_margin(
                global_emb,
                window_emb,
                window_mask,
                prefix,
                suffix,
                verdict_prefix_ids,
                verdict_prefix_mask,
                verdict_prefix_lens,
                pos_ids,
                neg_ids,
                speaker_emb=speaker_emb,
                speaker_mask=speaker_mask,
            )
            return lr, verdict, lm_loss.float(), tok_margin, tok_llr
        return lr, verdict, lm_loss.float()

    def forward_uni_joint(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        return_scores: bool = True,
        verdict_prefix_ids: torch.Tensor | None = None,
        verdict_prefix_mask: torch.Tensor | None = None,
        verdict_prefix_lens: torch.Tensor | None = None,
        pos_ids: list[int] | tuple[int, ...] | None = None,
        neg_ids: list[int] | tuple[int, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        prompt, prompt_mask, prefix_len = self.prompt_embeds(
            global_emb, window_emb, window_mask, prefix, suffix, speaker_emb=speaker_emb, speaker_mask=speaker_mask
        )
        prev_ids = target_ids[:, :-1]
        prev_mask = target_mask[:, :-1]
        prev_emb = self.emb_layer(prev_ids).to(dtype=prompt.dtype)
        seq = torch.cat([prompt, prev_emb], dim=1)
        attn_mask = torch.cat([prompt_mask, prev_mask.bool()], dim=1)
        prompt_len = prompt.shape[1]
        with self.backbone_context():
            out = self.backbone(inputs_embeds=seq, attention_mask=attn_mask.long(), use_cache=False, return_dict=True)
            hidden = out.last_hidden_state
        valid = target_mask.reshape(-1).bool()
        if torch.any(valid):
            lm_hidden = hidden[:, prompt_len - 1 : prompt_len - 1 + target_ids.shape[1], :].reshape(-1, hidden.shape[-1])[valid]
            head_dtype = next(self.lm_head.parameters()).dtype
            logits = self.lm_head(lm_hidden.to(dtype=head_dtype))
            lm_loss = F.cross_entropy(logits.float(), target_ids.reshape(-1)[valid])
        else:
            lm_loss = hidden.sum() * 0.0
        if return_scores:
            soft_mask = prompt_mask[:, prefix_len : prefix_len + self.soft_token_count]
            heads = self.uni_heads_from_hidden(hidden, prefix_len, prompt_len, soft_mask, window_emb=window_emb, window_mask=window_mask)
        else:
            heads = {
                "lr": hidden.new_zeros((global_emb.shape[0], 4), dtype=torch.float32),
                "verdict": hidden.new_zeros((global_emb.shape[0],), dtype=torch.float32),
                "speaker": hidden.new_zeros(
                    (global_emb.shape[0], self.speaker_dim if self.speaker_readout_head is not None else 0),
                    dtype=torch.float32,
                ),
                "region": hidden.new_zeros(
                    (global_emb.shape[0], self.max_windows if self.enable_region else 0),
                    dtype=torch.float32,
                ),
                "method": hidden.new_zeros((global_emb.shape[0], self.num_methods), dtype=torch.float32),
            }
        heads["lm_loss"] = lm_loss.float()
        if verdict_prefix_ids is not None:
            if verdict_prefix_mask is None or verdict_prefix_lens is None or pos_ids is None or neg_ids is None:
                raise ValueError("token calibration requested without complete verdict-prefix inputs")
            tok_margin, tok_llr = self.forward_verdict_token_margin(
                global_emb,
                window_emb,
                window_mask,
                prefix,
                suffix,
                verdict_prefix_ids,
                verdict_prefix_mask,
                verdict_prefix_lens,
                pos_ids,
                neg_ids,
                speaker_emb=speaker_emb,
                speaker_mask=speaker_mask,
            )
            heads["tok_margin"] = tok_margin.float()
            heads["tok_llr"] = tok_llr.float()
        return heads

    def calibrate_verdict_margin(self, margin: torch.Tensor) -> torch.Tensor:
        return self.tokcal_scale.float() * margin.float() + self.tokcal_bias.float()

    def forward_verdict_token_margin(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        verdict_prefix_ids: torch.Tensor,
        verdict_prefix_mask: torch.Tensor,
        verdict_prefix_lens: torch.Tensor,
        pos_ids: list[int] | tuple[int, ...],
        neg_ids: list[int] | tuple[int, ...],
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_logp = self._verdict_label_logprob(
            global_emb,
            window_emb,
            window_mask,
            prefix,
            suffix,
            verdict_prefix_ids,
            verdict_prefix_mask,
            verdict_prefix_lens,
            [int(item) for item in pos_ids],
            speaker_emb=speaker_emb,
            speaker_mask=speaker_mask,
        )
        neg_logp = self._verdict_label_logprob(
            global_emb,
            window_emb,
            window_mask,
            prefix,
            suffix,
            verdict_prefix_ids,
            verdict_prefix_mask,
            verdict_prefix_lens,
            [int(item) for item in neg_ids],
            speaker_emb=speaker_emb,
            speaker_mask=speaker_mask,
        )
        margin = pos_logp - neg_logp
        return margin.float(), self.calibrate_verdict_margin(margin).float()

    def _verdict_label_logprob(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        verdict_prefix_ids: torch.Tensor,
        verdict_prefix_mask: torch.Tensor,
        verdict_prefix_lens: torch.Tensor,
        label_ids: list[int],
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not label_ids:
            raise ValueError("label_ids must not be empty")
        prompt, prompt_mask, _prefix_len = self.prompt_embeds(
            global_emb, window_emb, window_mask, prefix, suffix, speaker_emb=speaker_emb, speaker_mask=speaker_mask
        )
        batch = int(global_emb.shape[0])
        label_len = len(label_ids)
        forced_width = int(verdict_prefix_ids.shape[1]) + max(0, label_len - 1)
        forced_ids = torch.full(
            (batch, forced_width),
            int(verdict_prefix_ids[0, -1].item()),
            dtype=verdict_prefix_ids.dtype,
            device=verdict_prefix_ids.device,
        )
        forced_mask = torch.zeros((batch, forced_width), dtype=torch.bool, device=verdict_prefix_ids.device)
        for row in range(batch):
            plen = int(verdict_prefix_lens[row].item())
            forced_ids[row, :plen] = verdict_prefix_ids[row, :plen]
            forced_mask[row, :plen] = True
            if label_len > 1:
                label_prev = torch.tensor(label_ids[:-1], dtype=verdict_prefix_ids.dtype, device=verdict_prefix_ids.device)
                forced_ids[row, plen : plen + label_len - 1] = label_prev
                forced_mask[row, plen : plen + label_len - 1] = True
        forced_emb = self.emb_layer(forced_ids).to(dtype=prompt.dtype)
        seq = torch.cat([prompt, forced_emb], dim=1)
        attn_mask = torch.cat([prompt_mask, forced_mask], dim=1)
        prompt_len = int(prompt.shape[1])
        with self.backbone_context():
            out = self.backbone(inputs_embeds=seq, attention_mask=attn_mask.long(), use_cache=False, return_dict=True)
            head_dtype = next(self.lm_head.parameters()).dtype
            logprob = torch.zeros((batch,), dtype=torch.float32, device=global_emb.device)
            row_idx = torch.arange(global_emb.shape[0], device=global_emb.device)
            for offset, token_id in enumerate(label_ids):
                positions = prompt_len + verdict_prefix_lens.long() - 1 + offset
                hidden = out.last_hidden_state[row_idx, positions, :]
                logits = self.lm_head(hidden.to(dtype=head_dtype)).float()
                logprob = logprob + torch.log_softmax(logits, dim=-1)[:, int(token_id)]
        return logprob

    @torch.no_grad()
    def generate_ids(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        *,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        eos_id: int,
        pad_id: int,
    ) -> torch.Tensor:
        prompt, prompt_mask, _prefix_len = self.prompt_embeds(
            global_emb, window_emb, window_mask, prefix, suffix, speaker_emb=speaker_emb, speaker_mask=speaker_mask
        )
        batch = int(prompt.shape[0])
        out_ids = torch.full((batch, max_new_tokens), pad_id, dtype=torch.long, device=prompt.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=prompt.device)
        for step in range(max_new_tokens):
            prev = out_ids[:, :step]
            if step:
                prev_emb = self.emb_layer(prev).to(dtype=prompt.dtype)
                seq = torch.cat([prompt, prev_emb], dim=1)
                prev_mask = ~prev.eq(pad_id)
                attn_mask = torch.cat([prompt_mask, prev_mask], dim=1)
            else:
                seq = prompt
                attn_mask = prompt_mask
            with self.backbone_context():
                out = self.backbone(inputs_embeds=seq, attention_mask=attn_mask.long(), use_cache=False, return_dict=True)
                logits = self.lm_head(out.last_hidden_state[:, -1:, :])[:, -1, :].float()
            next_id = torch.argmax(logits, dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, pad_id), next_id)
            out_ids[:, step] = next_id
            finished = finished | next_id.eq(eos_id)
            if bool(finished.all()):
                break
        return out_ids

    def forward(self, task: str, **kwargs):
        if task == "scores":
            return self.forward_scores(**kwargs)
        if task == "joint":
            return self.forward_joint(**kwargs)
        if task == "uni_joint":
            return self.forward_uni_joint(**kwargs)
        if task == "tokcal":
            return self.forward_verdict_token_margin(**kwargs)
        raise ValueError(f"unsupported task: {task}")


class SpoofStudentMHFAReasoner(SpoofStudentReasoner):
    """Frozen XLS-R hidden states -> trainable MHFA -> existing Qwen student."""

    def __init__(
        self,
        speech_ssl_model: nn.Module,
        lm_backbone: nn.Module,
        lm_head: nn.Module,
        emb_layer: nn.Module,
        hidden: int,
        *,
        n_global_tokens: int = 8,
        max_windows: int = 30,
        n_speaker_tokens: int = 0,
        speaker_dim: int = 256,
        num_methods: int = 0,
        readout: str = "last",
        region_head: str = "global",
        enable_region: bool = True,
        enable_speaker: bool = True,
        input_embed_rms: float = 1.0,
        use_bf16_autocast: bool = True,
        use_speech_bf16_autocast: bool = True,
        mhfa_nb_layer: int = 49,
        mhfa_feat_dim: int = 1280,
        mhfa_head_nb: int = 64,
        mhfa_compression_dim: int = 128,
        mhfa_embed_dim: int = 256,
        window_s: float = 0.4,
        hop_s: float = 0.2,
        sample_rate: int = 16000,
        ssl_input_samples: int = 64600,
        ssl_batch_size: int = 1,
        train_speech_encoder: bool = False,
        encoder_train_top_k: int = 0,
    ):
        super().__init__(
            lm_backbone,
            lm_head,
            emb_layer,
            hidden,
            in_dim=mhfa_embed_dim,
            n_global_tokens=n_global_tokens,
            max_windows=max_windows,
            n_speaker_tokens=n_speaker_tokens,
            speaker_dim=speaker_dim,
            num_methods=num_methods,
            readout=readout,
            region_head=region_head,
            enable_region=enable_region,
            enable_speaker=enable_speaker,
            input_embed_rms=input_embed_rms,
            use_bf16_autocast=use_bf16_autocast,
        )
        for param in self.window_adapter.parameters():
            param.requires_grad = False
        self.region_adapter = nn.Linear(mhfa_compression_dim, hidden)
        self.train_speech_encoder = bool(train_speech_encoder)
        self.speech_ssl_model = speech_ssl_model
        if self.train_speech_encoder:
            # v3.2.2: low-LR finetune of the XLS-R encoder (grads flow; see train()/_mhfa_dual no_grad gates)
            self.speech_ssl_model.train()
            top_k = int(encoder_train_top_k)
            if top_k > 0:
                # depth axis: unfreeze only the top-K transformer layers; rest stay frozen
                for param in self.speech_ssl_model.parameters():
                    param.requires_grad = False
                enc_layers = self.speech_ssl_model.encoder.layers
                k = min(top_k, len(enc_layers))
                for layer in enc_layers[-k:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                ln = getattr(self.speech_ssl_model.encoder, "layer_norm", None)
                if ln is not None:
                    for param in ln.parameters():
                        param.requires_grad = True
            else:
                for param in self.speech_ssl_model.parameters():
                    param.requires_grad = True
        else:
            self.speech_ssl_model.eval()
            for param in self.speech_ssl_model.parameters():
                param.requires_grad = False

        self.mhfa = SSL_BACKEND_MHFA_LayerNorm(
            head_nb=mhfa_head_nb,
            feat_dim=mhfa_feat_dim,
            compression_dim=mhfa_compression_dim,
            embed_dim=mhfa_embed_dim,
            nb_layer=mhfa_nb_layer,
            feature_grad_mult=1.0,
            projection_dim=1,
        )
        for param in self.mhfa.projection.parameters():
            param.requires_grad = False

        self.use_speech_bf16_autocast = bool(use_speech_bf16_autocast)
        self.mhfa_nb_layer = int(mhfa_nb_layer)
        self.mhfa_compression_dim = int(mhfa_compression_dim)
        self.mhfa_embed_dim = int(mhfa_embed_dim)
        self.sample_rate = int(sample_rate)
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        self.window_samples = max(1, int(round(self.window_s * self.sample_rate)))
        self.hop_samples = max(1, int(round(self.hop_s * self.sample_rate)))
        self.ssl_input_samples = max(1, int(ssl_input_samples))
        self.ssl_batch_size = max(1, int(ssl_batch_size))

    def _project_region_windows(self, window_emb: torch.Tensor) -> torch.Tensor:
        return self.region_adapter(window_emb.float())

    def train(self, mode: bool = True):
        super().train(mode)
        if not getattr(self, "train_speech_encoder", False):
            self.speech_ssl_model.eval()
        return self

    def speech_context(self) -> contextlib.AbstractContextManager:
        if self.use_speech_bf16_autocast and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def soft_tokens(
        self,
        global_emb: torch.Tensor,
        window_emb: torch.Tensor,
        window_mask: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(global_emb.shape[0])
        global_tok = self.global_adapter(global_emb.float()).view(batch, self.n_global_tokens, self.hidden)
        global_tok = global_tok + self.global_type.to(device=global_tok.device, dtype=global_tok.dtype)
        global_tok = self._normalize_tokens(global_tok)

        n_win = min(int(window_emb.shape[1]), self.max_windows)
        win = self.region_adapter(window_emb[:, :n_win, :].float())
        win = win + self.window_type.to(device=win.device, dtype=win.dtype) + self.window_pos[:, :n_win, :].to(device=win.device, dtype=win.dtype)
        win = self._normalize_tokens(win)
        if n_win < self.max_windows:
            pad = win.new_zeros((batch, self.max_windows - n_win, self.hidden))
            win = torch.cat([win, pad], dim=1)
            mask_pad = torch.zeros((batch, self.max_windows - n_win), dtype=torch.bool, device=window_mask.device)
            window_mask = torch.cat([window_mask[:, :n_win].bool(), mask_pad], dim=1)
        else:
            window_mask = window_mask[:, : self.max_windows].bool()

        tokens = torch.cat([global_tok, win], dim=1)
        global_mask = torch.ones((batch, self.n_global_tokens), dtype=torch.bool, device=window_mask.device)
        soft_mask = torch.cat([global_mask, window_mask], dim=1)
        if self.n_speaker_tokens > 0:
            if self.speaker_adapter is None or self.speaker_type is None:
                raise RuntimeError("speaker token bank requested without speaker modules")
            if speaker_emb is None:
                speaker_emb = global_emb.new_zeros((batch, self.speaker_dim), dtype=torch.float32)
                speaker_mask = torch.zeros((batch,), dtype=torch.bool, device=global_emb.device)
            else:
                speaker_emb = speaker_emb.to(device=global_emb.device, dtype=torch.float32)
                speaker_mask = (
                    torch.ones((batch,), dtype=torch.bool, device=global_emb.device)
                    if speaker_mask is None
                    else speaker_mask.to(device=global_emb.device).bool()
                )
            spk = self.speaker_adapter(speaker_emb.float()).view(batch, self.n_speaker_tokens, self.hidden)
            spk = spk + self.speaker_type.to(device=spk.device, dtype=spk.dtype)
            spk = self._normalize_tokens(spk)
            spk_mask = speaker_mask[:, None].expand(batch, self.n_speaker_tokens)
            tokens = torch.cat([tokens, spk], dim=1)
            soft_mask = torch.cat([soft_mask, spk_mask], dim=1)
        return tokens, soft_mask

    def _dfarena_repeat_pad(self, wav: torch.Tensor, wav_len: torch.Tensor) -> torch.Tensor:
        """Match DF-Arena's local feature extractor: truncate or repeat-pad to 64600 samples."""
        rows: list[torch.Tensor] = []
        for idx in range(int(wav.shape[0])):
            length = max(1, min(int(wav_len[idx].item()), int(wav.shape[1])))
            piece = wav[idx, :length].float()
            if piece.numel() >= self.ssl_input_samples:
                rows.append(piece[: self.ssl_input_samples])
            else:
                repeats = int(self.ssl_input_samples / max(1, piece.numel())) + 1
                rows.append(piece.repeat(repeats)[: self.ssl_input_samples])
        return torch.stack(rows, dim=0)

    def _window_audio(self, wav: torch.Tensor, wav_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        windows: list[torch.Tensor] = []
        owners: list[tuple[int, int]] = []
        mask = torch.zeros((wav.shape[0], self.max_windows), dtype=torch.bool, device=wav.device)
        for row in range(int(wav.shape[0])):
            length = max(1, min(int(wav_len[row].item()), int(wav.shape[1])))
            starts = list(range(0, max(1, length - self.window_samples + 1), self.hop_samples))
            starts = starts[: self.max_windows] or [0]
            for slot, start in enumerate(starts):
                piece = wav[row, start : min(start + self.window_samples, length)].float()
                if piece.numel() < self.window_samples:
                    piece = F.pad(piece, (0, self.window_samples - piece.numel()))
                windows.append(piece)
                owners.append((row, slot))
                mask[row, slot] = True
        return torch.stack(windows, dim=0), mask, owners

    def _frame_window_count(self) -> int:
        if self.ssl_input_samples <= self.window_samples:
            return 1
        return min(self.max_windows, 1 + max(0, (self.ssl_input_samples - self.window_samples) // self.hop_samples))

    def _window_frame_features(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frame_count, feat_dim = frames.shape
        n_win = self._frame_window_count()
        window_emb = frames.new_zeros((batch, self.max_windows, feat_dim))
        window_mask = torch.zeros((batch, self.max_windows), dtype=torch.bool, device=frames.device)
        frame_step_s = float(self.ssl_input_samples) / float(self.sample_rate) / float(max(1, frame_count))
        frame_centers = (torch.arange(frame_count, device=frames.device, dtype=torch.float32) + 0.5) * frame_step_s
        for slot in range(n_win):
            start_s = float(slot * self.hop_samples) / float(self.sample_rate)
            stop_s = start_s + self.window_s
            keep = (frame_centers >= start_s) & (frame_centers < stop_s)
            if bool(keep.any()):
                pooled = frames[:, keep, :].mean(dim=1)
            else:
                center = start_s + 0.5 * self.window_s
                nearest = int(torch.argmin(torch.abs(frame_centers - center)).item())
                pooled = frames[:, nearest, :]
            window_emb[:, slot, :] = pooled
            window_mask[:, slot] = True
        return window_emb, window_mask

    def _mhfa_dual_from_prepared_ssl_inputs(self, prepared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_outputs: list[torch.Tensor] = []
        window_outputs: list[torch.Tensor] = []
        mask_outputs: list[torch.Tensor] = []
        for start in range(0, int(prepared.shape[0]), self.ssl_batch_size):
            end = min(start + self.ssl_batch_size, int(prepared.shape[0]))
            with _profile_phase("A1_ssl_fwd"):
                grad_ctx = contextlib.nullcontext() if self.train_speech_encoder else torch.no_grad()
                with grad_ctx, self.speech_context():
                    ssl_out = self.speech_ssl_model(
                        prepared[start:end],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    if self.train_speech_encoder:
                        states = tuple(state.float() for state in ssl_out.hidden_states)
                    else:
                        states = tuple(state.detach().float() for state in ssl_out.hidden_states)
            if len(states) != self.mhfa_nb_layer:
                raise RuntimeError(f"expected {self.mhfa_nb_layer} XLS-R hidden states, got {len(states)}")
            with _profile_phase("A2_mhfa"):
                mhfa_input = mhfa_input_from_hidden_states(states, kept_layers=self.mhfa_nb_layer)
                mhfa_out = self.mhfa.dual_readout(mhfa_input)
            global_outputs.append(mhfa_out["embedding"].float())
            with _profile_phase("A3_window_loop"):
                window_emb, window_mask = self._window_frame_features(mhfa_out["frame_features"].float())
            window_outputs.append(window_emb)
            mask_outputs.append(window_mask)
        return torch.cat(global_outputs, dim=0), torch.cat(window_outputs, dim=0), torch.cat(mask_outputs, dim=0)

    def extract_mhfa_embeddings(self, wav: torch.Tensor, wav_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wav = wav.float()
        wav_len = wav_len.to(device=wav.device, dtype=torch.long)
        global_inputs = self._dfarena_repeat_pad(wav, wav_len)
        return self._mhfa_dual_from_prepared_ssl_inputs(global_inputs)

    def forward_scores_audio(
        self,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_emb, window_emb, window_mask = self.extract_mhfa_embeddings(wav, wav_len)
        return self.forward_scores(global_emb, window_emb, window_mask, prefix, suffix, speaker_emb=speaker_emb, speaker_mask=speaker_mask)

    def forward_joint_audio(
        self,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        global_emb, window_emb, window_mask = self.extract_mhfa_embeddings(wav, wav_len)
        return self.forward_joint(
            global_emb,
            window_emb,
            window_mask,
            prefix,
            suffix,
            target_ids,
            target_mask,
            **kwargs,
        )

    def forward_uni_joint_audio(
        self,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        with _profile_phase("A_encoder_mhfa"):
            global_emb, window_emb, window_mask = self.extract_mhfa_embeddings(wav, wav_len)
        with _profile_phase("B_llm_fwd"):
            heads = self.forward_uni_joint(
                global_emb,
                window_emb,
                window_mask,
                prefix,
                suffix,
                target_ids,
                target_mask,
                **kwargs,
            )
        _profile_tick()
        heads["window_mask"] = window_mask
        return heads

    @torch.no_grad()
    def generate_ids_audio(
        self,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int,
        pad_id: int,
        speaker_emb: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global_emb, window_emb, window_mask = self.extract_mhfa_embeddings(wav, wav_len)
        return self.generate_ids(
            global_emb,
            window_emb,
            window_mask,
            prefix,
            suffix,
            speaker_emb=speaker_emb,
            speaker_mask=speaker_mask,
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
            pad_id=pad_id,
        )

    def mhfa_layer_profile(self, top_k: int = 8) -> dict[str, Any]:
        def profile(name: str, weights: torch.Tensor) -> dict[str, Any]:
            probs = torch.softmax(weights.detach().float().cpu(), dim=-1)
            top = torch.topk(probs, k=min(int(top_k), int(probs.numel())))
            return {
                "name": name,
                "nb_layer": int(probs.numel()),
                "top": [
                    {"layer": int(idx), "weight": float(value)}
                    for value, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True)
                ],
                "weights": [float(item) for item in probs.tolist()],
            }

        return {
            "weights_k": profile("weights_k", self.mhfa.weights_k),
            "weights_v": profile("weights_v", self.mhfa.weights_v),
        }

    def forward(self, task: str, **kwargs):
        if task == "scores_audio":
            return self.forward_scores_audio(**kwargs)
        if task == "joint_audio":
            return self.forward_joint_audio(**kwargs)
        if task == "uni_joint_audio":
            return self.forward_uni_joint_audio(**kwargs)
        return super().forward(task, **kwargs)


def force_trainable_fp32(model: nn.Module) -> dict[str, int]:
    converted = 0
    trainable = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        trainable += param.numel()
        if param.dtype != torch.float32:
            param.data = param.data.float()
            converted += param.numel()
    return {"trainable": int(trainable), "converted_to_fp32": int(converted)}


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
    own = dict(model.named_parameters())
    missing: list[str] = []
    unexpected: list[str] = []
    for name, value in state.items():
        if name not in own:
            unexpected.append(name)
            continue
        own[name].data.copy_(value.to(device=own[name].device, dtype=own[name].dtype))
    if strict:
        for name, param in own.items():
            if param.requires_grad and name not in state:
                missing.append(name)
        if missing or unexpected:
            raise RuntimeError(f"trainable state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    return {"missing": missing, "unexpected": unexpected}


def input_embedding_stats(emb_layer: nn.Module) -> dict[str, float]:
    with torch.no_grad():
        weight = emb_layer.weight.detach()
        if weight.shape[0] > 8192:
            idx = torch.linspace(0, weight.shape[0] - 1, 8192, device=weight.device).long()
            sample = weight.index_select(0, idx)
        else:
            sample = weight
        return {
            "rms": float(sample.float().pow(2).mean(dim=-1).sqrt().mean().item()),
            "norm": float(sample.float().norm(dim=-1).mean().item()),
        }


def text_embedding(tok: Any, emb_layer: nn.Module, text: str, bos: bool, device: torch.device) -> torch.Tensor:
    ids = tok(text, add_special_tokens=bos).input_ids
    with torch.no_grad():
        return emb_layer(torch.tensor(ids, device=device)).float()


def build_qwen_student(
    llm: str | Path | None = DEFAULT_LLM,
    *,
    device: torch.device,
    lora: bool,
    readout: str,
    n_global_tokens: int = 8,
    max_windows: int = 30,
    n_speaker_tokens: int = 0,
    speaker_dim: int = 256,
    num_methods: int = 0,
    in_dim: int = 1280,
    region_head: str = "global",
    llm_fp32_forward: bool = False,
    attn_implementation: str = "eager",
) -> tuple[SpoofStudentReasoner, Any, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]:
    llm = Path(DEFAULT_LLM if llm is None else llm)
    dtype = torch.float32 if llm_fp32_forward else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(llm, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = int(tok.eos_token_id)
    pad_id = int(tok.pad_token_id)

    causal = AutoModelForCausalLM.from_pretrained(
        llm,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation=attn_implementation,
    ).to(device)
    for param in causal.parameters():
        param.requires_grad = False
    emb_layer = causal.get_input_embeddings()
    if lora:
        from peft import LoraConfig, get_peft_model

        causal.model = get_peft_model(
            causal.model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
    causal.config.use_cache = False
    for param in causal.lm_head.parameters():
        param.requires_grad = False
    stats = input_embedding_stats(emb_layer)
    prefix = text_embedding(tok, emb_layer, "Audio evidence tokens: ", True, device)
    suffix = text_embedding(tok, emb_layer, " Generate the spoof forensic trace using the fixed schema:\n", False, device)
    model = SpoofStudentReasoner(
        causal.model,
        causal.lm_head,
        emb_layer,
        int(causal.config.hidden_size),
        in_dim=in_dim,
        n_global_tokens=n_global_tokens,
        max_windows=max_windows,
        n_speaker_tokens=n_speaker_tokens,
        speaker_dim=speaker_dim,
        num_methods=num_methods,
        readout=readout,
        region_head=region_head,
        enable_region=enable_region,
        enable_speaker=enable_speaker,
        input_embed_rms=stats["rms"],
        use_bf16_autocast=not llm_fp32_forward,
    ).to(device)
    meta = {
        "llm": str(llm),
        "hidden": int(causal.config.hidden_size),
        "llm_dtype": str(dtype),
        "llm_fp32_forward": bool(llm_fp32_forward),
        "attn_implementation": attn_implementation,
        "lora": bool(lora),
        "readout": readout,
        "n_global_tokens": int(n_global_tokens),
        "max_windows": int(max_windows),
        "n_speaker_tokens": int(n_speaker_tokens),
        "speaker_dim": int(speaker_dim),
        "num_methods": int(num_methods),
        "in_dim": int(in_dim),
        "input_embed_rms": stats["rms"],
        "input_embed_norm": stats["norm"],
        "prompt_lens": {
            "prefix": int(prefix.shape[0]),
            "suffix": int(suffix.shape[0]),
            "soft": int(n_global_tokens + max_windows + n_speaker_tokens),
            "total": int(prefix.shape[0] + suffix.shape[0] + n_global_tokens + max_windows + n_speaker_tokens),
        },
    }
    if str(region_head) != "global":
        meta["region_head"] = str(region_head)
    return model, tok, prefix, suffix, eos_id, pad_id, meta


def build_qwen_mhfa_student(
    speech_ssl_model: nn.Module,
    llm: str | Path | None = DEFAULT_LLM,
    *,
    device: torch.device,
    lora: bool,
    readout: str,
    n_global_tokens: int = 8,
    max_windows: int = 30,
    n_speaker_tokens: int = 0,
    speaker_dim: int = 256,
    num_methods: int = 0,
    llm_fp32_forward: bool = False,
    speech_bf16_forward: bool = True,
    attn_implementation: str = "eager",
    region_head: str = "global",
    enable_region: bool = True,
    enable_speaker: bool = True,
    mhfa_nb_layer: int = 49,
    mhfa_feat_dim: int = 1280,
    mhfa_head_nb: int = 64,
    mhfa_compression_dim: int = 128,
    mhfa_embed_dim: int = 256,
    window_s: float = 0.4,
    hop_s: float = 0.2,
    ssl_input_samples: int = 64600,
    ssl_batch_size: int = 1,
    train_speech_encoder: bool = False,
    encoder_train_top_k: int = 0,
) -> tuple[SpoofStudentMHFAReasoner, Any, torch.Tensor, torch.Tensor, int, int, dict[str, Any]]:
    llm = Path(DEFAULT_LLM if llm is None else llm)
    dtype = torch.float32 if llm_fp32_forward else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(llm, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = int(tok.eos_token_id)
    pad_id = int(tok.pad_token_id)

    causal = AutoModelForCausalLM.from_pretrained(
        llm,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation=attn_implementation,
    ).to(device)
    for param in causal.parameters():
        param.requires_grad = False
    emb_layer = causal.get_input_embeddings()
    if lora:
        from peft import LoraConfig, get_peft_model

        causal.model = get_peft_model(
            causal.model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
    causal.config.use_cache = False
    for param in causal.lm_head.parameters():
        param.requires_grad = False
    stats = input_embedding_stats(emb_layer)
    prefix = text_embedding(tok, emb_layer, "Audio evidence tokens: ", True, device)
    suffix = text_embedding(tok, emb_layer, " Generate the spoof forensic trace using the fixed schema:\n", False, device)
    speech_ssl_model = speech_ssl_model.to(device)
    model = SpoofStudentMHFAReasoner(
        speech_ssl_model,
        causal.model,
        causal.lm_head,
        emb_layer,
        int(causal.config.hidden_size),
        n_global_tokens=n_global_tokens,
        max_windows=max_windows,
        n_speaker_tokens=n_speaker_tokens,
        speaker_dim=speaker_dim,
        num_methods=num_methods,
        readout=readout,
        region_head=region_head,
        enable_region=enable_region,
        enable_speaker=enable_speaker,
        input_embed_rms=stats["rms"],
        use_bf16_autocast=not llm_fp32_forward,
        use_speech_bf16_autocast=speech_bf16_forward,
        mhfa_nb_layer=mhfa_nb_layer,
        mhfa_feat_dim=mhfa_feat_dim,
        mhfa_head_nb=mhfa_head_nb,
        mhfa_compression_dim=mhfa_compression_dim,
        mhfa_embed_dim=mhfa_embed_dim,
        window_s=window_s,
        hop_s=hop_s,
        ssl_input_samples=ssl_input_samples,
        ssl_batch_size=ssl_batch_size,
        train_speech_encoder=train_speech_encoder,
        encoder_train_top_k=encoder_train_top_k,
    ).to(device)
    meta = {
        "llm": str(llm),
        "hidden": int(causal.config.hidden_size),
        "llm_dtype": str(dtype),
        "llm_fp32_forward": bool(llm_fp32_forward),
        "speech_bf16_forward": bool(speech_bf16_forward),
        "attn_implementation": attn_implementation,
        "lora": bool(lora),
        "readout": readout,
        "n_global_tokens": int(n_global_tokens),
        "max_windows": int(max_windows),
        "n_speaker_tokens": int(n_speaker_tokens),
        "speaker_dim": int(speaker_dim),
        "num_methods": int(num_methods),
        "in_dim": int(mhfa_embed_dim),
        "input_embed_rms": stats["rms"],
        "input_embed_norm": stats["norm"],
        "frontend": {
            "name": "dfarena_xlsr_1b_mhfa",
            "encoder": "DF-Arena backbone.ssl_model only",
            "encoder_frozen": True,
            "dfarena_conformer_used": False,
            "readout": "dual_shared_mhfa_global_pooled_region_frame_windowed",
            "global_adapter": "Linear(256,1024)->GELU->Linear(1024,n_global_tokens*hidden)->view(B,8,hidden)->LayerNorm/RMS",
            "region_adapter": "Linear(mhfa_compression_dim,hidden) over value-branch frame-window means",
            "region_tap": "v after weights_v layer weighting, cmp_linear_v, and norm_v; no second MHFA/window forward",
            "mhfa_nb_layer": int(mhfa_nb_layer),
            "mhfa_feat_dim": int(mhfa_feat_dim),
            "mhfa_head_nb": int(mhfa_head_nb),
            "mhfa_compression_dim": int(mhfa_compression_dim),
            "mhfa_embed_dim": int(mhfa_embed_dim),
            "window_s": float(window_s),
            "hop_s": float(hop_s),
            "ssl_input_samples": int(ssl_input_samples),
            "ssl_batch_size": int(ssl_batch_size),
            "projection": "unused frozen 1-way copy of speakerllm MHFA projection slot",
        },
        "prompt_lens": {
            "prefix": int(prefix.shape[0]),
            "suffix": int(suffix.shape[0]),
            "soft": int(n_global_tokens + max_windows + n_speaker_tokens),
            "total": int(prefix.shape[0] + suffix.shape[0] + n_global_tokens + max_windows + n_speaker_tokens),
        },
    }
    if str(region_head) != "global":
        meta["region_head"] = str(region_head)
    return model, tok, prefix, suffix, eos_id, pad_id, meta


def tokenize_texts(tok: Any, texts: Sequence[str], eos_id: int, pad_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ids_list: list[list[int]] = []
    for text in texts:
        ids = tok(str(text), add_special_tokens=False).input_ids + [int(eos_id)]
        ids_list.append([int(item) for item in ids])
    max_len = max(len(ids) for ids in ids_list)
    ids_np = np.full((len(ids_list), max_len), int(pad_id), dtype=np.int64)
    mask_np = np.zeros((len(ids_list), max_len), dtype=np.bool_)
    for row, ids in enumerate(ids_list):
        ids_np[row, : len(ids)] = np.asarray(ids, dtype=np.int64)
        mask_np[row, : len(ids)] = True
    return ids_np, mask_np, {"target_len": int(max_len), "unique_trace_targets": int(len(set(texts)))}


def verdict_token_info(tok: Any) -> dict[str, Any]:
    pos_ids = tok("spoof", add_special_tokens=False).input_ids
    neg_ids = tok("bonafide", add_special_tokens=False).input_ids
    return {
        "positive_label": "spoof",
        "negative_label": "bonafide",
        "positive_token_ids": [int(item) for item in pos_ids],
        "negative_token_ids": [int(item) for item in neg_ids],
        "single_token_labels": bool(len(pos_ids) == 1 and len(neg_ids) == 1),
        "positive_token_decode": tok.decode(pos_ids),
        "negative_token_decode": tok.decode(neg_ids),
        "score_definition": "a * (logp(spoof label tokens) - logp(bonafide label tokens)) + b at teacher-forced verdict slot",
    }


def verdict_prefix_before_label_from_text(text: str, verdict: str) -> str:
    marker = f"verdict={verdict}"
    if marker not in text:
        raise ValueError(f"could not locate verdict marker {marker!r} in trace {text!r}")
    before, _after = text.split(marker, 1)
    return before + "verdict="


def tokenize_verdict_prefixes(
    tok: Any,
    texts: Sequence[str],
    verdicts: Sequence[str],
    pad_id: int,
    token_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pos_ids = [int(item) for item in token_info["positive_token_ids"]]
    neg_ids = [int(item) for item in token_info["negative_token_ids"]]
    prefix_ids: list[list[int]] = []
    lengths: list[int] = []
    examples: list[dict[str, Any]] = []
    for row, (text, verdict) in enumerate(zip(texts, verdicts, strict=True)):
        prefix_text = verdict_prefix_before_label_from_text(str(text), str(verdict))
        ids = tok(prefix_text, add_special_tokens=False).input_ids
        full_ids = tok(str(text), add_special_tokens=False).input_ids
        expected = pos_ids if str(verdict) == "spoof" else neg_ids
        got = [int(item) for item in full_ids[len(ids) : len(ids) + len(expected)]]
        if not ids or len(full_ids) < len(ids) + len(expected) or got != expected:
            raise ValueError(f"verdict prefix sanity failed at row={row} verdict={verdict}")
        prefix_ids.append([int(item) for item in ids])
        lengths.append(len(ids))
        if len(examples) < 3:
            examples.append(
                {
                    "row": int(row),
                    "verdict": str(verdict),
                    "prefix_len_tokens": int(len(ids)),
                    "target_label_token_ids": got,
                    "prefix_tail_decode": tok.decode(ids[-12:]),
                }
            )
    max_len = max(lengths)
    ids_np = np.full((len(prefix_ids), max_len), int(pad_id), dtype=np.int64)
    mask_np = np.zeros((len(prefix_ids), max_len), dtype=np.bool_)
    for row, ids in enumerate(prefix_ids):
        ids_np[row, : len(ids)] = np.asarray(ids, dtype=np.int64)
        mask_np[row, : len(ids)] = True
    return ids_np, mask_np, np.asarray(lengths, dtype=np.int64), {
        "token_info": token_info,
        "n_prefixes": int(len(prefix_ids)),
        "prefix_len_min": int(min(lengths)),
        "prefix_len_max": int(max(lengths)),
        "prefix_len_mean": float(np.mean(lengths)),
        "sanity_examples": examples,
    }
