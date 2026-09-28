#!/usr/bin/env python
"""Loader for the DF-Arena-1B snapshot, which supplies the XLS-R-1B encoder.

SpoofLLM reuses the DF-Arena-1B repository only for its encoder architecture and
its custom Transformers modeling files; the SpoofLLM checkpoint then overwrites
the encoder weights. The snapshot asks Transformers for the XLS-R-1B config by
name, so this module supplies that config locally and keeps the load offline.

`load_model(path)` returns the DF-Arena-1B module. Running this file directly is
a standalone check of that snapshot on its own:

    python df_arena_loader.py --ckpt /path/to/df_arena_1b [--audio speech.wav]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Config

sys.dont_write_bytecode = True


DF_ARENA_HF_REPO = "Speech-Arena-2025/DF_Arena_1B_V_1"   # config.json + modeling_antispoofing.py + pytorch_model.bin


def xlsr_1b_config() -> Wav2Vec2Config:
    return Wav2Vec2Config(
        hidden_size=1280,
        num_hidden_layers=48,
        num_attention_heads=16,
        intermediate_size=5120,
        feat_extract_norm="layer",
        feat_proj_dropout=0.0,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        hidden_act="gelu",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=True,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=True,
        output_hidden_states=True,
    )


def patch_wav2vec2_config() -> None:
    original = Wav2Vec2Config.from_pretrained

    @classmethod
    def offline_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if str(pretrained_model_name_or_path) == "facebook/wav2vec2-xls-r-1b":
            return xlsr_1b_config()
        return original(pretrained_model_name_or_path, *args, **kwargs)

    Wav2Vec2Config.from_pretrained = offline_from_pretrained


def import_snapshot_package(ckpt: Path):
    pkg_name = f"_df_arena_1b_{abs(hash(str(ckpt.resolve())))}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ckpt)]
    sys.modules[pkg_name] = pkg
    return (
        importlib.import_module(f"{pkg_name}.configuration_antispoofing"),
        importlib.import_module(f"{pkg_name}.modeling_antispoofing"),
    )


def load_model(ckpt: Path):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    patch_wav2vec2_config()
    cfg_mod, model_mod = import_snapshot_package(ckpt)
    config = cfg_mod.DF_Arena_1B_Config.from_pretrained(ckpt)
    model = model_mod.DF_Arena_1B_Antispoofing(config)
    state = torch.load(ckpt / "pytorch_model.bin", map_location="cpu", mmap=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    return model


def load_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).numpy().astype(np.float32)


def prepare_input(model, audio: np.ndarray) -> torch.Tensor:
    features = model.feature_extractor(audio, sampling_rate=16000)
    values = features["input_values"]
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    return values.float()


def forward_one(model, audio: np.ndarray) -> tuple[np.ndarray, float, int]:
    cache = {}

    def capture_ssl(_module, _inputs, output):
        cache["hidden_states"] = output.hidden_states

    handle = model.backbone.ssl_model.register_forward_hook(capture_ssl)
    try:
        input_values = prepare_input(model, audio)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(input_values)
        elapsed = time.perf_counter() - start
    finally:
        handle.remove()

    logits = outputs["logits"].detach().cpu().float().numpy()
    hidden_states = cache.get("hidden_states")
    if not hidden_states:
        raise RuntimeError("SSL hidden states were not captured")
    pooled = hidden_states[-1].detach().float().mean(dim=1).squeeze(0)
    return logits, elapsed, int(pooled.numel())


def score_from_logits(logits: np.ndarray) -> float:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1, 2)
    return float(arr[0, 0] - arr[0, 1])


def run_case(model, name: str, audio: np.ndarray) -> str:
    logits, elapsed, emb_dim = forward_one(model, audio)
    raw_score = score_from_logits(logits)
    return (
        f"{name}: raw_score_logit_spoof_minus_bonafide={raw_score:.6f} "
        f"logits={np.asarray(logits).reshape(-1, 2).round(6).tolist()} "
        f"ssl_pooled_embedding_dim={emb_dim} wall_time_sec={elapsed:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="DF-Arena-1B snapshot directory")
    parser.add_argument("--audio", type=Path, action="append", default=None, help="optional real audio file(s) to score")
    parser.add_argument("--seed", type=int, default=606060)
    args = parser.parse_args()

    ckpt = args.ckpt.resolve()
    print(f"ckpt_path={ckpt}")
    model = load_model(ckpt)
    print(f"param_count={sum(p.numel() for p in model.parameters())}")
    print("score_field=logits[0, spoof_id=0] - logits[0, bonafide_id=1]; positive => spoof")
    print("ssl_backbone=facebook/wav2vec2-xls-r-1b hidden_size=1280 layers=48")

    rng = np.random.default_rng(args.seed)
    random_audio = (0.01 * rng.standard_normal(4 * 16000)).astype(np.float32)
    print(run_case(model, "random_noise_4s", random_audio))

    for real in args.audio or ():
        audio = load_audio(Path(real))
        print(f"real_audio_path={real}")
        print(run_case(model, "real_audio", audio))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
