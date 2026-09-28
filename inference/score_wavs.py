#!/usr/bin/env python3
"""Score audio files with the SpoofLLM detection model (weight-merged frozen+fine-tuned encoder, 4.77% macro EER).

Output per file: fused log-odds on the teacher scale (positive = spoof), the three component
scores (S1 artifact, S2 naturalness, S3 residual, same scale), the verdict-head probability of spoof,
and the standardized raw head outputs.

    python score_wavs.py --ckpt models/merge_a0.5_b0.5_ep3.pt \
        --llm models/Qwen2.5-1.5B-Instruct \
        --wavs list.txt            # one path per line (or a directory; wav/flac/mp3/ogg via soundfile)
        --out scores.jsonl [--bs 8] [--device cuda]

Audio handling mirrors evaluation: mono, resampled to 16 kHz, then the model takes the leading
64,600 samples (4.04 s; shorter files are repeat-padded).

The checkpoint carries the complete XLS-R-1B encoder, so nothing but this file, the checkpoint and
the frozen Qwen2.5-1.5B-Instruct base is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
try:
    import soundfile as sf
except ImportError:          # fallback: torchaudio backend, then scipy (wav only)
    sf = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # spoofllm/ package + xlsr_encoder.py live next to this file
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from xlsr_encoder import build_xlsr_encoder  # noqa: E402
from spoofllm.student.model import build_qwen_mhfa_student, force_trainable_fp32, load_trainable_state_dict  # noqa: E402

FIELDS = ("lr_S1", "lr_S2", "lr_S3", "fused_log_lr")   # order of the 4-d LLR head


def build_kwargs_from_ckpt(ckpt: dict, attn: str) -> dict:
    cfg = ckpt.get("config", {}); m = cfg.get("model", {}); a = cfg.get("args", {}); fe = m.get("frontend", {})
    st = ckpt["trainable_state_dict"]
    def shape0(k): return int(st[k].shape[0]) if k in st else None
    num_methods = m.get("num_methods") or shape0("method_head.2.weight") or 0
    max_windows = m.get("max_windows") or shape0("region_head.2.weight") or 30
    kw = dict(
        lora=bool(m.get("lora", a.get("lora", True))), readout=str(m.get("readout", a.get("readout", "last"))),
        n_global_tokens=int(m.get("n_global_tokens", 8)), max_windows=int(max_windows),
        n_speaker_tokens=int(m.get("n_speaker_tokens", 8)), speaker_dim=int(m.get("speaker_dim", 256)),
        num_methods=int(num_methods), llm_fp32_forward=bool(m.get("llm_fp32_forward", False)),
        speech_bf16_forward=bool(m.get("speech_bf16_forward", True)), attn_implementation=attn,
        mhfa_nb_layer=int(fe.get("mhfa_nb_layer", 49)), mhfa_feat_dim=int(fe.get("mhfa_feat_dim", 1280)),
        mhfa_head_nb=int(fe.get("mhfa_head_nb", 64)), mhfa_compression_dim=int(fe.get("mhfa_compression_dim", 128)),
        mhfa_embed_dim=int(fe.get("mhfa_embed_dim", 256)), window_s=float(fe.get("window_s", 0.4)),
        hop_s=float(fe.get("hop_s", 0.2)), ssl_batch_size=int(fe.get("ssl_batch_size", 8)),
    )
    if "ssl_input_samples" in fe:
        kw["ssl_input_samples"] = int(fe["ssl_input_samples"])
    return kw


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    if sf is not None:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    else:
        try:
            import torchaudio
            t, sr = torchaudio.load(str(path)); wav = t.mean(dim=0).numpy().astype(np.float32)
        except Exception:
            from scipy.io import wavfile
            sr, data = wavfile.read(str(path)); data = np.asarray(data)
            if data.ndim > 1: data = data.mean(axis=1)
            wav = (data / 32768.0 if data.dtype == np.int16 else data).astype(np.float32)
    if sr != target_sr:
        wav = AF.resample(torch.from_numpy(wav), sr, target_sr).numpy()
    return wav.astype(np.float32)


def list_inputs(spec: str) -> list[Path]:
    p = Path(spec)
    if p.is_dir():
        exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
        return sorted(q for q in p.rglob("*") if q.suffix.lower() in exts)
    return [Path(l.strip()) for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="SpoofLLM checkpoint (.pt)")
    ap.add_argument("--llm", type=Path, required=True, help="Qwen2.5-1.5B-Instruct snapshot directory")
    ap.add_argument("--wavs", required=True, help="text file with one audio path per line, or a directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn-implementation", default="eager")
    args = ap.parse_args()
    device = torch.device(args.device)

    t0 = time.time()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    scale = ckpt["config"]["llr_scale"]["fields"]          # train-pool mean/std per field
    speech_ssl_model = build_xlsr_encoder()      # architecture only; the checkpoint supplies the weights
    model, tok, prefix, suffix, eos_id, pad_id, meta = build_qwen_mhfa_student(
        speech_ssl_model, args.llm, device=device, **build_kwargs_from_ckpt(ckpt, args.attn_implementation))
    force_trainable_fp32(model)
    report = load_trainable_state_dict(model, ckpt["trainable_state_dict"], strict=False)
    state_keys = set(ckpt["trainable_state_dict"])
    n_enc = sum(1 for k in state_keys if k.startswith("speech_ssl_model."))
    uncovered = [n for n, _ in model.named_parameters()
                 if n.startswith("speech_ssl_model.") and n not in state_keys]
    print(f"[load] tensors={len(state_keys)} (encoder {n_enc}) unexpected={len(report['unexpected'])} "
          f"encoder_uncovered={len(uncovered)} in {time.time()-t0:.0f}s", flush=True)
    if uncovered:
        print(f"[load] WARNING encoder parameters absent from the checkpoint: {uncovered[:5]}", flush=True)
    if report["unexpected"]:
        print("[load] WARNING unexpected keys:", report["unexpected"][:5], flush=True)
    model.eval()

    files = list_inputs(args.wavs)
    print(f"[data] {len(files)} files", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh, torch.no_grad():
        for i in range(0, len(files), args.bs):
            batch = files[i:i + args.bs]
            wavs = [load_audio(p) for p in batch]
            lengths = torch.tensor([w.size for w in wavs], dtype=torch.long)
            padded = torch.zeros((len(wavs), int(lengths.max())), dtype=torch.float32)
            for j, w in enumerate(wavs):
                padded[j, :w.size] = torch.from_numpy(w)
            lr, verdict = model(task="scores_audio", wav=padded.to(device), wav_len=lengths.to(device), prefix=prefix, suffix=suffix)
            lr = lr.detach().cpu().float().numpy(); pv = torch.sigmoid(verdict.detach().cpu().float()).numpy()
            for j, p in enumerate(batch):
                rec = {"path": str(p), "dur_s": round(float(wavs[j].size) / 16000.0, 3)}
                for k, name in enumerate(FIELDS):
                    rec[name] = float(lr[j, k] * scale[name]["std"] + scale[name]["mean"])
                rec["spoof_score"] = rec["fused_log_lr"]           # positive = spoof; teacher log-odds scale
                rec["p_spoof_verdict_head"] = float(pv[j])
                rec["z_standardized"] = [float(v) for v in lr[j]]
                fh.write(json.dumps(rec) + "\n")
            print(f"[score] {min(i+args.bs, len(files))}/{len(files)}", flush=True)
    print(f"[done] wrote {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
