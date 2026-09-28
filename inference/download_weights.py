#!/usr/bin/env python3
"""Fetch what score_wavs.py needs: the SpoofLLM checkpoint and Qwen2.5-1.5B-Instruct.

    python download_weights.py --out models/

About 7 GB. The acoustic encoder is not downloaded: the checkpoint carries every
XLS-R-1B weight and `xlsr_encoder.py` builds the architecture locally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

SPOOFLLM_REPO = "JYP2024/SpoofLLM-detector"
SPOOFLLM_FILE = "merge_a0.5_b0.5_ep3.pt"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--only", choices=["ckpt", "qwen"], action="append", default=None)
    args = ap.parse_args()
    want = set(args.only or ["ckpt", "qwen"])
    args.out.mkdir(parents=True, exist_ok=True)

    if "ckpt" in want:
        print(f"[ckpt] {hf_hub_download(SPOOFLLM_REPO, SPOOFLLM_FILE, local_dir=args.out)}")
    if "qwen" in want:
        p = snapshot_download(QWEN_REPO, local_dir=args.out / "Qwen2.5-1.5B-Instruct",
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
        print(f"[qwen] {p}")

    print("\nNext:")
    print(f"  python score_wavs.py --ckpt {args.out}/{SPOOFLLM_FILE} \\")
    print(f"    --llm {args.out}/Qwen2.5-1.5B-Instruct \\")
    print("    --wavs smoke/list.txt --out smoke/scores.jsonl --bs 2")


if __name__ == "__main__":
    main()
