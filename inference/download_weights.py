#!/usr/bin/env python3
"""Fetch everything score_wavs.py needs: the SpoofLLM checkpoint, the DF-Arena-1B
snapshot that supplies the XLS-R-1B encoder architecture, and Qwen2.5-1.5B-Instruct.

    python download_weights.py --out models/

About 11 GB in total. Pass --only to fetch one component, e.g. --only ckpt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

SPOOFLLM_REPO = "JYP2024/SpoofLLM-detector"
SPOOFLLM_FILE = "merge_a0.5_b0.5_ep3.pt"
DF_ARENA_REPO = "Speech-Arena-2025/DF_Arena_1B_V_1"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--only", choices=["ckpt", "df-arena", "qwen"], action="append", default=None)
    args = ap.parse_args()
    want = set(args.only or ["ckpt", "df-arena", "qwen"])
    args.out.mkdir(parents=True, exist_ok=True)

    if "ckpt" in want:
        p = hf_hub_download(SPOOFLLM_REPO, SPOOFLLM_FILE, local_dir=args.out)
        print(f"[ckpt]      {p}")
    if "df-arena" in want:
        p = snapshot_download(DF_ARENA_REPO, local_dir=args.out / "df_arena_1b")
        print(f"[df-arena]  {p}")
    if "qwen" in want:
        p = snapshot_download(QWEN_REPO, local_dir=args.out / "Qwen2.5-1.5B-Instruct",
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "tokenizer*"])
        print(f"[qwen]      {p}")

    print("\nNext:")
    print(f"  python score_wavs.py --ckpt {args.out}/{SPOOFLLM_FILE} \\")
    print(f"    --df-arena-dir {args.out}/df_arena_1b --llm {args.out}/Qwen2.5-1.5B-Instruct \\")
    print("    --wavs smoke/list.txt --out smoke/scores.jsonl --bs 2")


if __name__ == "__main__":
    main()
