#!/usr/bin/env python3
"""The XLS-R-1B acoustic encoder of SpoofLLM, built from its architecture alone.

The released checkpoint carries every encoder weight, so nothing is downloaded and
no pretrained encoder is needed: this module builds the bare architecture and
`score_wavs.py` writes the checkpoint's own weights into it.

The config below is `facebook/wav2vec2-xls-r-1b` with dropout, layerdrop and
SpecAugment disabled, which is the configuration the model was trained and
evaluated under. Those fields change behaviour, never parameter shapes.
"""
from __future__ import annotations

import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model


def xlsr_1b_config() -> Wav2Vec2Config:
    """Architecture of facebook/wav2vec2-xls-r-1b: 48 layers, width 1280."""
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
        apply_spec_augment=False,
    )


def build_xlsr_encoder() -> Wav2Vec2Model:
    """Bare XLS-R-1B encoder. Weights are meaningless until the checkpoint is loaded."""
    model = Wav2Vec2Model(xlsr_1b_config())
    model.eval()
    return model


if __name__ == "__main__":
    enc = build_xlsr_encoder()
    n = sum(p.numel() for p in enc.parameters())
    print(f"tensors={len(enc.state_dict())} parameters={n/1e6:.1f}M")
    with torch.no_grad():
        out = enc(torch.zeros(1, 16000), output_hidden_states=True, return_dict=True)
    print(f"hidden states={len(out.hidden_states)} shape={tuple(out.hidden_states[-1].shape)}")
