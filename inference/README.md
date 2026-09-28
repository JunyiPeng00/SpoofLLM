# SpoofLLM detection inference

Scores audio files with the released SpoofLLM detector: the fine-tuned-encoder system that reaches
**4.77 % macro EER** over the fourteen DF-Arena protocols in the paper. This directory covers
detection only. Localization is not part of this release.

## Contents

| File | Purpose |
|---|---|
| `score_wavs.py` | the scorer; reads a file list or a directory, writes one JSON line per file |
| `eer_from_scores.py` | EER and *C*<sub>llr</sub> of a scored file against a label list |
| `download_weights.py` | pulls the checkpoint and Qwen2.5-1.5B-Instruct |
| `xlsr_encoder.py` | builds the bare XLS-R-1B encoder architecture |
| `spoofllm/student/` | the model: dual-stream MHFA adapter, Qwen + LoRA, the score heads |
| `smoke/` | two PartialSpoof evaluation files and their reference scores |

## Install

```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verified on Python 3.12.9 with torch 2.6.0 (ROCm 6.2.4), transformers 4.51.1 and peft 0.15.1.
The reported results were produced on AMD MI250X. CUDA builds use the same code path.

## Get the weights

```bash
python download_weights.py --out models/
```

Two components, about 7 GB:

- `merge_a0.5_b0.5_ep3.pt` (3.97 GB) — the SpoofLLM checkpoint. Holds the dual-stream adapter, the
  Qwen LoRA parameters, the score heads, the complete fine-tuned XLS-R-1B encoder and the
  training-pool statistics used to restore the score scale.
- `Qwen/Qwen2.5-1.5B-Instruct` — the frozen language backend and its tokenizer.

No pretrained acoustic encoder is downloaded. The checkpoint's 806 encoder tensors are exactly the
state dict of an XLS-R-1B `Wav2Vec2Model`, so `xlsr_encoder.py` builds the architecture and the
checkpoint fills every weight.

## Run the smoke test first

```bash
python score_wavs.py \
  --ckpt models/merge_a0.5_b0.5_ep3.pt \
  --llm models/Qwen2.5-1.5B-Instruct \
  --wavs smoke/list.txt --out smoke/scores.jsonl --bs 2 --device cpu
```

The load report must read `tensors=1082 (encoder 806) unexpected=0 encoder_uncovered=0`. Reference
output, CPU, `smoke/expected_scores_cpu.jsonl`:

| file | label | `spoof_score` | `lr_S1` | `lr_S2` | `lr_S3` | `p_spoof_verdict_head` |
|---|---|---|---|---|---|---|
| CON_E_0000368 | spoof | +16.71 | 0.82 | 11.55 | 3.58 | 1.000 |
| LA_E_1617471 | bona fide | −1.82 | −14.01 | −4.67 | 1.62 | 0.0002 |

A GPU in bf16 moves the second decimal. The signs and the ordering must not move.

## Score your own audio

```bash
python score_wavs.py \
  --ckpt models/merge_a0.5_b0.5_ep3.pt \
  --llm models/Qwen2.5-1.5B-Instruct \
  --wavs my_files.txt --out scores.jsonl --bs 8 --device cuda
```

`--wavs` takes a text file with one path per line, or a directory that is searched recursively for
`wav`, `flac`, `mp3`, `ogg`, `m4a` and `opus`. Any sample rate works: files are mixed to mono and
resampled to 16 kHz. Roughly 12 GB of accelerator memory at batch size 8. CPU works too: the smoke test above loads
the model in about 16 s and scores its two files in about 26 s on 16 threads.

## Reading the output

One JSON object per line:

| Field | Meaning |
|---|---|
| `spoof_score` | the detection score, a log-odds on the teacher scale. **Positive means spoof.** This is the score behind every EER and *C*<sub>llr</sub> in the paper. |
| `lr_S1`, `lr_S2`, `lr_S3` | the three component scores on the same scale: artifact, naturalness, residual SSL |
| `p_spoof_verdict_head` | sigmoid of the auxiliary binary head, trained with cross-entropy and less well calibrated than the score |
| `z_standardized` | the raw 4-dimensional head output before de-standardization |

The score carries no post-hoc calibration: it is the head output restored with the training-pool
mean and standard deviation, exactly as evaluated in the paper. Thresholding at 0 decides at equal
priors, but the teacher scale keeps the calibration-set prior, so a threshold picked on a small
labelled development set of your own domain will do better. The paper's raw and oracle affine
*C*<sub>llr</sub> columns quantify how much a per-corpus affine map would recover.

## Audio length

The model scores the leading 4.04 s (64,600 samples) of each file and repeat-pads anything shorter.
That is the protocol behind the reported numbers. For longer recordings, cut them into 4 s chunks
yourself and aggregate the scores, by mean or by max depending on whether you expect the whole
recording or only part of it to be manipulated.

## Measuring EER

```bash
python eer_from_scores.py --scores scores.jsonl --labels labels.txt --out metrics.json
```

`labels.txt` holds one line per file, `<path-or-basename-or-stem> <spoof|bonafide>`. Rows are
matched by full path, then basename, then stem. The reported `min_cllr_affine` is fitted on the
scores being evaluated, so it is an oracle bound and not a calibration you can deploy.
