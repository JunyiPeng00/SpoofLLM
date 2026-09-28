# SpoofLLM

**A prompt-conditioned large audio language model for speech spoof detection and temporal localization.**

One instruction-conditioned model produces two outputs: a calibrated utterance-level detection
score, and the spoofed time intervals of a partially manipulated recording as JSON. The task
instruction selects which.

Junyi Peng¹, Lichun Fan², Lin Zhang⁴, Oldřich Plchot¹, Themos Stafylakis³, Jian Luan², Jan Černocký¹

¹ Brno University of Technology, Czechia  ·  ² Xiaomi Inc.  ·  ³ Athens University of Economics and Business, Greece  ·  ⁴ Independent Researcher

---

## Abstract

Speech anti-spoofing requires utterance-level detection and temporal localization of manipulated
speech. For instruction-conditioned large audio language models (LALMs), we investigate whether
continuous countermeasure scores improve detection beyond binary labels. We propose SpoofLLM, which
learns a corpus-dependent primary teacher score and three auxiliary component scores alongside
binary classification. A dual-stream adapter provides global and frame tokens for detection and
autoregressive interval generation. Across fourteen DF-Arena protocols, score supervision reduces
macro equal error rate (EER) from 6.16 % to 4.89 % for the language decoder and from 6.30 % to
5.10 % for a Transformer mixer approximately matched in trainable backend parameters. Both backends
also obtain lower raw and oracle affine-calibrated log-likelihood-ratio costs. On the same
protocols, the released HoliAntiSpoof checkpoint yields 13.47 % macro EER under different training
conditions. On PartialSpoof, under matched total update budgets, the fine-tuned joint model achieves
6.37 % balanced segment error, below both single-task configurations (9.11 % with a frozen encoder
and 10.20 % with encoder fine-tuning). It also achieves 92.36 % segment F1 on the 160 ms grid while
supporting utterance-level detection.

---

## Model

![SpoofLLM architecture](assets/spoofllm_system.png)

**(a)** A task instruction, eight global tokens and `T` frame tokens enter the backend, which emits
either a detection score or a JSON list of spoofed intervals.
**(b)** The language backend is Qwen2.5-1.5B-Instruct adapted with LoRA (rank 16, `q/k/v/o`).
**(c)** A randomly initialized bidirectional Transformer mixer replaces Qwen and LoRA, matched to
within 1 % of its trainable backend parameters (4.40 M vs 4.36 M), and tests whether the same
supervision helps a non-language backend.
**(d)** The dual-stream adapter extends multi-head factorized attention pooling: one set of
attention weights and values feeds both a pooled global stream and a time-resolved frame stream,
over a learned mixture of all 49 XLS-R-1B hidden states.

Text embeddings are frozen. The encoder is frozen unless fine-tuning is stated.

**Supervision.** Alongside the binary label, the student regresses a primary detection target and
three auxiliary component scores: an artifact score (AASIST + RawNet2-DF), a naturalness score
(UTMOS with pitch, duration, voicing, jitter and shimmer statistics), and a residual SSL
countermeasure score. All are logistic-calibrated log-odds. At inference the student restores only
the primary prediction; no teacher and no post-hoc calibrator is involved.

---

## Results

### Detection: fourteen DF-Arena protocols

Macro columns are unweighted protocol averages. Seen/unseen refer to SpoofLLM's training coverage.
Oracle affine calibration is fitted and evaluated on each protocol's own evaluation trials, so it is
a diagnostic bound rather than a deployable calibration. External systems keep their original
training conditions. Bold marks the best value within each backend.

| System / supervision | 19LA | ASV5 | CFake | LSeVoc | DFADD | **Macro₅** | 21LA | 21DF | ITW | FoR | SONAR | A22T1 | A22T3 | A23R1 | A23R2 | **Macro₉** | **Macro₁₄ ↓** | Raw *C*<sub>llr</sub> ↓ | Oracle affine ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Language decoder (A)* |
| token softmax | 0.76 | 14.39 | 5.41 | 0.08 | **0.00** | 4.13 | 5.88 | 2.73 | 2.35 | 9.53 | 1.57 | 23.71 | 3.48 | 7.85 | 11.31 | 7.60 | 6.36 | 1.30 | 0.26 |
| binary cross-entropy | 1.01 | 15.80 | **4.98** | 0.08 | **0.00** | 4.37 | 7.31 | 3.10 | 1.32 | 7.95 | 1.54 | 23.66 | 3.05 | 6.66 | 9.83 | 7.16 | 6.16 | 1.36 | 0.22 |
| primary score only | 0.35 | 13.68 | 8.28 | 0.05 | **0.00** | 4.47 | 6.62 | 1.56 | 1.09 | **2.87** | 0.79 | 23.44 | 2.12 | 4.83 | 7.43 | 5.64 | 5.22 | 0.52 | 0.18 |
| primary + components | 0.26 | **12.82** | 7.18 | 0.11 | **0.00** | **4.07** | 5.08 | 0.96 | 0.93 | 3.80 | **0.57** | 21.94 | **2.08** | **4.65** | 8.03 | 5.34 | 4.89 | 0.46 | **0.17** |
| &nbsp;&nbsp;+ encoder fine-tuning **(released)** | **0.14** | 13.23 | 9.13 | **0.00** | **0.00** | 4.50 | **3.10** | **0.79** | **0.74** | 3.75 | 1.05 | **20.60** | 2.21 | 5.17 | **6.86** | **4.92** | **4.77** | **0.41** | 0.19 |
| *Mixer (B)* |
| binary cross-entropy | 1.05 | 16.05 | **4.96** | 0.09 | **0.00** | 4.43 | **4.43** | 3.34 | 1.54 | 9.58 | 1.63 | 24.85 | 3.64 | 6.27 | 10.76 | 7.34 | 6.30 | 0.98 | 0.23 |
| primary + components | **0.28** | **12.98** | 8.04 | **0.08** | **0.00** | **4.28** | 4.57 | **1.21** | **1.02** | **3.27** | **0.88** | **23.11** | **2.08** | **5.34** | **8.56** | **5.56** | **5.10** | **0.47** | **0.17** |
| *Teachers and external systems* |
| five-member fusion | 0.10 | 12.71 | 31.45 | 0.56 | 0.13 | 8.99 | 2.43 | 1.33 | 2.82 | 1.46 | 7.93 | 22.51 | 4.11 | 15.31 | 17.90 | 8.42 | 8.62 | 3.71 | 0.27 |
| DF-Arena-1B | 1.13 | 17.41 | 8.07 | 0.19 | 0.00 | 5.36 | 4.94 | 1.93 | 0.91 | 3.02 | 1.14 | 22.27 | 2.20 | 5.14 | 11.56 | 5.90 | 5.71 | 0.30 | 0.19 |
| HoliAntiSpoof (released ckpt, re-scored here) | 1.09 | 24.85 | 15.27 | 1.13 | 24.90 | 13.45 | 10.28 | 8.17 | 0.81 | 1.55 | 20.06 | 26.88 | 8.85 | 24.57 | 20.17 | 13.48 | 13.47 | 1.81 | 0.41 |

Detection EER in %. Protocols: ASVspoof 2019 LA, ASVspoof 5 Track 1, CodecFake, LibriSeVoc, DFADD,
ASVspoof 2021 LA, ASVspoof 2021 DF, In-the-Wild, FoR, SONAR, ADD 2022 Track 1 and 3, ADD 2023
Round 1 and 2.

Both students beat DF-Arena-1B, which supplies their own primary targets on seven training corpora.

<p align="center"><img src="assets/spoofllm_radar.png" width="560" alt="Per-protocol detection EER"></p>

Axes are normalized independently per protocol. The best of the three systems lies on the rim and
the boxed value gives that system's EER in %.

### Calibration

<p align="center"><img src="assets/spoofllm_calibration.png" width="620" alt="Raw and oracle affine-calibrated Cllr"></p>

Score supervision lowers the raw cost far more than the oracle affine cost, so most of what binary
training loses is scale and offset that a per-corpus affine map could have recovered. The raw-to-
affine gap falls from 1.14 to 0.29 for the decoder and from 0.75 to 0.30 for the mixer.

### Detection ablations

Data, seed and optimizer are held fixed. Evaluation always uses full-length audio; only the training
window varies.

| System | Training window | EER % ↓ | Raw *C*<sub>llr</sub> ↓ |
|---|---|---|---|
| *Language decoder (A)* |
| input up to 20 s | 20 s | 6.35 | 0.65 |
| **leading window** | **4.04 s** | **4.89** | **0.46** |
| &nbsp;&nbsp;final encoder layer only | 4.04 s | 6.11 | 0.60 |
| &nbsp;&nbsp;frame tokens only, no global tokens | 4.04 s | 5.38 | 0.53 |
| *Mixer (B)* |
| input up to 20 s | 20 s | 6.20 | 0.61 |
| random crop | 4.04 s random | 6.68 | 0.63 |
| **leading window** | **4.04 s** | **5.10** | **0.47** |

Cross-layer aggregation and global tokens both earn their place, and a leading training window beats
both a longer window and an equal-duration random crop.

### Localization: PartialSpoof, 160 ms grid

Encoders are frozen unless stated. Single-task and joint comparisons match total optimizer updates,
so joint training gives localization about half the updates of its single-task counterpart. Segment
error is threshold-swept segEER for frame-based systems and balanced error at one operating point
for generated spans.

| Configuration | Det. EER % single | Det. EER % joint | Seg. error % single | Seg. error % joint | segF1 % joint | JSON % joint |
|---|---|---|---|---|---|---|
| *Language decoder (A), score-based readout* |
| frame head, text instruction | 4.89 | 5.28 | 7.65 | 7.81 | 91.18 | – |
| *Language decoder (A), generated spans (single operating point)* |
| span generation, text instruction | 4.89 | 5.13 | 9.11 | 11.56 | 88.15 | 99.48 |
| &nbsp;&nbsp;+ encoder fine-tuning | 4.77 | **5.05** | 10.20 | **6.37** | **92.36** | 99.49 |
| span generation, code₁ selector | 4.89 | 5.21 | 9.11 | 11.92 | 87.74 | 99.49 |
| *Mixer (B), task embedding, score-based readout* |
| frame head | 6.81 | 6.65 | 7.60 | 9.46 | 89.74 | – |

Single-task reference with a fine-tuned encoder, same grid: CFPRF at 6.20 % segEER and 93.81 % segF1.

Joint training hurts localization with a frozen encoder and helps it with a fine-tuned one. About
99.5 % of generated outputs are valid JSON under a strict criterion: the text must parse as a list
of `{"s", "e"}` objects with finite two-decimal endpoints, positive duration, chronological and
non-overlapping, and must re-serialize byte-for-byte.

**Example output.** `CON_E_0000368`, PartialSpoof evaluation, 3.09 s.

```
reference   [{"s":0.32,"e":1.76},{"s":2.24,"e":2.88}]
prediction  [{"s":0.32,"e":1.76},{"s":2.24,"e":3.04}]
```

The first interval matches; the second ends one 0.16 s grid step late.

---

## Detection inference

The released checkpoint is the fine-tuned-encoder system: **4.77 % macro EER** over the fourteen
protocols above. See [`inference/`](inference/) for the full instructions.

```bash
cd inference
pip install torch torchaudio            # pick the build for your CUDA / ROCm
pip install -r requirements.txt
python download_weights.py --out models/          # checkpoint + the frozen Qwen base

python score_wavs.py \
  --ckpt models/merge_a0.5_b0.5_ep3.pt \
  --llm models/Qwen2.5-1.5B-Instruct \
  --wavs my_files.txt --out scores.jsonl --device cuda
```

The checkpoint carries the complete fine-tuned XLS-R-1B encoder, so no pretrained acoustic model is
downloaded and none of the teachers are needed at inference.

Each line of `scores.jsonl` carries `spoof_score`, a log-odds on the teacher scale where positive
means spoof, plus the three component scores and the verdict-head probability. `eer_from_scores.py`
turns a scored file plus a label list into EER and *C*<sub>llr</sub>.

A two-file smoke test with reference outputs ships in `inference/smoke/`. Run it first.

---

## Training data

The detection pool holds 4.57 M utterances from ten corpora: SpoofCeleb, CodecFake, MLAAD, DFADD,
EnvSDD, PartialSpoof, ASVspoof 5, CtrSVDD, LibriSeVoc and ASVspoof 2019 LA. PartialSpoof also
supplies the official 160 ms segment labels behind both localization outputs.

Teacher fits use ASVspoof 2019 LA development data: 24,844 utterances, 22,296 spoof and 2,548 bona
fide. Corpus identity selects the training-target source only. Inference uses one student, with no
teacher and no corpus identity.

---

## Citation

```bibtex
@article{peng2026spoofllm,
  title   = {{SpoofLLM}: A Prompt-Conditioned {LALM} for Spoof Detection and Localization},
  author  = {Peng, Junyi and Fan, Lichun and Zhang, Lin and Plchot, Old{\v{r}}ich and
             Stafylakis, Themos and Luan, Jian and {\v{C}}ernock{\'y}, Jan},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Code in this repository is released under Apache-2.0.

The checkpoint is released **for research use only**, because it was trained against teacher scores
that include DF-Arena-1B, whose terms are non-commercial.

| Component | Role | At inference | Terms |
|---|---|---|---|
| DF-Arena-1B | primary teacher score on seven of the ten training corpora, one of two members of the residual component, and the initialization of the acoustic encoder | not used | non-commercial |
| Qwen2.5-1.5B-Instruct | frozen language backend | required | Apache-2.0 |
| XLS-R-1B | acoustic encoder architecture | built locally, weights come from the checkpoint | Apache-2.0 |
| AASIST, RawNet2-DF, UTMOS | artifact and naturalness teacher components | not used | respective upstream licenses |

The released encoder is fine-tuned away from its DF-Arena-1B initialization, and no DF-Arena weight
or file is read at inference. The non-commercial condition follows from training, not from loading.
Clear the upstream terms yourself before any commercial use.
