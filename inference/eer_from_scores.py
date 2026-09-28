#!/usr/bin/env python3
"""EER / Cllr of score_wavs.py output against a label file (CPU, no torch needed).

    python eer_from_scores.py --scores scores.jsonl --labels labels.txt [--out metrics.json]

labels.txt: one line per file, `<path-or-basename-or-stem> <spoof|bonafide>` (whitespace or tab
separated; `fake/1` = spoof, `real/genuine/0` = bona fide).  Rows are matched by full path, then
basename, then stem.  Metrics follow the paper: EER at the crossing of the bona-fide-as-spoof and
spoof-missed rates; raw Cllr of `spoof_score` (positive = spoof); affine-min Cllr from a balanced
logistic re-map fitted on these scores (an oracle bound, not a deployable calibration).
"""
import argparse, json, math
from pathlib import Path
import numpy as np

SPOOF = {"spoof", "fake", "1"}; BONA = {"bonafide", "bona_fide", "bona-fide", "genuine", "real", "0"}


def eer_pct(scores, y):
    order = np.argsort(-scores); ys = y[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    far = fp / max(1, n_neg); frr = 1.0 - tp / max(1, n_pos)
    i = int(np.argmin(np.abs(far - frr)))
    return 100.0 * 0.5 * (far[i] + frr[i]), float(scores[order][i])


def cllr(llr, y):
    sp, bo = llr[y == 1], llr[y == 0]
    return 0.5 * (np.mean(np.logaddexp(0.0, -sp)) + np.mean(np.logaddexp(0.0, bo))) / math.log(2.0)


def affine_min_cllr(s, y):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    lr = LogisticRegression(C=1e6, class_weight="balanced", solver="lbfgs", max_iter=2000).fit(s.reshape(-1, 1), y)
    return cllr(lr.decision_function(s.reshape(-1, 1)), y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    labels = {}
    for line in Path(a.labels).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2: continue
        key, lab = parts[0], parts[-1].lower()
        if lab in SPOOF: labels[key] = 1
        elif lab in BONA: labels[key] = 0
        else: raise SystemExit(f"unknown label {lab!r} in line: {line}")
    rows = [json.loads(l) for l in Path(a.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    s, y, unmatched = [], [], 0
    for r in rows:
        p = r["path"]
        for k in (p, Path(p).name, Path(p).stem):
            if k in labels:
                s.append(float(r["spoof_score"])); y.append(labels[k]); break
        else:
            unmatched += 1
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.int64)
    if y.size == 0 or y.min() == y.max():
        raise SystemExit(f"need both classes: matched {y.size} rows (spoof {int(y.sum())}), unmatched {unmatched}")
    eer, thr = eer_pct(s, y)
    res = {"n_scored_rows": len(rows), "n_matched": int(y.size), "n_unmatched": unmatched, "n_spoof": int(y.sum()),
           "n_bonafide": int((1 - y).sum()), "eer_pct": eer, "eer_threshold": thr, "cllr_raw": cllr(s, y),
           "min_cllr_affine": affine_min_cllr(s, y), "acc_at_threshold_0_pct": float(100.0 * np.mean((s > 0) == (y == 1)))}
    print(json.dumps(res, indent=1))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
