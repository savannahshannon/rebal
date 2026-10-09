#!/usr/bin/env python3
"""
Does REBAL add anything on top of logit adjustment?

Post-hoc logit adjustment (Menon et al., ICLR 2021) subtracts tau*log(prior)
from the logits at inference. It needs no retraining, so this question can be
answered from the checkpoints the corrected sweep already wrote -- minutes,
not the 15 hours a retraining sweep would cost.

For every seed it evaluates four arms:

    baseline          the pipeline's own baseline checkpoint
    baseline + LA     the same weights, prior-corrected at inference
    rebal             the full REBAL checkpoint
    rebal + LA        the same weights, prior-corrected at inference

The comparison that matters is (rebal + LA) against (baseline + LA). If
REBAL's synthesis and regularization contribute anything a prior correction
does not already supply, that gap is where it shows. If the two are equal,
REBAL is dominated and the paper needs a different claim.

tau is SELECTED ON THE VALIDATION SPLIT, never on test. Choosing it by test
accuracy is the same error this whole re-run exists to undo. The default
tau = 1.0 is reported alongside for reference.

Usage
-----
    python posthoc_la.py
    python posthoc_la.py --seeds 0 1 2 --out la_ontop.json
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

import framework as fw
from framework import CFG

TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
METRICS = ["acc", "bacc", "head", "mid", "tail",
           "head_tail_gap", "worst_f1", "worst_acc"]


def rebuild_data(seed):
    """Replay the exact data pipeline the run used. Deterministic given seed."""
    CFG.SEED = seed
    fw.set_seeds(seed)
    x_imb, y_imb, x_te, y_te = fw.load_imbalanced_cifar100()
    if CFG.USE_ECOD:
        x_clean, y_clean = fw.ecod_clean(x_imb, y_imb)
    else:
        x_clean, y_clean = x_imb, y_imb
    x_te_01 = x_te.astype("float32") / 255.0
    x_clean_01 = (x_clean + 1.0) / 2.0

    if CFG.VAL_FRACTION > 0:
        tr_idx, val_idx = fw.stratified_val_indices(y_clean)
        x_val, y_val = x_clean_01[val_idx], y_clean[val_idx]
        y_train = y_clean[tr_idx]
    else:
        raise SystemExit("VAL_FRACTION is 0 -- set REBAL_VAL_FRACTION=0.1 so "
                         "tau is chosen on a validation split rather than on test.")
    return x_val, y_val, x_te_01, y_te, y_train


def log_prior(y_train):
    n = np.maximum(np.bincount(y_train, minlength=CFG.NUM_CLASSES), 1).astype("float64")
    return np.log(n / n.sum()).astype("float32")


def logits_for(weights_path, x):
    model = fw.build_classifier(feature_dim=CFG.FEATURE_DIM)
    model.load_weights(weights_path)
    chunks = []
    for i in range(0, len(x), CFG.BATCH_SIZE):
        cl, _ = model(x[i:i + CFG.BATCH_SIZE], training=False)
        chunks.append(cl.numpy())
    del model
    tf.keras.backend.clear_session()
    return np.concatenate(chunks, axis=0)


def balanced_acc(logits, y, lp, tau):
    pred = np.argmax(logits - tau * lp[None, :], axis=1)
    recalls = [float((pred[y == k] == k).mean())
               for k in np.unique(y)]
    return float(np.mean(recalls))


def metrics_at(logits, y, lp, tau):
    pred = np.argmax(logits - tau * lp[None, :], axis=1)
    m = fw._compute_metrics(y, pred)
    return {k: float(m[k]) for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/corrected/rho_100/main")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="posthoc_la.json")
    args = ap.parse_args()

    # The checkpoints were written under the corrected protocol; replaying the
    # data pipeline must use the same settings or the prior will not match.
    os.environ.setdefault("REBAL_ARCH", "resnet32")
    if CFG.ARCHITECTURE != "resnet32":
        CFG.ARCHITECTURE = "resnet32"
        if hasattr(fw, "_apply_arch_preset"):
            fw._apply_arch_preset()
    CFG.IMBALANCE_PROTOCOL = "cui2019"
    CFG.IMBALANCE_RATIO = 100.0
    if CFG.VAL_FRACTION <= 0:
        CFG.VAL_FRACTION = 0.1

    results = {}
    for seed in args.seeds:
        ck = os.path.join(args.root, f"seed_{seed}", "checkpoints")
        paths = {"baseline": os.path.join(ck, "baseline.weights.h5"),
                 "rebal": os.path.join(ck, "final.weights.h5")}
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            print(f"seed {seed}: missing {missing}, skipping")
            continue

        print(f"\n=== seed {seed} ===")
        x_val, y_val, x_te, y_te, y_train = rebuild_data(seed)
        lp = log_prior(y_train)
        print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_te)}")

        results[str(seed)] = {}
        for arm, wp in paths.items():
            lg_val = logits_for(wp, x_val)
            lg_te = logits_for(wp, x_te)

            # tau chosen on validation balanced accuracy -- never on test.
            scores = {t: balanced_acc(lg_val, y_val, lp, t) for t in TAUS}
            best_tau = max(scores, key=scores.get)

            entry = {
                "tau_selected": best_tau,
                "val_bacc_by_tau": {str(t): round(v, 4) for t, v in scores.items()},
                "plain": metrics_at(lg_te, y_te, lp, 0.0),
                "la_selected": metrics_at(lg_te, y_te, lp, best_tau),
                "la_tau1": metrics_at(lg_te, y_te, lp, 1.0),
            }
            results[str(seed)][arm] = entry
            p, s = entry["plain"], entry["la_selected"]
            print(f"  {arm:<9} tau*={best_tau:<4}  "
                  f"acc {p['acc']:.4f} -> {s['acc']:.4f}   "
                  f"tail {p['tail']:.4f} -> {s['tail']:.4f}   "
                  f"worst {p['worst_f1']:.4f} -> {s['worst_f1']:.4f}")

    if not results:
        sys.exit("no seeds evaluated")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    # ---- the comparison this script exists for ----
    seeds = sorted(results)
    def get(arm, key, m):
        return np.array([results[s][arm][key][m] for s in seeds
                         if arm in results[s]])

    print("\n" + "=" * 70)
    print(" does REBAL add anything on top of logit adjustment?")
    print("=" * 70)
    print(f"{'arm':<22}" + "".join(f"{m:>11}" for m in
                                   ["acc", "head", "mid", "tail", "gap", "worst"]))
    rows = [("baseline", "plain"), ("baseline + LA", "la_selected"),
            ("rebal", "plain"), ("rebal + LA", "la_selected")]
    keys = ["acc", "head", "mid", "tail", "head_tail_gap", "worst_f1"]
    for label, key in rows:
        arm = "baseline" if label.startswith("baseline") else "rebal"
        vals = [get(arm, key, m) for m in keys]
        print(f"{label:<22}" + "".join(f"{v.mean():11.4f}" for v in vals))

    print("\n  (rebal + LA) minus (baseline + LA):")
    for m in keys:
        a, b = get("rebal", "la_selected", m), get("baseline", "la_selected", m)
        d = a - b
        line = f"    {m:>14}  {d.mean():+.4f}"
        if len(d) > 1 and d.std(ddof=1) > 0:
            try:
                from scipy import stats
                line += f"   p={stats.ttest_rel(a, b).pvalue:.4f}"
            except ImportError:
                line += f"   dz={d.mean() / d.std(ddof=1):+.2f}"
        print(line)
    print("\n  If these deltas are near zero, REBAL contributes nothing that a "
          "prior\n  correction does not already supply, and the paper needs a "
          "different claim.")


if __name__ == "__main__":
    main()
