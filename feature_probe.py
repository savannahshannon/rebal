#!/usr/bin/env python3
"""
Is REBAL's REPRESENTATION better, independent of its classifier?

This is the question the earlier post-hoc test could not answer. That test
applied logit adjustment on top of REBAL's own head, which reweighting and
SMOTE had already prior-corrected -- so it was double-correcting, which is
why the selected tau collapsed to zero on two seeds. It told us REBAL's
decision rule is redundant with LA. It did not tell us whether REBAL's
features are any good.

Features are the one thing post-hoc logit adjustment cannot fix. If REBAL's
backbone is genuinely better, the method has a contribution and the fix is
to pair it with a clean decision rule. If it is not, that is conclusive.

Three decision rules are applied to each frozen backbone:

  NCM      nearest class mean on L2-normalised features. No learned
           classifier at all, so no classifier bias -- the purest available
           measure of representation quality.
  probe    a linear head retrained on the natural (imbalanced) distribution
           with NO class weighting, so the head carries no rebalancing.
  probe+LA the same head with post-hoc logit adjustment, tau selected on the
           validation split. This is the clean version of the earlier test.

The comparison that matters is REBAL's backbone against the baseline's under
the SAME rule. Runs in minutes; no retraining of any backbone.

Usage
-----
    python feature_probe.py
    python feature_probe.py --seeds 0 1 2
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

import framework as fw
from framework import CFG

TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
METRICS = ["acc", "bacc", "head", "mid", "tail", "head_tail_gap",
           "worst_f1", "worst_acc"]
ARMS = {"baseline": "baseline.weights.h5", "rebal": "final.weights.h5"}


def replay_data(seed):
    CFG.SEED = seed
    fw.set_seeds(seed)
    x_imb, y_imb, x_te, y_te = fw.load_imbalanced_cifar100()
    x_clean, y_clean = (fw.ecod_clean(x_imb, y_imb) if CFG.USE_ECOD
                        else (x_imb, y_imb))
    x_te_01 = x_te.astype("float32") / 255.0
    x_clean_01 = (x_clean + 1.0) / 2.0
    tr_idx, val_idx = fw.stratified_val_indices(y_clean)
    return (x_clean_01[tr_idx], y_clean[tr_idx],
            x_clean_01[val_idx], y_clean[val_idx], x_te_01, y_te)


def features(weights_path, xs):
    model = fw.build_classifier(feature_dim=CFG.FEATURE_DIM)
    model.load_weights(weights_path)
    out = []
    for x in xs:
        chunks = []
        for i in range(0, len(x), CFG.BATCH_SIZE):
            _, f = model(x[i:i + CFG.BATCH_SIZE], training=False)
            chunks.append(f.numpy())
        out.append(np.concatenate(chunks, axis=0))
    del model
    tf.keras.backend.clear_session()
    return out


def ncm_logits(f_train, y_train, f_eval):
    """Nearest class mean on unit-normalised features. No learned classifier."""
    def norm(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    ft, fe = norm(f_train), norm(f_eval)
    cents = np.zeros((CFG.NUM_CLASSES, ft.shape[1]), dtype="float32")
    for k in range(CFG.NUM_CLASSES):
        m = y_train == k
        if m.any():
            c = ft[m].mean(axis=0)
            cents[k] = c / max(np.linalg.norm(c), 1e-8)
    return fe @ cents.T          # cosine similarity acts as the logit


def train_probe(f_train, y_train, epochs=30):
    """Plain linear head, natural distribution, no class weighting."""
    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(f_train.shape[1],)),
        tf.keras.layers.Dense(CFG.NUM_CLASSES),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    head.fit(f_train, y_train, batch_size=256, epochs=epochs, verbose=0)
    return head


def log_prior(y):
    n = np.maximum(np.bincount(y, minlength=CFG.NUM_CLASSES), 1).astype("float64")
    return np.log(n / n.sum()).astype("float32")


def met(logits, y, lp=None, tau=0.0):
    z = logits if tau == 0.0 else logits - tau * lp[None, :]
    m = fw._compute_metrics(y, np.argmax(z, axis=1))
    return {k: float(m[k]) for k in METRICS}


def bacc(logits, y, lp, tau):
    z = logits - tau * lp[None, :]
    p = np.argmax(z, axis=1)
    return float(np.mean([float((p[y == k] == k).mean()) for k in np.unique(y)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/corrected/rho_100/main")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="feature_probe.json")
    args = ap.parse_args()

    os.environ.setdefault("REBAL_ARCH", "resnet32")
    if CFG.ARCHITECTURE != "resnet32":
        CFG.ARCHITECTURE = "resnet32"
        if hasattr(fw, "_apply_arch_preset"):
            fw._apply_arch_preset()
    CFG.IMBALANCE_PROTOCOL = "cui2019"
    CFG.IMBALANCE_RATIO = 100.0
    if CFG.VAL_FRACTION <= 0:
        CFG.VAL_FRACTION = 0.1

    res = {}
    for seed in args.seeds:
        ck = os.path.join(args.root, f"seed_{seed}", "checkpoints")
        if not all(os.path.exists(os.path.join(ck, f)) for f in ARMS.values()):
            print(f"seed {seed}: checkpoints missing, skipping")
            continue
        print(f"\n=== seed {seed} ===")
        xtr, ytr, xva, yva, xte, yte = replay_data(seed)
        lp = log_prior(ytr)
        res[str(seed)] = {}

        for arm, fn in ARMS.items():
            f_tr, f_va, f_te = features(os.path.join(ck, fn), [xtr, xva, xte])

            ncm_te = ncm_logits(f_tr, ytr, f_te)
            head = train_probe(f_tr, ytr)
            pr_va = head.predict(f_va, verbose=0)
            pr_te = head.predict(f_te, verbose=0)
            tau = max(TAUS, key=lambda t: bacc(pr_va, yva, lp, t))

            res[str(seed)][arm] = {
                "ncm": met(ncm_te, yte),
                "probe": met(pr_te, yte),
                "probe_la": met(pr_te, yte, lp, tau),
                "tau": tau,
            }
            r = res[str(seed)][arm]
            print(f"  {arm:<9} NCM acc {r['ncm']['acc']:.4f} tail {r['ncm']['tail']:.4f} | "
                  f"probe acc {r['probe']['acc']:.4f} tail {r['probe']['tail']:.4f} | "
                  f"+LA(t={tau}) acc {r['probe_la']['acc']:.4f} tail {r['probe_la']['tail']:.4f}")
            tf.keras.backend.clear_session()

    if not res:
        raise SystemExit("no seeds evaluated")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

    seeds = sorted(res)
    def g(arm, rule, m):
        return np.array([res[s][arm][rule][m] for s in seeds])

    print("\n" + "=" * 78)
    print(" SAME decision rule on each backbone: are REBAL's FEATURES better?")
    print("=" * 78)
    keys = ["acc", "head", "mid", "tail", "head_tail_gap", "worst_f1"]
    for rule, label in [("ncm", "nearest class mean"),
                        ("probe", "linear probe (unweighted)"),
                        ("probe_la", "linear probe + logit adj")]:
        print(f"\n  {label}")
        print("    " + f"{'backbone':<12}" + "".join(f"{k:>11}" for k in keys))
        for arm in ARMS:
            print("    " + f"{arm:<12}" + "".join(f"{g(arm, rule, k).mean():11.4f}"
                                                  for k in keys))
        print("    " + f"{'rebal - base':<12}", end="")
        for k in keys:
            d = g("rebal", rule, k) - g("baseline", rule, k)
            s = ""
            try:
                from scipy import stats
                if d.std(ddof=1) > 0:
                    p = stats.ttest_rel(g("rebal", rule, k), g("baseline", rule, k)).pvalue
                    s = "*" if p < 0.05 else " "
            except ImportError:
                pass
            print(f"{d.mean():+10.4f}{s}", end="")
        print()

    print("\n  Read it this way: if REBAL's row beats the baseline's under NCM -- a")
    print("  rule with no learned classifier at all -- then the synthesis really did")
    print("  improve the representation, and the method's problem is only its")
    print("  decision rule, which is fixable. If NCM is a wash, the features are not")
    print("  better and no classifier will rescue them.")


if __name__ == "__main__":
    main()
