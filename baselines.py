#!/usr/bin/env python3
"""
Long-tailed baselines for matched comparison against REBAL.

Adds the three methods every reviewer asked for and that framework.py does
not implement:

    cb        Class-Balanced softmax CE          (Cui et al., CVPR 2019)
    ldam      LDAM margin loss                   (Cao et al., NeurIPS 2019)
    ldam_drw  LDAM + deferred reweighting        (Cao et al., NeurIPS 2019)
    la        Logit adjustment, loss variant     (Menon et al., ICLR 2021)
    la_posthoc  Logit adjustment, post-hoc       (Menon et al., ICLR 2021)
    ce        Plain cross-entropy (sanity check; equals framework.py baseline)

Design notes
------------
This module IMPORTS framework.py and does not modify it. It reuses the same
data loading, imbalance protocol, architecture, optimizer, schedule,
augmentation and metrics, so the comparison is matched by construction
rather than by assertion. The only thing that differs between arms is the
classification loss.

`la_posthoc` is free: it is `ce` at training time with the prior subtracted
from the logits at inference, so one training run yields both `ce` and
`la_posthoc` rows.

Usage
-----
    python baselines.py --method ldam_drw --rho 100 --seed 0 \
        --arch resnet32 --protocol cui2019 --out-dir experiments/baselines/rho_100/ldam_drw/seed_0

See run_baselines.sh for the full sweep.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

import framework as fw
from framework import CFG


# ======================================================================
# Cosine-normalised classifier head (required by LDAM)
# ======================================================================
class NormalizedDense(layers.Layer):
    """Cosine classifier: L2-normalise both features and weight columns.

    LDAM's margins are defined on cosine similarities, so the standard
    unnormalised Dense head from framework.py would make the margin scale
    meaningless. Cao et al. use exactly this head.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        x_n = tf.math.l2_normalize(x, axis=1)
        w_n = tf.math.l2_normalize(self.kernel, axis=0)
        return tf.matmul(x_n, w_n)          # cosine similarities in [-1, 1]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def build_model(method, feature_dim=CFG.FEATURE_DIM):
    """Same trunk as framework.build_classifier; cosine head only for LDAM.

    The unused original `logits` Dense is not part of the returned graph, so
    its weights and its L2 penalty are excluded from the model.
    """
    base = fw.build_classifier(feature_dim=feature_dim)
    if not method.startswith("ldam"):
        return base
    inp = base.input
    feat = base.outputs[1]                   # post-dropout feature tensor
    cos_logits = NormalizedDense(CFG.NUM_CLASSES, name="cosine_logits")(feat)
    return tf.keras.Model(inp, [cos_logits, feat], name="classifier_cosine")


# ======================================================================
# Losses
# ======================================================================
LDAM_MAX_MARGIN = 0.5     # Cao et al. default
LDAM_SCALE = 30.0         # Cao et al. default
LA_TAU = 1.0              # Menon et al. default
DRW_START_FRAC = 0.8      # DRW switches on at 80% of the schedule


def class_counts(y):
    return np.maximum(
        np.bincount(y, minlength=CFG.NUM_CLASSES).astype("float64"), 1.0
    )


def ldam_margins(y):
    """Delta_j = C / n_j^(1/4), scaled so the largest margin equals MAX_MARGIN."""
    n = class_counts(y)
    m = 1.0 / np.power(n, 0.25)
    m = m * (LDAM_MAX_MARGIN / m.max())
    return m.astype("float32")


def log_prior(y):
    n = class_counts(y)
    p = n / n.sum()
    return np.log(p).astype("float32")


# ======================================================================
# Training
# ======================================================================
def train_baseline(model, method, x_train, y_train, x_val, y_val,
                   cls_weights, margins, logprior, verbose=True):
    """Mirror of framework.train_classifier with a pluggable loss.

    Optimizer, schedule, gradient clipping, early stopping and the
    validation protocol are identical to framework.train_classifier so that
    any difference in the results is attributable to the loss alone.
    """
    steps_per_epoch = max(1, len(x_train) // CFG.BATCH_SIZE)
    if CFG.USE_LR_SCHEDULE:
        decay_steps = steps_per_epoch * max(1, CFG.CLS_EPOCHS - CFG.WARMUP_EPOCHS)
        if CFG.WARMUP_EPOCHS > 0:
            lr = fw.WarmupCosineDecay(
                CFG.CLS_LR, steps_per_epoch * CFG.WARMUP_EPOCHS, decay_steps)
        else:
            lr = tf.keras.optimizers.schedules.CosineDecay(
                CFG.CLS_LR, decay_steps=decay_steps)
    else:
        lr = CFG.CLS_LR
    opt = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True,
                                  clipnorm=CFG.GRAD_CLIP_NORM)

    ce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    cw = tf.constant(cls_weights, dtype=tf.float32)
    mg = tf.constant(margins, dtype=tf.float32)
    lp = tf.constant(logprior, dtype=tf.float32)

    cur_epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
    drw_epoch = int(DRW_START_FRAC * CFG.CLS_EPOCHS)

    ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
          .shuffle(min(20000, len(x_train)), seed=CFG.SEED)
          .batch(CFG.BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

    uses_drw = (method == "ldam_drw")
    uses_weights = method in ("cb",)
    is_ldam = method.startswith("ldam")
    is_la_loss = (method == "la")

    @tf.function
    def train_step(xb, yb):
        # CIFAR-10 labels arrive as uint8; tf.gather and tf.one_hot
        # need int32/int64. Cast once here rather than in every path.
        yb = tf.cast(yb, tf.int32)
        with tf.GradientTape() as tape:
            logits, _ = model(xb, training=True)

            if is_ldam:
                # Subtract the class margin from the TRUE-class logit only,
                # then scale. cos in [-1,1] -> s*(cos - m_y * onehot).
                onehot = tf.one_hot(yb, CFG.NUM_CLASSES)
                margin_y = onehot * tf.reshape(mg, [1, -1])
                logits = LDAM_SCALE * (logits - margin_y)
            elif is_la_loss:
                # Menon et al.: add tau * log(prior) to every logit.
                logits = logits + LA_TAU * tf.reshape(lp, [1, -1])

            per_sample = ce(yb, logits)

            if uses_weights:
                sample_w = tf.gather(cw, yb)
                loss = tf.reduce_mean(per_sample * sample_w)
            elif uses_drw:
                on = tf.cast(cur_epoch >= drw_epoch, tf.float32)
                sample_w = on * tf.gather(cw, yb) + (1.0 - on)
                loss = tf.reduce_mean(per_sample * sample_w)
            else:
                loss = tf.reduce_mean(per_sample)

            if model.losses:
                loss = loss + tf.add_n([tf.cast(l, loss.dtype)
                                        for l in model.losses])

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, model.trainable_variables)]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_val, best_w, patience = -1.0, None, 0
    for ep in range(CFG.CLS_EPOCHS):
        cur_epoch.assign(ep)
        t0 = time.time()
        tot, n = 0.0, 0
        for xb, yb in ds:
            tot += float(train_step(xb, yb))
            n += 1

        chunks = []
        for i in range(0, len(x_val), CFG.BATCH_SIZE):
            cl, _ = model(x_val[i:i + CFG.BATCH_SIZE], training=False)
            chunks.append(tf.argmax(cl, axis=-1, output_type=tf.int32))
        val_preds = tf.concat(chunks, axis=0)
        val_acc = float(tf.reduce_mean(
            tf.cast(tf.equal(val_preds, y_val), tf.float32)))

        if verbose:
            print(f"  ep {ep + 1:3d}/{CFG.CLS_EPOCHS}  loss={tot / n:.3f}  "
                  f"val_acc={val_acc:.4f}  ({time.time() - t0:.1f}s)")

        if CFG.USE_EARLY_STOPPING:
            if val_acc > best_val:
                best_val, best_w, patience = val_acc, model.get_weights(), 0
            else:
                patience += 1
                if patience >= CFG.EARLY_STOP_PATIENCE:
                    if verbose:
                        print(f"  early stop (best val_acc={best_val:.4f})")
                    break

    if CFG.USE_EARLY_STOPPING and best_w is not None:
        model.set_weights(best_w)
    return best_val


# ======================================================================
# Evaluation
# ======================================================================
def predict_logits(model, x):
    chunks = []
    for i in range(0, len(x), CFG.BATCH_SIZE):
        cl, _ = model(x[i:i + CFG.BATCH_SIZE], training=False)
        chunks.append(cl)
    return tf.concat(chunks, axis=0)


def evaluate_method(model, x_te, y_te, method, logprior):
    logits = predict_logits(model, x_te)
    if method == "la_posthoc":
        logits = logits - LA_TAU * tf.reshape(
            tf.constant(logprior, dtype=tf.float32), [1, -1])
    y_hat = tf.argmax(logits, axis=-1).numpy()
    return fw._compute_metrics(y_te, y_hat)


def peak_memory_mb():
    """Peak memory, GPU if available, else host RSS."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info = tf.config.experimental.get_memory_info("GPU:0")
            return float(info["peak"]) / (1024 ** 2)
    except Exception:
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        div = 1024.0 if os.uname().sysname == "Linux" else 1024.0 ** 2
        return float(peak) / div
    except Exception:
        return float("nan")


# ======================================================================
# Main
# ======================================================================
def run(args):
    CFG.SEED = args.seed
    # DATASET and NUM_CLASSES must both be set BEFORE _apply_arch_preset.
    # _apply_arch_preset does not touch NUM_CLASSES, and the
    # DATASET-to-NUM_CLASSES mapping in framework.py only fires on an
    # env var, not on this CLI path.
    if args.dataset:
        CFG.DATASET = args.dataset
        CFG.NUM_CLASSES = 10 if CFG.DATASET == "cifar10" else 100
    if args.arch:
        CFG.ARCHITECTURE = args.arch
        # Without this a baseline runs 120 epochs / patience 12 / no
        # warmup while REBAL runs 200 / 10, and the comparison the
        # table exists to make is not a comparison at all.
        if hasattr(fw, "_apply_arch_preset"):
            fw._apply_arch_preset()
        else:
            raise SystemExit(
                "framework.py has no _apply_arch_preset; apply "
                "patch_arch_preset.py before running baselines.")
    if args.protocol:
        CFG.IMBALANCE_PROTOCOL = args.protocol
    if args.rho is not None:
        CFG.IMBALANCE_RATIO = args.rho
    if args.epochs is not None:
        CFG.CLS_EPOCHS = args.epochs
    # REBAL runs the full cosine schedule and keeps the best-validation
    # weights. Baselines must do the same or the arms differ by more
    # than their loss.
    CFG.USE_EARLY_STOPPING = bool(args.early_stopping)

    # Baselines are single-arm: no ECOD, no synthesis, no equalization.
    CFG.USE_CGAN_AUG = False
    CFG.USE_FEATURE_SMOTE = False
    CFG.USE_EQUALIZATION = False
    CFG.USE_REWEIGHTING = False

    fw.set_seeds(args.seed)

    out_dir = args.out_dir or os.path.join(
        "experiments", "baselines",
        f"rho_{int(args.rho or CFG.IMBALANCE_RATIO)}",
        args.method, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"BASELINE: {args.method}   seed={args.seed}   "
          f"rho={CFG.IMBALANCE_RATIO}   arch={CFG.ARCHITECTURE}")
    print(f"val-mode: {args.val_mode}")
    print(f"{'=' * 64}")

    t_start = time.time()

    x_imb, y_imb, x_te, y_te = fw.load_imbalanced_cifar100()
    # ECOD is a REBAL module, not part of any baseline.
    x_train_01 = (x_imb + 1.0) / 2.0
    x_te_01 = x_te.astype("float32") / 255.0
    del x_imb, x_te
    gc.collect()

    if args.val_mode == "holdout":
        # Proper protocol: stratified validation split carved out of TRAIN.
        rng = np.random.RandomState(args.seed)
        val_idx, tr_idx = [], []
        for k in range(CFG.NUM_CLASSES):
            idx = np.where(y_imb == k)[0]
            rng.shuffle(idx)
            n_val = max(1, int(round(args.val_frac * len(idx)))) if len(idx) > 2 else 0
            val_idx.extend(idx[:n_val])
            tr_idx.extend(idx[n_val:])
        val_idx = np.array(sorted(val_idx))
        tr_idx = np.array(sorted(tr_idx))
        x_val, y_val = x_train_01[val_idx], y_imb[val_idx]
        x_fit, y_fit = x_train_01[tr_idx], y_imb[tr_idx]
        print(f"  holdout split: train={len(y_fit)}  val={len(y_val)}")
    else:
        # 'match' reproduces framework.py's current behaviour, in which the
        # TEST set is passed as the validation set and early stopping selects
        # weights on it. Use this ONLY to produce numbers comparable with the
        # existing sprint2/sprint4 results. See the note printed below.
        x_fit, y_fit = x_train_01, y_imb
        x_val, y_val = x_te_01, y_te
        print("  NOTE: val-mode=match selects weights on the TEST set, "
              "reproducing framework.py's current protocol.")

    counts = class_counts(y_fit)
    cw = fw.effective_number_weights(y_fit)
    mg = ldam_margins(y_fit)
    lp = log_prior(y_fit)

    model = build_model(args.method)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))

    t_train0 = time.time()
    best_val = train_baseline(model, args.method, x_fit, y_fit,
                              x_val, y_val, cw, mg, lp)
    train_sec = time.time() - t_train0

    t_inf0 = time.time()
    metrics = evaluate_method(model, x_te_01, y_te, args.method, lp)
    infer_sec = time.time() - t_inf0

    extra = {}
    if args.method == "ce":
        # One training run also yields the post-hoc logit-adjustment arm.
        extra["la_posthoc"] = {
            k: float(v) for k, v in
            evaluate_method(model, x_te_01, y_te, "la_posthoc", lp).items()
            if k in ("acc", "bacc", "head", "mid", "tail",
                     "head_tail_gap", "worst_f1", "worst_acc")
        }

    print(f"\n--- {args.method} ---")
    for k in ("acc", "bacc", "head", "mid", "tail",
              "head_tail_gap", "worst_f1", "worst_acc"):
        print(f"  {k:>14}: {metrics[k]:.4f}")

    results = {
        "method": args.method,
        "seed": args.seed,
        "val_mode": args.val_mode,
        "use_early_stopping": CFG.USE_EARLY_STOPPING,
        "cls_epochs": CFG.CLS_EPOCHS,
        "warmup_epochs": CFG.WARMUP_EPOCHS,
        "config": {
            "architecture": CFG.ARCHITECTURE,
            "imbalance_protocol": CFG.IMBALANCE_PROTOCOL,
            "rho": CFG.IMBALANCE_RATIO,
            "dataset": CFG.DATASET,
            "epochs": CFG.CLS_EPOCHS,
            "batch_size": CFG.BATCH_SIZE,
            "lr": CFG.CLS_LR,
            "weight_decay": CFG.WEIGHT_DECAY,
            "effnum_beta": CFG.EFFNUM_BETA,
            "ldam_max_margin": LDAM_MAX_MARGIN,
            "ldam_scale": LDAM_SCALE,
            "la_tau": LA_TAU,
            "drw_start_frac": DRW_START_FRAC,
        },
        "cost": {
            "train_sec": train_sec,
            "infer_sec": infer_sec,
            "total_sec": time.time() - t_start,
            "peak_mem_mb": peak_memory_mb(),
            "trainable_params": n_params,
        },
        "n_train": int(len(y_fit)),
        "n_classes_present": int((counts > 1).sum()),
        "best_val_acc": float(best_val),
        "metrics": {k: float(metrics[k]) for k in
                    ("acc", "bacc", "head", "mid", "tail",
                     "head_tail_gap", "worst_f1", "worst_acc")},
    }
    results.update(extra)

    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {path}")
    print(f"train {train_sec / 60:.1f} min   infer {infer_sec:.1f} s   "
          f"peak mem {results['cost']['peak_mem_mb']:.0f} MB")
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   choices=["ce", "cb", "ldam", "ldam_drw", "la"],
                   help="baseline to train (la_posthoc is emitted alongside ce)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--arch", type=str, default="resnet32")
    p.add_argument("--protocol", type=str, default="cui2019")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--val-mode", choices=["match", "holdout"], default="holdout",
                   help="'match' reproduces framework.py (validates on the "
                        "test set) so numbers are comparable with sprint2; "
                        "'holdout' carves a stratified validation split out "
                        "of training data, which is what the paper claims.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--early-stopping", action="store_true",
                   help="re-enable early stopping (off by default, so the "
                        "full cosine schedule runs as it does for REBAL)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
