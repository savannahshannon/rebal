"""
REBAL pipeline v2: improved next-phase implementation.

This script is the follow-on to the pilot reported in Section VII of the paper.
It addresses the four limitations identified there:

    1. Pixel-space SMOTE -> Feature-space SMOTE.
    SMOTE is now applied to penultimate-layer CNN features rather than to
    raw pixels.  Linear interpolation in a learned feature space lies (much
    more) on the class manifold than linear interpolation in pixel space.

    2. Unused cGAN -> Active cGAN-augmented classifier training.
    cGAN samples are mixed into the classifier's training set, with a
    configurable budget per class.

    3. No reweighting -> Effective-number class-balanced loss (Cui et al. 2019).

    4. No representation equalization -> Per-batch equalization regularizer
    penalising small inter-class centroid separation relative to per-class
    feature spread.

    5. Overfitting -> Anti-overfitting block: data augmentation, BatchNorm,
    weight decay, cosine LR schedule, early stopping.

Every improvement is behind a toggle flag at the top of the file so you can
ablate them one at a time, which is the experimental protocol the paper
prescribes.

Tested with TensorFlow 2.15+, Python 3.10+.  Trains in ~30-45 minutes on a
single A100 / T4 / local RTX GPU.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import balanced_accuracy_score, recall_score
from imblearn.over_sampling import SMOTE
from pyod.models.ecod import ECOD

# Root directory for Sprint 1's multi-seed experiment artifacts
# (checkpoints, per-class CSVs, metrics) -- see experiments/phase2_multiseed/.
EXPERIMENT_ROOT = "experiments/phase2_multiseed"


# ======================================================================
# 0. CONFIGURATION
# ======================================================================
# Toggle each improvement independently to ablate.

@dataclass
class Config:
    # Data
    NUM_CLASSES: int = 100
    IMG_SHAPE: tuple = (32, 32, 3)
    IMBALANCE_SLOPE: float = 0.009     # r_k = max(0.10, 1 - slope*k)
    IMBALANCE_FLOOR: float = 0.10
    SEED: int = 42

    # Module toggles (set False to ablate the corresponding improvement)
    USE_ECOD:           bool = True
    USE_FEATURE_SMOTE:  bool = True    # if False, falls back to pixel SMOTE
    USE_CGAN_AUG:       bool = True    # if False, cGAN trained but unused
    USE_REWEIGHTING:    bool = True    # effective-number class weights
    USE_EQUALIZATION:   bool = True    # per-batch equalization regularizer
    USE_DATA_AUG:       bool = True    # crop/flip/color jitter
    USE_WEIGHT_DECAY:   bool = True
    USE_LR_SCHEDULE:    bool = True
    USE_EARLY_STOPPING: bool = True

    # ECOD
    ECOD_CONTAMINATION: float = 0.05

    # SMOTE
    SMOTE_K: int = 5
    FEATURE_DIM: int = 128             # penultimate dim of the classifier

    # cGAN
    LATENT_DIM: int = 128
    GAN_EPOCHS: int = 60
    GAN_LR: float = 2e-4
    CGAN_AUG_PER_CLASS: int = 200    # synthetic images per class
    CGAN_SEED: int = 1234
    CGAN_MIN_REAL: int = 30
    
    # SPRINT 3: pre-generated sample bank (StyleGAN2-ADA v2). When set to a
    # directory with class_XX/NNNN.png layout, sample_cgan() loads from disk
    # instead of training the built-in cGAN. Set to None to use the built-in
    # cGAN as before.
    SAMPLE_BANK_DIR: str = None
    
    # SPRINT 4: hybrid per-class gating. If PER_CLASS_FID_CSV is set to an
    # existing file, load_samples_from_bank() applies a second filter that
    # requires per-class FID < CGAN_MAX_FID for a class to be eligible for
    # cGAN augmentation. Backward-compatible: if the CSV is missing or the
    # field is None, the gate falls back to real-sample-count only.
    PER_CLASS_FID_CSV: str = None
    CGAN_MAX_FID: float = 300.0
    HYBRID_GATE_MIN_RHO: float = 75.0

    # Classifier
    BATCH_SIZE: int = 128
    CLS_EPOCHS: int = 120
    CLS_LR: float = 0.1
    GRAD_CLIP_NORM: float = 1.0
    WEIGHT_DECAY: float = 5e-4
    EARLY_STOP_PATIENCE: int = 12

    # --- Validation protocol -------------------------------------
    # VAL_FRACTION = 0.0 reproduces the original behaviour, in which
    # the TEST set is handed to train_classifier as the validation set
    # and early stopping restores the weights with the best TEST
    # accuracy. That is model selection on the test set. Set
    # VAL_FRACTION > 0 (0.1 is typical) to carve a stratified
    # validation split out of the training data instead, which is the
    # protocol described in the paper.
    VAL_FRACTION: float = 0.0
    VAL_MIN_CLASS_COUNT: int = 5   # smaller classes contribute no val samples
    # "acc"  = plain top-1 on the validation set (original behaviour)
    # "bacc" = mean per-class recall; keeps head classes from dominating
    #          the early-stopping decision on long-tailed validation data
    EARLY_STOP_METRIC: str = "acc"

    # Effective-number reweighting
    EFFNUM_BETA: float = 0.999

    # Equalization regularizer
    LAMBDA_EQ: float = 0.05            # weight on equalization loss
    MIN_PER_CLASS_IN_BATCH: int = 2    # skip eq term if fewer in batch

    # Sprint 2: architecture and imbalance protocol switches.
    # Defaults keep Sprint 1 behaviour intact; flip to "resnet32"/"cui2019"
    # for Sprint 2 runs.
    ARCHITECTURE: str = "small"        # "small" (Sprint 1) | "resnet32"
    IMBALANCE_PROTOCOL: str = "linear" # "linear" (Sprint 1) | "cui2019"
    IMBALANCE_RATIO: float = 100.0     # ρ; used when IMBALANCE_PROTOCOL=="cui2019"
    DATASET: str = "cifar100"          # "cifar100" | "cifar10"

    # Sprint 2.5: decouple cGAN seed, gate cGAN by per-class sample count,
    # defer rebalancing terms, and stabilize training at extreme imbalance.
    # All defaults below reproduce Sprint 1/2 behaviour exactly (no-op);
    # they only activate via ENV vars / CLI flags or the resnet32 preset.
    CGAN_SEED: int = 1234                  # independent of CFG.SEED
    CGAN_MIN_REAL: int = 30                # skip cGAN samples for classes below this real-sample count
    DEFER_REWEIGHTING_FRAC: float = 0.0    # fraction of CLS_EPOCHS before reweighting turns on (0 = always on)
    DEFER_EQUALIZATION_FRAC: float = 0.0   # fraction of CLS_EPOCHS before equalization turns on (0 = always on)
    WARMUP_EPOCHS: int = 0                 # linear LR warmup epochs before cosine decay (0 = no warmup)
    USE_BALANCED_BATCH: bool = False       # class-uniform batch sampling instead of natural distribution


CFG = Config()

# Sprint 2 ENV var overrides — backward-compatible with Sprint 1.
# run_multi_imbalance.py sets these before spawning each subprocess so that
# framework.py does not need to be edited between runs.
_rho_env = os.environ.get("REBAL_RHO")
if _rho_env is not None:
    # CFG.IMBALANCE_RHO = float(_rho_env)
    CFG.IMBALANCE_RATIO = float(_rho_env)
    CFG.IMBALANCE_PROTOCOL = "cui2019"
def _apply_arch_preset():
    """Sprint 2.5 stability preset for ResNet-32 on CIFAR-LT.

    Called from BOTH the REBAL_ARCH env path and the --arch CLI path so the
    two entry points cannot diverge. Previously only the env path applied it,
    so `--arch resnet32` ran ResNet-32 with the Sprint 1 defaults
    (120 epochs, no warmup, no deferral, patience 12).

    Individual REBAL_* env vars are applied after this at module load, and
    explicit CLI flags are applied after this in main(), so both still win.
    """
    if CFG.ARCHITECTURE != "resnet32":
        return
    CFG.CLS_EPOCHS = 200   # published training budget for ResNet-32 on CIFAR-LT
    # Sprint 2.5 stability preset (Cao et al. 2019 DRW-style deferral).
    CFG.WARMUP_EPOCHS = 10
    CFG.DEFER_REWEIGHTING_FRAC = 0.8
    CFG.DEFER_EQUALIZATION_FRAC = 0.8
    # At rho=100, val_acc swings ±5-8pp per epoch during the middle
    # training phase. Patience=12 (Sprint 1 default) fires too early,
    # stopping seeds 40-130 epochs before they would have converged.
    CFG.EARLY_STOP_PATIENCE = 40


if os.environ.get("REBAL_ARCH"):
    CFG.ARCHITECTURE = os.environ["REBAL_ARCH"]
    _apply_arch_preset()
if os.environ.get("REBAL_HYBRID_GATE_MIN_RHO"):
    CFG.HYBRID_GATE_MIN_RHO = float(os.environ["REBAL_HYBRID_GATE_MIN_RHO"])
if os.environ.get("REBAL_DATASET"):
    CFG.DATASET = os.environ["REBAL_DATASET"]
    CFG.NUM_CLASSES = 10 if CFG.DATASET == "cifar10" else 100
_ep_env = os.environ.get("REBAL_EPOCHS_OVERRIDE")
if os.environ.get("REBAL_SAMPLE_BANK_DIR"):
    CFG.SAMPLE_BANK_DIR = os.environ["REBAL_SAMPLE_BANK_DIR"]
if _ep_env:
    _ep = int(_ep_env)
    CFG.CLS_EPOCHS = _ep
    CFG.GAN_EPOCHS = min(CFG.GAN_EPOCHS, _ep)
if os.environ.get("REBAL_BASELINE_ONLY") == "1":
    CFG.USE_CGAN_AUG      = False
    CFG.USE_FEATURE_SMOTE = True
    CFG.USE_REWEIGHTING   = False
    CFG.USE_EQUALIZATION  = False

# Sprint 2.5 ENV var overrides — explicit knobs for the calibration sweep
# (warmup, LR, balanced batching) and the rho=100 module ablation grid.
if os.environ.get("REBAL_EARLY_STOP_PATIENCE"):
    CFG.EARLY_STOP_PATIENCE = int(os.environ["REBAL_EARLY_STOP_PATIENCE"])
if os.environ.get("REBAL_VAL_FRACTION"):
    CFG.VAL_FRACTION = float(os.environ["REBAL_VAL_FRACTION"])
if os.environ.get("REBAL_USE_EARLY_STOPPING") is not None:
    # "0" trains the full cosine schedule and keeps the best-validation
    # weights without ever terminating early. With cosine decay most of
    # the gain arrives in the final third, so early stopping and the
    # schedule work against each other.
    CFG.USE_EARLY_STOPPING = os.environ["REBAL_USE_EARLY_STOPPING"] not in ("0", "false", "False")
if os.environ.get("REBAL_EARLY_STOP_METRIC"):
    CFG.EARLY_STOP_METRIC = os.environ["REBAL_EARLY_STOP_METRIC"]
if os.environ.get("REBAL_CGAN_SEED"):
    CFG.CGAN_SEED = int(os.environ["REBAL_CGAN_SEED"])
if os.environ.get("REBAL_CGAN_MIN_REAL"):
    CFG.CGAN_MIN_REAL = int(os.environ["REBAL_CGAN_MIN_REAL"])
if os.environ.get("REBAL_WARMUP_EPOCHS"):
    CFG.WARMUP_EPOCHS = int(os.environ["REBAL_WARMUP_EPOCHS"])
if os.environ.get("REBAL_CLS_LR"):
    CFG.CLS_LR = float(os.environ["REBAL_CLS_LR"])
if os.environ.get("REBAL_BALANCED_BATCH") == "1":
    CFG.USE_BALANCED_BATCH = True
if os.environ.get("REBAL_BALANCE_BATCH") == "0":
    CFG.USE_BALANCED_BATCH = False
if os.environ.get("REBAL_DEFER_FRAC"):
    _frac = float(os.environ["REBAL_DEFER_FRAC"])
    CFG.DEFER_REWEIGHTING_FRAC = _frac
    CFG.DEFER_EQUALIZATION_FRAC = _frac
if os.environ.get("REBAL_SEED") is not None:
    CFG.SEED = int(os.environ["REBAL_SEED"])
if os.environ.get("REBAL_ABLATE"):
    _ablate = os.environ["REBAL_ABLATE"]
    if _ablate == "cgan":
        CFG.USE_CGAN_AUG = False
    elif _ablate == "eq":
        CFG.USE_EQUALIZATION = False
    elif _ablate == "rw":
        CFG.USE_REWEIGHTING = False
    elif _ablate == "ecod":
        CFG.USE_ECOD = False
    elif _ablate == "feature_smote":
        CFG.USE_FEATURE_SMOTE = False
    elif _ablate == "decoupled":
        CFG.USE_CGAN_AUG     = False
        CFG.USE_REWEIGHTING  = False
        CFG.USE_EQUALIZATION = False
        # USE_FEATURE_SMOTE stays True so the cRT step still runs
        # (Kang et al. 2020 decoupled recipe: vanilla trunk + cRT head).
        
# Sprint 4 Addition
if os.environ.get("REBAL_PER_CLASS_FID_CSV"):
    CFG.PER_CLASS_FID_CSV = os.environ["REBAL_PER_CLASS_FID_CSV"]
if os.environ.get("REBAL_CGAN_MAX_FID"):
    CFG.CGAN_MAX_FID = float(os.environ["REBAL_CGAN_MAX_FID"])


def set_seeds(seed):
    """Re-seed TF/NumPy/Python RNGs and update CFG.SEED, which is also read
    downstream (SMOTE random_state, dataset shuffle seed)."""
    CFG.SEED = seed
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


set_seeds(CFG.SEED)

# Prevent TensorFlow from grabbing all available memory at startup.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)


# ======================================================================
# 1. DATA LOADING + ARTIFICIAL IMBALANCE
# ======================================================================
def _load_raw_cifar():
    """Load the raw CIFAR dataset specified by CFG.DATASET, normalised to [-1,1]."""
    loader = (tf.keras.datasets.cifar10 if CFG.DATASET == "cifar10"
              else tf.keras.datasets.cifar100)
    (x_tr, y_tr), (x_te, y_te) = loader.load_data()
    y_tr, y_te = y_tr.flatten(), y_te.flatten()
    x_tr = (x_tr.astype("float32") - 127.5) / 127.5    # [-1, 1] for the GAN
    return x_tr, y_tr, x_te, y_te


def _load_imbalanced_cui(rho):
    """Cui et al. (2019) exponential-decay imbalance schedule.

    n_i = N_max * (1/rho)^(i/(K-1))

    Works for both CIFAR-100 (N_max=500) and CIFAR-10 (N_max=5000).
    Classes are sorted by index so class 0 is the majority class and
    class K-1 is the rarest, matching the published protocol.
    """
    x_tr, y_tr, x_te, y_te = _load_raw_cifar()
    K = CFG.NUM_CLASSES
    counts = np.bincount(y_tr, minlength=K)
    n_max = int(counts.max())
    idx_keep = []
    for k in range(K):
        idx_k = np.where(y_tr == k)[0]
        n_k = max(1, int(round(n_max * (1.0 / rho) ** (k / (K - 1)))))
        n_k = min(n_k, len(idx_k))
        idx_keep.append(np.random.choice(idx_k, size=n_k, replace=False))
    idx_keep = np.concatenate(idx_keep)
    return x_tr[idx_keep], y_tr[idx_keep], x_te, y_te


def load_imbalanced_cifar100():
    """Return imbalanced training set (x in [-1, 1]) and balanced test set.

    Dispatches to Cui et al. (2019) exponential-decay protocol when
    CFG.IMBALANCE_PROTOCOL == "cui2019", otherwise uses the Sprint 1
    linear-slope protocol.
    """
    if CFG.IMBALANCE_PROTOCOL == "cui2019":
        return _load_imbalanced_cui(CFG.IMBALANCE_RATIO)

    # Legacy linear protocol (Sprint 1 default)
    x_tr, y_tr, x_te, y_te = _load_raw_cifar()
    keep = np.maximum(CFG.IMBALANCE_FLOOR,
                      1.0 - CFG.IMBALANCE_SLOPE * np.arange(CFG.NUM_CLASSES))
    idx_keep = []
    for k in range(CFG.NUM_CLASSES):
        idx_k = np.where(y_tr == k)[0]
        n_keep = int(len(idx_k) * keep[k])
        idx_keep.append(np.random.choice(idx_k, size=n_keep, replace=False))
    idx_keep = np.concatenate(idx_keep)
    return x_tr[idx_keep], y_tr[idx_keep], x_te, y_te


# ======================================================================
# 2. ECOD CLEANING  (Module 1)
# ======================================================================
def ecod_clean(x, y, contamination=CFG.ECOD_CONTAMINATION):
    """Per-class outlier removal on flattened pixels."""
    keep_x, keep_y = [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        if len(idx) < 10:
            keep_x.append(x[idx]); keep_y.append(y[idx]); continue
        flat = x[idx].reshape(len(idx), -1)
        clf = ECOD(contamination=contamination, n_jobs=1)
        clf.fit(flat)
        inliers = clf.labels_ == 0
        keep_x.append(x[idx][inliers]); keep_y.append(y[idx][inliers])
    return np.concatenate(keep_x), np.concatenate(keep_y)


# ======================================================================
# 3. FEATURE-SPACE SMOTE  (Module 3, improved)
# ======================================================================
def smote_in_feature_space(features, labels, k_neighbors=CFG.SMOTE_K):
    """Apply SMOTE in feature space; returns (feat_aug, lab_aug)."""
    nan_mask = ~np.isnan(features).any(axis=1)
    if not nan_mask.all():
        n_bad = int((~nan_mask).sum())
        print(f"  [SMOTE] WARNING: dropping {n_bad}/{len(features)} samples "
            f"with NaN features — feature extractor may be unstable")
        features, labels = features[nan_mask], labels[nan_mask]
    counts = Counter(labels)
    target = max(counts.values())
    strategy = {c: target for c in counts if counts[c] < target}
    if not strategy:
        return features, labels
    smote = SMOTE(sampling_strategy=strategy,random_state=CFG.SEED,
    k_neighbors=min(k_neighbors, min(counts.values()) - 1))
    return smote.fit_resample(features, labels)


def smote_in_pixel_space(x_img, y, k_neighbors=CFG.SMOTE_K):
    """Pixel-space SMOTE (the pilot's approach), included for ablation."""
    flat = x_img.reshape(len(x_img), -1)
    counts = Counter(y)
    target = max(counts.values())
    strategy = {c: target for c in counts if counts[c] < target}
    if not strategy:
        return x_img, y
    smote = SMOTE(sampling_strategy=strategy, random_state=CFG.SEED,
    k_neighbors=min(k_neighbors, min(counts.values()) - 1))
    flat_bal, y_bal = smote.fit_resample(flat, y)
    return flat_bal.reshape(-1, *CFG.IMG_SHAPE), y_bal


# ======================================================================
# 4. cGAN  (Module 3, pixel-space synthesis)
# ======================================================================
def build_generator():
    z = layers.Input(shape=(CFG.LATENT_DIM,))
    lab = layers.Input(shape=(1,), dtype="int32")
    emb = layers.Flatten()(layers.Embedding(CFG.NUM_CLASSES, 100)(lab))
    h = layers.Concatenate()([z, emb])
    h = layers.Dense(4 * 4 * 256, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU(0.2)(h)
    h = layers.Reshape((4, 4, 256))(h)
    for ch in (128, 64, 32):
        h = layers.Conv2DTranspose(ch, 4, 2, "same", use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
    out = layers.Conv2D(3, 3, 1, "same", activation="tanh")(h)
    return tf.keras.Model([z, lab], out, name="generator")


def build_discriminator():
    img = layers.Input(shape=CFG.IMG_SHAPE)
    lab = layers.Input(shape=(1,), dtype="int32")
    emb = layers.Embedding(CFG.NUM_CLASSES, 100)(lab)
    emb = layers.Dense(CFG.IMG_SHAPE[0] * CFG.IMG_SHAPE[1])(emb)
    emb = layers.Reshape((CFG.IMG_SHAPE[0], CFG.IMG_SHAPE[1], 1))(emb)
    h = layers.Concatenate()([img, emb])
    for ch in (64, 128, 256):
        h = layers.Conv2D(ch, 3, 2, "same")(h)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Dropout(0.4)(h)
    h = layers.Flatten()(h)
    out = layers.Dense(1, activation="sigmoid")(h)
    return tf.keras.Model([img, lab], out, name="discriminator")


def train_cgan(x, y, epochs=CFG.GAN_EPOCHS):
    # Sprint 2.5: the cGAN gets its own seed namespace, independent of
    # CFG.SEED. Without this, every classifier seed inherits whatever
    # noise the cGAN's initialization happened to produce, which couples
    # classifier variance to generator variance and makes it impossible
    # to tell which one is responsible for an unstable run.
    _classifier_seed = CFG.SEED
    tf.keras.utils.set_random_seed(CFG.CGAN_SEED)
    g, d = build_generator(), build_discriminator()
    bce = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(CFG.GAN_LR, beta_1=0.5)
    d_opt = tf.keras.optimizers.Adam(CFG.GAN_LR, beta_1=0.5)
    ds = (tf.data.Dataset.from_tensor_slices((x, y))
        .shuffle(len(x)).batch(CFG.BATCH_SIZE))

    @tf.function
    def step(real, lab):
        z = tf.random.normal([tf.shape(real)[0], CFG.LATENT_DIM])
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = g([z, lab], training=True)
            r_out = d([real, lab], training=True)
            f_out = d([fake, lab], training=True)
            d_loss = bce(tf.ones_like(r_out) * 0.9, r_out) + \
                    bce(tf.zeros_like(f_out), f_out)
            g_loss = bce(tf.ones_like(f_out), f_out)
        g_opt.apply_gradients(zip(gt.gradient(g_loss, g.trainable_variables),
                                g.trainable_variables))
        d_opt.apply_gradients(zip(dt.gradient(d_loss, d.trainable_variables),
                                d.trainable_variables))
        return g_loss, d_loss

    for ep in range(epochs):
        t0 = time.time(); gl = dl = n = 0
        for xb, yb in ds:
            a, b = step(xb, yb)
            gl += a; dl += b; n += 1
        print(f"  cGAN ep {ep+1:3d}/{epochs}  "
            f"g={gl/n:.3f}  d={dl/n:.3f}  ({time.time()-t0:.1f}s)")
    tf.keras.utils.set_random_seed(_classifier_seed)   # restore classifier seed
    return g


def sample_cgan(generator, y_train_real, per_class=CFG.CGAN_AUG_PER_CLASS,
                min_real_samples=None):
    """Generate `per_class` images for each class with enough real support.

    Sprint 2.5: at extreme imbalance (e.g. rho=100) tail classes can have
    fewer than 10 real images. A cGAN trained on that few examples per
    class produces low-quality, often label-confused samples; mixing those
    into classifier training poisons the trunk rather than helping it.
    Classes below `min_real_samples` real images are skipped entirely —
    their representation increase comes from feature-space SMOTE instead,
    where even a handful of real samples is enough to interpolate from.
    """
    if min_real_samples is None:
        min_real_samples = CFG.CGAN_MIN_REAL
    real_counts = Counter(y_train_real)
    eligible = [c for c in range(CFG.NUM_CLASSES)
                if real_counts.get(c, 0) >= min_real_samples]
    if not eligible:
        print(f"  cGAN gated: 0/{CFG.NUM_CLASSES} classes meet the "
              f"{min_real_samples}-real-sample threshold — skipping cGAN augmentation")
        return (np.zeros((0, *CFG.IMG_SHAPE), dtype="float32"),
                np.zeros((0,), dtype="int32"))
    n = per_class * len(eligible)
    z = tf.random.normal([n, CFG.LATENT_DIM])
    lab = np.repeat(eligible, per_class).astype(np.int32)
    imgs = generator.predict([z, lab], batch_size=512, verbose=0)
    print(f"  cGAN gated: generated {n} images across "
          f"{len(eligible)}/{CFG.NUM_CLASSES} classes "
          f"(threshold = {min_real_samples} real samples)")
    return imgs, lab

def load_samples_from_bank(bank_dir, y_train_real,
                           per_class=CFG.CGAN_AUG_PER_CLASS,
                           min_real_samples=None,
                           rng_seed=None):
    """Load pre-generated PNG samples from disk (Sprint 3 StyleGAN2-ADA path).

    Applies the framework's per-class gating logic:
      1. Real-sample gate (always active): class requires >= min_real_samples
         real training images to be eligible for cGAN augmentation.
      2. Per-class FID gate (Sprint 4 hybrid gating): additionally requires
         per-class FID < CGAN_MAX_FID. Only activated when the current
         imbalance factor CFG.IMBALANCE_RHO >= CFG.HYBRID_GATE_MIN_RHO,
         because empirically the FID gate destabilizes training at low
         imbalance while recovering worst-class accuracy at high imbalance.

    Layout expected:
        bank_dir/
            class_00/ 0000.png 0001.png ...
            class_01/ ...
            ...

    Returns (imgs_uint8_01, labels) where imgs are in [0, 1] range,
    shape (N, H, W, C), matching what sample_cgan() would return after
    its post-processing.
    """
    from pathlib import Path
    from collections import Counter
    from PIL import Image as _PILImage

    bank = Path(bank_dir)
    if not bank.exists():
        raise FileNotFoundError(f"SAMPLE_BANK_DIR does not exist: {bank}")

    if min_real_samples is None:
        min_real_samples = CFG.CGAN_MIN_REAL
    if rng_seed is None:
        rng_seed = CFG.SEED
    rng = np.random.default_rng(rng_seed)

    # -------- Per-class real-sample counts (always used) --------
    real_counts = Counter(y_train_real.tolist()
                          if hasattr(y_train_real, "tolist")
                          else list(y_train_real))

    # -------- Per-class FID (used when hybrid gate is active) --------
    per_class_fid = load_per_class_fid(CFG.PER_CLASS_FID_CSV)

    # -------- Sprint 4 refinement: is the FID gate active at this rho? --------
    # Read from env var directly if present, since CFG.IMBALANCE_RHO may not
    # be updated by the framework's env-var override block.
    if os.environ.get("REBAL_RHO") is not None:
        rho = float(os.environ["REBAL_RHO"])
    else:
        rho = float(getattr(CFG, "IMBALANCE_RATIO", 100.0))
    hybrid_gate_min = float(getattr(CFG, "HYBRID_GATE_MIN_RHO", 75.0))
    fid_gate_active = bool(per_class_fid) and rho >= hybrid_gate_min

    def _passes_gate(c):
        if real_counts.get(c, 0) < min_real_samples:
            return False
        if fid_gate_active:
            fid = per_class_fid.get(c, float("inf"))
            if fid >= CFG.CGAN_MAX_FID:
                return False
        return True

    eligible = [c for c in range(CFG.NUM_CLASSES) if _passes_gate(c)]

    # -------- Logging: reflect which gate is actually active --------
    if fid_gate_active:
        excluded_by_fid = [
            c for c in range(CFG.NUM_CLASSES)
            if real_counts.get(c, 0) >= min_real_samples
               and per_class_fid.get(c, float("inf")) >= CFG.CGAN_MAX_FID
        ]
        print(f"  sample bank (hybrid gate active, rho={rho:.0f} "
              f">= HYBRID_GATE_MIN_RHO={hybrid_gate_min:.0f}): "
              f"{len(eligible)}/{CFG.NUM_CLASSES} classes eligible "
              f"(>= {min_real_samples} real AND per-class FID < {CFG.CGAN_MAX_FID})")
        print(f"  {len(excluded_by_fid)} classes excluded by FID filter alone: "
              f"{excluded_by_fid}")
    elif per_class_fid:
        print(f"  sample bank (FID gate available but inactive: "
              f"rho={rho:.0f} < HYBRID_GATE_MIN_RHO={hybrid_gate_min:.0f}): "
              f"{len(eligible)}/{CFG.NUM_CLASSES} classes eligible "
              f"(real-sample gate only)")
    else:
        print(f"  sample bank: {len(eligible)}/{CFG.NUM_CLASSES} classes "
              f"eligible (>= {min_real_samples} real samples)")

    if not eligible:
        return (np.zeros((0, *CFG.IMG_SHAPE), dtype="float32"),
                np.zeros((0,), dtype=np.int32))

    # -------- Load samples for eligible classes --------
    all_imgs, all_labels = [], []
    total_missing = 0
    for c in eligible:
        class_dir = bank / f"class_{c:02d}"
        if not class_dir.exists():
            total_missing += 1
            continue
        pngs = sorted(class_dir.glob("*.png"))
        if not pngs:
            total_missing += 1
            continue

        # Sample per_class images (with replacement if bank is smaller)
        replace = len(pngs) < per_class
        picks = rng.choice(pngs, size=per_class, replace=replace)
        for p in picks:
            img = np.array(_PILImage.open(p).convert("RGB"), dtype=np.uint8)
            all_imgs.append(img)
            all_labels.append(c)

    if total_missing:
        print(f"  WARNING: {total_missing} eligible classes had no PNGs "
              f"in the bank; those classes get no cGAN samples.")

    imgs = np.stack(all_imgs, axis=0).astype("float32") / 255.0
    labels = np.array(all_labels, dtype=np.int32)
    print(f"  sample bank: loaded {len(imgs)} images "
          f"from {bank} into [0, 1] range")
    return imgs, labels

# Sprint 4 Addition: hybrid per-class gating by FID. If a CSV is provided, only classes
def load_per_class_fid(csv_path):
    """Load per-class FID from a CSV file with columns:
    class_idx, class_name, per_class_fid, n_real, n_synth.

    Returns {class_idx: per_class_fid} dict, or {} if file missing.
    Missing values (NaN) are treated as +inf so those classes never pass
    the FID filter.
    """
    import csv, math
    result = {}
    if not csv_path:
        return result
    from pathlib import Path
    p = Path(csv_path)
    if not p.exists():
        print(f"  WARNING: PER_CLASS_FID_CSV not found at {p}; "
              f"hybrid gate falls back to real-sample-count only.")
        return result
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["class_idx"])
                fid_str = row.get("per_class_fid", "nan")
                fid = float(fid_str) if fid_str != "nan" else float("inf")
                if math.isnan(fid):
                    fid = float("inf")
                result[idx] = fid
            except (ValueError, KeyError):
                continue
    return result
# ======================================================================
# 5. CLASSIFIER  (improved: BN, augmentation, weight decay, ResNet-ish)
# ======================================================================
def res_block(x, ch, stride=1, l2=CFG.WEIGHT_DECAY):
    reg = regularizers.l2(l2) if CFG.USE_WEIGHT_DECAY else None
    h = layers.Conv2D(ch, 3, stride, "same",
                    kernel_regularizer=reg, use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Conv2D(ch, 3, 1, "same",
                    kernel_regularizer=reg, use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    if stride > 1 or x.shape[-1] != ch:
        x = layers.Conv2D(ch, 1, stride, "same",
                        kernel_regularizer=reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    return layers.ReLU()(layers.Add()([h, x]))


def build_resnet32_classifier(feature_dim=CFG.FEATURE_DIM):
    """Canonical ResNet-32 for CIFAR (He et al. 2016, Table 1).

    3 stages × 5 residual blocks, filters [16, 32, 64], ~464K params.
    No max-pool after the initial conv — CIFAR images are only 32×32.
    Reuses res_block() from the existing pipeline for consistency.
    """
    inp = layers.Input(shape=CFG.IMG_SHAPE)
    if CFG.USE_DATA_AUG:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
        ], name="aug")
        h = aug(inp)
    else:
        h = inp

    reg = regularizers.l2(CFG.WEIGHT_DECAY) if CFG.USE_WEIGHT_DECAY else None
    # Initial 3×3 conv with 16 filters; no max-pool (images are 32×32)
    h = layers.Conv2D(16, 3, 1, "same", use_bias=False,
                      kernel_regularizer=reg)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)

    # Stage 1: 5 blocks, 16 filters, 32×32
    for _ in range(5):
        h = res_block(h, 16)
    # Stage 2: 5 blocks, 32 filters, 16×16 (stride-2 on first block)
    h = res_block(h, 32, stride=2)
    for _ in range(4):
        h = res_block(h, 32)
    # Stage 3: 5 blocks, 64 filters, 8×8 (stride-2 on first block)
    h = res_block(h, 64, stride=2)
    for _ in range(4):
        h = res_block(h, 64)

    h = layers.GlobalAveragePooling2D()(h)
    feat = layers.Dense(feature_dim, activation="relu", name="feat",
                        kernel_regularizer=reg)(h)
    feat = layers.Dropout(0.3)(feat)
    logits = layers.Dense(CFG.NUM_CLASSES, name="logits",
                          kernel_regularizer=reg)(feat)
    return tf.keras.Model(inp, [logits, feat], name="resnet32")


def build_classifier(feature_dim=CFG.FEATURE_DIM):
    """Returns a model that outputs (logits, features) for the eq. regularizer."""
    if CFG.ARCHITECTURE == "resnet32":
        return build_resnet32_classifier(feature_dim)
    inp = layers.Input(shape=CFG.IMG_SHAPE)

    # Per-batch augmentation block (active only during training)
    if CFG.USE_DATA_AUG:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ], name="aug")
        h = aug(inp)
    else:
        h = inp

    reg = regularizers.l2(CFG.WEIGHT_DECAY) if CFG.USE_WEIGHT_DECAY else None
    h = layers.Conv2D(32, 3, 1, "same", kernel_regularizer=reg,
                    use_bias=False)(h)
    h = layers.BatchNormalization()(h); h = layers.ReLU()(h)

    h = res_block(h, 32)
    h = res_block(h, 64, 2)
    h = res_block(h, 128, 2)
    h = res_block(h, 256, 2)

    h = layers.GlobalAveragePooling2D()(h)
    feat = layers.Dense(feature_dim, activation="relu", name="feat",
                        kernel_regularizer=reg)(h)
    feat = layers.Dropout(0.3)(feat)
    logits = layers.Dense(CFG.NUM_CLASSES, name="logits",
                        kernel_regularizer=reg)(feat)
    return tf.keras.Model(inp, [logits, feat], name="classifier")


# ======================================================================
# 6. EFFECTIVE-NUMBER REWEIGHTING  (Module 4)
# ======================================================================
def stratified_val_indices(y, frac=None, seed=None, min_count=None):
    """Split training indices into (train, val). Never touches the test set.

    Classes with fewer than VAL_MIN_CLASS_COUNT samples contribute nothing
    to the validation set: at rho=100 the smallest CIFAR-100-LT classes
    hold only a handful of images, and giving one away costs more than the
    validation signal is worth. The validation set is therefore still
    long-tailed, which is why EARLY_STOP_METRIC='bacc' is recommended
    alongside this.
    """
    frac = CFG.VAL_FRACTION if frac is None else frac
    seed = CFG.SEED if seed is None else seed
    min_count = CFG.VAL_MIN_CLASS_COUNT if min_count is None else min_count
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for k in range(CFG.NUM_CLASSES):
        idx = np.where(y == k)[0]
        if len(idx) == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = int(round(frac * len(idx))) if len(idx) >= min_count else 0
        n_val = min(n_val, len(idx) - 1)          # never empty a class in train
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    return np.array(sorted(tr), dtype=int), np.array(sorted(va), dtype=int)


def effective_number_weights(y, beta=CFG.EFFNUM_BETA):
    counts = np.bincount(y, minlength=CFG.NUM_CLASSES).astype("float64")
    counts = np.maximum(counts, 1.0)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = (1.0 / eff)
    w = w / w.sum() * CFG.NUM_CLASSES    # normalise so mean weight is 1
    return w.astype("float32")


# ======================================================================
# 7. EQUALIZATION REGULARIZER  (Module 4)
# ======================================================================
def equalization_loss(features, labels, num_classes=CFG.NUM_CLASSES):
    """Penalise small inter-class centroid separation relative to spread.

    For each class present in the batch with at least
    MIN_PER_CLASS_IN_BATCH samples, compute centroid and average spread.
    Return:  -mean pairwise centroid distance^2  +  mean per-class spread.

    Features are L2-normalised onto the unit sphere before any computation so
    that all squared distances lie in [0, 4].  Without this the sep_mean term
    is unbounded, the loss diverges to -inf, and it overwhelms cls_loss.
    """
    # Project to unit sphere: bounds eq_loss to [-4, 4] regardless of
    # feature scale, keeping LAMBDA_EQ * eq_loss a small perturbation on
    # top of the ~4.6-nats cross-entropy at initialisation.
    features = tf.math.l2_normalize(features, axis=-1)

    one_hot = tf.one_hot(labels, num_classes)               # (B, K)
    n_c = tf.reduce_sum(one_hot, axis=0)                    # (K,)
    present = n_c >= CFG.MIN_PER_CLASS_IN_BATCH             # (K,)

    # Per-class sums of features:  (K, d)
    sum_c = tf.matmul(one_hot, features, transpose_a=True)
    mu_c = sum_c / tf.maximum(tf.reshape(n_c, [-1, 1]), 1.0)

    # Per-class spread: mean ||f - mu||^2 over samples in class
    diff = tf.expand_dims(features, 1) - tf.expand_dims(mu_c, 0)   # (B,K,d)
    sq = tf.reduce_sum(diff * diff, axis=-1)                       # (B,K)
    weighted_sq = sq * one_hot                                     # only own class
    spread_c = tf.reduce_sum(weighted_sq, axis=0) / tf.maximum(n_c, 1.0)
    spread_mean = tf.reduce_mean(tf.boolean_mask(spread_c, present))

    # Pairwise centroid distances among present classes
    mu_p = tf.boolean_mask(mu_c, present)
    if tf.shape(mu_p)[0] >= 2:
        diffs = tf.expand_dims(mu_p, 0) - tf.expand_dims(mu_p, 1)
        d2 = tf.reduce_sum(diffs * diffs, axis=-1)
        # Exclude self-pairs (zeros on the diagonal)
        n_p = tf.cast(tf.shape(mu_p)[0], tf.float32)
        sep_mean = tf.reduce_sum(d2) / tf.maximum(n_p * (n_p - 1.0), 1.0)
    else:
        sep_mean = tf.constant(0.0)

    # Lower spread + larger separation is better => loss = spread - sep
    return spread_mean - sep_mean


# ======================================================================
# 8. CUSTOM TRAINING LOOP  (visible enough that you can edit it)
# ======================================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear LR warmup for `warmup_steps`, then cosine decay to 0.

    Sprint 2.5: at high imbalance ratios, starting straight at the peak
    LR can destabilize BatchNorm running stats before the trunk has seen
    enough of the (very few) tail-class examples. A short linear ramp-up
    fixes this without changing the eventual peak LR or decay shape.
    """
    def __init__(self, peak_lr, warmup_steps, decay_steps):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)
        self.cosine = tf.keras.optimizers.schedules.CosineDecay(
            peak_lr, decay_steps=max(1, decay_steps))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        decayed_lr = self.cosine(step - tf.cast(self.warmup_steps, tf.float32))
        return tf.where(step < tf.cast(self.warmup_steps, tf.float32),
                        warmup_lr, decayed_lr)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps}


def _build_balanced_dataset(x_train, y_train):
    """Class-uniform batch sampling: each batch draws every class with
    equal probability, regardless of the natural (imbalanced) frequency.

    Sprint 2.5: under heavy imbalance, natural-distribution batches at
    rho=100 can go many steps without seeing a tail class at all, which
    destabilizes BatchNorm running statistics. Sampling per-class datasets
    uniformly keeps every class represented in (almost) every batch.
    """
    classes = np.unique(y_train)
    per_class_ds = []
    for c in classes:
        idx_c = np.where(y_train == c)[0]
        ds_c = (tf.data.Dataset.from_tensor_slices((x_train[idx_c], y_train[idx_c]))
                .shuffle(len(idx_c), seed=CFG.SEED, reshuffle_each_iteration=True)
                .repeat())
        per_class_ds.append(ds_c)
    balanced = tf.data.Dataset.sample_from_datasets(
        per_class_ds, weights=[1.0 / len(classes)] * len(classes), seed=CFG.SEED)
    steps_per_epoch = max(1, len(x_train) // CFG.BATCH_SIZE)
    return (balanced.batch(CFG.BATCH_SIZE)
            .take(steps_per_epoch)
            .prefetch(tf.data.AUTOTUNE))


def train_classifier(model, x_train, y_train, x_val, y_val, class_weights):
    steps_per_epoch = max(1, len(x_train) // CFG.BATCH_SIZE)
    if CFG.USE_LR_SCHEDULE:
        decay_steps = steps_per_epoch * max(1, CFG.CLS_EPOCHS - CFG.WARMUP_EPOCHS)
        if CFG.WARMUP_EPOCHS > 0:
            lr = WarmupCosineDecay(CFG.CLS_LR, steps_per_epoch * CFG.WARMUP_EPOCHS,
                                    decay_steps)
        else:
            lr = tf.keras.optimizers.schedules.CosineDecay(
                CFG.CLS_LR, decay_steps=decay_steps)
    else:
        lr = CFG.CLS_LR
    opt = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True,
                                clipnorm=CFG.GRAD_CLIP_NORM)

    ce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    cw = tf.constant(class_weights, dtype=tf.float32)

    # Sprint 2.5: deferred rebalancing (Cao et al. 2019 DRW). cur_epoch is a
    # tf.Variable (not a Python int) so train_step reads it without
    # retracing the @tf.function every epoch.
    cur_epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
    defer_rw_epoch = int(CFG.DEFER_REWEIGHTING_FRAC * CFG.CLS_EPOCHS)
    defer_eq_epoch = int(CFG.DEFER_EQUALIZATION_FRAC * CFG.CLS_EPOCHS)

    if CFG.USE_BALANCED_BATCH:
        ds = _build_balanced_dataset(x_train, y_train)
    else:
        ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(min(20000, len(x_train)), seed=CFG.SEED)
            .batch(CFG.BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

    @tf.function
    def train_step(xb, yb):
        # CIFAR-10 labels arrive as uint8; tf.gather and tf.one_hot need
        # int32/int64. Cast once here rather than in every call site.
        yb = tf.cast(yb, tf.int32)
        with tf.GradientTape() as tape:
            logits, feats = model(xb, training=True)
            per_sample = ce(yb, logits)
            if CFG.USE_REWEIGHTING:
                use_w = tf.cast(cur_epoch >= defer_rw_epoch, tf.float32)
                sample_w = use_w * tf.gather(cw, yb) + (1.0 - use_w)
                cls_loss = tf.reduce_mean(per_sample * sample_w)
            else:
                cls_loss = tf.reduce_mean(per_sample)
            loss = cls_loss
            if CFG.USE_EQUALIZATION:
                use_eq = tf.cast(cur_epoch >= defer_eq_epoch, tf.float32)
                loss = loss + use_eq * CFG.LAMBDA_EQ * equalization_loss(feats, yb)
            loss = loss + tf.add_n([tf.cast(l, loss.dtype)
                                    for l in model.losses]) if model.losses else loss
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g
                for g, v in zip(grads, model.trainable_variables)]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, cls_loss

    if defer_rw_epoch > 0 or defer_eq_epoch > 0:
        print(f"  deferred rebalancing: reweighting@{defer_rw_epoch} "
              f"equalization@{defer_eq_epoch} (of {CFG.CLS_EPOCHS} epochs)")

    best_val, best_w, patience = -1.0, None, 0
    for ep in range(CFG.CLS_EPOCHS):
        cur_epoch.assign(ep)
        t0 = time.time(); tl = tcl = 0.0; n = 0
        for xb, yb in ds:
            l, cl = train_step(xb, yb)
            tl += float(l); tcl += float(cl); n += 1
        # Validation — run in batches to avoid an OOM spike each epoch
        val_chunks = []
        for i in range(0, len(x_val), CFG.BATCH_SIZE):
            chunk_logits, _ = model(x_val[i:i + CFG.BATCH_SIZE], training=False)
            val_chunks.append(tf.argmax(chunk_logits, axis=-1, output_type=tf.int32))
        val_preds = tf.concat(val_chunks, axis=0)
        if CFG.EARLY_STOP_METRIC == "bacc":
            _vp = val_preds.numpy()
            _vy = np.asarray(y_val)
            _rec = [float((_vp[_vy == k] == k).mean())
                    for k in np.unique(_vy)]
            val_acc = float(np.mean(_rec)) if _rec else 0.0
        else:
            val_acc = float(tf.reduce_mean(tf.cast(tf.equal(val_preds, y_val), tf.float32)))
        print(f"  ep {ep+1:3d}/{CFG.CLS_EPOCHS}  loss={tl/n:.3f}  "
            f"cls={tcl/n:.3f}  val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)")
        if CFG.USE_EARLY_STOPPING:
            if val_acc > best_val:
                best_val, best_w, patience = val_acc, model.get_weights(), 0
            else:
                patience += 1
                if patience >= CFG.EARLY_STOP_PATIENCE:
                    print(f"  early stop (best val_acc={best_val:.4f})")
                    break
    if CFG.USE_EARLY_STOPPING and best_w is not None:
        model.set_weights(best_w)


# ======================================================================
# 9. EVALUATION
# ======================================================================
def _compute_metrics(y_te, y_hat):
    """Per-class F1/recall plus the head/mid/tail and worst-class summaries.

    Per-class accuracy == recall here, since each test class occupies a
    disjoint set of samples; worst-class accuracy is the min over classes.
    """
    per_class_f1 = f1_score(y_te, y_hat, average=None,
                            labels=list(range(CFG.NUM_CLASSES)),
                            zero_division=0)
    per_class_recall = recall_score(y_te, y_hat, average=None,
                                    labels=list(range(CFG.NUM_CLASSES)),
                                    zero_division=0)
    K = CFG.NUM_CLASSES
    h_end, m_end = K // 3, 2 * K // 3   # CIFAR-100: 33/66; CIFAR-10: 3/6
    head = per_class_f1[:h_end].mean()
    mid  = per_class_f1[h_end:m_end].mean()
    tail = per_class_f1[m_end:].mean()
    return dict(acc=(y_hat == y_te).mean(),
                bacc=balanced_accuracy_score(y_te, y_hat),
                head=head, mid=mid, tail=tail,
                head_tail_gap=head - tail,
                worst_f1=per_class_f1.min(),
                worst_acc=per_class_recall.min(),
                per_class_f1=per_class_f1,
                per_class_recall=per_class_recall)


def evaluate(model, x_te, y_te, name):
    chunks = []
    for i in range(0, len(x_te), CFG.BATCH_SIZE):
        chunk_logits, _ = model(x_te[i:i + CFG.BATCH_SIZE], training=False)
        chunks.append(chunk_logits)
    logits = tf.concat(chunks, axis=0)
    y_hat = tf.argmax(logits, axis=-1).numpy()
    m = _compute_metrics(y_te, y_hat)

    print(f"\n--- {name} ---")
    print(f"  top-1 accuracy   : {m['acc']:.4f}")
    print(f"  balanced accuracy: {m['bacc']:.4f}")
    print(f"  macro F1 (head)  : {m['head']:.3f}")
    print(f"  macro F1 (mid)   : {m['mid']:.3f}")
    print(f"  macro F1 (tail)  : {m['tail']:.3f}")
    print(f"  head-tail F1 gap : {m['head_tail_gap']:.3f}")
    print(f"  worst-class F1   : {m['worst_f1']:.3f}")
    print(f"  worst-class acc  : {m['worst_acc']:.3f}")
    return m


# ======================================================================
# 10. MAIN
# ======================================================================
def main(seed=None):
    if seed is not None:
        set_seeds(seed)
    print(f"\n{'=' * 60}\nSeed: {CFG.SEED}  arch={CFG.ARCHITECTURE}  "
          f"protocol={CFG.IMBALANCE_PROTOCOL}"
          + (f"  rho={CFG.IMBALANCE_RATIO}" if CFG.IMBALANCE_PROTOCOL == "cui2019" else "")
          + f"  dataset={CFG.DATASET}\n"
          + f"modules: cgan={CFG.USE_CGAN_AUG} rw={CFG.USE_REWEIGHTING} "
          + f"eq={CFG.USE_EQUALIZATION} smote={CFG.USE_FEATURE_SMOTE}\n"
          + f"sprint2.5: cgan_seed={CFG.CGAN_SEED} cgan_min_real={CFG.CGAN_MIN_REAL} "
          + f"warmup={CFG.WARMUP_EPOCHS} defer_rw={CFG.DEFER_REWEIGHTING_FRAC} "
          + f"defer_eq={CFG.DEFER_EQUALIZATION_FRAC} balanced_batch={CFG.USE_BALANCED_BATCH}"
          + f"\n{'=' * 60}")
    print(f"protocol: epochs={CFG.CLS_EPOCHS} patience={CFG.EARLY_STOP_PATIENCE} "
          f"early_stop={CFG.USE_EARLY_STOPPING} metric={CFG.EARLY_STOP_METRIC} "
          f"val_frac={CFG.VAL_FRACTION} warmup={CFG.WARMUP_EPOCHS} "
          f"defer_rw={CFG.DEFER_REWEIGHTING_FRAC} defer_eq={CFG.DEFER_EQUALIZATION_FRAC}")
    t_start = time.time()

    _out_override = os.environ.get("REBAL_OUT_DIR")
    exp_dir = _out_override if _out_override else os.path.join(EXPERIMENT_ROOT, f"seed_{CFG.SEED}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading and imbalancing CIFAR-100...")
    x_imb, y_imb, x_te, y_te = load_imbalanced_cifar100()
    print(f"  imbalanced: {x_imb.shape}, classes: {len(np.unique(y_imb))}")

    # ---- Module 1: detection/cleaning ----
    if CFG.USE_ECOD:
        print("\nECOD cleaning...")
        x_clean, y_clean = ecod_clean(x_imb, y_imb)
        print(f"  cleaned: {x_clean.shape}")
    else:
        x_clean, y_clean = x_imb, y_imb
    del x_imb, y_imb; gc.collect()   # [-1,1] raw copy no longer needed

    # Prepare classifier inputs in [0, 1]
    x_te_01 = x_te.astype("float32") / 255.0
    del x_te; gc.collect()
    x_clean_01 = (x_clean + 1.0) / 2.0

    # ---- Validation split ------------------------------------------
    # Taken here, BEFORE any synthesis, so no validation image can reach
    # training through the cGAN or SMOTE paths.
    if CFG.VAL_FRACTION > 0:
        tr_idx, val_idx = stratified_val_indices(y_clean)
        x_val, y_val = x_clean_01[val_idx], y_clean[val_idx]
        x_clean_01 = x_clean_01[tr_idx]
        x_clean    = x_clean[tr_idx]
        y_clean    = y_clean[tr_idx]
        print(f"  validation split: train={len(y_clean)} val={len(y_val)} "
              f"({len(np.unique(y_val))} classes contribute val samples)")
    else:
        x_val, y_val = x_te_01, y_te
        print("  WARNING: VAL_FRACTION=0 -> validating and early-stopping "
              "on the TEST set. Set CFG.VAL_FRACTION=0.1 for the protocol "
              "described in the paper.")

    # ---- Module 2: train a baseline feature extractor on imbalanced data ----
    print("\nTraining baseline feature extractor on imbalanced data...")
    feat_extractor = build_classifier(feature_dim=CFG.FEATURE_DIM)
    weights_uniform = np.ones(CFG.NUM_CLASSES, dtype="float32")
    train_classifier(feat_extractor, x_clean_01, y_clean,
                    x_val, y_val, weights_uniform)
    base_metrics = evaluate(feat_extractor, x_te_01, y_te, "BASELINE")
    feat_extractor.save_weights(os.path.join(ckpt_dir, "baseline.weights.h5"))

    # ---- Module 3a: cGAN training + sample bank ----
    if CFG.USE_CGAN_AUG:
        if CFG.SAMPLE_BANK_DIR:
            print(f"\nLoading samples from bank: {CFG.SAMPLE_BANK_DIR}")
            gan_x_01, gan_y = load_samples_from_bank(
                CFG.SAMPLE_BANK_DIR, y_train_real=y_clean)
        else:
            print("\nTraining cGAN on cleaned (imbalanced) data...")
            gen = train_cgan(x_clean, y_clean)
            print(f"\nGenerating up to {CFG.CGAN_AUG_PER_CLASS} cGAN "
                  f"samples per class...")
            gan_x, gan_y = sample_cgan(gen, y_train_real=y_clean)
            gan_x_01 = (gan_x + 1.0) / 2.0
            gan_x_01 = np.clip(gan_x_01, 0.0, 1.0)
    else:
        gan_x_01 = np.zeros((0, *CFG.IMG_SHAPE), dtype="float32")
        gan_y    = np.zeros((0,), dtype="int32")

    # ---- Module 3b: feature-space SMOTE  --------------------------------
    if CFG.USE_FEATURE_SMOTE:
        print("\nExtracting features for feature-space SMOTE...")
        n_clean = len(x_clean_01)
        feat_chunks = []
        for i in range(0, n_clean, CFG.BATCH_SIZE):
            _, chunk = feat_extractor(x_clean_01[i:min(i + CFG.BATCH_SIZE, n_clean)],
                                      training=False)
            feat_chunks.append(chunk.numpy())
        feats = np.concatenate(feat_chunks, axis=0)
        del feat_chunks; gc.collect()
        feats_bal, y_bal = smote_in_feature_space(feats, y_clean)
        del feats; gc.collect()
        print(f"  feature-balanced: {feats_bal.shape}")
        # Note: feature-space SMOTE samples retrain the linear head only.
        # The convolutional trunk stays fixed (Kang et al. 2020 cRT).
    else:
        print("\nPixel-space SMOTE (ablation)...")
        x_smote, y_smote = smote_in_pixel_space(x_clean_01, y_clean)
        feats_bal, y_bal = None, None

    # ---- Module 4: train the final classifier ---------------------------
    if CFG.USE_FEATURE_SMOTE:
        x_train_final = np.concatenate([x_clean_01, gan_x_01], axis=0)
        y_train_final = np.concatenate([y_clean, gan_y], axis=0)
    else:
        x_train_final = np.concatenate([x_smote, gan_x_01], axis=0)
        y_train_final = np.concatenate([y_smote, gan_y], axis=0)
        del x_smote; gc.collect()
    del x_clean_01, gan_x_01; gc.collect()   # subsumed into x_train_final
    print(f"\nFinal training set: {x_train_final.shape}")

    print("Training final classifier (full REBAL configuration)...")
    final_model = build_classifier(feature_dim=CFG.FEATURE_DIM)
    cw = effective_number_weights(y_train_final) if CFG.USE_REWEIGHTING \
        else np.ones(CFG.NUM_CLASSES, dtype="float32")
    train_classifier(final_model, x_train_final, y_train_final,
                    x_val, y_val, cw)

    # ---- cRT: retrain linear head on final_model features ---------------
    # IMPORTANT: features must be re-extracted from final_model BEFORE
    # x_train_final is freed.  The earlier feats_bal came from feat_extractor
    # (different weights) — using that head with final_model's trunk is a
    # feature-space mismatch and produces worse-than-random results.
    if CFG.USE_FEATURE_SMOTE:
        n_real = len(y_clean)   # real images occupy first n_real rows
        print("\nRe-extracting features from final model for cRT...")
        crt_feat_chunks = []
        for i in range(0, n_real, CFG.BATCH_SIZE):
            _, chunk = final_model(x_train_final[i:min(i + CFG.BATCH_SIZE, n_real)],
                                   training=False)
            crt_feat_chunks.append(chunk.numpy())
        feats_for_crt = np.concatenate(crt_feat_chunks, axis=0)
        del crt_feat_chunks; gc.collect()
        feats_bal_crt, y_bal_crt = smote_in_feature_space(feats_for_crt, y_clean)
        del feats_for_crt; gc.collect()
        print(f"  cRT feature-balanced: {feats_bal_crt.shape}")

    del x_train_final, y_train_final; gc.collect()
    del feats_bal, y_bal; gc.collect()   # baseline features no longer needed

    crt_metrics = None
    if CFG.USE_FEATURE_SMOTE:
        print("\nRetraining linear head on SMOTE-augmented features (cRT step)...")
        for v in final_model.layers:
            if v.name != "logits":
                v.trainable = False
        new_head = tf.keras.Sequential([
            layers.Input(shape=(CFG.FEATURE_DIM,)),
            layers.Dense(CFG.NUM_CLASSES,
                        kernel_regularizer=regularizers.l2(CFG.WEIGHT_DECAY)
                        if CFG.USE_WEIGHT_DECAY else None),
        ])
        new_head.compile(
            optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        new_head.fit(feats_bal_crt, y_bal_crt, batch_size=512, epochs=20, verbose=2)
        del feats_bal_crt, y_bal_crt; gc.collect()
        new_head.save_weights(os.path.join(ckpt_dir, "crt_head.weights.h5"))

        def predict_with_new_head(x):
            _, f = final_model(x, training=False)
            return new_head(f, training=False)

        # Evaluate the cRT version — batched to stay within memory
        crt_chunks = []
        for i in range(0, len(x_te_01), CFG.BATCH_SIZE):
            crt_chunks.append(predict_with_new_head(x_te_01[i:i + CFG.BATCH_SIZE]))
        logits = tf.concat(crt_chunks, axis=0)
        y_hat = tf.argmax(logits, axis=-1).numpy()
        crt_metrics = _compute_metrics(y_te, y_hat)

        print(f"\n--- FINAL + cRT (linear head on SMOTE features) ---")
        print(f"  top-1 accuracy   : {crt_metrics['acc']:.4f}")
        print(f"  balanced accuracy: {crt_metrics['bacc']:.4f}")
        print(f"  macro F1 (head)  : {crt_metrics['head']:.3f}")
        print(f"  macro F1 (mid)   : {crt_metrics['mid']:.3f}")
        print(f"  macro F1 (tail)  : {crt_metrics['tail']:.3f}")
        print(f"  head-tail F1 gap : {crt_metrics['head_tail_gap']:.3f}")
        print(f"  worst-class F1   : {crt_metrics['worst_f1']:.3f}")
        print(f"  worst-class acc  : {crt_metrics['worst_acc']:.3f}")

    final_metrics = evaluate(final_model, x_te_01, y_te,
                            "FINAL (full REBAL)")
    final_model.save_weights(os.path.join(ckpt_dir, "final.weights.h5"))

    print("\n=== SUMMARY ===")
    print(f"  baseline    top-1: {base_metrics['acc']:.4f}   "
        f"BAcc: {base_metrics['bacc']:.4f}   "
        f"tail F1: {base_metrics['tail']:.3f}   "
        f"head-tail gap: {base_metrics['head_tail_gap']:.3f}   "
        f"worst-acc: {base_metrics['worst_acc']:.3f}")
    print(f"  final REBAL top-1: {final_metrics['acc']:.4f}   "
        f"BAcc: {final_metrics['bacc']:.4f}   "
        f"tail F1: {final_metrics['tail']:.3f}   "
        f"head-tail gap: {final_metrics['head_tail_gap']:.3f}   "
        f"worst-acc: {final_metrics['worst_acc']:.3f}")
    if crt_metrics is not None:
        print(f"  REBAL + cRT top-1: {crt_metrics['acc']:.4f}   "
            f"BAcc: {crt_metrics['bacc']:.4f}   "
            f"tail F1: {crt_metrics['tail']:.3f}   "
            f"head-tail gap: {crt_metrics['head_tail_gap']:.3f}   "
            f"worst-acc: {crt_metrics['worst_acc']:.3f}")

    runtime_sec = time.time() - t_start
    try:
        if tf.config.list_physical_devices("GPU"):
            peak_mem_mb = float(
                tf.config.experimental.get_memory_info("GPU:0")["peak"]) / (1024 ** 2)
        else:
            import resource
            _p = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_mem_mb = float(_p) / (1024.0 if os.uname().sysname == "Linux"
                                       else 1024.0 ** 2)
    except Exception:
        peak_mem_mb = float("nan")
    print(f"\nTotal runtime: {runtime_sec / 60:.1f} min")

    def _scalarize(m):
        return {k: float(m[k]) for k in
                ("acc", "bacc", "head", "mid", "tail",
                "head_tail_gap", "worst_f1", "worst_acc")}

    results = {
        "seed": CFG.SEED,
        "runtime_sec": runtime_sec,
        "peak_mem_mb": peak_mem_mb,
        "val_fraction": CFG.VAL_FRACTION,
        "early_stop_metric": CFG.EARLY_STOP_METRIC,
        "use_early_stopping": CFG.USE_EARLY_STOPPING,
        "early_stop_patience": CFG.EARLY_STOP_PATIENCE,
        "cls_epochs": CFG.CLS_EPOCHS,
        "config": {
            "architecture": CFG.ARCHITECTURE,
            "imbalance_protocol": CFG.IMBALANCE_PROTOCOL,
            "rho": CFG.IMBALANCE_RATIO if CFG.IMBALANCE_PROTOCOL == "cui2019" else None,
            "dataset": CFG.DATASET,
            "use_cgan_aug": CFG.USE_CGAN_AUG,
            "use_reweighting": CFG.USE_REWEIGHTING,
            "use_equalization": CFG.USE_EQUALIZATION,
            "use_feature_smote": CFG.USE_FEATURE_SMOTE,
            "cgan_seed": CFG.CGAN_SEED,
            "cgan_min_real": CFG.CGAN_MIN_REAL,
            "warmup_epochs": CFG.WARMUP_EPOCHS,
            "defer_reweighting_frac": CFG.DEFER_REWEIGHTING_FRAC,
            "defer_equalization_frac": CFG.DEFER_EQUALIZATION_FRAC,
            "balanced_batch": CFG.USE_BALANCED_BATCH,
            "cls_lr": CFG.CLS_LR,
        },
        "baseline": _scalarize(base_metrics),
        "rebal": _scalarize(final_metrics),
    }
    if crt_metrics is not None:
        results["rebal_crt"] = _scalarize(crt_metrics)

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Per-class F1 / accuracy CSV for reproducibility (one row per class).
    csv_path = os.path.join(exp_dir, "per_class_f1.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["class_idx", "region",
                "baseline_f1", "baseline_acc",
                "rebal_f1", "rebal_acc"]
        if crt_metrics is not None:
            header += ["rebal_crt_f1", "rebal_crt_acc"]
        writer.writerow(header)
        for c in range(CFG.NUM_CLASSES):
            region = ("head" if c < CFG.NUM_CLASSES // 3 else
                      "mid" if c < 2 * CFG.NUM_CLASSES // 3 else "tail")
            row = [c, region,
                f"{base_metrics['per_class_f1'][c]:.6f}",
                f"{base_metrics['per_class_recall'][c]:.6f}",
                f"{final_metrics['per_class_f1'][c]:.6f}",
                f"{final_metrics['per_class_recall'][c]:.6f}"]
            if crt_metrics is not None:
                row += [f"{crt_metrics['per_class_f1'][c]:.6f}",
                        f"{crt_metrics['per_class_recall'][c]:.6f}"]
            writer.writerow(row)
    print(f"Wrote {csv_path}")
    print(f"Wrote {os.path.join(exp_dir, 'metrics.json')}")
    print(f"Wrote checkpoints to {ckpt_dir}/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the REBAL pipeline once.")
    parser.add_argument("--seed", type=int, default=CFG.SEED,
                        help="random seed for this run")
    parser.add_argument("--output", type=str, default=None,
                        help="path to write this run's metrics as JSON")
    # Sprint 2 flags
    parser.add_argument("--rho", type=float, default=None,
                        help="imbalance ratio ρ (enables Cui2019 protocol)")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["small", "resnet32"],
                        help="classifier architecture (default: small)")
    parser.add_argument("--protocol", type=str, default=None,
                        choices=["linear", "cui2019"],
                        help="imbalance protocol (default: linear)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar100", "cifar10"],
                        help="dataset (default: cifar100)")
    parser.add_argument("--rebal-off", action="store_true",
                        help="disable all REBAL modules (baseline cross-entropy only)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory for artifacts (overrides EXPERIMENT_ROOT)")
    # Sprint 2.5 flags
    parser.add_argument("--ablate", type=str, default=None,
                        choices=["cgan", "eq", "rw", "decoupled"],
                        help="ablate a single REBAL module (rho=100 ablation grid)")
    parser.add_argument("--cgan-seed", type=int, default=None,
                        help="independent seed for cGAN init/training (default: 1234)")
    parser.add_argument("--cgan-min-real", type=int, default=None,
                        help="skip cGAN samples for classes with fewer real images than this")
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="linear LR warmup epochs before cosine decay")
    parser.add_argument("--defer-frac", type=float, default=None,
                        help="fraction of CLS_EPOCHS before reweighting/equalization activate")
    parser.add_argument("--balanced-batch", action="store_true",
                        help="class-uniform batch sampling instead of natural distribution")
    args = parser.parse_args()

    # Apply Sprint 2 CLI overrides (ENV vars already applied at module load)
    if args.rho is not None:
        CFG.IMBALANCE_RATIO = args.rho
        CFG.IMBALANCE_PROTOCOL = "cui2019"
    if args.arch is not None:
        CFG.ARCHITECTURE = args.arch
        # Only when the env path did not already apply it, so individual
        # REBAL_* overrides are not clobbered. Explicit CLI flags below
        # (--warmup-epochs, --defer-frac, ...) are applied after this and
        # therefore still win.
        if not os.environ.get("REBAL_ARCH"):
            _apply_arch_preset()
            print(f"  applied {CFG.ARCHITECTURE} preset: "
                  f"epochs={CFG.CLS_EPOCHS} warmup={CFG.WARMUP_EPOCHS} "
                  f"defer_rw={CFG.DEFER_REWEIGHTING_FRAC} "
                  f"defer_eq={CFG.DEFER_EQUALIZATION_FRAC} "
                  f"patience={CFG.EARLY_STOP_PATIENCE}")
    if args.protocol is not None:
        CFG.IMBALANCE_PROTOCOL = args.protocol
    if args.dataset is not None:
        CFG.DATASET = args.dataset
        CFG.NUM_CLASSES = 10 if args.dataset == "cifar10" else 100
    if args.rebal_off:
        CFG.USE_CGAN_AUG      = False
        CFG.USE_FEATURE_SMOTE = False
        CFG.USE_REWEIGHTING   = False
        CFG.USE_EQUALIZATION  = False
    if args.out_dir is not None:
        os.environ["REBAL_OUT_DIR"] = args.out_dir
    if args.ablate is not None:
        if args.ablate == "cgan":
            CFG.USE_CGAN_AUG = False
        elif args.ablate == "eq":
            CFG.USE_EQUALIZATION = False
        elif args.ablate == "rw":
            CFG.USE_REWEIGHTING = False
        elif args.ablate == "decoupled":
            CFG.USE_CGAN_AUG = False
            CFG.USE_REWEIGHTING = False
            CFG.USE_EQUALIZATION = False
    if args.cgan_seed is not None:
        CFG.CGAN_SEED = args.cgan_seed
    if args.cgan_min_real is not None:
        CFG.CGAN_MIN_REAL = args.cgan_min_real
    if args.warmup_epochs is not None:
        CFG.WARMUP_EPOCHS = args.warmup_epochs
    if args.defer_frac is not None:
        CFG.DEFER_REWEIGHTING_FRAC = args.defer_frac
        CFG.DEFER_EQUALIZATION_FRAC = args.defer_frac
    if args.balanced_batch:
        CFG.USE_BALANCED_BATCH = True

    results = main(args.seed)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote metrics to {args.output}")
