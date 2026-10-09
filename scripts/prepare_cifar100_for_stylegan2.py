"""
Prepare CIFAR-100 imbalanced splits for StyleGAN2-ADA training.

Produces one PNG folder per (rho, class), matching the imbalance protocol used
in Sprint 2.5. StyleGAN2-ADA's dataset_tool.py can then convert each folder
into its native ZIP dataset format inside Colab.

Run locally on your Mac:
    python prepare_cifar100_for_stylegan2.py

Output structure (roughly ~200 MB total):
    stylegan2_data/
        rho_10/
            class_00/
                00000.png
                00001.png
                ...
            class_01/
                ...
            ...
        rho_50/
            ...
        rho_100/
            ...

Upload the stylegan2_data/ folder to your Google Drive under
`MyDrive/rebal/stylegan2_data/` before starting the Colab notebook.

Notes:
    * Uses the same imbalance protocol as Sprint 2.5 (Cui et al. exponential).
    * Applies ECOD cleaning per class before saving, matching framework.py.
    * Saves images at 32x32 in RGB PNG format (StyleGAN2-ADA's default).
    * Optional --skip_ecod flag if you want to include the outliers (useful
      for training a generator on unfiltered data, which the framework does
      in the ablation branch).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from pyod.models.ecod import ECOD

OUT_ROOT = Path("stylegan2_data")
RHOS = [10, 50, 100]
NUM_CLASSES = 100
IMG_SHAPE = (32, 32, 3)
SEED = 42


def cifar100_imbalanced(rho: float, rng: np.random.Generator):
    """Return imbalanced (x_uint8, y) at the given imbalance factor using
    the Cui et al. exponential decay protocol.
    """
    (x_tr, y_tr), _ = tf.keras.datasets.cifar100.load_data()
    y_tr = y_tr.flatten()

    mu = (1.0 / rho) ** (1.0 / (NUM_CLASSES - 1))
    keep_frac = np.power(mu, np.arange(NUM_CLASSES))

    idx_all = []
    for k in range(NUM_CLASSES):
        idx_k = np.where(y_tr == k)[0]
        n_keep = max(1, int(len(idx_k) * keep_frac[k]))
        idx_all.append(rng.choice(idx_k, size=n_keep, replace=False))
    idx_all = np.concatenate(idx_all)
    return x_tr[idx_all], y_tr[idx_all]


def ecod_clean(x_uint8: np.ndarray, y: np.ndarray, contamination: float = 0.05):
    """Per-class ECOD outlier removal, matching framework.py's Module 1."""
    keep_x, keep_y = [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        if len(idx) < 10:
            keep_x.append(x_uint8[idx]); keep_y.append(y[idx]); continue
        flat = x_uint8[idx].reshape(len(idx), -1).astype("float32")
        clf = ECOD(contamination=contamination, n_jobs=-1)
        clf.fit(flat)
        inliers = clf.labels_ == 0
        keep_x.append(x_uint8[idx][inliers]); keep_y.append(y[idx][inliers])
    return np.concatenate(keep_x), np.concatenate(keep_y)


def save_dataset(x_uint8: np.ndarray, y: np.ndarray, out_dir: Path):
    """Save images as PNGs organized in per-class subfolders."""
    out_dir.mkdir(parents=True, exist_ok=True)
    class_counter = {}
    for xi, yi in zip(x_uint8, y):
        yi = int(yi)
        class_dir = out_dir / f"class_{yi:02d}"
        class_dir.mkdir(exist_ok=True)
        n = class_counter.get(yi, 0)
        Image.fromarray(xi.astype("uint8"), mode="RGB").save(
            class_dir / f"{n:05d}.png", optimize=True)
        class_counter[yi] = n + 1


def summarize(x, y, label: str):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    print(f"  [{label}] total={len(x)}  min_class_count={counts.min()}  "
          f"max_class_count={counts.max()}  imbalance_factor={counts.max()/max(1,counts.min()):.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_ecod", action="store_true",
                   help="Skip ECOD cleaning (save the pre-cleaned dataset).")
    p.add_argument("--rhos", nargs="+", type=int, default=RHOS)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    for rho in args.rhos:
        print(f"\n=== rho = {rho} ===")
        x_imb, y_imb = cifar100_imbalanced(rho, rng)
        summarize(x_imb, y_imb, "imbalanced")

        if args.skip_ecod:
            x_final, y_final = x_imb, y_imb
        else:
            print("  running ECOD cleaning...")
            x_final, y_final = ecod_clean(x_imb, y_imb, contamination=0.05)
            summarize(x_final, y_final, "after ECOD")

        out_dir = OUT_ROOT / f"rho_{rho}"
        print(f"  saving PNGs to {out_dir} ...")
        save_dataset(x_final, y_final, out_dir)

    print("\nDone. Upload the following folder to Google Drive under")
    print(f"    MyDrive/rebal/{OUT_ROOT.name}/")
    print(f"before starting the Colab training notebook.")
    print(f"\nTotal size on disk:")
    total_bytes = sum(f.stat().st_size for f in OUT_ROOT.rglob('*.png'))
    print(f"    ~{total_bytes / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
