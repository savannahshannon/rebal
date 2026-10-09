#!/usr/bin/env python3
"""
Aggregate baseline runs, fold in the existing REBAL results, and emit
paper-ready tables plus paired significance tests.

Outputs (into --out-dir, default the repo root):
  baselines_summary.csv        per-method mean/std for every metric and rho
  baseline_vs_rebal.csv        REBAL minus each baseline, with paired t-test
                               and Wilcoxon p-values across seeds
  compute_cost.csv             train/infer seconds, peak memory, params
  significance_existing.csv    paired tests on the sprint2 and sprint4 results
                               that already exist (no new runs needed)

Usage
-----
    python aggregate_baselines.py
    python aggregate_baselines.py --rho 100
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

METRICS = ["acc", "bacc", "head", "mid", "tail",
           "head_tail_gap", "worst_f1", "worst_acc"]

# Lower is better for these; everything else is higher-is-better.
LOWER_IS_BETTER = {"head_tail_gap"}


# ----------------------------------------------------------------------
def load_baselines(root="experiments/baselines"):
    """-> {(rho, method): {seed: metrics}}, {(rho, method): {seed: cost}}"""
    metrics = defaultdict(dict)
    costs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "rho_*", "*", "seed_*",
                                              "results.json"))):
        with open(path) as f:
            r = json.load(f)
        rho = int(float(r["config"]["rho"]))
        seed = int(r["seed"])
        metrics[(rho, r["method"])][seed] = r["metrics"]
        costs[(rho, r["method"])][seed] = r.get("cost", {})
        if "la_posthoc" in r:
            metrics[(rho, "la_posthoc")][seed] = r["la_posthoc"]
            costs[(rho, "la_posthoc")][seed] = r.get("cost", {})
    return metrics, costs


def load_sprint2(root="experiments/sprint2_grid"):
    """Parse the existing per-seed REBAL tables into the same shape."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "table_rho*.csv"))):
        rho = int(os.path.basename(path).replace("table_rho", "").replace(".csv", ""))
        with open(path) as f:
            for row in csv.DictReader(f):
                cfg, metric = row["config"], row["metric"]
                # sprint2 uses slightly different metric names
                name = {"top1_acc": "acc", "balanced_acc": "bacc",
                        "head_f1": "head", "mid_f1": "mid", "tail_f1": "tail",
                        "head_tail_f1_gap": "head_tail_gap",
                        "worst_class_f1": "worst_f1",
                        "worst_class_acc": "worst_acc"}.get(metric)
                if name is None:
                    continue
                for key, val in row.items():
                    if not key.startswith("seed_") or val in ("", None):
                        continue
                    seed = int(key.split("_")[1])
                    out[(rho, cfg)].setdefault(seed, {})[name] = float(val)
    return out


def mean_std(per_seed, metric):
    vals = [m[metric] for m in per_seed.values() if metric in m]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)


def paired_tests(a_per_seed, b_per_seed, metric):
    """Paired comparison of a - b over the seeds both arms share."""
    seeds = sorted(set(a_per_seed) & set(b_per_seed))
    a = np.array([a_per_seed[s][metric] for s in seeds
                  if metric in a_per_seed[s] and metric in b_per_seed[s]])
    b = np.array([b_per_seed[s][metric] for s in seeds
                  if metric in a_per_seed[s] and metric in b_per_seed[s]])
    if len(a) < 2:
        return None
    d = a - b
    res = {"n_pairs": len(a), "delta_mean": float(d.mean()),
           "delta_std": float(d.std(ddof=1)) if len(a) > 1 else 0.0,
           "t_p": "", "wilcoxon_p": "", "cohens_dz": ""}
    if d.std(ddof=1) > 0:
        res["cohens_dz"] = float(d.mean() / d.std(ddof=1))
    if HAVE_SCIPY and len(a) >= 2:
        try:
            res["t_p"] = float(stats.ttest_rel(a, b).pvalue)
        except Exception:
            pass
        try:
            if np.any(d != 0):
                res["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except Exception:
            pass
    return res


# ----------------------------------------------------------------------
def write_summary(metrics, out_path):
    rows = []
    for (rho, method) in sorted(metrics):
        for m in METRICS:
            mu, sd, n = mean_std(metrics[(rho, method)], m)
            if mu is None:
                continue
            rows.append({"rho": rho, "method": method, "metric": m,
                         "n_seeds": n, "mean": round(mu, 4),
                         "std": round(sd, 4)})
    _write(out_path, rows, ["rho", "method", "metric", "n_seeds", "mean", "std"])
    return rows


def write_comparison(metrics, rebal, out_path):
    """REBAL (and REBAL+cRT) against each baseline, paired across seeds."""
    rows = []
    rhos = sorted({r for (r, _) in metrics} & {r for (r, _) in rebal})
    for rho in rhos:
        for rebal_cfg in ("rebal", "rebal_crt"):
            ref = rebal.get((rho, rebal_cfg))
            if not ref:
                continue
            for (r2, method) in sorted(metrics):
                if r2 != rho:
                    continue
                for m in METRICS:
                    t = paired_tests(ref, metrics[(rho, method)], m)
                    if t is None:
                        continue
                    better = (t["delta_mean"] < 0) if m in LOWER_IS_BETTER \
                        else (t["delta_mean"] > 0)
                    rows.append({
                        "rho": rho, "rebal_config": rebal_cfg,
                        "baseline": method, "metric": m,
                        "n_pairs": t["n_pairs"],
                        "delta_mean": round(t["delta_mean"], 4),
                        "delta_std": round(t["delta_std"], 4),
                        "cohens_dz": round(t["cohens_dz"], 3)
                        if t["cohens_dz"] != "" else "",
                        "t_p": round(t["t_p"], 5) if t["t_p"] != "" else "",
                        "wilcoxon_p": round(t["wilcoxon_p"], 5)
                        if t["wilcoxon_p"] != "" else "",
                        "rebal_better": better,
                    })
    _write(out_path, rows,
           ["rho", "rebal_config", "baseline", "metric", "n_pairs",
            "delta_mean", "delta_std", "cohens_dz", "t_p", "wilcoxon_p",
            "rebal_better"])
    return rows


def write_costs(costs, out_path):
    rows = []
    for (rho, method) in sorted(costs):
        per = costs[(rho, method)]
        def agg(field):
            vals = [c[field] for c in per.values()
                    if field in c and c[field] == c[field]]
            return (round(float(np.mean(vals)), 2),
                    round(float(np.std(vals)), 2)) if vals else ("", "")
        tm, ts = agg("train_sec")
        im, isd = agg("infer_sec")
        mm, ms = agg("peak_mem_mb")
        params = [c.get("trainable_params") for c in per.values()
                  if c.get("trainable_params")]
        rows.append({
            "rho": rho, "method": method, "n_seeds": len(per),
            "train_sec_mean": tm, "train_sec_std": ts,
            "train_min_mean": round(tm / 60, 2) if tm != "" else "",
            "infer_sec_mean": im, "infer_sec_std": isd,
            "peak_mem_mb_mean": mm, "peak_mem_mb_std": ms,
            "trainable_params": int(np.median(params)) if params else "",
        })
    _write(out_path, rows,
           ["rho", "method", "n_seeds", "train_sec_mean", "train_sec_std",
            "train_min_mean", "infer_sec_mean", "infer_sec_std",
            "peak_mem_mb_mean", "peak_mem_mb_std", "trainable_params"])
    return rows


def write_existing_significance(rebal, out_path):
    """Paired tests on results that already exist. No new runs required.

    This alone answers Reviewer 4's significance request for the sprint2 grid.
    """
    rows = []
    pairs = [("rebal", "baseline"), ("rebal_crt", "baseline"),
             ("rebal_crt", "rebal")]
    for rho in sorted({r for (r, _) in rebal}):
        for a, b in pairs:
            pa, pb = rebal.get((rho, a)), rebal.get((rho, b))
            if not pa or not pb:
                continue
            for m in METRICS:
                t = paired_tests(pa, pb, m)
                if t is None:
                    continue
                rows.append({
                    "rho": rho, "comparison": f"{a} - {b}", "metric": m,
                    "n_pairs": t["n_pairs"],
                    "delta_mean": round(t["delta_mean"], 4),
                    "delta_std": round(t["delta_std"], 4),
                    "cohens_dz": round(t["cohens_dz"], 3)
                    if t["cohens_dz"] != "" else "",
                    "t_p": round(t["t_p"], 5) if t["t_p"] != "" else "",
                    "wilcoxon_p": round(t["wilcoxon_p"], 5)
                    if t["wilcoxon_p"] != "" else "",
                    "significant_at_05": (t["t_p"] != "" and t["t_p"] < 0.05),
                })
    _write(out_path, rows,
           ["rho", "comparison", "metric", "n_pairs", "delta_mean",
            "delta_std", "cohens_dz", "t_p", "wilcoxon_p",
            "significant_at_05"])
    return rows


def _write(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baselines-root", default=None,
                   help="default: experiments/baselines for cifar100, "
                        "experiments/baselines/cifar10 for cifar10")
    p.add_argument("--sprint2-root", default="experiments/sprint2_grid")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100")
    p.add_argument("--rho", type=int, default=None,
                   help="restrict every table to one imbalance factor")
    args = p.parse_args()

    if args.baselines_root is None:
        args.baselines_root = ("experiments/baselines" if args.dataset == "cifar100"
                               else f"experiments/baselines/{args.dataset}")
    suffix = "" if args.dataset == "cifar100" else f"_{args.dataset}"

    if not HAVE_SCIPY:
        print("NOTE: scipy not found -- effect sizes still computed, "
              "p-values left blank. pip install scipy")

    bl_metrics, bl_costs = load_baselines(args.baselines_root)
    rebal = load_sprint2(args.sprint2_root) if args.dataset == "cifar100" else {}

    if args.rho is not None:
        keep = lambda d: {k: v for k, v in d.items() if k[0] == args.rho}
        bl_metrics, bl_costs, rebal = keep(bl_metrics), keep(bl_costs), keep(rebal)

    print(f"\nloaded {len(bl_metrics)} baseline arms, {len(rebal)} REBAL arms")

    od = args.out_dir
    # This one works even with zero baseline runs -- run it today.
    if rebal:
        write_existing_significance(
            rebal, os.path.join(od, f"significance_existing{suffix}.csv"))

    if not bl_metrics:
        print("\nNo baseline runs found yet. significance_existing.csv is "
              "still written from the sprint2 results.\nRun ./run_baselines.sh "
              "to populate the rest.")
        return

    write_summary(bl_metrics, os.path.join(od, f"baselines_summary{suffix}.csv"))
    write_comparison(bl_metrics, rebal, os.path.join(od, f"baseline_vs_rebal{suffix}.csv"))
    write_costs(bl_costs, os.path.join(od, f"compute_cost{suffix}.csv"))
    print("\ndone.")


if __name__ == "__main__":
    main()
