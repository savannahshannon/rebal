#!/usr/bin/env python3
"""
Aggregate the corrected-protocol runs into paper-ready tables.

Reads  experiments/corrected/rho_<R>/{main,ablate_*}/seed_*/
         metrics.json      baseline / rebal / rebal_crt summary metrics
         per_class_f1.csv   per-class F1 and accuracy for all three arms

Writes (into --out-dir, default the repo root):
  table2_corrected.csv     Table II: mean +/- std over seeds for every arm
                           and metric, plus paired tests against the baseline
  redistribution.csv       the head/middle/tail decomposition and the
                           per-class win counts -- the paper's headline
  ablation_corrected.csv   leave-one-out deltas with paired tests, reported
                           per region rather than on top-1 alone
  protocol_check.csv       the protocol fields each run actually used, so a
                           mismatched cell is caught before it reaches a table

Usage
-----
    python aggregate_corrected.py
    python aggregate_corrected.py --rho 100 --out-dir tables/
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

from scripts.run_corrected import protocol_mismatches, cell_mismatches

ROOT = Path(__file__).resolve().parents[1]

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

ARMS = ["baseline", "rebal", "rebal_crt"]
METRICS = ["acc", "bacc", "head", "mid", "tail",
           "head_tail_gap", "worst_f1", "worst_acc"]
LOWER_IS_BETTER = {"head_tail_gap"}
REGIONS = ["head", "mid", "tail"]


def _write(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


def load_runs(root, rho):
    """-> {cell: {seed: metrics_json}}"""
    out = defaultdict(dict)
    pat = os.path.join(root, f"rho_{rho}", "*", "seed_*", "metrics.json")
    for p in sorted(glob.glob(pat)):
        try:
            with open(p) as handle:
                j = json.load(handle)
            seed = int(Path(p).parent.name.removeprefix("seed_"))
            if not isinstance(j, dict) or not isinstance(j.get("config"), dict):
                raise ValueError("result and config must be JSON objects")
        except Exception as e:
            print(f"  skipping unreadable {p}: {e}")
            continue
        cell = p.split(os.sep)[-3]
        out[cell][seed] = j
    return out


def load_per_class(root, rho, cell):
    """-> {seed: list of row dicts}"""
    out = {}
    pat = os.path.join(root, f"rho_{rho}", cell, "seed_*", "per_class_f1.csv")
    for p in sorted(glob.glob(pat)):
        seed = int(os.path.basename(os.path.dirname(p)).split("_")[1])
        with open(p) as handle:
            out[seed] = list(csv.DictReader(handle))
    return out


def paired(a, b):
    """a, b: equal-length arrays. Returns delta stats for a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return None
    d = a - b
    r = {"n": len(a), "delta": float(d.mean()),
         "delta_std": float(d.std(ddof=1)), "dz": "", "p": ""}
    if d.std(ddof=1) > 0:
        r["dz"] = float(d.mean() / d.std(ddof=1))
    if HAVE_SCIPY:
        try:
            r["p"] = float(stats.ttest_rel(a, b).pvalue)
        except Exception:
            pass
    return r


# ----------------------------------------------------------------------
def table2(runs, out_path):
    main = runs.get("main", {})
    if not main:
        print("  no main/ runs yet -- skipping table2")
        return
    seeds = sorted(main)
    rows = []
    for arm in ARMS:
        # Feature-SMOTE ablations can legitimately have no cRT result.
        arm_seeds = [s for s in seeds if arm in main[s]]
        if not arm_seeds:
            continue
        for m in METRICS:
            vals = [main[s][arm][m] for s in arm_seeds]
            row = {"arm": arm, "metric": m, "n_seeds": len(arm_seeds),
                   "mean": round(float(np.mean(vals)), 4),
                   "std": round(float(np.std(vals, ddof=0)), 4),
                   "delta_vs_baseline": "", "dz": "", "p": "",
                   "sig_at_05": ""}
            if arm != "baseline":
                t = paired(vals, [main[s]["baseline"][m] for s in arm_seeds])
                if t:
                    row["delta_vs_baseline"] = round(t["delta"], 4)
                    row["dz"] = round(t["dz"], 3) if t["dz"] != "" else ""
                    row["p"] = round(t["p"], 5) if t["p"] != "" else ""
                    row["sig_at_05"] = (t["p"] != "" and t["p"] < 0.05)
            rows.append(row)
    _write(out_path, rows,
           ["arm", "metric", "n_seeds", "mean", "std",
            "delta_vs_baseline", "dz", "p", "sig_at_05"])

    # The null result on top-1 needs stating explicitly, not inferring.
    t = paired([main[s]["rebal"]["acc"] for s in seeds],
               [main[s]["baseline"]["acc"] for s in seeds])
    if t:
        print(f"\n  top-1, REBAL - baseline: {t['delta']:+.4f} "
              f"(n={t['n']}, p={t['p'] if t['p']=='' else round(t['p'],4)})")
        if t["p"] != "" and t["p"] >= 0.05:
            print("  -> consistent with no aggregate effect. State it as a null "
                  "WITH the test and the seed spread; a null asserted from means "
                  "alone will not be believed.")
        elif t["p"] != "":
            print("  -> aggregate difference IS significant. The redistribution "
                  "framing in the draft assumes it is not; revise Section VI-A.")


def redistribution(root, rho, runs, out_path):
    main = runs.get("main", {})
    pc = load_per_class(root, rho, "main")
    if not main or not pc:
        print("  no per-class data yet -- skipping redistribution")
        return
    seeds = sorted(set(main) & set(pc))
    if not seeds:
        print("  no matching valid per-class seeds -- skipping redistribution")
        return
    for seed in seeds:
        if any(not any(row["region"] == region for row in pc[seed])
               for region in REGIONS):
            raise ValueError(f"seed {seed}: per-class data lacks head/mid/tail "
                             "regions; check dataset-specific class boundaries")
    rows = []
    for reg in REGIONS:
        b = [np.mean([float(r["baseline_f1"]) for r in pc[s] if r["region"] == reg])
             for s in seeds]
        v = [np.mean([float(r["rebal_f1"]) for r in pc[s] if r["region"] == reg])
             for s in seeds]
        t = paired(v, b)
        wins = [sum(1 for r in pc[s]
                    if r["region"] == reg
                    and float(r["rebal_f1"]) > float(r["baseline_f1"]))
                for s in seeds]
        n_cls = len([r for r in pc[seeds[0]] if r["region"] == reg])
        rows.append({
            "region": reg, "n_classes": n_cls, "n_seeds": len(seeds),
            "baseline_mean": round(float(np.mean(b)), 4),
            "rebal_mean": round(float(np.mean(v)), 4),
            "delta": round(t["delta"], 4) if t else "",
            "dz": round(t["dz"], 3) if t and t["dz"] != "" else "",
            "p": round(t["p"], 5) if t and t["p"] != "" else "",
            "classes_improved_mean": round(float(np.mean(wins)), 1),
        })
    # Classes the baseline cannot classify at all.
    z_b = [sum(1 for r in pc[s] if float(r["baseline_f1"]) == 0.0) for s in seeds]
    z_r = [sum(1 for r in pc[s] if float(r["rebal_f1"]) == 0.0) for s in seeds]
    rows.append({"region": "ALL (F1==0 count)", "n_classes": len(pc[seeds[0]]),
                 "n_seeds": len(seeds),
                 "baseline_mean": round(float(np.mean(z_b)), 2),
                 "rebal_mean": round(float(np.mean(z_r)), 2),
                 "delta": round(float(np.mean(z_r) - np.mean(z_b)), 2),
                 "dz": "", "p": "", "classes_improved_mean": ""})
    _write(out_path, rows,
           ["region", "n_classes", "n_seeds", "baseline_mean", "rebal_mean",
            "delta", "dz", "p", "classes_improved_mean"])

    d = {r["region"]: r["delta"] for r in rows if r["region"] in REGIONS}
    if all(k in d and d[k] != "" for k in REGIONS):
        print(f"\n  redistribution: head {d['head']:+.4f}   "
              f"mid {d['mid']:+.4f}   tail {d['tail']:+.4f}")
        if d["mid"] < 0 <= d["tail"] and d["head"] >= 0:
            print("  -> middle absorbs the cost, head does not. This is the "
                  "draft's headline claim and it holds across seeds.")
        else:
            print("  -> pattern differs from the single-seed run the draft is "
                  "written around. Section VI-A needs rewriting to match.")


def ablations(runs, out_path):
    main = runs.get("main", {})
    cells = {k: v for k, v in runs.items() if k.startswith("ablate_")}
    if not main or not cells:
        print("  no ablation runs yet -- skipping ablation table")
        return
    rows = []
    for cell in sorted(cells):
        seeds = sorted(set(main) & set(cells[cell]))
        if len(seeds) < 2:
            continue
        for m in METRICS:
            full = [main[s]["rebal"][m] for s in seeds]
            abl = [cells[cell][s]["rebal"][m] for s in seeds]
            t = paired(abl, full)
            if not t:
                continue
            # Positive "hurt" means removing the module made things worse.
            hurt = (t["delta"] < 0) if m not in LOWER_IS_BETTER else (t["delta"] > 0)
            rows.append({
                "ablation": cell.replace("ablate_", ""), "metric": m,
                "n_seeds": len(seeds),
                "full_mean": round(float(np.mean(full)), 4),
                "ablated_mean": round(float(np.mean(abl)), 4),
                "delta_removed_minus_full": round(t["delta"], 4),
                "dz": round(t["dz"], 3) if t["dz"] != "" else "",
                "p": round(t["p"], 5) if t["p"] != "" else "",
                "removing_hurts": hurt,
            })
    _write(out_path, rows,
           ["ablation", "metric", "n_seeds", "full_mean", "ablated_mean",
            "delta_removed_minus_full", "dz", "p", "removing_hurts"])

    eq = [r for r in rows if r["ablation"] == "eq" and r["metric"] == "tail"]
    if eq:
        r = eq[0]
        print(f"\n  removing R_eq changes tail F1 by "
              f"{r['delta_removed_minus_full']:+.4f} (p={r['p']}).")
        print("  This single row is what the paper's novelty claim rests on. "
              "Report it per region, not on top-1.")


def protocol_check(runs, out_path, dataset="cifar100", rho=100):
    """Write an audit and return only runs proved compatible for aggregation."""
    rows, bad = [], 0
    valid = defaultdict(dict)
    for cell in sorted(runs):
        for seed in sorted(runs[cell]):
            j = runs[cell][seed]
            row = {"cell": cell, "seed": seed,
                   "arch": j.get("config", {}).get("architecture", "?"),
                   "rho": j.get("config", {}).get("rho", "?"),
                   "dataset": j.get("config", {}).get("dataset", "?")}
            mism = protocol_mismatches(j, "framework", dataset, rho)
            expected = {"kind": "main", "seed": seed}
            if cell.startswith("ablate_"):
                expected.update(kind="ablate", ablate=cell[len("ablate_"):])
                if expected["ablate"] not in ("cgan", "eq", "rw", "feature_smote"):
                    mism.append("unknown_ablation")
            elif cell != "main":
                mism.append("unknown_cell")
            mism.extend(cell_mismatches(j, expected))
            for arm in ("baseline", "rebal") + (("rebal_crt",) if "rebal_crt" in j else ()):
                if not isinstance(j.get(arm), dict) or any(
                        not isinstance(j[arm].get(metric), (int, float))
                        or not np.isfinite(j[arm][metric]) for metric in METRICS):
                    mism.append(arm + "_metrics")
            for k in ("val_fraction", "use_early_stopping", "early_stop_metric", "cls_epochs"):
                got = j.get(k, "MISSING")
                row[k] = got
            row["mismatch"] = ";".join(mism)
            if mism:
                bad += 1
            else:
                valid[cell][seed] = j
            rows.append(row)
    _write(out_path, rows,
           ["cell", "seed", "arch", "rho", "dataset", "val_fraction", "use_early_stopping",
            "early_stop_metric", "cls_epochs", "mismatch"])
    if bad:
        print(f"\n  WARNING: {bad} run(s) did not use the corrected protocol. "
              "They are excluded from all result tables; see the mismatch column.")
    else:
        print("\n  protocol check: all runs used the corrected protocol.")
    return valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="default: experiments/corrected for cifar100, "
                         "experiments/corrected/cifar10 for cifar10")
    ap.add_argument("--rho", type=int, default=100)
    ap.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100")
    ap.add_argument("--out-dir", default=str(ROOT))
    args = ap.parse_args()

    if args.root is None:
        args.root = str(ROOT / "experiments" / "corrected")
        if args.dataset != "cifar100":
            args.root = os.path.join(args.root, args.dataset)
    suffix = "" if args.dataset == "cifar100" else f"_{args.dataset}"

    if not HAVE_SCIPY:
        print("NOTE: scipy not found -- effect sizes computed, p-values blank. "
              "pip install scipy")

    runs = load_runs(args.root, args.rho)
    if not runs:
        print(f"No runs under {args.root}/rho_{args.rho}/. "
              "Run: python run_corrected.py --tier 1")
        return
    for cell in sorted(runs):
        print(f"  {cell}: {len(runs[cell])} seed(s)")

    od = args.out_dir
    runs = protocol_check(runs, os.path.join(od, f"protocol_check{suffix}.csv"),
                          dataset=args.dataset, rho=args.rho)
    if not runs:
        raise SystemExit("No compatible corrected-protocol runs remain.")
    table2(runs, os.path.join(od, f"table2_corrected{suffix}.csv"))
    redistribution(args.root, args.rho, runs,
                   os.path.join(od, f"redistribution{suffix}.csv"))
    ablations(runs, os.path.join(od, f"ablation_corrected{suffix}.csv"))
    print("\ndone.")


if __name__ == "__main__":
    main()
