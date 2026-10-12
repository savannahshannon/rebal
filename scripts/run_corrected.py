#!/usr/bin/env python3
"""
Driver for the blocking revision work, under the corrected validation protocol.

Everything published so far was produced with the test set used as the
validation set and early stopping on it. None of it can be reported. This
script re-runs what the paper needs, in priority order, and is RESUMABLE:
a cell that already has results.json is skipped, so a dropped Colab session
costs you one run, not the whole sweep.

Tiers
-----
  1  main      REBAL grid at rho=100, 5 seeds. Nothing in the paper can be
               written until this exists. One run yields baseline, REBAL
               and REBAL+cRT together.                        ~5 runs
  2  eq        The no_eq ablation, 5 seeds. The paper's novelty claim rests
               on this single comparison (Reviewer 2 asked for it by name).
                                                              ~5 runs
  3  baselines LDAM, LDAM-DRW, class-balanced, logit adjustment and plain CE
               at rho=100, 5 seeds. Three of four reviewers named this.
               Cheaper per run: no cGAN, no SMOTE, no cRT.   ~25 runs
  4  ablations The remaining leave-one-out cells (smote, cgan, rw), 5 seeds.
               Good to have, not blocking.                   ~15 runs

Usage
-----
    python scripts/run_corrected.py --dry-run          # show the plan
    python scripts/run_corrected.py --tier 1           # just the main grid
    python scripts/run_corrected.py --tier 1 2 3       # the blocking set
    python scripts/run_corrected.py                    # all tiers, in order
    python scripts/run_corrected.py --tier 1 --seeds 0 # one cell

Re-running is always safe. Add --force to redo completed cells.
The sweep supplies its own protocol settings. External sample banks can be
selected with --sample-bank-dir and --per-class-fid-csv; the corresponding
REBAL_SAMPLE_BANK_DIR and REBAL_PER_CLASS_FID_CSV environment variables remain
supported. Other inherited REBAL_* overrides are ignored.

Colab
-----
Runs are ~2.85 h each and sessions drop. Practical pattern: start the
script, let it get through what it can, and re-run the identical command
after a reconnect -- it picks up where it stopped. `--max-runs N` caps a
single invocation so you can fit it inside a session you expect to keep.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework.py"
BASELINES = ROOT / "baselines.py"
RHO = 100
SEEDS = [0, 1, 2, 3, 4]
DATASET = "cifar100"  # overridden by --dataset
ARTIFACT_ENV_KEYS = ("REBAL_SAMPLE_BANK_DIR", "REBAL_PER_CLASS_FID_CSV")
ARTIFACT_OVERRIDES = {}


def _corrected_root():
    return (ROOT / "experiments" / "corrected" if DATASET == "cifar100"
            else ROOT / "experiments" / "corrected" / DATASET)


def _baselines_root():
    return (ROOT / "experiments" / "baselines" if DATASET == "cifar100"
            else ROOT / "experiments" / "baselines" / DATASET)

# The corrected protocol. These are the whole point of the re-run.
PROTOCOL = {
    "REBAL_ARCH": "resnet32",          # also applies the Sprint 2.5 preset
    "REBAL_RHO": str(RHO),
    "REBAL_VAL_FRACTION": "0.1",       # stratified holdout, taken pre-synthesis
    "REBAL_EARLY_STOP_METRIC": "bacc",
    "REBAL_USE_EARLY_STOPPING": "0",   # full cosine schedule, keep best
}

def _protocol():
    p = dict(PROTOCOL)
    p["REBAL_DATASET"] = DATASET
    p.update(ARTIFACT_OVERRIDES)
    return p

EST_HOURS = {"main": 2.85, "eq": 2.85, "ablation": 2.85,
             "ablation_cgan": 2.5, "baseline": 1.2}


# ----------------------------------------------------------------------
def cells_for(tiers, seeds):
    """Build the ordered work list. Each cell is a dict."""
    out = []
    cor = _corrected_root()
    if 1 in tiers:
        for s in seeds:
            out.append(dict(tier=1, kind="main", name=f"main/seed_{s}", seed=s,
                            out=cor / f"rho_{RHO}/main/seed_{s}",
                            est="main"))
    if 2 in tiers:
        for s in seeds:
            out.append(dict(tier=2, kind="ablate", ablate="eq",
                            name=f"ablate_eq/seed_{s}", seed=s,
                            out=cor / f"rho_{RHO}/ablate_eq/seed_{s}",
                            est="eq"))
    bas = _baselines_root()
    if 3 in tiers:
        for m in ["ce", "cb", "ldam", "ldam_drw", "la"]:
            for s in seeds:
                out.append(dict(tier=3, kind="baseline", method=m,
                                name=f"baseline_{m}/seed_{s}", seed=s,
                                out=bas / f"rho_{RHO}/{m}/seed_{s}",
                                est="baseline"))
    if 4 in tiers:
        for ab in ["feature_smote", "cgan", "rw"]:
            for s in seeds:
                out.append(dict(tier=4, kind="ablate", ablate=ab,
                                name=f"ablate_{ab}/seed_{s}", seed=s,
                                out=cor / f"rho_{RHO}/ablate_{ab}/seed_{s}",
                                est="ablation_cgan" if ab == "cgan" else "ablation"))
    return out


def protocol_of(j, kind):
    """Extract the protocol fields a result file should carry."""
    cfg = j.get("config", {})
    got = {"arch": cfg.get("architecture"),
           "rho": cfg.get("rho"),
           "epochs": j.get("cls_epochs"),
           "early_stop": j.get("use_early_stopping")}
    got["val"] = j.get("val_mode") if kind == "baseline" else j.get("val_fraction")
    return got


def protocol_mismatches(j, kind, dataset=None, rho=None):
    """List unproved or incompatible protocol fields, without loading TF.

    Older baseline records do not store the holdout fraction; ``holdout``
    verifies the source of validation data, but not the exact split identity.
    """
    dataset = DATASET if dataset is None else dataset
    rho = RHO if rho is None else rho
    if not isinstance(j, dict) or not isinstance(j.get("config"), dict):
        return ["config"]
    cfg = j["config"]
    want = {"architecture": "resnet32", "dataset": dataset,
            "imbalance_protocol": "cui2019"}
    bad = [key for key, value in want.items() if cfg.get(key) != value]
    try:
        if float(cfg.get("rho")) != float(rho):
            bad.append("rho")
    except (TypeError, ValueError):
        bad.append("rho")
    if j.get("cls_epochs") != 200:
        bad.append("cls_epochs")
    if j.get("use_early_stopping") is not False:
        bad.append("use_early_stopping")
    if kind == "baseline":
        if j.get("val_mode") != "holdout":
            bad.append("val_mode")
    else:
        if j.get("val_fraction") != 0.1:
            bad.append("val_fraction")
        if j.get("early_stop_metric") != "bacc":
            bad.append("early_stop_metric")
    return bad


def protocol_ok(j, kind):
    """True only if the run used the corrected protocol.

    Existence of a results file is NOT enough. We have twice been bitten by a
    run that completed happily under the wrong config, so a cell only counts
    as done if it can prove which protocol it used, and on the right dataset.
    """
    return not protocol_mismatches(j, kind)


def cell_mismatches(j, cell):
    """Ensure a result belongs to the seed and method at this output path."""
    if not isinstance(j, dict) or not isinstance(j.get("config"), dict):
        return ["config"]
    bad = []
    if j.get("seed") != cell["seed"]:
        bad.append("seed")
    if cell["kind"] == "baseline":
        if j.get("method") != cell["method"]:
            bad.append("method")
    else:
        ablation = cell.get("ablate")
        flags = {"cgan": "use_cgan_aug", "eq": "use_equalization",
                 "rw": "use_reweighting", "feature_smote": "use_feature_smote"}
        for module, key in flags.items():
            if j.get("config", {}).get(key) is not (module != ablation):
                bad.append(key)
    return bad


def done(cell, verbose=False):
    """Complete = a parseable results file that used the corrected protocol."""
    kind = "baseline" if cell["kind"] == "baseline" else "framework"
    for fn in (("results.json",) if kind == "baseline" else ("metrics.json",)):
        p = cell["out"] / fn
        if not p.exists():
            continue
        try:
            with p.open() as handle:
                j = json.load(handle)
        except Exception:
            if verbose:
                print(f"  ! {cell['name']}: unreadable {fn}, will re-run")
            return False
        if protocol_ok(j, kind) and not cell_mismatches(j, cell):
            return True
        if verbose:
            g = protocol_mismatches(j, kind)
            if isinstance(j, dict):
                g += cell_mismatches(j, cell)
            print(f"  ! {cell['name']}: results exist but under the WRONG "
                  f"protocol ({g}) -- will re-run and overwrite")
        return False
    return False


def build_cmd_env(cell):
    # A previous shell experiment must not silently turn the main arm into
    # an ablation or shorten its schedule. External bank paths are explicit
    # artifact inputs; preserve the two documented environment controls.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("REBAL_")}
    if cell["kind"] == "baseline":
        cmd = [sys.executable, "-u", str(BASELINES),
               "--method", cell["method"],
               "--seed", str(cell["seed"]),
               "--rho", str(RHO),
               "--arch", "resnet32",
               "--protocol", "cui2019",
               "--dataset", DATASET,
               "--epochs", "200", "--val-frac", "0.1",
               # Explicit even though baselines.py now defaults to holdout:
               # this sweep must never silently inherit the old protocol.
               "--val-mode", "holdout",
               "--out-dir", str(cell["out"])]
        return cmd, env

    env.update({key: os.environ[key] for key in ARTIFACT_ENV_KEYS
                if os.environ.get(key)})
    env.update(_protocol())
    env["REBAL_OUT_DIR"] = str(cell["out"])
    if cell["kind"] == "ablate":
        env["REBAL_ABLATE"] = cell["ablate"]
    cmd = [sys.executable, "-u", str(FRAMEWORK), "--seed", str(cell["seed"])]
    return cmd, env


def run_cell(cell, log_path):
    cmd, env = build_cmd_env(cell)
    cell["out"].mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write("# " + " ".join(cmd) + "\n")
        logf.write(f"# DATASET={DATASET}\n")
        for k in sorted(key for key in env if key.startswith("REBAL_")):
            logf.write(f"# {k}={env[k]}\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              env=env, cwd=str(ROOT))
    return proc.returncode, time.time() - t0


def fmt_h(h):
    return f"{h:.1f} h" if h >= 1 else f"{h * 60:.0f} min"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, nargs="+", default=[1, 2, 3, 4],
                    choices=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells that already have results")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="stop after this many runs in this invocation")
    ap.add_argument("--only", nargs="+", default=None, metavar="SUBSTR",
                    help="keep only cells whose name contains one of these "
                         "substrings, e.g. --only cgan  |  --only ldam_drw la")
    ap.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100",
                    help="cifar100 (default; writes to the original paths) or "
                         "cifar10 (writes to experiments/corrected/cifar10/...)")
    ap.add_argument("--sample-bank-dir", type=Path,
                    help="external cGAN sample directory; overrides REBAL_SAMPLE_BANK_DIR")
    ap.add_argument("--per-class-fid-csv", type=Path,
                    help="per-class FID gate file; overrides REBAL_PER_CLASS_FID_CSV")
    args = ap.parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        ap.error("--max-runs must be positive")

    global DATASET
    DATASET = args.dataset
    ARTIFACT_OVERRIDES.clear()
    for option, key, directory in ((args.sample_bank_dir, "REBAL_SAMPLE_BANK_DIR", True),
                                    (args.per_class_fid_csv, "REBAL_PER_CLASS_FID_CSV", False)):
        supplied = option or os.environ.get(key)
        if supplied:
            path = Path(supplied).expanduser().resolve()
            if not (path.is_dir() if directory else path.is_file()):
                ap.error(f"{key}: path does not exist or has the wrong type: {path}")
            ARTIFACT_OVERRIDES[key] = str(path)
    inherited = sorted(key for key in os.environ if key.startswith("REBAL_")
                       and key not in ARTIFACT_ENV_KEYS)
    if inherited:
        print("Sweep ignores inherited overrides: " + ", ".join(inherited))

    if not FRAMEWORK.exists():
        sys.exit(f"error: {FRAMEWORK} not found in the repository root.")
    src = FRAMEWORK.read_text()
    missing = [f for f in ("VAL_FRACTION", "_apply_arch_preset",
                           "REBAL_USE_EARLY_STOPPING") if f not in src]
    if missing:
        sys.exit("error: framework.py is missing " + ", ".join(missing) +
                 ".\nApply patch_framework.py and patch_arch_preset.py first.")
    if 3 in args.tier and not BASELINES.exists():
        sys.exit(f"error: {BASELINES} not found (needed for tier 3).")

    cells = cells_for(sorted(args.tier), args.seeds)
    if args.only:
        cells = [c for c in cells
                 if any(o in c["name"] for o in args.only)]
        if not cells:
            sys.exit(f"--only {args.only} matched no cells in tier(s) "
                     f"{sorted(args.tier)}. Run --dry-run without --only "
                     f"to see the available cell names.")
    todo = [c for c in cells if args.force or not done(c, verbose=True)]
    finished = len(cells) - len(todo)
    est = sum(EST_HOURS[c["est"]] for c in todo)

    print("=" * 66)
    print(" corrected-protocol sweep")
    print(f"   tiers      : {sorted(args.tier)}")
    print(f"   dataset    : {DATASET}")
    print(f"   seeds      : {args.seeds}")
    if args.only:
        print(f"   only       : {args.only}")
    print(f"   cells      : {len(cells)}  ({finished} already done, {len(todo)} to run)")
    print(f"   estimate   : {fmt_h(est)}")
    for k, v in PROTOCOL.items():
        print(f"   {k:<26} = {v}")
    print("=" * 66)

    if not todo:
        print("\nNothing to do -- every cell has results. Aggregate with:")
        print("  python scripts/aggregate_corrected.py")
        print("  python scripts/aggregate_baselines.py")
        return

    by_tier = {}
    for c in todo:
        by_tier.setdefault(c["tier"], []).append(c)
    for t in sorted(by_tier):
        hrs = sum(EST_HOURS[c["est"]] for c in by_tier[t])
        print(f"  tier {t}: {len(by_tier[t]):3d} runs  ~{fmt_h(hrs)}")
    print()

    if args.dry_run:
        print("--dry-run, nothing executed. Planned order:")
        for i, c in enumerate(todo, 1):
            print(f"  {i:3d}. {c['name']}")
        return

    t_start = time.time()
    ok = fail = 0
    for i, cell in enumerate(todo, 1):
        if args.max_runs and (ok + fail) >= args.max_runs:
            print(f"\n--max-runs {args.max_runs} reached; stopping cleanly.")
            break
        remaining = sum(EST_HOURS[c["est"]] for c in todo[i - 1:])
        print(f"[{i}/{len(todo)}] {cell['name']}   "
              f"(~{fmt_h(EST_HOURS[cell['est']])}, ~{fmt_h(remaining)} left)  "
              f"{time.strftime('%H:%M')}")
        log_path = cell["out"] / "run.log"
        cell["out"].mkdir(parents=True, exist_ok=True)
        rc, secs = run_cell(cell, log_path)
        if rc == 0 and done(cell):
            ok += 1
            print(f"        done in {fmt_h(secs / 3600)}")
        else:
            fail += 1
            print(f"        FAILED (exit {rc}) after {fmt_h(secs / 3600)}")
            print(f"        last lines of {log_path}:")
            try:
                tail = log_path.read_text().splitlines()[-12:]
                for ln in tail:
                    print("          " + ln)
            except Exception:
                pass
            print("        continuing to the next cell; re-run this script "
                  "to retry the failed one.")

    print()
    print(f"this invocation: {ok} ok, {fail} failed, "
          f"{fmt_h((time.time() - t_start) / 3600)} elapsed")
    left = [c for c in cells if not done(c)]  # protocol-validated
    if left:
        print(f"{len(left)} cells still outstanding -- re-run the same command "
              "to continue.")
    else:
        print("all cells complete. Aggregate with:")
        print("  python scripts/aggregate_corrected.py")
        print("  python scripts/aggregate_baselines.py")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
