"""Small regression checks for experiment routing and provenance (no TF)."""
import ast
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_corrected as runner
from scripts import aggregate_corrected as corrected
from scripts import aggregate_baselines as baselines


def record(seed=0):
    scores = {metric: 0.5 for metric in corrected.METRICS}
    return {
        "seed": seed, "val_fraction": 0.1, "early_stop_metric": "bacc",
        "use_early_stopping": False, "cls_epochs": 200,
        "config": {"architecture": "resnet32", "dataset": "cifar100",
                   "imbalance_protocol": "cui2019", "rho": 100,
                   "use_cgan_aug": True, "use_equalization": True,
                   "use_reweighting": True, "use_feature_smote": True},
        "baseline": dict(scores), "rebal": dict(scores),
        "rebal_crt": dict(scores),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RunnerTests(unittest.TestCase):
    def test_paths_and_dry_run_work_outside_repository(self):
        self.assertEqual(runner.ROOT, ROOT)
        self.assertTrue(runner.FRAMEWORK.is_file())
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_corrected.py"),
                 "--dry-run", "--tier", "1", "3", "--seeds", "0"],
                cwd=tmp, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cells      : 6", result.stdout)
        self.assertIn("nothing executed", result.stdout)

    def test_output_roots_separate_datasets(self):
        with patch.object(runner, "DATASET", "cifar10"):
            cell = runner.cells_for([1], [4])[0]
        self.assertEqual(cell["out"], ROOT / "experiments/corrected/cifar10/rho_100/main/seed_4")

    def test_inherited_overrides_cannot_change_main_or_baselines(self):
        hostile = {"REBAL_ABLATE": "eq", "REBAL_EPOCHS_OVERRIDE": "1",
                   "REBAL_BASELINE_ONLY": "1"}
        with patch.dict(os.environ, hostile):
            main, baseline = runner.cells_for([1, 3], [0])[:2]
            _, env = runner.build_cmd_env(main)
            cmd, baseline_env = runner.build_cmd_env(baseline)
        for key in hostile:
            self.assertNotIn(key, env)
        self.assertFalse(any(key.startswith("REBAL_") for key in baseline_env))
        self.assertEqual(env["REBAL_VAL_FRACTION"], "0.1")
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "200")
        self.assertEqual(cmd[cmd.index("--val-frac") + 1], "0.1")

    def test_external_bank_artifacts_remain_supported(self):
        with patch.dict(os.environ, {"REBAL_SAMPLE_BANK_DIR": "/bank",
                                     "REBAL_PER_CLASS_FID_CSV": "/gate.csv"}):
            _, env = runner.build_cmd_env(runner.cells_for([1], [0])[0])
        self.assertEqual(env["REBAL_SAMPLE_BANK_DIR"], "/bank")
        self.assertEqual(env["REBAL_PER_CLASS_FID_CSV"], "/gate.csv")
        with patch.dict(runner.ARTIFACT_OVERRIDES, {"REBAL_SAMPLE_BANK_DIR": "/cli-bank"}):
            _, env = runner.build_cmd_env(runner.cells_for([1], [0])[0])
        self.assertEqual(env["REBAL_SAMPLE_BANK_DIR"], "/cli-bank")

    def test_ablation_is_explicit_and_not_inherited(self):
        cell = runner.cells_for([2], [0])[0]
        _, env = runner.build_cmd_env(cell)
        self.assertEqual(env["REBAL_ABLATE"], "eq")

    def test_protocol_rejects_missing_and_malformed_fields(self):
        good = record()
        self.assertTrue(runner.protocol_ok(good, "framework"))
        for key, value in [("rho", "broken"), ("rho", 50),
                           ("dataset", "cifar10"), ("imbalance_protocol", "linear")]:
            bad = copy.deepcopy(good)
            bad["config"][key] = value
            self.assertFalse(runner.protocol_ok(bad, "framework"), key)
        good["early_stop_metric"] = "acc"
        self.assertFalse(runner.protocol_ok(good, "framework"))
        self.assertFalse(runner.protocol_ok([], "framework"))

    def test_resume_requires_correct_seed_and_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = {"kind": "main", "seed": 0, "out": Path(tmp), "name": "main/seed_0"}
            path = Path(tmp) / "metrics.json"
            result = record()
            write_json(path, result)
            self.assertTrue(runner.done(cell))
            result["seed"] = 1
            write_json(path, result)
            self.assertFalse(runner.done(cell))
            result["seed"] = 0
            result["config"]["use_equalization"] = False
            write_json(path, result)
            self.assertFalse(runner.done(cell))
            cell.update(kind="ablate", ablate="eq")
            self.assertTrue(runner.done(cell))

    def test_run_cell_uses_repository_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = runner.cells_for([1], [0])[0]
            cell["out"] = Path(tmp)
            with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run:
                code, _ = runner.run_cell(cell, Path(tmp) / "run.log")
        self.assertEqual(code, 7)
        self.assertEqual(run.call_args.kwargs["cwd"], str(ROOT))


class AggregationTests(unittest.TestCase):
    def test_aggregator_clis_create_output_directories_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corr_root, base_root = tmp / "corrected", tmp / "baselines"
            for seed in (0, 1):
                result = record(seed)
                result["rebal"] = {key: 0.55 + 0.05 * seed for key in corrected.METRICS}
                result["rebal_crt"] = {key: 0.56 + 0.04 * seed for key in corrected.METRICS}
                run_dir = corr_root / f"rho_100/main/seed_{seed}"
                write_json(run_dir / "metrics.json", result)
                (run_dir / "per_class_f1.csv").write_text(
                    "class_idx,region,baseline_f1,rebal_f1\n"
                    f"0,head,0.5,{0.55 + 0.05 * seed}\n"
                    f"1,mid,0.5,{0.55 + 0.05 * seed}\n"
                    f"2,tail,0.5,{0.55 + 0.05 * seed}\n")
                result.update(method="ce", val_mode="holdout", metrics=result["baseline"])
                write_json(base_root / f"rho_100/ce/seed_{seed}/results.json", result)
            commands = [
                ("aggregate_corrected.py", ["--root", str(corr_root)]),
                ("aggregate_baselines.py", ["--corrected-root", str(corr_root),
                                             "--baselines-root", str(base_root)]),
            ]
            out = tmp / "new" / "tables"
            for script, args in commands:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), *args,
                     "--out-dir", str(out)], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ("protocol_check.csv", "table2_corrected.csv", "redistribution.csv",
                         "baselines_summary.csv", "baseline_vs_rebal.csv", "compute_cost.csv"):
                self.assertTrue((out / name).is_file(), name)
            self.assertFalse((out / "significance_legacy_sprint2.csv").exists())

    def test_protocol_audit_excludes_mismatch_and_keeps_evidence(self):
        good, wrong_dataset, wrong_seed = record(), record(1), record(99)
        wrong_dataset["config"]["dataset"] = "cifar10"
        runs = {"main": {0: good, 1: wrong_dataset, 2: wrong_seed}}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "audit.csv"
            valid = corrected.protocol_check(runs, path)
            with path.open() as handle:
                audit = list(csv.DictReader(handle))
        self.assertEqual(set(valid["main"]), {0})
        self.assertIn("dataset", audit[1]["mismatch"])
        self.assertIn("seed", audit[2]["mismatch"])

    def test_loader_uses_directory_seed_to_detect_misplaced_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "rho_100/main/seed_0/metrics.json", record(99))
            runs = corrected.load_runs(tmp, 100)
        self.assertEqual(set(runs["main"]), {0})
        self.assertEqual(runs["main"][0]["seed"], 99)

    def test_baseline_loader_excludes_test_selected_and_misidentified(self):
        result = record()
        result.update(method="ce", val_mode="holdout", metrics=result["baseline"])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "rho_100/ce/seed_0/results.json"
            write_json(path, result)
            metrics, _ = baselines.load_baselines(tmp)
            self.assertIn((100, "ce"), metrics)
            result["val_mode"] = "match"
            write_json(path, result)
            self.assertFalse(baselines.load_baselines(tmp)[0])
            result.update(val_mode="holdout", method="ldam")
            write_json(path, result)
            self.assertFalse(baselines.load_baselines(tmp)[0])

    def test_comparison_source_is_corrected_and_dataset_checked(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            write_json(Path(tmp) / "rho_100/main/seed_0/metrics.json", record())
            self.assertIn((100, "rebal"), baselines.load_corrected(tmp))
            self.assertFalse(baselines.load_corrected(tmp, dataset="cifar10"))

    def test_paired_comparison_uses_only_shared_seed_metrics(self):
        a = {0: {"acc": 0.4}, 1: {"acc": 0.7}, 2: {"acc": 99}}
        b = {0: {"acc": 0.2}, 1: {"acc": 0.6}, 3: {"acc": -99}}
        result = baselines.paired_tests(a, b, "acc")
        self.assertEqual(result["n_pairs"], 2)
        self.assertAlmostEqual(result["delta_mean"], 0.15)

    def test_missing_per_class_regions_raise_instead_of_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rho_100/main/seed_0/per_class_f1.csv"
            path.parent.mkdir(parents=True)
            path.write_text("class_idx,region,baseline_f1,rebal_f1\n0,head,0.1,0.2\n")
            with self.assertRaisesRegex(ValueError, "lacks head/mid/tail"):
                corrected.redistribution(tmp, 100, {"main": {0: record()}}, Path(tmp) / "out.csv")

    def test_framework_csv_boundaries_match_summary_regions_without_import(self):
        tree = ast.parse((ROOT / "framework.py").read_text())
        region = next(node.value for node in ast.walk(tree)
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == "region"
                          for target in node.targets))
        expression = compile(ast.Expression(region), "framework-region", "eval")
        for classes in (10, 100):
            actual = [eval(expression, {"CFG": SimpleNamespace(NUM_CLASSES=classes), "c": c})
                      for c in range(classes)]
            self.assertEqual(actual.count("head"), classes // 3)
            self.assertEqual(actual.count("mid"), 2 * classes // 3 - classes // 3)
            self.assertEqual(actual.count("tail"), classes - 2 * classes // 3)


if __name__ == "__main__":
    unittest.main()
