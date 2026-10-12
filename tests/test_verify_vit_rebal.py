"""Independent recomputation, calibrated aliases, and falsified-evidence checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
fixture_module=module("vit_verifier_fixture",HERE/"test_vit_rebal.py")
runner=fixture_module.v
checker=module("vit_independent_checker",HERE.parent/"scripts/verify_vit_rebal.py")


class IndependentVitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads(); torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name)
        cls.caches,cls.eligible=fixture_module.fixture()
        cls.args=fixture_module.args_for(cls.root); cls.args.out.mkdir()
        cls.protocol=runner.make_protocol(cls.caches,runner.validate_cache_pair(cls.caches,production=False),{},cls.args)
        cls.results={}
        for encoder in runner.ENCODERS:
            cls.results[encoder]=runner.run_cell(cls.caches,cls.eligible,encoder,2.,3,cls.args,cls.protocol)
        cls.aggregate=runner.aggregate(cls.caches,cls.eligible,cls.args,cls.protocol)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup(); torch.set_num_threads(cls.previous_threads)

    def cell(self,encoder="deit_tiny"):
        return self.args.out/(encoder+"__rho2")/"seed_3"

    def verify_one(self,encoder="deit_tiny"):
        cache=self.caches[encoder]
        split=checker.independent_split(cache["labels"],cache["manifest"],3,2.,3)
        return checker.verify_cell(self.cell(encoder),cache,split,self.protocol,encoder,2.,3)

    def test_confusion_metrics_have_hand_computable_values(self):
        y=np.array([0,0,1,1,2,2]); pred=np.array([0,1,1,2,2,2])
        m=checker.v.metrics(y,pred,np.arange(3))
        self.assertAlmostEqual(m["accuracy"],4/6)
        self.assertTrue(np.allclose(m["per_class_f1"],[2/3,1/2,4/5]))
        self.assertEqual(m["confusion_matrix"],[[1,1,0],[0,1,1],[0,0,2]])

    def test_source_zip_labels_are_parsed_without_feature_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/"source.zip";prefix="tiny-imagenet-200/"
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr(prefix+"wnids.txt","c\na\nb\n")
                z.writestr(prefix+"val/val_annotations.txt","v0.JPEG\tc\t0\n")
                for name in ("train/b/images/b0.JPEG","val/images/v0.JPEG","train/a/images/a0.JPEG","train/c/images/c0.JPEG"):
                    z.writestr(prefix+name,b"filename and annotations only fixture")
            classes,train,evaluation,labels=checker.source_labels(archive)
            self.assertEqual(classes,["a","b","c"])
            self.assertTrue(np.array_equal(labels,[0,1,2,2]))
            self.assertEqual(len(train),3);self.assertEqual(len(evaluation),1)

    def test_split_reconstruction_handles_duplicates_independently(self):
        cache=copy.deepcopy(self.caches["deit_tiny"]); manifest=cache["manifest"]
        manifest["train_pixel_sha256"][0]=manifest["eval_pixel_sha256"][0]
        manifest["train_pixel_sha256"][5]=manifest["train_pixel_sha256"][4]
        # The original corpus hygiene helper deliberately requires200 classes
        # and10000 evaluation images. These two synthetic exclusions have an
        # explicit hand-checked answer, independent of either implementation.
        eligible=np.setdiff1d(np.arange(60,dtype=np.int64),[0,5])
        reference=runner.split_helper.make_split(cache["labels"][:60],3,2.,3,eligible)
        reference["evaluation_indices"]=np.arange(60,69,dtype=np.int64)
        recomputed=checker.independent_split(cache["labels"],manifest,3,2.,3)
        self.assertEqual(set(reference),set(recomputed))
        for k in reference: self.assertTrue(np.array_equal(reference[k],recomputed[k]),k)

    def test_completed_cells_aliases_and_aggregate_recompute(self):
        rows=[]; reports=[]
        for encoder in runner.ENCODERS:
            one,report=self.verify_one(encoder); rows+=one; reports.append(report)
        expected=checker.expected_aggregate(self.protocol,rows)
        for key in ("rows","summary","paired_comparisons"):
            sort=lambda r:json.dumps({k:v for k,v in r.items() if k in ("condition","seed","method","encoder_a","method_a","encoder_b","method_b","rho")},sort_keys=True)
            checker.close(sorted(self.aggregate[key],key=sort),sorted(expected[key],key=sort),key)
        self.assertEqual(len(rows),54)
        self.assertTrue(all(r["calibration_curves_verified"]==12 for r in reports))
        self.assertEqual(reports[0]["initial_head_sha256"],reports[1]["initial_head_sha256"])

    def mutate_metric(self,mutate,pattern):
        path=self.cell()/"metrics.json"; original=path.read_bytes()
        value=json.loads(original); mutate(value); path.write_text(json.dumps(value))
        try:
            with self.assertRaisesRegex(ValueError,pattern): self.verify_one()
        finally: path.write_bytes(original)

    def test_modified_accuracy_is_rejected(self):
        self.mutate_metric(lambda r:r["metrics"]["ce"].update(accuracy=.12345),"numeric mismatch")

    def test_invalid_uniform_prior_adjustment_is_rejected(self):
        self.mutate_metric(lambda r:r["metrics"]["rebal_crt_smote_la"]["adjustment"].update(tau=1.),"LA choice")

    def test_changed_budget_is_rejected(self):
        self.mutate_metric(lambda r:r["trunks"]["ce"].update(optimizer_updates=0),"adapter budget")

    def test_leaked_scaler_rejected_even_with_updated_hashes(self):
        cell=self.cell(); names=("metrics.json","scaler.json","scaler.npz")
        original={name:(cell/name).read_bytes() for name in names}
        try:
            raw=self.caches["deit_tiny"]["features"].astype(np.float64)
            np.savez(cell/"scaler.npz",mean=raw.mean(0),scale=raw.std(0))
            sm=json.loads(original["scaler.json"]); sm["artifact_sha256"]=checker.sha(cell/"scaler.npz")
            (cell/"scaler.json").write_text(json.dumps(sm))
            r=json.loads(original["metrics.json"])
            for name in ("scaler.npz","scaler.json"):r["artifacts"][name]=checker.sha(cell/name)
            (cell/"metrics.json").write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,"exclusively"):self.verify_one()
        finally:
            for name,data in original.items():(cell/name).write_bytes(data)

    def test_incomplete_grid_refuses_before_cache_or_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); args=fixture_module.args_for(root)
            args.seeds=[0,1,2,3,4]; args.rhos=[10.,50.,100.];args.epochs=50;args.head_steps=2000
            args.warmup_epochs=5;args.batch_size=1024;args.lr=.05;args.head_batch_size=2048;args.holdout_per_class=50;args.smote_k=5
            protocol=runner.make_protocol(self.caches,runner.validate_cache_pair(self.caches,production=False),{},args)
            (root/"protocol.json").write_text(json.dumps(protocol))
            result=checker.verify([root/"absent1",root/"absent2"],root)
            self.assertEqual(result["status"],"incomplete");self.assertEqual(result["method_evaluations_verified"],0)
            self.assertEqual(len(result["missing_cells"]),30)
            changed=copy.deepcopy(protocol);changed["helper_sha256"]={}
            changed["fingerprint"]=checker.fingerprint({k:v for k,v in changed.items() if k!="fingerprint"})
            (root/"protocol.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"helper source inventory"):checker.verify([],root)
            changed=copy.deepcopy(protocol);changed["encoders"]["deit_tiny"]={}
            changed["fingerprint"]=checker.fingerprint({k:v for k,v in changed.items() if k!="fingerprint"})
            (root/"protocol.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"provenance fields"):checker.verify([],root)
            protocol["execution"]="train_only_pilot"
            (root/"protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError,"Pilot"):checker.verify([],root)


if __name__=="__main__":unittest.main()
