"""Matched encoders, fit-only scaling, resumability and result integrity checks."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("vit_rebal", Path(__file__).resolve().parents[1]/"scripts/benchmark_vit_rebal.py")
v = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(v)


def fixture():
    ntrain, neval = 60, 9
    labels = np.concatenate([np.repeat(np.arange(3),20), np.repeat(np.arange(3),3)]).astype(np.int64)
    preprocessing = dict(resize=256,crop=224,interpolation="bicubic")
    shared = dict(classes=["a","b","c"], train_files=[f"train/{i}.JPEG" for i in range(ntrain)],
                  eval_files=[f"val/{i}.JPEG" for i in range(neval)],
                  train_pixel_sha256=[f"{i:064x}" for i in range(ntrain)],
                  eval_pixel_sha256=[f"{i+1000:064x}" for i in range(neval)],
                  archive_sha256="1"*64,ntrain=ntrain,n_eval=neval,
                  original_manifest_sha256="4"*64,source_reader_sha256="5"*64,
                  extractor_source_sha256="6"*64,encoder_parameter_count=100,
                  encoder_parameter_count_scope="Synthetic fixture only",
                  preprocessing=preprocessing,preprocessing_sha256=v.fingerprint(preprocessing))
    caches = {}
    for encoder, dimension in (("resnet18",7),("deit_tiny",5)):
        m = dict(shared,encoder_id=encoder,feature_dim=dimension,weights_sha256="2"*64,encoder_state_sha256="3"*64)
        caches[encoder] = dict(manifest=copy.deepcopy(m),manifest_sha256=v.fingerprint(m),labels=labels.copy(),
                              features=np.random.default_rng(dimension).normal(size=(len(labels),dimension)).astype(np.float32))
    return caches, np.arange(ntrain,dtype=np.int64)


def args_for(root):
    return argparse.Namespace(features=[],out=root/"run",seeds=[3],rhos=[2.],epochs=2,warmup_epochs=0,batch_size=16,lr=.01,
        head_steps=6,head_batch_size=16,holdout_per_class=3,smote_k=3,checkpoint_every=1,head_checkpoint_every=2,
        device="cpu",knn_device="cpu",knn_query_block=8,knn_reference_block=16,threads=1,pilot_steps=0,
        aggregate_only=False,defer_aggregate=False,worker_count=1,worker_index=0)


class VitRebalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_pair_identity_rejects_label_order_preprocess_or_source_drift(self):
        caches,_ = fixture(); v.validate_cache_pair(caches,production=False)
        for mutation in ("labels","raw_row","preprocess"):
            changed=copy.deepcopy(caches)
            if mutation=="labels": changed["deit_tiny"]["labels"][0]=1
            if mutation=="raw_row": changed["deit_tiny"]["manifest"]["train_files"][0]="different.JPEG"
            if mutation=="preprocess": changed["deit_tiny"]["manifest"]["preprocessing"]["crop"]=64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                v.validate_cache_pair(changed,production=False)
        with self.assertRaisesRegex(ValueError,"production"):
            v.validate_cache_pair(caches)

    def test_shared_split_copies_are_identical_and_ratios_nested(self):
        caches,eligible=fixture()
        with tempfile.TemporaryDirectory() as directory:
            args=args_for(Path(directory)); protocol={"fingerprint":"p","row_identity_sha256":"r"}
            a,ah,_=v.ensure_shared_split(caches,eligible,args,protocol,3,1.)
            b,bh,_=v.ensure_shared_split(caches,eligible,args,protocol,3,2.)
            replay,rh,_=v.ensure_shared_split(caches,eligible,args,protocol,3,2.)
            self.assertEqual(bh,rh); self.assertTrue(np.array_equal(b["train_indices"],replay["train_indices"]))
            self.assertTrue(set(b["train_indices"])<=set(a["train_indices"]))
            self.assertTrue(np.array_equal(a["holdout_indices"],b["holdout_indices"]))
            self.assertFalse(set(b["train_indices"])&set(b["holdout_indices"]))
            self.assertTrue(np.array_equal(a["class_rank"],b["class_rank"]))
            self.assertTrue((b["evaluation_indices"]>=60).all())

    def test_scaler_uses_fit_only_and_rejects_tampered_resume(self):
        raw=np.arange(90,dtype=np.float32).reshape(18,5); raw[:,4]=1; fit=np.arange(7)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); x,info=v.prepare_scaler(raw,fit,root/"a",dict(fit="fixed"))
            changed=raw.copy(); changed[7:]=1e7
            other,_=v.prepare_scaler(changed,fit,root/"b",dict(fit="fixed"))
            self.assertTrue(np.array_equal(x.mean,other.mean)); self.assertTrue(np.array_equal(x.scale,other.scale))
            self.assertTrue(np.allclose(x[fit].mean(0),0.,atol=1e-7)); self.assertEqual(x.scale[-1],1.)
            self.assertGreater(other[7:][0,0],1000)
            (root/"a/scaler.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"checksum"):
                v.prepare_scaler(raw,fit,root/"a",dict(fit="fixed"))

    def test_shared_native_and_retrained_initialization_across_dimensions(self):
        a,b,c=v.Adapter(7,3,5),v.Adapter(5,3,5),v.Adapter(7,3,5)
        self.assertEqual(v.common.state_hash(a.state_dict()),v.common.state_hash(c.state_dict()))
        self.assertNotEqual(v.common.state_hash(a.state_dict()),v.common.state_hash(b.state_dict()))
        self.assertEqual(v.common.state_hash(a.head.state_dict()),v.common.state_hash(b.head.state_dict()))
        v.common.set_seed(5+90909); head=torch.nn.Linear(128,3); v.common.initialize(head)
        self.assertEqual(v.common.state_hash(head.state_dict()),v.common.state_hash(a.head.state_dict()))
        logits,features=a(torch.ones(4,7)); self.assertEqual(logits.shape,(4,3)); self.assertEqual(features.shape,(4,128))

    def test_adapter_and_streaming_head_resume_are_exact(self):
        caches,eligible=fixture(); record=caches["deit_tiny"]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); args=args_for(root); x=record["features"]; y=record["labels"]
            identity=dict(condition="fixture",seed=3)
            direct,_=v.train_adapter(x,y,eligible,3,"rebal",3,args,root/"direct",identity)
            original=v.common.save_torch
            def interrupt_adapter(path,value):
                original(path,value)
                if value.get("completed_epochs")==1: raise InterruptedError("synthetic interruption")
            with patch.object(v.common,"save_torch",side_effect=interrupt_adapter),self.assertRaises(InterruptedError):
                v.train_adapter(x,y,eligible,3,"rebal",3,args,root/"resume",identity)
            resumed,_=v.train_adapter(x,y,eligible,3,"rebal",3,args,root/"resume",identity)
            self.assertEqual(v.common.state_hash(direct.state_dict()),v.common.state_hash(resumed.state_dict()))
            def interrupt_final(path,value):
                original(path,value)
                if value.get("completed_epochs")==args.epochs: raise InterruptedError("final sidecar gap")
            with patch.object(v.common,"save_torch",side_effect=interrupt_final),self.assertRaises(InterruptedError):
                v.train_adapter(x,y,eligible,3,"rebal",3,args,root/"final_gap",identity)
            recovered,_=v.train_adapter(x,y,eligible,3,"rebal",3,args,root/"final_gap",identity)
            self.assertEqual(v.common.state_hash(direct.state_dict()),v.common.state_hash(recovered.state_dict()))
            self.assertEqual(len(json.loads((root/"final_gap/history.json").read_text())),args.epochs)
            features=np.random.default_rng(9).normal(size=(18,128)).astype(np.float32); labels=np.repeat(np.arange(3),[10,5,3])
            graph,_=v.nonimage.neighbor_graph(features,labels,3,k=2)
            head,record=v.train_head(features,labels,graph,"smote",3,args,3,root/"direct_head.pt",identity)
            def interrupt_head(path,value):
                original(path,value)
                if value.get("completed_updates")==2: raise InterruptedError("synthetic interruption")
            with patch.object(v.common,"save_torch",side_effect=interrupt_head),self.assertRaises(InterruptedError):
                v.train_head(features,labels,graph,"smote",3,args,3,root/"resume_head.pt",identity)
            replay,replayed_record=v.train_head(features,labels,graph,"smote",3,args,3,root/"resume_head.pt",identity)
            self.assertEqual(v.common.state_hash(head.state_dict()),v.common.state_hash(replay.state_dict()))
            self.assertEqual(record["losses"],replayed_record["losses"])

    def test_vectorized_sampler_preserves_rng_records_and_head_trajectory(self):
        configurations=(np.array([10,10,3]),np.floor(445*10.**(-np.arange(200)/199)).astype(int),
                        np.floor(445*100.**(-np.arange(200)/199)).astype(int))
        for counts in configurations:
            labels=np.repeat(np.arange(len(counts)),counts)
            labels=labels[np.random.default_rng(451).permutation(len(labels))]
            features=np.random.default_rng(19).normal(size=(len(labels),128)).astype(np.float32)
            k=min(5,int(counts.min())-1);graph=np.full((len(labels),k),-1,dtype=np.int64)
            for c,count in enumerate(counts):
                ids=np.flatnonzero(labels==c)
                if count<counts.max():graph[ids]=ids[(np.arange(count)[:,None]+1+np.arange(k)[None,:])%count]
            for seed in (0,7,91):
                for kind in v.HEADS:
                    baseline=v.nonimage.LatentSampler(features,labels,graph,"cpu",seed)
                    candidate=v.VectorizedLatentSampler(features,labels,graph,"cpu",seed)
                    for size in (1,31,2048):
                        x,y,r=baseline.sample(kind,size);xx,yy,rr=candidate.sample(kind,size)
                        self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy))
                        self.assertTrue(all(np.array_equal(r[key],rr[key]) for key in r))
                        self.assertEqual(baseline.rng.bit_generator.state,candidate.rng.bit_generator.state)
            if len(counts)==3:
                with tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);args=args_for(root)
                    for kind in v.HEADS:
                        candidate,record=v.train_head(features,labels,graph,kind,3,args,3,root/(kind+"_candidate.pt"),{"test":"equivalence"})
                        with patch.object(v,"VectorizedLatentSampler",v.nonimage.LatentSampler):
                            baseline,original=v.train_head(features,labels,graph,kind,3,args,3,root/(kind+"_baseline.pt"),{"test":"equivalence"})
                        self.assertEqual(v.common.state_hash(candidate.state_dict()),v.common.state_hash(baseline.state_dict()))
                        self.assertEqual(record["losses"],original["losses"])

    def test_end_to_end_aliases_pairing_aggregate_and_tamper(self):
        caches,eligible=fixture(); row_identity=v.validate_cache_pair(caches,production=False)
        with tempfile.TemporaryDirectory() as directory:
            args=args_for(Path(directory)); args.out.mkdir()
            protocol=v.make_protocol(caches,row_identity,{},args); records=[]
            for encoder in v.ENCODERS:
                result=v.run_cell(caches,eligible,encoder,2.,3,args,protocol); records.append(result)
                self.assertEqual(len(result["metrics"]),27)
                self.assertEqual(result,v.run_cell(caches,eligible,encoder,2.,3,args,protocol))
                cell=args.out/v.cell_name(encoder,2.)/"seed_3"
                with np.load(cell/"predictions.npz",allow_pickle=False) as arrays:
                    self.assertEqual(len(arrays.files),33)
                    self.assertFalse(any(name.endswith("_la__test_logits") for name in arrays.files))
                    for trunk in v.TRUNKS:
                        for head in ("ros","smote"):
                            name=f"{trunk}_crt_{head}"
                            self.assertEqual(result["metrics"][name+"_la"]["adjustment"]["tau"],0.)
                            self.assertEqual(result["metrics"][name]["macro_f1"],result["metrics"][name+"_la"]["macro_f1"])
            self.assertEqual(records[0]["identity"]["split_sha256"],records[1]["identity"]["split_sha256"])
            self.assertEqual(records[0]["trunks"]["ce"]["initial_head_sha256"],records[1]["trunks"]["ce"]["initial_head_sha256"])
            summary=v.aggregate(caches,eligible,args,protocol)
            self.assertEqual((summary["completed_cells"],summary["evaluations"],len(summary["paired_comparisons"])),(2,54,5))
            self.assertTrue(all("signflip_holm" in c for c in summary["paired_comparisons"]))
            marker=args.out/"deit_tiny__rho2/seed_3/metrics.json"; original=marker.read_bytes(); marker.unlink()
            partial=v.aggregate(caches,eligible,args,protocol)
            self.assertEqual(partial["status"],"incomplete")
            self.assertTrue(all("signflip_holm" not in c for c in partial["paired_comparisons"]))
            marker.write_bytes(original)
            artifact=args.out/"deit_tiny__rho2/seed_3/scaler.npz"; artifact.write_bytes(artifact.read_bytes()+b"corrupt")
            with self.assertRaisesRegex(ValueError,"checksum"):
                v.aggregate(caches,eligible,args,protocol)

    def test_pilot_never_models_heldout_or_scores(self):
        caches,eligible=fixture(); row_identity=v.validate_cache_pair(caches,production=False)
        with tempfile.TemporaryDirectory() as directory:
            args=args_for(Path(directory)); args.pilot_steps=2; args.out.mkdir()
            protocol=v.make_protocol(caches,row_identity,{},args)
            allowed=set(v.make_shared_split(caches,eligible,3,2.,3)["train_indices"].tolist())
            for encoder in v.ENCODERS:
                with patch.object(v.nonimage,"encode",side_effect=AssertionError("heldout encoding forbidden")), \
                     patch.object(v.nonimage,"neighbor_graph",side_effect=AssertionError("neighbor graph forbidden")), \
                     patch.object(v.nonimage,"scores",side_effect=AssertionError("scores forbidden")), \
                     patch.object(v.nonimage,"adjust",side_effect=AssertionError("calibration forbidden")), \
                     patch.object(v,"train_head",side_effect=AssertionError("classifier retraining forbidden")), \
                     patch.object(v.nonimage,"input_batch",wraps=v.nonimage.input_batch) as batches:
                    result=v.train_only_pilot(caches,eligible,encoder,args,protocol)
                self.assertFalse(result["test_evaluated"]); self.assertFalse(result["calibration_evaluated"])
                self.assertTrue(all(set(call.args[1])<=allowed for call in batches.call_args_list))
            self.assertFalse(list(args.out.rglob("metrics.json"))); self.assertFalse(list(args.out.rglob("predictions.npz")))


if __name__=="__main__": unittest.main()
