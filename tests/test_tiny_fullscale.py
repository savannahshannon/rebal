"""Meaningful controls for the separate from-scratch PyTorch adaptation."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

spec=importlib.util.spec_from_file_location('fullscale',Path(__file__).resolve().parents[1]/'scripts/train_tiny_fullscale.py')
f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def mini_archive(path):
    rng=np.random.default_rng(818); classes=['n00000001','n00000002','n00000003']; prefix='tiny-imagenet-200/'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(prefix+'wnids.txt','\n'.join(classes)); annotations=[]
        for k,c in enumerate(classes):
            for i in range(12):
                a=rng.integers(0,256,(64,64,3),dtype=np.uint8); buf=io.BytesIO(); Image.fromarray(a).save(buf,format='JPEG')
                z.writestr(prefix+f'train/{c}/images/{c}_{i:03d}.JPEG',buf.getvalue())
            for i in range(3):
                a=rng.integers(0,256,(64,64,3),dtype=np.uint8); buf=io.BytesIO(); Image.fromarray(a).save(buf,format='JPEG')
                name=f'val_{k:03d}_{i:03d}.JPEG'; z.writestr(prefix+'val/images/'+name,buf.getvalue()); annotations.append(name+'\t'+c+'\t0\t0\t64\t64')
        z.writestr(prefix+'val/val_annotations.txt','\n'.join(annotations))


def args_for(path):
    return argparse.Namespace(command='run',cache_dir=path/'cache',out=path/'runs',rhos=[2.],seeds=[7],epochs=2,
        crt_epochs=1,warmup_epochs=0,lr=.01,batch_size=8,crt_batch_size=16,holdout_per_class=2,workers=0,device='cpu',
        checkpoint_every=1,threads=1,no_ecod=True,blocks=1,aggregate_only=False)


class FullscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_equalization_matches_numpy_and_has_gradients(self):
        x=torch.tensor([[1.,0.],[.8,.6],[0.,1.],[.6,.8],[1.,1.]],requires_grad=True); y=torch.tensor([0,0,1,1,2])
        a=x.detach().numpy(); a=a/np.linalg.norm(a,axis=1,keepdims=True)
        centers=np.stack([a[y.numpy()==k].mean(0) for k in (0,1)])
        spread=np.mean([np.square(a[y.numpy()==k]-centers[k]).sum(1).mean() for k in (0,1)])
        expected=spread-np.square(centers[0]-centers[1]).sum()
        loss=f.equalization(x,y,3); self.assertAlmostEqual(float(loss.detach()),float(expected),places=6)
        loss.backward(); self.assertTrue(torch.isfinite(x.grad).all())

    def test_singleton_equalization_finite_zero(self):
        x=torch.randn(3,5,requires_grad=True); value=f.equalization(x,torch.arange(3),3)
        self.assertEqual(float(value.detach()),0.); value.backward(); self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))

    def test_architecture_and_kernel_only_l2(self):
        model=f.ResNet32(200); self.assertEqual(len(model.blocks),15)
        self.assertEqual(model.bn.eps,.001); self.assertEqual(model.bn.momentum,.01)
        for module in model.modules():
            if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d)) and module.bias is not None: self.assertTrue(torch.count_nonzero(module.bias)==0)
        expected=sum(m.weight.square().sum() for m in model.modules() if isinstance(m,(torch.nn.Linear,torch.nn.Conv2d)))
        self.assertEqual(float(f.kernel_l2(model).detach()),float(expected.detach()))
        model.eval(); logits,features=model(torch.rand(2,3,64,64)); self.assertEqual(logits.shape,(2,200)); self.assertEqual(features.shape,(2,128))

    def test_dedup_and_nested_split(self):
        labels=np.repeat(np.arange(3),20); labels=np.concatenate([labels,np.repeat(np.arange(3),2)])
        hashes=[f'{i:064x}' for i in range(len(labels))]; hashes[1]=hashes[0]; hashes[2]=hashes[60]
        manifest=dict(train_count=60,classes=['a','b','c'],pixel_sha256=hashes)
        a=f.make_split(labels,manifest,8,1,2); b=f.make_split(labels,manifest,8,2,2)
        self.assertEqual(a['excluded_duplicate'].tolist(),[1]); self.assertEqual(a['excluded_overlap'].tolist(),[2])
        self.assertTrue(np.array_equal(a['holdout'],b['holdout'])); self.assertTrue(set(b['fit_before_ecod'])<=set(a['fit_before_ecod']))
        self.assertFalse(set(a['fit_before_ecod'])&set(a['holdout'])); self.assertTrue(np.array_equal(a['class_rank'],b['class_rank']))

    def test_loss_weighting_and_schedule(self):
        weights=f.cb_weights(np.array([20,10,4])); self.assertAlmostEqual(float(weights.mean()),1.,places=6)
        self.assertGreater(weights[2],weights[0]); self.assertEqual(f.learning_rate(0,100,10,.1),0)
        self.assertAlmostEqual(f.learning_rate(10,100,10,.1),.1); self.assertAlmostEqual(f.learning_rate(100,100,10,.1),0.)

    def test_la_uses_holdout_and_tail_prior(self):
        dev=np.array([[3.,0.,0.],[3.,1.,0.],[3.,0.,1.]],dtype=np.float32); labels=np.arange(3); counts=np.array([100,10,1])
        tau,curve,prior=f.choose_tau(dev,labels,counts); self.assertIn(tau,f.TAUS)
        self.assertTrue(np.allclose(prior,np.log(counts/counts.sum()))); self.assertEqual(tau,max(curve,key=lambda row:row['macro_recall'])['tau'])

    def test_crt_update_initialization_control(self):
        rng=np.random.default_rng(31); x=rng.normal(size=(18,128)).astype(np.float32); y=np.repeat(np.arange(3),[10,5,3])
        with tempfile.TemporaryDirectory() as d:
            args=args_for(Path(d)); records=[]
            for kind in ('natural','ros','smote'):
                _,record,_=f.train_crt(x,y,kind,4,args,3); records.append(record)
            self.assertEqual(len({r['initial_state_sha256'] for r in records}),1)
            self.assertEqual(len({r['optimizer_updates'] for r in records}),1); self.assertEqual(records[2]['k_neighbors'],2)

    def test_end_to_end_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'mini.zip'; mini_archive(archive); args=args_for(root)
            f.prepare(archive,args.cache_dir,3); images,labels,manifest=f.load_cache(args.cache_dir); protocol=f.make_protocol(args,manifest)
            result=f.run_cell(images,labels,manifest,args,protocol,7,2.)
            self.assertEqual(len(result['metrics']),17); self.assertEqual(len(result['trunks']),3)
            self.assertEqual(len({r['initial_state_sha256'] for r in result['trunks'].values()}),1)
            resumed=f.run_cell(images,labels,manifest,args,protocol,7,2.); self.assertEqual(result,resumed)
            aggregate=f.aggregate(args,protocol); self.assertEqual(aggregate['completed_evaluations'],17)
            bad=copy.deepcopy(result); bad['identity']['seed']=99
            metricpath=args.out/'rho_2'/'seed_7'/'metrics.json'; f.write_json(metricpath,bad)
            with self.assertRaisesRegex(ValueError,'identity mismatch'): f.aggregate(args,protocol)
            f.write_json(metricpath,result)
            splitpath=args.out/'rho_2'/'seed_7'/'split.npz'; splitpath.write_bytes(splitpath.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'identity mismatch'): f.run_cell(images,labels,manifest,args,protocol,7,2.)

    def test_interrupted_trunk_resume_exact_cpu_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'mini.zip'; mini_archive(archive); args=args_for(root)
            f.prepare(archive,args.cache_dir,3); images,labels,manifest=f.load_cache(args.cache_dir)
            split=f.make_split(labels,manifest,7,2,2); split['fit']=split['fit_before_ecod']; identity=dict(seed=7,rho=2.,protocol='test')
            direct,record=f.train_trunk('rebal',images,labels,split,args,root/'direct',identity)
            real_save=f.save_torch
            def interrupt(path,obj):
                real_save(path,obj)
                if obj.get('completed_epochs')==1: raise InterruptedError('intentional test interruption')
            with patch.object(f,'save_torch',side_effect=interrupt):
                with self.assertRaises(InterruptedError): f.train_trunk('rebal',images,labels,split,args,root/'resumed',identity)
            resumed,_=f.train_trunk('rebal',images,labels,split,args,root/'resumed',identity)
            self.assertEqual(f.state_hash(direct.state_dict()),f.state_hash(resumed.state_dict()))
            self.assertEqual(record['completed_epochs'],2)


if __name__=='__main__': unittest.main()
