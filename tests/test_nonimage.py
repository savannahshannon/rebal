"""Leakage, sampling, evidence and resume checks for non-image adaptation."""
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

spec=importlib.util.spec_from_file_location('nonimage',Path(__file__).resolve().parents[1]/'scripts/train_nonimage.py')
n=importlib.util.module_from_spec(spec); spec.loader.exec_module(n)


def fixture(root,kind):
    root.mkdir(); labels=np.repeat(np.arange(3),18).astype(np.int64)
    train=np.concatenate([np.arange(c*18,c*18+count) for c,count in enumerate((12,9,6))])
    cal=np.concatenate([np.arange(c*18+12,c*18+15) for c in range(3)]); test=np.concatenate([np.arange(c*18+15,c*18+18) for c in range(3)])
    np.save(root/'labels.npy',labels); n.save_npz(root/'split.npz',train=train,calibration=cal,test=test,class_rank=np.array([2,0,1]))
    names=['labels.npy','split.npz']
    if kind=='tabular':
        raw=np.random.default_rng(73).normal(size=(54,54)).astype(np.float32); raw[:,10:]=(raw[:,10:]>0).astype(np.float32); raw[cal,0]=10000.
        np.save(root/'features.npy',raw); names.append('features.npy')
    else:
        with open(root/'texts.jsonl','w') as f:
            for i,y in enumerate(labels):
                token='calibrationonly' if i in cal else 'testonly' if i in test else 'trainingword'
                f.write(json.dumps(dict(title=f'Class{y} row{i}',content=f'{token} class{y} ordinary text'))+'\n')
        names.append('texts.jsonl')
    meta=dict(dataset='covertype' if kind=='tabular' else 'dbpedia',condition='fixture',task_type=kind,n_classes=3,classes=['a','b','c'],
            rank_semantics='ordered_class_ids_head_to_tail',files_sha256={name:n.sha(root/name) for name in names})
    n.write_json(root/'metadata.json',meta); return root


def args_for(root):
    return argparse.Namespace(command='run',data=[],out=root/'runs',seeds=[3],epochs=2,warmup_epochs=0,batch_size=8,lr=.01,
        head_steps=3,head_batch_size=16,max_words=100,max_tokens=12,device='cpu',knn_device='cpu',knn_query_block=4,knn_reference_block=8,
        smote_k=3,ecod=False,threads=1,checkpoint_every=1,pilot_steps=0,aggregate_only=False,worker_index=0,worker_count=1,defer_aggregate=False)


class NonimageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_train_only_scaler_and_rank(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); condition=n.load_condition(fixture(root/'tabular','tabular')); directory,meta,y,split=condition; args=args_for(root)
            x,fit,info=n.prepare_transforms(directory,meta,y,split,args,root/'transforms'); raw=np.load(directory/'features.npy')
            scaler=np.load(root/'transforms/scaler.npz'); self.assertTrue(np.allclose(scaler['mean'],raw[fit,:10].astype(float).mean(0)))
            self.assertTrue(np.array_equal(x[:,10:],raw[:,10:])); self.assertLess(abs(x[fit,0].mean()),1e-6)
            self.assertEqual(split['rank'].tolist(),[1,2,0]); self.assertTrue(x[split['calibration'],0].mean()>1000)
            replay=n.prepare_transforms(directory,meta,y,split,args,root/'transforms'); self.assertEqual(info,replay[2])

    def test_train_only_vocabulary_and_specials(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); directory,meta,y,split=n.load_condition(fixture(root/'text','text')); args=args_for(root)
            x,fit,info=n.prepare_transforms(directory,meta,y,split,args,root/'transforms'); vocab=json.loads((root/'transforms/vocabulary.json').read_text())
            self.assertNotIn('calibrationonly',vocab); self.assertNotIn('testonly',vocab); self.assertIn('trainingword',vocab)
            self.assertEqual([vocab[k] for k in ('<pad>','<unk>','<sep>')],[0,1,2]); self.assertTrue((x[split['test']]==1).any())
            self.assertEqual(n.tokens('CAFÉ','ＡＢＣ'),['café','<sep>','abc']); self.assertEqual(info['detail']['vocabulary_size'],len(vocab))

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=fixture(Path(d)/'tabular','tabular'); meta=json.loads((path/'metadata.json').read_text())
            with np.load(path/'split.npz') as z: split={k:z[k] for k in z.files}
            split['calibration'][0]=split['train'][0]; n.save_npz(path/'split.npz',**split); meta['files_sha256']['split.npz']=n.sha(path/'split.npz'); n.write_json(path/'metadata.json',meta)
            with self.assertRaisesRegex(ValueError,'overlap'): n.load_condition(path)

    def test_count_aware_cb_does_not_saturate_and_scale_one(self):
        counts=np.array([300000,200000,1000]); w,beta=n.cb_weights(counts)
        self.assertAlmostEqual(float((counts*w).sum()/counts.sum()),1.,places=6); self.assertGreater(w[-1],100*w[0]); self.assertAlmostEqual(beta,1-1/300000)

    def test_exact_graph_streaming_interpolation(self):
        features=np.random.default_rng(63).normal(size=(18,128)).astype(np.float32); labels=np.repeat(np.arange(3),[10,5,3])
        graph,meta=n.neighbor_graph(features,labels,3,k=5,query_block=2)
        self.assertTrue((graph[:10]==-1).all()); self.assertEqual(meta['k_neighbors'],2)
        for i in range(10,18):
            candidates=np.flatnonzero(labels==labels[i]); candidates=candidates[candidates!=i]
            expected=candidates[np.argsort(((features[candidates]-features[i])**2).sum(1))[:2]]
            self.assertEqual(graph[i].tolist(),expected.tolist())
        sampler=n.LatentSampler(features,labels,graph,'cpu',54); x,y,record=sampler.sample('smote',2000)
        expected=features[record['base']]+record['alpha'][:,None]*(features[record['neighbor']]-features[record['base']])
        self.assertTrue(np.allclose(x.numpy(),expected,atol=5e-7)); self.assertTrue(record['synthetic'].any())
        self.assertTrue(np.array_equal(labels[record['neighbor']],y.numpy())); self.assertFalse(record['synthetic'][y.numpy()==0].any())

    def test_balanced_head_la_uniform_prior_has_no_effect(self):
        logits=np.random.default_rng(52).normal(size=(12,3)).astype(np.float32); labels=np.tile(np.arange(3),4)
        cal,test,meta=n.adjust(logits,labels,logits,np.ones(3)/3)
        self.assertEqual(meta['tau'],0.); self.assertTrue(np.array_equal(cal,logits)); self.assertTrue(np.array_equal(test,logits))

    def test_tabular_and_text_end_to_end(self):
        for kind in ('tabular','text'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d); condition=n.load_condition(fixture(root/'input',kind)); args=args_for(root); args.out.mkdir()
                protocol=n.make_protocol([condition],args); result=n.run_seed(condition,args,protocol,3)
                self.assertEqual(len(result['metrics']),27); self.assertEqual(len(result['trunks']),3)
                initials={result['metrics'][f'{m}_crt_{h}']['training']['initial_state_sha256'] for m in n.TRUNKS for h in n.HEADS}; self.assertEqual(len(initials),1)
                self.assertEqual(n.run_seed(condition,args,protocol,3),result)
                agg=n.aggregate([condition],args,protocol); self.assertEqual(agg['completed_cells'],1); self.assertEqual(agg['evaluations'],27)
                name=n.identifier(condition[1]); path=args.out/name/'seed_3/metrics.json'; bad=copy.deepcopy(result); bad['identity']['seed']=4; n.write_json(path,bad)
                with self.assertRaisesRegex(ValueError,'identity mismatch'): n.aggregate([condition],args,protocol)
                n.write_json(path,result)
                transform=args.out/name/'transforms'/('scaler.npz' if kind=='tabular' else 'vocabulary.json'); transform.write_bytes(b'corrupt transform')
                with self.assertRaisesRegex(ValueError,'Transform artifact checksum mismatch'): n.aggregate([condition],args,protocol)

    def test_interrupted_resume_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); directory,meta,y,split=n.load_condition(fixture(root/'input','tabular')); args=args_for(root)
            x,fit,_=n.prepare_transforms(directory,meta,y,split,args,root/'transform'); identity=dict(source='test',seed=3)
            direct,_=n.train_trunk(x,y,fit,'tabular',3,None,'rebal',3,args,root/'direct',identity); real=n.common.save_torch
            def interrupt(path,value):
                real(path,value)
                if value['completed_epochs']==1: raise InterruptedError('test')
            with patch.object(n.common,'save_torch',side_effect=interrupt):
                with self.assertRaises(InterruptedError): n.train_trunk(x,y,fit,'tabular',3,None,'rebal',3,args,root/'resume',identity)
            resumed,_=n.train_trunk(x,y,fit,'tabular',3,None,'rebal',3,args,root/'resume',identity)
            self.assertEqual(n.common.state_hash(direct.state_dict()),n.common.state_hash(resumed.state_dict()))

    def test_train_only_pilot_never_scores_or_builds_neighbors(self):
        for kind in ('tabular','text'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d); condition=n.load_condition(fixture(root/'input',kind)); args=args_for(root); args.pilot_steps=2; args.out.mkdir()
                protocol=n.make_protocol([condition],args)
                with patch.object(n,'encode',side_effect=AssertionError('held-out model encoding forbidden')),patch.object(n,'neighbor_graph',side_effect=AssertionError('neighbor graph forbidden')),patch.object(n,'scores',side_effect=AssertionError('scoring forbidden')),patch.object(n,'adjust',side_effect=AssertionError('calibration forbidden')),patch.object(n,'train_head',side_effect=AssertionError('readout training forbidden')),patch.object(n,'input_batch',wraps=n.input_batch) as model_inputs:
                    report=n.throughput_pilot(condition,args,protocol)
                allowed=set(condition[3]['train'].tolist())
                self.assertTrue(all(set(call.args[1].tolist())<=allowed for call in model_inputs.call_args_list))
                self.assertFalse(report['test_evaluated']); self.assertFalse(report['calibration_evaluated']); self.assertFalse(report['neighbor_graph_built'])
                self.assertEqual(len(report['records']),3); self.assertTrue(all(r['optimizer_updates']==2 for r in report['records']))
                self.assertEqual(len({r['initial_state_sha256'] for r in report['records']}),1)
                self.assertFalse(list(args.out.rglob('metrics.json'))); self.assertFalse(list(args.out.rglob('predictions.npz')))


if __name__=='__main__': unittest.main()
