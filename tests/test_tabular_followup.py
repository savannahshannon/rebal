"""Leakage, categorical semantics, attention and numeric replay checks."""
import argparse, importlib.util
from pathlib import Path
import tempfile, unittest
from unittest.mock import patch
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('tabular_followup',Path(__file__).resolve().parents[1]/'scripts/benchmark_tabular_followup.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
 y=np.repeat(np.arange(7),40).astype(np.int64);rng=np.random.default_rng(42)
 x=rng.normal(size=(len(y),54)).astype(np.float32);x[:,10:]=0
 a=rng.integers(0,4,len(y));b=rng.integers(0,40,len(y));x[np.arange(len(y)),10+a]=1;x[np.arange(len(y)),14+b]=1
 parts=m.stratified_split(y);mean,scale=m.scaler_fit(x,parts['fit']);numeric=((x[:,:10]-mean)/scale).astype(np.float32);flat=x.copy();flat[:,:10]=numeric
 return dict(flat=flat,numeric=numeric,categorical=m.decode_categories(x)),y,parts

def args():return argparse.Namespace(epochs=2,mlp_batch_size=32,ft_batch_size=32,threads=1)

class TabularFollowupTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_disjoint_split_exact_replay_and_seed(self):
  _,y,parts=fixture();replay=m.stratified_split(y)
  for k in parts:self.assertTrue(np.array_equal(parts[k],replay[k]))
  self.assertFalse(np.array_equal(parts['confirmation'],m.stratified_split(y,1729)['confirmation']))
  self.assertEqual(sum(len(v) for v in parts.values()),len(y))
  bad={k:v.copy() for k,v in parts.items()};bad['confirmation'][0]=bad['fit'][0]
  with self.assertRaisesRegex(ValueError,'overlaps'):m.validate_split(bad,y)
 def test_fit_only_scaler_unaffected_by_heldout(self):
  arrays,y,parts=fixture();x=arrays['flat'].copy();a=m.scaler_fit(x,parts['fit']);x[parts['confirmation'],:10]=100000
  b=m.scaler_fit(x,parts['fit']);self.assertTrue(all(np.array_equal(i,j) for i,j in zip(a,b)))
 def test_lossless_categorical_groups_and_invalid_rejected(self):
  arrays,_,_=fixture();cat=m.decode_categories(arrays['flat']);self.assertEqual(cat.shape,(280,2))
  self.assertTrue(np.array_equal(np.eye(4)[cat[:,0]],arrays['flat'][:,10:14]));self.assertTrue(np.array_equal(np.eye(40)[cat[:,1]],arrays['flat'][:,14:54]))
  bad=arrays['flat'].copy();bad[0,10:14]=0
  with self.assertRaisesRegex(ValueError,'one-hot'):m.decode_categories(bad)
 def test_real_feature_tokens_attention_and_gradients(self):
  arrays,_,_=fixture();model=m.FTTransformer(width=16,blocks=2,heads=4,dropout=0)
  num=torch.tensor(arrays['numeric'][:8]);cat=torch.tensor(arrays['categorical'][:8]);tokens=model.tokenizer(num,cat)
  self.assertEqual(tokens.shape,(8,13,16));out=model(num,cat);self.assertEqual(out.shape,(8,7));out.square().sum().backward()
  self.assertGreater(float(model.blocks[0].attention.in_proj_weight.grad.abs().sum()),0)
  self.assertGreater(float(model.tokenizer.numeric_weight.grad.abs().sum()),0)
  self.assertGreater(float(model.tokenizer.categorical[1].weight.grad.abs().sum()),0)
  model.eval();changed=num.clone();changed[:,0]+=10;self.assertFalse(torch.equal(model(num,cat),model(changed,cat)))
 def test_cls_only_final_attention_preserves_full_cls_output(self):
  torch.manual_seed(17); full=m.TransformerBlock(16,4,0,cls_only=False); cls=m.TransformerBlock(16,4,0,cls_only=True);cls.load_state_dict(full.state_dict())
  x=torch.randn(5,13,16);self.assertTrue(torch.allclose(full(x)[:,-1:],cls(x),atol=1e-6,rtol=0))
 def test_independent_weighted_metrics_match(self):
  y=np.repeat(np.arange(7),[100,80,60,40,20,10,5]);pred=np.random.default_rng(34).integers(0,7,len(y));a=m.metrics(y,pred,np.arange(7));b=m.independent_metrics(y,pred,np.arange(7));m.compare_numbers(a,b)
 def test_loss_weights_expected_one_and_rare_class_larger(self):
  counts=np.array([1000,100,10]);weights=m.cb_weights(counts)
  self.assertAlmostEqual(float(np.dot(counts,weights)/counts.sum()),1,places=6);self.assertGreater(weights[-1],weights[0]*10)
 def test_pilot_only_fit_no_prediction(self):
  arrays,y,parts=fixture()
  for family in ['mlp_ce','mlp_cb','ft_transformer']:
   with tempfile.TemporaryDirectory() as d,patch.object(m,'logits_for',side_effect=AssertionError('heldoutpredictionforbidden')):
    _,record=m.train_neural(arrays,y,parts,family,m.GRIDS[family][0],9,args(),Path(d),pilot_steps=2)
    self.assertFalse(record['heldout_evaluated']);self.assertEqual(record['steps'],2)
 def test_neural_training_exact_reproduction_and_best_tune(self):
  arrays,y,parts=fixture()
  with tempfile.TemporaryDirectory() as d:
   a,ra=m.train_neural(arrays,y,parts,'ft_transformer',dict(lr=.001,width=16,blocks=1,heads=4,dropout=.1,weight_decay=.0001),10,args(),Path(d)/'a')
   b,rb=m.train_neural(arrays,y,parts,'ft_transformer',dict(lr=.001,width=16,blocks=1,heads=4,dropout=.1,weight_decay=.0001),10,args(),Path(d)/'b')
   self.assertEqual(m.state_hash(a),m.state_hash(b));self.assertEqual(ra['optimizer_updates'],14);self.assertEqual(ra['best_epoch'],rb['best_epoch'])
   c,rc=m.train_neural(arrays,y,parts,'ft_transformer',dict(lr=.001,width=16,blocks=1,heads=4,dropout=.1,weight_decay=.0001),10,args(),Path(d)/'a')
   self.assertEqual(m.state_hash(a),m.state_hash(c));self.assertEqual(rc['optimizer_updates'],14)
 def test_metrics_against_sklearn_and_tau_tie(self):
  from sklearn.metrics import f1_score,confusion_matrix,recall_score
  y=np.tile(np.arange(7),8);p=np.roll(y,3);p[:10]=y[:10];record=m.metrics(y,p)
  self.assertEqual(record['macro_f1'],f1_score(y,p,average='macro'));self.assertEqual(record['macro_recall'],recall_score(y,p,average='macro'))
  self.assertEqual(record['confusion_matrix'],confusion_matrix(y,p).tolist())
  logits=np.random.default_rng(22).normal(size=(56,7));adj,la=m.adjust(logits,y,logits,np.ones(7)/7)
  self.assertEqual(la['tau'],0);self.assertTrue(np.array_equal(adj,logits))
 def test_tree_uses_only_fit_and_tune_for_training(self):
  import xgboost
  arrays,y,parts=fixture()
  with tempfile.TemporaryDirectory() as d:
   _,record=m.train_tree(arrays,y,parts,dict(max_depth=2,learning_rate=.1,subsample=.8,colsample_bytree=.9,reg_lambda=1.,min_child_weight=1.,n_estimators=50),10,args(),Path(d))
   self.assertEqual(record['completed_rounds'],50);self.assertEqual(record['selected_rounds'],50);self.assertEqual(record['tree_count'],350)

if __name__=='__main__':unittest.main()
