"""Invariants protecting the pretrained text follow-up."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py')); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
extract=load('extract_text_followup_features')

class TextPreparationTests(unittest.TestCase):
 def test_masked_mean_padding_and_alignment(self):
  h=torch.tensor([[[1.,3.],[3.,5.],[100.,100.]]]); mask=torch.tensor([[1,1,0]])
  self.assertTrue(torch.equal(extract.masked_mean(h,mask),torch.tensor([[2.,4.]])))
  with self.assertRaises(ValueError):extract.masked_mean(h,torch.tensor([[0,0,0]]))
  with self.assertRaises(ValueError):extract.masked_mean(h,torch.ones(1,2))
 def test_fresh_splits_disjoint_deterministic_all_classes(self):
  # Synthetic source retains native boundary to test exact row-ID contract.
  y=np.repeat(np.arange(14),40000); y=np.concatenate([y,np.repeat(np.arange(14),3)])
  canonical=np.arange(560000,dtype=np.int64)
  a,detail=extract.fresh_splits(y,canonical,confirmation=4,tune=2,calibration=2)
  b,_=extract.fresh_splits(y,canonical,confirmation=4,tune=2,calibration=2)
  for k in a:self.assertTrue(np.array_equal(a[k],b[k]))
  self.assertEqual(detail['split_counts']['confirmation'],56)
  self.assertEqual(detail['split_class_counts']['confirmation'],[4]*14)
  self.assertEqual(a['fit'].dtype,np.int64)
  self.assertEqual(set(y[a['fit']]),set(range(14)))
  a['tune'][0]=a['fit'][0]
  with self.assertRaises(ValueError):extract.validate_split(y,a)
 def test_resume_journal_detects_feature_and_prefix_corruption(self):
  features=np.arange(16,dtype=np.float32).reshape(8,2);prefix=np.arange(24,dtype=np.float16).reshape(4,3,2)
  order=np.array([3,0,2,1,7,4,6,5]);ids=np.arange(8);lookup=np.array([0,-1,1,-1,2,-1,3,-1])
  blocks=[]
  for begin,end in [(0,4),(4,8)]:
   positions=order[begin:end];loc=lookup[positions];loc=loc[loc>=0]
   blocks.append(dict(begin=begin,end=end,features_sha256=extract.array_digest(features,positions),prefix_rows=len(loc),prefix_sha256=extract.array_digest(prefix,loc)))
  state=dict(fingerprint='test',completed_rows=8,blocks=blocks)
  self.assertEqual(extract.verify_block_journal(state,'test',order,ids,features,prefix,lookup),8)
  features[0,0]+=1
  with self.assertRaises(ValueError):extract.verify_block_journal(state,'test',order,ids,features,prefix,lookup)
  features[0,0]-=1;prefix[0,0,0]+=1
  with self.assertRaises(ValueError):extract.verify_block_journal(state,'test',order,ids,features,prefix,lookup)
  with self.assertRaises(ValueError):extract.verify_block_journal(state,'wrong',order,ids,features,prefix,lookup)
 def test_unsafe_assets_rejected(self):
  with self.assertRaises(ValueError):extract.safe_name('../config.json')
 def test_offline_tokenizer_truncation_pair_and_pad(self):
  from transformers import BertTokenizerFast
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory);(p/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n')
   tokenizer=BertTokenizerFast(vocab_file=str(p/'vocab.txt'),do_lower_case=True)
   b=tokenizer(['HELLO '*20,'world'],['world '*20,'hello'],padding='max_length',max_length=12,truncation='longest_first',return_tensors='np')
   self.assertEqual(b['input_ids'].shape,(2,12));self.assertEqual(int(b['attention_mask'][0].sum()),12)
   self.assertTrue(np.all(b['input_ids'][1,b['attention_mask'][1]==0]==0))
   self.assertEqual(int(b['input_ids'][0,0]),tokenizer.cls_token_id)
   self.assertEqual(int((b['input_ids'][0]==tokenizer.sep_token_id).sum()),2)

bench=load('benchmark_text_followup')

class TextTrainingTests(unittest.TestCase):
 def test_balanced_softmax_formula(self):
  from torch.nn import functional as F
  logits=torch.tensor([[1.,2.],[3.,1.]],requires_grad=True);labels=torch.tensor([0,1]);counts=np.array([100,1])
  expected=F.cross_entropy(logits+torch.tensor(counts).log(),labels)
  self.assertTrue(torch.allclose(bench.balanced_softmax(logits,labels,counts),expected))
  with self.assertRaises(ValueError):bench.balanced_softmax(logits,labels,[1,0])
 def test_bucket_batches_preserve_every_fit_row_and_are_deterministic(self):
  y=np.tile(np.arange(14),20);fit=np.arange(len(y),dtype=np.int64);lengths=np.resize(np.array([17,50,90,128]),len(y))
  a=bench.epoch_batches(fit,y,10,0,64,lengths=lengths);b=bench.epoch_batches(fit,y,10,0,64,lengths=lengths)
  self.assertEqual(set(np.concatenate(a)),set(fit));self.assertEqual(len(np.concatenate(a)),len(fit))
  self.assertEqual(len(a),len(b))
  for left,right in zip(a,b):self.assertTrue(np.array_equal(left,right));self.assertEqual(len(set(((lengths[left]+31)//32)*32)),1)
 def test_history_rejects_short_overrun_and_wrong_updates(self):
  history=[dict(epoch=i,optimizer_updates=i*10,batches=10,loss=1.,seconds=.1,EQ_eligible_batches_per_class=[2]*14) for i in (1,2)]
  bench.validate_history(history,2,2,10)
  with self.assertRaises(ValueError):bench.validate_history(history,2,1,10)
  with self.assertRaises(ValueError):bench.validate_history(history,1,2,10)
  history[1]['optimizer_updates']=19
  with self.assertRaises(ValueError):bench.validate_history(history,2,2,10)
 def test_sampling_seed_alignment_and_balanced_frequency(self):
  y=np.repeat(np.arange(14),np.arange(14)+5);fit=np.arange(len(y),dtype=np.int64)
  natural=bench.sampling_ids(fit,y,10,0,False)
  self.assertEqual(set(natural),set(fit))
  self.assertTrue(np.array_equal(natural,bench.sampling_ids(fit,y,10,0,False)))
  draws=np.concatenate([bench.sampling_ids(fit,y,10,i,True) for i in range(100)])
  freq=np.bincount(y[draws],minlength=14)/len(draws)
  self.assertTrue(np.max(abs(freq-1/14))<.01)
 def test_partial_freeze_and_dropout_modes(self):
  from transformers import BertConfig,BertModel
  bert=BertModel(BertConfig(hidden_size=24,num_hidden_layers=12,num_attention_heads=4,intermediate_size=48,vocab_size=31))
  model=bench.PartialBert(bert,1)
  self.assertTrue(all(not p.requires_grad for p in model.layers[0].parameters()))
  self.assertTrue(all(p.requires_grad for p in model.layers[1].parameters()))
  self.assertFalse(model.layers[0].training);self.assertTrue(model.layers[1].training)
  model.eval();self.assertFalse(model.layers[1].training)
  model.train();self.assertFalse(model.layers[0].training)
  logits,features=model(torch.rand(3,5,24),torch.ones(3,5,dtype=torch.long))
  self.assertEqual(logits.shape,(3,14));self.assertEqual(features.shape,(3,128))
  model=bench.PartialBert(bert,2);self.assertTrue(all(p.requires_grad for p in model.layers.parameters()))
 def test_calibration_uniform_prior_no_rank_change_and_smallest_tau_tie(self):
  rng=np.random.default_rng(88);cal=rng.normal(size=(28,14)).astype(np.float32);test=rng.normal(size=(21,14)).astype(np.float32)
  labels=np.tile(np.arange(14),2);adjusted,choice=bench.adjust(cal,labels,test,np.ones(14)/14)
  self.assertEqual(choice['tau'],0.)
  self.assertTrue(np.array_equal(adjusted.argmax(1),test.argmax(1)))
  with self.assertRaises(ValueError):bench.adjust(cal,labels,test,np.zeros(14))
 def test_mps_rng_resume_cpu_roundtrip(self):
  bench.common.set_seed(33);saved=bench.get_rng();a=torch.rand(8);b=np.random.rand(8);bench.restore_rng(saved)
  self.assertTrue(torch.equal(a,torch.rand(8)));self.assertTrue(np.array_equal(b,np.random.rand(8)))
 def test_partial_bucket_prediction_scatter_preserves_original_source_order(self):
  import argparse
  from unittest.mock import patch
  from transformers import BertConfig,BertModel
  cfg=BertConfig(hidden_size=24,num_hidden_layers=12,num_attention_heads=4,intermediate_size=48,vocab_size=31);cfg._attn_implementation='sdpa'
  y=np.tile(np.arange(14),6);fit=np.arange(14,dtype=np.int64);tune=np.arange(14,28,dtype=np.int64);lengths=np.resize(np.array([17,50,90,128]),84)
  mask=np.zeros((84,128),dtype=np.uint8)
  for i,n in enumerate(lengths):mask[i,:n]=1
  hidden=np.random.default_rng(312).normal(size=(84,128,24)).astype(np.float16);lookup=np.arange(84,dtype=np.int64)
  d=dict(y=y,split={'fit':fit,'tune':tune},_lengths=lengths,_cls_scaler=(np.zeros(24),np.ones(24)))
  with tempfile.TemporaryDirectory() as directory:
   args=argparse.Namespace(out=Path(directory),protocol_fingerprint='synthetic',modeldir=Path('unused'),device='cpu',attention='sdpa',partial_epochs=2,adapter_epochs=2,partial_batch_size=14,batch_size=14,partial_length_buckets=True,partial_precision='amp_float16',eq_weight=.05)
   with patch.object(bench.extract,'load_bert',side_effect=lambda *a,**k:BertModel(cfg,add_pooling_layer=False)):
    selection,predict,model=bench.train_model(d,'last1_ce',10,.001,args,None,prefix=(hidden,lookup,mask))
   ids=np.array([37,32,41,28,40,35,38,31,29,36,30,39,34,33],dtype=np.int64);optimized=predict(ids);model.eval()
   with torch.inference_mode():reference=model(torch.tensor(hidden[ids],dtype=torch.float32),torch.tensor(mask[ids],dtype=torch.long))[0].numpy()
   np.testing.assert_allclose(optimized,reference,atol=2e-5,rtol=2e-5)
   self.assertEqual(selection['epochs'],2);self.assertEqual(selection['updates'],8)
 def test_cls_only_full_layer_value_input_and_parameter_gradients(self):
  import copy
  from transformers import BertConfig
  from transformers.models.bert.modeling_bert import BertLayer
  torch.manual_seed(62);cfg=BertConfig(hidden_size=24,num_attention_heads=4,intermediate_size=48,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.);cfg._attn_implementation='sdpa'
  full=BertLayer(cfg).double();fast=copy.deepcopy(full);x=torch.randn(2,9,24,dtype=torch.float64,requires_grad=True);xx=x.detach().clone().requires_grad_()
  keep=torch.tensor([[1,1,1,1,0,0,0,0,0],[1,1,1,1,1,1,1,1,1]],dtype=torch.float64);mask=(1-keep[:,None,None,:])*torch.finfo(torch.float64).min
  a=full(x,attention_mask=mask)[0][:,:1];b=bench.cls_only_layer(fast,xx,mask);weight=torch.randn_like(a)
  self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12));(a*weight).sum().backward();(b*weight).sum().backward()
  self.assertTrue(torch.allclose(x.grad,xx.grad,atol=1e-12,rtol=1e-12))
  for left,right in zip(full.parameters(),fast.parameters()):self.assertTrue(torch.allclose(left.grad,right.grad,atol=1e-12,rtol=1e-12))
  fast.attention.self.position_embedding_type='relative_key'
  with self.assertRaises(ValueError):bench.cls_only_layer(fast,xx,mask)
 def test_sdpa_padding_and_gradients_match_eager(self):
  from transformers import BertConfig,BertModel
  torch.manual_seed(92)
  cfg=BertConfig(hidden_size=24,num_hidden_layers=12,num_attention_heads=4,intermediate_size=48,vocab_size=31,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
  cfg._attn_implementation='eager';a=BertModel(cfg,add_pooling_layer=False)
  cfg2=BertConfig(**cfg.to_dict());cfg2._attn_implementation='sdpa';b=BertModel(cfg2,add_pooling_layer=False);b.load_state_dict(a.state_dict());a.eval();b.eval()
  ids=torch.randint(1,31,(2,7));mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]])
  outa=a(input_ids=ids,attention_mask=mask).last_hidden_state;outb=b(input_ids=ids,attention_mask=mask).last_hidden_state
  self.assertTrue(torch.allclose(outa,outb,atol=2e-5,rtol=2e-5))
  weights=torch.randn_like(outa);(outa*weights).sum().backward();(outb*weights).sum().backward()
  self.assertTrue(torch.allclose(a.encoder.layer[11].attention.self.query.weight.grad,b.encoder.layer[11].attention.self.query.weight.grad,atol=2e-5,rtol=2e-5))
  # PAD trimming preserves valid token positions and masked mean semantics.
  full=a(input_ids=ids[:1],attention_mask=mask[:1]).last_hidden_state
  short=a(input_ids=ids[:1,:4],attention_mask=mask[:1,:4]).last_hidden_state
  self.assertTrue(torch.allclose(full[:,:4],short,atol=2e-5,rtol=2e-5))
 def test_signflip_10paired_exact_resolution(self):
  result=bench.paired_stats([.01]*10)
  self.assertEqual(result['signflip_p'],2/1024)
  self.assertAlmostEqual(result['mean'],.01,places=14)

if __name__=='__main__':unittest.main()
