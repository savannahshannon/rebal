import importlib.util
from pathlib import Path
import unittest
import tempfile,json,types
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
spec=importlib.util.spec_from_file_location('vision_followup',Path(__file__).parents[1]/'scripts/benchmark_vision_followup.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(90);self.blocks=nn.ModuleList([nn.Linear(192,192) for _ in range(12)]);self.norm=nn.LayerNorm(192)

class VisionFollowupTests(unittest.TestCase):
    def test_protocol_survives_phase_json_roundtrip_and_rejects_real_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cache=p/'cache';cache.mkdir();partition=p/'partition.npz';partition.write_bytes(b'fixture partition')
            (cache/'features.npy').write_bytes(b'fixture feature cache')
            (cache/'manifest.json').write_text(json.dumps({'artifacts':{'features.npy':b.sha(cache/'features.npy')}}))
            args=types.SimpleNamespace(out=p/'run',deit_cache=cache,partition=partition,epochs=15,batch_size=128,device='cpu',threads=1)
            split={'fit':np.array([0,1]),'tune':np.array([2]),'calibration':np.array([3]),'confirmation':np.array([4])}
            with patch.object(b,'load_split',return_value=(split,np.arange(5),np.arange(200))):
                first,*_=b.prepare(args);second,*_=b.prepare(args)
                self.assertEqual(first,second);self.assertEqual(first,json.loads((args.out/'protocol.json').read_text()))
                args.epochs=16
                with self.assertRaisesRegex(ValueError,'Protocol changed'):b.prepare(args)

    def test_cls_only_deit_block_preserves_values_and_all_gradients(self):
        from timm.models.vision_transformer import Block
        import copy
        torch.manual_seed(771);full=Block(dim=192,num_heads=3,qkv_bias=True).double().eval();efficient=copy.deepcopy(full)
        x=torch.randn(2,17,192,dtype=torch.float64,requires_grad=True);xx=x.detach().clone().requires_grad_(True)
        y=full(x)[:,:1];yy=b.final_cls_block(efficient,xx)
        self.assertTrue(torch.allclose(y,yy,atol=1e-10,rtol=1e-10));y.square().sum().backward();yy.square().sum().backward()
        self.assertTrue(torch.allclose(x.grad,xx.grad,atol=1e-9,rtol=1e-9))
        for p,q in zip(full.parameters(),efficient.parameters()):self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-9,rtol=1e-9))
    def test_partial_training_changes_only_last_blocks_norm_and_head(self):
        backbone=FakeBackbone();original={k:v.clone() for k,v in backbone.state_dict().items()}
        model=b.LastBlocks(backbone,3,10,train_blocks=True);before={k:v.clone() for k,v in model.state_dict().items()}
        loss=F.cross_entropy(model(torch.randn(5,4,192)),torch.tensor([0,1,2,0,1]));loss.backward();torch.optim.SGD(model.parameters(),lr=.1).step()
        self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('blocks.')))
        self.assertTrue(all(torch.equal(v,backbone.state_dict()[k]) for k,v in original.items()))
    def test_frozen_and_partial_have_identical_initial_outputs(self):
        backbone=FakeBackbone();scaler=(np.arange(192,dtype=np.float32)/192,np.ones(192,dtype=np.float32)*2)
        frozen=b.LastBlocks(backbone,3,10,scaler,False).eval();partial=b.LastBlocks(backbone,3,10,scaler,True).eval();x=torch.randn(6,4,192)
        self.assertTrue(torch.equal(frozen(x),partial(x)))
        self.assertEqual(sum(p.numel() for p in frozen.parameters() if p.requires_grad),3*193)
        frozen.train();self.assertFalse(frozen.blocks.training);self.assertTrue(frozen.head.training)
    def test_frozen_block_values_do_not_change_on_head_update(self):
        model=b.LastBlocks(FakeBackbone(),3,11,train_blocks=False);before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('head.')}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01);opt.zero_grad();model(torch.randn(4,3,192)).sum().backward();opt.step()
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
    def test_balanced_softmax_matches_explicit_probability(self):
        z=torch.tensor([[.2,.8,-.4],[.9,-.2,.4]],requires_grad=True);labels=torch.tensor([2,0]);counts=np.array([11,5,2])
        actual=b.objective(z,labels,counts,'balanced_softmax');weighted=torch.exp(z)*torch.tensor(counts)
        expected=-torch.log(weighted[torch.arange(2),labels]/weighted.sum(1)).mean()
        self.assertTrue(torch.allclose(actual,expected));actual.backward();self.assertTrue(torch.isfinite(z.grad).all())
    def test_metrics_use_semantic_frequency_order(self):
        labels=np.arange(6);logits=np.full((6,6),-2.);logits[np.arange(6),labels]=2;logits[5,0]=3
        order=np.array([4,5,2,3,0,1]);r=b.metric(logits,labels,order)
        self.assertAlmostEqual(r['accuracy'],5/6);self.assertEqual(r['head_f1'],.5);self.assertEqual(r['mid_f1'],1)
        self.assertEqual(r['per_class_support'],[1]*6)
    def test_uniform_prior_calibration_preserves_all_predictions(self):
        labels=np.arange(6);z=np.eye(6);tau,curve=b.calibration(z,labels,np.ones(6)/6,np.arange(6))
        self.assertEqual(tau,0.);self.assertTrue(all(r['macro_f1']==1 for r in curve))
    def test_missing_prefix_row_fails_instead_of_negative_index(self):
        with self.assertRaisesRegex(ValueError,'Missing cache row'):b.batch(np.ones((3,2)),np.array([0,-1,2]),np.array([1]),None,'cpu')
    def test_nonfinite_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonfinite'):b.metric(np.full((3,3),np.nan),np.arange(3),np.arange(3))
        with self.assertRaises(ValueError):b.metric(np.ones((3,2)),np.arange(3),np.arange(3))
    def test_progress_rejects_gaps_and_false_completion(self):
        b.verify_progress(dict(completed=3,blocks=[dict(start=0,end=3)]),5)
        for p in [dict(completed=4,blocks=[dict(start=0,end=3)]),dict(completed=5,blocks=[dict(start=1,end=5)]),dict(completed=7,blocks=[])]:
            with self.assertRaises(ValueError):b.verify_progress(p,5)
    def test_completed_receipt_rejects_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'checkpoint.pt').write_bytes(b'checkpoint')
            identity=dict(name='test');history=[dict(epoch=1,optimizer_updates=3,tuning_macro_f1=.4),dict(epoch=2,optimizer_updates=6,tuning_macro_f1=.5)]
            (p/'history.json').write_text(json.dumps(history));r=dict(identity=identity,epochs=2,optimizer_updates=6,checkpoint_sha256=b.sha(p/'checkpoint.pt'),best_epoch=2)
            (p/'training_complete.json').write_text(json.dumps(r));b.validate_training_receipt(types.SimpleNamespace(epochs=2),p,identity,3)
            (p/'checkpoint.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):b.validate_training_receipt(types.SimpleNamespace(epochs=2),p,identity,3)
    def test_extraction_receipt_rejects_wrong_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'extraction_complete.json').write_text(json.dumps(dict(protocol='old')))
            with self.assertRaisesRegex(ValueError,'protocol mismatch'):b.validate_extraction(types.SimpleNamespace(out=p),dict(fingerprint='new'))
    def test_confirmation_gate_refuses_missing_all_model_receipt(self):
        with tempfile.TemporaryDirectory() as d,patch.object(b,'prepare',return_value=({}, {}, None,None)),patch.object(b,'validate_extraction'):
            with self.assertRaisesRegex(ValueError,'All final training'):b.evaluate(types.SimpleNamespace(out=Path(d)))
    def test_resume_rejects_nonfinite_or_invalid_checkpoint_state(self):
        model=b.FrozenHead(2,3,10);state={k:v.clone() for k,v in model.state_dict().items()};identity=dict(name='test');initial=b.extractor.state_hash(model)
        cp=dict(identity=identity,initial=initial,history=[dict(epoch=1,optimizer_updates=3)],completed_epochs=1,best=.5,seconds=1.,model=state,best_state={k:v.clone() for k,v in state.items()})
        b.validate_checkpoint(cp,model,identity,initial,2,3)
        cp['best_state']['head.weight'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'state shape'):b.validate_checkpoint(cp,model,identity,initial,2,3)
    def test_aggregate_refuses_duplicate_seed_grid(self):
        with tempfile.TemporaryDirectory() as d,patch.object(b,'prepare',return_value=(dict(fingerprint='new'),{},None,None)):
            p=Path(d);(p/'selection.json').write_text('{}');(p/'results.json').write_text(json.dumps(dict(protocol='new',selection_sha256=b.sha(p/'selection.json'),rows=[dict(arm='deit_frozen__ce',seed=10)]*60)))
            with self.assertRaisesRegex(ValueError,'duplicated'):b.aggregate(types.SimpleNamespace(out=p))
    def test_epoch_checkpoint_resume_matches_uninterrupted_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rng=np.random.default_rng(18);np.save(p/'dino_features.npy',rng.normal(size=(600,384)).astype(np.float32))
            args=types.SimpleNamespace(out=p,deit_cache=p,deit_weights=p,device='cpu',epochs=2,batch_size=128,eval_batch=256)
            split=dict(fit=np.arange(200),tune=np.arange(200,400),calibration=np.arange(400,600));y=np.tile(np.arange(200),3);order=np.arange(200);protocol=dict(fingerprint='fixture')
            with patch.object(b.extractor,'load_encoder',return_value=(FakeBackbone(),{})),patch('builtins.print'):
                full,*_=b.train(args,split,y,order,'dino_frozen','ce',10,.001,p/'full',protocol)
            with patch.object(b.extractor,'load_encoder',return_value=(FakeBackbone(),{})),patch('builtins.print',side_effect=InterruptedError('fixture interruption')):
                with self.assertRaises(InterruptedError):b.train(args,split,y,order,'dino_frozen','ce',10,.001,p/'resumed',protocol)
            with patch.object(b.extractor,'load_encoder',return_value=(FakeBackbone(),{})),patch('builtins.print'):
                resumed,*_=b.train(args,split,y,order,'dino_frozen','ce',10,.001,p/'resumed',protocol)
            self.assertTrue(all(torch.equal(v,resumed.state_dict()[k]) for k,v in full.state_dict().items()))
            self.assertEqual(json.loads((p/'resumed'/'training_complete.json').read_text())['optimizer_updates'],4)

    def test_constant_paired_null_is_finite_and_valid(self):
        self.assertEqual(b.constant_safe_t_p(np.zeros(10)),1.)
        self.assertEqual(b.constant_safe_t_p(np.full(10,.01)),0.)
        with self.assertRaises(ValueError):b.constant_safe_t_p(np.zeros(9))
        with self.assertRaises(ValueError):b.constant_safe_t_p(np.full(10,np.nan))

if __name__=='__main__':unittest.main()
