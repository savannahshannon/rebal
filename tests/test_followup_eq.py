"""Mathematical controls, heldout boundaries and resumable matched budgets."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
SPEC=importlib.util.spec_from_file_location('followup',Path(__file__).resolve().parents[1]/'scripts/followup_eq.py')
e=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(e)


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.old_threads)

    def test_importance_exact_same_class_balanced_expectation(self):
        counts=np.array([2,5,13]); values=np.arange(20,dtype=float)+1
        labels=np.repeat(np.arange(3),counts)
        natural=np.mean(values*e.importance_weights(counts,'natural')[labels])
        balanced=sum(values[labels==c].mean() for c in range(3))/3
        self.assertAlmostEqual(natural,balanced,places=6)
        self.assertTrue(np.array_equal(e.importance_weights(counts,'balanced'),np.ones(3)))
        with self.assertRaises(ValueError): e.importance_weights([2,0],'natural')

    def test_partition_disjoint_fixed_before_thinning_and_rank_inverse(self):
        y=np.repeat(np.arange(4),180); eligible=np.arange(len(y)); split=e.make_partition(y,eligible,classes=4,rho=10)
        again=e.make_partition(y,eligible,classes=4,rho=100)
        for n in ('tune','calibration','confirmation','class_rank','class_order','fit_pool'):
            self.assertTrue(np.array_equal(split[n],again[n]))
        self.assertTrue(set(again['fit'])<=set(split['fit']))
        counts=np.bincount(y[split['fit']],minlength=4)
        self.assertTrue(np.array_equal(np.argsort(-counts),split['class_order']))
        self.assertTrue(np.array_equal(split['class_rank'][split['class_order']],np.arange(4)))
        overlap={k:v.copy() for k,v in split.items()}; overlap['tune'][0]=overlap['confirmation'][0]
        with self.assertRaisesRegex(ValueError,'overlap'): e.validate_partition(overlap,y,eligible,4)

    def test_sampling_determinism_full_equal_updates_and_balanced_mass(self):
        labels=np.repeat(np.arange(3),[3,7,20]); fit=np.arange(len(labels))
        for kind in ('natural','balanced'):
            a=list(e.epoch_batches(fit,labels,3,12,0,16,kind)); b=list(e.epoch_batches(fit,labels,3,12,0,16,kind))
            self.assertEqual(len(a),2); self.assertTrue(all(len(x)==16 for x in a))
            self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
        ids=np.concatenate(list(e.epoch_batches(fit,labels,3,12,0,100000,'balanced')))
        self.assertTrue(np.max(np.abs(np.bincount(labels[ids])/len(ids)-1/3))<.01)

    def test_balanced_softmax_matches_explicit_probability(self):
        logits=torch.tensor([[.3,.1,-.2],[1.,.5,.1]],dtype=torch.float64); counts=np.array([3,8,19]); labels=torch.tensor([0,2])
        c=dict(objective='balanced_softmax',sampler='natural',onset=0.,coefficient=0.)
        actual=e.loss_for(logits,logits,labels,counts,c,0,50)
        explicit=-sum(torch.log((torch.exp(logits[i])*torch.tensor(counts))[labels[i]]/(torch.exp(logits[i])*torch.tensor(counts)).sum()) for i in range(2))/2
        self.assertAlmostEqual(actual.item(),explicit.item(),places=12)

    def test_bcl_matches_loop_value_and_gradient_with_missing_class(self):
        torch.manual_seed(44)
        f=torch.randn(4,2,5,dtype=torch.float64,requires_grad=True); proto=torch.randn(4,5,dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([0,0,0,1])  # classes2/3 represented only by their prototypes.
        actual=e.balanced_contrastive(f,labels,proto,.1)
        z=F.normalize(torch.cat((f[:,0],f[:,1],proto)),dim=1); y=torch.cat((labels,labels,torch.arange(4))); counts=torch.bincount(y)
        losses=[]
        for i in range(8):
            positive=[j for j in range(len(y)) if i!=j and y[j]==y[i]]
            terms=[]
            for j in range(len(y)):
                if i==j: continue
                divisor=counts[y[j]]-(1 if y[j]==y[i] else 0)
                terms.append((z[i]@z[j])/.1-torch.log(divisor.to(f.dtype)))
            denom=torch.logsumexp(torch.stack(terms),0)
            losses.append(-torch.stack([(z[i]@z[j])/.1-denom for j in positive]).mean())
        explicit=torch.stack(losses).mean()
        self.assertAlmostEqual(actual.item(),explicit.item(),places=12)
        ga=torch.autograd.grad(actual,(f,proto),retain_graph=True); gb=torch.autograd.grad(explicit,(f,proto))
        self.assertTrue(all(torch.allclose(a,b,rtol=0,atol=1e-11) for a,b in zip(ga,gb)))
        self.assertGreater(proto.grad.abs().sum().item() if proto.grad is not None else ga[1][2:].abs().sum().item(),0.)

    def test_equalization_moment_formula_matches_pairwise_value_and_gradient(self):
        torch.manual_seed(19); features=torch.randn(13,5,dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([0,0,0,0,1,1,2,2,2,2,2,3,4])
        actual=e.equalization(features,labels,6); reference=e.common.equalization(features,labels,6)
        self.assertAlmostEqual(actual.item(),reference.item(),places=12)
        a=torch.autograd.grad(actual,features,retain_graph=True)[0]; b=torch.autograd.grad(reference,features)[0]
        self.assertTrue(torch.allclose(a,b,rtol=0,atol=1e-12))

    def test_resume_exact_state_budget_and_identity(self):
        x=np.random.default_rng(4).normal(size=(24,6)).astype(np.float32); y=np.repeat(np.arange(3),8); fit=np.arange(len(y))
        a=argparse.Namespace(device='cpu',lr=.01,batch_size=8,epochs=3,warmup_epochs=0,checkpoint_every=1)
        configs=[dict(objective='class_balanced',sampler='balanced',onset=.2,coefficient=.05),dict(objective='bcl_dropout',sampler='natural',onset=0.,coefficient=.05,temperature=.1)]
        with tempfile.TemporaryDirectory() as directory:
            for j,c in enumerate(configs):
                root=Path(directory)/str(j); identity={'fixed':j}
                full,fi=e.train_adapter(x,y,fit,3,c,12,a,root/'full',identity)
                e.train_adapter(x,y,fit,3,c,12,a,root/'resume',identity,stop_after=1)
                resumed,ri=e.train_adapter(x,y,fit,3,c,12,a,root/'resume',identity)
                self.assertEqual(fi['final'],ri['final']); self.assertEqual(ri['optimizer_updates'],9)
                self.assertTrue(all(torch.equal(full.state_dict()[k],resumed.state_dict()[k]) for k in full.state_dict()))
                with self.assertRaisesRegex(ValueError,'identity'): e.train_adapter(x,y,fit,3,c,12,a,root/'resume',{'fixed':99})

    def test_selection_missing_training_cannot_access_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(out=Path(directory),seeds=[10,11])
            with patch.object(e,'prepare',return_value=({'fingerprint':'fixed'},None,None,None,None)),patch.object(e.nonimage,'encode') as encode:
                with self.assertRaises(FileNotFoundError): e.evaluate(args)
                encode.assert_not_called()
                self.assertFalse((args.out/'confirmation').exists())
        self.assertEqual(len(e.configurations()),32)
        names=[e.configuration_name(c) for c in e.configurations()]
        self.assertEqual(len(set(names)),32)

    def test_factorial_holm_and_exact_ten_seed_resolution(self):
        rows=[e.paired([.01]*10,'a'),e.paired([-.01]*10,'b'),e.paired([0.]*10,'c'),e.paired([.01,-.01]*5,'d')]
        self.assertEqual(rows[0]['signflip_p'],2/1024)
        e.holm(rows,'signflip_p'); self.assertEqual(rows[0]['signflip_p_holm'],8/1024)
        self.assertEqual(rows[2]['signflip_p_holm'],1.)

if __name__=='__main__': unittest.main()
