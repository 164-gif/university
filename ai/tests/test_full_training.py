import unittest
import torch
from ai.training.prepare_full import variants
from ai.training.mlm import masked_batch
from ai.training.semantic_teacher import semantic_loss
from ai.model.encoder import UniversityEncoder
from ai.model.config import EncoderConfig

class FullTrainingTests(unittest.TestCase):
    def test_variants_preserve_content(self):
        q="Почему A не равно 2?"
        v=list(variants(q))
        self.assertGreaterEqual(len(v),17)
        self.assertTrue(all(q.rstrip("?") in s for s in v))

    def test_mask_specials(self):
        ids=torch.tensor([[1,300,310,2,0]])
        mask=torch.tensor([[1,1,1,1,0]])
        noisy,chosen=masked_batch(ids,mask,400,probability=1)
        self.assertEqual(chosen.tolist(),[[False,True,True,False,False]])
        self.assertEqual(noisy[0,[0,3,4]].tolist(),[1,2,0])

    def test_hidden_and_semantic_loss(self):
        model=UniversityEncoder(EncoderConfig(vocab_size=300,hidden_size=32,num_heads=4,num_layers=1,intermediate_size=64,max_length=16))
        ids=torch.tensor([[1,100,2]])
        mask=torch.ones_like(ids)
        self.assertEqual(model.hidden_states(ids,mask).shape,(1,3,32))
        s=torch.tensor([[1.,0.,10.]],requires_grad=True)
        t=torch.tensor([[.8,.5,.9]])
        loss=semantic_loss(s,t,torch.tensor([[True,False,False]]),torch.tensor([[1.,1.,0.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(s.grad[0,2].item(),0.)
