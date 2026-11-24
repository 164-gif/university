import tempfile
from pathlib import Path
import unittest

import torch

from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.model.inputs import collate_texts
from ai.model.checkpoint import save_checkpoint, load_checkpoint
from ai.tokenizer.bpe import train

class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.tokenizer=train(["Скалярное произведение","Векторное произведение"]*4,vocab_size=300)
        self.config=EncoderConfig(vocab_size=self.tokenizer.vocab_size,hidden_size=32,
                                  num_heads=4,num_layers=2,intermediate_size=64,
                                  embedding_size=16,max_length=64)
        self.model=UniversityEncoder(self.config)

    def test_default_parameter_count(self):
        m=UniversityEncoder(EncoderConfig(vocab_size=16384))
        self.assertEqual(m.parameter_count(),2972800)

    def test_shape_normalization_and_padding(self):
        self.model.eval()
        one=collate_texts(self.tokenizer,["Скалярное произведение"],self.config)
        many=collate_texts(self.tokenizer,["Скалярное произведение","Векторное произведение"],self.config,pad_to=64)
        with torch.no_grad():
            a=self.model(**one)
            b=self.model(**many)
            changed=many["input_ids"].clone()
            changed[many["attention_mask"]==0]=10
            c=self.model(changed,many["attention_mask"])
        self.assertEqual(tuple(b.shape),(2,16))
        self.assertTrue(torch.isfinite(b).all())
        self.assertTrue(torch.allclose(b.norm(dim=-1),torch.ones(2),atol=1e-5))
        self.assertTrue(torch.allclose(a[0],b[0],atol=1e-5))
        self.assertTrue(torch.allclose(b,c,atol=1e-5))
        self.assertFalse(torch.allclose(b[0],b[1]))

    def test_gradient_and_optimizer(self):
        batch=collate_texts(self.tokenizer,["Скалярное произведение","Векторное произведение"],self.config)
        opt=torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        before=self.model.projection.weight.detach().clone()
        self.model.train()
        out=self.model(**batch)
        loss=out[:,0].mean()
        loss.backward()
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(self.model.blocks[0].attention.qkv.weight.grad.abs().sum()>0)
        opt.step()
        self.assertFalse(torch.equal(before,self.model.projection.weight))

    def test_all_masked_is_finite_zero(self):
        x=torch.zeros((2,8),dtype=torch.long)
        out=self.model(x,x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.abs().sum().item(),0)

    def test_input_errors(self):
        with self.assertRaises(ValueError):
            collate_texts(self.tokenizer,["x"*1000],self.config)
        with self.assertRaises(ValueError):
            collate_texts(self.tokenizer,[],self.config)
        with self.assertRaises(ValueError):
            EncoderConfig(vocab_size=300,hidden_size=30,num_heads=4)

    def test_checkpoint_roundtrip_and_hash(self):
        self.model.eval()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"tokenizer.json"
            self.tokenizer.save(path)
            checkpoint=Path(d)/"model.pt"
            save_checkpoint(checkpoint,self.model,path)
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint,path)
            loaded,steps=load_checkpoint(checkpoint,path,require_trained=False)
            self.assertEqual(steps,0)
            batch=collate_texts(self.tokenizer,["Скалярное произведение"],self.config)
            with torch.no_grad():
                self.assertTrue(torch.equal(self.model(**batch),loaded(**batch)))
            path.write_text(path.read_text()+" ",encoding="utf-8")
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint,path,require_trained=False)

if __name__=="__main__":
    unittest.main()
