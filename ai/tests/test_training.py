import unittest
import torch

from ai.training.losses import contrastive_loss,teacher_loss
from ai.training.retrieval_eval import metrics,rrf,choose_alpha
from ai.training.train import TrainingCorpus
from ai.tokenizer.bpe import train
from ai.indexing.chunker import chunk_document
from ai.training.dataset import encoder_text
import random

class TrainingTests(unittest.TestCase):
    def test_multi_positive_and_ignored_negative(self):
        x=torch.tensor([[2.,1.,100.,0.]],requires_grad=True)
        pos=torch.tensor([[True,True,False,False]])
        w=torch.tensor([[1.,1.,0.,.25]])
        loss=contrastive_loss(x,pos,w)
        expected=torch.log1p(.25/(torch.exp(torch.tensor(2.))+torch.exp(torch.tensor(1.))))
        self.assertTrue(torch.allclose(loss,expected,atol=1e-6))
        loss.backward()
        self.assertEqual(x.grad[0,2].item(),0.)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_teacher_gradient(self):
        x=torch.tensor([[.1,.2]],requires_grad=True)
        t=torch.tensor([[3.,1.]])
        loss=teacher_loss(x,t,[(0,[0,1])])
        loss.backward()
        self.assertGreater(loss.item(),0)
        self.assertLess(x.grad[0,0].item(),0)
        self.assertEqual(teacher_loss(x,t,[]).item(),0)

    def test_rrf_and_metrics(self):
        self.assertEqual(rrf(["a","b"],["b","c"],1),["a","b"])
        self.assertEqual(rrf(["a","b"],["b","c"],0),["b","c"])
        self.assertEqual(rrf(["a","b"],["b","c"],.5)[0],"b")
        self.assertEqual(metrics([1,2,None])["top5"],2/3)
        self.assertEqual(choose_alpha({"0.0":metrics([1]),"1.0":metrics([1])}),1.)

    def test_false_negative_mask(self):
        rows=[]
        for subject,name,body in [("physics","A","Ускорение — изменение скорости."),("physics","B","Ускорение — изменение скорости."),("obzh","C","Пожар — неконтролируемое горение.")]:
            r=chunk_document(f"subjects/{subject}/notes/{name}.md",f"# {name}\n\n{body}\n")[0]
            rows.append({**r,"encoder_text":encoder_text(r)})
        tok=train([r["text"] for r in rows],vocab_size=300)
        examples=[{"id":str(i),"query":r["text"],"positive_ids":[r["id"]],
                   "cohort":r["cohort"],"subject":r["subject"],"negative_candidates":[]} for i,r in enumerate(rows)]
        corpus=TrainingCorpus(rows,examples,tok)
        ids,pos,w,_,_=corpus.prepare(examples,random.Random(1),"cpu")
        self.assertTrue(pos[0,ids.index(rows[0]["id"])])
        self.assertEqual(w[0,ids.index(rows[1]["id"])].item(),0)
        self.assertEqual(w[0,ids.index(rows[2]["id"])].item(),1)
