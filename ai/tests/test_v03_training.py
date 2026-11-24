import random
import unittest

import torch

from ai.training.losses import contrastive_loss
from ai.training.v03_experiments import choose_queries


class V03TrainingTests(unittest.TestCase):
    def test_multiple_positives_and_negatives_have_expected_gradients(self):
        logits = torch.tensor([[0., 0., 0., 0.]], requires_grad=True)
        positive = torch.tensor([[True, True, False, False]])
        confidence = torch.tensor([[1., 1., 1., 0.]])
        loss = contrastive_loss(logits, positive, confidence, per_query=True)
        self.assertEqual(loss.shape, (1,))
        loss.sum().backward()
        self.assertTrue(torch.all(logits.grad[0, :2] < 0))
        self.assertGreater(logits.grad[0, 2].item(), 0)
        self.assertEqual(logits.grad[0, 3].item(), 0)

    def test_sampling_keeps_one_request_per_family(self):
        clean = [{"id":str(i),"query":f"clean {i}"} for i in range(10)]
        families = {r["id"]:[{"query":f"noisy {r['id']} {j}"} for j in range(5)] for r in clean}
        for ratio, count in ((.2,2),(.4,4),(.5,5)):
            sampled = choose_queries(clean, families, ratio, random.Random(42))
            self.assertEqual(len(sampled), 10)
            self.assertEqual(sum(s.startswith("noisy") for s in sampled), count)
            for i,text in enumerate(sampled):
                self.assertEqual(text.split()[1], str(i))
