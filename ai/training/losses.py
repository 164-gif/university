"""Multi-positive contrastive loss and optional lexical teacher distillation."""
import torch
from torch.nn import functional as F

def contrastive_loss(logits, positive, weights):
    """weights: 0=ignore, (0,1]=negative confidence; positives always weight 1."""
    positive=positive.bool()
    if not positive.any(dim=1).all():
        raise ValueError("every query must have a positive")
    weights=torch.where(positive,torch.ones_like(weights),weights)
    numerator=torch.logsumexp(logits.masked_fill(~positive,-torch.inf),dim=1)
    denominator=torch.logsumexp(logits+weights.clamp_min(1e-12).log().masked_fill(weights<=0,-torch.inf),dim=1)
    return (denominator-numerator).mean()

def teacher_loss(logits, teacher_scores, pairs, *, temperature=2.0):
    """KL on a small, explicitly selected list of lexical teacher candidates.

    pairs is a sequence of (query index, document indices). Teacher scores
    come from BM25F, not a pretrained semantic language model.
    """
    losses=[]
    for i,columns in pairs:
        if len(columns)<2:
            continue
        t=teacher_scores[i,columns]
        s=logits[i,columns]
        losses.append(F.kl_div(F.log_softmax(s/temperature,dim=0),
                               F.softmax(t/temperature,dim=0),reduction="sum")*temperature**2)
    return torch.stack(losses).mean() if losses else logits.sum()*0
