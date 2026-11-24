"""Masked language pretraining with tied token embeddings; train passages only."""
import random
import torch
from torch.nn import functional as F
from ai.model.inputs import collate_texts
from ai.tokenizer.bpe import BYTE_OFFSET, MASK

def masked_batch(ids, mask, vocab_size, probability=.15):
    eligible=mask.bool() & (ids>=BYTE_OFFSET)
    chosen=(torch.rand(ids.shape,device=ids.device)<probability)&eligible
    # Guarantee at least one selected token for a nonempty content row.
    for i in range(ids.shape[0]):
        if eligible[i].any() and not chosen[i].any():
            indices=eligible[i].nonzero().flatten()
            chosen[i,indices[torch.randint(len(indices),(1,),device=ids.device)]]=True
    noisy=ids.clone()
    draw=torch.rand(ids.shape,device=ids.device)
    noisy[chosen&(draw<.8)]=MASK
    random_positions=chosen&(draw>=.8)&(draw<.9)
    noisy[random_positions]=torch.randint(BYTE_OFFSET,vocab_size,(int(random_positions.sum()),),device=ids.device)
    return noisy,chosen

def pretrain(model,tokenizer,rows,*,epochs,batch_size,device,seed=42,lr=3e-4):
    rng=random.Random(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    texts=sorted(set(r["encoder_text"] for r in rows))
    history=[]
    for epoch in range(epochs):
        order=list(texts)
        rng.shuffle(order)
        batches=[]
        for start in range(0,len(order),batch_size*12):
            pool=sorted(order[start:start+batch_size*12],key=lambda t:len(tokenizer.encode(t)))
            batches.extend(pool[i:i+batch_size] for i in range(0,len(pool),batch_size))
        rng.shuffle(batches)
        model.train()
        total=steps=0
        for b in batches:
            inp=collate_texts(tokenizer,b,model.config,device=device)
            noisy,chosen=masked_batch(inp["input_ids"],inp["attention_mask"],model.config.vocab_size)
            if not chosen.any():
                continue
            h=model.hidden_states(noisy,inp["attention_mask"])
            # Project only masked positions, not all 512 positions.
            logits=F.linear(h[chosen],model.token_embedding.weight)
            loss=F.cross_entropy(logits,inp["input_ids"][chosen])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total+=loss.item()
            steps+=1
        if not steps:
            raise RuntimeError("no MLM steps")
        record={"epoch":epoch+1,"steps":steps,"loss":total/steps}
        history.append(record)
        print("MLM="+str(record),flush=True)
    return history
