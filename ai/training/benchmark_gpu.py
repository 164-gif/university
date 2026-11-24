"""Short CUDA encoder benchmark; no corpus text is written to the report."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from ai.model.attention import SelfAttention
from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder


class ExplicitAttention(SelfAttention):
    """Pre-SDPA reference math, retained only for benchmark comparison."""
    def forward(self,x,attention_mask):
        b,n,h=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        keep=attention_mask[:,None,None,:].bool()
        logits=(q@k.transpose(-2,-1))*(self.head_dim**-.5)
        weights=torch.softmax(logits.masked_fill(~keep,-1e4),dim=-1)*keep.to(logits.dtype)
        weights=self.dropout(weights)
        out=(weights@v).transpose(1,2).reshape(b,n,h)
        return self.output(out)*attention_mask.unsqueeze(-1).to(out.dtype)


def make_model(config,attention,device):
    model=UniversityEncoder(config)
    if attention=="explicit":
        for block in model.blocks:
            original=block.attention
            replacement=ExplicitAttention(config)
            replacement.load_state_dict(original.state_dict())
            block.attention=replacement
    return model.to(device)


def benchmark_case(config,batch_size,dtype,attention,device,*,compile_model=False,warmup=3,repeats=10):
    model=make_model(config,attention,device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,fused=True)
    runner=torch.compile(model) if compile_model else model
    ids=torch.randint(4,config.vocab_size,(batch_size,128),device=device)
    mask=torch.ones_like(ids)
    target=F.normalize(torch.randn(batch_size,config.embedding_size,device=device),dim=-1)
    scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=dtype,enabled=dtype is not None):
            out=runner(ids,mask)
            loss=F.mse_loss(out,target)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        return float(loss.detach())
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup): step()
    torch.cuda.synchronize(device)
    losses=[]; start=perf_counter()
    for _ in range(repeats): losses.append(step())
    torch.cuda.synchronize(device); seconds=perf_counter()-start
    return {"batch_size":batch_size,"sequence_length":128,"dtype":str(dtype or torch.float32),
        "attention":attention,"compiled":compile_model,"steps":repeats,
        "step_ms":seconds*1000/repeats,"queries_per_second":batch_size*repeats/seconds,
        "tokens_per_second":batch_size*128*repeats/seconds,"peak_vram_mib":torch.cuda.max_memory_allocated(device)/1024**2,
        "loss_first":losses[0],"loss_last":losses[-1],"loss_std":pstdev(losses),
        "stable":all(math.isfinite(x) for x in losses)}


def main(args):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; benchmark refuses CPU fallback")
    device=torch.device("cuda:0")
    config=EncoderConfig(vocab_size=args.vocab_size,max_length=512)
    rows=[]
    candidates=(("fp32",None),("fp16",torch.float16),("bf16",torch.bfloat16))
    for size in (8,16,32,48,64,96,128):
        for name,dtype in candidates:
            for attention in ("explicit","sdpa"):
                try:
                    rows.append(benchmark_case(config,size,dtype,attention,device,
                                               warmup=args.warmup,repeats=args.repeats))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rows.append({"batch_size":size,"dtype":name,"attention":attention,"status":"cuda_oom"})
                except Exception as exc:
                    rows.append({"batch_size":size,"dtype":name,"attention":attention,
                                 "status":"error","error":f"{type(exc).__name__}: {exc}"})
    compile_row=None
    try:
        eager=benchmark_case(config,16,torch.bfloat16,"sdpa",device,warmup=4,repeats=15)
        compiled=benchmark_case(config,16,torch.bfloat16,"sdpa",device,compile_model=True,warmup=4,repeats=15)
        compile_row={"eager":eager,"compiled":compiled,
                     "speedup":eager["step_ms"]/compiled["step_ms"],
                     "keep_enabled":compiled["stable"] and compiled["step_ms"]<eager["step_ms"]}
    except Exception as exc:
        compile_row={"status":"unavailable","error":f"{type(exc).__name__}: {exc}","keep_enabled":False}
    safe_limit=.75*torch.cuda.get_device_properties(device).total_memory/1024**2
    valid=[r for r in rows if r.get("stable") and r.get("dtype")=="torch.bfloat16" and
           r.get("attention")=="sdpa" and r.get("peak_vram_mib",math.inf)<=safe_limit]
    best=max(valid,key=lambda r:r["queries_per_second"]) if valid else None
    report={"device":torch.cuda.get_device_name(device),"device_properties":{"memory_total_mib":torch.cuda.get_device_properties(device).total_memory/1024**2,
           "bf16_supported":torch.cuda.is_bf16_supported()},"config":config.to_dict(),
           "benchmark_sequence_length":128,"results":rows,"compile_comparison":compile_row,
           "recommended_bf16_sdpa":best,"recommended_compile":bool(compile_row and compile_row.get("keep_enabled"))}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(report,indent=2)+"\n"); tmp.replace(path)
    print(json.dumps({"output":str(path),"best":best,"compile":compile_row},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--vocab-size",type=int,default=15352)
    parser.add_argument("--warmup",type=int,default=2); parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--output",default="ai/generated/full/trained/benchmark-report.json")
    main(parser.parse_args())
