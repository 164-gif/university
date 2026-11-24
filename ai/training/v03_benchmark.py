"""Compare batch sizes and compilation on representative v0.3 sequence lengths."""
import argparse
from pathlib import Path
from time import perf_counter

import torch

from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.training.v03_data import write_json
from ai.training.v03_experiments import device_for_cuda


def measure(config, batch_size, compiled):
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = UniversityEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),fused=True)
    runner = torch.compile(model,dynamic=True) if compiled else model
    inputs = [(torch.randint(4,config.vocab_size,(batch_size,length),device=device),
               torch.ones((batch_size,length),dtype=torch.bool,device=device))
              for length in (64,128,256)]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)

    def step(index):
        ids,mask = inputs[index % len(inputs)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=dtype):
            embeddings = runner(ids,mask)
            loss = ((embeddings @ embeddings.T).float()-torch.eye(batch_size,device=device)).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return ids.numel(), loss.detach()

    torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    for i in range(6):
        step(i)
    torch.cuda.synchronize(device)
    warmup = perf_counter()-start
    start = perf_counter()
    steps = [step(i) for i in range(12)]
    torch.cuda.synchronize(device)
    elapsed = perf_counter()-start
    return {"batch":batch_size,"compiled":compiled,"dtype":str(dtype),"warmup_seconds":warmup,
            "step_ms":elapsed/len(steps)*1000,"tokens_per_second":sum(x[0] for x in steps)/elapsed,
            "queries_per_second":batch_size*len(steps)/elapsed,
            "peak_mib":torch.cuda.max_memory_allocated(device)/2**20,
            "stable":bool(torch.stack([x[1] for x in steps]).isfinite().all())}


def run(output):
    device_for_cuda()
    config = EncoderConfig(vocab_size=16384,hidden_size=256,num_layers=6,num_heads=8,
                           embedding_size=256,intermediate_size=1024)
    cases = []
    for batch,compiled in [(16,False),(32,False),(48,False),(64,False),(96,False),(48,True)]:
        torch.cuda.empty_cache()
        try:
            result = measure(config,batch,compiled)
        except Exception as exc:
            result = {"batch":batch,"compiled":compiled,"error":f"{type(exc).__name__}: {exc}"}
        cases.append(result)
        write_json(output,{"device":torch.cuda.get_device_name(),"config":config.to_dict(),"cases":cases})
        print(result,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
