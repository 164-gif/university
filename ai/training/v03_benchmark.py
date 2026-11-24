"""Compare batch sizes and compilation on representative v0.3 sequence lengths."""
import argparse
from pathlib import Path
from time import perf_counter

import torch

from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.training.v03_data import load_rows, write_json
from ai.training.semantic_teacher import MODEL, REVISION
from ai.training.v03_experiments import BASE, BASE_REVISION, device_for_cuda


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


def teacher_memory(data):
    """Measure equal-sized FP32 batches independently of cached embeddings."""
    from transformers import AutoModel, AutoTokenizer

    train_rows = [row for row in load_rows(data,"passages.jsonl") if row["split"]=="train"]
    texts = sorted(train_rows,key=lambda row:-len(row["encoder_text"]))[:16]
    device = torch.device("cuda")
    results = []
    for name,revision in ((MODEL,REVISION),(BASE,BASE_REVISION)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(name,revision=revision,local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModel.from_pretrained(name,revision=revision,local_files_only=True,
                                          trust_remote_code=False,use_safetensors=True).to(device).eval()
        inputs = tokenizer(["passage: "+row["encoder_text"] for row in texts],padding=True,
                           truncation=True,max_length=512,return_tensors="pt").to(device)
        with torch.inference_mode():
            result = model(**inputs).last_hidden_state
        torch.cuda.synchronize()
        results.append({"model":name,"revision":revision,"precision":"FP32",
                        "input_shape":list(inputs["input_ids"].shape),
                        "peak_mib":torch.cuda.max_memory_allocated(device)/2**20})
        del model,inputs,result
    return results


def run(output, data):
    device_for_cuda()
    config = EncoderConfig(vocab_size=16384,hidden_size=256,num_layers=6,num_heads=8,
                           embedding_size=256,intermediate_size=1024)
    report = {"device":torch.cuda.get_device_name(),"config":config.to_dict(),
              "teacher_memory":teacher_memory(data),"cases":[]}
    cases = report["cases"]
    for batch,compiled in [(16,False),(32,False),(48,False),(64,False),(96,False),(48,True)]:
        torch.cuda.empty_cache()
        try:
            result = measure(config,batch,compiled)
        except Exception as exc:
            result = {"batch":batch,"compiled":compiled,"error":f"{type(exc).__name__}: {exc}"}
        cases.append(result)
        write_json(output,report)
        print(result,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data",type=Path,default=Path("ai/data/v03"))
    args = parser.parse_args()
    run(args.output,args.data)
