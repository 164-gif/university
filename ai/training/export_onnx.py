"""Export a trained University Encoder checkpoint for local browser inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from ai.model.checkpoint import load_checkpoint,tokenizer_sha256


class ExportModel(nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,input_ids,attention_mask):
        return self.model(input_ids,attention_mask)


def run(args):
    model,steps=load_checkpoint(args.checkpoint,args.tokenizer,map_location="cpu")
    model.eval(); wrapper=ExportModel(model).eval()
    ids=torch.tensor([[1,17,21,3,2,0,0],[1,25,26,27,28,29,2]],dtype=torch.long)
    mask=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,1]],dtype=torch.long)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(args.output.suffix+".tmp")
    torch.onnx.export(wrapper,(ids,mask),tmp,dynamo=False,opset_version=18,
        input_names=["input_ids","attention_mask"],output_names=["embedding"],
        dynamic_axes={"input_ids":{0:"batch",1:"sequence"},"attention_mask":{0:"batch",1:"sequence"},
                      "embedding":{0:"batch"}},do_constant_folding=True)
    graph=onnx.load(tmp); onnx.checker.check_model(graph); tmp.replace(args.output)
    session=ort.InferenceSession(str(args.output),providers=["CPUExecutionProvider"])
    actual=session.run(["embedding"],{"input_ids":ids.numpy(),"attention_mask":mask.numpy()})[0]
    with torch.inference_mode(): expected=wrapper(ids,mask).numpy()
    max_error=float(np.max(np.abs(actual-expected)))
    if max_error>1e-4: raise RuntimeError(f"ONNX mismatch: max abs error {max_error}")
    manifest={"format":"onnx","opset":18,"checkpoint":str(args.checkpoint),
        "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "tokenizer":"tokenizer.json","tokenizer_sha256":tokenizer_sha256(args.tokenizer),
        "trained_steps":steps,"parameters":model.parameter_count(),
        "onnx_bytes":args.output.stat().st_size,"onnx_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "validation":{"runtime":"onnxruntime-cpu","max_abs_error":max_error,"dynamic_batch_and_sequence":True}}
    manifest_path=args.output.with_suffix(".manifest.json")
    temp_manifest=manifest_path.with_suffix(manifest_path.suffix+".tmp")
    temp_manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temp_manifest.replace(manifest_path)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,default=Path("ai/generated/full/trained/encoder-best.pt"))
    parser.add_argument("--tokenizer",type=Path,default=Path("ai/generated/full/tokenizer.json"))
    parser.add_argument("--output",type=Path,default=Path("ai/generated/full/trained/encoder-v0.2.onnx"))
    run(parser.parse_args())
