"""Publish checked-in ONNX v0.2 with an index built from the current site."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import onnxruntime as ort

from ai.indexing.chunker import chunk_document
from ai.tokenizer.bpe import BPETokenizer
from ai.training.dataset import split_passage, encoder_text
from ai.search.bm25 import DEFAULT_LAYERS

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/"ai/weights/v0.2"
SITE=ROOT/"_site"
OUT=SITE/"ai-search-assets"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")

def main():
    model=WEIGHTS/"encoder-v0.2.onnx"
    tokpath=WEIGHTS/"tokenizer.json"
    meta=json.loads((WEIGHTS/"encoder-v0.2.manifest.json").read_text(encoding="utf-8"))
    if (sha(model)!=meta["onnx_sha256"] or sha(tokpath)!=meta["tokenizer_sha256"]
            or model.stat().st_size!=meta["onnx_bytes"] or meta["trained_steps"]<=0):
        raise ValueError("ONNX/tokenizer integrity mismatch")
    tokenizer=BPETokenizer.load(tokpath)
    site=json.loads((SITE/"content.json").read_text(encoding="utf-8"))
    rows=[]
    omitted=warned=0
    for doc in sorted(site["docs"],key=lambda d:d["path"]):
        rel=doc["path"]
        if not rel.startswith(("subjects/","cohorts/")):
            continue
        path=ROOT/rel
        if path.is_symlink() or not path.is_file():
            continue
        raw=path.read_bytes().decode("utf-8")
        lines=raw.splitlines(keepends=True)
        for row in chunk_document(rel,raw):
            if row["layer"] not in DEFAULT_LAYERS:
                continue
            if row["passage_warning"]:
                warned+=1
                continue
            pieces=list(split_passage(row,tokenizer,512))
            if not pieces:
                omitted+=1
            for p in pieces:
                if p["text"]!="".join(lines[p["line_start"]-1:p["line_end"]]):
                    raise ValueError("passage is not verbatim")
                rows.append(p)
    if not rows or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("empty or duplicate passage index")
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=2
    opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=opts,providers=["CPUExecutionProvider"])
    encoded=[tokenizer.encode(encoder_text(r)) for r in rows]
    order=sorted(range(len(rows)),key=lambda i:len(encoded[i]))
    vectors=np.empty((len(rows),128),dtype="<f4")
    for start in range(0,len(order),16):
        indices=order[start:start+16]
        length=max(len(encoded[i]) for i in indices)
        if length>512:
            raise ValueError("context overflow")
        ids=np.zeros((len(indices),length),dtype=np.int64)
        mask=np.zeros_like(ids)
        for j,i in enumerate(indices):
            seq=encoded[i]
            ids[j,:len(seq)]=seq
            mask[j,:len(seq)]=1
        values=session.run(["embedding"],{"input_ids":ids,"attention_mask":mask})[0]
        if values.shape!=(len(indices),128) or not np.isfinite(values).all():
            raise ValueError("bad encoder output")
        if not np.allclose(np.linalg.norm(values,axis=1),1.,atol=2e-4):
            raise ValueError("encoder output not normalized")
        vectors[indices]=values
    OUT.mkdir(parents=True,exist_ok=True)
    fields=("id","source","title","cohort","subject","layer","headings","line_start","line_end","text","document_warning")
    write_json(OUT/"passages.json",[{k:r[k] for k in fields} for r in rows])
    (OUT/"embeddings.f32").write_bytes(vectors.tobytes(order="C"))
    shutil.copy2(model,OUT/"encoder.onnx")
    shutil.copy2(tokpath,OUT/"tokenizer.json")
    runtime=OUT/"runtime"
    runtime.mkdir(exist_ok=True)
    dist=ROOT/"ai/browser-runtime/node_modules/onnxruntime-web/dist"
    for name in ("ort.wasm.min.mjs","ort-wasm-simd-threaded.mjs","ort-wasm-simd-threaded.wasm"):
        shutil.copy2(dist/name,runtime/name)
    names=("encoder.onnx","tokenizer.json","passages.json","embeddings.f32")
    manifest={"format":"university-browser-index-v1","model_version":"v0.2",
              "trained_steps":meta["trained_steps"],"rows":len(rows),"dimensions":128,
              "dtype":"float32-little-endian","max_length":512,
              "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "files":{n:{"sha256":sha(OUT/n),"bytes":(OUT/n).stat().st_size} for n in names},
              "excluded_warned":warned,"excluded_oversized":omitted}
    write_json(OUT/"manifest.json",manifest)
    samples=["Что такое скалярное произведение?",r"\vec a\cdot\vec b=|\vec a||\vec b|","ёж, ЁЖ, 3.14, ∫ α β 中文"]
    write_json(ROOT/"web/.ai-tokenizer-fixtures.json",{"samples":[{"text":x,"ids":tokenizer.encode(x)} for x in samples]})
    print("AI_SEARCH_BUILD="+json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
