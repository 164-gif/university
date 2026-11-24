"""Publish the experimental ONNX search with content-addressed incremental vectors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

from ai.indexing.chunker import chunk_document
from ai.tokenizer.bpe import BPETokenizer
from ai.training.dataset import split_passage, encoder_text
from ai.search.bm25 import DEFAULT_LAYERS

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/".ai-search-cache/model"
MODEL_METADATA=ROOT/"ai/weights/v0.3/model.json"
MODEL_VERSION="v0.3"
SITE=ROOT/"_site"
OUT=SITE/"ai-search-assets"
CACHE=ROOT/".ai-search-cache/bundle"
RUNTIME=("ort.wasm.min.mjs","ort-wasm-simd-threaded.mjs","ort-wasm-simd-threaded.wasm")
FILES=("encoder.onnx","tokenizer.json","passages.json","embeddings.f32","vector-keys.json",
       *("runtime/"+n for n in RUNTIME))
DIMENSIONS=256
SCHEMA="university-ai-bundle-v3"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")

def hash_paths(paths):
    h=hashlib.sha256()
    for p in sorted(set(paths)):
        h.update(p.relative_to(ROOT).as_posix().encode("utf-8")+b"\0")
        h.update(bytes.fromhex(sha(p)))
    return h.hexdigest()

def fingerprints():
    # Same exact encoder input and tokenizer implementation + weights are
    # required to reuse vectors. Runtime/website changes only invalidate bundle.
    vector=hash_paths([WEIGHTS/"encoder.onnx",WEIGHTS/"tokenizer.json",ROOT/"ai/tokenizer/bpe.py",
                       ROOT/"ai/browser-runtime/requirements.txt"])
    code=hash_paths([ROOT/p for p in (
        "ai/publish_search.py","ai/indexing/chunker.py","ai/training/dataset.py","ai/search/bm25.py",
        "ai/search/lexical.py","ai/tokenizer/bpe.py","ai/weights/v0.3/model.json",
        "ai/browser-runtime/package.json","ai/browser-runtime/package-lock.json",
        "ai/browser-runtime/requirements.txt","ai/browser-runtime/smoke.mjs",
        "web/build_site.py","web/ai-search.html","web/ai-search.css","web/ai-search.mjs",
        "web/ai-search-tokenizer.mjs","web/ai-search-core.mjs","web/tests/ai-search.test.mjs")])
    site=read_json(SITE/"content.json")
    sources=[]
    for doc in site["docs"]:
        rel=doc["path"]
        p=ROOT/rel
        if rel.startswith(("subjects/","cohorts/")) and p.is_file() and not p.is_symlink():
            sources.append(p)
    corpus=hash_paths(sources)
    return {"schema":SCHEMA,"vector_key":vector,"code_key":code,"corpus_key":corpus}

def emit(**values):
    for name,value in values.items():
        v=str(value).lower() if isinstance(value,bool) else str(value)
        print(f"{name}={v}",flush=True)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"],"a",encoding="utf-8") as file:
                file.write(f"{name}={v}\n")

def verify_bundle(directory, expected_vector=None):
    """Verify cached bundle independently of numpy/onnxruntime."""
    m=read_json(directory/"manifest.json")
    if m.get("cache_schema")!=SCHEMA or m.get("model_version")!=MODEL_VERSION or m.get("dimensions")!=DIMENSIONS:
        raise ValueError("unsupported cached bundle")
    if expected_vector is not None and m.get("vector_key")!=expected_vector:
        raise ValueError("different vector identity")
    if set(m["files"])!=set(FILES):
        raise ValueError("incomplete bundle")
    for n in FILES:
        p=directory/n
        entry=m["files"][n]
        if p.is_symlink() or p.stat().st_size!=entry["bytes"] or sha(p)!=entry["sha256"]:
            raise ValueError("cache integrity mismatch: "+n)
    rows=read_json(directory/"passages.json")
    keys=read_json(directory/"vector-keys.json")
    if not rows or len(rows)!=m["rows"] or len(keys)!=len(rows) or (directory/"embeddings.f32").stat().st_size!=len(rows)*DIMENSIONS*4:
        raise ValueError("cached shape mismatch")
    if any(not isinstance(k,str) or len(k)!=64 for k in keys):
        raise ValueError("invalid vector keys")
    return m

def cached(finger):
    try:
        return verify_bundle(CACHE,finger["vector_key"])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print("No usable previous AI bundle:",exc,flush=True)
        return None

def restore():
    finger=fingerprints()
    previous=cached(finger)
    if previous and all(previous.get(k)==finger[k] for k in ("vector_key","code_key","corpus_key")):
        if OUT.exists():
            shutil.rmtree(OUT)
        shutil.copytree(CACHE,OUT)
        emit(ready=True,smoke_required=False)
        print("AI_CACHE_HIT: verified complete bundle",flush=True)
    else:
        emit(ready=False,smoke_required=not bool(previous and previous.get("code_key")==finger["code_key"]))

def load_vectors(finger):
    previous=cached(finger)
    if not previous:
        return {}
    keys=read_json(CACHE/"vector-keys.json")
    raw=(CACHE/"embeddings.f32").read_bytes()
    result={}
    stride=DIMENSIONS*4
    for i,key in enumerate(keys):
        data=raw[i*stride:(i+1)*stride]
        vector=struct.unpack(f"<{DIMENSIONS}f",data)
        norm=math.sqrt(sum(x*x for x in vector))
        if not math.isfinite(norm) or abs(norm-1)>2e-4:
            raise ValueError("bad cached vector")
        result[key]=data
    return result

def main():
    start=time.perf_counter()
    finger=fingerprints()
    model=WEIGHTS/"encoder.onnx"
    tokpath=WEIGHTS/"tokenizer.json"
    meta=read_json(MODEL_METADATA)
    if (meta["model_version"]!=MODEL_VERSION or meta["dimensions"]!=DIMENSIONS
            or sha(model)!=meta["onnx_sha256"] or sha(tokpath)!=meta["tokenizer_sha256"]
            or model.stat().st_size!=meta["onnx_bytes"] or meta["trained_steps"]<=0):
        raise ValueError("ONNX/tokenizer integrity mismatch")
    tokenizer=BPETokenizer.load(tokpath)
    site=read_json(SITE/"content.json")
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
    # Hash actual model input rather than line numbers; unrelated line shifts
    # and edits elsewhere in a source file do not invalidate its vectors.
    texts=[encoder_text(r) for r in rows]
    keys=[digest(t) for t in texts]
    try:
        cache=load_vectors(finger)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print("Ignoring damaged vector cache:",exc,flush=True)
        cache={}
    reused=sum(k in cache for k in keys)
    missing={}
    for key,t in zip(keys,texts):
        if key not in cache:
            missing.setdefault(key,t)
    encoded={key:tokenizer.encode(t) for key,t in missing.items()}
    order=sorted(encoded,key=lambda k:len(encoded[k]))
    if order:
        import numpy as np
        import onnxruntime as ort
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=2
        opts.inter_op_num_threads=1
        session=ort.InferenceSession(str(model),sess_options=opts,providers=["CPUExecutionProvider"])
        for offset in range(0,len(order),16):
            batch=order[offset:offset+16]
            length=max(len(encoded[k]) for k in batch)
            if length>512:
                raise ValueError("context overflow")
            ids=np.zeros((len(batch),length),dtype=np.int64)
            mask=np.zeros_like(ids)
            for j,key in enumerate(batch):
                seq=encoded[key]
                ids[j,:len(seq)]=seq
                mask[j,:len(seq)]=1
            values=session.run(["embedding"],{"input_ids":ids,"attention_mask":mask})[0]
            if values.shape!=(len(batch),DIMENSIONS) or not np.isfinite(values).all():
                raise ValueError("bad encoder output")
            if not np.allclose(np.linalg.norm(values,axis=1),1.,atol=2e-4):
                raise ValueError("encoder output not normalized")
            for key,vec in zip(batch,values):
                cache[key]=vec.astype("<f4",copy=False).tobytes()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    fields=("id","source","title","cohort","subject","layer","headings","line_start","line_end","text","document_warning")
    write_json(OUT/"passages.json",[{k:r[k] for k in fields} for r in rows])
    write_json(OUT/"vector-keys.json",keys)
    (OUT/"embeddings.f32").write_bytes(b"".join(cache[k] for k in keys))
    shutil.copy2(model,OUT/"encoder.onnx")
    shutil.copy2(tokpath,OUT/"tokenizer.json")
    runtime=OUT/"runtime"
    runtime.mkdir()
    dist=ROOT/"ai/browser-runtime/node_modules/onnxruntime-web/dist"
    for name in RUNTIME:
        shutil.copy2(dist/name,runtime/name)
    manifest={"format":"university-browser-index-v1","model_version":MODEL_VERSION,
              "trained_steps":meta["trained_steps"],"rows":len(rows),"dimensions":DIMENSIONS,
              "dtype":"float32-little-endian","max_length":512,"cache_schema":SCHEMA,
              **{k:finger[k] for k in ("vector_key","code_key","corpus_key")},
              "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "files":{n:{"sha256":sha(OUT/n),"bytes":(OUT/n).stat().st_size} for n in FILES},
              "excluded_warned":warned,"excluded_oversized":omitted,
              "reused_passages":reused,"computed_unique_vectors":len(order),
              "build_seconds":round(time.perf_counter()-start,3)}
    write_json(OUT/"manifest.json",manifest)
    verify_bundle(OUT,finger["vector_key"])
    samples=["Что такое скалярное произведение?",r"\vec a\cdot\vec b=|\vec a||\vec b|","ёж, ЁЖ, 3.14, ∫ α β 中文"]
    write_json(ROOT/"web/.ai-tokenizer-fixtures.json",{"samples":[{"text":x,"ids":tokenizer.encode(x)} for x in samples]})
    print("AI_SEARCH_BUILD="+json.dumps(manifest,ensure_ascii=False),flush=True)

def save_cache():
    verify_bundle(OUT,fingerprints()["vector_key"])
    if CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(OUT,CACHE)
    print("AI_CACHE_READY",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",nargs="?",choices=("build","plan","restore","save-cache"),default="build")
    args=p.parse_args()
    if args.action=="plan":
        emit(**fingerprints())
    elif args.action=="restore":
        restore()
    elif args.action=="save-cache":
        save_cache()
    else:
        main()
