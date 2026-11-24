"""Export and measure local static inference assets for a selected v0.3 model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.transformers.float16 import convert_float_to_float16
import torch

from ai.model.inputs import collate_texts
from ai.training.export_onnx import run as export_onnx
from ai.training.v03_data import digest, load_rows, write_json
from ai.training.v03_eval import Ranking, latency
from ai.training.v03_experiments import cached_tokenizer


def encode(session, tok, texts, batch_size):
    order = sorted(range(len(texts)),key=lambda i:len(tok.encode(texts[i])))
    result = None
    config = SimpleNamespace(vocab_size=tok.vocab_size,max_length=512)
    for start in range(0,len(order),batch_size):
        idx = order[start:start+batch_size]
        batch = collate_texts(tok,[texts[i] for i in idx],config)
        values = session.run(["embedding"],{k:v.numpy() for k,v in batch.items()})[0]
        if result is None:
            result = np.empty((len(texts),values.shape[1]),np.float32)
        result[idx] = values
    return result


def cached_index(session,tok,rows,model,tokenizer,cache):
    """Reuse only vectors keyed by the exact model, tokenizer and input text."""
    cache.mkdir(parents=True,exist_ok=True)
    path = cache / f"{digest(model)}-{digest(tokenizer)}.npz"
    keys = [hashlib.sha256(row["encoder_text"].encode("utf-8")).hexdigest() for row in rows]
    saved = {}
    if path.exists():
        with np.load(path,allow_pickle=False) as content:
            saved = dict(zip(content["keys"].tolist(),content["vectors"]))
    missing = {}
    for row,key in zip(rows,keys):
        if key not in saved:
            missing[key] = row["encoder_text"]
    if missing:
        values = encode(session,tok,list(missing.values()),48)
        saved.update(zip(missing,values))
    vectors = np.stack([saved[key] for key in keys]).astype("<f4")
    if missing:
        temp = path.with_suffix(".tmp.npz")
        np.savez(temp,keys=np.array(keys),vectors=vectors)
        temp.replace(path)
    return vectors,len(missing)


def quantized(source, destination):
    """Store int8 weights and keep float activations consistent across runtimes."""
    graph = onnx.load(source)
    weights = {item.name:item for item in graph.graph.initializer}
    nodes = []
    stored = {}
    graph.graph.initializer.append(numpy_helper.from_array(np.array([-1],np.int64),"quant_scale_axis"))
    for node in graph.graph.node:
        index = 0 if node.op_type=="Gather" else 1
        if node.op_type not in ("MatMul","Gemm","Gather") or node.input[index] not in weights:
            nodes.append(node)
            continue
        name = node.input[index]
        original = weights[name]
        values = numpy_helper.to_array(original)
        if values.ndim!=2 or values.dtype!=np.float32:
            nodes.append(node)
            continue
        transposed = any(attr.name=="transB" and attr.i for attr in node.attribute)
        axis = 0 if node.op_type=="Gather" or transposed else 1
        scales = np.maximum(np.max(np.abs(values),axis=1-axis)/127,1e-10).astype(np.float32)
        if name not in stored:
            graph.graph.initializer.remove(original)
            broadcast = scales[:,None] if axis==0 else scales[None,:]
            compressed = np.rint(values/broadcast).clip(-127,127).astype(np.int8)
            graph.graph.initializer.extend([numpy_helper.from_array(compressed,name+"_int8"),
                                           numpy_helper.from_array(scales,name+"_scale")])
            stored[name] = axis
        if node.op_type=="Gather":
            output = node.output[0]
            node.input[0] = name+"_int8"
            node.output[0] = output+"_int8"
            nodes.extend([node,
                          onnx.helper.make_node("Cast",[node.output[0]],[output+"_fp32"],to=onnx.TensorProto.FLOAT),
                          onnx.helper.make_node("Gather",[name+"_scale",node.input[1]],[output+"_scale"]),
                          onnx.helper.make_node("Unsqueeze",[output+"_scale","quant_scale_axis"],[output+"_scale_row"]),
                          onnx.helper.make_node("Mul",[output+"_fp32",output+"_scale_row"],[output])])
        else:
            output = node.output[0]+"_weight_fp32"
            nodes.append(onnx.helper.make_node("DequantizeLinear",[name+"_int8",name+"_scale"],[output],axis=axis))
            node.input[1] = output
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph,destination)


def validate_export(args):
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    fp32 = output/"encoder-fp32.onnx"
    torch.set_num_threads(4)
    export_onnx(SimpleNamespace(checkpoint=args.checkpoint,tokenizer=args.tokenizer,output=fp32))
    files = {"fp32":fp32,"int8":output/"encoder-int8.onnx","fp16":output/"encoder-fp16.onnx"}
    quantized(fp32,files["int8"])
    converted = convert_float_to_float16(onnx.load(fp32),keep_io_types=True)
    onnx.save(converted,files["fp16"])
    tokenizer = cached_tokenizer(args.tokenizer)
    val_examples = load_rows(args.data,"validation-examples.jsonl")
    rows = load_rows(args.data,"passages.jsonl")
    docs = [r for r in rows if r["split"]=="validation"]
    ranker = Ranking(docs,val_examples)
    report, embeddings = {}, {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    for name,path in files.items():
        item = {"bytes":path.stat().st_size,"sha256":digest(path)}
        try:
            session = ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
            queries = encode(session,tokenizer,[r["query"] for r in val_examples],48)
            values = encode(session,tokenizer,[r["encoder_text"] for r in docs],48)
            result,_ = ranker.dense(torch.from_numpy(queries),torch.from_numpy(values))
            item["validation"] = ranker.summaries(result,detailed=True)
            times = []
            for text in [row["query"] for row in val_examples[:100]]:
                start = perf_counter()
                encode(session,tokenizer,[text],1)
                times.append((perf_counter()-start)*1000)
            item["query_latency"] = latency(times)
            embeddings[name] = (queries,values)
            item["query_max_abs_delta_vs_fp32"] = float(np.abs(embeddings["fp32"][0]-queries).max())
            item["group_mrr_delta_vs_fp32"] = (item["validation"]["group_mean"]["mrr_at5"] -
                report.get("fp32",item)["validation"]["group_mean"]["mrr_at5"])
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        report[name] = item
        print(name,item["bytes"],item.get("query_latency"),item.get("group_mrr_delta_vs_fp32"),flush=True)

    accepted = [name for name,item in report.items() if "validation" in item and item["group_mrr_delta_vs_fp32"]>=-.005]
    if not accepted:
        raise RuntimeError("no validated ONNX export")
    recommended = min(accepted,key=lambda name:report[name]["bytes"])
    picked = args.format or recommended
    if "validation" not in report[picked]:
        raise ValueError(f"requested format failed validation: {picked}")
    session = ort.InferenceSession(str(files[picked]),sess_options=options,providers=["CPUExecutionProvider"])
    vectors,computed = cached_index(session,tokenizer,rows,files[picked],args.tokenizer,output/"embedding-cache")
    source_commit = json.loads((args.data/"manifest.json").read_text(encoding="utf-8"))["source_commit"]
    browser = output/"browser"
    runtime = browser/"runtime"
    runtime.mkdir(parents=True,exist_ok=True)
    package = Path("web/node_modules/onnxruntime-web/dist")
    for asset in ("ort.wasm.min.mjs","ort-wasm-simd-threaded.mjs","ort-wasm-simd-threaded.wasm"):
        shutil.copyfile(package/asset,runtime/asset)
    shutil.copyfile(args.tokenizer,browser/"tokenizer.json")
    shutil.copyfile(files[picked],browser/"encoder.onnx")
    shutil.copyfile("web/ai-search-tokenizer.js",browser/"ai-search-tokenizer.js")
    shutil.copyfile("web/university-v03-demo.js",browser/"app.js")
    shutil.copyfile("web/university-v03-demo.html",browser/"index.html")
    write_json(browser/"passages.json",[{key:r[key] for key in ("id","text","source","line_start","line_end","subject","cohort")}
                                        for r in rows])
    (browser/"embeddings.f32").write_bytes(vectors.tobytes())
    write_json(browser/"manifest.json",{"rows":len(rows),"dimensions":vectors.shape[1],
               "source_commit":source_commit,"model_sha256":digest(browser/"encoder.onnx"),"tokenizer_sha256":digest(args.tokenizer),
               "passages_sha256":digest(browser/"passages.json"),"embeddings_sha256":digest(browser/"embeddings.f32")})
    probes = ["Что такое множество?", "Формула: $x^2 + y^2 = 1$.", "Скорость и ускорение?", "е\u0301 №² €🙂", "yt ghjcnsq"]
    expected = [{"text":text,"ids":tokenizer.encode(text)} for text in probes]
    for length in (7,13):
        ids = 1+np.arange(2*length,dtype=np.int64).reshape(2,length)%31
        reference = session.run(["embedding"],{"input_ids":ids,"attention_mask":np.ones_like(ids)})[0]
        expected.append({"length":length,"embedding":reference.flatten().tolist()})
    write_json(browser/"tokenizer-parity.json",expected)
    write_json(output/"export-report.json",{"selected":picked,"recommended":recommended,"vectors_computed":computed,
               "variants":report,"checkpoint_sha256":digest(args.checkpoint)})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("ai/data/v03"))
    parser.add_argument("--format",choices=("fp32","fp16","int8"))
    for name in ("checkpoint","tokenizer","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    validate_export(parser.parse_args())
