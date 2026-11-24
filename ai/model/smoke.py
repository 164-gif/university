"""Architecture smoke benchmark with real BPE and source passages (random weights)."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from ai.model.config import EncoderConfig
from ai.model.encoder import UniversityEncoder
from ai.model.inputs import collate_texts
from ai.model.checkpoint import tokenizer_sha256
from ai.tokenizer.bpe import BPETokenizer
from ai.indexing.chunker import digest

def run(directory):
    directory = Path(directory)
    path = directory/"tokenizer.json"
    tokenizer = BPETokenizer.load(path)
    config = EncoderConfig(vocab_size=tokenizer.vocab_size)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = UniversityEncoder(config).eval()
    raw = (directory/"passages.jsonl").read_bytes().decode("utf-8")
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if digest(raw) != manifest["index_sha256"]:
        raise ValueError("index integrity mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    sample = next(r["text"] for r in rows if r["source"].endswith("05 - Скалярное произведение.md")
                  and r"\vec a\cdot\vec b" in r["text"])
    question = "Что такое скалярное произведение?"
    batch = collate_texts(tokenizer,[question,sample],config)
    one = collate_texts(tokenizer,[question],config)
    times = []
    with torch.inference_mode():
        output = model(**batch)
        assert torch.isfinite(output).all()
        assert torch.allclose(output.norm(dim=-1),torch.ones(2),atol=1e-5)
        assert torch.allclose(output[0],model(**one)[0],atol=1e-5)
        for _ in range(3):
            model(**one)
        for _ in range(20):
            start=perf_counter()
            model(**one)
            times.append((perf_counter()-start)*1000)
    params=model.parameter_count()
    report={"architecture":"University Encoder v0.1","weights":"random_untrained",
            "config":config.to_dict(),"parameters":params,
            "fp32_parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "int8_theoretical_parameter_bytes":params,
            "tokenizer_sha256":tokenizer_sha256(path),
            "output_shape":list(output.shape),"finite":True,"unit_norm":True,
            "padding_invariant":True,"question_tokens":one["input_ids"].shape[1],
            "median_forward_ms_cpu_1_thread":round(median(times),3),
            "torch_version":torch.__version__}
    (directory/"encoder-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,default=Path("ai/generated"))
    args=p.parse_args()
    print(json.dumps(run(args.directory),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
