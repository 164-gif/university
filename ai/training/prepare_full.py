"""Build train-only BPE, strict dataset and deterministic query variants."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from ai.indexing.chunker import digest, SCHEMA_VERSION
from ai.search.bm25 import DEFAULT_LAYERS
from ai.tokenizer.bpe import train, BPETokenizer
from ai.training.dataset import HOLDOUT, normalize, question_key
from ai.training.build_dataset import run as build_dataset

# Preserve the entire original query, including negation, numbers and math.
# These are controlled intent/register augmentations, not independent paraphrases.
PREFIXES = ("Подскажите: ", "Объясните, пожалуйста: ", "Нужно разобраться: ",
            "Ответьте по конспекту: ", "Вопрос к зачёту: ", "Хочу понять: ",
            "Напомните: ", "Помогите разобраться: ", "Найдите в лекции: ",
            "Как это объясняют в конспекте: ", "Уточните: ",
            "Вопрос по теории: ", "Нужна формулировка из лекции: ",
            "Для подготовки к экзамену: ", "Где в конспекте объясняется: ",
            "Можно пояснить: ")

def variants(query):
    q=query.strip()
    yield q
    alternatives=[]
    for start in ("Что такое ", "Дайте определение: "):
        if q.startswith(start):
            term=q[len(start):].rstrip("?").strip()
            if term and len(term)<180:
                alternatives.extend(("Как определяется "+term+"?",
                                     "Что означает термин: "+term+"?",
                                     "Как в лекции объясняется понятие: "+term+"?"))
            break
    if q.startswith("Объясните: "):
        topic=q[len("Объясните: "):].strip()
        if topic:
            alternatives.extend(("Как в лекции раскрывается тема: "+topic+"?",
                                 "Что нужно знать о теме: "+topic+"?"))
    yield from alternatives
    for prefix in PREFIXES[:len(PREFIXES)-len(alternatives)]:
        yield prefix+q
    # Only change terminal punctuation, never mathematical content.
    if q.endswith("?"):
        yield q[:-1]
    else:
        yield q+"?"

def prepare(source, output, vocab_size=16384):
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():
        raise ValueError("use a separate directory for strict training")
    output.mkdir(parents=True,exist_ok=True)
    raw=(source/"passages.jsonl").read_bytes().decode("utf-8")
    manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"]!=SCHEMA_VERSION or digest(raw)!=manifest["index_sha256"]:
        raise ValueError("source index integrity mismatch")
    rows=[json.loads(x) for x in raw.splitlines() if x]
    if len(rows)!=manifest["passages"]:
        raise ValueError("source index count mismatch")
    # Fit only eligible training texts; remove exact normalized bodies also
    # present in a held-out subject before fitting the vocabulary.
    held={normalize(r["text"]) for r in rows if r["subject"] in HOLDOUT}
    unique={}
    for r in rows:
        if (r["subject"] not in HOLDOUT and r["layer"] in DEFAULT_LAYERS
                and not r["passage_warning"] and normalize(r["text"]) not in held):
            unique.setdefault(normalize(r["text"]),r["text"])
    if not unique:
        raise ValueError("empty train-only tokenizer corpus")
    tok=train(unique.values(),vocab_size=vocab_size)
    tok_path=output/"tokenizer.json"
    tok.save(tok_path)
    tok_hash=hashlib.sha256(tok_path.read_bytes()).hexdigest()
    report={"corpus_sha256":manifest["index_sha256"],"tokenizer_sha256":tok_hash,
            "fit_scope":"train_subjects_only","fit_passages":len(unique),
            "actual_vocab_size":tok.vocab_size,"excluded_holdout_subjects":sorted(HOLDOUT),
            "fit_text_sha256":digest(json.dumps(sorted(unique.values()),ensure_ascii=False))}
    (output/"tokenizer-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for name in ("passages.jsonl","manifest.json"):
        shutil.copyfile(source/name,output/name)
    dataset=output/"dataset"
    dm=build_dataset(output,dataset,max_length=512,negatives=3)
    tok=BPETokenizer.load(tok_path)
    train_examples=[json.loads(x) for x in (dataset/"train-examples.jsonl").read_text(encoding="utf-8").splitlines()]
    holdout=set()
    for s in ("validation","test"):
        holdout.update(question_key(json.loads(x)["query"]) for x in (dataset/f"{s}-examples.jsonl").read_text(encoding="utf-8").splitlines())
    manual=Path(__file__).parents[1]/"evaluation"/"paraphrases_test.json"
    holdout.update(question_key(c["q"]) for c in json.loads(manual.read_text(encoding="utf-8")))
    # A variant may belong to several base queries: do not attach conflicting
    # labels to the same normalized query in the same subject and cohort.
    owners=defaultdict(list)
    for e in train_examples:
        seen=set()
        for q in variants(e["query"]):
            key=question_key(q)
            if key in seen or key in holdout or len(tok.encode(q))>512:
                continue
            seen.add(key)
            owners[(e["cohort"],e["subject"],key)].append((e["id"],q))
    augmented=defaultdict(list)
    for key,choices in sorted(owners.items()):
        if len(choices)==1:
            ident,q=choices[0]
            augmented[ident].append(q)
    # Keep original base examples even if no unambiguous augmentation exists.
    data=[{"id":e["id"],"queries":sorted(set([e["query"],*augmented[e["id"]]]))}
          for e in train_examples]
    body="".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in data)
    (dataset/"train-variants.jsonl").write_bytes(body.encode("utf-8"))
    ar={"variant_count":sum(len(x["queries"]) for x in data),"base_queries":len(data),
        "variant_sha256":digest(body),"dataset_manifest_sha256":hashlib.sha256((dataset/"dataset-manifest.json").read_bytes()).hexdigest(),
        "augmentation":"controlled_intent_and_style_templates","tokenizer_fit_scope":"train_subjects_only"}
    (dataset/"augmentation-manifest.json").write_text(json.dumps(ar,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PREPARE="+json.dumps({**report,**ar,"splits":dm["splits"]},ensure_ascii=False),flush=True)
    return ar

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="ai/generated")
    p.add_argument("--output",default="ai/generated/full")
    p.add_argument("--vocab-size",type=int,default=16384)
    a=p.parse_args()
    prepare(a.source,a.output,a.vocab_size)
