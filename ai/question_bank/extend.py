"""Extend the committed bank without replacing its original questions or splits."""
from collections import Counter,defaultdict,deque
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from itertools import chain

from ai.question_bank.build import ROOT,example,topic,clean,REVIEW,DEFINITION,question_pool,scope
from ai.question_bank.typos import augment,load,write,norm,ident
from ai.tokenizer.bpe import BPETokenizer
from ai.search.lexical import tokens

BASE_COUNT=1000
TARGET=3000
# Definitions in prose and tables complement dash-style definitions.
PROSE=re.compile(r"(?m)^\s*(?:[-*]\s+)?(?:\*\*)?([А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z0-9 ()«»\-]{2,85}?)(?:\*\*)?\s+(?:называется|представляет собой|это|означает)\s+([^\n]{15,260})$")
TABLE=re.compile(r"(?m)^\|\s*([^|\n]{3,90})\s*\|\s*([^|\n]{15,250})\s*\|\s*$")
BULLET=re.compile(r"(?m)^\s*[-*]\s+([^|\n]{5,100})$")
HEADER=re.compile(r"(?m)^\s*\*\*([^*\n]{4,100})\*\*\s*:\s*([^\n]{15,260})$")
META=re.compile(r"без самостоятельного|не (?:содержит|дано|приводится) определени|в лекции упомянут|в записи упомянут|вопросы|навиг|источник|проверить|неразборчив|предварительн|^\s*(?:термин|понятие|определение|описание|содержание|значение|пример|да|нет|вопрос|ответ|вариант|фото)\s*$",re.I)
LIST_HEAD=re.compile(r"вид[ыов]|тип[ыов]|классификац|свойств|признак|функци|этап|состав|структур|элемент|принцип|правил|метод|способ|ресурс|требован|услови|причин|фактор",re.I)

def key(q):
    # Ignore presentation markup and punctuation at boundaries, not math.
    return norm(q.replace("«","").replace("»","").replace("**","").replace('"',""))

def phrase(s):
    s=clean(s).strip(" .:;")
    if not s or META.search(s) or REVIEW.search(s) or len(s)>200 or re.search(r"https?://|[\[\]<>|]",s):
        return None
    return s

def additional_pool(rows,existing):
    # A clean unique heading or an explicit definition provides an actual
    # answer-bearing anchor; a generic heading alone is not a gold label.
    headings=Counter((r["source"],tuple(r["headings"])) for r in rows)
    for r in rows:
        text=r["text"]
        if r["layer"] not in {"notes","concepts","exam","laboratory_physics","practice"} or REVIEW.search(text) or len(text.strip())<30:
            continue
        definitions=list(chain(DEFINITION.finditer(text),PROSE.finditer(text),HEADER.finditer(text)))
        if r["layer"]=="concepts":
            definitions+=list(TABLE.finditer(text))
        for m in definitions:
            term=topic(clean(m[1]))
            clue=phrase(m[2])
            if not term or not clue or META.search(term) or len(term)>75 or len(tokens(term))<1:
                continue
            term=term.strip("«»")
            # Different lookup directions, including conversational queries.
            for q,kind in (
                (term+" простыми словами","short_definition"),
                ("Какое определение дано термину "+term+"?","definition_lookup"),
                ("Объясните, что понимается под понятием "+term+".","definition_lookup"),
                ("Что имеется в виду, когда в лекции говорят про "+term+"?","conversational_definition"),
            ):
                yield example(q,[r],kind,"generated_weak")
            if 25<=len(clue)<=180 and "$" not in clue and "\\" not in clue and term.casefold() not in clue.casefold():
                clue=clue[0].lower()+clue[1:]
                yield example("Как называется понятие, которое описывается так: "+clue+"?",[r],"reverse_definition_extra","generated_weak")
                yield example("Какой термин подходит к определению: "+clue+"?",[r],"reverse_definition_extra","generated_weak")
        if not r["headings"] or headings[(r["source"],tuple(r["headings"]))]!=1 or r.get("split_fragment"):
            continue
        h=topic(r["headings"][-1])
        if not h or META.search(h):
            continue
        h=h.strip("«»")
        # List-oriented prompts only when an actual list is present.
        if LIST_HEAD.search(h) and len(BULLET.findall(text))+len(re.findall(r"(?m)^\s*\d+[.)]\s+\S",text))>=2:
            yield example("Что входит в перечень по теме "+h+"?",[r],"list_retrieval","generated_weak")
            yield example("Перечислите пункты, указанные в разделе "+h+".",[r],"list_retrieval","generated_weak")
        if "$$" in text or "\\frac" in text:
            yield example(h+" формулы и обозначения",[r],"formula_lookup","generated_weak")
            yield example("Как записываются математические соотношения по теме "+h+"?",[r],"formula_lookup","generated_weak")
        # Topic queries are weaker than specific definitions, and are
        # intentionally marked as such.
        yield example(h+" кратко",[r],"short_topic","generated_weak")
        yield example("Нужен материал для ответа на вопрос по теме "+h+".",[r],"conversational_topic","generated_weak")
        yield example("Какие сведения приведены в разделе "+h+"?",[r],"topic_lookup","generated_weak")
    for e in existing:
        if not e.get("positive_ids") or e.get("label")=="review":
            continue
        q=e["query"].strip().rstrip("? .")
        low=q.lower()
        # Reframe only known interrogative/imperative forms without
        # changing quantities, operators or negation.
        for prefix,templates in (
            ("что такое ",("Дайте определение понятия {}.","Что понимают под термином {}?","Объясните значение термина {}.")),
            ("как называется ",("Назовите термин: {}.","Какой термин обозначает то, что {}?")),
            ("какие бывают ",("Перечислите, какие бывают {}.","Какие виды выделяются: {}?")),
            ("как определить ",("Каким способом определяют {}?","Опишите способ определения: {}.")),
            ("как вычислить ",("Как рассчитывают {}?","Какой способ расчёта используется для {}?")),
            ("чему равно ",("Укажите значение: {}.","Какое значение имеет {}?")),
            ("дайте определение ",("Как определяется {}?","Что означает {}?")),
            ("перечислите ",("Какие пункты нужно перечислить: {}?","Что входит в перечень: {}?")),
            ("назовите ",("Какие сведения требуется назвать: {}?","Укажите: {}.")),
            ("что означает ",("Как понимать {}?","Объясните, что значит {}.")),
        ):
            if low.startswith(prefix):
                tail=q[len(prefix):]
                for template in templates:
                    query=template.format(tail)
                    yield {**e,"id":ident(e["id"]+"\0"+query),"query":query,"origin":"qa_rephrase_extra","label":"generated_weak","parent_id":e["id"]}
                break

def verify(directory,manifest):
    for name,info in manifest["files"].items():
        raw=(directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=info["sha256"]:
            raise ValueError("base file integrity mismatch: "+name)

def extend(directory,target=TARGET,seed=20261124,variants=5):
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    verify(directory,manifest)
    existing=load(directory/"existing.jsonl")
    old=load(directory/"generated.jsonl")
    noisy=load(directory/"typos.jsonl")
    rows=load(directory/"passages.jsonl")
    tokenizer=BPETokenizer.load(ROOT/"ai/weights/v0.2/tokenizer.json")
    lookup={r["id"]:r for r in rows}
    if len(old)==target:
        print("Target already present; no changes.")
        return manifest
    if len(old)!=BASE_COUNT or target<BASE_COUNT:
        raise ValueError("expected original 1000-question bank; refusing to overwrite an unexpected dataset")
    used={key(e["query"]) for e in existing+old+noisy}
    old_ids={e["id"] for e in existing+old+noisy}
    candidates={}
    # Old pool contains many valid questions not selected in the first run.
    for e in chain(question_pool(rows,existing),additional_pool(rows,existing)):
        k=key(e["query"])
        if k in used or k in candidates or e["id"] in old_ids:
            continue
        if not e.get("positive_ids") or len(e["query"])>400 or len(tokenizer.encode(e["query"]))>512:
            continue
        if not all(i in lookup and scope(e)==scope(lookup[i]) and e["split"]==lookup[i]["split"] for i in e["positive_ids"]):
            continue
        candidates[k]=e
    print("EXTENSION_POOL="+json.dumps(dict(Counter(e["origin"] for e in candidates.values())),ensure_ascii=False),flush=True)
    # Prefer diverse passages and task types rather than thousands of
    # variations of the same few terms. Round-robin by course subject.
    groups=defaultdict(list)
    seen_passages=Counter(i for e in old for i in e["positive_ids"])
    priority={"reverse_definition":0,"reverse_definition_extra":1,"definition_question":2,
              "list_retrieval":2,"formula_question":2,"formula_lookup":3,
              "qa_rewrite":3,"qa_rephrase_extra":3,"short_definition":4,"definition_lookup":4,
              "conversational_definition":5,"topic_question":6,"short_topic":6,"topic_lookup":7,
              "conversational_topic":8}
    rng=random.Random(seed)
    for e in candidates.values():
        groups[e["subject"]].append(e)
    queues={}
    for subject,items in groups.items():
        rng.shuffle(items)
        items.sort(key=lambda e:(min(seen_passages[i] for i in e["positive_ids"]),priority.get(e["origin"],8)))
        queues[subject]=deque(items)
    new=[]
    while len(new)<target-len(old) and any(queues.values()):
        for subject in sorted(queues):
            if queues[subject] and len(new)<target-len(old):
                e=queues[subject].popleft()
                k=key(e["query"])
                if k not in used and e["id"] not in old_ids:
                    used.add(k);old_ids.add(e["id"]);new.append(e)
    if len(new)!=target-BASE_COUNT:
        raise ValueError(f"insufficient unique candidates: {len(new)}")
    # Preserve original noisy data byte-for-byte at the record level.
    # New noise uses the same generator and excludes old clean/noisy queries.
    extra_noisy=augment(new,variants,seed,reserved=[e["query"] for e in existing+old+noisy])
    extra_noisy=[e for e in extra_noisy if len(tokenizer.encode(e["query"]))<=512 and key(e["query"]) not in used]
    dedup=[]
    for e in extra_noisy:
        k=key(e["query"])
        if k not in used and e["id"] not in old_ids:
            used.add(k);old_ids.add(e["id"]);dedup.append(e)
    extra_noisy=dedup
    all_generated=old+new
    all_noise=noisy+extra_noisy
    assert len(all_generated)==target
    assert len({key(e["query"]) for e in new})==len(new)
    assert not {key(e["query"]) for e in new}&{key(e["query"]) for e in existing+old+noisy}
    for e in new+extra_noisy:
        assert e["positive_ids"] and all(scope(e)==scope(lookup[i]) and e["split"]==lookup[i]["split"] for i in e["positive_ids"])
    parent={e["id"]:e for e in new}
    for e in extra_noisy:
        p=parent[e["parent_id"]]
        assert e["positive_ids"]==p["positive_ids"] and e["split"]==p["split"]
    data={"generated.jsonl":all_generated,"typos.jsonl":all_noise,
          "generated-extra-2000.jsonl":new,"typos-extra.jsonl":extra_noisy}
    for s in ("train","validation","test"):
        name=f"{s}-examples.jsonl"
        base=load(directory/name)
        data[name]=base+[e for e in new+extra_noisy if e["split"]==s]
    for name,items in data.items():
        write(directory/name,items)
        manifest["files"][name]={"rows":len(items),"sha256":hashlib.sha256((directory/name).read_bytes()).hexdigest()}
    manifest.update(requested_generated=target,generated=target,typos=len(all_noise),
        generated_by_origin=dict(Counter(e["origin"] for e in all_generated)),
        generated_by_subject=dict(Counter(e["subject"] for e in all_generated)),
        typo_operations=dict(Counter(x["type"] for e in all_noise for x in e["edits"])),
        splits={s:len(data[f"{s}-examples.jsonl"]) for s in ("train","validation","test")},
        extension={"base_generated":BASE_COUNT,"added":len(new),"added_typos":len(extra_noisy),
            "seed":seed,"by_origin":dict(Counter(e["origin"] for e in new)),
            "by_subject":dict(Counter(e["subject"] for e in new)),
            "unique_positive_passages":len({i for e in new for i in e["positive_ids"]}),
            "candidate_pool":len(candidates),"uniqueness":"normalized text; semantic equivalence not independently reviewed",
            "all_added_labels":"generated_weak"})
    (directory/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    verify(directory,manifest)
    print("QUESTION_BANK_EXTENSION="+json.dumps(manifest["extension"],ensure_ascii=False),flush=True)
    print("QUESTION_BANK_TOTAL="+json.dumps({k:manifest[k] for k in ("existing_unique","generated","typos","splits")},ensure_ascii=False),flush=True)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=ROOT/"ai/data/v03")
    p.add_argument("--target",type=int,default=TARGET)
    p.add_argument("--seed",type=int,default=20261124)
    p.add_argument("--variants",type=int,default=5)
    a=p.parse_args()
    extend(a.directory,a.target,a.seed,a.variants)
