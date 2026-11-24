"""Build reproducible, source-grounded University question bank v0.3."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict,deque
import hashlib
import json
from pathlib import Path
import re
import subprocess
import random

from ai.indexing.chunker import classify,chunk_document,UNCERTAIN
from ai.training.dataset import split_passage,encoder_text,HOLDOUT,clean_heading,specific
from ai.search.bm25 import BM25,DEFAULT_LAYERS
from ai.search.lexical import tokens
from ai.tokenizer.bpe import BPETokenizer
from ai.question_bank.typos import augment,ident,norm,write

ROOT=Path(__file__).resolve().parents[2]
ELIGIBLE={"notes","concepts","exam","tests","teacher_questions","laboratory_physics","quick_cr_prep","practice"}
REVIEW=re.compile(r"предварительн|не подтвержд|неподтвержд|неизвест|неясн|неразборчив|нет (?:прямого )?ответа|не найден|требует (?:сверки|уточнения)|не указан|не предоставлен|не уверен|не раскрыт|не позволяет установить|не проверен",re.I)
ANSWER=re.compile(r"^\s*(?:>\s*)?(?:\*\*)?(?:(?:правильный|верный|предварительный)\s+)?ответ(?:\s+по[^:\n]+)?(?:\*\*)?\s*[:—]|^\s*>\s*\[!answer\]",re.I)
QSTART=re.compile(r"^\s*(?:\*\*)?(?:Вопрос\s*(?:№\s*)?\d*\s*[.:)]\s*|\d+\s*[.)]\s*)",re.I)
COMMAND=re.compile(r"^(?:что|как|какие?|какая|каков[аыо]?|сколько|чем|чему|кто|где|когда|почему|зачем|при каких|при каком|в ч[её]м|в каких|назовите|перечислите|дайте|укажите|сформулируйте|определите|запишите|объясните|выведите|докажите|найдите|вычислите|рассчитайте)\b",re.I)
QUESTION_SECTION=re.compile(r"вопрос[ыа]?|самопровер|повторени|контрольн|коллоквиум|тест",re.I)
DEFINITION=re.compile(r"(?m)^\s*(?:[-*]\s+)?(?:\*\*)?([А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z0-9 «»()\-]{2,95}?)(?:\*\*)?\s+[—–]\s+([^\n]{15,300})$")
BAD_HEADING=re.compile(r"^(?:свойств|пример|связан|навигац|источник|вопрос|ответ|вывод|доказательств|план|итог|замечан|вариант|фото|задани|упражнен|порядок|что важно|см\.|общие|теория|формул|определени|содержание|примечан)",re.I)

def clean(s):
    return re.sub(r"\s+"," ",re.sub(r"</?(?:strong|em|b|i)>","",s).replace("**","").strip()).strip()

def scope(row):
    return row["cohort"],row["subject"]

def split_for(subject):
    return HOLDOUT.get(subject,"train")

def ref(row):
    return {k:row[k] for k in ("id","source","line_start","line_end","text_sha256")}

def example(q,rows,origin,label,question_source=None):
    r=rows[0] if rows else question_source
    return {"id":ident("\0".join([r["cohort"],r["subject"],norm(q),origin])),
        "query":q,"cohort":r["cohort"],"subject":r["subject"],"split":split_for(r["subject"]),
        "origin":origin,"label":label,"positive_ids":sorted({x["id"] for x in rows}),
        "positive_sources":[ref(x) for x in rows],"question_source":question_source,
        "negative_candidates":[]}

def corpus(root,tokenizer):
    docs={}; rows=[]
    for folder in ("subjects","cohorts"):
        for p in sorted((root/folder).rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            path=p.relative_to(root).as_posix()
            info=classify(path)
            if info is None or info[2] not in ELIGIBLE:
                continue
            raw=p.read_bytes().decode("utf-8")
            docs[path]=(info,raw)
            lines=raw.splitlines(keepends=True)
            for row in chunk_document(path,raw):
                if row["passage_warning"] or row["layer"] not in DEFAULT_LAYERS:
                    continue
                for piece in split_passage(row,tokenizer,512):
                    assert piece["text"]=="".join(lines[piece["line_start"]-1:piece["line_end"]])
                    rows.append({**piece,"encoder_text":encoder_text(piece),"split":split_for(piece["subject"])})
    # Keep duplicate answer bodies out of competing subject holdouts.
    body_splits=defaultdict(set)
    for r in rows:
        body_splits[norm(r["text"])].add(r["split"])
    rows=[r for r in rows if len(body_splits[norm(r["text"])])==1]
    assert len({r["id"] for r in rows})==len(rows)
    return docs,rows

def scan_questions(path,info,raw,rows):
    """Capture Markdown question lists and explicit numbered Q/A headings."""
    cohort,subject,layer=info
    lines=raw.splitlines()
    headers=[]; section=""; fenced=False; math=False
    qa_file=bool(re.search(r"вопрос|тест|question|test|qa|коллоквиум|подготовка|variant|вариант|answer",path,re.I))
    for i,line in enumerate(lines):
        stripped=line.strip()
        if re.match(r"^(?:\x60{3,}|~{3,})",stripped):
            fenced=not fenced; continue
        if fenced: continue
        if stripped.count("$$")%2: math=not math; continue
        if math or not stripped or stripped.startswith((">","|","!","[[")):continue
        h=re.match(r"^(#{2,6})\s+(.+)",stripped)
        body=clean(h[2] if h else stripped)
        numbered=QSTART.match(body)
        if h:
            section=body
        question=clean(body[numbered.end():]) if numbered else body
        if len(question)<8 or len(question)>650 or question.startswith(("Ответ","Варианты","http")):continue
        looks=bool(COMMAND.match(question) or "?" in question or (numbered and qa_file and h))
        if not looks:continue
        if not h and not numbered and not stripped.startswith(("- ","**Вопрос")):continue
        if layer in {"notes","concepts","practice"} and not (QUESTION_SECTION.search(section) or h or "?" in question):continue
        headers.append((i,question,bool(h),bool(numbered)))
    for j,(i,q,is_heading,numbered) in enumerate(headers):
        end=headers[j+1][0] if j+1<len(headers) else len(lines)
        # A new non-question heading also ends this question block.
        for k in range(i+1,end):
            if re.match(r"^#{1,6}\s+",lines[k]):
                end=k;break
        tail=lines[i+1:end]
        marker=next((k for k,l in enumerate(tail) if ANSWER.match(l)),None)
        direct=marker is not None
        # Some teacher-control Q&A uses a numbered question heading followed
        # immediately by the answer, without an Answer marker.
        if marker is None and is_heading and numbered and "контрольные вопросы" in path.lower() and tail:
            marker=next((k for k,l in enumerate(tail) if l.strip()),None)
            direct=marker is not None
        a=i+2+(marker or 0)
        answer="\n".join(lines[a-1:end]).strip() if direct else ""
        source={"source":path,"line_start":i+1,"line_end":i+1,"cohort":cohort,"subject":subject,"text":lines[i]}
        positives=[]
        if direct and len(answer)>=8 and not REVIEW.search(answer) and not UNCERTAIN.search(answer):
            positives=[r for r in rows if r["line_start"]<=end and r["line_end"]>=a
                       and r["text"].strip() and not REVIEW.search(r["text"])]
        # Limit very broad multi-section matches; don't silently label an
        # entire chapter as an answer.
        if len(positives)>4:positives=[]
        yield example(q,positives,"repository","source_answer" if positives else "review",source)

def existing_bank(docs,rows):
    by_path=defaultdict(list)
    for r in rows:by_path[r["source"]].append(r)
    all_q=[]
    for path,(info,raw) in docs.items():
        all_q.extend(scan_questions(path,info,raw,by_path[path]))
    # Merge exact repeated questions in same course+subject; keep all
    # matching answer IDs and provenance, not duplicate training weights.
    groups={}
    for q in all_q:
        key=(*scope(q),norm(q["query"]))
        if key not in groups:
            groups[key]=q
            q["question_sources"]=[q["question_source"]]
        else:
            old=groups[key]
            old["question_sources"].append(q["question_source"])
            refs={r["id"]:r for r in old["positive_sources"]+q["positive_sources"]}
            old["positive_sources"]=list(refs.values());old["positive_ids"]=sorted(refs)
            if refs:old["label"]="source_answer"
    return list(groups.values()),len(all_q)

def authored_bank(rows):
    by_subject=defaultdict(list)
    for r in rows:by_subject[r["subject"]].append(r)
    result=[]; missing=[]
    for line in (Path(__file__).parent/"authored.tsv").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):continue
        subject,filename,anchor,questions=line.split("\t",3)
        hits=[r for r in by_subject[subject] if r["source"].endswith("/"+filename) and anchor in r["text"]]
        if not hits:
            missing.append({"subject":subject,"file":filename,"anchor":anchor,"questions":questions.split("|")})
            continue
        for q in questions.split("|"):
            # A precise literal anchor points to the actual passage, never a guessed answer.
            result.append(example(q,hits,"assistant_authored","authored"))
    return result,missing

def topic(s):
    s=clean_heading(clean(s))
    if BAD_HEADING.search(s) or not specific(s) or len(s)>90 or "?" in s or re.search(r"\b(?:вариант|фото|страница|слайд)\s*\d+",s,re.I):
        return None
    return s

def question_pool(rows,existing):
    """Generate weak labels tied to definitions/unique headings, not invented answers."""
    heading_counts=Counter((r["source"],tuple(r["headings"])) for r in rows)
    for r in rows:
        if r["layer"] not in {"notes","concepts","exam","laboratory_physics","practice"} or REVIEW.search(r["text"]) or len(r["text"].strip())<30:
            continue
        # Distinct definitions support both term -> explanation and reverse lookup.
        for m in DEFINITION.finditer(r["text"]):
            term=topic(m[1])
            if not term:continue
            term=term[0].lower()+term[1:] if not term.isupper() else term
            yield example("Что означает термин «"+term+"»?",[r],"definition_question","generated_weak")
            yield example("Как в конспекте определяется «"+term+"»?",[r],"definition_question","generated_weak")
            clue=clean(m[2]).rstrip(".;")
            if 25<=len(clue)<=180 and "$" not in clue and "\\" not in clue and not any(c in clue for c in "[]<>|") and term.casefold() not in clue.casefold() and not clue.endswith(":"):
                yield example("Как называется следующее понятие: "+clue+"?",[r],"reverse_definition","generated_weak")
        # Use the entire section only when it is one eligible passage.
        if not r["headings"] or heading_counts[(r["source"],tuple(r["headings"]))]!=1 or r.get("split_fragment"):continue
        h=topic(r["headings"][-1])
        if not h:continue
        if len(tokens(h))<1:continue
        yield example("Что нужно знать по теме «"+h+"»?",[r],"topic_question","generated_weak")
        yield example("Объясните основное содержание темы «"+h+"».",[r],"topic_question","generated_weak")
        if "$$" in r["text"] or "\\frac" in r["text"]:
            yield example("Какие формулы приведены для темы «"+h+"»?",[r],"formula_question","generated_weak")
    for e in existing:
        if not e["positive_ids"]:continue
        q=e["query"];low=q.lower()
        # Rewrite the interrogative construction without inventing facts.
        repl=None
        for prefix,target in (("что такое ","Дайте определение: "),("как называется ","Назовите, как называется "),("какие бывают ","Перечислите виды: "),("чему равно ","Укажите, чему равно "),("в чём заключается ","Объясните, в чём заключается "),("в чем заключается ","Объясните, в чём заключается "),("как определить ","Каким способом можно определить "),("как вычислить ","По какой формуле можно вычислить "),("что означает ","Объясните значение: ")):
            if low.startswith(prefix):
                repl=target+q[len(prefix):].rstrip("?")+"."
                break
        if repl:
            yield {**e,"id":ident(e["id"]+"\0"+repl),"query":repl,"origin":"qa_rewrite","label":"generated_weak","parent_id":e["id"]}

def choose_generated(rows,existing,count,seed):
    # Global exact query dedup avoids split leakage through identical wording.
    used={norm(e["query"]) for e in existing}
    authored,missing=authored_bank(rows)
    selected=[]
    for e in authored:
        k=norm(e["query"])
        if k not in used:
            selected.append(e);used.add(k)
    rng=random.Random(seed)
    buckets=defaultdict(list)
    for e in question_pool(rows,existing):
        key=norm(e["query"])
        if key in used:continue
        used.add(key)
        buckets[e["subject"]].append(e)
    queues={}
    # Round-robin subjects, first prefer new passage IDs and reverse/definition questions.
    for subject,items in buckets.items():
        rng.shuffle(items)
        items.sort(key=lambda e:{"reverse_definition":0,"definition_question":1,"formula_question":2,"qa_rewrite":3,"topic_question":4}.get(e["origin"],5))
        queues[subject]=deque(items)
    while len(selected)<count and any(queues.values()):
        for subject in sorted(queues):
            if queues[subject] and len(selected)<count:
                selected.append(queues[subject].popleft())
    if len(selected)!=count:
        raise ValueError(f"only {len(selected)} unique questions available; requested {count}")
    return selected,missing

def prepare_review(existing,rows):
    engine=BM25(rows)
    for e in existing:
        if e["positive_ids"]:continue
        hits=engine.search(e["query"],cohort=e["cohort"],subject=e["subject"],layers={"notes","concepts","laboratory_physics","practice","exam"},limit=10)
        # Candidates are explicitly not gold. Exclude question's own source.
        e["candidate_sources"]=[{**ref(h.passage),"score":round(h.score,6)} for h in hits if h.passage["source"]!=e["question_source"]["source"]][:3]
        yield e

def build(root,output,count=1000,variants=5,seed=42):
    tokenizer=BPETokenizer.load(root/"ai/weights/v0.2/tokenizer.json")
    docs,rows=corpus(root,tokenizer)
    existing,raw_count=existing_bank(docs,rows)
    # Drop cross-holdout identical questions from training labels rather than
    # letting a shared wording enter train/test.
    q_splits=defaultdict(set)
    for e in existing:q_splits[norm(e["query"])].add(e["split"])
    for e in existing:
        if len(q_splits[norm(e["query"])])>1:
            e["label"]="review";e["positive_ids"]=[];e["positive_sources"]=[]
    generated,missing=choose_generated(rows,existing,count,seed)
    reviews=list(prepare_review(existing,rows))
    clean=existing+generated
    # Token length is an actual compatibility constraint, not a claim that
    # the v0.2 vocabulary is appropriate for the next training experiment.
    for e in clean:
        if e["positive_ids"] and len(tokenizer.encode(e["query"]))>512:
            e["label"]="review";e["positive_ids"]=[];e["positive_sources"]=[]
    noisy=augment(clean,variants,seed)
    noisy=[e for e in noisy if len(tokenizer.encode(e["query"]))<=512]
    lookup={r["id"]:r for r in rows}
    # Validate provenance, scope and split of every labeled example.
    for e in clean+noisy:
        for i in e["positive_ids"]:
            r=lookup[i]
            assert scope(e)==scope(r) and e["split"]==r["split"]
    assert len(generated)==count and len({norm(e["query"]) for e in generated})==count
    assert len({e["id"] for e in clean+noisy})==len(clean+noisy)
    trainable=[e for e in clean+noisy if e["positive_ids"] and e["label"]!="review"]
    output.mkdir(parents=True,exist_ok=True)
    data={"existing.jsonl":existing,"generated.jsonl":generated,"typos.jsonl":noisy,"review.jsonl":[e for e in clean if not e["positive_ids"]],"passages.jsonl":rows}
    for s in ("train","validation","test"):
        data[f"{s}-examples.jsonl"]=[e for e in trainable if e["split"]==s]
    for name,items in data.items():write(output/name,items)
    manifest={"format":"university-question-bank-v1","source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        "seed":seed,"requested_generated":count,"existing_occurrences":raw_count,"existing_unique":len(existing),
        "existing_labeled":sum(bool(e["positive_ids"]) for e in existing),
        "existing_review":sum(not e["positive_ids"] for e in existing),
        "generated":len(generated),"generated_by_origin":dict(Counter(e["origin"] for e in generated)),
        "generated_by_subject":dict(Counter(e["subject"] for e in generated)),
        "missing_authored_anchors":missing,"typos":len(noisy),
        "typo_operations":dict(Counter(x["type"] for e in noisy for x in e["edits"])),
        "splits":{s:len(data[f"{s}-examples.jsonl"]) for s in ("train","validation","test")},
        "files":{name:{"rows":len(items),"sha256":hashlib.sha256((output/name).read_bytes()).hexdigest()} for name,items in data.items()},
        "limitations":["source labels are not independently verified","generated_weak requires review","review rows are not training positives","v0.2 holdout subjects preserved"]}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("QUESTION_BANK="+json.dumps({k:v for k,v in manifest.items() if k!="missing_authored_anchors"},ensure_ascii=False),flush=True)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,default=ROOT/"ai/data/v03")
    p.add_argument("--count",type=int,default=1000)
    p.add_argument("--variants",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    build(a.root.resolve(),a.output.resolve(),a.count,a.variants,a.seed)
