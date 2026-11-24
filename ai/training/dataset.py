"""Weakly supervised retrieval dataset; no generated factual answers."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

from ai.indexing.chunker import digest, paragraphs
from ai.search.bm25 import BM25, DEFAULT_LAYERS
from ai.search.lexical import tokens
from ai.tokenizer.bpe import BPETokenizer

VERSION = 1
SPLITS = ("train", "validation", "test")
# Keep every version of a subject together, including other cohorts.
HOLDOUT = {"mathematical_analysis": "validation",
           "russian_statehood_fundamentals": "test"}
QA = re.compile(r"^\s*(?:\*\*)?(?:(?:Вопрос(?:\s+\d+)?|\d+)[.:)]\s*)?(.{5,350}?\?)\s*(?:\*\*)?\s*$", re.I)
ANSWER = re.compile(r"(?im)^\s*(?:\*\*)?ответ(?:\s+по[^:\n]+)?(?:\*\*)?\s*:")
DEFINITION = re.compile(r"(?m)^\s*(?:[-*]\s+)?(?:\*\*)?([А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z \-]{1,85}?)(?:\*\*)?\s+[—–]\s+(\S.*)$")
GENERIC = re.compile(r"^(?:свойства|определение|примеры?|вопросы|ответы|навигация|итоги|доказательство|введение|заключение|замечание|формулы|теория|вывод|задачи|фото|вариант)(?:\s*\d+)?$", re.I)

def normalize(s):
    return " ".join(unicodedata.normalize("NFKC",s).casefold().replace("ё","е").split())

def question_key(s):
    return normalize(s).strip(" ?.!")

def clean_heading(s):
    s = re.sub(r"^\s*\d+(?:\.\d+)*[.) —–-]+\s*", "", s)
    s = re.sub(r"\s*\((?:страниц[аы]?|стр\.|фото)\s*[^)]*\)\s*$","",s,flags=re.I)
    return s.strip(" *#")

def specific(s):
    return (3 <= len(s) <= 100 and len(tokens(s)) >= 1 and not GENERIC.fullmatch(s)
            and not any(c in s for c in "$|[]{}<>") and bool(re.search("[а-яА-ЯёЁ]",s)))

def explicit_questions(text):
    lines=text.splitlines()
    for i,line in enumerate(lines):
        m=QA.match(line)
        if not m:
            continue
        # Require a nearby explicit answer marker, before another question.
        tail=[]
        for next_line in lines[i+1:]:
            if QA.match(next_line):
                break
            tail.append(next_line)
        if ANSWER.search("\n".join(tail)):
            yield m[1].strip("* ")

def make_queries(row, unique_heading):
    text=row["text"]
    out=[]
    for q in explicit_questions(text):
        out.append((q,"explicit_qa"))
        # Only light stylistic variants; never change numbers, formulas or polarity.
        if q.startswith("Что такое "):
            term=q[len("Что такое "):].rstrip("?")
            out.append(("Дайте определение: "+term,"qa_template"))
        elif q.startswith("Каковы "):
            out.append(("Назовите: "+q[len("Каковы "):].rstrip("?"),"qa_template"))
    for m in DEFINITION.finditer(text):
        term=m[1].strip()
        if specific(term) and len(m[2]) >= 12:
            out.extend([("Что такое "+term+"?","definition_template"),
                        ("Дайте определение: "+term,"definition_template")])
    heading=clean_heading(row["headings"][-1]) if row["headings"] else ""
    if unique_heading and specific(heading) and len(text.strip()) >= 45:
        if heading.endswith("?"):
            out.append((heading,"heading_question"))
        else:
            out.extend([(heading,"heading_topic"),("Объясните: "+heading,"heading_template")])
    # Do not produce a query from arbitrary prose or from a partial formula.
    seen=set()
    for q,kind in out:
        key=question_key(q)
        if key and key not in seen and 5 <= len(q) <= 400 and tokens(q):
            seen.add(key)
            yield q,kind

def encoder_text(row):
    heading=" / ".join(row["headings"])
    return heading+"\n\n"+row["text"] if heading else row["text"]

def split_passage(row, tokenizer, max_length):
    """Keep exact contiguous source slices. Never cut an indivisible math/code block."""
    def fits(r):
        return len(tokenizer.encode(encoder_text(r))) <= max_length
    if fits(row):
        yield {**row,"parent_id":row["id"],"split_fragment":False}
        return
    # Explicit Q&A may lose qualifications if divided; leave it out instead.
    if any(explicit_questions(row["text"])):
        return
    lines=row["text"].splitlines(keepends=True)
    # Paragraph parser preserves fenced code and display equations.
    # Merge adjacent formula paragraphs with their preceding prose.
    blocks=[]
    for a,b in paragraphs(lines,0,len(lines)):
        first=lines[a].lstrip()
        if blocks and (first.startswith(("$$",r"\["))
                       or re.match(r"^(?:[-*+]|\d+[.)])\s",first)):
            blocks[-1]=(blocks[-1][0],b)
        else:
            blocks.append((a,b))
    def item(a,b):
        text="".join(lines[a:b])
        start=row["line_start"]+a
        end=row["line_start"]+b-1
        return {**row,"id":digest(f"{row['id']}\0{a}\0{b}")[:32],
                "parent_id":row["id"],"split_fragment":True,"text":text,
                "text_sha256":digest(text),"line_start":start,"line_end":end}
    pending=None
    for a,b in blocks:
        if not fits(item(a,b)):
            if pending:
                yield item(*pending)
                pending=None
            continue
        if pending and fits(item(pending[0],b)):
            pending=(pending[0],b)
        else:
            if pending:
                yield item(*pending)
            pending=(a,b)
    if pending:
        yield item(*pending)

def near_duplicate(a,b):
    if not a or not b:
        return False
    intersection=len(a & b)
    return intersection/max(1,min(len(a),len(b))) >= .8

def build_rows(records, tokenizer, *, max_length=512, negatives=3):
    if max_length < 8 or negatives < 0:
        raise ValueError("invalid dataset configuration")
    stats=Counter()
    by_split={s:[] for s in SPLITS}
    for r in records:
        if r["layer"] not in DEFAULT_LAYERS:
            stats["excluded_layer"]+=1
            continue
        if r["passage_warning"]:
            stats["excluded_passage_warning"]+=1
            continue
        pieces=list(split_passage(r,tokenizer,max_length))
        if not pieces:
            stats["excluded_overflow"]+=1
            continue
        if pieces[0]["split_fragment"]:
            stats["split_parents"]+=1
        s=HOLDOUT.get(r["subject"],"train")
        for p in pieces:
            by_split[s].append({**p,"encoder_text":encoder_text(p),"split":s})
    # Remove cross-split exact normalized body matches, even across cohorts.
    hashes=defaultdict(set)
    for s,rows in by_split.items():
        for r in rows:
            hashes[normalize(r["text"])].add(s)
    for s,rows in by_split.items():
        kept=[r for r in rows if len(hashes[normalize(r["text"])])==1]
        stats["excluded_cross_split_duplicate"]+=len(rows)-len(kept)
        by_split[s]=kept

    groups={s:defaultdict(dict) for s in SPLITS}
    for s,rows in by_split.items():
        headings=Counter((r["source"],tuple(r["headings"])) for r in rows)
        for r in rows:
            unique=headings[(r["source"],tuple(r["headings"]))]==1 and not r["split_fragment"]
            for q,kind in make_queries(r,unique):
                if len(tokenizer.encode(q))>max_length:
                    stats["excluded_long_query"]+=1
                    continue
                key=(r["cohort"],r["subject"],question_key(q))
                g=groups[s][key]
                if not g:
                    g.update(query=q,kinds=set(),positive_ids=set())
                g["kinds"].add(kind)
                g["positive_ids"].add(r["id"])
    query_splits=defaultdict(set)
    for s,gs in groups.items():
        for _,_,q in gs:
            query_splits[q].add(s)

    result={}
    for s,rows in by_split.items():
        engine=BM25(rows)
        lookup={r["id"]:r for r in rows}
        body_terms={r["id"]:set(tokens(r["text"])) for r in rows}
        examples=[]
        for (cohort,subject,key),g in sorted(groups[s].items()):
            if len(query_splits[key])>1:
                stats["excluded_cross_split_query"]+=1
                continue
            positive=sorted(g["positive_ids"])
            positive_rows=[lookup[i] for i in positive]
            pos_sources={r["source"] for r in positive_rows}
            pos_hashes={normalize(r["text"]) for r in positive_rows}
            pos_headings={normalize(r["headings"][-1]) for r in positive_rows if r["headings"]}
            candidates=[]
            if negatives:
                for hit in engine.search(g["query"],cohort=cohort,subject=subject,limit=40,fuzzy=False):
                    r=hit.passage
                    if r["id"] in positive or r["source"] in pos_sources:
                        continue
                    if normalize(r["text"]) in pos_hashes or (r["headings"] and normalize(r["headings"][-1]) in pos_headings):
                        continue
                    if any(near_duplicate(body_terms[r["id"]],body_terms[i]) for i in positive):
                        continue
                    candidates.append({"id":r["id"],"bm25_score":round(hit.score,6),"label":"candidate_unverified"})
                    if len(candidates)==negatives:
                        break
            examples.append({"id":digest(f"{s}\0{cohort}\0{subject}\0{key}")[:32],
                             "split":s,"cohort":cohort,"subject":subject,
                             "query":g["query"],"query_kinds":sorted(g["kinds"]),
                             "positive_ids":positive,"negative_candidates":candidates})
        result[s]=(rows,examples)
    return result,dict(sorted(stats.items()))

def validate(result, tokenizer, max_length):
    seen_sources={}
    seen_bodies={}
    seen_queries={}
    for s,(rows,examples) in result.items():
        lookup={r["id"]:r for r in rows}
        if len(lookup)!=len(rows):
            raise ValueError("duplicate passage IDs")
        for r in rows:
            for table,key in ((seen_sources,r["source"]),(seen_bodies,normalize(r["text"]))):
                if table.setdefault(key,s)!=s:
                    raise ValueError("cross-split source/body leakage")
            if len(tokenizer.encode(r["encoder_text"]))>max_length:
                raise ValueError("passage too long")
        for e in examples:
            if seen_queries.setdefault(question_key(e["query"]),s)!=s:
                raise ValueError("cross-split query leakage")
            pos=set(e["positive_ids"])
            neg={n["id"] for n in e["negative_candidates"]}
            if not pos or pos&neg or not pos|neg <= lookup.keys():
                raise ValueError("invalid example references")
            if len(tokenizer.encode(e["query"]))>max_length:
                raise ValueError("query too long")
            if any(lookup[i]["subject"]!=e["subject"] or lookup[i]["cohort"]!=e["cohort"] for i in pos|neg):
                raise ValueError("example scope mismatch")
