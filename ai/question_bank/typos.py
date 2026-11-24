"""Deterministic keyboard corruption, without modifying source answers."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from collections import Counter

RUS=("йцукенгшщзхъ","фывапролджэ","ячсмитьбю")
ENG=("qwertyuiop[]","asdfghjkl;'","zxcvbnm,.")
LAYOUT=str.maketrans("".join(RUS),"".join(ENG))
PROTECTED=set("не нет без кроме только либо если когда чтобы после перед всегда никогда должен должна должны может могут нельзя можно является являются равно равна равны больше меньше выше ниже минимум максимум между внутри снаружи".split())
WORD=re.compile(r"[а-яё]{5,}",re.I)
# Skip all math, code, Markdown links, quoted names and commands.
SPECIAL=re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\(.+?\\\)|\\\[.+?\\\]|\\[A-Za-z]+|[<][^>]+[>]|[\x60][^\x60]*[\x60]|\[[^\]]+\]\([^)]*\)|\[\[.*?\]\]|«[^»]*»|\"[^\"]*\"",re.S)

def norm(s):
    return " ".join(unicodedata.normalize("NFKC",s).casefold().replace("ё","е").split()).strip(" ?.!,;:")

def ident(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

def keyboard_neighbors(letter):
    for r,row in enumerate(RUS):
        if letter in row:
            c=row.index(letter)
            return sorted({RUS[y][x] for y in range(max(0,r-1),min(len(RUS),r+2))
                for x in range(max(0,c-1),min(len(RUS[y]),c+2)) if RUS[y][x]!=letter})
    return []

def choices(text):
    protected=[m.span() for m in SPECIAL.finditer(text)]
    return [m for m in WORD.finditer(text) if not m[0].isupper()
        and m[0].lower() not in PROTECTED and not any(a<m.end() and m.start()<b for a,b in protected)
        and not (m.start()>0 and (text[m.start()-1].isalnum() or text[m.start()-1] in "_\\"))
        and not (m.end()<len(text) and (text[m.end()].isalnum() or text[m.end()]=="_"))]

def mutate(word,kind,rng):
    # Preserve first character/capitalization except layout. Never edit digits.
    if kind=="layout":
        return word.lower().translate(LAYOUT)
    i=rng.randrange(1,len(word)-1)
    if kind=="delete":
        return word[:i]+word[i+1:]
    if kind=="duplicate":
        return word[:i]+word[i]+word[i:]
    if kind=="transpose":
        if word[i]==word[i+1]:
            return word
        return word[:i]+word[i+1]+word[i]+word[i+2:]
    if kind=="keyboard":
        near=keyboard_neighbors(word[i].lower())
        return word[:i]+rng.choice(near)+word[i+1:] if near else word
    raise ValueError(kind)

def corrupt(text,kind,rng):
    words=choices(text)
    if not words:
        return text,[]
    m=rng.choice(words)
    changed=mutate(m[0],kind,rng)
    if changed==m[0]:
        return text,[]
    return text[:m.start()]+changed+text[m.end():],[{"type":kind,"before":m[0],"after":changed}]

def augment(rows,variants=5,seed=42):
    if variants<0 or variants>30:
        raise ValueError("variants must be 0..30")
    original={norm(r["query"]) for r in rows}
    seen=set(original)
    output=[]
    for row in sorted(rows,key=lambda r:r["id"]):
        if not row.get("positive_ids") or row.get("label")=="review":
            continue
        rng=random.Random(f"{seed}:{row['id']}")
        kinds=["delete","duplicate","transpose","keyboard","layout"]
        count=0
        for attempt in range(max(30,variants*20)):
            if count>=variants:
                break
            kind=kinds[(count+attempt)%len(kinds)]
            text,edits=corrupt(row["query"],kind,rng)
            if not edits:
                continue
            if count>=3 and kind!="layout" and rng.random()<.5:
                text2,edits2=corrupt(text,rng.choice(kinds[:4]),rng)
                if edits2:
                    text,edits=text2,edits+edits2
            key=norm(text)
            if not key or key in seen:
                continue
            seen.add(key)
            output.append({**row,"id":ident(row["id"]+"\0"+text),"query":text,
                "origin":"typo","parent_id":row["id"],"clean_query":row["query"],"edits":edits})
            count+=1
    return output

def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def write(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",nargs="+",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--variants",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    rows=[r for path in args.input for r in load(path)]
    noisy=augment(rows,args.variants,args.seed)
    write(args.output,noisy)
    print(json.dumps({"input":len(rows),"typos":len(noisy),"operations":Counter(e["type"] for r in noisy for e in r["edits"])},ensure_ascii=False))

if __name__=="__main__":
    main()
