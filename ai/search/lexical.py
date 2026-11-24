"""Small dependency-free Russian lexical normalizer (not a learned tokenizer)."""
from __future__ import annotations

import re
import unicodedata

WORDS = re.compile(r"[a-zа-я]+|\d+(?:[.,]\d+)?|[∩∪⊂⊆∈∉≠≤≥±×]", re.I)
STOP = frozenset(("что такое как какой какая какие какое каков какова каковы чем чему "
                  "сколько почему где когда дай дайте назови назовите опиши объясни "
                  "расскажи пожалуйста определение формулировка является называется "
                  "в на по из за для от до у о об к ко и а или ли же бы это "
                  "его ее их он она они мы вы я ты который которая которые "
                  "которого которых при со с во").split())
VOWELS = "аеиоуыэюя"

def stem(word: str) -> str:
    """Conservative Russian suffix normalization in RV, based on Porter steps."""
    if not re.fullmatch("[а-я]+", word) or len(word) < 4:
        return word
    match = re.search(f"[{VOWELS}]", word)
    if not match:
        return word
    prefix, rv = word[:match.end()], word[match.end():]
    old = rv
    rv = re.sub(r"(?:ив|ивши|ившись|ыв|ывши|ывшись)$|(?<=[ая])(?:в|вши|вшись)$", "", rv)
    if rv == old:
        rv = re.sub(r"(?:ся|сь)$", "", rv)
        before = rv
        rv = re.sub(r"(?:ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$", "", rv)
        if rv != before:
            rv = re.sub(r"(?:ивш|ывш|ующ)$|(?<=[ая])(?:ем|нн|вш|ющ|щ)$", "", rv)
        else:
            before = rv
            rv = re.sub(r"(?:ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)$|(?<=[ая])(?:ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ны|ть|ешь|нно)$", "", rv)
            if rv == before:
                rv = re.sub(r"(?:иями|ями|ами|ией|иям|ием|иях|ев|ов|ие|ье|еи|ии|ей|ой|ий|ям|ем|ам|ом|ах|ях|ию|ью|ия|ья|а|е|и|й|о|у|ы|ь|ю|я)$", "", rv)
    rv = re.sub(r"и$", "", rv)
    result = prefix + rv
    if len(result) < 3:
        return word
    return result

def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    # Strip URL destinations and LaTeX command noise, retain numeric and
    # literal operands. Display text is never changed by this function.
    normalized = re.sub(r"https?://[^\s)>]+", " ", normalized)
    normalized = re.sub(r"\\(?:frac|vec|left|right|mathrm|text|boxed|begin|end|qquad|quad|underline|operatorname)\b", " ", normalized)
    normalized = re.sub(r"\\(cap|cup|subseteq|subsetneq|in|notin|neq|leq|geq|pm|times)\b",
                        lambda m: {"cap":"∩","cup":"∪","subseteq":"⊆","subsetneq":"⊂","in":"∈","notin":"∉","neq":"≠","leq":"≤","geq":"≥","pm":"±","times":"×"}[m[1]], normalized)
    return [stem(w) for w in WORDS.findall(normalized)
            if w not in STOP and (len(w) >= 2 or w.isdigit() or w in "∩∪⊂⊆∈∉≠≤≥±×")]

def near(a: str, b: str) -> bool:
    """One insertion/deletion/substitution or adjacent transposition."""
    if a == b or abs(len(a)-len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i, (x,y) in enumerate(zip(a,b)) if x != y]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0]+1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if len(a) > len(b):
        a,b = b,a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i+1:]
