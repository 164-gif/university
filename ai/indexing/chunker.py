"""Extract verbatim, source-addressable passages from University Markdown."""
from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

SCHEMA_VERSION = 1
DEFAULT_COHORT = "ib-2026-1kurs"
LAYERS = {"notes", "concepts", "exam", "tests", "summaries", "transcripts",
          "teacher_questions", "laboratory_physics", "quick_cr_prep", "practice"}
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(?P<marker>~{3,}|\x60{3,})")
QUESTION = re.compile(r"^\s*(?:\*\*)?(?:Вопрос\s*(?:\d+)?[.:)]|(?:\d+[.)]\s+.+\?))", re.I)
ANSWER = re.compile(r"^\s*(?:\*\*)?Ответ(?:\s+по[^:]+)?(?:\*\*)?\s*:", re.I)
NAV = re.compile(r"^(?:\s*>\s*)?(?:\[\[.*\]\].*|\[!\[.*|!\[\[.*|!\[.*)\s*$")
UNCERTAIN = re.compile(r"\[неразборчиво[^\]]*\]|\[предположительно:[^\]]*\]|\[формула неразборчива\]|\[пропуск в исходнике\]|<u>|\\underline\{|неподтвержд|не подтвержд|предварительн|требу[ею]т сверки|не официальный ключ|there might be a mistake", re.I)

def classify(path: str) -> tuple[str, str, str] | None:
    p = PurePosixPath(path)
    parts = p.parts
    if p.suffix.lower() not in {".md", ".txt"} or any(x.startswith(".") for x in parts):
        return None
    if len(parts) >= 3 and parts[0] == "subjects":
        cohort, subject, tail = DEFAULT_COHORT, parts[1], parts[2:]
    elif len(parts) >= 5 and parts[0] == "cohorts" and parts[2] == "subjects":
        cohort, subject, tail = parts[1], parts[3], parts[4:]
    else:
        return None
    if any(x in {"raw", "checks", "assets", "graphs"} for x in tail):
        return None
    if p.name.upper() in {"README.MD", "RECOVERY_MANIFEST.MD", "OBSIDIAN_NOTES_RULES.MD",
                          "TEST_ANSWER_POLICY.MD", "SOURCE_INDEX.MD", "IMAGE_INDEX.MD"}:
        return None
    if subject == "laboratory_physics" and tail[0].startswith("lab_"):
        layer = "teacher_questions" if p.name == "teacher_questions.md" else "laboratory_physics"
    elif len(tail) >= 2 and tail[0] in LAYERS:
        layer = tail[0]
    else:
        return None
    return cohort, subject, layer

def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def metadata(lines: list[str]) -> tuple[int, dict[str, str]]:
    if not lines or lines[0].strip() != "---":
        return 0, {}
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            meta = {}
            for line in lines[1:i]:
                m = re.match(r"^([a-zA-Z_][\w-]*):\s*(.*?)\s*$", line)
                if m:
                    meta[m[1]] = m[2].strip("\"'")
            return i + 1, meta
    return 0, {}

def sections(lines: list[str], start: int, title: str):
    """Yield (heading hierarchy, start, end); headings in code/math are not structural."""
    stack: list[tuple[int, str]] = []
    fence = None
    math = False
    begin = start
    for i in range(start, len(lines)):
        line = lines[i]
        if fence:
            m = FENCE.match(line)
            if m and m["marker"][0] == fence[0] and len(m["marker"]) >= len(fence) and not line[m.end():].strip():
                fence = None
            continue
        if math:
            if line.count("$$") % 2:
                math = False
            continue
        m = FENCE.match(line)
        if m:
            fence = m["marker"]
            continue
        if line.count("$$") % 2:
            math = True
            continue
        h = HEADING.match(line)
        if h:
            if i > begin:
                yield [v for _, v in stack] or [title], begin, i
            level = len(h[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, h[2].strip().rstrip("#").rstrip()))
            begin = i + 1
    if begin < len(lines):
        yield [v for _, v in stack] or [title], begin, len(lines)

def paragraphs(lines: list[str], start: int, end: int):
    """Keep fenced code and display math intact, including internal blank lines."""
    begin = None
    fence = None
    math = False
    for i in range(start, end):
        line = lines[i]
        if not line.strip() and not fence and not math:
            if begin is not None:
                yield begin, i
                begin = None
            continue
        if begin is None:
            begin = i
        m = FENCE.match(line)
        if fence:
            if m and m["marker"][0] == fence[0] and len(m["marker"]) >= len(fence) and not line[m.end():].strip():
                fence = None
        elif not math and m:
            fence = m["marker"]
        elif line.count("$$") % 2:
            math = not math
    if begin is not None:
        yield begin, end

def passages(lines: list[str], start: int, end: int, target: int):
    """Keep explicit Q&A, consecutive formulas and list runs together.
    The size limit is soft: an indivisible passage may exceed it.
    """
    current = None
    qa = False
    for a, b in paragraphs(lines, start, end):
        first = lines[a].strip()
        block = "".join(lines[a:b])
        if all(NAV.match(line.strip()) for line in lines[a:b] if line.strip()):
            if current:
                yield current
            current, qa = None, False
            continue
        question = bool(QUESTION.match(first))
        if current is None:
            current, qa = (a, b), question
            continue
        x, y = current
        prev = lines[x].strip()
        continuation = (first.startswith("$$") or bool(ANSWER.match(first))
                        or (re.match(r"^(?:[-*+]|\d+[.)])\s", first)
                            and re.match(r"^(?:[-*+]|\d+[.)])\s", lines[y-1].strip()))
                        or prev.endswith(":"))
        if question or (not qa and not continuation and len("".join(lines[x:b])) > target):
            yield current
            current, qa = (a, b), question
        else:
            current = (x, b)
    if current:
        yield current

def chunk_document(path: str, source: str, target: int = 1000) -> list[dict]:
    if target < 1:
        raise ValueError("target must be positive")
    info = classify(path)
    if info is None:
        return []
    cohort, subject, layer = info
    lines = source.splitlines(keepends=True)
    start, meta = metadata(lines)
    title = meta.get("title") or PurePosixPath(path).stem
    source_sha = digest(source)
    status = meta.get("status", "")
    # Conservative document-level warning: a local warning can belong to a
    # different passage. Keep it separate from the passage-specific flag.
    document_warning = bool(UNCERTAIN.search("".join(lines[start:]))) or any(
        term in status.lower() for term in ("draft", "unverified", "proofreading"))
    out = []
    for headings, a, b in sections(lines, start, title):
        for x, y in passages(lines, a, b, target):
            text = "".join(lines[x:y])
            if not text.strip() or re.fullmatch(r"[\s*_=-]+", text):
                continue
            out.append({
                "id": digest(f"{path}\0{x+1}\0{y}\0{digest(text)}")[:32],
                "cohort": cohort, "subject": subject, "layer": layer,
                "title": title, "headings": headings, "source": path,
                "line_start": x + 1, "line_end": y,
                "source_sha256": source_sha, "text_sha256": digest(text),
                "status": status, "document_warning": document_warning,
                "passage_warning": bool(UNCERTAIN.search(text)),
                "text": text,
            })
    return out
