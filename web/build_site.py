#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
WEB = ROOT / "web"
BASE_URL = "https://164-gif.github.io/university"

DOC_EXTS = {".md", ".txt"}
ASSET_EXTS = {
    ".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif",
    ".pdf", ".mp3", ".wav", ".ogg", ".m4a", ".mp4", ".webm",
}
SKIP_PARTS = {".git", ".github", ".obsidian", "_site", "web", "templates"}

DEFAULT_COHORT = "ib-2026-1kurs"
COHORTS = [
    {"id": "ib-2026-1kurs", "title": "ИБ-2026 — 1 курс", "description": "Текущие материалы репозитория"},
    {"id": "ib-2025-1kurs", "title": "ИБ-2025 — 1 курс", "description": "Пока материалов нет"},
    {"id": "ib-2026-2kurs", "title": "ИБ-2026 — 2 курс", "description": "Пока материалов нет"},
]


def strip_frontmatter(text: str) -> tuple[str, dict[str, str]]:
    if not text.startswith("---\n"):
        return text, {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return text, {}
    raw = text[4:end]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"\'')
    return text[end + 5 :], meta


def title_for(path: Path, text: str, meta: dict[str, str]) -> str:
    if meta.get("title"):
        return meta["title"]
    body, _ = strip_frontmatter(text)
    match = re.search(r"(?m)^#\s+(.+?)\s*$", body)
    if match:
        return re.sub(r"[*_`]+", "", match.group(1)).strip()
    return path.stem


def cohort_for(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "cohorts":
        return parts[1]
    if parts and parts[0] == "subjects":
        return DEFAULT_COHORT
    return "repository"


def subject_for(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "subjects":
        return parts[1]
    if len(parts) >= 4 and parts[0] == "cohorts" and parts[2] == "subjects":
        return parts[3]
    return "repository"


def section_for(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "subjects":
        return parts[2]
    if len(parts) >= 5 and parts[0] == "cohorts" and parts[2] == "subjects":
        return parts[4]
    return "root"


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return any(part in SKIP_PARTS or part.startswith(".") for part in rel.parts)


def copy_asset(src: Path) -> None:
    rel = src.relative_to(ROOT)
    dst = OUT / "content" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_ai_source(src: Path) -> None:
    rel = src.relative_to(ROOT)
    dst = OUT / "source" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_ai_text(src: Path, text: str) -> None:
    rel = src.relative_to(ROOT)
    dst = OUT / "ai" / "source" / Path(rel.as_posix() + ".txt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def source_url(path: str) -> str:
    return "source/" + "/".join(quote(part) for part in Path(path).parts)


def ai_text_url(path: str) -> str:
    encoded = "/".join(quote(part) for part in Path(path).parts)
    return f"{BASE_URL}/ai/source/{encoded}.txt"


def read_root_text(name: str) -> str:
    path = ROOT / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for name in (
        "index.html",
        "styles.css",
        "enhancements.css",
        "app.js",
        "geometry-quiz.js",
        "geometry-quiz-bootstrap.js",
        "numbering.js",
        "subject-labels.js",
        "navigation-search.js",
    ):
        shutil.copy2(WEB / name, OUT / name)

    docs = []
    assets = []

    roots = [ROOT / "subjects", ROOT / "cohorts"]
    for root_name in ("README.md", "AI_FOR_USERS.md", "AI_GUIDE.md", "AGENTS.md", "usethis.md"):
        root_file = ROOT / root_name
        if root_file.exists():
            roots.append(root_file)

    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file())

    for path in sorted(set(paths)):
        if should_skip(path):
            continue
        rel = path.relative_to(ROOT)
        suffix = path.suffix.lower()

        if suffix in DOC_EXTS:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="replace")
            _, meta = strip_frontmatter(text)
            docs.append(
                {
                    "path": rel.as_posix(),
                    "title": title_for(path, text, meta),
                    "cohort": cohort_for(rel),
                    "subject": subject_for(rel),
                    "section": section_for(rel),
                    "ext": suffix[1:],
                    "content": text,
                    "meta": meta,
                }
            )
            copy_ai_source(path)
            copy_ai_text(path, text)
        elif suffix in ASSET_EXTS:
            copy_asset(path)
            assets.append(rel.as_posix())

    study_docs = [d for d in docs if d["subject"] != "repository"]

    lines = [
        "# AI-карта University Notes",
        "",
        "Эта страница создана для AI, которые не могут прочитать JavaScript-интерфейс сайта или весь GitHub-репозиторий.",
        "Сначала прочитайте [инструкцию для AI](source/AI_GUIDE.md). Затем открывайте нужные текстовые источники ниже.",
        "",
        f"Всего учебных страниц: **{len(study_docs)}**.",
        "",
    ]
    current_group = None
    for doc in sorted(study_docs, key=lambda d: (d["cohort"], d["subject"], d["section"], d["title"].lower())):
        group = (doc["cohort"], doc["subject"])
        if group != current_group:
            if current_group is not None:
                lines.append("")
            lines.extend([f"## {doc['cohort']} / {doc['subject']}", ""])
            current_group = group
        lines.append(f"- [{doc['title']}]({source_url(doc['path'])}) — `{doc['path']}`")
    lines.extend([
        "",
        "## Служебные инструкции",
        "",
        "- [AI_GUIDE.md](source/AI_GUIDE.md)",
        "- [usethis.md](source/usethis.md)",
        "- [AGENTS.md](source/AGENTS.md)",
        "- [README.md](source/README.md)",
        "",
    ])
    (OUT / "AI_SITEMAP.md").write_text("\n".join(lines), encoding="utf-8")

    text_lines = [
        "UNIVERSITY NOTES — AI SITEMAP",
        "",
        "Direct plain-text URLs for specific materials.",
        f"Total study documents: {len(study_docs)}",
        "",
    ]
    current_group = None
    for doc in sorted(study_docs, key=lambda d: (d["cohort"], d["subject"], d["section"], d["title"].lower())):
        group = (doc["cohort"], doc["subject"])
        if group != current_group:
            if current_group is not None:
                text_lines.append("")
            text_lines.append(f"[{doc['cohort']} / {doc['subject']}]")
            current_group = group
        text_lines.append(f"{doc['title']} | {doc['section']} | {ai_text_url(doc['path'])}")
    sitemap_text = "\n".join(text_lines)
    (OUT / "AI_SITEMAP.txt").write_text(sitemap_text, encoding="utf-8")

    guide_text = read_root_text("AI_GUIDE.md")
    agents_text = read_root_text("AGENTS.md")
    usethis_text = read_root_text("usethis.md")
    readme_text = read_root_text("README.md")

    ai_entry = "\n".join([
        "UNIVERSITY NOTES — SELF-CONTAINED AI ENTRY POINT",
        "",
        "This single file contains all mandatory initialization rules and the complete material map.",
        "Do not require the user to provide any additional initialization page.",
        "Read this file completely before declaring readiness.",
        "",
        "READINESS RULE",
        "After reading this complete file, answer exactly:",
        "Готов. Ознакомился с правилами University Notes и картой материалов. ⟦UNIV-AI-READY⟧",
        "",
        "IMPORTANT:",
        "- You no longer need to open AI_SITEMAP.txt, AGENTS.md.txt, usethis.md.txt or README.md.txt to initialize.",
        "- Their required contents are embedded below in this file.",
        "- URLs in the material map are only needed later when the user asks about a specific subject or topic.",
        "- If a specific material URL cannot be opened later, say which material is unavailable; do not pretend you read it.",
        "",
        "================ AI GUIDE ================",
        guide_text,
        "",
        "================ AGENT RULES ================",
        agents_text,
        "",
        "================ REPOSITORY INSTRUCTIONS ================",
        usethis_text,
        "",
        "================ REPOSITORY README ================",
        readme_text,
        "",
        "================ COMPLETE MATERIAL MAP ================",
        sitemap_text,
        "",
        "================ END ================",
    ])
    (OUT / "AI_ENTRY.txt").write_text(ai_entry, encoding="utf-8")

    cohort_counts = {c["id"]: 0 for c in COHORTS}
    for doc in docs:
        if doc["cohort"] in cohort_counts:
            cohort_counts[doc["cohort"]] += 1

    cohorts = [dict(c, documents=cohort_counts[c["id"]]) for c in COHORTS]
    manifest = {
        "version": 6,
        "repository": "164-gif/university",
        "defaultCohort": DEFAULT_COHORT,
        "cohorts": cohorts,
        "docs": docs,
        "assets": assets,
        "stats": {"documents": len(docs), "assets": len(assets), "cohorts": len(cohorts)},
    }
    (OUT / "content.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built {len(docs)} documents, {len(assets)} assets and {len(cohorts)} cohorts into {OUT}")


if __name__ == "__main__":
    main()
