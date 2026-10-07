#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
WEB = ROOT / "web"

DOC_EXTS = {".md", ".txt"}
ASSET_EXTS = {
    ".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif",
    ".pdf", ".mp3", ".wav", ".ogg", ".m4a", ".mp4", ".webm",
}
SKIP_PARTS = {".git", ".github", ".obsidian", "_site", "web", "templates"}


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


def subject_for(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "subjects":
        return parts[1]
    return "repository"


def section_for(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "subjects":
        return parts[2]
    return "root"


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return any(part in SKIP_PARTS or part.startswith(".") for part in rel.parts)


def copy_asset(src: Path) -> None:
    rel = src.relative_to(ROOT)
    dst = OUT / "content" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(WEB / name, OUT / name)

    docs = []
    assets = []

    # Public study content. Repository-internal agent instructions stay out of the site.
    roots = [ROOT / "subjects"]
    if (ROOT / "README.md").exists():
        roots.append(ROOT / "README.md")

    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file())

    for path in sorted(paths):
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
                    "subject": subject_for(rel),
                    "section": section_for(rel),
                    "ext": suffix[1:],
                    "content": text,
                    "meta": meta,
                }
            )
        elif suffix in ASSET_EXTS:
            copy_asset(path)
            assets.append(rel.as_posix())

    manifest = {
        "version": 1,
        "repository": "164-gif/university",
        "docs": docs,
        "assets": assets,
        "stats": {"documents": len(docs), "assets": len(assets)},
    }
    (OUT / "content.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built {len(docs)} documents and {len(assets)} assets into {OUT}")


if __name__ == "__main__":
    main()
