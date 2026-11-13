"""Document-aware chunking that preserves source and section provenance."""
from __future__ import annotations
from hashlib import sha1
from pathlib import Path
import re

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

def _medication(path: Path) -> str | None:
    match = re.match(r"(?:dailymed|medlineplus)-(.+)\.md$", path.name, re.I)
    return match.group(1).replace("-", " ").title() if match else None

def _pieces(text: str, limit: int = 950, overlap: int = 140) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= limit:
            current = (current + "\n\n" + paragraph).strip()
            continue
        if current:
            out.append(current)
        # Keep sentences together where possible; fall back only for a huge paragraph.
        while len(paragraph) > limit:
            cut = paragraph.rfind(". ", 0, limit)
            cut = cut + 1 if cut > limit // 2 else limit
            out.append(paragraph[:cut].strip())
            paragraph = paragraph[max(0, cut - overlap):].strip()
        current = paragraph
    if current:
        out.append(current)
    return out

def chunk_document(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    headings = list(HEADING.finditer(text))
    spans = headings or [None]
    records: list[dict] = []
    for index, match in enumerate(spans):
        start = match.end() if match else 0
        end = headings[index + 1].start() if match and index + 1 < len(headings) else len(text)
        section = match.group(2).strip() if match else "Document overview"
        for piece_number, piece in enumerate(_pieces(text[start:end]), start=1):
            digest = sha1(f"{path.name}|{section}|{piece_number}|{piece}".encode()).hexdigest()[:12]
            records.append({
                "id": f"{path.stem}-{digest}", "source": path.name,
                "source_type": path.stem.split("-", 1)[0].title(),
                "medication": _medication(path), "section": section,
                "text": piece,
            })
    return records
