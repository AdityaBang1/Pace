"""Download real course documents from universities the rules were never tuned on.

The evaluation corpus that ships with the repository covers fourteen
institutions, and by the time the split was written only three of them were
still untouched — too few for a confidence claim to rest on. This script adds
documents from two more institutions (Yale Summer Session and the University of
Florida archive), both of which publish course documents openly.

Selection is deliberately mechanical and happens *before* the extractor runs:
candidates are sorted by URL, each is fetched, text is extracted, and the first
``--per-school`` documents that clear the quality gates are kept. The gates only
ask whether a document is readable at all (a PDF with a real text layer, a
plausible length); they never look at what the extractor will later say about
it, so the sample cannot be cherry-picked towards good results.

Usage::

    python scripts/fetch_eval_syllabi.py --manifest data/raw/field-eval/sources.json \
        --output data/raw/field-eval/syllabi.jsonl --per-school 6
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "PACE-eval/1.0 (academic extraction evaluation; contact: local)"
MIN_CHARS = 1200
MAX_CHARS = 20000


def _fetch(url: str, timeout: float = 40.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _pdf_text(payload: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(payload))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def _html_text(payload: bytes) -> str:
    from pace.field_corpus import clean_html

    # clean_html expects the markup a syllabus page would carry; the raw bytes
    # are decoded leniently because these pages predate consistent encoding.
    text, _ = clean_html(payload.decode("utf-8", errors="ignore"))
    return text


def extract_text(payload: bytes, url: str) -> str:
    if payload[:4] == b"%PDF" or url.lower().endswith(".pdf"):
        text = _pdf_text(payload)
    else:
        text = _html_text(payload)
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def readable(text: str) -> tuple[bool, str]:
    """Whether a document is usable at all — never whether it looks easy to parse."""
    if len(text) < MIN_CHARS:
        return False, f"too short ({len(text)} chars)"
    if len(text) > MAX_CHARS:
        return False, f"too long ({len(text)} chars)"
    printable = sum(char.isprintable() or char.isspace() for char in text)
    if printable / len(text) < 0.97:
        return False, "mostly non-printable (scanned image?)"
    letters = sum(char.isalpha() for char in text)
    if letters / len(text) < 0.45:
        return False, "mostly numbers (tables or a slide deck?)"
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 150:
        return False, f"only {len(words)} words"
    return True, ""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/raw/field-eval/sources.json")
    parser.add_argument("--output", default="data/raw/field-eval/syllabi.jsonl")
    parser.add_argument("--per-school", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=40.0)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for school, entries in manifest["schools"].items():
        taken = 0
        for entry in sorted(entries, key=lambda item: item["url"]):
            if taken >= args.per_school:
                break
            url = entry["url"]
            try:
                payload = _fetch(url, args.timeout)
                text = extract_text(payload, url)
            except Exception as error:  # noqa: BLE001 - one unreadable PDF must not stop the run
                rejected.append({"school": school, "url": url, "reason": f"fetch/parse: {error}"})
                print(f"  reject {url.rsplit('/', 1)[-1]}: {error}")
                continue
            ok, reason = readable(text)
            if not ok:
                rejected.append({"school": school, "url": url, "reason": reason})
                print(f"  reject {url.rsplit('/', 1)[-1]}: {reason}")
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            kept.append({
                "id": digest,
                "school": school,
                "url": url,
                "title": entry.get("title"),
                "chars": len(text),
                "text": text,
            })
            taken += 1
            print(f"  keep   {url.rsplit('/', 1)[-1]} ({len(text)} chars)")

    with out.open("w", encoding="utf-8") as handle:
        for document in kept:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    provenance = out.with_name(out.stem + "_provenance.json")
    provenance.write_text(json.dumps({
        "note": "Public course documents fetched for local evaluation of the extractor. "
                "Copyright remains with the instructors and institutions; this file is a "
                "local research corpus, not a redistributable dataset.",
        "manifest": str(args.manifest),
        "gates": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                  "selection": "candidates sorted by URL; first N that clear the gates"},
        "per_school": args.per_school,
        "kept": [{key: document[key] for key in ("id", "school", "url", "chars")}
                 for document in kept],
        "rejected": rejected,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nkept {len(kept)} documents -> {out}")
    print(f"rejected {len(rejected)} -> {provenance}")


if __name__ == "__main__":
    main()
