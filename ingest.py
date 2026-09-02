"""Turn PDFs into text files in data/, then (re)build the vector index.

    python ingest.py            # convert new PDFs and rebuild the index
    python ingest.py --force    # re-extract every PDF from scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import config
from vector_store import VectorStore, VectorStoreError


def find_pdfs() -> List[Path]:
    """PDFs in pdfs/ and in the project root (deduplicated, sorted)."""
    found: dict[str, Path] = {}
    for directory in config.PDF_SEARCH_DIRS:
        directory = Path(directory)
        if not directory.exists():
            continue
        for pdf in sorted(directory.glob("*.pdf")):
            found.setdefault(pdf.resolve().as_posix(), pdf)
    return sorted(found.values(), key=lambda p: p.name.lower())


def pdf_to_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise SystemExit(
            "pdfplumber is not installed. Run: pip install -r requirements.txt"
        ) from exc

    pages: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            # extract_text() returns None for image-only pages.
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
            else:
                print(f"    page {number}: no extractable text (scanned image?)")
    return "\n\n".join(pages)


def convert_pdfs(force: bool = False) -> List[Path]:
    """Write data/<name>.txt for every PDF found. Returns the files written."""
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs()
    if not pdfs:
        print(f"[ingest] no PDFs found in {', '.join(str(d) for d in config.PDF_SEARCH_DIRS)}")
        return []

    written: List[Path] = []
    for pdf in pdfs:
        target = data_dir / f"{pdf.stem}.txt"
        if target.exists() and not force and target.stat().st_mtime >= pdf.stat().st_mtime:
            print(f"[ingest] {pdf.name} -> {target.name} (already up to date)")
            continue

        print(f"[ingest] {pdf.name} -> {target.name}")
        text = pdf_to_text(pdf)
        if not text.strip():
            print(f"[ingest]   no text extracted from {pdf.name}, skipping. "
                  "A scanned PDF needs OCR first.")
            continue
        target.write_text(text, encoding="utf-8")
        written.append(target)
        print(f"[ingest]   {len(text.split())} words written")
    return written


def build_index() -> VectorStore:
    store = VectorStore()
    store.build()
    return store


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PDFs and build the index.")
    parser.add_argument("--force", action="store_true",
                        help="re-extract PDFs even if the text file looks current")
    parser.add_argument("--no-build", action="store_true",
                        help="only convert PDFs, do not rebuild the index")
    args = parser.parse_args()

    convert_pdfs(force=args.force)
    if args.no_build:
        return 0

    try:
        stats = build_index().stats()
    except VectorStoreError as exc:
        print(f"\n[ingest] {exc}")
        return 1

    print(f"\n[ingest] done - {stats['chunks']} chunks from "
          f"{len(stats['sources'])} file(s): {', '.join(stats['sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
