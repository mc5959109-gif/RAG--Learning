"""Kept for compatibility -- PDF conversion now lives in ingest.py.

    python pdf_to_docs.py        # convert PDFs to data/*.txt (no index rebuild)
"""

from ingest import convert_pdfs, find_pdfs  # noqa: F401  (re-exported)

if __name__ == "__main__":
    written = convert_pdfs()
    print(f"Converted {len(written)} PDF(s). "
          "Run 'python main.py build' to refresh the index.")
