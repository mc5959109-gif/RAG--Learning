"""FAISS-backed vector store: chunk documents, embed them, search them.

Improvements over the first version:
  * paths are absolute, so it works no matter which folder you run from
  * cosine similarity (normalised vectors + inner product) instead of raw L2,
    so scores are comparable between queries and easy to threshold
  * each chunk keeps its source file and position, so answers can cite them
  * the embedding model loads lazily -- importing this module is cheap
  * an index built with different settings is detected and rebuilt instead of
    silently returning nonsense
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

import config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class VectorStoreError(RuntimeError):
    """Raised when the store cannot be built or loaded."""


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class VectorStore:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        docs_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        model_name: str = config.EMBED_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.index_path = Path(index_path or config.INDEX_PATH)
        self.docs_path = Path(docs_path or config.DOCS_PATH)
        self.meta_path = Path(meta_path or config.META_PATH)
        self.model_name = model_name
        self.chunk_size = max(50, int(chunk_size))
        # Overlap must stay below chunk size or chunking never advances.
        self.chunk_overlap = min(int(chunk_overlap), self.chunk_size // 2)

        self.index = None
        self.chunks: List[dict] = []
        self.dim: Optional[int] = None
        self._model = None

    # ------------------------------------------------------------------
    # Embedding model (loaded on first use -- it is ~90 MB and slow to init)
    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise VectorStoreError(
                    "sentence-transformers is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from exc
            print(f"[vector_store] loading embedding model '{self.model_name}' ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts into L2-normalised float32 vectors (cosine ready)."""
        vecs = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32,
        ).astype("float32")
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\r", "\n")
        # Repair words hyphenated across a line break (common in PDF text).
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping, sentence-aligned chunks."""
        text = self.clean_text(text)
        if not text:
            return []

        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            # A single very long sentence is hard-split on word count.
            if len(words) > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                step = self.chunk_size - self.chunk_overlap
                for start in range(0, len(words), step):
                    chunks.append(" ".join(words[start:start + self.chunk_size]))
                continue

            if current_len + len(words) > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Carry the tail of the last chunk over as overlap.
                tail: List[str] = []
                tail_len = 0
                for word in reversed(current):
                    if tail_len >= self.chunk_overlap:
                        break
                    tail.insert(0, word)
                    tail_len += 1
                current, current_len = list(tail), tail_len

            current.extend(words)
            current_len += len(words)

        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Reading source documents
    # ------------------------------------------------------------------
    def collect_chunks(self) -> List[dict]:
        if not self.data_dir.exists():
            raise VectorStoreError(
                f"Data folder not found: {self.data_dir}\n"
                "Create it and add .txt files, or run: python main.py ingest"
            )

        files = sorted(
            p for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".md"}
        )
        print(f"[vector_store] {len(files)} source file(s) in {self.data_dir}")

        chunks: List[dict] = []
        for path in files:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"[vector_store] could not read {path.name}: {exc}")
                continue
            pieces = self.chunk_text(raw)
            for i, piece in enumerate(pieces):
                chunks.append({
                    "text": piece,
                    "source": path.name,
                    "chunk": i,
                })
            print(f"[vector_store]   {path.name}: {len(pieces)} chunk(s)")
        return chunks

    # ------------------------------------------------------------------
    # Build / save / load
    # ------------------------------------------------------------------
    def _fingerprint(self) -> dict:
        return {
            "version": config.INDEX_VERSION,
            "embed_model": self.model_name,
            "metric": "cosine",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def build(self, chunks: Optional[Iterable[dict]] = None) -> "VectorStore":
        import faiss  # imported here so the module imports without faiss present

        chunks = list(chunks) if chunks is not None else self.collect_chunks()
        if not chunks:
            raise VectorStoreError(
                f"No text found in {self.data_dir}. Add .txt files there, or put a "
                "PDF in pdfs/ and run: python main.py ingest"
            )

        print(f"[vector_store] embedding {len(chunks)} chunk(s) ...")
        embeddings = self.encode([c["text"] for c in chunks], show_progress=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine on unit vectors
        index.add(embeddings)

        self.index = index
        self.chunks = chunks
        self.dim = int(embeddings.shape[1])

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        np.save(str(self.docs_path), np.array(chunks, dtype=object), allow_pickle=True)

        meta = self._fingerprint()
        meta.update({
            "dim": self.dim,
            "count": len(chunks),
            "sources": sorted({c["source"] for c in chunks}),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[vector_store] index built: {len(chunks)} chunks, dim {self.dim}")
        return self

    def is_stale(self) -> bool:
        """True when the saved index is missing or built with other settings."""
        if not (self.index_path.exists() and self.docs_path.exists()):
            return True
        if not self.meta_path.exists():
            return True
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        return any(meta.get(k) != v for k, v in self._fingerprint().items())

    def load(self, auto_build: bool = True) -> "VectorStore":
        import faiss

        if self.is_stale():
            if not auto_build:
                raise VectorStoreError(
                    "No usable index found. Run: python main.py build"
                )
            print("[vector_store] index missing or outdated -- rebuilding ...")
            return self.build()

        self.index = faiss.read_index(str(self.index_path))
        raw = np.load(str(self.docs_path), allow_pickle=True)
        # Tolerate the old format, where docs.npy held plain strings.
        self.chunks = [
            item if isinstance(item, dict)
            else {"text": str(item), "source": "unknown", "chunk": i}
            for i, item in enumerate(raw)
        ]
        self.dim = self.index.d

        if self.index.ntotal != len(self.chunks):
            if not auto_build:
                raise VectorStoreError("Index and document store are out of sync.")
            print("[vector_store] index/doc mismatch -- rebuilding ...")
            return self.build()

        print(f"[vector_store] loaded {len(self.chunks)} chunks (dim {self.dim})")
        return self

    def ensure_ready(self) -> "VectorStore":
        if self.index is None:
            self.load()
        return self

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None,
               min_score: float = 0.0) -> List[dict]:
        self.ensure_ready()
        k = config.TOP_K if k is None else k
        query = (query or "").strip()
        if not query:
            return []

        k = max(1, min(int(k), len(self.chunks)))
        scores, ids = self.index.search(self.encode([query]), k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:  # FAISS pads with -1 when it has fewer hits than k
                continue
            score = float(score)
            if score < min_score:
                continue
            chunk = self.chunks[int(idx)]
            results.append({
                "text": chunk["text"],
                "doc": chunk["text"],          # kept for backwards compatibility
                "source": chunk.get("source", "unknown"),
                "chunk": chunk.get("chunk", int(idx)),
                "score": score,
                "idx": int(idx),
            })
        # Highest cosine similarity first.
        return sorted(results, key=lambda r: r["score"], reverse=True)

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "dim": self.dim,
            "sources": sorted({c.get("source", "unknown") for c in self.chunks}),
            "embed_model": self.model_name,
            "index_path": str(self.index_path),
        }


if __name__ == "__main__":
    store = VectorStore().load()
    print(json.dumps(store.stats(), indent=2))
    for hit in store.search("What is this about?", k=3):
        print(f"\n[{hit['score']:.3f}] {hit['source']} #{hit['chunk']}")
        print(hit["text"][:200], "...")
