# vector_store.py
import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim embeddings


class VectorStore:
    def __init__(self, dim=None, index_path="faiss.index", docs_path="docs.npy"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index_path = index_path
        self.docs_path = docs_path
        self.index = None
        self.docs = []
        self.dim = dim or 384  # all-MiniLM-L6-v2 → 384 dimensions

        # Absolute path to /data folder
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(base_dir, "data")
        print(f"[init] Looking for documents in: {self.data_dir}")

    # -----------------------------------------------------------
    # TEXT CLEANING
    # -----------------------------------------------------------
    def clean_text(self, text):
        text = text.replace("\n", " ").strip()
        text = " ".join(text.split())       # remove extra spaces
        return text

    # -----------------------------------------------------------
    # CHUNKING
    # -----------------------------------------------------------
    def chunk_text(self, text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # -----------------------------------------------------------
    # LOAD TEXT FILES
    # -----------------------------------------------------------
    def get_docs_from_data(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.txt")))
        print(f"[get_docs_from_data] Found files: {files}")

        docs = []
        for f in files:
            try:
                with open(f, "r", encoding="utf8") as fh:
                    raw = fh.read().strip()
                    cleaned = self.clean_text(raw)

                    # apply chunking
                    chunks = self.chunk_text(cleaned)
                    docs.extend(chunks)

            except Exception as e:
                print(f"[get_docs_from_data] Error reading {f}: {e}")

        print(f"[get_docs_from_data] Total chunks: {len(docs)}")
        return docs

    # -----------------------------------------------------------
    # BUILD VECTOR STORE
    # -----------------------------------------------------------
    def build(self, docs=None):
        if docs is None:
            docs = self.get_docs_from_data()

        if not docs:
            raise ValueError(f"No documents found in {self.data_dir}!")

        print(f"[build] Encoding {len(docs)} documents...")

        embs = self.model.encode(
            docs,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        # Build FAISS L2 index
        self.index = faiss.IndexFlatL2(embs.shape[1])
        self.index.add(embs)
        self.docs = docs

        # Save index + docs metadata
        faiss.write_index(self.index, self.index_path)
        np.save(self.docs_path, np.array(docs, dtype=object))

        print("[build] Index built and saved successfully.")

    # -----------------------------------------------------------
    # LOAD VECTOR STORE
    # -----------------------------------------------------------
    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docs_path):
            raise FileNotFoundError(
                "Index or docs missing. Run VectorStore().build() first."
            )

        self.index = faiss.read_index(self.index_path)
        self.docs = list(np.load(self.docs_path, allow_pickle=True))
        self.dim = self.index.d

        print(f"[load] LOADED index with {len(self.docs)} docs.")

    # -----------------------------------------------------------
    # SEARCH
    # -----------------------------------------------------------
    def search(self, query, k=3):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({
                "doc": self.docs[int(idx)],
                "score": float(score),
                "idx": int(idx)
            })
        return results


# -----------------------------------------------------------
# Quick CLI Test
# -----------------------------------------------------------
if __name__ == "__main__":
    vs = VectorStore()
    vs.build()
    vs.load()
    print("Search example:", vs.search("What is this about?", k=2))
