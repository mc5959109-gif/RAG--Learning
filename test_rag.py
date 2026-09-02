"""Checks for the RAG pipeline.

    python test_rag.py          offline checks (no Ollama, no model download)
    python test_rag.py --live   also query a server running on port 8000

The offline checks swap the sentence-transformers model for a tiny
deterministic stand-in, so they run in seconds and never touch the network.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import config
import rag_query
from vector_store import VectorStore


class StubEncoder:
    """Hashing bag-of-words encoder - deterministic, no downloads."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False,
               batch_size=32):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in str(text).lower().split():
                vectors[row][hash(word) % self.dim] += 1.0
        return vectors


def make_store(tmp: Path) -> VectorStore:
    data_dir = tmp / "data"
    data_dir.mkdir()
    (data_dir / "bikes.txt").write_text(
        "The Yamaha FZ25 uses a 249cc air-cooled single-cylinder engine. "
        "It produces about 20.8 PS at 8000 rpm and 20.1 Nm of torque.",
        encoding="utf-8",
    )
    (data_dir / "coffee.txt").write_text(
        "Espresso is brewed by forcing hot water through finely ground coffee "
        "under roughly nine bars of pressure.",
        encoding="utf-8",
    )
    store = VectorStore(
        data_dir=data_dir,
        index_path=tmp / "faiss.index",
        docs_path=tmp / "docs.npy",
        meta_path=tmp / "meta.json",
        chunk_size=60,
        chunk_overlap=10,
    )
    store.model = StubEncoder()
    return store


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail and not condition else ''}")
    return condition


def run_offline() -> bool:
    results = []
    print("\nOffline checks")

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        store = make_store(tmp)

        # --- chunking -------------------------------------------------
        small = VectorStore(chunk_size=50, chunk_overlap=999)
        results.append(check(
            "overlap is clamped below chunk size",
            small.chunk_overlap < small.chunk_size,
            f"got overlap={small.chunk_overlap}",
        ))
        long_text = " ".join(f"word{i}" for i in range(500))
        pieces = small.chunk_text(long_text)
        results.append(check("long text is chunked", len(pieces) > 1,
                             f"got {len(pieces)} chunk(s)"))
        results.append(check("no chunk exceeds the size limit",
                             all(len(p.split()) <= small.chunk_size for p in pieces)))
        results.append(check("empty text yields no chunks", small.chunk_text("   ") == []))
        results.append(check("hyphen line-breaks are repaired",
                             "cylinder" in VectorStore.clean_text("single-\ncylinder")))

        # --- build + search -------------------------------------------
        store.build()
        results.append(check("index has chunks", len(store.chunks) >= 2))
        results.append(check("chunks carry their source file",
                             all("source" in c for c in store.chunks)))

        hits = store.search("How much torque does the FZ25 make?", k=2)
        results.append(check("search returns hits", len(hits) > 0))
        results.append(check("top hit comes from the right document",
                             bool(hits) and hits[0]["source"] == "bikes.txt",
                             hits[0]["source"] if hits else "no hits"))
        results.append(check("scores are sorted high to low",
                             [h["score"] for h in hits] ==
                             sorted((h["score"] for h in hits), reverse=True)))
        results.append(check("k larger than the corpus is safe",
                             len(store.search("engine", k=500)) <= len(store.chunks)))
        results.append(check("empty query returns nothing", store.search("  ") == []))
        results.append(check("high threshold filters everything out",
                             store.search("engine", k=3, min_score=1e9) == []))

        # --- persistence ----------------------------------------------
        results.append(check("freshly built index is not stale", not store.is_stale()))
        reopened = VectorStore(
            data_dir=store.data_dir, index_path=store.index_path,
            docs_path=store.docs_path, meta_path=store.meta_path,
            chunk_size=60, chunk_overlap=10,
        )
        reopened.model = StubEncoder()
        reopened.load(auto_build=False)
        results.append(check("index reloads from disk",
                             len(reopened.chunks) == len(store.chunks)))

        changed = VectorStore(
            data_dir=store.data_dir, index_path=store.index_path,
            docs_path=store.docs_path, meta_path=store.meta_path,
            chunk_size=120, chunk_overlap=10,
        )
        results.append(check("different settings mark the index stale",
                             changed.is_stale()))

        # --- pipeline --------------------------------------------------
        rag_query.set_store(store)
        passages = rag_query.retrieve("torque of the FZ25", top_k=2, min_score=0.0)
        prompt = rag_query.build_prompt("torque of the FZ25", passages)
        results.append(check("prompt embeds the question",
                             "torque of the FZ25" in prompt))
        results.append(check("prompt embeds the context",
                             "bikes.txt" in prompt))

        real_generate = rag_query.model_utils.generate
        real_pick = rag_query.model_utils.pick_model
        try:
            rag_query.model_utils.pick_model = lambda *a, **k: "stub-model"
            rag_query.model_utils.generate = lambda *a, **k: "20.1 Nm of torque."
            answered = rag_query.answer("What torque does the FZ25 make?",
                                        min_score=0.0)
            results.append(check("answer() returns the model text",
                                 answered["answer"] == "20.1 Nm of torque."))
            results.append(check("answer() attaches sources",
                                 len(answered["sources"]) > 0))
            results.append(check("answer() reports the model used",
                                 answered["model"] == "stub-model"))

            def boom(*_a, **_k):
                raise rag_query.OllamaError("Ollama is not running")

            rag_query.model_utils.generate = boom
            degraded = rag_query.answer("What torque does the FZ25 make?",
                                        min_score=0.0)
            results.append(check("LLM failure is reported, not raised",
                                 degraded["llm_ok"] is False and degraded["answer"] is None))
            results.append(check("sources survive an LLM failure",
                                 len(degraded["sources"]) > 0))
            results.append(check("blank question is rejected",
                                 rag_query.answer("   ")["error"] is not None))
        finally:
            rag_query.model_utils.generate = real_generate
            rag_query.model_utils.pick_model = real_pick

    # --- API layer ----------------------------------------------------
    print("\nAPI checks")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  SKIP  fastapi TestClient unavailable (pip install httpx)")
    else:
        import app as app_module
        with tempfile.TemporaryDirectory() as raw_tmp:
            store = make_store(Path(raw_tmp))
            store.build()
            app_module.STORE = store
            app_module.STARTUP_ERROR = None
            rag_query.set_store(store)

            real_generate = rag_query.model_utils.generate
            real_pick = rag_query.model_utils.pick_model
            real_min = config.MIN_SCORE
            rag_query.model_utils.pick_model = lambda *a, **k: "stub-model"
            rag_query.model_utils.generate = lambda *a, **k: "A 249cc engine."
            # The stub encoder's similarities are not on the same scale as a
            # real embedding model, so drop the relevance floor for this check.
            config.MIN_SCORE = 0.0
            try:
                # lifespan is skipped so the injected store survives
                client = TestClient(app_module.app)
                health = client.get("/health").json()
                results.append(check("/health reports a ready index",
                                     health["index_ready"] is True))

                response = client.post("/ask", json={"question": "engine size?"})
                body = response.json()
                results.append(check("/ask returns 200", response.status_code == 200,
                                     str(response.status_code)))
                results.append(check("/ask returns an answer",
                                     body.get("answer") == "A 249cc engine."))
                results.append(check("/ask returns sources",
                                     len(body.get("sources", [])) > 0))
                results.append(check("/ask rejects an empty question",
                                     client.post("/ask", json={"question": ""})
                                     .status_code == 422))
                results.append(check("/ serves the web page",
                                     client.get("/").status_code == 200))

                # Nothing relevant in the corpus -> honest fallback, no LLM call.
                config.MIN_SCORE = 1e9
                off_topic = client.post(
                    "/ask", json={"question": "who won the 1998 world cup?"}
                ).json()
                results.append(check(
                    "off-topic question gets the honest fallback",
                    off_topic["answer"] == "The documents don't cover that."
                    and off_topic["sources"] == []))
            finally:
                rag_query.model_utils.generate = real_generate
                rag_query.model_utils.pick_model = real_pick
                config.MIN_SCORE = real_min

    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} checks passed")
    return all(results)


def run_live(url: str = "http://127.0.0.1:8000") -> bool:
    import requests
    print(f"\nLive check against {url}")
    try:
        health = requests.get(f"{url}/health", timeout=10).json()
        print(f"  index ready: {health['index_ready']}, LLM: {health['llm'].get('model')}")
        response = requests.post(
            f"{url}/ask",
            json={"question": "Explain the main idea of the document?"},
            timeout=300,
        ).json()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  could not reach the server: {exc}")
        print("  Start it with: python main.py serve")
        return False

    if response.get("answer"):
        print(f"  PASS  answer: {response['answer'][:300]}")
        for s in response.get("sources", []):
            print(f"        source: {s['source']} #{s['chunk']} ({s['score']:.3f})")
        return True
    print(f"  FAIL  {response.get('error')}")
    return False


if __name__ == "__main__":
    ok = run_offline()
    if "--live" in sys.argv:
        ok = run_live() and ok
    print("\nRESULT:", "OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
