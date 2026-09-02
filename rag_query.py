"""The RAG pipeline: retrieve relevant chunks, then ask the LLM about them.

Run it directly for an interactive shell:
    python rag_query.py
"""

from __future__ import annotations

from typing import List, Optional

import config
import model_utils
from model_utils import OllamaError
from vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a careful assistant answering strictly from the provided context. "
    "If the context does not contain the answer, say so plainly instead of "
    "guessing. Keep answers concise and factual."
)

PROMPT_TEMPLATE = """Context passages:

{context}

---
Answer the question using ONLY the context above. If the context does not
contain the answer, reply exactly: "The documents don't cover that."

Question: {question}

Answer:"""

_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """Load the vector store once and reuse it."""
    global _store
    if _store is None:
        _store = VectorStore().load()
    return _store


def set_store(store: VectorStore) -> None:
    """Inject a store (used by the API at startup and by the tests)."""
    global _store
    _store = store


def retrieve(question: str, top_k: Optional[int] = None,
             min_score: Optional[float] = None) -> List[dict]:
    # Defaults are read at call time so config (and env overrides) stay live.
    top_k = config.TOP_K if top_k is None else top_k
    min_score = config.MIN_SCORE if min_score is None else min_score
    return get_store().search(question, k=top_k, min_score=min_score)


def build_prompt(question: str, passages: List[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source: {p['source']})\n{p['text']}"
        for i, p in enumerate(passages)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer(question: str,
           top_k: Optional[int] = None,
           model: Optional[str] = None,
           min_score: Optional[float] = None) -> dict:
    """Answer a question. Always returns a dict -- never raises for LLM issues.

    Keys: question, answer, sources, model, llm_ok, error
    """
    question = (question or "").strip()
    if not question:
        return {"question": question, "answer": None, "sources": [],
                "model": None, "llm_ok": False,
                "error": "Please type a question."}

    passages = retrieve(question, top_k=top_k, min_score=min_score)

    sources = [
        {
            "source": p["source"],
            "chunk": p["chunk"],
            "score": round(p["score"], 4),
            "text": p["text"],
        }
        for p in passages
    ]

    if not passages:
        return {"question": question,
                "answer": "The documents don't cover that.",
                "sources": [], "model": None, "llm_ok": True, "error": None}

    try:
        # Always resolve through pick_model so a short name like "llama2"
        # is validated and reported as the real tag ("llama2:7b").
        chosen = model_utils.pick_model(preferred=model)
        text = model_utils.generate(
            build_prompt(question, passages),
            model=chosen,
            system=SYSTEM_PROMPT,
        )
        return {"question": question, "answer": text, "sources": sources,
                "model": chosen, "llm_ok": True, "error": None}
    except OllamaError as exc:
        # Retrieval worked; only the generation step failed. Hand back the
        # passages so the pipeline is still visibly doing its job.
        return {"question": question, "answer": None, "sources": sources,
                "model": None, "llm_ok": False, "error": str(exc)}


def rag_query(user_query: str, top_k: Optional[int] = None,
              model: Optional[str] = None) -> str:
    """Backwards-compatible helper that returns just the answer text."""
    result = answer(user_query, top_k=top_k, model=model)
    return result["answer"] or f"[LLM unavailable] {result['error']}"


def _interactive() -> None:
    print("Loading index and embedding model ...")
    store = get_store()
    stats = store.stats()
    print(f"Ready - {stats['chunks']} chunks from {len(stats['sources'])} file(s).")

    status = model_utils.health()
    if status["model"]:
        print(f"LLM: {status['model']} via {status['host']}")
    else:
        print(f"LLM unavailable:\n{status['error']}")
    print("Type a question, or 'exit' to quit.")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        result = answer(question)
        if result["answer"]:
            print(f"\n{result['answer']}")
        else:
            print(f"\n[LLM unavailable] {result['error']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                preview = s["text"][:110].replace("\n", " ")
                print(f"  - {s['source']} #{s['chunk']} "
                      f"(score {s['score']:.3f}): {preview}...")


if __name__ == "__main__":
    _interactive()
