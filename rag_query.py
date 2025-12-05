# rag_query.py
import os
import subprocess
from vector_store import VectorStore

# Avoid HuggingFace parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------
# Load Vector Store
# -----------------------------------------------------------
vs = VectorStore()
vs.load()


# -----------------------------------------------------------
# OLLAMA LLM CALL
# -----------------------------------------------------------
def ask_ollama(prompt: str, model="llama2"):
    """
    Calls an Ollama model through command line.
    Returns its output or an error string.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        return f"[Ollama Error] {e.stderr}"


# -----------------------------------------------------------
# SIMPLE RE-RANKER (optional improvement)
# -----------------------------------------------------------
def simple_rerank(query, results):
    """
    Improves ranking by sorting chunks by score similarity (lower = better)
    """
    return sorted(results, key=lambda x: x["score"])


# -----------------------------------------------------------
# RAG QUERY
# -----------------------------------------------------------
def rag_query(user_query, top_k=3, model="llama2"):
    # Step 1: retrieve chunks
    results = vs.search(user_query, k=top_k)

    if not results:
        return "No relevant documents found."

    # Step 2: rerank
    results = simple_rerank(user_query, results)

    # Step 3: extract text chunks
    retrieved_docs = [r["doc"] for r in results]

    # Build context block
    context_block = "\n\n---\n".join(retrieved_docs)

    # Step 4: construct an improved prompt
    prompt = f"""
You are an AI assistant using Retrieval Augmented Generation (RAG).

Relevant context from the database (these may be chunked from longer documents):

{context_block}

---

Using ONLY the context above, answer the following question clearly and concisely.

Question: {user_query}

Answer:
""".strip()

    # Step 5: call Ollama
    answer = ask_ollama(prompt, model=model)
    return answer


# -----------------------------------------------------------
# CLI TEST MODE
# -----------------------------------------------------------
if __name__ == "__main__":
    print("RAG with Ollama — READY ✔")
    while True:
        query = input("\nAsk something (type 'exit' to quit): ")
        if query.lower().strip() == "exit":
            break
        print("\nAnswer:\n", rag_query(query))

