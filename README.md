# Local RAG — ask your own documents

Retrieval-Augmented Generation over local files: your documents are chunked and
embedded into a FAISS index, the most relevant passages are retrieved for each
question, and a local Ollama model writes the answer from those passages only.
Nothing leaves your machine.

```
PDF / TXT  ──▶  ingest.py  ──▶  data/*.txt
                                   │
                          vector_store.py  (chunk ▸ embed ▸ FAISS)
                                   │
question ─▶ retrieve top-k ─▶ rag_query.py builds the prompt ─▶ Ollama ─▶ answer + sources
                                   │
                                app.py  (FastAPI + web UI)
```

## Setup

```bash
conda activate rag310                 # or: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You also need [Ollama](https://ollama.com) running with at least one model:

```bash
ollama serve          # usually already running after install
ollama pull llama3.2  # any chat model works; the app auto-detects it
```

Check everything at once:

```bash
python main.py doctor
```

## Build the index

```bash
python main.py ingest     # convert PDFs in ./ and ./pdfs, then build the index
python main.py build      # rebuild from data/*.txt only
```

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) once.

## Ask questions

```bash
python main.py serve                    # http://127.0.0.1:8000
python main.py ask "What is RAG?"       # one-shot, prints answer + sources
python main.py chat                     # interactive loop
```

`uvicorn app:app --reload` still works if you prefer it.

## API

| Method | Path       | Purpose                                            |
| ------ | ---------- | -------------------------------------------------- |
| GET    | `/`        | Web UI                                              |
| GET    | `/health`  | Index stats, Ollama status, selected model          |
| POST   | `/ask`     | `{"question": "...", "top_k": 4, "model": null}`    |
| POST   | `/reindex` | Convert new PDFs and rebuild the index              |
| GET    | `/docs`    | Swagger UI                                          |

`/ask` always returns the retrieved passages, even when the LLM is unavailable,
so you can see what retrieval found:

```json
{
  "answer": "...",
  "sources": [{"source": "RAGlearn.txt", "chunk": 2, "score": 0.61, "text": "..."}],
  "model": "llama3.2:latest",
  "llm_ok": true,
  "error": null
}
```

## Adding documents

Drop `.txt` or `.md` files into `data/`, or PDFs into `pdfs/` (or the project
root), then run `python main.py ingest`. Scanned PDFs with no text layer are
reported and skipped — they need OCR first.

## Configuration

Everything is overridable by environment variable; defaults live in `config.py`.

| Variable            | Default                  | Meaning                              |
| ------------------- | ------------------------ | ------------------------------------ |
| `RAG_EMBED_MODEL`   | `all-MiniLM-L6-v2`       | sentence-transformers model          |
| `RAG_CHUNK_SIZE`    | `220`                    | words per chunk                      |
| `RAG_CHUNK_OVERLAP` | `50`                     | words repeated between chunks        |
| `RAG_TOP_K`         | `4`                      | passages retrieved per question      |
| `RAG_MIN_SCORE`     | `0.15`                   | cosine floor for "relevant"          |
| `RAG_LLM_MODEL`     | *(auto-detect)*          | pin a specific Ollama model          |
| `OLLAMA_HOST`       | `http://127.0.0.1:11434` | where Ollama listens                 |
| `RAG_LLM_TIMEOUT`   | `180`                    | seconds to wait for a completion     |
| `RAG_PORT`          | `8000`                   | web server port                      |

Changing the embedding model or chunk settings makes the saved index stale; it
is detected and rebuilt automatically on the next run.

## Tests

```bash
python test_rag.py          # 31 offline checks, no Ollama and no downloads
python test_rag.py --live   # also queries a running server
```

## Files

| File              | Role                                                        |
| ----------------- | ----------------------------------------------------------- |
| `config.py`       | All settings, env-overridable                                |
| `vector_store.py` | Chunking, embedding, FAISS index, cosine search              |
| `model_utils.py`  | Ollama HTTP client, model auto-detection, health check       |
| `rag_query.py`    | Retrieval → prompt → answer, plus an interactive CLI         |
| `ingest.py`       | PDF → text → index (`pdf_to_docs.py` calls into it)          |
| `app.py`          | FastAPI server                                               |
| `main.py`         | `doctor` / `ingest` / `build` / `ask` / `chat` / `serve`     |
| `test_rag.py`     | Offline checks with a stub encoder + live API check          |
| `static/`         | Web UI                                                       |

Generated files (`faiss.index`, `docs.npy`, `index_meta.json`, and the `.txt`
extracted from PDFs) are git-ignored — rebuild them with `python main.py ingest`.
