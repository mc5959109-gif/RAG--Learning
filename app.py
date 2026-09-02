"""FastAPI server for the RAG demo.

    uvicorn app:app --reload
    python main.py serve

    GET  /          web UI
    GET  /health    index + Ollama status
    POST /ask       {"question": "...", "top_k": 4}
    POST /reindex   rebuild the index from data/ (and any new PDFs)
    GET  /docs      Swagger UI
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import model_utils
import rag_query
from vector_store import VectorStore

STATIC_DIR = config.BASE_DIR / "static"

# Set at startup; None means the index failed to load.
STORE: Optional[VectorStore] = None
STARTUP_ERROR: Optional[str] = None


def _load_store() -> None:
    """Load (or build) the index and hand it to the query pipeline."""
    global STORE, STARTUP_ERROR
    try:
        store = VectorStore().load()
        rag_query.set_store(store)
        STORE, STARTUP_ERROR = store, None
        print(f"[app] index ready: {store.stats()}")
    except Exception as exc:  # noqa: BLE001 - any failure should be reported, not fatal
        STORE, STARTUP_ERROR = None, str(exc)
        print(f"[app] index NOT ready: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_store()
    yield


app = FastAPI(
    title="RAG with Ollama",
    description="Retrieval-augmented question answering over local documents.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class Query(BaseModel):
    question: str = Field(..., min_length=1, description="What to ask the documents")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    model: Optional[str] = Field(None, description="Ollama model; blank = auto")


@app.get("/", include_in_schema=False)
def serve_frontend():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"detail": "static/index.html is missing"}, status_code=404)
    return FileResponse(str(index_file))


@app.get("/health")
def health():
    return {
        "index_ready": STORE is not None,
        "index": STORE.stats() if STORE else None,
        "index_error": STARTUP_ERROR,
        "llm": model_utils.health(),
    }


@app.post("/ask")
def ask_api(data: Query):
    if STORE is None:
        return JSONResponse(
            {"answer": None, "sources": [], "llm_ok": False,
             "error": f"Index not loaded: {STARTUP_ERROR}"},
            status_code=503,
        )
    try:
        result = rag_query.answer(
            data.question,
            top_k=data.top_k or config.TOP_K,
            model=data.model or None,
        )
    except Exception as exc:  # noqa: BLE001 - surface the reason to the UI
        return JSONResponse(
            {"answer": None, "sources": [], "llm_ok": False, "error": str(exc)},
            status_code=500,
        )
    # 200 even when the LLM is down: retrieval still returned sources.
    return JSONResponse(result)


@app.post("/reindex")
def reindex():
    """Re-read data/ (converting any new PDFs first) and rebuild the index."""
    try:
        from ingest import convert_pdfs
        convert_pdfs()
        _load_store_rebuild()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return {"ok": True, "index": STORE.stats() if STORE else None}


def _load_store_rebuild() -> None:
    global STORE, STARTUP_ERROR
    store = VectorStore()
    store.build()
    rag_query.set_store(store)
    STORE, STARTUP_ERROR = store, None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)
