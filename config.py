"""Central configuration for the RAG project.

Every setting can be overridden with an environment variable, so you can point
the app at a different data folder, embedding model or Ollama host without
touching the code.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Where documents and index artifacts live
# --------------------------------------------------------------------------
DATA_DIR = _env_path("RAG_DATA_DIR", BASE_DIR / "data")
PDF_DIR = _env_path("RAG_PDF_DIR", BASE_DIR / "pdfs")

# PDFs are picked up from pdfs/ *and* from the project root, so a PDF dropped
# next to the code is indexed too.
PDF_SEARCH_DIRS = [PDF_DIR, BASE_DIR]

INDEX_PATH = _env_path("RAG_INDEX_PATH", BASE_DIR / "faiss.index")
DOCS_PATH = _env_path("RAG_DOCS_PATH", BASE_DIR / "docs.npy")
META_PATH = _env_path("RAG_META_PATH", BASE_DIR / "index_meta.json")

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")  # 384 dims
CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 220)      # words per chunk
CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 50)  # words repeated between chunks
TOP_K = _env_int("RAG_TOP_K", 4)
# Cosine similarity below this is treated as "not really related".
MIN_SCORE = _env_float("RAG_MIN_SCORE", 0.15)

# Bumped whenever the on-disk index format changes, so stale indexes rebuild.
INDEX_VERSION = 2

# --------------------------------------------------------------------------
# LLM (Ollama)
# --------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
# Leave empty to auto-detect whichever model is installed.
OLLAMA_MODEL = os.getenv("RAG_LLM_MODEL", "").strip()
OLLAMA_TIMEOUT = _env_float("RAG_LLM_TIMEOUT", 180.0)
OLLAMA_TEMPERATURE = _env_float("RAG_LLM_TEMPERATURE", 0.2)

# Preferred chat models, best first, used when auto-detecting.
MODEL_PREFERENCES = [
    "llama3.2", "llama3.1", "llama3", "qwen2.5", "mistral",
    "phi3", "gemma2", "gemma", "llama2",
]

# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
HOST = os.getenv("RAG_HOST", "127.0.0.1")
PORT = _env_int("RAG_PORT", 8000)

# Quiet the HuggingFace tokenizers fork warning everywhere at once.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
