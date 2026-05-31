import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from vector_store import VectorStore
import subprocess

# Disable HuggingFace tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="RAG with Ollama"

# Load vector store
vs = VectorStore()
vs.load()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve frontend
@app.get("/", response_class=FileResponse)
def serve_frontend():
    return os.path.join(STATIC_DIR, "index.html")

# API model
class Query(BaseModel):
    question: str

# Ask Ollama
def ask_ollama(prompt: str):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama2"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error calling Ollama: {e.stderr}"

# RAG query
def rag_answer(query: str, top_k=3):
    results = vs.search(query, k=top_k)
    docs = [r["doc"] for r in results]

    context = "\n\n".join(docs) if docs else "No matching documents."
    prompt = f"Use the context below to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    return ask_ollama(prompt)

# API endpoint
@app.post("/ask")
def ask_api(data: Query):
    try:
        answer = rag_answer(data.question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

