"""Single entry point for the project.

    python main.py doctor          check the environment
    python main.py ingest          convert PDFs to text and build the index
    python main.py build           rebuild the index from data/*.txt
    python main.py ask "question"  one-shot question
    python main.py chat            interactive question loop
    python main.py serve           run the web app on http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys

import config


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"Project: {config.BASE_DIR}")
    print(f"Python : {sys.version.split()[0]}")

    missing = []
    for module, package in [
        ("numpy", "numpy"), ("faiss", "faiss-cpu"),
        ("sentence_transformers", "sentence-transformers"),
        ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
        ("pdfplumber", "pdfplumber"), ("requests", "requests"),
    ]:
        try:
            __import__(module)
            print(f"  [ok]      {package}")
        except ImportError:
            missing.append(package)
            print(f"  [MISSING] {package}")

    data_files = sorted(p.name for p in config.DATA_DIR.glob("*.txt")) \
        if config.DATA_DIR.exists() else []
    print(f"\nData folder: {config.DATA_DIR} ({len(data_files)} text file(s))")
    for name in data_files:
        print(f"  - {name}")

    print(f"\nIndex file : {config.INDEX_PATH} "
          f"({'present' if config.INDEX_PATH.exists() else 'not built yet'})")

    import model_utils
    status = model_utils.health()
    if status["model"]:
        print(f"\nOllama     : {status['host']} -> using '{status['model']}'")
        print(f"  installed: {', '.join(status['models'])}")
    else:
        print(f"\nOllama     : unavailable\n{status['error']}")

    if missing:
        print(f"\nInstall the missing packages:\n  pip install {' '.join(missing)}")
        return 1
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from ingest import build_index, convert_pdfs
    convert_pdfs(force=args.force)
    stats = build_index().stats()
    print(f"\nIndexed {stats['chunks']} chunks from: {', '.join(stats['sources'])}")
    return 0


def cmd_build(_: argparse.Namespace) -> int:
    from vector_store import VectorStore
    stats = VectorStore().build().stats()
    print(f"\nIndexed {stats['chunks']} chunks from: {', '.join(stats['sources'])}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    import rag_query
    result = rag_query.answer(" ".join(args.question), top_k=args.top_k)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["llm_ok"] else 1

    if result["answer"]:
        print(f"\n{result['answer']}")
    else:
        print(f"\n[LLM unavailable] {result['error']}")
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s['source']} #{s['chunk']} (score {s['score']:.3f})")
    return 0 if result["llm_ok"] else 1


def cmd_chat(_: argparse.Namespace) -> int:
    import rag_query
    rag_query._interactive()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install -r requirements.txt")
        return 1
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Local RAG over your documents, answered by Ollama."
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="check packages, data and Ollama").set_defaults(func=cmd_doctor)

    p_ingest = sub.add_parser("ingest", help="convert PDFs then build the index")
    p_ingest.add_argument("--force", action="store_true", help="re-extract every PDF")
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("build", help="rebuild the index from data/*.txt").set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="ask one question and exit")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--top-k", type=int, default=config.TOP_K)
    p_ask.add_argument("--json", action="store_true", help="print the raw result")
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="interactive question loop").set_defaults(func=cmd_chat)

    p_serve = sub.add_parser("serve", help="run the FastAPI web app")
    p_serve.add_argument("--host", default=config.HOST)
    p_serve.add_argument("--port", type=int, default=config.PORT)
    p_serve.add_argument("--reload", action="store_true", default=True)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
