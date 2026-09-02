"""Talking to Ollama.

The original code shelled out to `ollama run llama2`, which pulls a model you
may not have, gives no timeout and returns spinner characters mixed into the
answer. This module uses Ollama's HTTP API instead, picks whichever model you
actually have installed, and fails with a message that says what to do.
"""

from __future__ import annotations

from typing import List, Optional

import requests

import config


class OllamaError(RuntimeError):
    """Ollama is unreachable, has no models, or returned an error."""


_NOT_RUNNING = (
    "Cannot reach Ollama at {host}.\n"
    "  1. Install it from https://ollama.com\n"
    "  2. Start it:  ollama serve\n"
    "  3. Pull a model:  ollama pull llama3.2"
)

_NO_MODELS = (
    "Ollama is running at {host} but no models are installed.\n"
    "  Pull one first:  ollama pull llama3.2"
)


def list_models(host: str = config.OLLAMA_HOST, timeout: float = 5.0) -> List[str]:
    """Names of the models installed in Ollama. Raises OllamaError if down."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        raise OllamaError(_NOT_RUNNING.format(host=host)) from exc
    except ValueError as exc:
        raise OllamaError(f"Ollama at {host} returned an unreadable response.") from exc

    return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]


def pick_model(preferred: Optional[str] = None,
               host: str = config.OLLAMA_HOST) -> str:
    """Choose a chat model: explicit request > config > best installed."""
    installed = list_models(host=host)
    if not installed:
        raise OllamaError(_NO_MODELS.format(host=host))

    wanted = (preferred or config.OLLAMA_MODEL or "").strip()
    if wanted:
        # Accept "llama3.2" for an installed "llama3.2:latest".
        for name in installed:
            if name == wanted or name.split(":")[0] == wanted.split(":")[0]:
                return name
        raise OllamaError(
            f"Model '{wanted}' is not installed in Ollama.\n"
            f"  Installed: {', '.join(installed)}\n"
            f"  Or pull it:  ollama pull {wanted}"
        )

    # Embedding-only models cannot answer questions.
    chat_models = [n for n in installed if "embed" not in n.lower()] or installed
    for family in config.MODEL_PREFERENCES:
        for name in chat_models:
            if name.split(":")[0] == family:
                return name
    return chat_models[0]


def generate(prompt: str,
             model: Optional[str] = None,
             system: Optional[str] = None,
             host: str = config.OLLAMA_HOST,
             timeout: float = config.OLLAMA_TIMEOUT,
             temperature: float = config.OLLAMA_TEMPERATURE) -> str:
    """Send a prompt to Ollama and return the completion text."""
    model = model or pick_model(host=host)
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        body["system"] = system

    try:
        response = requests.post(f"{host}/api/generate", json=body, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            f"'{model}' did not answer within {timeout:.0f}s. Try a smaller model "
            "or raise RAG_LLM_TIMEOUT."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError(_NOT_RUNNING.format(host=host)) from exc

    if response.status_code != 200:
        raise OllamaError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
        )

    answer = (response.json().get("response") or "").strip()
    if not answer:
        raise OllamaError(f"'{model}' returned an empty answer.")
    return answer


def health(host: str = config.OLLAMA_HOST) -> dict:
    """Status summary that never raises -- handy for /health and diagnostics."""
    try:
        models = list_models(host=host)
    except OllamaError as exc:
        return {"reachable": False, "host": host, "models": [],
                "model": None, "error": str(exc)}

    try:
        selected = pick_model(host=host)
        error = None
    except OllamaError as exc:
        selected = None
        error = str(exc)

    return {"reachable": True, "host": host, "models": models,
            "model": selected, "error": error}


if __name__ == "__main__":
    import json
    status = health()
    print(json.dumps(status, indent=2))
    if status["model"]:
        print("\nTest answer:\n", generate("Say 'ready' and nothing else."))
