"""
ThreadWeave AI - LLM utility layer

Wraps ChatOllama so every agent can call the local LLM the same way, and so
that every call has a uniform fallback path if Ollama isn't running or the
model isn't pulled yet. This mirrors the "robustness" principle from the
report: every LLM call has a Python-based fallback so the pipeline never
fails completely due to LLM errors.
"""
import socket
import urllib.request
import urllib.error

from . import config

_LLM_STATUS_CHECKED = False
_LLM_AVAILABLE = False
_llm_instance = None


def _ollama_reachable() -> bool:
    """Quick, cheap check that the Ollama HTTP server is up before we ever
    try to import/instantiate LangChain's ChatOllama (which can hang)."""
    try:
        url = f"{config.OLLAMA_BASE_URL}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, socket.timeout, ConnectionRefusedError, OSError):
        return False


def is_llm_available() -> bool:
    """Checked once per process and cached."""
    global _LLM_STATUS_CHECKED, _LLM_AVAILABLE
    if _LLM_STATUS_CHECKED:
        return _LLM_AVAILABLE
    _LLM_STATUS_CHECKED = True
    _LLM_AVAILABLE = _ollama_reachable()
    if not _LLM_AVAILABLE:
        print(f"[llm_utils] Ollama not reachable at {config.OLLAMA_BASE_URL} "
              f"-> agents will use Python fallback logic instead of LLM reasoning.")
    return _LLM_AVAILABLE


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from langchain_ollama import ChatOllama
        _llm_instance = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )
    return _llm_instance


def call_llm(prompt: str, system: str = None) -> str:
    """
    Calls the local LLM with a prompt. Raises RuntimeError if the LLM is
    unavailable or the call fails - callers are expected to catch this and
    fall back to deterministic Python logic.
    """
    if not is_llm_available():
        raise RuntimeError("LLM unavailable (Ollama not reachable)")

    try:
        llm = _get_llm()
        messages = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


def safe_llm_call(prompt: str, system: str = None, fallback: str = "") -> tuple:
    """
    Convenience wrapper: returns (text, used_llm: bool).
    Never raises - always returns something usable.
    """
    try:
        text = call_llm(prompt, system=system)
        return text, True
    except Exception:
        return fallback, False
