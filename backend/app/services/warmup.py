"""Loads the models in the background as soon as the backend starts.

Without this, each model loaded (and on the first run downloaded) only when someone clicked Check,
so the first check of each kind waited minutes on a laptop. Now the backend starts at once, the models
get ready one after another behind it, and the app shows the progress.
Only the models the current settings use are loaded: with the LLM checker on, the big NLI and
BGE-M3 models of the fallback pipeline are not loaded at all.
"""
import threading
import time

from ..config import get_settings
from .common import ModelNotReadyError, logger

state = {"running": False, "done": 0, "total": 0, "current": None, "failed": [], "seconds": None}


def _steps():
    from .claim_extractor import _nlp
    from .evidence import get_retriever
    from .image_detector import get_image_detector
    from .llm_verifier import get_embedder, llm_available
    from .text_classifier import get_text_classifier

    return [
        ("English text model", lambda: (get_text_classifier("en"), _nlp.get())),
        ("Evidence search model", get_embedder if llm_available() else get_retriever),
        ("Image models", get_image_detector),
        ("Telugu text model", lambda: get_text_classifier("te")),
    ]


def _run():
    start = time.time()
    steps = _steps()
    state.update(running=True, done=0, total=len(steps), failed=[])
    for name, load in steps:
        state["current"] = name
        try:
            load()
        except ModelNotReadyError:
            pass  # not trained yet: that page says so
        except Exception as exc:  # e.g. no internet on the first run; it is retried on first use
            logger.warning("Could not preload %s: %s", name, exc)
            state["failed"].append(name)
        state["done"] += 1
    state.update(running=False, current=None, seconds=round(time.time() - start))
    logger.info("All models ready in %ss.", state["seconds"])


def start_warmup() -> None:
    if get_settings().preload_models and not state["running"]:
        threading.Thread(target=_run, name="model-warmup", daemon=True).start()
