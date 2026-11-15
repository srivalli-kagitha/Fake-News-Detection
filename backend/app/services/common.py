"""Shared helpers: device selection, errors and lazy model loading."""
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Generic, TypeVar

import torch

from ..config import get_settings

logger = logging.getLogger("truthlens")
T = TypeVar("T")

# PyTorch uses one thread per logical CPU by default. With hyper-threading those threads fight over
# the same cores: on a 16-thread CPU the models ran up to 7x slower than with 8 threads.
# Use about one thread per physical core (at most 8).
torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))


class ModelNotReadyError(RuntimeError):
    """Raised when a model is not trained / not found. Returned to the client as HTTP 503."""


def get_device() -> torch.device:
    choice = get_settings().device.lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


# Only the files a model needs. Many Hugging Face repos hold the same weights twice (model.safetensors
# AND pytorch_model.bin, sometimes ONNX too); downloading everything doubled the first-run download.
MODEL_FILES = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "sentencepiece*", "spm*",
               "1_Pooling/*"]


def hf_path(repo: str) -> str:
    """Local folder of a Hugging Face model, downloading only the needed files the first time.
    Works offline once the files are on disk."""
    if Path(repo).exists():
        return repo
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo, allow_patterns=MODEL_FILES)
    except Exception:  # no internet: use what is already downloaded
        path = snapshot_download(repo, allow_patterns=MODEL_FILES, local_files_only=True)
    if not any(Path(path).glob("*.safetensors")):  # older repos only ship pytorch_model.bin
        path = snapshot_download(repo, allow_patterns=MODEL_FILES + ["pytorch_model.bin"])
    return path


class Lazy(Generic[T]):
    """Builds an object the first time it is needed (thread-safe), then reuses it."""

    def __init__(self, name: str, factory: Callable[[], T]):
        self.name = name
        self._factory = factory
        self._value: T | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._value is not None

    def get(self) -> T:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    logger.info("Loading %s ...", self.name)
                    self._value = self._factory()
                    logger.info("%s ready.", self.name)
        return self._value
