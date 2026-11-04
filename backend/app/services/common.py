"""Shared helpers: device selection, errors and lazy model loading."""
import logging
import threading
from typing import Callable, Generic, TypeVar

import torch

from ..config import get_settings

logger = logging.getLogger("truthlens")
T = TypeVar("T")


class ModelNotReadyError(RuntimeError):
    """Raised when a model is not trained / not found. Returned to the client as HTTP 503."""


def get_device() -> torch.device:
    choice = get_settings().device.lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


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
