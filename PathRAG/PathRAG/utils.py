"""Utility helpers shared by the simplified PathRAG implementation."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import tiktoken

LOGGER = logging.getLogger("PathRAG")


def set_logger(log_file: str) -> None:
    """Configure the package wide logger."""

    LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not LOGGER.handlers:
        LOGGER.addHandler(handler)


@lru_cache(maxsize=4)
def _encoder(model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model)


def count_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    return len(_encoder(model).encode(text))


def truncate_by_tokens(text: str, max_tokens: int, model: str) -> str:
    if max_tokens <= 0:
        return ""
    encoding = _encoder(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def join_non_empty(parts: Iterable[str], delimiter: str = "\n") -> str:
    return delimiter.join(part for part in parts if part)
