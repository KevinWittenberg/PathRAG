"""Helper routines that build context windows for the retriever."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .base import ChunkMatch, ContextWindow, EntityMatch
from .storage import StorageAdapter
from .utils import join_non_empty, truncate_by_tokens


def build_graph_windows(
    adapter: StorageAdapter,
    seeds: Sequence[EntityMatch],
    *,
    depth: int,
    max_windows: int,
    max_tokens: int,
    model: str,
) -> List[ContextWindow]:
    descriptions = adapter.describe_subgraph([seed.name for seed in seeds], depth=depth)
    windows: List[ContextWindow] = []
    score_by_seed = {seed.name: seed.score for seed in seeds}
    for seed, text in descriptions:
        trimmed = truncate_by_tokens(text, max_tokens, model)
        windows.append(
            ContextWindow(label=f"graph::{seed}", text=trimmed, score=score_by_seed.get(seed, 1.0))
        )
        if len(windows) >= max_windows:
            break
    return windows


def build_chunk_windows(
    chunks: Sequence[ChunkMatch],
    *,
    max_windows: int,
    max_tokens: int,
    model: str,
) -> List[ContextWindow]:
    windows: List[ContextWindow] = []
    for chunk in chunks[:max_windows]:
        snippet = truncate_by_tokens(chunk.text, max_tokens, model)
        label = f"chunk::{chunk.filename or chunk.document_id}" if chunk.filename else "chunk"
        windows.append(ContextWindow(label=label, text=snippet, score=chunk.score))
    return windows


def merge_windows(graph_windows: Sequence[ContextWindow], chunk_windows: Sequence[ContextWindow]) -> List[ContextWindow]:
    return list(graph_windows) + list(chunk_windows)


def format_windows_for_prompt(windows: Iterable[ContextWindow]) -> str:
    blocks = [f"[{window.label}]\n{window.text}" for window in windows]
    return join_non_empty(blocks, delimiter="\n\n")
