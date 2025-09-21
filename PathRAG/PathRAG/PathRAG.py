"""Simplified PathRAG retriever wired to the ingestion storage backend."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from .base import RetrievalResult
from .llm import AsyncChat
from .operate import (
    build_chunk_windows,
    build_graph_windows,
    format_windows_for_prompt,
    merge_windows,
)
from .prompt import RAG_PROMPT
from .storage import StorageAdapter, StoragePaths
from .utils import LOGGER, set_logger


@dataclass
class RetrieverConfig:
    """Tunable knobs that control how evidence is assembled."""

    entity_top_k: int = 5
    relation_top_k: int = 5
    chunk_top_k: int = 6
    graph_depth: int = 2
    graph_windows: int = 3
    chunk_windows: int = 3
    graph_window_tokens: int = 512
    chunk_window_tokens: int = 512
    tiktoken_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2


class PathRAG:
    """Minimal retriever that relies on the ingestion storage backend."""

    def __init__(
        self,
        *,
        storage_paths: Optional[StoragePaths] = None,
        system_prompt: Optional[str] = None,
        log_file: str = "PathRAG.log",
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        set_logger(log_file)
        LOGGER.info("Initialising PathRAG retriever")
        self._config = config or RetrieverConfig()
        self._storage = StorageAdapter(paths=storage_paths)
        self._chat = AsyncChat(system_prompt=system_prompt)

    # ------------------------------------------------------------------
    # Retrieval entry points
    # ------------------------------------------------------------------
    async def aretrieve(self, question: str) -> RetrievalResult:
        LOGGER.info("Running retrieval for query: %s", question)
        cfg = self._config

        entity_matches = self._storage.query_entities(question, limit=cfg.entity_top_k)
        relation_matches = self._storage.query_relations(question, limit=cfg.relation_top_k)
        chunk_matches = self._storage.query_chunks(question, limit=cfg.chunk_top_k)

        graph_windows = build_graph_windows(
            self._storage,
            entity_matches,
            depth=cfg.graph_depth,
            max_windows=cfg.graph_windows,
            max_tokens=cfg.graph_window_tokens,
            model=cfg.tiktoken_model,
        )
        chunk_windows = build_chunk_windows(
            chunk_matches,
            max_windows=cfg.chunk_windows,
            max_tokens=cfg.chunk_window_tokens,
            model=cfg.tiktoken_model,
        )
        context_windows = merge_windows(graph_windows, chunk_windows)
        context_block = format_windows_for_prompt(context_windows)

        prompt = RAG_PROMPT.format(context=context_block, question=question)
        answer = await self._chat.generate(
            prompt,
            max_tokens=cfg.llm_max_tokens,
            temperature=cfg.llm_temperature,
        )
        return RetrievalResult(
            answer=answer,
            context_windows=context_windows,
            entity_matches=entity_matches,
            relation_matches=relation_matches,
            chunk_matches=chunk_matches,
        )

    def retrieve(self, question: str) -> RetrievalResult:
        """Synchronous helper that creates an event loop if needed."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve(question))
        else:  # pragma: no cover - usage depends on embedding application
            raise RuntimeError(
                "retrieve() cannot be used when an event loop is already running. "
                "Use 'await aretrieve(...)' instead."
            )
