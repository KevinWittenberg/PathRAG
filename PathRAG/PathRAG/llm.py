"""Async bridge to the ingestion package's chat helper."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional


try:  # pragma: no cover - consumers are expected to provide this module
    from llm import Chat as IngestionChat
except ImportError as exc:  # pragma: no cover - surface a useful error message
    raise ImportError(
        "The ingestion 'llm' module could not be imported. Ensure the PathRAG "
        "ingestion package (or equivalent) is available on PYTHONPATH before "
        "initialising the retriever."
    ) from exc


@dataclass
class AsyncChat:
    """Lightweight async wrapper around :class:`llm.Chat`."""

    system_prompt: Optional[str] = None

    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Run :meth:`llm.Chat.generate` in a worker thread."""

        chat = IngestionChat.singleton()
        return await asyncio.to_thread(
            chat.generate,
            prompt,
            system=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
