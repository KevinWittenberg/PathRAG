"""Utilities that bridge the ingestion storage with the PathRAG retriever."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from .base import ChunkMatch, EntityMatch, RelationMatch

try:  # pragma: no cover - the ingestion package provides this module
    from storage import Storage as IngestionStorage
    from storage import StoragePaths
except ImportError as exc:  # pragma: no cover - surface a helpful error message
    raise ImportError(
        "The ingestion storage module could not be imported. "
        "Ensure that the PathRAG ingestion package is installed and available "
        "on PYTHONPATH before initialising the retriever."
    ) from exc

LOGGER = logging.getLogger("PathRAG")


@dataclass(frozen=True)
class GraphSnapshot:
    """A cached view of the knowledge graph stored in SQLite."""

    graph: nx.Graph


class StorageAdapter:
    """High level helper around the ingestion ``Storage`` facade.

    The adapter exposes convenience methods that are tailored for retrieval use
    cases. It translates low level database rows and Chroma query responses into
    the dataclasses used by the simplified retriever implementation.
    """

    def __init__(self, paths: Optional[StoragePaths] = None):
        self._storage = IngestionStorage(paths=paths)
        # Ensure tables exist before we start querying them.
        self._storage.init()
        self._graph_snapshot: Optional[GraphSnapshot] = None

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    @property
    def graph(self) -> nx.Graph:
        if self._graph_snapshot is None:
            self._graph_snapshot = self._load_graph()
        return self._graph_snapshot.graph

    def refresh_graph(self) -> None:
        """Reload the graph from SQLite."""

        self._graph_snapshot = self._load_graph()

    def _load_graph(self) -> GraphSnapshot:
        graph = nx.Graph()

        with self._storage.graph.connect() as con:
            node_rows = con.execute(
                "SELECT name, type, description, source_id, filepath FROM nodes;"
            ).fetchall()
            edge_rows = con.execute(
                "SELECT source_name, target_name, weight, description, keywords, "
                "source_id, filepath FROM edges;"
            ).fetchall()

        for name, type_, description, source_id, filepath in node_rows:
            node_id = (name or "").strip()
            if not node_id:
                continue
            graph.add_node(
                node_id,
                type=(type_ or "unknown").strip() or "unknown",
                description=(description or "").strip(),
                source_id=(source_id or "").strip(),
                filepath=(filepath or "").strip(),
            )

        for row in edge_rows:
            source, target, weight, description, keywords, source_id, filepath = row
            src_id = (source or "").strip()
            tgt_id = (target or "").strip()
            if not src_id or not tgt_id:
                continue
            if src_id not in graph or tgt_id not in graph:
                # The ingestion pipeline guarantees referential integrity, but we
                # guard here to avoid breaking retrieval if data becomes skewed.
                LOGGER.debug("Skipping edge with missing endpoints: %s -> %s", src_id, tgt_id)
                continue
            graph.add_edge(
                src_id,
                tgt_id,
                weight=float(weight) if weight is not None else 1.0,
                description=(description or "").strip(),
                keywords=(keywords or "").strip(),
                source_id=(source_id or "").strip(),
                filepath=(filepath or "").strip(),
            )

        LOGGER.debug(
            "Loaded graph snapshot with %d nodes and %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return GraphSnapshot(graph=graph)

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        node_name = name.strip()
        if not node_name:
            return None
        graph = self.graph
        if node_name not in graph:
            return None
        data = dict(graph.nodes[node_name])
        data["name"] = node_name
        return data

    def get_neighbors(self, name: str) -> List[str]:
        graph = self.graph
        if name not in graph:
            return []
        return list(graph.neighbors(name))

    # ------------------------------------------------------------------
    # Vector queries
    # ------------------------------------------------------------------
    def query_entities(self, text: str, limit: int = 5) -> List[EntityMatch]:
        results = self._storage.entity_vectors.query(text=text, n_results=limit) or []
        matches: List[EntityMatch] = []
        for result in results:
            metadata = result.get("metadatas") or {}
            matches.append(
                EntityMatch(
                    name=metadata.get("name", result.get("ids", "")),
                    type=metadata.get("type"),
                    description=metadata.get("description", ""),
                    score=_distance_to_similarity(result.get("distances")),
                )
            )
        return matches

    def query_relations(self, text: str, limit: int = 5) -> List[RelationMatch]:
        results = self._storage.relation_vectors.query(text=text, n_results=limit) or []
        matches: List[RelationMatch] = []
        for result in results:
            metadata = result.get("metadatas") or {}
            matches.append(
                RelationMatch(
                    source_name=metadata.get("source_name", ""),
                    target_name=metadata.get("target_name", ""),
                    description=metadata.get("description", ""),
                    keywords=metadata.get("keywords", ""),
                    score=_distance_to_similarity(result.get("distances")),
                )
            )
        return matches

    def query_chunks(self, text: str, limit: int = 5) -> List[ChunkMatch]:
        results = self._storage.chunk_vectors.query(text=text, n_results=limit) or []
        matches: List[ChunkMatch] = []
        for result in results:
            metadata = result.get("metadatas") or {}
            matches.append(
                ChunkMatch(
                    chunk_uuid=result.get("ids", ""),
                    document_id=str(metadata.get("doc_id", "")),
                    filename=str(metadata.get("filename", "")),
                    text=result.get("documents", ""),
                    score=_distance_to_similarity(result.get("distances")),
                )
            )
        return matches

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def describe_subgraph(self, seeds: Sequence[str], depth: int = 2) -> List[Tuple[str, str]]:
        graph = self.graph
        descriptions: List[Tuple[str, str]] = []
        for seed in seeds:
            if seed not in graph:
                continue
            visited = {seed}
            frontier = [(seed, 0)]
            nodes_in_scope = {seed}
            while frontier:
                current, dist = frontier.pop(0)
                if dist >= depth:
                    continue
                for neighbor in graph.neighbors(current):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    nodes_in_scope.add(neighbor)
                    frontier.append((neighbor, dist + 1))

            subgraph = graph.subgraph(nodes_in_scope)
            descriptions.append((seed, self._format_subgraph(seed, subgraph)))
        return descriptions

    @staticmethod
    def _format_subgraph(seed: str, subgraph: nx.Graph) -> str:
        node_lines = []
        for node, data in subgraph.nodes(data=True):
            prefix = "Seed" if node == seed else "Node"
            node_lines.append(
                f"{prefix} {node} ({data.get('type', 'unknown')}): {data.get('description', '')}"
            )

        edge_lines = []
        for src, tgt, data in subgraph.edges(data=True):
            edge_lines.append(
                f"Relation {src} ↔ {tgt}: {data.get('description', '')} | Keywords: {data.get('keywords', '')}"
            )
        return "\n".join(node_lines + edge_lines)

    def get_documents(self, doc_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        documents: Dict[str, Dict[str, Any]] = {}
        for doc_id in doc_ids:
            if not doc_id or doc_id in documents:
                continue
            record = self._storage.get_document(doc_id)
            if record:
                documents[doc_id] = record
        return documents


def _distance_to_similarity(value: Any) -> float:
    try:
        return max(0.0, 1.0 - float(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0
