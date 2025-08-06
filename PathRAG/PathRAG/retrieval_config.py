"""Configuration placeholders for PathRAG retrieval stage.

This module centralizes all external connections required by the retrieval
functions. Replace the stub implementations with your own knowledge graph,
vector database and text chunk store backends.
"""
from dataclasses import dataclass
from .base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage, QueryParam


@dataclass
class RetrievalConfig:
    """Holds all dependencies needed for retrieval.

    graph:      Storage providing knowledge graph access.
    entity_store: Vector DB for entity lookups.
    text_store:   KV store containing text chunks.
    query_param:  Parameters controlling retrieval sizes.
    """
    graph: BaseGraphStorage
    entity_store: BaseVectorStorage
    text_store: BaseKVStorage
    query_param: QueryParam


# ---------------------------------------------------------------------------
# Example placeholders
# ---------------------------------------------------------------------------
# Paths to prebuilt resources. Update these to point at your own files or
# connection strings.
GRAPH_PATH = "graph_chunk_entity_relation.graphml"
ENTITY_VECTOR_PATH = "entity_vectors.db"
TEXT_CHUNK_PATH = "text_chunks.db"

# Default query configuration used by :func:`get_context`.
DEFAULT_QUERY_PARAM = QueryParam()

# Consumers are expected to build concrete storage instances and populate the
# RetrievalConfig, for example:
#
# from PathRAG.storage import NetworkXStorage, NanoVectorDBStorage, JsonKVStorage
# from PathRAG.llm import OpenAIEmbedding
#
# embedding = OpenAIEmbedding("text-embedding-3-large")
# graph = NetworkXStorage(namespace="kg", global_config={"working_dir": "."})
# entity_store = NanoVectorDBStorage(namespace="entities", global_config={"working_dir": "."}, embedding_func=embedding)
# text_store = JsonKVStorage(namespace="chunks", global_config={"working_dir": "."}, embedding_func=embedding)
# CONFIG = RetrievalConfig(graph, entity_store, text_store, DEFAULT_QUERY_PARAM)
"""
The CONFIG instance is intentionally left uninitialised so that the retrieval
stage can be "drag-and-dropped" into any project. Construct it with the
storage backends of your choice before calling :func:`get_context`.
"""
