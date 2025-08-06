"""Standalone retrieval routines for PathRAG.

This module bundles all graph/text retrieval helpers so it can be reused in
other projects without pulling in the ingestion pipeline.  The functions here
operate on the abstract storage interfaces defined in :mod:`PathRAG.base` and
consume a :class:`RetrievalConfig` describing concrete storage backends.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import List, Tuple

import networkx as nx

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .utils import (
    logger,
    list_of_list_to_csv,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .retrieval_config import RetrievalConfig


async def find_paths_and_edges_with_stats(graph, target_nodes: List[str]):
    """Depth limited DFS to enumerate paths up to three hops."""
    result = defaultdict(lambda: {"paths": [], "edges": set()})
    path_stats = {"1-hop": 0, "2-hop": 0, "3-hop": 0}
    one_hop_paths: List[List[str]] = []
    two_hop_paths: List[List[str]] = []
    three_hop_paths: List[List[str]] = []

    async def dfs(current, target, path, depth):
        if depth > 3:
            return
        if current == target:
            result[(path[0], target)]["paths"].append(list(path))
            for u, v in zip(path[:-1], path[1:]):
                result[(path[0], target)]["edges"].add(tuple(sorted((u, v))))
            if depth == 1:
                path_stats["1-hop"] += 1
                one_hop_paths.append(list(path))
            elif depth == 2:
                path_stats["2-hop"] += 1
                two_hop_paths.append(list(path))
            elif depth == 3:
                path_stats["3-hop"] += 1
                three_hop_paths.append(list(path))
            return
        for neighbor in graph.neighbors(current):
            if neighbor not in path:
                await dfs(neighbor, target, path + [neighbor], depth + 1)

    for node1 in target_nodes:
        for node2 in target_nodes:
            if node1 != node2:
                await dfs(node1, node2, [node1], 0)

    for key in result:
        result[key]["edges"] = list(result[key]["edges"])

    return dict(result), path_stats, one_hop_paths, two_hop_paths, three_hop_paths


def bfs_weighted_paths(G, path, source, target, threshold, alpha):
    """Breadth-first path scoring used for relation ranking."""
    results = []
    edge_weights = defaultdict(float)
    node = source
    follow_dict = {}

    for p in path:
        for i in range(len(p) - 1):
            current = p[i]
            next_num = p[i + 1]
            follow_dict.setdefault(current, set()).add(next_num)

    for neighbor in follow_dict.get(node, []):
        edge_weights[(node, neighbor)] += 1 / len(follow_dict[node])
        if neighbor == target:
            results.append([node, neighbor])
            continue
        if edge_weights[(node, neighbor)] > threshold:
            for second_neighbor in follow_dict.get(neighbor, []):
                weight = edge_weights[(node, neighbor)] * alpha / len(
                    follow_dict[neighbor]
                )
                edge_weights[(neighbor, second_neighbor)] += weight
                if second_neighbor == target:
                    results.append([node, neighbor, second_neighbor])
                    continue
                if edge_weights[(neighbor, second_neighbor)] > threshold:
                    for third_neighbor in follow_dict.get(second_neighbor, []):
                        weight = (
                            edge_weights[(neighbor, second_neighbor)]
                            * alpha
                            / len(follow_dict[second_neighbor])
                        )
                        edge_weights[(second_neighbor, third_neighbor)] += weight
                        if third_neighbor == target:
                            results.append(
                                [node, neighbor, second_neighbor, third_neighbor]
                            )
                            continue
    path_weights = []
    for p in path:
        path_weight = 0
        for i in range(len(p) - 1):
            edge = (p[i], p[i + 1])
            path_weight += edge_weights.get(edge, 0)
        path_weights.append(path_weight / (len(p) - 1))

    return list(zip(path, path_weights))


async def _find_most_related_text_unit_from_entities(
    node_datas: List[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [","])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [","]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }
            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    return [t["data"] for t in all_text_units]


async def _find_most_related_edges_from_entities3(
    node_datas: List[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    G = nx.Graph()
    edges = await knowledge_graph_inst.edges()
    nodes = await knowledge_graph_inst.nodes()
    for u, v in edges:
        G.add_edge(u, v)
    G.add_nodes_from(nodes)
    source_nodes = [dp["entity_name"] for dp in node_datas]
    result, path_stats, one_hop_paths, two_hop_paths, three_hop_paths = await find_paths_and_edges_with_stats(
        G, source_nodes
    )

    threshold = 0.3
    alpha = 0.8
    all_results = []
    for node1 in source_nodes:
        for node2 in source_nodes:
            if node1 != node2 and (node1, node2) in result:
                sub_G = nx.Graph()
                paths = result[(node1, node2)]["paths"]
                edges = result[(node1, node2)]["edges"]
                sub_G.add_edges_from(edges)
                results = bfs_weighted_paths(G, paths, node1, node2, threshold, alpha)
                all_results += results
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    seen = set()
    result_edge = []
    for edge, weight in all_results:
        sorted_edge = tuple(sorted(edge))
        if sorted_edge not in seen:
            seen.add(sorted_edge)
            result_edge.append((edge, weight))

    length_1 = int(len(one_hop_paths) / 2)
    length_2 = int(len(two_hop_paths) / 2)
    length_3 = int(len(three_hop_paths) / 2)
    results = []
    if one_hop_paths:
        results = one_hop_paths[0:length_1]
    if two_hop_paths:
        results = results + two_hop_paths[0:length_2]
    if three_hop_paths:
        results = results + three_hop_paths[0:length_3]

    total_edges = min(len(results), 15)
    sort_result = result_edge[0:total_edges] if result_edge else []

    final_result = [edge for edge, _ in sort_result]

    relationship = []
    for path in final_result:
        if len(path) == 4:
            s_name, b1_name, b2_name, t_name = path
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1], path[2]) or await knowledge_graph_inst.get_edge(path[2], path[1])
            edge2 = await knowledge_graph_inst.get_edge(path[2], path[3]) or await knowledge_graph_inst.get_edge(path[3], path[2])
            if edge0 is None or edge1 is None or edge2 is None:
                continue
            e1 = f"through edge ({edge0['keywords']}) to connect to {s_name} and {b1_name}."
            e2 = f"through edge ({edge1['keywords']}) to connect to {b1_name} and {b2_name}."
            e3 = f"through edge ({edge2['keywords']}) to connect to {b2_name} and {t_name}."
            s = await knowledge_graph_inst.get_node(s_name)
            b1 = await knowledge_graph_inst.get_node(b1_name)
            b2 = await knowledge_graph_inst.get_node(b2_name)
            t = await knowledge_graph_inst.get_node(t_name)
            relationship.append([
                "The entity "
                + s_name
                + " is a "
                + s["entity_type"]
                + " with the description(" + s["description"] + ")"
                + e1
                + "The entity "
                + b1_name
                + " is a "
                + b1["entity_type"]
                + " with the description(" + b1["description"] + ")"
                + "and"
                + "The entity "
                + b1_name
                + " is a "
                + b1["entity_type"]
                + " with the description(" + b1["description"] + ")"
                + e2
                + "The entity "
                + b2_name
                + " is a "
                + b2["entity_type"]
                + " with the description(" + b2["description"] + ")"
                + "and"
                + "The entity "
                + b2_name
                + " is a "
                + b2["entity_type"]
                + " with the description(" + b2["description"] + ")"
                + e3
                + "The entity "
                + t_name
                + " is a "
                + t["entity_type"]
                + " with the description(" + t["description"] + ")"
            ])
        elif len(path) == 3:
            s_name, b_name, t_name = path
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1], path[2]) or await knowledge_graph_inst.get_edge(path[2], path[1])
            if edge0 is None or edge1 is None:
                continue
            e1 = f"through edge({edge0['keywords']}) to connect to {s_name} and {b_name}."
            e2 = f"through edge({edge1['keywords']}) to connect to {b_name} and {t_name}."
            s = await knowledge_graph_inst.get_node(s_name)
            b = await knowledge_graph_inst.get_node(b_name)
            t = await knowledge_graph_inst.get_node(t_name)
            relationship.append([
                "The entity "
                + s_name
                + " is a "
                + s["entity_type"]
                + " with the description(" + s["description"] + ")"
                + e1
                + "The entity "
                + b_name
                + " is a "
                + b["entity_type"]
                + " with the description(" + b["description"] + ")"
                + "and"
                + "The entity "
                + b_name
                + " is a "
                + b["entity_type"]
                + " with the description(" + b["description"] + ")"
                + e2
                + "The entity "
                + t_name
                + " is a "
                + t["entity_type"]
                + " with the description(" + t["description"] + ")"
            ])
        elif len(path) == 2:
            s_name, t_name = path
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            if edge0 is None:
                continue
            e = f"through edge({edge0['keywords']}) to connect to {s_name} and {t_name}."
            s = await knowledge_graph_inst.get_node(s_name)
            t = await knowledge_graph_inst.get_node(t_name)
            relationship.append([
                "The entity "
                + s_name
                + " is a "
                + s["entity_type"]
                + " with the description(" + s["description"] + ")"
                + e
                + "The entity "
                + t_name
                + " is a "
                + t["entity_type"]
                + " with the description(" + t["description"] + ")"
            ])

    relationship = truncate_list_by_token_size(
        relationship,
        key=lambda x: x[0],
        max_token_size=query_param.max_token_for_local_context,
    )
    return relationship[::-1]


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
) -> Tuple[str, str, str]:
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not results:
        return "", "", ""
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities3(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [["id", "context"]]
    for i, e in enumerate(use_relations):
        relations_section_list.append([i, e])
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entities_context, relations_context, text_units_context


async def get_context(query: str, config: RetrievalConfig) -> str:
    """Return a single context string suitable for LLM prompts."""
    entities_ctx, relations_ctx, text_ctx = await _get_node_data(
        query,
        config.graph,
        config.entity_store,
        config.text_store,
        config.query_param,
    )
    return "\n\n".join([entities_ctx, relations_ctx, text_ctx])

