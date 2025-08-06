# PathRAG Retrieval

Standalone retrieval stage for the paper **"PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths"**. The package queries a knowledge graph, ranks relational paths and gathers supporting text chunks to build context for language model prompts.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Implement the storage interfaces in `PathRAG/base.py` for your own knowledge graph, entity vector store and text chunk store. Instantiate these backends and pass them to `RetrievalConfig` from `PathRAG/retrieval_config.py`.

```python
import asyncio
from PathRAG import (
    DEFAULT_QUERY_PARAM,
    RetrievalConfig,
    get_context,
)

# graph, entity_store and text_store are your implementations of the
# BaseGraphStorage, BaseVectorStorage and BaseKVStorage interfaces.
config = RetrievalConfig(graph, entity_store, text_store, DEFAULT_QUERY_PARAM)

context = asyncio.run(get_context("your question", config))
print(context)
```

The `get_context` function returns a string containing entity summaries, relation paths and related text chunks. Insert this string into the `{context}` portion of your LLM prompt.

## Code structure

- `base.py` – abstract storage interfaces and query parameters.
- `retrieval.py` – graph and text retrieval routines.
- `retrieval_config.py` – configuration container used to wire in your storage backends.
- `utils.py` – helper utilities.

## Cite

Please cite our paper if you use this code in your own work:

```bibtex
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```
