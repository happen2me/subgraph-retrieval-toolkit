% srtk documentation master file, created by
% sphinx-quickstart on Sat Apr 22 01:32:21 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# Subgraph Retrieval Toolkit (SRTK)

The Subgraph Retrieval Toolkit (SRTK) is a Python-based toolkit for retrieving subgraphs from large-scale knowledge graphs. SRTK provides a unified interface to access multiple knowledge graphs, including Wikidata and Freebase, and ships with state-of-the-art subgraph retrieval algorithms.

SRTK CLI Interface supports the following subcommands:

- `preprocess`: Preprocess a dataset for training a subgraph retrieval model.
- `train`: Train a subgraph retrieval model on a preprocessed dataset.
- `link`: Link entity mentions in a text to a knowledge graph. Currently only Wikidata is supported out of the box.
- `retrieve`: Retrieve a semantic-relevant subgraph from a knowledge graph with a trained retriever. It can also be used to evaluate a trained retriever.
- `visualize`: Visualize a retrieved subgraph using a graph visualization tool.

```{toctree}
:caption: 'Contents:'

getting_started
cli
```

Besides, `srtk` can also be used as a python library.

```{toctree}
:caption: Python API Reference
:maxdepth: 2

srtk
```

```{eval-rst}
.. automodule:: srtk
   :members:
   :undoc-members:
   :show-inheritance:
```

```{toctree}
:caption: 'Tutorials'

tutorials
```
