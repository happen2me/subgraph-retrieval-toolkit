
# Subgraph Retrieval Toolkit: Retrieving Semantic-relevant Subgraphs from Large-scale Knowledge Graphs


Alternative: 
- Subgraph Retrieval from Large-Scale Knowledge Graphs: A Toolkit for Semantic Analysis
- Subgraph Retrieval Toolkit: A Toolkit for Retrieving Semantic-relevant Subgraphs from Large-scale Knowledge Graphs
- Subgraph Retrieval Toolkit: (Unified) Semantic-Driven Subgraph Retrieval from Wikidata and Freebase
- Universal Retrieval of Semantic-Relevant Subgraphs from Knowledge Graphs for Question Answering

## Abstract

Version 1:
Retrieving a resonably small but highly relevant subgraph is a crucial step for knowledge graph related semantic analysis. Existing approaches are flawed in either entity linking or path expansion. Besides, many of them are based on alreay-ceased Freebase. Moreover, most of the existing approaches are not easily extensible to new knowledge graphs and subgraph retrieval algorithms. In this paper, we present a toolkit for subgraph retrieval from large-scale knowledge graphs. The toolkit provides a unified interface to access different knowledge graphs, and ships with the state-of-the-art subgraph retrieval algorithm. We also provide a visualization tool to visualize the retrieved subgraphs. We evaluate the toolkit on two large-scale knowledge graphs, Wikidata and Freebase, and show that the toolkit can retrieve the relevant subgraphs for a given query. We also expose inferfaces that allows easy extension to new knowledge graphs, datasets and subgraph retrieval algorithms.

Version2:
A reasonably small but concise subgraph is crucial for knowledge graph based semantic analysis. Three prevalent problems in existing approaches hinder researchers from integrating relevant subgraphs into their applications. First, there is not an off-the-shelf framework for semantic-relevant subgraph retrieval. Second, existing retrieval methods are KG-dependent. Even the most recent studies base their model on Freebase, which ceased in updating in 2015, because of the difficulty in adapting predesessor methods. Third, existing approaches are either flawed in entity linking or path expanding. In this paper, we present a toolkit for subgraph retrieval from large-scale knowledge graphs. The toolkit provides a unified interface to access different knowledge graphs, and ships with the state-of-the-art subgraph retrieval algorithms that can be used out of box. We also provide a visualization tool to visualize the retrieved subgraphs. We evaluate the toolkit on two large-scale knowledge graphs, Wikidata and Freebase, and show that the toolkit can retrieve the relevant subgraphs for a given query. We also show that the toolkit can be easily extended to new knowledge graphs and subgraph retrieval algorithms.

Knowledge graph provides a structural data source for semantic analysis. To facilitate the semantic analysis, it is important to retrieve the relevant subgraphs from the knowledge graph for a given query.

However, the large-scale and complex nature of knowledge graphs makes it difficult to retrieve the relevant subgraphs for a given query. In this paper, we present a toolkit for subgraph retrieval from large-scale knowledge graphs. The toolkit provides a unified interface to access different knowledge graphs, and ships with the state-of-the-art subgraph retrieval algorithms. We also provide a visualization tool to visualize the retrieved subgraphs. We evaluate the toolkit on two large-scale knowledge graphs, Wikidata and Freebase, and show that the toolkit can retrieve the relevant subgraphs for a given query. We also show that the toolkit can be easily extended to new knowledge graphs and subgraph retrieval algorithms.

## Core Features

- An off-the-shelf toolkit for subgraph retrieval
  From my personal research experience, it is extremely difficult to find a usable toolkit for subgraph retrieval. There exists significant problems with even the most recent and popular retrieval studies, either in entity linking or path expansion. For example, DRAGON uses lexical-based mention match for entity linking, and expanding the subgraph by blindly adding all neighbors connected to the recognized entites, resulting in virtually no useful information in the subgraph. EXAQT adopts semantic based entity linking but expands the subgraph by adding all neighbors connected to the recognized entities,
  Most of the existing papers either rely on very coarse lexical-based mention matching and expanding, or 

- For Freebase, Wikidata and more
  Most of the current research on subgraph retrieval are still using the already ceased Freebase. Even the most recent researches still sticks to this outdated data source [][][]. This potentially  This toolkit provides a unified interface to access different knowledge graphs, which is can accelerate the migration of existing research to the most recent knowledge graphs.

- Visualization of subgraphs

- Ships with the state of the art subgraph retrieval algorithms

- Extensible to new knowledge graphs and subgraph retrieval algorithms

- Cached knowledge graph for fast access


## References

- [1]: Easily setting up a local Wikidata SPARQL endpoint using the qEndpoint