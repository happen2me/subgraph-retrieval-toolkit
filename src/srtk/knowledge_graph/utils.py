from .wikidata import Wikidata
from .freebase import Freebase
from .dbpedia import DBpedia


def get_knowledge_graph(knowledge_graph_type, sparql_endpoint, prepend_prefixes=False,
                        exclude_qualifiers=True):
    """Create a knowledge graph object.

    Args:
        knowledge_graph_type (str): Knowledge graph type. One of 'freebase', 'wikidata', 'dbpedia'.
        sparql_endpoint (str): The SPARQL endpoint of the knowledge graph.
        prepend_prefixes (bool): Whether to prepend prefixes to the SPARQL query. Defaults to False.
        exclude_qualifiers (bool, optional): Whether to exclude qualifiers, only valid for Wikidata. Defaults to True.

    Raises:
        ValueError: If the knowledge graph type is not supported.

    Returns:
        KnowledgeGraphBase: A knowledge graph object.
    """
    if knowledge_graph_type == 'freebase':
        kg = Freebase(sparql_endpoint, prepend_prefixes=prepend_prefixes)
    elif knowledge_graph_type == 'wikidata':
        kg = Wikidata(sparql_endpoint, prepend_prefixes=prepend_prefixes, exclude_qualifiers=exclude_qualifiers)
    elif knowledge_graph_type == 'dbpedia':
        kg = DBpedia(sparql_endpoint, prepend_prefixes=prepend_prefixes)
    else:
        raise ValueError(f'Unknown knowledge graph type: {knowledge_graph_type}')
    return kg