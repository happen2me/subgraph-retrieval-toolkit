import socket
from urllib.parse import urlparse

from .knowledge_graph import DBpedia, Freebase, Wikidata, KnowledgeGraphBase

def get_host_port(url):
    """Get the host and port from a URL"""
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port
    if port is None:
        if parsed_url.scheme == 'http':
            port = 80
        elif parsed_url.scheme == 'https':
            port = 443
    return host, port


def socket_reachable(url):
    """Check if a socket is reachable
    """
    host, port = get_host_port(url)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2) # set a timeout value for the socket
        s.connect((host, port))
        s.close()
        return True
    except Exception as err:
        print(err)
        return False

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
