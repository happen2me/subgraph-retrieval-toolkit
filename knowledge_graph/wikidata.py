from .knowledge_graph import KnowledgeGraph
from SPARQLWrapper import SPARQLWrapper, JSON


def get_pid_from_uri(uri):
    """Get property id from uri."""
    return uri.split('/')[-1]


class Wikidata:
    PREFIXES: str = """PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX wdv: <http://www.wikidata.org/value/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        """

    def __init__(self, endpoint, prepend_prefixes=False):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.prepend_prefixes = prepend_prefixes
    
    def queryWikidata(self, query):
        if self.prepend_prefixes:
            query = self.PREFIXES + query

        self.sparql.setQuery(query)
        try:
            ret = self.sparql.queryAndConvert()
            result = ret['results']['bindings']
        except Exception as exeption:
            print(exeption)
            print(f'Failed executing query: {query}')
            result = []
        return result

    def search_one_hop_relation(self, src, dst):
        """Search one hop relation between src and dst.
        
        Args:
            src (str): source entity
            dst (str): destination entity
        
        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        query = f"""
            SELECT DISTINCT ?r WHERE {{
                wd:{src} ?r wd:{dst}.
            }}
            """
        paths = self.queryWikidata(query)
        # Keep only PIDs in the paths
        paths = [[get_pid_from_uri(path['r']['value'])] for path in paths]
        return paths

    def search_two_hop_relation(self, src, dst):
        """Search two hop relation between src and dst.
        
        Args:
            src (str): source entity
            dst (str): destination entity
        
        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        query = f"""
            SELECT DISTINCT ?r1 ?r2 WHERE {{
                wd:{src} ?r1 ?x.
                ?x ?r2 wd:{dst}.
            }}
            """
        paths = self.queryWikidata(query)
        # Keep only PIDs in the paths
        paths = [[get_pid_from_uri(path['r1']['value']),
                  get_pid_from_uri(path['r2']['value'])]
                 for path in paths]
        return paths

    def deduce_leaves(self, src, path):
        """Deduce leave entities from source entity following the path.
        
        Args:
            src_entity (str): source entity
            path (list[str]): path from source entity to destination entity
        
        Returns:
            list[str]: list of leaves. Each leaf is a QID.
        """
        assert len(path) < 3, f'Currenly only support paths with length less than 3, got {len(path)}'
        if len(path) == 0:
            return [src]
        if len(path) == 1:
            query = f"""
                SELECT DISTINCT ?x WHERE {{
                    wd:{src} wdt:{path[0]} ?x.
                    }}
                LIMIT 2000
            """
        else: # len(path) == 2
            query = f"""
                SELECT DISTINCT ?x WHERE {{
                    wd:{src} wdt:{path[0]} ?y.
                    ?y wdt:{path[1]} ?x.
                }}
                LIMIT 2000
            """
        if self.prepend_prefixes:
            query = self.PREFIXES + query
        leaves = self.queryWikidata(query)
        # Keep only QIDs in the leaves
        leaves = [leaf['x']['value'].split('/')[-1] for leaf in leaves]
        return leaves
