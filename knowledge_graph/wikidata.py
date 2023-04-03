from .knowledge_graph import KnowledgeGraph
from SPARQLWrapper import SPARQLWrapper, JSON

class Wikidata(KnowledgeGraph):
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

    def __init__(self, endpoint='http://localhost:1234/api/endpoint/sparql', prepend_prefixes=False):
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
            print(f'Failed execuring query: {query}')
            result = []
        return result

    def search_one_hop_relation(self, src, dst):
        """Search one hop relation between src and dst."""
        query = f"""
            SELECT DISTINCT ?r WHERE {{
                wd:{src} ?r wd:{dst}.
            }}
            """
        result = self.queryWikidata(query)
        return result
    
    def search_two_hop_relation(self, src, dst):
        """Search two hop relation between src and dst."""
        query = f"""
            SELECT DISTINCT ?r1 ?r2 WHERE {{
                wd:{src} ?r1 ?x.
                ?x ?r2 wd:{dst}.
            }}
            """
        result = self.queryWikidata(query)
        return result


