from functools import lru_cache
from typing import List

from SPARQLWrapper import SPARQLWrapper, JSON

from .graph_base import KnowledgeGraphBase


class DBpedia(KnowledgeGraphBase):
    PREFIXES: str = """PREFIX dbo: <http://dbpedia.org/ontology/>
                       PREFIX dbr: <http://dbpedia.org/resource/>
                       """

    def __init__(self, endpoint, prepend_prefixes=False):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.prepend_prefixes = prepend_prefixes
        self.name = 'dbpedia'

    def queryDBPedia(self, query):
        if self.prepend_prefixes:
            query = self.PREFIXES + query

        self.sparql.setQuery(query)
        try:
            ret = self.sparql.queryAndConvert()
            result = ret['results']['bindings']
        except Exception as exeption:
            print(f'Failed executing query: {query}')
            print(f'Exception: {exeption}')
            result = []
        return result

    def get_id_from_uri(self, uri):
        return uri.split('/')[-1]

    @lru_cache
    def search_one_hop_relations(self, src, dst):
        query = f"""
                SELECT DISTINCT ?r WHERE {{
                    dbr:{src} ?r dbr:{dst}.
                    FILTER regex(str(?r), "^http://dbpedia.org/ontology/")
                    FILTER (?r != dbo:wikiPageWikiLink)
                }}
                """
        paths = self.queryDBPedia(query)
        # Keep only identifiers in the paths
        paths = [[self.get_id_from_uri(path['r']['value'])] for path in paths]
        return paths

    @lru_cache
    def search_two_hop_relations(self, src: str, dst: str) -> List[List[str]]:
        """Search two hop relations between src and dst.

        Args:
            src (str): source entity
            dst (str): destination entity

        Returns:
            list[list[str]]: list of paths, each path is a list of IDs
        """
        query = f"""
                SELECT DISTINCT ?r1 ?r2 WHERE {{
                    dbr:{src} ?r1 ?mid.
                    ?mid ?r2 dbr:{dst}.
                    FILTER regex(str(?r1), "^http://dbpedia.org/ontology/")
                    FILTER regex(str(?r2), "^http://dbpedia.org/ontology/")
                    FILTER (?r1 != dbo:wikiPageWikiLink)
                    FILTER (?r2 != dbo:wikiPageWikiLink)
                }}
                """
        paths = self.queryDBPedia(query)
        # Keep only identifiers in the paths
        paths = [[self.get_id_from_uri(path['r1']['value']), self.get_id_from_uri(path['r2']['value'])] for path in paths]
        return paths

    def deduce_leaves(self, src: str, path: List[str], limit: int) -> List[str]:
        """Deduce leave entities from source entity following the path.

        Args:
            src_entity (str): source entity
            path (tuple[str]): path from source entity to destination entity
            limit (int, optional): limit of the number of leaves.

        Returns:
            list[str]: list of leaves. Each leaf is a ID.
        """
        if len(path) > 3:
            raise NotImplementedError('Deduce leaves for paths longer than 3 is not implemented.')
        
        if len(path) == 0:
            return [src]
        
        if len(path) == 1:
            query = f"""
                SELECT DISTINCT ?dst WHERE {{
                    dbr:{src} dbo:{path[0]} ?dst.
                    FILTER(STRSTARTS(str(?dst), "http://dbpedia.org/resource/"))
                }}
                LIMIT {limit}
                """
        else:
            query = f"""
                SELECT DISTINCT ?dst WHERE {{
                    dbr:{src} dbo:{path[0]} ?mid.
                    ?mid dbo:{path[1]} ?dst.
                    FILTER(STRSTARTS(str(?dst), "http://dbpedia.org/resource/"))
                }}
                LIMIT {limit}
                """
        leaves = self.queryDBPedia(query)
        leaves = [self.get_id_from_uri(leaf['dst']['value']) for leaf in leaves]
        return leaves

    def deduce_leaves_from_multiple_srcs(self, srcs, path, limit=2000):
        """Deuce leave entities from multiple source entities following the path.

        Args:
            srcs (list[str]): list of source entities
            path (list[str]): path from source entity to destination entity
            limit (int, optional): limit of the number of leaves. Defaults to 200.

        Returns:
            list[str]: list of leaves. Each leaf is a QID.
        """
        if len(path) >= 2:
            raise NotImplementedError(f'Currenly only support paths with length less than 2, got {len(path)}')
        if len(path) == 0:
            return srcs
        if len(srcs) == 0:
            return []

        query = f"""
            SELECT DISTINCT ?x WHERE {{
                VALUES ?src {{dbr:{' dbr:'.join(srcs)}}}
                ?src dbo:{path[0]} ?x.
                FILTER(STRSTARTS(str(?x), "http://dbpedia.org/resource/"))
            }}
            LIMIT {limit}
            """
        if self.prepend_prefixes:
            query = self.PREFIXES + query
        leaves = self.queryDBPedia(query)
        # Keep only QIDs in the leaves
        leaves = [leaf['x']['value'].split('/')[-1] for leaf in leaves]
        return leaves

    @lru_cache
    def get_neighbor_relations(self, src, hop=1, limit=100):
        """Get all relations connected to an entity. The relations are
        limited to direct relations (those with wdt: prefix).

        Args:
            src (str): source entity
            hop (int, optional): hop of the relations. Defaults to 1.
            limit (int, optional): limit of the number of relations. Defaults to 100.

        Returns:
            list[str] | list[tuple(str,)]: list of relations. Each relation is a PID or a tuple of PIDs.
        """
        if hop > 2:
            raise NotImplementedError('Get neighbor relations for hop larger than 2 is not implemented.')

        if hop == 1:
            query = f"""
                SELECT DISTINCT ?r
                WHERE {{
                    dbr:Charles_III ?r ?neighbor .
                    FILTER (STRSTARTS(STR(?r), "http://dbpedia.org/ontology/") && !STRSTARTS(STR(?r), "http://dbpedia.org/ontology/wiki"))
                }}
                LIMIT {limit}
                """
        else:
            query = f"""
                SELECT DISTINCT ?r1 ?r2 WHERE {{
                    dbr:{src} ?r1 ?mid.
                    ?mid ?r2 ?dst.
                    FILTER (STRSTARTS(STR(?r1), "http://dbpedia.org/ontology/") && !STRSTARTS(STR(?r1), "http://dbpedia.org/ontology/wiki"))
                    FILTER (STRSTARTS(STR(?r2), "http://dbpedia.org/ontology/") && !STRSTARTS(STR(?r2), "http://dbpedia.org/ontology/wiki"))
                }}
                LIMIT {limit}
                """

        relations = self.queryDBPedia(query)

        if hop == 1:
            relations = [self.get_id_from_uri(relation['r']['value'])
                         for relation in relations]
        else:
            relations = [(self.get_id_from_uri(relation['r1']['value']),
                          self.get_id_from_uri(relation['r2']['value']))
                         for relation in relations]
        return relations

    @lru_cache
    def get_label(self, identifier):
        """Get label of an entity or a relation. If no label is found, return None.

        Args:
            identifier (str): entity or relation, a QID or a PID

        Returns:
            str | None: label of the entity or relation
        """
        query = f"""
                SELECT (str(?label) AS ?name)
                WHERE {{
                dbr:{identifier} rdfs:label ?label .
                FILTER (lang(?label) = "en")
                }}
                LIMIT 1
                """
        labels = self.queryDBPedia(query)
        if len(labels) == 0:
            print(f'No label found for {identifier}')
            return None
        label = labels[0]['name']['value']
        return label
