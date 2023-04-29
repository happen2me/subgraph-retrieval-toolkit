from functools import lru_cache
from SPARQLWrapper import SPARQLWrapper, JSON

from .graph_base import KnowledgeGraphBase


class Freebase(KnowledgeGraphBase):
    PREFIXES: str = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns: <http://rdf.freebase.com/ns/>
        """

    def __init__(self, endpoint, prepend_prefixes=True) -> None:
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.prepend_prefixes = prepend_prefixes
        self.name = 'freebase'

    def queryFreebase(self, query):
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

    @staticmethod
    def get_id_from_uri(uri):
        """Get id from uri."""
        return uri.split('/')[-1]

    @lru_cache
    def search_one_hop_relations(self, src, dst):
        """Search one hop relation between src and dst.
        
        Args:
            src (str): source entity
            dst (str): destination entity
        
        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        query = f"""
            SELECT distinct ?r1 where {{
                ns:{src} ?r1_ ns:{dst} . 
                FILTER REGEX(?r1_, "http://rdf.freebase.com/ns/")
                BIND(STRAFTER(STR(?r1_),str(ns:)) AS ?r1)
            }}
        """
        paths = self.queryFreebase(query)
        paths = [[path['r1']['value']] for path in paths]
        return paths

    @lru_cache
    def search_two_hop_relations(self, src, dst):
        query = f"""
            SELECT distinct ?r1 ?r2 where {{
                ns:{src} ?r1_ ?e1 . 
                ?e1 ?r2_ ns:{dst} .
                FILTER REGEX(?e1, "http://rdf.freebase.com/ns/")
                FILTER REGEX(?r1_, "http://rdf.freebase.com/ns/")
                FILTER REGEX(?r2_, "http://rdf.freebase.com/ns/")
                FILTER (?r1_ != <http://rdf.freebase.com/ns/type.object.type>)
                FILTER (?r2_ != <http://rdf.freebase.com/ns/type.object.type>)
                BIND(STRAFTER(STR(?r1_),str(ns:)) AS ?r1)
                BIND(STRAFTER(STR(?r2_),str(ns:)) AS ?r2)
            }}
        """
        paths = self.queryFreebase(query)
        paths = [[path['r1']['value'], path['r2']['value']] for path in paths]
        return paths

    @lru_cache
    def deduce_leaves(self, src, path, limit=2000):
        """Deduce leave entities from source entity following the path.
        
        Args:
            src_entity (str): source entity
            path (tuple[str]): path from source entity to destination entity
            limit (int, optional): limit of the number of leaves. Defaults to 2000.
        
        Returns:
            list[str]: list of leaves. Each leaf is a QID.
        """
        if len(path) >= 3:
            raise NotImplementedError(f'Currenly only support paths with length less than 3, got {len(path)}')
        if len(path) == 0:
            return [src]
        if len(path) == 1:
            query = f"""
                SELECT DISTINCT ?leaf WHERE {{
                    ns:{src} ns:{path[0]} ?t0_ .
                    FILTER REGEX(?t0_, "http://rdf.freebase.com/ns/")
                    BIND(STRAFTER(STR(?t0_),STR(ns:)) AS ?leaf)
                }} LIMIT {limit}
                """
        else: # len(path) == 2:
            query = f"""
                SELECT DISTINCT ?leaf WHERE {{
                    ns:{src} ns:{path[0]} ?e1_ . 
                    ?e1_ ns:{path[1]} ?e2_ .
                    FILTER REGEX(?e1_, "http://rdf.freebase.com/ns/")
                    FILTER REGEX(?e2_, "http://rdf.freebase.com/ns/")
                    BIND(STRAFTER(STR(?e2_),str(ns:)) AS ?leaf)
                }} LIMIT {limit}
                """
        results = self.queryFreebase(query)
        return [i['leaf']['value'] for i in results]

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
        query = f"""
            SELECT DISTINCT ?leaf WHERE {{
                VALUES ?src {{ns:{' ns:'.join(srcs)}}}
                ?src ns:{path[0]} ?t0_ .
                FILTER REGEX(?t0_, "http://rdf.freebase.com/ns/")
                BIND(STRAFTER(STR(?t0_),STR(ns:)) AS ?leaf)
            }} LIMIT {limit}
            """
        results = self.queryFreebase(query)
        return [i['leaf']['value'] for i in results]

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
        if hop >= 3:
            raise NotImplementedError(f'Currenly only support relations with hop less than 3, got {hop}')
        if hop == 1:
            query = f"""
                SELECT DISTINCT ?rel WHERE {{
                    ns:{src} ?r0_ ?t0 .
                    FILTER REGEX(?r0_, "http://rdf.freebase.com/ns/")
                    FILTER REGEX(?t0, "http://rdf.freebase.com/ns/")
                    FILTER (?r0_ != <http://rdf.freebase.com/ns/type.object.type>)
                    BIND(STRAFTER(STR(?r0_),STR(ns:)) AS ?rel)
                }} LIMIT {limit}
                """
        elif hop == 2:
            query = f"""
                SELECT DISTINCT ?rel0, ?rel1 WHERE {{     
                    {{
                        SELECT DISTINCT ?t0, ?rel0 WHERE {{
                        ns:{src} ?r0_ ?t0 .
                        FILTER REGEX(?r0_, "http://rdf.freebase.com/ns/")
                        FILTER REGEX(?t0, "http://rdf.freebase.com/ns/")
                        FILTER (?r0_ != <http://rdf.freebase.com/ns/type.object.type>)
                        BIND(STRAFTER(STR(?r0_),STR(ns:)) AS ?rel0)
                        }} LIMIT 10
                    }}
                    ?t0 ?r1_ ?t1 .
                    FILTER REGEX(?r1_, "http://rdf.freebase.com/ns/")
                    FILTER REGEX(?t1, "http://rdf.freebase.com/ns/")
                    FILTER (?r1_ != <http://rdf.freebase.com/ns/type.object.type>)
                    BIND(STRAFTER(STR(?r1_),STR(ns:)) AS ?rel1)
                }} LIMIT {limit}
                """
        results = self.queryFreebase(query)
        if hop == 1:
            paths = [path['rel']['value'] for path in results]
        else:
            paths = [(path['rel0']['value'], path['rel1']['value'])
                     for path in results]
        return paths

    @lru_cache
    def get_label(self, identifier):
        query = f"""
            SELECT ?label
            WHERE {{
                ns:{identifier} rdfs:label ?label .
                FILTER (langMatches(lang(?label), "EN"))
            }} LIMIT 1
            """
        results = self.queryFreebase(query)
        return results[0]['label']['value'] if results else None

    def get_relation_label(self, relation):
        """For freebase, relation label is the same as the relation identifier."""
        return relation
