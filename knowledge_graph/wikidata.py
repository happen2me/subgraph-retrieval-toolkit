from functools import lru_cache
from SPARQLWrapper import SPARQLWrapper, JSON

from .base_graph import KnowledgeGraphBase


class Wikidata(KnowledgeGraphBase):
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

    @staticmethod
    def get_pid_from_uri(uri):
        """Get property id from uri."""
        return uri.split('/')[-1]

    def search_one_hop_relations(self, src, dst):
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
                FILTER(STRSTARTS(STR(?dst), "http://www.wikidata.org/entity/Q"))
            }}
            """
        paths = self.queryWikidata(query)
        # Keep only PIDs in the paths
        paths = [[self.get_pid_from_uri(path['r']['value'])] for path in paths]
        return paths

    def search_two_hop_relations(self, src, dst):
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
                FILTER(STRSTARTS(STR(?x), "http://www.wikidata.org/entity/Q"))
            }}
            """
        paths = self.queryWikidata(query)
        # Keep only PIDs in the paths
        paths = [[self.get_pid_from_uri(path['r1']['value']),
                  self.get_pid_from_uri(path['r2']['value'])]
                 for path in paths]
        return paths

    @lru_cache
    def deduce_leaves(self, src, path, limit=2000):
        """Deduce leave entities from source entity following the path.
        
        Args:
            src_entity (str): source entity
            path (list[str]): path from source entity to destination entity
            limit (int, optional): limit of the number of leaves. Defaults to 2000.
        
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
                    FILTER(STRSTARTS(STR(?x), "http://www.wikidata.org/entity/Q"))
                    }}
                LIMIT {limit}
            """
        else: # len(path) == 2
            query = f"""
                SELECT DISTINCT ?x WHERE {{
                    wd:{src} wdt:{path[0]} ?y.
                    ?y wdt:{path[1]} ?x.
                    FILTER(STRSTARTS(STR(?x), "http://www.wikidata.org/entity/Q"))
                    FILTER(STRSTARTS(STR(?y), "http://www.wikidata.org/entity/Q"))
                }}
                LIMIT {limit}
            """
        if self.prepend_prefixes:
            query = self.PREFIXES + query
        leaves = self.queryWikidata(query)
        # Keep only QIDs in the leaves
        leaves = [leaf['x']['value'].split('/')[-1] for leaf in leaves]
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
        assert len(path) < 2, f'Currenly only support paths with length less than 2, got {len(path)}'
        if len(path) == 0:
            return srcs
        # len(path) == 1
        query = f"""
            SELECT DISTINCT ?x WHERE {{
                VALUES ?src {{wd:{' wd:'.join(srcs)}}}
                ?src wdt:{path[0]} ?x.
                FILTER(STRSTARTS(STR(?x), "http://www.wikidata.org/entity/Q"))
                    }}
            LIMIT {limit}
            """
        if self.prepend_prefixes:
            query = self.PREFIXES + query
        leaves = self.queryWikidata(query)
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
        assert hop < 3, f'Currenly only support relations with hop less than 3, got {hop}'
        if hop == 1:
            query = f"""SELECT ?rel WHERE {{
                wd:{src} ?rel ?obj .
                FILTER(REGEX(STR(?rel), "^http://www.wikidata.org/prop/direct/"))
                FILTER(STRSTARTS(STR(?obj), "http://www.wikidata.org/entity/Q"))
                }}
                LIMIT {limit}
                """
        else: # hop == 2
            query = f"""SELECT ?rel1, ?rel2 WHERE {{
                wd:{src} ?rel1 ?obj1 .
                ?obj1 ?rel2 ?obj2 .
                FILTER(REGEX(STR(?rel1), "^http://www.wikidata.org/prop/direct/"))
                FILTER(REGEX(STR(?rel2), "^http://www.wikidata.org/prop/direct/"))
                FILTER(STRSTARTS(STR(?obj1), "http://www.wikidata.org/entity/Q"))
                FILTER(STRSTARTS(STR(?obj2), "http://www.wikidata.org/entity/Q"))
                }}
                LIMIT {limit}
                """
        if self.prepend_prefixes:
            query = self.PREFIXES + query

        relations = self.queryWikidata(query)
        if hop == 1:
            relations = [self.get_pid_from_uri(relation['rel']['value'])
                         for relation in relations]
        else:
            relations = [(self.get_pid_from_uri(relation['rel1']['value']),
                          self.get_pid_from_uri(relation['rel2']['value']))
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
            SELECT ?label
                WHERE {{
                    BIND(wd:{identifier} AS ?identifier)
                    ?identifier rdfs:label ?label .
                    FILTER(LANG(?label) = "en")
                    }}
                LIMIT 1
            """
        if self.prepend_prefixes:
            query = self.PREFIXES + query
        label = self.queryWikidata(query)
        if len(label) == 0:
            print(f'No label for identifier {identifier}.')
            return None
        label = label[0]['label']['value']
        return label

    def get_relation_label(self, relation):
        """Get label of a relation. If no label is found, return None.

        Args:
            relation (str): relation, a PID

        Returns:
            str | None: label of the relation
        """
        return self.get_label(relation)

    def get_entity_label(self, entity):
        """Get label of an entity. If no label is found, return None.

        Args:
            entity (str): entity, a QID

        Returns:
            str | None: label of the entity
        """
        return self.get_label(entity)
