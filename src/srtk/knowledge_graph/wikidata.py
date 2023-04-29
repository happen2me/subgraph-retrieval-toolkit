from functools import lru_cache
from SPARQLWrapper import SPARQLWrapper, JSON

from .graph_base import KnowledgeGraphBase


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
    ENTITY_PREFIX: str = "http://www.wikidata.org/entity/Q"

    def __init__(self, endpoint, prepend_prefixes=False, exclude_qualifiers=True):
        """Create a Wikidata query handler.

        Args:
            endpoint (str): SPARQL endpoint, e.g. https://query.wikidata.org/sparql
                Note that the protocal part (like https) is necessary.
            prepend_prefixes (bool, optional): whether to prepend prefixes to the query.
                Necessary for endpoints without pre-defined prefixes. Defaults to False.
            exclude_qualifiers (bool, optional): whether to filter out qualifiers in the
                queried entities. If set to True, only Wikidata entities (QXX) will be
                considered. Defaults to True.
        """
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.prepend_prefixes = prepend_prefixes
        self.exclude_qualifiers = exclude_qualifiers
        self.name = 'wikidata'

    def queryWikidata(self, query):
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

    @staticmethod
    def get_pid_from_uri(uri):
        """Get property id from uri."""
        return uri.split('/')[-1]

    @staticmethod
    def is_qid(qid):
        """Check if qid is a valid Wikidata entity id."""
        return qid.startswith('Q') and qid[1:].isdigit()

    @staticmethod
    def is_pid(pid):
        """Check if pid is a valid Wikidata property id."""
        return pid.startswith('P') and pid[1:].isdigit()

    def get_quantifier_filter(self, var_name):
        """Get quantifier filter string where the var is restricted to be entities.
        If exclude_qualifiers is set to False, return empty string.

        Note: in Wikidata, entities are prefixed with "http://www.wikidata.org/entity/Q",
            while qualifiers are non-entity (and mostly string) values.
        """
        return f'FILTER(STRSTARTS(STR(?{var_name}), "{self.ENTITY_PREFIX}"))' if self.exclude_qualifiers else ''

    def search_one_hop_relations(self, src, dst):
        """Search one hop relation between src and dst.
        
        Args:
            src (str): source entity
            dst (str): destination entity
        
        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        if not self.is_qid(src) or not self.is_qid(dst):
            return []

        query = f"""
            SELECT DISTINCT ?r WHERE {{
                wd:{src} ?r wd:{dst}.
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
        if not self.is_qid(src) or not self.is_qid(dst):
            return []

        query = f"""
            SELECT DISTINCT ?r1 ?r2 WHERE {{
                wd:{src} ?r1 ?x.
                ?x ?r2 wd:{dst}.
                {self.get_quantifier_filter('x')}
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
            path (tuple[str]): path from source entity to destination entity
            limit (int, optional): limit of the number of leaves. Defaults to 2000.
        
        Returns:
            list[str]: list of leaves. Each leaf is a QID.
        """
        if len(path) >= 3:
            raise NotImplementedError(f'Currenly only support paths with length less than 3, got {len(path)}')
        if not self.is_qid(src):
            return []

        if len(path) == 0:
            return [src]
        if len(path) == 1:
            query = f"""
                SELECT DISTINCT ?x WHERE {{
                    wd:{src} wdt:{path[0]} ?x.
                    {self.get_quantifier_filter('x')}
                    }}
                LIMIT {limit}
            """
        else: # len(path) == 2
            query = f"""
                SELECT DISTINCT ?x WHERE {{
                    wd:{src} wdt:{path[0]} ?y.
                    ?y wdt:{path[1]} ?x.
                    {self.get_quantifier_filter('y')}
                    {self.get_quantifier_filter('x')}
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
        if len(path) >= 2:
            raise NotImplementedError(f'Currenly only support paths with length less than 2, got {len(path)}')
        if len(path) == 0:
            return srcs
        srcs = [src for src in srcs if self.is_qid(src)]
        if len(srcs) == 0:
            return []

        query = f"""
            SELECT DISTINCT ?x WHERE {{
                VALUES ?src {{wd:{' wd:'.join(srcs)}}}
                ?src wdt:{path[0]} ?x.
                {self.get_quantifier_filter('x')}
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
        if hop >= 3:
            raise NotImplementedError(f'Currenly only support relations with hop less than 3, got {hop}')
        if not self.is_qid(src):
            return []

        if hop == 1:
            query = f"""SELECT DISTINCT ?rel WHERE {{
                wd:{src} ?rel ?obj .
                FILTER(REGEX(STR(?rel), "^http://www.wikidata.org/prop/direct/"))
                {self.get_quantifier_filter('obj')}
                }}
                LIMIT {limit}
                """
        else: # hop == 2
            query = f"""SELECT DISTINCT ?rel1 ?rel2 WHERE {{
                wd:{src} ?rel1 ?obj1 .
                ?obj1 ?rel2 ?obj2 .
                FILTER(REGEX(STR(?rel1), "^http://www.wikidata.org/prop/direct/"))
                FILTER(REGEX(STR(?rel2), "^http://www.wikidata.org/prop/direct/"))
                {self.get_quantifier_filter('obj1')}
                {self.get_quantifier_filter('obj2')}
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
        if not self.is_qid(identifier) and not self.is_pid(identifier):
            return identifier

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
