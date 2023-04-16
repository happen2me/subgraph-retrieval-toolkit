from collections import defaultdict

from tqdm import tqdm

from .graph_base import KnowledgeGraphBase


class CachedGraph(KnowledgeGraphBase):
    """CachedGraph serves as an in-memory cache for a knowledge graph.
    It stores the knowledge graph in a dictionary consisting of {head: {relation: {tail, ...}}}.
    The knowledge should be loaded from a text file where each line is a (head, relation, tail) triplet.
    """
    def __init__(self, triplet_path, name):
        self.name = name  # wiki or freebase
        # adjacency: {head: {relation: {tail, }}}; type: dict[dict[set]]
        self.adjacency = self._load_triplets(triplet_path)

    def _load_triplets(self, triplet_path):
        adjacency = defaultdict(lambda: defaultdict(set))
        with open(triplet_path, encoding='utf-8') as f:
            for line in tqdm(f, desc='Loading triplets', total=15838623):
                head, relation, tail = line.strip().split('\t')
                if self.name == 'freebase' and (relation.startswith('kg') or relation.startswith('type')) :
                    continue
                adjacency[head][relation].add(tail)
        return adjacency

    def search_one_hop_relations(self, src, dst):
        """Search one hop relations between src and dst.

        Args:
            src (str): source entity
            dst (str): destination entity

        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        return [[relation] for relation, tails in self.adjacency[src].items() if dst in tails]

    def search_two_hop_relations(self, src, dst):
        """Search two hop relations between src and dst.

        Args:
            src (str): source entity
            dst (str): destination entity

        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        return [[relation1, relation2]
                for relation1, tails1 in self.adjacency[src].items()
                for tail1 in tails1
                for relation2, tails2 in self.adjacency[tail1].items()
                if dst in tails2]

    def deduce_leaves(self, src, path, limit=2000):
        """Deduce leave entities from source entity following the path.

        Args:
            src_entity (str): source entity
            path (tuple[str]): path from source entity to destination entity
            limit (int, optional): limit of the number of leaves.

        Returns:
            list[str]: list of leaves. Each leaf is a QID.
        """
        leaves = [src]
        for relation in path:
            new_leaves = set()
            for entity in leaves:
                new_leaves.update(self.adjacency[entity][relation])
            leaves = new_leaves
        return list(leaves)[:limit]

    def get_neighbor_relations(self, src, hop=1, limit=2000):
        """Get relations between src and dst.

        Args:
            src (str): source entity
            hop (int, optional): hop of the relations. Defaults to 1.
            limit (int, optional): limit of the number of relations.

        Returns:
            list[str] | list[tuple[str]]: list of relations (one-hop)
                or list of tuples of relations (multi-hop)
        """
        neighbor_entities = [src]
        neighbor_relations = [()]
        for _ in range(hop):
            new_neighbor_relations = set()
            new_neighbor_entities = set()
            for relation in neighbor_relations:
                for entity in neighbor_entities:
                    new_neighbor_relations.update((relation + (neighbor_relation, )
                                                   for neighbor_relation in self.adjacency[entity].keys()))
                    new_neighbor_entities.update(set().union(*self.adjacency[entity].values()))
            neighbor_relations = list(new_neighbor_relations)[:limit]
            neighbor_entities = list(new_neighbor_entities)[:limit]
        return neighbor_relations

    def get_label(self, identifier):
        return identifier


if __name__ == '__main__':
    freebase_cache = 'data/freebase-cache.txt'
    cached_graph = CachedGraph(freebase_cache, name='freebase')
    print(cached_graph.get_neighbor_relations('m.010131yb'))
