'''Provide protocal for different kinds of knowledge graphs.'''

from abc import abstractmethod
from typing import List


class KnowledgeGraphBase:
    """Knowledge graph base class."""

    @abstractmethod
    def search_one_hop_relations(self, src: str, dst: str) -> List[List[str]]:
        """Search one hop relations between src and dst.

        Args:
            src (str): source entity
            dst (str): destination entity

        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        raise NotImplementedError

    @abstractmethod
    def search_two_hop_relations(self, src: str, dst: str) -> List[List[str]]:
        """Search two hop relations between src and dst.

        Args:
            src (str): source entity
            dst (str): destination entity

        Returns:
            list[list[str]]: list of paths, each path is a list of PIDs
        """
        raise NotImplementedError

    @abstractmethod
    def deduce_leaves(self, src: str, path: List[str], limit: int) -> List[str]:
        """Deduce leave entities from source entity following the path.

        Args:
            src_entity (str): source entity
            path (tuple[str]): path from source entity to destination entity
            limit (int, optional): limit of the number of leaves.

        Returns:
            list[str]: list of leaves. Each leaf is a QID.
        """
        raise NotImplementedError

    @abstractmethod
    def get_neighbor_relations(self, src: str, hop: int, limit: int) -> List[str]:
        """Get n-hop neighbor relations of src.

        Args:
            src (str): source entity
            hop (int, optional): hop of the relations. Defaults to 1.
            limit (int, optional): limit of the number of relations.

        Returns:
            list[str] | list[tuple[str]]: list of relations (one-hop)
                or list of tuples of relations (multi-hop)
        """
        raise NotImplementedError

    @abstractmethod
    def get_label(self, identifier: str) -> str:
        """Get label of an entity or a relation.

        Args:
            identifier (str): entity or relation identifier

        Returns:
            str: label of the entity or the relation
        """
        raise NotImplementedError

    def get_relation_label(self, relation: str) -> str:
        """Get the label of a relation. Defaults to get_label.

        Args:
            relation (str): relation identifier

        Returns:
            str: label of the relation
        """
        return self.get_label(relation)

    def get_entity_label(self, entity: str) -> str:
        """Get the label of an entity. Defaults to get_label.

        Args:
            entity (str): entity identifier

        Returns:
            str: label of the entity
        """
        return self.get_label(entity)
