from abc import abstractmethod


class LinkerBase:
    """Base class for entity linking"""

    @abstractmethod
    def annotate(self, text: str):
        """Annotate a text with the entities in the knowledge graph

        The returned dictionary should at least have the following fields:
        - question: The input text
        - question_entities: The entity ids of the entities in the text

        Args:
            text (str): The text to annotate

        Returns:
            dictionary: The annotated text with linked entities
        """
