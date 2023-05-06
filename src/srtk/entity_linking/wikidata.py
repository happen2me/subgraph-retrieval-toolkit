import requests

from wikimapper import WikiMapper

from .linker_base import LinkerBase


class WikidataLinker(LinkerBase):
    """Link entitiy mentions to Wikidata entities using the REL endpoint"""
    def __init__(self, endpoint, wikimapper_db):
        """Initialize the linker

        Args:
            endpoint (str): The endpoint of the REL service
            wikimapper_db (str): The path to the Wikimapper database
        """
        self.endpoint = endpoint
        self.mapper = WikiMapper(wikimapper_db)

    def annotate(self, text):
        """Annotate a text with the entities in the Wikidata knowledge graph

        Args:
            text (str): The text to annotate

        Returns:
            dict: A dictionary with the following keys:
                question: The input text
                question_entities: The Wikidata ids of the entities in the text
                spans: The spans of the entities in the text
                entity_names: The names of the entities in the text
                not_converted_entities: The entities that are not converted to Wikidata ids
        """
        document = {
            'text': text,
        }
        api_results = requests.post(self.endpoint, json=document, timeout=60).json()
        qids = []
        spans = []
        entities = []
        not_converted_entities = []
        for result in api_results:
            start_pos, mention_length, mention, entity, disambiguation_cofidence, mention_detection_confidence, tag = result
            qid = self.mapper.title_to_id(entity)
            span = (start_pos, start_pos + mention_length)
            if qid is None:
                not_converted_entities.append(entity)
            else:
                qids.append(qid)
                entities.append(entity)
                spans.append(span)
        linked = {
            "question": text,
            "question_entities": qids,
            "spans": spans,
            "entity_names": entities,
            "not_converted_entities": not_converted_entities,
        }
        return linked
