import json
import logging
import requests

from wikimapper import WikiMapper

from .linker_base import LinkerBase


class WikidataLinker(LinkerBase):
    """Link entitiy mentions to Wikidata entities using the REL endpoint"""
    def __init__(self, endpoint, wikimapper_db, service='rel'):
        """Initialize the linker

        Args:
            endpoint (str): The endpoint of the REL service
            wikimapper_db (str): The path to the Wikimapper database
        """
        self.endpoint = endpoint
        self.mapper = WikiMapper(wikimapper_db)
        self.service = service

    def annotate_rel(self, text, token=None):
        """Annotate using a local REL service.
        Check https://rel.readthedocs.io/en/latest/tutorials/e2e_entity_linking for setup instructions.

        Args:
            text (str): The text to annotate

        Returns:
            dict: annotation results
        """
        document = {
            'text': text,
        }
        headers = {
            # 'Content-Type': 'application/json'
        }
        if token is not None:
            headers['gcube-token'] = token
        api_results = requests.post(self.endpoint, json=document, timeout=60,
                                    headers=headers)
        if api_results.status_code != 200:
            logging.error(f"Error in REL service: {api_results.text}")
            decoded = []
        else:
            decoded = api_results.json()

        qids = []
        spans = []
        entities = []
        not_converted_entities = []
        for result in decoded:
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

    def annotate_tagme_wat(self, text, token):
        """Annotate using WAT or REL online services

        Args:
            text (str): The text to annotate
            token (str): The token to access the service

        Returns:
            dict: annotation results
        """
        params = {
            'gcube-token': token,
            'text': text
        }
        response = requests.get(self.endpoint, params=params, timeout=60)
        # Parse the JSON response
        data = json.loads(response.text)
        qids = []
        spans = []
        entity_names = []
        not_converted_entities = []
        if "annotations" in data:
            for annotation in data["annotations"]:
                title = annotation["title"]
                qid = self.mapper.title_to_id(title)
                if qid is None:
                    not_converted_entities.append(title)
                else:
                    qids.append(qid)
                    spans.append((annotation["start"], annotation["end"]))
                    entity_names.append(title)
        linked = {
            "question": text,
            "question_entities": qids,
            "spans": spans,
            "entity_names": entity_names,
            "not_converted_entities": not_converted_entities,
        }
        return linked

    def annotate(self, text, **kwargs):
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
        token = kwargs.get("token", None)
        if self.service in ['tagme', 'wat']:
            if token is None:
                raise ValueError(f"The {self.service} service requires a token")
            return self.annotate_tagme_wat(text, token)

        if self.service == 'rel':
            return self.annotate_rel(text, token)

        raise NotImplementedError(f"Service {self.service} is not implemented")


# if __name__ == '__main__':
#     linker = WikidataLinker('http://localhost:5000/rel', 'data/index_enwiki-latest-uncased.db')
#     text = "The city of [[Amsterdam]] is the capital of [[Netherlands]]."
#     print(linker.annotate(text))