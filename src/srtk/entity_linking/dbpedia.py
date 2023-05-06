"""
Link entitiy mentions to DBpedia entities with the DBpedia Spotlight endpoint.

Know more about DBpedia Spotlight at https://www.dbpedia-spotlight.org/.
"""
import requests

from .linker_base import LinkerBase

class DBpediaLinker(LinkerBase):
    """Link entitiy mentions to DBpedia entities with the DBpedia Spotlight endpoint"""
    def __init__(self, endpoint):
        """Initialize the linker
        
        Args:
            endpoint (str): The endpoint of the DBpedia Spotlight service
                e.g. https://api.dbpedia-spotlight.org/en/annotate
        """
        self.endpoint = endpoint

    def annotate(self, text):
        """Annotate a text with the entities in the DBpedia knowledge graph

        Args:
            text (str): The text to annotate

        Returns:
            dict: A dictionary with the following keys:
                question: The input text
                question_entities: The DBpedia entities in the text
                spans: The spans of the entities in the text
                similarity_scores: The similarity scores of the entities in the text
        """
        params = {'text': text}
        headers = {'Accept': 'application/json'}
        response = requests.get(self.endpoint, params=params, headers=headers,
                                timeout=60).json()
        resources = response['Resources']
        question_entities = []
        spans = []
        similarity_scores = []
        for resource in resources:
            uri = resource['@URI']
            if not uri.startswith('http://dbpedia.org/resource/'):
                continue
            entity = uri[len('http://dbpedia.org/resource/'):]
            offset = int(resource['@offset'])
            surface_form = resource['@surfaceForm']
            similarity_score = float(resource['@similarityScore'])
            span = (offset, offset + len(surface_form))
            question_entities.append(entity)
            spans.append(span)
            similarity_scores.append(similarity_score)
        linked = {
            "question": text,
            "question_entities": question_entities,
            "spans": spans,
            "similarity_scores": similarity_scores,
        }
        return linked
