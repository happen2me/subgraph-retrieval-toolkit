from functools import lru_cache

import torch
from transformers import AutoTokenizer

from .encoder import LitSentenceEncoder


class Scorer:
    """Scorer for relation paths."""

    def __init__(self, pretrained_name_or_path, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)
        # Set pooling method to average if the model does not have a CLS token.
        if self.tokenizer.cls_token is None:
            pool_method = 'avg'
        else:
            pool_method = 'cls'
        self.model = LitSentenceEncoder(pretrained_name_or_path, pool=pool_method)
        if device:
            self.model = self.model.to(device)

    @lru_cache
    def score(self, question, prev_relations, next_relation):
        """Score a relation path.

        Args:
            question (str): question
            prev_relations (tuple[str]): tuple of relation **labels** that have been traversed.
            next_relation (str): next relation *label* to be traversed

        Returns:
            similarity: similarity between the query and the candidate
        """
        # Prepending 'query' and 'relation' corresponds to the way the model was trained (check collate_fn)
        query = f"query: {question} [SEP] {' # '.join(prev_relations)}"
        next_relation = 'relation: ' + next_relation
        text_pair = [query, next_relation]
        inputs = self.tokenizer(text_pair, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            query_embedding = outputs.last_hidden_state[0:1]
            sample_embedding = outputs.last_hidden_state[1:2]
            similarity = self.model.compute_sentence_similarity(
                query_embedding, sample_embedding)
        return similarity.item()

    @lru_cache
    def batch_score(self, question, prev_relations, next_relations):
        """Score next relations in batch.

        Args:
            question (str): question
            prev_relations (tuple[str]): tuple of relation **labels** that have been traversed.
            next_relations (tuple[str]): tuple of candidate next relation *labels* that are
                pertinent to the question and the previous relations.

        Returns:
            similarities (list[float]): list of similarities between the query and each candidate
        """
        query = f"query: {question} [SEP] {' # '.join(prev_relations)}"
        next_relations = ['relation: ' + next_relation for next_relation in next_relations]
        text_pair = [query] + next_relations
        inputs = self.tokenizer(text_pair, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            query_embedding = outputs.last_hidden_state[0:1]
            sample_embeddings = outputs.last_hidden_state[1:]
            similarities = self.model.compute_sentence_similarity(
                query_embedding, sample_embeddings)
            similarities = similarities.view(-1).tolist()
            if len(similarities) != len(next_relations):
                raise ValueError(f"Sanity check failed: {len(similarities)} != {len(next_relations)}")
        return similarities
