"""Retrieve triplets from a question and its grounding.

Personal note: this is not an efficient and accurate implementation. It first ranks relation paths,
and the reconstruct the graph via the paths. The two steps are separated, making the retrieved
graph containing paths that was not retrieved. A better implementation would be to take the
entities into consideration when training the scorer, and taking each new relation-entity pair
as an expanding step.
"""
import argparse
import heapq
from collections import namedtuple
from functools import lru_cache

import srsly
import torch

from knowledge_graph.wikidata import Wikidata


END_REL = 'END_REL'


class Scorer:
    """Scorer for relation paths."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @lru_cache
    def score(self, question, prev_relations, next_relation):
        """Score a relation path."""
        query = f"{question} [SEP] {' # '.join(prev_relations)}"
        inputs = self.tokenizer(query, next_relation, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.item()


# Path collects the information at each traversal step
# - last_nodes stores the nodes that are connected to the last node by the last relation
# - prev_relations stores the relations that have been traversed
# - score stores the score of the relation path
# - history_triplets stores the triplets that have been traversed, each triplet is a tuple
#   of (node, relation, node)
Path = namedtuple('Path', ['last_nodes', 'prev_relations', 'score'],
                  defaults=[(), (), 0])


def beam_search_path(graph: Wikidata, scorer, question, question_entities, beam_width, max_depth):
    """Beam search for paths."""
    # Each path is a tuple of (last_nodes, prev_relations, score, history_nodes)
    # When a relation is added, the last_nodes are updated by the new nodes that are connected to one of the last node by the relation
    paths = [Path(tuple(question_entities), (), 0)]
    best_paths = []

    for _ in range(max_depth):
        new_paths = []
        for last_nodes, prev_relations, _ in paths:
            for last_node in last_nodes:
                neighbor_relations = graph.get_relations(last_node, limit=beam_width)
                neighbor_relations += [END_REL]
                for relation in neighbor_relations:
                    new_prev_relations = prev_relations + (relation,)
                    if relation == END_REL:
                        neighbor_nodes = ()
                    else:
                        neighbor_nodes = tuple(graph.deduce_leaves(last_node, [relation], limit=beam_width))
                    score = scorer.score(question, prev_relations, relation)
                    # new_prev_relations include the previous relation and the next relation, the scores
                    # of those with the same new_prev_relations should be the same.
                    new_path = Path(last_nodes=neighbor_nodes,
                                    prev_relations=new_prev_relations,
                                    score=score)
                    new_paths.append(new_path)

        best_paths = heapq.nlargest(beam_width, new_paths, key=lambda x: x.score)
        paths = best_paths
    return paths


def retrieve_entities_and_triplets_from_relation_path(src, relation_path):
    """Retrieve entities and triplets from a path.
    
    Args:
        src: the source node
        relation_path: a list of relations

    Returns:
        entities: a list of entities
        triplets: a list of triplets
    """
    pass


def retrieve_triplets_from_paths(src_entities, paths):
    """Retrieve triplets as subgraphs from paths.
    """
    return []

def main(args):
    groundings = srsly.read_json(args.input)
    wikidata = None
    scorer = None
    for ground in groundings:
        question = ground['question']
        question_entities = ground['question_entities']
        paths = beam_search_path(wikidata, scorer, question, question_entities, args.beam_width, args.max_depth)
        triplets = retrieve_triplets_from_paths(question_entities, paths)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to the grounded qustions')
    parser.add_argument('--beam-width', type=int, default=5, help='beam width for beam search')
    parser.add_argument('--max-depth', type=int, default=2, help='maximum depth of the search tree')
    args = parser.parse_args()
    main(args)
