"""Retrieve triplets from a question and its grounding.

Personal note: this is not an efficient and accurate implementation. It first ranks relation paths,
and the reconstruct the graph via the paths. The two steps are separated, making the retrieved
graph containing paths that was not retrieved. A better implementation would be to take the
entities into consideration when training the scorer, and taking each new relation-entity pair
as an expanding step.

e.g.
python retrieve.py --scorer-model-path intfloat/e5-small --input data/ground.jsonl \
    --output-path data/subgraph.jsonl --beam-width 2
"""
import argparse
import heapq
from collections import namedtuple
from functools import cache

import srsly
import torch
from tqdm import tqdm

from knowledge_graph import KnowledgeGraphBase, Freebase, Wikidata
from scorer import Scorer

END_REL = 'END OF HOP'


# Path collects the information at each traversal step
# - last_nodes stores the nodes that are connected to the last node by the last relation
# - prev_relations stores the relations that have been traversed
# - score stores the score of the relation path
# - history_triplets stores the triplets that have been traversed, each triplet is a tuple
#   of (node, relation, node)
Path = namedtuple('Path', ['last_nodes', 'prev_relations', 'score'],
                  defaults=[(), (), 0])


def beam_search_path(kg: KnowledgeGraphBase, scorer, question, question_entities, beam_width, max_depth):
    """Beam search for paths."""
    # Each path is a tuple of (last_nodes, prev_relations, score, history_nodes)
    # When a relation is added, the last_nodes are updated by the new nodes that are connected to
    # one of the last node by the relation
    paths = [Path(tuple(question_entities), (), 0)]
    best_paths = []

    for _ in range(max_depth):
        new_paths = []
        for last_nodes, prev_relations, prev_score in paths:
            if prev_relations and prev_relations[-1] == END_REL:
                continue
            prev_relation_labels = tuple(kg.get_relation_label(relation) or relation
                                         for relation in prev_relations)
            for last_node in last_nodes:
                neighbor_relations = kg.get_neighbor_relations(last_node, limit=beam_width * 2)
                neighbor_relations += [END_REL]
                neighbor_relation_labels = tuple(kg.get_relation_label(relation) or relation
                                            if relation != END_REL else END_REL
                                            for relation in neighbor_relations)
                scores = scorer.batch_score(question, prev_relation_labels, neighbor_relation_labels)
                for relation, score in zip(neighbor_relations, scores):
                    if relation == END_REL:
                        neighbor_nodes = ()
                    else:
                        neighbor_nodes = tuple(kg.deduce_leaves(
                            last_node, (relation,), limit=beam_width))
                    # new_prev_relations include the previous relation and the next relation, the scores
                    # of those with the same new_prev_relations should be the same.
                    new_prev_relations = prev_relations + (relation,)
                    new_path = Path(last_nodes=neighbor_nodes,
                                    prev_relations=new_prev_relations,
                                    score=score + prev_score)
                    new_paths.append(new_path)

        best_paths = heapq.nlargest(beam_width, new_paths + paths, key=lambda x: x.score)
        paths = best_paths
    return paths


def exhaustive_search_path(kg: KnowledgeGraphBase, scorer: Scorer, question, question_entities, beam_width, max_depth):
    """This function reimplement RUC's paper's solution. In the search process, only the history
    paths are recorded; each new relation is looked up via looking up the end relations from the
    question entities following a history path.
    """
    candidate_paths = [Path(prev_relations=(), score=0)]
    result_paths = []
    depth = 0

    @cache
    def expand_relations(src, prev_relations):
        leaves = kg.deduce_leaves(src, prev_relations, limit=beam_width * 2)
        relations = set()
        for leaf in leaves:
            relations.update(kg.get_neighbor_relations(leaf, limit=beam_width * 2))
        return relations

    while candidate_paths and len(result_paths) < beam_width and depth < max_depth:
        tracked_paths = []
        for question_entity in question_entities:
            for _, prev_relations, prev_score in candidate_paths:
                candidate_relations = expand_relations(question_entity, prev_relations)
                prev_relation_labels = tuple(kg.get_relation_label(relation) or relation
                                             for relation in prev_relations)
                candidate_relation_labels = tuple(kg.get_relation_label(relation) or relation
                                                  for relation in candidate_relations)
                scores = scorer.batch_score(question, prev_relation_labels,
                                            candidate_relation_labels)
                tracked_paths += [Path(last_nodes=(question_entity,),
                                        prev_relations=prev_relations + (relation,),
                                        score=score + prev_score) \
                                  for relation, score in zip(candidate_relations, scores)]
        tracked_paths = heapq.nlargest(beam_width, tracked_paths, key=lambda x: x.score)
        depth += 1
        candidate_paths = []
        for path in tracked_paths:
            if path.prev_relations and path.prev_relations[-1] == END_REL:
                result_paths.append(path)
            else:
                candidate_paths.append(path)
    # Rest of the candidate paths are added to the result paths
    result_paths = heapq.nlargest(beam_width, result_paths + candidate_paths, key=lambda x: x.score)
    return result_paths


def retrieve_triplets_from_relation_path(src, relation_path, kg: KnowledgeGraphBase, beam_width=10):
    """Retrieve entities and triplets from a path.
    Since each relation can have multiple entities, the retrieved triplets actually
    form a tree-like graph.

    Args:
        src: the source node
        relation_path: a list of relations

    Returns:
        entities: a list of entities
        triplets: a list of triplets
    """
    tracked_entities = [src]
    visited_entities = set([src])
    triplets = []
    # At each step, each tracked entity and the relation at this step result in entities
    # that are tracked in the next step.
    for relation in relation_path:
        new_tracked_entities = []
        if relation == END_REL:
            continue
        for entity in tracked_entities:
            leaves = kg.deduce_leaves(entity, (relation,), limit=beam_width)
            leaves = [leaf for leaf in leaves if leaf not in visited_entities]
            visited_entities.update(leaves)
            triplets += [(entity, relation, leaf) for leaf in leaves]
            new_tracked_entities += leaves
        tracked_entities = new_tracked_entities
    return triplets


def retrieve_triplets_from_paths(src_entities, paths, kg: KnowledgeGraphBase):
    """Retrieve triplets as subgraphs from paths.
    """
    triplets = set()
    for src in src_entities:
        for path in paths:
            retrieved_triplets = retrieve_triplets_from_relation_path(
                src, path.prev_relations, kg)
            triplets.update(retrieved_triplets)
    return list(triplets)


def main(args):
    groundings = srsly.read_jsonl(args.input)
    total = sum(1 for _ in srsly.read_jsonl(args.input))
    if args.knowledge_graph == 'freebase':
        knowledge_graph = Freebase(args.sparql_endpoint)
    else:
        knowledge_graph = Wikidata(args.sparql_endpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scorer = Scorer(args.scorer_model_path, device)
    outputs = []
    for ground in tqdm(groundings, total=total, desc='Retrieving subgraphs'):
        question = ground['question']
        question_entities = ground['question_entities']
        paths = exhaustive_search_path(
            knowledge_graph, scorer, question, question_entities, args.beam_width, args.max_depth)
        triplets = retrieve_triplets_from_paths(question_entities, paths, knowledge_graph)
        ground['triplets'] = triplets
        outputs.append(ground)
    srsly.write_jsonl(args.output_path, outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparql-endpoint', type=str, default='http://localhost:1234/api/endpoint/sparql',
                        help='endpoint of the wikidata or freebase sparql service')
    parser.add_argument('-kg', '--knowledge-graph', type=str, choices=('wikidata', 'freebase'), default='wikidata')
    parser.add_argument('--scorer-model-path', type=str, default='models/roberta-base', help='path to the scorer model')
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the grounded qustions')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='path to the output file')
    parser.add_argument('--beam-width', type=int, default=5, help='beam width for beam search')
    parser.add_argument('--max-depth', type=int, default=2, help='maximum depth of the search tree')
    args = parser.parse_args()
    main(args)
