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

import srsly
from tqdm import tqdm

from knowledge_graph import Wikidata
from scorer import Scorer

END_REL = 'END_REL'


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
    # When a relation is added, the last_nodes are updated by the new nodes that are connected to
    # one of the last node by the relation
    paths = [Path(tuple(question_entities), (), 0)]
    best_paths = []

    for _ in range(max_depth):
        new_paths = []
        for last_nodes, prev_relations, _ in paths:
            for last_node in last_nodes:
                neighbor_relations = graph.get_relations(
                    last_node, limit=beam_width)
                neighbor_relations += [END_REL]
                for relation in neighbor_relations:
                    new_prev_relations = prev_relations + (relation,)
                    if relation == END_REL:
                        neighbor_nodes = ()
                    else:
                        neighbor_nodes = tuple(graph.deduce_leaves(
                            last_node, (relation,), limit=beam_width))
                    prev_relation_labels = tuple(graph.get_relation_label(relation) or relation
                                                 for relation in prev_relations)
                    if relation == END_REL:
                        next_relation_label = END_REL
                    else:
                        next_relation_label = graph.get_relation_label(relation) or relation
                    score = scorer.score(question, prev_relation_labels, next_relation_label)
                    # new_prev_relations include the previous relation and the next relation, the scores
                    # of those with the same new_prev_relations should be the same.
                    new_path = Path(last_nodes=neighbor_nodes,
                                    prev_relations=new_prev_relations,
                                    score=score)
                    new_paths.append(new_path)

        best_paths = heapq.nlargest(
            beam_width, new_paths, key=lambda x: x.score)
        paths = best_paths
    return paths


def retrieve_triplets_from_relation_path(src, relation_path, graph: Wikidata, beam_width=10):
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
            leaves = graph.deduce_leaves(entity, (relation,), limit=beam_width)
            leaves = [leaf for leaf in leaves if leaf not in visited_entities]
            visited_entities.update(leaves)
            triplets += [(entity, relation, leaf) for leaf in leaves]
            new_tracked_entities += leaves
        tracked_entities = new_tracked_entities
    return triplets


def retrieve_triplets_from_paths(src_entities, paths, graph: Wikidata):
    """Retrieve triplets as subgraphs from paths.
    """
    triplets = set()
    for src in src_entities:
        for path in paths:
            retrieved_triplets = retrieve_triplets_from_relation_path(
                src, path.prev_relations, graph)
            triplets.update(retrieved_triplets)
    return list(triplets)


def main(args):
    groundings = srsly.read_jsonl(args.input)
    total = sum(1 for _ in srsly.read_jsonl(args.input))
    wikidata = Wikidata(args.wikidata_endpoint)
    scorer = Scorer(args.scorer_model_path)
    outputs = []
    for ground in tqdm(groundings, total=total):
        question = ground['question']
        question_entities = ground['question_entities']
        paths = beam_search_path(
            wikidata, scorer, question, question_entities, args.beam_width, args.max_depth)
        triplets = retrieve_triplets_from_paths(
            question_entities, paths, wikidata)
        ground['triplets'] = triplets
        outputs.append(ground)
    srsly.write_jsonl(args.output_path, outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikidata-endpoint', type=str, default='http://localhost:1234/api/endpoint/sparql',
                        help='endpoint of the wikidata sparql service')
    parser.add_argument('--scorer-model-path', type=str,
                        default='models/roberta-base', help='path to the scorer model')
    parser.add_argument('--input', type=str, required=True,
                        help='path to the grounded qustions')
    parser.add_argument('--output-path', type=str,
                        required=True, help='path to the output file')
    parser.add_argument('--beam-width', type=int, default=5,
                        help='beam width for beam search')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='maximum depth of the search tree')
    args = parser.parse_args()
    main(args)
