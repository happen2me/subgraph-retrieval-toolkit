"""4. Score path

The score of a relation path is defined as the HIT rate of the prediction
with the ground truth entities. The *prediction* refers to the search results
from the question entities following the relation path.

Personal notes:
Why this is necessary? Isn't the relation path already the path from the question
entities to the ground truth entities?
In my understanding, this is similar to TF-IDF, the path is more precise if the results is
a smaller set of entities but have a higher intersection with the ground truth entities.

e.g.
python preprocess/score_path.py --paths-file data/preprocess/paths.jsonl --output-path data/preprocess/paths_scored.jsonl
"""
import os
import sys
import argparse

import srsly
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from knowledge_graph import KnowledgeGraphBase, get_knowledge_graph


def score_path(kg: KnowledgeGraphBase, src, path, answers, metric='jaccard'):
    """Calculate the HIT score of a given path.
    
    Args:
        kg (KnowledgeGraphBase): knowledge graph instance
        src (str): the source entity
        path (list): the path
        answers (list): the ground truth entities
        metric (str): how the paths are scored. 'jaccard' or 'recall'
            Default: 'jaccard', per the original implementation
    """
    if metric not in ('jaccard', 'recall'):
        raise ValueError(f'Unknown metric: {metric}')
    leaves = kg.deduce_leaves(src, path)
    leaves = set(leaves)
    answers = set(answers)
    hit = leaves.intersection(answers)
    if not leaves:
        # In the original implementation, they return 1 if the leaves is empty
        # I think it's more meaningful to set it to 0, as this path leads to no results
        return 0
    if metric == 'jaccard':
        score = len(hit) / len(leaves)
    else: # metric == 'recall':
        score = len(hit) / len(answers)
    return score


def main(args):
    kg = get_knowledge_graph(args.knowledge_graph_type, args.sparql_endpoint)
    samples = srsly.read_jsonl(args.paths_file)
    total_lines = sum(1 for _ in srsly.read_jsonl(args.paths_file))
    processed_samples = []  # adds path_scores to each sample
    # Each sample is a dict with the following fields:
    # - id: sample id
    # - question: question text
    # - question_entities: list of question entities
    # - answer_entities: list of answer entities
    # - paths: list of paths
    for sample in tqdm(samples, total=total_lines, desc='Scoring paths'):
        question_entities = sample['question_entities']
        answer_entities = sample['answer_entities']
        paths = sample['paths']
        path_scores = []
        for path in paths:
            # path score is the max score of all possible source entities following the path
            # Personal note: this is weird, why don't you start from the question entity where the
            # path was originally found?
            path = tuple(path)  # this makes it hashable
            score = max(score_path(kg, src, path, answer_entities, metric=args.metric) for src in question_entities)
            path_scores.append(score)
        sample['path_scores'] = path_scores
        processed_samples.append(sample)
    srsly.write_jsonl(args.output_path, processed_samples)
    print(f'Scored paths saved to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--sparql-endpoint', default='http://localhost:1234/api/endpoint/sparql', help='knowledge graph endpoint')
    parser.add_argument('-kg', '--knowledge-graph', default='wikidata', choices=('wikidata', 'freebase', 'dbpedia'),
                        help='knowledge graph name')
    parser.add_argument('--paths-file', help='the file where the paths are stored')
    parser.add_argument('--output-path', help='the file where the scores are stored')
    parser.add_argument('--metric', default='jaccard', choices=('jaccard', 'recall'), help='the metric used to score the paths')
    args = parser.parse_args()

    main(args)
