"""This script creates the training data from the grounded questions.

Inputs should be a jsonl file, with each line representing a grounded question.
The format of each line should be like this example:
```json
{
  "id": "sample-id",
  "question": "Which universities did Barack Obama graduate from?",
  "question_entities": [
    "Q76"
  ],
  "answer_entities": [
    "Q49122",
    "Q1346110",
    "Q4569677"
  ]
}
```
"""
import os
import argparse
from argparse import Namespace

from preprocess.search_path import main as search_path
from preprocess.score_path import main as score_path
from preprocess.negative_sampling import main as negative_sampling


def main(args):
    paths_file = os.path.join(args.output_dir, 'paths.jsonl')
    scores_file = os.path.join(args.output_dir, 'scores.jsonl')
    output_file = os.path.join(args.output_dir, 'train.jsonl')
    search_path_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                 knowledge_graph=args.knowledge_graph,
                                 ground_path=args.ground_path,
                                 output_path=paths_file,
                                 remove_sample_without_path=True)
    score_path_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                knowledge_graph=args.knowledge_graph,
                                paths_file=paths_file,
                                output_path=scores_file)
    negative_sampling_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                       knowledge_graph=args.knowledge_graph,
                                       scored_path_file=scores_file,
                                       positive_threshold=args.positive_threshold,
                                       output_file=output_file,)
    search_path(search_path_args)
    score_path(score_path_args)
    negative_sampling(negative_sampling_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparql-endpoint', type=str, required=True)
    parser.add_argument('-kg', '--knowledge-graph', type=str, required=True, choices=('wikidata', 'freebase'))
    parser.add_argument('--ground-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--positive-threshold', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
