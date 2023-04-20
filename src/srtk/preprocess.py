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

from .preprocessing.search_path import main as search_path
from .preprocessing.score_path import main as score_path
from .preprocessing.negative_sampling import main as negative_sampling


def main(args):
    paths_file = os.path.join(args.output_dir, 'paths.jsonl')
    scores_file = os.path.join(args.output_dir, 'scores.jsonl')
    output_file = os.path.join(args.output_dir, 'train.jsonl')
    search_path_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                 knowledge_graph=args.knowledge_graph,
                                 ground_path=args.input_file,
                                 output_path=paths_file,
                                 remove_sample_without_path=True)
    score_path_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                knowledge_graph=args.knowledge_graph,
                                paths_file=paths_file,
                                output_path=scores_file,
                                metric=args.metric,)
    negative_sampling_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                       knowledge_graph=args.knowledge_graph,
                                       scored_path_file=scores_file,
                                       positive_threshold=args.positive_threshold,
                                       output_file=output_file,)
    search_path(search_path_args)
    score_path(score_path_args)
    negative_sampling(negative_sampling_args)


def add_arguments(parser):
    """Add preprocess arguments to a parser in place."""
    parser.add_argument('--sparql-endpoint', type=str, required=True,
                        help="SPARQL endpoint URL for either Wikidata or Freebase (e.g., 'http://localhost:1234/api/endpoint/sparql' for default local qEndpoint)")
    parser.add_argument('-kg', '--knowledge-graph', type=str, required=True, choices=('wikidata', 'freebase'),
                        help='knowledge graph name, either wikidata or freebase')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='The grounded questions file with question, question & answer entities')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='The output directory where the training train and the \
                        data (paths, scores) will be saved.')
    parser.add_argument('--metric', choices=('jaccard', 'recall'), default='jaccard',
                        help='The metric used to score the paths. recall will usually result in a lager size of training dataset.')
    parser.add_argument('--positive-threshold', type=float, default=0.5,
                        help='The threshold to determine whether a path is positive or negative. The default value is 0.5.\
                        If you want to use a larger training dataset, you can set this value to a smaller value.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
