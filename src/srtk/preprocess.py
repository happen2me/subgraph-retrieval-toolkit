"""This script creates the training data from the grounded questions.

Inputs should be a jsonl file, with each line representing a grounded question.
The format of each line should be like this example:

.. code-block:: json

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
"""

import os
import argparse
from argparse import Namespace
from pathlib import Path

from .preprocessing.search_path import main as search_path
from .preprocessing.score_path import main as score_path
from .preprocessing.negative_sampling import main as negative_sampling


def preprocess(args):
    output_path = args.output
    # Create parent dir for output if not exists.
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    if args.search_path:
        intermediate_dir = args.intermediate_dir
        if intermediate_dir is None:
            intermediate_dir = os.path.dirname(output_path)
        Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
        paths_file = os.path.join(intermediate_dir, 'paths.jsonl')
        scores_file = os.path.join(intermediate_dir, 'scores.jsonl')
        search_path_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                    knowledge_graph=args.knowledge_graph,
                                    ground_path=args.input,
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
                                           num_negative=args.num_negative,
                                           positive_threshold=args.positive_threshold,
                                           output_path=output_path,)
        search_path(search_path_args)
        score_path(score_path_args)
    else:
        negative_sampling_args = Namespace(sparql_endpoint=args.sparql_endpoint,
                                           knowledge_graph=args.knowledge_graph,
                                           scored_path_file=args.input,
                                           num_negative=args.num_negative,
                                           positive_threshold=args.positive_threshold,
                                           output_path=output_path,)
    negative_sampling(negative_sampling_args)


def _add_arguments(parser):
    """Add preprocess arguments to a parser in place."""
    parser.description = 'Create the training data from the grounded questions.'
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The grounded questions file with question, question & answer entities')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output path where the final training data will be saved.')
    parser.add_argument('--intermediate-dir', type=str, help="The directory to save intermediate files. If not specified, the intermediate \
                        files will be saved in the same directory as the output file, with the name paths.jsonl and scores.jsonl")
    parser.add_argument('-e', '--sparql-endpoint', type=str, required=True,
                        help="SPARQL endpoint URL for either Wikidata or Freebase\
                        (e.g., 'http://localhost:1234/api/endpoint/sparql' for default local qEndpoint)")
    parser.add_argument('-kg', '--knowledge-graph', type=str, required=True, choices=('wikidata', 'freebase', 'dbpedia'),
                        help='knowledge graph name, either wikidata or freebase')
    parser.add_argument('--search-path', action='store_true',
                        help='Whether to search paths between question and answer entities. If not specified, paths and scores fields\
                        must present in the input file. You **have to** specify this for weakly supervised learning. (default: False)')
    parser.add_argument('--metric', choices=('jaccard', 'recall'), default='jaccard',
                        help='The metric used to score the paths. recall will usually result in a lager size of training dataset.\
                        (default: jaccard))')
    parser.add_argument('--num-negative', type=int, default=15,
                        help='The number of negative relations to sample for each positive relation. (default: 15)')
    parser.add_argument('--positive-threshold', type=float, default=0.5,
                        help='The threshold to determine whether a path is positive or negative. If you want to use \
                        a larger training dataset, you can set this value to a smaller value. (default: 0.5)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()
    preprocess(args)
