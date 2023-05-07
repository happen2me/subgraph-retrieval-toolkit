"""This script retrieves subgraphs from a knowledge graph according to a natural
language query (usually a question). This command can also be used to evaluate
a trained retriever when the answer entities are known.

The expected fields of one sample are:
- question: question text
- question_entities: list of grounded question entities (ids)

For evaluation, the following field is also required:
- answer_entities: list of grounded answer entities (ids)
"""
import argparse
import heapq
import os
import pathlib
from collections import namedtuple
from typing import List, Any, Dict

import srsly
import torch
from tqdm import tqdm

from .knowledge_graph import KnowledgeGraphBase
from .scorer import Scorer
from .utils import get_knowledge_graph


END_REL = 'END OF HOP'


# Path collects the information at each traversal step
# - prev_relations stores the relations that have been traversed
# - score stores the score of the relation path
Path = namedtuple('Path', ['prev_relations', 'score'], defaults=[(), 0])


class KnowledgeGraphTraverser:
    '''KnowledgeGraphTraverser is a helper class that traverses a knowledge graph'''
    def __init__(self, kg: KnowledgeGraphBase):
        self.kg = kg

    def retrive_subgraph(self, entity, path):
        '''Retrive subgraph entities and triplets by traversing from an entity following
        a relation path hop by hop.

        Args:
            entity (str): the identifier of the source node
            path (list[str]): a list of relation identifiers

        Returns:
            entities: a list of entity identifiers
            triplets: a list of triplets
        '''
        entities, triplets = set(), set()
        tracked_entities = set((entity,))
        for relation in path:
            next_hops = set()
            if relation == END_REL:
                continue
            for e in tracked_entities:
                leaves = set(self.kg.deduce_leaves(e, (relation,)))
                next_hops |= leaves
                triplets |= {(e, relation, leaf) for leaf in leaves}
            entities |= next_hops
            tracked_entities = next_hops
        return list(entities), list(triplets)

    def deduce_leaves(self, entity, path):
        """Deduce leaves from an entity following a path hop by hop.

        Args:
            entity (str): the identifier of the source node
            path (list[str]): a list of relation identifiers

        Returns:
            set[str]: a set of leave identifiers that are n-hop away from the source node,
                where n is the length of the path
        """
        leaves = set((entity,))
        for relation in path:
            if relation == END_REL:
                continue
            leaves = set().union(*(self.kg.deduce_leaves(leaf, (relation,)) for leaf in leaves))
        return leaves

    def deduce_leaf_relations(self, entity, path):
        """Deduce leaf relations from an entity following a path hop by hop.

        Args:
            entity (str): the identifier of the source node
            path (list[str]): a list of relation identifiers

        Returns:
            tuple[str]: a tuple of relations that are n-hop away from the source node,
                where n is the length of the path
        """
        leaves = self.deduce_leaves(entity, path)
        relations = set().union(*(self.kg.get_neighbor_relations(leaf) for leaf in leaves))
        # Special filter relation for freebase
        if self.kg.name == 'freebase':
            relations = [r for r in relations if r.split('.')[0] not in ['kg', 'common']]
        return tuple(relations)

    def get_relation_label(self, identifier):
        """Get the relation label of an entity or a relation.

        It serves as a proxy to the knowledge graph's get_label function. For freebase,
        we directly use the identifier as the label. For others, we return the retrieved
        label if it exists, otherwise return the identifier.

        Args:
            identifier (str): the identifier of an entity or a relation

        Returns:
            str: the label of the entity or the relation
        """
        # For freebase, the relation identifier contains enough information
        if self.kg.name == 'freebase':
            return identifier
        if identifier == END_REL:
            return END_REL
        label =  self.kg.get_label(identifier)
        if label is None:
            return identifier
        return label


class Retriever:
    '''Retriever retrieves subgraphs from a knowledge graph with a question and its
    linked entities. The retrieval process takes the semantic information of the question
    and the expanding path into consideration.
    '''
    def __init__(self, kg: KnowledgeGraphBase, scorer: Scorer, beam_width: int, max_depth: int):
        self.kgh = KnowledgeGraphTraverser(kg)
        self.scorer = scorer
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.num_entity_threshold = 1000

    def retrieve_subgraph_triplets(self, sample: Dict[str, Any]):
        """Retrieve triplets as subgraphs from paths.

        Args:
            sample (dict): a sample from the dataset, which contains at least the following fields:
                question: a string
                question_entities: a list of entities

        Returns:
            list(tuple): a list of triplets
        """
        question = sample['question']
        triplets = []
        for question_entity in sample['question_entities']:
            path_score_list = self.beam_search_path(question, question_entity)
            n_nodes = 0
            for relations, _ in path_score_list:
                partial_nodes, partial_triples = self.kgh.retrive_subgraph(question_entity, relations)
                if len(partial_nodes) > self.num_entity_threshold:
                    continue
                n_nodes += len(partial_nodes)
                triplets.extend(partial_triples)

                if n_nodes > self.num_entity_threshold:
                    break
        triplets = list(set(triplets))
        return triplets

    def beam_search_path(self, question: str, question_entity: str):
        '''This function reimplement RUC's paper's solution. In the search process, only the history
        paths are recorded; each new relation is looked up via looking up the end relations from the
        question entities following a history path.

        Args:
            question (str): a natural language question
            question_entity (str): a grounded question entity

        Returns:
            list[Path]: path score list, a list of (path, score) tuples
        '''
        candidate_paths = [Path()]  # path and its score
        result_paths = []
        depth = 0

        while candidate_paths and len(result_paths) < self.beam_width and depth < self.max_depth:
            next_relations_batched = []
            for path in candidate_paths:
                prev_relations = path.prev_relations
                next_relations = self.kgh.deduce_leaf_relations(question_entity, prev_relations)
                next_relations = next_relations + (END_REL,)
                next_relations_batched.append(next_relations)

            tracked_paths = self.expand_and_score_paths(question, candidate_paths, next_relations_batched)
            tracked_paths = heapq.nlargest(self.beam_width, tracked_paths, key=lambda x: x.score)
            depth += 1
            # Update candidate_paths
            candidate_paths = []
            for path in tracked_paths:
                if  path.prev_relations and path.prev_relations[-1] == END_REL:
                    result_paths.append(path)
                else:
                    candidate_paths.append(path)
        # Merge not-yet-ended paths into the result paths 
        candidate_paths = [Path(prev_relations + (END_REL,), score) for prev_relations, score in candidate_paths]
        result_paths = heapq.nlargest(self.beam_width, result_paths + candidate_paths, key=lambda x: x.score)
        return result_paths

    def expand_and_score_paths(self, question: str, paths: List[Path], relations_batched: List[List[str]]) -> List[Path]:
        '''Expand the paths by one hop and score them by comparing the embedding similarity between
        the query (question + prev_relations) and the next relation.

        Args:
            question (str)
            paths (list[Path]): a list of current paths
            relations_batched (list[list[str]]): a list of next relations for each path

        Returns:
            list[Path]: scored_paths, a list of newly expanded and scored paths
        '''
        scored_paths = []
        score_matrix = []
        for path, next_relations in zip(paths, relations_batched):
            prev_relation_labels = tuple(self.kgh.get_relation_label(r) for r in path.prev_relations)
            next_relation_labels = tuple(self.kgh.get_relation_label(r) for r in next_relations)
            scores = self.scorer.batch_score(question, prev_relation_labels, next_relation_labels)
            score_matrix.append(scores)

        for i, (path, next_relations) in enumerate(zip(paths, relations_batched)):
            for j, relation in enumerate(next_relations):
                new_prev_relations = path.prev_relations + (relation,)
                score = float(score_matrix[i][j]) + path.score
                scored_paths.append(Path(new_prev_relations, score))
        return scored_paths


def calculate_hit_and_miss(retrieved_path):
    """Calculate the recall of answer entities in retrieved triplets,
    if answer_entities exists in each sample.
    
    Args:
        retrieved_path (str): path to the retrieved triplets
    
    Returns:
        tuple(int, int): number of samples that have at least one answer entity in retrieved triplets,
            and number of samples that have no answer entity in retrieved triplets
        
    """
    retrieval = srsly.read_jsonl(retrieved_path)
    hit = 0
    miss = 0
    for sample in retrieval:
        if 'answer_entities' not in sample:
            continue
        answers = sample['answer_entities']
        entities = set().union(*((triplet[0], triplet[-1]) for triplet in sample['triplets']))
        if any([entity in answers for entity in entities]):
            hit += 1
        else:
            miss += 1
    return hit, miss


def print_and_save_recall(retrieved_path):
    """Calculate and print the recall of answer entities in retrieved triplets,
    If any answer from the answer entities is in the retrieved entities, the sample
    counts as a hit.
    """
    hit, miss = calculate_hit_and_miss(retrieved_path)
    print(f"Answer coverage rate: {hit / (hit + miss)} ({hit} / {hit + miss})")
    info = {}
    if hit + miss != 0:
        info = {
            'hit': hit,
            'miss': miss,
            'recall': hit / (hit + miss)
        }
    # path/to/subgraph.jsonl -> path/to/subgraph.metric
    recall_path = os.path.splitext(retrieved_path)[0] + '.metric'
    srsly.write_json(recall_path, info)


def retrieve(args):
    """Retrieve subgraphs from a knowledge graph.

    Args:
        args (Namespace): arguments for subgraph retrieval
    """
    pathlib.Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    kg = get_knowledge_graph(args.knowledge_graph, args.sparql_endpoint,
                             exclude_qualifiers=not args.include_qualifiers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scorer = Scorer(args.scorer_model_path, device)
    retriever = Retriever(kg, scorer, args.beam_width, args.max_depth)
    samples = list(srsly.read_jsonl(args.input))
    total = sum(1 for _ in srsly.read_jsonl(args.input))
    for sample in tqdm(samples, desc='Retrieving subgraphs', total=total):
        triplets = retriever.retrieve_subgraph_triplets(sample)
        sample['triplets'] = triplets
    srsly.write_jsonl(args.output, samples)
    print(f'Retrieved subgraphs saved to to {args.output}')
    if args.evaluate:
        print_and_save_recall(args.output)


def _add_arguments(parser):
    """Add retrieve arguments to the parser in place."""
    parser.description = '''Retrieve subgraphs with a trained model on a dataset that entities are linked.
    This command can also be used to evaluate a trained retriever when the answer entities are known.

    Provide a JSON file as input, where each JSON object must contain at least the 'question' and 'question_entities' fields.
    When ``--evaluate`` is set, the input JSON file must also contain the 'answer_entities' field.

    The output JSONL file will include an added 'triplets' field, based on the input JSONL file. This field consists of a list of triplets,
    with each triplet representing a (head, relation, tail) tuple.
    When ``--evaluate`` is set, a metric file will also be saved to the same directory as the output JSONL file.
    '''
    parser.add_argument('-i', '--input', type=str, required=True, help='path to input jsonl file. it should contain at least \
                        ``question`` and ``question_entities`` fields.')
    parser.add_argument('-o', '--output', type=str, required=True, help='output file path for storing retrieved triplets.')
    parser.add_argument('-e', '--sparql-endpoint', type=str, help='SPARQL endpoint for Wikidata or Freebase services.')
    parser.add_argument('-kg', '--knowledge-graph', type=str, required=True, choices=('freebase', 'wikidata', 'dbpedia'),
                        help='choose the knowledge graph: currently supports ``freebase`` and ``wikidata``.')
    parser.add_argument('-m', '--scorer-model-path', type=str, required=True, help='Path to the scorer model, containing \
                        both the saved model and its tokenizer in the Huggingface models format.\
                        Such a model is saved automatically when using the ``srtk train`` command.\
                        Alternatively, provide a pre-trained model name from the Hugging Face model hub.\
                        In practice it supports any Huggingface transformers encoder model, though models that do not use [CLS] \
                        tokens may require modifications on similarity function.')
    parser.add_argument('--beam-width', type=int, default=10, help='beam width for beam search (default: 10).')
    parser.add_argument('--max-depth', type=int, default=2, help='maximum depth for beam search (default: 2).')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the retriever model. When the answer \
                        entities are known, the recall can be evluated as the number of samples that any of the \
                        answer entities are retrieved in the subgraph by the number of all samples. This equires \
                        `answer_entities` field in the input jsonl.')
    parser.add_argument('--include-qualifiers', action='store_true', help='Include qualifiers from the retrieved triplets. \
                        Qualifiers are informations represented in non-entity form, like date, count etc.\
                        This is only relevant for Wikidata.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()
    if not args.sparql_endpoint:
        if args.knowledge_graph == 'freebase':
            args.sparql_endpoint = 'http://localhost:3001/sparql'
        elif args.knowledge_graph == 'wikidata':
            args.sparql_endpoint = 'http://localhost:1234/api/endpoint/sparql'
        else:
            args.sparql_endpoint = 'https://dbpedia.org/sparql'
        print(f'Using default sparql endpoint for {args.knowledge_graph}: {args.sparql_endpoint}')
    retrieve(args)
