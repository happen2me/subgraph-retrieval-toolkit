"""5. Negative Sampling

Regarding negative sampling method, the author states in the paper:
> We replace the observed relation at each time step with other sampled relations as the
negative instances to optimize the probability of the observed ones.

e.g.
python preprocess/negative_sampling.py \
    --scored-path-file data/preprocess/paths_scored.jsonl \
    --output-file data/preprocess/train_.jsonl\
    --positive-threshold 0.3
"""
import os
import sys
import argparse
import random
from collections import defaultdict
from functools import lru_cache

import srsly
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from knowledge_graph import KnowledgeGraphBase, get_knowledge_graph


END_REL = "END OF HOP"


def sample_negative_relations(soruce_entities, prev_path, positive_connections,
                              num_negative, kg: KnowledgeGraphBase):
    """A helper function to sample negative relations.
    
    Args:
        soruce_entities (list[str]): list of source entities
        prev_path (list[str]): previous path / relations
        positive_connections (dict): a dictionary of positive connections
        num_negative (int): number of negative relations to sample
        kg (KnowledgeGraphBase): a knoledge graph instance
        
    Returns:
        list[str]: list of negative relations
    """
    negative_relations = set()
    for src in soruce_entities:
        # get all relations connected to current tracked entities (question or intermediate entities)
        negative_relations |= set(kg.get_neighbor_relations(src, limit=50))
        if len(negative_relations) > 100:  # yet another magic number :(
            break
    negative_relations = negative_relations - positive_connections[tuple(prev_path)]
    if len(negative_relations) == 0:
        return []
    negative_relations = random.choices(list(negative_relations), k=num_negative)
    return negative_relations


def is_candidate_space_too_large(path, question_entities, kg: KnowledgeGraphBase, candidate_depth_multiplier=5):
    """Check whether the number of the candidate entities along the path is too large.
    
    Args:
        path (list[str]): path from source entity to destination entity
        question_entities (list[str]): list of question entities
        kg (KnowledgeGraphBase): a knowledge graph instance
        candidate_depth_multiplier (int, optional): a multiplier to control the number of candidate entities
            at each depth. Defaults to 10.
    """
    flag_too_large = False
    for i in range(1, len(path)):
        prev_path = tuple(path[:i])
        # The further the path is from the question, the greater the search space becomes.
        limit = candidate_depth_multiplier ** i
        candidate_entities = set()
        for src in question_entities:
            candidate_entities |= set(kg.deduce_leaves(src, prev_path, limit=limit))
            # Stop early if the number of candidate entities is already very large
            if len(candidate_entities) > limit:
                break
        # Check the entity space at each depth, if it is too large at any depth, we flag the path as too large.
        if len(candidate_entities) > limit:
            flag_too_large = True
            break
    return flag_too_large


def sample_records_from_path(path, question, question_entities, positive_connections,
                             kg: KnowledgeGraphBase, num_negative):
    """Sample training records from a path.
    
    Returns:
        list[dict]: list of training records, each record has the following fields:
            - question (str): the question
            - prev_path (list): previous relations up to a relation (positive_relation) in the path
            - positive_relation (str): the next relation of the prev_path is regarded as the positive relation
            - negative_relations (list): a list of negative relations, the number is specified by num_negative
    """
    # My interpretation: If the number of candidate entities is too large, we simply discard this path.
    # But isn't it weird? The author checks the number of connected entities to the question entities
    # with each relation along the path, and simply discard those paths with too many connected entities.
    if is_candidate_space_too_large(path, question_entities, kg, candidate_depth_multiplier=5):
        return []

    path = path + [END_REL]

    records = []
    tracked_entities = question_entities
    for i, current_relation in enumerate(path):
        prev_path = path[:i]
        negative_relations = sample_negative_relations(tracked_entities, prev_path, positive_connections,
                                                       num_negative, kg)
        if len(negative_relations) == 0:
            continue
        record = {
            'question': question,
            'prev_path': prev_path,
            'positive_relation': current_relation,
            'negative_relations': negative_relations
        }
        records.append(record)
        if current_relation != END_REL:
            # update tracked entities
            tracked_entities = kg.deduce_leaves_from_multiple_srcs(tracked_entities, [current_relation], limit=100)
    return records


def get_positive_connections_along_paths(paths):
    """Collect positive connections along paths. A positive connection is defined as
    {prev_relations: next_relation}. END_REL is added to the end of each path.
    
    Returns:
        dict: a dictionary of positive connections
    """
    positive_connections = defaultdict(set)
    for path in paths:
        path = path + [END_REL]
        for i, rel in enumerate(path):
            positive_connections[tuple(path[:i])].add(rel)
    return positive_connections


def convert_records_relation_id_to_lable(records, kg):
    """Convert relation ids to relation labels in each record.
    """
    processed_records = []

    @lru_cache
    def get_label(rel):
        if kg.name == 'dbpedia' or kg.name == 'freebase':
            return rel
        if rel == END_REL:
            return END_REL
        return kg.get_relation_label(rel) or rel

    for record in tqdm(records, desc='Converting relation ids to labels'):
        record['prev_path'] = [get_label(rel) for rel in record['prev_path']]
        record['positive_relation'] = get_label(record['positive_relation'])
        record['negative_relations'] = [get_label(rel) for rel in record['negative_relations']]
        processed_records.append(record)
    return processed_records


def create_jsonl_dataset(records):
    """It combines the question and prev_path to query. Each train sample is a dict with the following fields:
    - query (str): question + prev_path
    - positive (str): the next relation of the prev_path is regarded as the positive relation
    - negatives (list): a list of negative relations
    
    Args:
        records (list[dict]): list of records
        
    Returns:
        list[dict]: list of train samples
    """
    samples = []
    for record in records:
        sample = {
            'query': record['question'] + ' [SEP] ' + ' # '.join(record['prev_path']),
            'positive': record['positive_relation'],
            'negatives': record['negative_relations']
        }
        samples.append(sample)
    return samples


def main(args):
    kg = get_knowledge_graph(args.knowledge_graph, args.sparql_endpoint)
    positive_threshold = args.positive_threshold
    # Each sample has the following fields:
    # - id: sample id
    # - question: question text
    # - question_entities: list of question entities (ids)
    # - answer_entities: list of answer entities (ids)
    # - question: question text
    # - paths: list of paths
    # - path_scores: list of path scores
    samples = srsly.read_jsonl(args.scored_path_file)
    total = sum(1 for _ in srsly.read_jsonl(args.scored_path_file))
    train_records = []
    for sample in tqdm(samples, total=total, desc='Negative sampling'):
        paths = sample['paths']
        path_scores = sample['path_scores']
        question = sample['question']
        question_entities = sample['question_entities']

        # Filter out paths with low scores.
        paths = [path for path, score in zip(paths, path_scores) if score >= positive_threshold]
        path_scores = [score for score in path_scores if score >= positive_threshold]
        if len(paths) != len(path_scores):
            raise ValueError(f'The number of paths and path scores are not equal. {len(paths)} != {len(path_scores)}')

        # A dictionary of {prev_rels: {next_rel, next_rel, ...}, ...},
        # where prev_rels is a tuple of previous relations.
        positive_connections = get_positive_connections_along_paths(paths)

        for path in paths:
            train_records.extend(sample_records_from_path(path, question, question_entities,
                                                          positive_connections, kg, args.num_negative))
    print(f"Number of training records: {len(train_records)}")
    train_records = convert_records_relation_id_to_lable(train_records, kg)
    train_records = create_jsonl_dataset(train_records)
    srsly.write_jsonl(args.output_path, train_records)
    print(f"Training samples are saved to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--sparql-endpoint', default='http://localhost:1234/api/endpoint/sparql',
                        help='knowledge graph endpoint (default: http://localhost:1234/api/endpoint/sparql)')
    parser.add_argument('-kg', '--knowledge-graph', default='wikidata', choices=['wikidata', 'freebase', 'dbpedia'],
                        help='knowledge graph name')
    parser.add_argument('--scored-path-file', help='The file containing scored paths')
    parser.add_argument('--output-path', help='The path to the output file')
    parser.add_argument('--positive-threshold', type=float, default=0.5, help='The threshold to determine whether a path is positive or negative')
    parser.add_argument('--num-negative', type=int, default=15, help='The number of negative relations to sample for each positive relation')
    args = parser.parse_args()

    main(args)
