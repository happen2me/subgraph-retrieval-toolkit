"""5. Negative Sampling

Regarding negative sampling method, the author states in the paper:
> We replace the observed relation at each time step with other sampled relations as the
negative instances to optimize the probability of the observed ones.

e.g.
python preprocess/negative_sampling.py \
    --scored-path-file data/retrieval/paths_scored.jsonl \
    --output-file data/retrieval/train.csv
"""
import os
import sys
import argparse
import random
from collections import defaultdict

import srsly
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from knowledge_graph.wikidata import Wikidata

END_REL = "END_REL"


def sample_negative_relations(soruce_entities, prev_path, positive_connections,
                              num_negative, wikidata):
    """A helper function to sample negative relations.
    
    Args:
        soruce_entities (list[str]): list of source entities
        prev_path (list[str]): previous path / relations
        positive_connections (dict): a dictionary of positive connections
        num_negative (int): number of negative relations to sample
        wikidata (Wikidata): a Wikidata instance
        
    Returns:
        list[str]: list of negative relations
    """
    negative_relations = set()
    for src in soruce_entities:
        # get all relations connected to current tracked entities (question or intermediate entities)
        negative_relations |= set(wikidata.get_relations(src))
        if len(negative_relations) > 100:  # yet another magic number :(
            break
    negative_relations = negative_relations - positive_connections[tuple(prev_path)]
    if len(negative_relations) == 0:
        return []
    negative_relations = random.choices(list(negative_relations), k=num_negative)
    return negative_relations


def is_candidate_space_too_large(path, question_entities, wikidata, candidate_depth_multiplier=10):
    """Check whether the number of the candidate entities along the path is too large.
    
    Args:
        path (list[str]): path from source entity to destination entity
        question_entities (list[str]): list of question entities
        wikidata (Wikidata): a Wikidata instance
        candidate_depth_multiplier (int, optional): a multiplier to control the number of candidate entities
            at each depth. Defaults to 10.
    """
    flag_too_large = False
    for i in range(1, len(path)):
        prev_path = path[:i]
        # The further the path is from the question, the greater the search space becomes.
        limit = candidate_depth_multiplier ** i
        candidate_entities = set()
        for src in question_entities:
            candidate_entities |= set(wikidata.deduce_leaves(src, prev_path, limit=limit))
            # Stop early if the number of candidate entities is already very large
            if len(candidate_entities) > limit:
                break
        # Check the entity space at each depth, if it is too large at any depth, we flag the path as too large.
        if len(candidate_entities) > limit:
            flag_too_large = True
            break
    return flag_too_large


def sample_records_from_path(path, question, question_entities, positive_connections,
                             wikidata, num_negative=15):
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
    if is_candidate_space_too_large(path, question_entities, wikidata, candidate_depth_multiplier=5):
        return []

    path = path + [END_REL]

    records = []
    tracked_entities = question_entities
    for i, current_relation in enumerate(path):
        prev_path = path[:i]
        negative_relations = sample_negative_relations(tracked_entities, prev_path, positive_connections,
                                                       num_negative, wikidata)
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
            tracked_entities = wikidata.deduce_leaves_from_multiple_srcs(tracked_entities, [current_relation], limit=100)
    return records


def get_positive_connections_along_paths(paths, path_scores, positive_threshold):
    """Collect positive connections along paths. A positive connection is defined as
    {prev_relations: next_relation}. END_REL is added to the end of each path.
    
    Returns:
        dict: a dictionary of positive connections
    """
    positive_connections = defaultdict(set)
    for path, score in zip(paths, path_scores):
        if score < positive_threshold:
            continue
        path = path + [END_REL]
        for i, rel in enumerate(path):
            positive_connections[tuple(path[:i])].add(rel)
    return positive_connections


def convert_records_relation_id_to_lable(records, wikidata):
    """Convert relation ids to relation labels in each record.
    """
    processed_records = []

    def get_relation_label(rel):
        if rel == END_REL:
            return END_REL
        return wikidata.get_relation_label(rel) or rel

    for record in tqdm(records, desc='Converting relation ids to labels'):
        record['prev_path'] = [get_relation_label(rel) for rel in record['prev_path']]
        record['positive_relation'] = get_relation_label(record['positive_relation'])
        record['negative_relations'] = [get_relation_label(rel) for rel in record['negative_relations']]
        processed_records.append(record)
    return processed_records


def write_records_to_csv(records, output_file):
    """Write records to a csv file. Per the original implementation, each record is saved as a line:
    question [SEP] prev_path separated by #, positive_relation, negative_relation1, negative_relation2, ...
    
    Args:
        records (list[dict]): list of records
        output_file (str): output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            question = record['question']
            prev_path = '#'.join(record['prev_path'])
            positive_relation = record['positive_relation']
            negative_relations = ','.join(record['negative_relations'])
            line = f"{question} [SEP] {prev_path}, {positive_relation}, {negative_relations}"
            f.write(line + '\n')


def main(args):
    wikidata = Wikidata(args.wikidata_endpoint)
    positive_threshold = args.positive_threshold
    # Each sample has the following fields:
    # - id: sample id
    # - question: question text
    # - question_entities: list of question entities
    # - answer_entities: lis
    # - question: question text
    # - question_entities: list of question entities
    # - answer_entities: list of answer entities
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

        # A dictionary of {prev_rels: {next_rel, next_rel, ...}, ...},
        # where prev_rels is a tuple of previous relations.
        positive_connections = get_positive_connections_along_paths(paths, path_scores, positive_threshold)

        for path in paths:
            train_records.extend(sample_records_from_path(path, question, question_entities,
                                                          positive_connections, wikidata))
    print(f"Number of training records: {len(train_records)}")
    train_records = convert_records_relation_id_to_lable(train_records, wikidata)
    write_records_to_csv(train_records, args.output_file)
    print(f"Saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikidata-endpoint', default='http://localhost:1234/api/endpoint/sparql', help='wikidata endpoint')
    parser.add_argument('--scored-path-file', help='The file containing scored paths')
    parser.add_argument('--output-file', help='The output file')
    parser.add_argument('--positive-threshold', type=float, default=0.5, help='The threshold to determine whether a path is positive or negative')
    args = parser.parse_args()
    
    main(args)
