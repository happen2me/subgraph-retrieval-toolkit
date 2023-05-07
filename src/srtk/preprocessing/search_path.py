"""3. Search Path

This corresponds to search_to_get_path.py in the RUC's code.
It enumerates all paths from the question entities to answer entities.

python preprocess/search_path.py --ground-path data/preprocess/merged-ground.jsonl --output-path data/preprocess/paths.jsonl --remove-sample-without-path
"""
import sys
import os
import argparse

import srsly
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from knowledge_graph import KnowledgeGraphBase, get_knowledge_graph


def generate_paths(src_entities, dst_entities, kg: KnowledgeGraphBase, max_path=50):
    """Generate paths from question entities to answer entities.
    """
    paths = []
    for src in src_entities:
        for dst in dst_entities:
            if len(paths) >= max_path:
                break
            one_hop_paths = kg.search_one_hop_relations(src, dst)
            paths.extend(one_hop_paths)
            # If there are alreay one-hop paths between the two entities, 
            # we don't need to look further.
            if len(one_hop_paths) == 0:
                paths.extend(kg.search_two_hop_relations(src, dst))
    paths = [tuple(path) for path in paths]
    paths = list(set(paths))
    return paths[:max_path]


def has_type_relation(path):
    """A utility function to check whether the path contain certain relations."""
    for rel in path:
        if rel in ('type.object.type', 'type.type.instance'):
            return False
    return True


def main(args):
    # Each ground sample has the following fields:
    # - id: sample id
    # - question: question text
    # - question_entities: list of question entities
    # - answer_entities: list of answer entities
    ground_samples = srsly.read_jsonl(args.ground_path)
    total_samples = sum(1 for _ in srsly.read_jsonl(args.ground_path))
    kg = get_knowledge_graph(args.knowledge_graph, args.sparql_endpoint)
    processed_samples = []
    skipped = 0
    for sample in tqdm(ground_samples, total=total_samples, desc='Searching paths'):
        question_entities = sample['question_entities']
        answer_entities = sample['answer_entities']
        try:
            paths = generate_paths(question_entities, answer_entities, kg)
        except Exception as e:
            skipped += 1
            print(e)
            continue
        if args.remove_sample_without_path and not paths:
            skipped += 1
            continue
        # Special filter for Freebase
        if args.knowledge_graph == 'freebase':
            paths = list(filter(has_type_relation, paths))
        sample['paths'] = paths
        processed_samples.append(sample)
    print(f'Processed {len(processed_samples)} samples; skipped {skipped} samples without any paths between question entities and answer entities; total {total_samples} samples')
    srsly.write_jsonl(args.output_path, processed_samples)
    print(f'Retrieved paths saved to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparql-endpoint', default='http://localhost:1234/api/endpoint/sparql', help='knowledge graph SPARQL endpoint')
    parser.add_argument('--knowledge-graph', type=str, default='wikidata', choices=('wikidata', 'freebase', 'dbpedia'), help='knowledge graph name (default: wikidata)')
    parser.add_argument('--ground-path', type=str, required=True, help='grounded file where the question and answer entities are stored')
    parser.add_argument('--output-path', type=str, required=True, help='path file where several paths for each sample stored')
    parser.add_argument('--remove-sample-without-path', action='store_true', help='remove samples without paths')
    args = parser.parse_args()

    main(args)
