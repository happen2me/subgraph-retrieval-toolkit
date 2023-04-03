"""3. Search Path

This corresponds to search_to_get_path.py in the RUC's code.
It enumerates all paths from the question entities to answer entities.
"""
import argparse
import srsly

# from knowledge_graph.knowledge_graph import KnowledgeGraph


def generate_paths(src_entities, dst_entities, kg, max_path=100):
    """Generate paths from question entities to answer entities.
    """
    paths = []
    for src in src_entities:
        for dst in dst_entities:
            if len(paths) >= max_path:
                break
            paths.extend(kg.search_one_hop_relaiotn(src, dst))
            paths.extend(kg.search_two_hop_relaiotn(dst, src))
    return paths[:max_path]

            
def main(args):
    # Each ground sample has the following fields:
    # - id: sample id
    # - question: question text
    # - question_entities: list of question entities
    # - answer_entities: list of answer entities
    ground_samples = srsly.read_jsonl(args.ground_file)
    # TODO: implement wikidata knowledge graph
    kg = None
    processed_samples = []
    skipped = 0
    for sample in ground_samples:
        question_entities = sample['question_entities']
        answer_entities = sample['answer_entities']
        try:
            paths = generate_paths(question_entities, answer_entities, kg)
        except Exception:
            skipped += 1
            continue
        sample['paths'] = paths
        processed_samples.append(sample)
    print(f'Processed {len(processed_samples)} samples, skipped {skipped} samples, total {len(ground_samples)} samples')
    srsly.write_jsonl(args.output_path, processed_samples)
    print(f'Output saved to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-path', type=str, required=True, help='grounded file where the question and answer entities are stored')
    parser.add_argument('--output-path', type=str, required=True, help='path file where several paths for each sample stored')
    args = parser.parse_args()

    main(args)
