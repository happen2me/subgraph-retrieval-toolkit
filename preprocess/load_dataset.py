"""1. Load the dataset
This program filters the grounded samples, removing those without any answer or question entity.

- For Mintaka dataset, those without any answer entity are removed.
$ python load_dataset.py --dataset mintaka --ground-path data/mintaka/grounded/train.jsonl --output-path data/preprocess/mintaka.grounded.train.jsonl
Train: Processed 8537 samples, skipped 5463 samples, total 14000 samples
Validation: Processed 1223 samples, skipped 777 samples, total 2000 samples

- For MKQA dataset, those without any question entity are removed.
$ python preprocess/load_dataset.py --dataset mkqa --ground-path data/mkqa/grounded/ground.jsonl --output-path data/preprocess/mkqa.grounded.jsonl
Processed 2112 samples, skipped 7888 samples, total 10000 samples
"""
import argparse
from pathlib import Path

import srsly
from tqdm import tqdm

def main(args):
    samples= srsly.read_jsonl(args.ground_path)
    total_lines = sum(1 for _ in srsly.read_jsonl(args.ground_path))
    skipped = 0
    processed_samples = []
    for sample in tqdm(samples, total=total_lines):
        if len(sample['qc']) == 0 or len(sample['ac']) == 0 or None in sample['ac']:
            skipped += 1
            continue
        processed_sample = {
            'id': args.dataset + '_' + str(sample['id']),
            'question': sample['sent'],
            'question_entities': sample['qc'],
            'answer_entities': sample['ac'],
        }
        processed_samples.append(processed_sample)

    output_path = Path(args.output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    srsly.write_jsonl(output_path, processed_samples)
    print(f'Processed {len(processed_samples)} samples, skipped {skipped} samples, total {total_lines} samples')
    print(f'Output saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help='dataset name, which will be prepend to sample ids')
    parser.add_argument('--ground-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()
    main(args)
