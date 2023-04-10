"""2. This script merges the grounded data into one training data.

e.g.
python preprocess/merge_ground.py  --output-path data/preprocess/merged-ground.jsonl --ground-files\
    data/preprocess/mintaka-ground.jsonl data/preprocess/mkqa-ground.jsonl
"""
import argparse
import srsly

def main(args):
    merged_samples = []
    for ground_file in args.ground_files:
        samples = srsly.read_jsonl(ground_file)
        merged_samples.extend(samples)
    srsly.write_jsonl(args.output_path, merged_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--ground-files', nargs='+', required=True)
    args = parser.parse_args()
    main(args)
