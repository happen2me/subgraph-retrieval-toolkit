"""The script to train the scorer model.

e.g.
python train.py --data-file data/train.jsonl --model-name-or-path intfloat/e5-small --save-model-path artifacts/scorer
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from scorer.encoder import LitSentenceEncoder


def concate_all(example):
    """Concatenate all columns into one column for input.
    The resulted 'input_text' column is a list of strings.
    """
    query = 'query: ' + example['query']
    rels = [example['positive']] + example['negatives']
    rels = ['relation: ' + rel for rel in rels]
    example['input_text'] = [query] + rels
    return example


@dataclass
class Collator:
    """Collate a list of examples into a batch."""
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        batched = defaultdict(list)
        for item in features:
            for key, value in item.items():
                value = torch.tensor(value)
                if key == 'attention_mask':
                    value = value.bool()
                batched[key].append(value)
        for key, value in batched.items():
            batched[key] = torch.stack(value, dim=0)
        return batched


def prepare_dataloaders(data_file, model_name_or_path, batch_size):
    """Prepare dataloaders for training and validation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    def tokenize(example):
        tokenized = tokenizer(example['input_text'], padding='max_length', truncation=True, return_tensors='pt', max_length=32)
        return tokenized
    train_dataset = load_dataset('json', data_files=data_file, split='train[:95%]')
    train_dataset = train_dataset.map(concate_all, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    validation_dataset = load_dataset('json', data_files=data_file, split='train[95%:]')
    validation_dataset = validation_dataset.map(concate_all, remove_columns=validation_dataset.column_names)
    validation_dataset = validation_dataset.map(tokenize, remove_columns=validation_dataset.column_names)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Collator(tokenizer), num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer), num_workers=8)
    return train_loader, validation_loader


def main(args):
    model = LitSentenceEncoder(args.model_name_or_path)
    train_loader, validation_loader = prepare_dataloaders(args.data_file, args.model_name_or_path, args.batch_size)
    trainer = pl.Trainer(accelerator=args.accelerator, default_root_dir=args.save_model_path, fast_dev_run=args.fast_dev_run)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    model.save_huggingface_model(args.save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default='data/retrieval/train.jsonl', help='train data')
    parser.add_argument('--model-name-or-path', default='intfloat/e5-small', help='pretrained model name or path')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--max-epochs', default=10, type=int, help='max epochs')
    parser.add_argument('--accelerator', default='gpu', help='accelerator, can be cpu, gpu, or tpu')
    parser.add_argument('--save-model-path', default='artifacts/scorer', help='output model checkpoint path')
    parser.add_argument('--fast-dev-run', action='store_true')
    args = parser.parse_args()

    main(args)
