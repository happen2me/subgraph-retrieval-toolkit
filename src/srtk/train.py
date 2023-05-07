"""The script to train the scorer model.

e.g.
python train.py --data-file data/train.jsonl --model-name-or-path intfloat/e5-small --save-model-path artifacts/scorer
"""
import argparse
import datetime
from collections import defaultdict
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .scorer import LitSentenceEncoder


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


def prepare_dataloaders(train_data, validation_data, tokenizer, batch_size):
    """Prepare dataloaders for training and validation.

    If validation dataset is not provided, 5 percent of the training data will be used as validation data.
    """
    def tokenize(example):
        tokenized = tokenizer(example['input_text'], padding='max_length', truncation=True, return_tensors='pt', max_length=32)
        return tokenized

    train_split = 'train[:95%]' if validation_data is None else 'train'
    validation_split = 'train[95%:]' if validation_data is None else 'train'
    if validation_data is None:
        validation_data = train_data

    train_dataset = load_dataset('json', data_files=train_data, split=train_split)
    train_dataset = train_dataset.map(concate_all, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    validation_dataset = load_dataset('json', data_files=validation_data, split=validation_split)
    validation_dataset = validation_dataset.map(concate_all, remove_columns=validation_dataset.column_names)
    validation_dataset = validation_dataset.map(tokenize, remove_columns=validation_dataset.column_names)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Collator(tokenizer), num_workers=8)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer), num_workers=8)
    return train_loader, validation_loader


def train(args):
    """Train the scorer model.

    The model compares the similarity between [question; previous relation] and the next relation.
    """
    torch.set_float32_matmul_precision('medium')
    model = LitSentenceEncoder(args.model_name_or_path, lr=args.learning_rate, loss=args.loss)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_loader, validation_loader = prepare_dataloaders(args.train_dataset, args.validation_dataset, tokenizer, args.batch_size)
    day_hour = datetime.datetime.now().strftime('%m%d%H%M')
    wandb_logger = WandbLogger(project=args.wandb_project, name=day_hour , group=args.wandb_group, save_dir=args.wandb_savedir)
    trainer = pl.Trainer(accelerator=args.accelerator, default_root_dir=args.output_dir,
                         fast_dev_run=args.fast_dev_run, max_epochs=args.max_epochs, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    model.save_huggingface_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def _add_arguments(parser):
    """Add train arguments to a parser in place."""
    parser.add_argument('-t', '--train-dataset', required=True,
                        help='path to the training dataset. It should be a JSONL file with fields: query, positive, negatives')
    parser.add_argument('-v', '--validation-dataset',
                        help='path to the validation dataset. If not provided, 5 percent of the training data will be used as validation data.\
                        (default: None)')
    parser.add_argument('-o', '--output-dir', default='artifacts/scorer',
                        help='output model path. the model will be saved in the format of huggingface models,\
                        which can be uploaded to the huggingface hub and shared with the community.\
                        (default: artifacts/scorer)')
    parser.add_argument('-m', '--model-name-or-path', default='intfloat/e5-small',
                        help='pretrained model name or path. It is fully compatible with HuggingFace models.\
                        You can specify either a local path where a model is saved, or an encoder model identifier\
                        from huggingface hub. (default: intfloat/e5-small)')
    parser.add_argument('-lr', '--learning-rate', default=5e-5, type=float, help='learning rate (default: 5e-5)')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size (default: 16)')
    parser.add_argument('--loss', default='cross_entropy', choices=['cross_entropy', 'contrastive'],
                        help='loss function, can be cross_entropy or contrastive (default: cross_entropy)')
    parser.add_argument('--max-epochs', default=10, type=int, help='max epochs (default: 10)')
    parser.add_argument('--accelerator', default='gpu', help='accelerator, can be cpu, gpu, or tpu (default: gpu)')
    parser.add_argument('--fast-dev-run', action='store_true',
                        help='fast dev run for debugging, only use 1 batch for training and validation')
    parser.add_argument('--wandb-project', default='retrieval', help='wandb project name (default: retrieval)')
    parser.add_argument('--wandb-group', default='contrastive', help='wandb group name (default: contrastive)')
    parser.add_argument('--wandb-savedir', default='artifacts', help='wandb save directory (default: artifacts)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()

    train(args)
