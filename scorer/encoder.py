import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from pytorch_metric_learning.losses import NTXentLoss
from transformers import AutoModel


class LitSentenceEncoder(pl.LightningModule):
    """Lightning module """
    def __init__(self, model_name_or_path, temperature=0.07):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.model.config
        self.loss_fn = NTXentLoss(temperature=temperature)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        """Average pool the sentence embedding
        Ref: huggingface.co/intfloat/e5-large
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def training_step(self, batch, batch_idx):
        # batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
        # input_ids: [batch_size, 1(query) + 1(positive) + k(negative), seq_len]
        batch_size, n_sentences, seq_len = batch['input_ids'].shape
        input_ids = batch['input_ids'].view(-1, seq_len)
        attention_mask = batch['attention_mask'].view(-1, seq_len)
        outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        embeddings = self.average_pool(outputs.last_hidden_state, attention_mask)
        embeddings = embeddings.view(batch_size, n_sentences, -1)
        # In each sentence group, the first sentence is the query, the second is the positive,
        # and the rest are negatives. We set the positive to have the same label as the query,
        # and the negatives to have different labels, so that the query and the positive will
        # be pulled together, and the query and the negatives will be pushed apart.
        # Ref: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
        n_neg = n_sentences - 2
        # I manurally create positive pairs as (0, 1), and negative pairs as (0, 2), (0, 3), ...
        # indices_tuple (anchor1, postives, anchor2, negatives)
        indices_tuple = (torch.zeros((1,), dtype=torch.long), torch.ones((1,), dtype=torch.long),
                         torch.zeros((n_neg,), dtype=torch.long), torch.arange(2, n_sentences, dtype=torch.long))
        train_loss = 0
        for sentence_group in embeddings:
            loss = self.loss_fn(sentence_group, indices_tuple=indices_tuple)
            train_loss = train_loss + loss
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self) :
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def save_huggingface_model(self, save_dir):
        """Will save the model, so you can reload it using `from_pretrained()`."""
        save_path = Path(save_dir)
        if not save_path.exists():
            save_path.mkdir(parents=True)
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, 'pytorch_model.bin'))
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f)
