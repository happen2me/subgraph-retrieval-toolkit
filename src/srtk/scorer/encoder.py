import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel


class LitSentenceEncoder(pl.LightningModule):
    """Lightning module """
    def __init__(self, model_name_or_path, temperature=0.07, lr=5e-5, pool='cls'):
        if pool not in ['cls', 'avg']:
            raise ValueError(f"pool method must be either cls or avg, got {pool}")
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if self.model.config.is_encoder_decoder:
            self.model = self.model.encoder
            print("The model is an encoder-decoder model, only the encoder will be used.")
        self.config = self.model.config
        self.temperature = temperature
        self.lr = lr
        self.pool_method = pool
        self.loss_fn = F.cross_entropy

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def cls_pool(last_hidden_states, attention_mask=None):
        """CLS pool the sentence embedding.
        This is the pooling method adopted by RUC's SR paper.

        Args:
            last_hidden_states: [..., seq_len, embedding_dim]
            attention_mask: [..., seq_len] silently ignored!
                It exists for compatibility with other pooling methods.

        Returns:
            torch.Tensor: pooled_embedding [..., embedding_dim]
        """
        return last_hidden_states[..., 0, :]

    @staticmethod
    def avg_pool(last_hidden_states, attention_mask=None):
        """Average pool the sentence embedding.

        Args:
            last_hidden_states (torch.Tensor): [..., seq_len, embedding_dim]
            attention_mask (torch.Tensor): [..., seq_len]

        Returns:
            torch.Tensor: pooled_embedding [..., embedding_dim]
        """
        # Compute the average embedding, ignoring the padding tokens.
        if attention_mask is None:
            attention_mask = torch.ones(last_hidden_states.shape[:-1], device=last_hidden_states.device)
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=-2) / attention_mask.sum(dim=-1)[..., None]

    def compute_embedding_similarity(self, query, target):
        """Compute the similarity between query and target(s) embeddings.

        Args:
            query (torch.Tensor): [batch_size, 1, embedding_dim]
            target (torch.Tensor): [batch_size, k, embedding_dim]

        Returns:
            torch.Tensor: similarity [batch_size, k]
        """
        return F.cosine_similarity(query, target, dim=-1) / self.temperature

    def compute_sentence_similarity(self, query, target, query_mask=None, target_mask=None):
        """Compute the similarity between query and target(s) sentence embeddings.
        The query & target(s) sentence embedding are first pooled. Then the similarity
        is computed between the pooled query and target(s) embeddings. 

        Args:
            query (torch.Tensor): [..., 1, seq_len, embedding_dim]
                query sentence embedding
            target (torch.Tensor): [..., k, seq_len, embedding_dim]
                target sentence(s) embedding

        Returns:
            torch.Tensor: similarity [..., k]
        """
        embeddings = torch.cat([query, target], dim=-3)  # [..., 1 + k, seq_len, embedding_dim]
        if self.pool_method == 'cls':
            embeddings = self.cls_pool(embeddings) # [..., 1 + k, embedding_dim]
        else:
            if query_mask is None:
                query_mask = torch.ones(query.shape[:-1], device=query.device)
            if target_mask is None:
                target_mask = torch.ones(target.shape[:-1], device=target.device)
            attention_mask = torch.cat([query_mask, target_mask], dim=-2)  # [..., 2 + k, seq_len]
            embeddings = self.avg_pool(embeddings, attention_mask)
        query_embeddings = embeddings[..., 0:1, :]  # [..., 1, embedding_dim]
        samples_embeddings = embeddings[..., 1:, :]  # [..., k, embedding_dim]
        similarity = self.compute_embedding_similarity(query_embeddings, samples_embeddings)
        return similarity

    def batch_forward(self, batch):
        """The common forward function for both training and inference."""
        # In each sentence group, the first sentence is the query, the second is the positive,
        # and the rest are negatives.
        # batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
        # input_ids: [batch_size, 1(query) + 1(positive) + neg (negative), seq_len]
        batch_size, n_samples, seq_len = batch['input_ids'].shape  # n_sentences = 1 + 1 + neg
        input_ids = batch['input_ids'].view(-1, seq_len)
        attention_mask = batch['attention_mask'].view(-1, seq_len)
        # outputs.last_hidden_state: [batch_size * (1 + 1 + neg), seq_len, embedding_dim]
        outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        embeddings = outputs.last_hidden_state.view(batch_size, n_samples, seq_len, -1)
        query_embedding = embeddings[:, 0:1]  # [batch_size, 1, seq_len, embedding_dim]
        samples_embedding = embeddings[:, 1:]  # [batch_size, 1 + neg, seq_len, embedding_dim]
        # similarity: [batch_size, 1 + neg]
        query_samples_similarity = self.compute_sentence_similarity(query_embedding, samples_embedding,
            query_mask=batch['attention_mask'][:, 0:1], target_mask=batch['attention_mask'][:, 1:])
        # The zerot-th label, where positive sample locates, is set to 0.
        labels = torch.zeros(query_samples_similarity.shape[0], dtype=torch.long,
                             device=query_samples_similarity.device)
        loss = self.loss_fn(query_samples_similarity, labels)
        return loss

    def training_step(self, batch, batch_idx):
        train_loss =  self.batch_forward(batch)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss =  self.batch_forward(batch)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self) :
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def save_huggingface_model(self, save_dir):
        """Will save the model, so you can reload it using `from_pretrained()`."""
        save_path = Path(save_dir)
        if not save_path.exists():
            save_path.mkdir(parents=True)
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, 'pytorch_model.bin'))
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f)
