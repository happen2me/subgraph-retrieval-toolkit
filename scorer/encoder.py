import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel


class LitSentenceEncoder(pl.LightningModule):
    """Lightning module """
    def __init__(self, model_name_or_path, temperature=0.07):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.model.config
        self.loss_fn = F.cross_entropy
        self.temperature = temperature

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def cls_pool(last_hidden_states):
        """CLS pool the sentence embedding.
        This is the pooling method adopted by RUC's SR paper.
        
        Args:
            last_hidden_states: [..., seq_len, embedding_dim]
        Returns:
            pooled_embedding: [..., embedding_dim]
        """
        return last_hidden_states[..., 0, :]

    def compute_embedding_similarity(self, query, target):
        """Compute the similarity between query and target(s) embeddings.
        
        Args:
            query: [batch_size, 1, embedding_dim]
            target: [batch_size, k, embedding_dim]
        
        Returns:
            similarity: [batch_size, k]
        """
        return F.cosine_similarity(query, target, dim=-1) / self.temperature

    def compute_sentence_similarity(self, query, target):
        """Compute the similarity between query and target(s) sentence embeddings.
        The query & target(s) sentence embedding are first pooled. Then the similarity
        is computed between the pooled query and target(s) embeddings. 
        
        Args:
            query: [batch_size, 1, seq_len, embedding_dim]
                query sentence embedding
            target: [batch_size, k, seq_len, embedding_dim]
                target sentence(s) embedding
        
        Returns:
            similarity: [batch_size, k]    
        """        
        embeddings = torch.cat([query, target], dim=-3)  # [batch_size, 1 + k, seq_len, embedding_dim]
        embeddings = self.cls_pool(embeddings) # [batch_size, 1 + k, embedding_dim]
        query_embeddings = embeddings[..., 0:1, :]  # [batch_size, 1, embedding_dim]
        samples_embeddings = embeddings[..., 1:, :]  # [batch_size, k, embedding_dim]
        similarity = self.compute_embedding_similarity(query_embeddings, samples_embeddings)
        return similarity

    def training_step(self, batch, batch_idx):
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
        query_samples_similarity = self.compute_sentence_similarity(query_embedding, samples_embedding)
        # The zerot-th label, where positive sample locates, is set to 0.
        labels = torch.zeros(query_samples_similarity.shape[0], dtype=torch.long,
                             device=query_samples_similarity.device)
        train_loss = self.loss_fn(query_samples_similarity, labels)
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
