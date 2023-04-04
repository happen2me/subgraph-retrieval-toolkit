import torch
import lightning.pytorch as pl
from transformers import AutoModel
from pytorch_metric_learning import losses


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class LitSentenceEncoder(pl.LightningModule):
    def __init__(self, model_name_or_path, temperature=0.07):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.model.config
        self.loss_fn = losses.NTXentLoss(temperature=temperature)

    def training_step(self, batch, batch_idx):
        # batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
        # input_ids: [batch_size, 1(query) + 1(positive) + k(negative), seq_len]
        batch_size, n_sentences, seq_len = batch['input_ids'].shape
        input_ids = batch['input_ids'].view(-1, seq_len)
        attention_mask = batch['attention_mask'].view(-1, seq_len)
        outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        embeddings = embeddings.view(batch_size, n_sentences, -1)
        # In each sentence group, the first sentence is the query, the second is the positive,
        # and the rest are negatives. We set the positive to have the same label as the query,
        # and the negatives to have different labels, so that the query and the positive will
        # be pulled together, and the query and the negatives will be pushed apart.
        # Ref: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
        labels = torch.arange(n_sentences).to(embeddings.device)
        labels[1] = labels[0]
        train_loss = 0
        for sentence_group in embeddings:
            loss = self.loss_fn(sentence_group, labels)
            train_loss = train_loss + loss
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self) :
        return torch.optim.Adam(self.parameters(), lr=1e-5)
