import pytest
import torch

from srtk.scorer import LitSentenceEncoder


@pytest.fixture
def model():
    return LitSentenceEncoder('smallbenchnlp/roberta-small')


def examine_single_pooler(pooler):
    # Test pooler without batch size
    hidden_states = torch.randn(10, 256)
    attention_mask = torch.ones(10)
    pooled = pooler(hidden_states, attention_mask)
    assert pooled.shape == (256,)
    # Test pooler with batch size
    hidden_states = torch.randn(2, 10, 256)
    attention_mask = torch.ones(2, 10)
    pooled = pooler(hidden_states, attention_mask)
    assert pooled.shape == (2, 256)

def test_avg_pooler(model):
    if hasattr(model, 'avg_pool'):
        examine_single_pooler(model.avg_pool)
    else:
        pytest.skip('No avg pooler found')

def test_cls_pooler(model):
    if hasattr(model, 'cls_pool'):
        examine_single_pooler(model.cls_pool)
    else:
        pytest.skip('No cls pooler found')

def test_compute_sentence_similarity(model):
    # Test batched similarity
    query = torch.randn(2, 1, 10, 256)
    samples = torch.randn(2, 3, 10, 256)
    similarity = model.compute_sentence_similarity(query, samples)
    assert similarity.shape == (2, 3)
    # Test single pair similarity
    query = torch.randn(1, 10, 256)
    sample = torch.randn(1, 10, 256)
    similarity = model.compute_sentence_similarity(query, sample)
    assert similarity.shape == (1,)

def test_training_step(model):
    # Test training step
    batch = {
        'input_ids': torch.randint(0, 100, (2, 4, 10)),
        'attention_mask': torch.ones(2, 4, 10),
    }
    loss = model.training_step(batch, 0)
    assert loss.shape == ()
