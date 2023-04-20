import pytest
from srtk.scorer import Scorer


@pytest.fixture
def scorer():
    """Whether the scorer can be initialized"""
    return Scorer('smallbenchnlp/roberta-small')

def test_score(scorer):
    """Whether the scorer can run and return a digit, without crashing."""
    query = 'query: how old is the natural satellite of the earch?'
    prev_relations = ('satellite of',)
    next_relation = 'relation: age'
    scorer.score(query, prev_relations, next_relation)

def test_batch_score(scorer):
    """Whether the batch scorer is capable of batch scoring."""
    query = 'query: how old is the natural satellite of the earch?'
    prev_relations = ('satellite of',)
    next_relations = ('relation: age', 'relation: age')
    scores = scorer.batch_score(query, prev_relations, next_relations)
    assert isinstance(scores, list)
    assert len(scores) == len(next_relations)
    assert scores[0] == scores[1], 'The scores should be the same for the same samples.'
