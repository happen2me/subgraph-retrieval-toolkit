from scorer import Scorer


def test_scorer():
    """Whether the scorer can be initialized, run and return a digit, without crashing."""
    scorer = Scorer('smallbenchnlp/roberta-small')
    query = 'query: how old is the natural satellite of the earch?'
    prev_relations = ('satellite of',)
    next_relation = 'relation: age'
    scorer.score(query, prev_relations, next_relation)
