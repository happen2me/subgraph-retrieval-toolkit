import requests

import pytest

from srtk.knowledge_graph import Freebase

FREEBASE_ENDPOINT = "http://localhost:3001/sparql"


def endpoint_reachable():
    try:
        response = requests.get(FREEBASE_ENDPOINT)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

skip_if_unreachable = pytest.mark.skipif(
    not endpoint_reachable(), reason="Freebase endpoint not reachable"
)


@pytest.fixture
def freebase():
    return Freebase(FREEBASE_ENDPOINT, prepend_prefixes=True)

@skip_if_unreachable
def test_get_label(freebase: Freebase):
    # Skip the test if the fixture is skipped
    label = freebase.get_label("m.03_r3")
    assert label == "Jamaica"

@skip_if_unreachable
def test_search_one_hop_relations(freebase: Freebase):
    src = "m.03_r3"
    dst = "m.01428y"
    relations = freebase.search_one_hop_relations(src, dst)
    assert ["location.country.languages_spoken"] in relations, "(Jamaica --speak--> Jamaican English)"

@skip_if_unreachable
def test_search_two_hop_relations(freebase: Freebase):
    src = "m.0157m"
    dst = "m.0crpbj"
    relations = freebase.search_two_hop_relations(src, dst)
    assert ["people.person.education", "education.education.institution"] in relations, \
        "Bill Clinton --education-->  ?? --institution--> Hot Springs High School"

@skip_if_unreachable
def test_deduce_leaves(freebase: Freebase):
    src = "m.0157m"

    # Test one-hop leaves
    path = ("people.person.education",)
    leaves = freebase.deduce_leaves(src, path, limit=10)
    assert "m.0125cddf"  in leaves, "(Bill Clinton --education-->  )"
    
    # Test two-hop leaves
    path = ("people.person.education", "education.education.institution")
    leaves = freebase.deduce_leaves(src, path, limit=10)
    assert "m.0crpbj" in leaves, "(Bill Clinton --education-->  --insittution--> Hot Springs High School"

@skip_if_unreachable
def test_get_neighbor_relations(freebase):
    src = "m.0157m"
    # Test 1-hop relations
    relations = freebase.get_neighbor_relations(src, hop=1, limit=200)
    assert "government.us_president.vice_president" in relations and "government.politician.party" in relations,\
        "Billclinton has 1-hop relations vice_president and party"

    # Test 2-hop relations
    src = "m.0157m"
    relations = freebase.get_neighbor_relations(src, hop=2, limit=200)
    assert ('government.us_president.vice_president', 'people.person.gender') in relations,\
        "Bill Clinton --vice president--> --gender--> ..."

@skip_if_unreachable
def test_deduce_leaves_from_multiple_srcs(freebase: Freebase):
    srcs = ["m.0157m", "m.02mjmr"]  # Bill and Barack

    # Test searching one-hop common leaves
    path = ("government.us_president.vice_president",)
    leaves = freebase.deduce_leaves_from_multiple_srcs(srcs, path, limit=10)
    assert "m.0d05fv" in leaves, "(Bill | Barack --vice president--> Joe Biden )"
