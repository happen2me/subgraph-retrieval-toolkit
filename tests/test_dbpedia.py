import pytest

from srtk.knowledge_graph.dbpedia import DBpedia

DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"

@pytest.fixture
def dbpedia():
    return DBpedia(DBPEDIA_ENDPOINT, prepend_prefixes=False)

def test_get_label(dbpedia: DBpedia):
    label = dbpedia.get_label("Elizabeth_II")
    assert label == "Elizabeth II"

def test_search_one_hop_relations(dbpedia: DBpedia):
    src = "Elizabeth_II"
    dst = "Charles_III"
    relations = dbpedia.search_one_hop_relations(src, dst)
    assert ["child"] in relations and ["successor"] in relations

def test_search_two_hop_relations(dbpedia: DBpedia):
    src = "Technical_University_of_Munich"
    dst = "Bavaria"
    relations = dbpedia.search_two_hop_relations(src, dst)
    assert ["city", "federalState"] in relations

def test_deduce_leaves(dbpedia: DBpedia):
    src = "Technical_University_of_Munich"

    # Test one-hop leaves
    path = ("city",)
    leaves = dbpedia.deduce_leaves(src, path, limit=10)
    assert "Munich"  in leaves, "(TUM --city--> Munich )"
    
    # Test two-hop leaves
    path = ("city", "federalState")
    leaves = dbpedia.deduce_leaves(src, path, limit=10)
    assert "Bavaria" in leaves

def test_get_neighbor_relations(dbpedia):
    src = "Charles_III"

    # Test 1-hop relations
    relations = dbpedia.get_neighbor_relations(src, hop=1, limit=200)
    assert  all(r in relations for r in ["predecessor", "parent", "successor"]),\
        "'predecessor', 'parent', 'successor' should present in the relations"

    # Test 2-hop relations
    relations = dbpedia.get_neighbor_relations(src, hop=2)
    assert ("parent", "title") in relations, "Charles III --parent--> Elizabeth II --title--> Queen"

def test_deduce_leaves_from_multiple_srcs(dbpedia):
    srcs = ["Charles_III", "Elizabeth_II"]
    path = ("successor",)
    leaves = dbpedia.deduce_leaves_from_multiple_srcs(srcs, path, limit=10)
    assert "Charles_III" in leaves and "William,_Prince_of_Wales" in leaves
