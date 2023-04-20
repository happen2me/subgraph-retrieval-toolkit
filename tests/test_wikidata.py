import pytest

from srtk.knowledge_graph import Wikidata

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

@pytest.fixture
def wikidata():
    return Wikidata(WIKIDATA_ENDPOINT, prepend_prefixes=False, exclude_qualifiers=True)

def test_get_label(wikidata: Wikidata):
    label = wikidata.get_label("Q157808")
    assert label == "Technical University of Munich"

def test_search_one_hop_relations(wikidata: Wikidata):
    src = "Q157808"
    dst = "Q183"
    relations = wikidata.search_one_hop_relations(src, dst)
    assert ["P17"] in relations, "(TUM --country--> Germany)"

def test_search_two_hop_relations(wikidata: Wikidata):
    src = "Q157808"
    dst = "Q458"
    relations = wikidata.search_two_hop_relations(src, dst)
    assert ["P17", "P361"] in relations, "(TUM --country--> Germany --part of--> European Union)"

def test_deduce_leaves(wikidata: Wikidata):
    src = "Q157808"

    # Test one-hop leaves
    path = ("P17",)
    leaves = wikidata.deduce_leaves(src, path, limit=10)
    assert "Q183"  in leaves, "(TUM --country--> Germany )"
    
    # Test two-hop leaves
    path = ("P17", "P361")
    leaves = wikidata.deduce_leaves(src, path, limit=10)
    assert "Q458" in leaves and "Q27509" in leaves, "(TUM --country--> Germany --part of--> European Union | Central Europe)"

def test_get_neighbor_relations():
    src = "Q157808"
    wikidata = Wikidata(WIKIDATA_ENDPOINT, exclude_qualifiers=True)

    # Test 1-hop relations AND exclude_qualifiers=True
    relations = wikidata.get_neighbor_relations(src, hop=1, limit=200)
    assert "P17" in relations and "P112" in relations,  "TUM has 1-hop relations country and founded by"
    assert "P571" not in relations, "inception should be filtered out as it is a quantifier"

    # Test 2-hop relations
    relations = wikidata.get_neighbor_relations(src, hop=2)
    assert ("P17", "P361") in relations, "TUM --country--> Germany --part of--> European Union"

    # The wikidata endpoint is re-initialized to avoid caching
    wikidata = Wikidata(WIKIDATA_ENDPOINT, prepend_prefixes=False, exclude_qualifiers=False)
    # Test exclude_qualifiers=False
    relations = wikidata.get_neighbor_relations(src, hop=1, limit=200)
    assert "P571" in relations, "inception should be present when qualifiers are not excluded"

def test_deduce_leaves_from_multiple_srcs(wikidata: Wikidata):
    srcs = ["Q157808", "Q55044"]  # TUM and LMU

    # Test searching one-hop common leaves
    path = ("P17",)
    leaves = wikidata.deduce_leaves_from_multiple_srcs(srcs, path, limit=10)
    assert "Q183" in leaves, "(TUM | LMU --country--> Germany )"
