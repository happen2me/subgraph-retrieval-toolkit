import json
import requests
from argparse import Namespace

import pytest
from srtk.link import link

def check_url_availability(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def test_dbpedia_linker():
    question = {"id": "berlin", "question": "Berlin is the capital of Germany."}
    with open("question.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(question) + "\n")
    dbpedia_endpoint = "https://api.dbpedia-spotlight.org/en/annotate"
    if not check_url_availability(dbpedia_endpoint):
        pytest.skip("DBpedia endpoint is not available.")
    args = Namespace(
        input="question.jsonl",
        output="linked.jsonl",
        knowledge_graph="dbpedia",
        ground_on="question",
        el_endpoint=dbpedia_endpoint,
        service="spotlight",
        token=None,
    )
    link(args)
    with open("linked.jsonl", "r", encoding="utf-8") as f:
        linked = json.loads(f.readline())
    assert "Berlin" in linked["question_entities"]
    assert "Germany" in linked["question_entities"]
