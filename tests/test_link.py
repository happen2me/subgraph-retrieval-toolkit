import json
from argparse import Namespace

from srtk.link import link


def test_dbpedia_linker():
    question = {"id": "berlin", "question": "Berlin is the capital of Germany."}
    with open("question.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(question) + "\n")
    args = Namespace(
        input="question.jsonl",
        output="linked.jsonl",
        knowledge_graph="dbpedia",
        ground_on="question",
        el_endpoint="https://api.dbpedia-spotlight.org/en/annotate",
    )
    link(args)
    with open("linked.jsonl", "r", encoding="utf-8") as f:
        linked = json.loads(f.readline())
    assert "Berlin" in linked["question_entities"]
    assert "Germany" in linked["question_entities"]
