"""1. Entity linking
This step links the entity mentions in the question to the entities in the Wikidata knowledge graph.
It inference on the REL endpoint.
"""
import argparse
import requests
from pathlib import Path

import srsly
from tqdm import tqdm

from wikimapper import WikiMapper

import socket

def socket_reachable(host, port):
    """
    Check if a socket is reachable
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2) # set a timeout value for the socket
        s.connect((host, port))
        s.close()
        return True
    except Exception as err:
        print(err)
        return False


def main(args):
    try:
        host = args.endpoint.split(':')[-2].split('/')[-1]
        port = int(args.endpoint.split(':')[-1])
    except Exception as exc:
        raise ValueError(f"Can't parse the endpoint {args.endpoint}") from exc
    if not socket_reachable(host, port):
        raise RuntimeError(f"Can't reach the endpoint {args.endpoint}")
    mapper = WikiMapper(args.wikimapper_db)

    with open(args.statement_path, "r", encoding="utf-8") as f:
        total_lines = len(f.readlines())
    statements = srsly.read_jsonl(args.statement_path)
    cnt_qid_not_found = 0
    cnt_total_ground = 0
    all_linked = []
    for statement in tqdm(statements, total=total_lines, desc=f"Entity linking {args.statement_path}"):
        document = {
            'text': statement[args.ground_on],
        }
        api_results = requests.post(args.endpoint, json=document, timeout=60).json()
        cnt_total_ground += len(api_results)
        qids = []
        spans = []
        entities = []
        for result in api_results:
            start_pos, mention_length, mention, entity, disambiguation_cofidence, mention_detection_confidence, tag = result
            qid = mapper.title_to_id(entity)
            span = (start_pos, start_pos + mention_length)
            if qid is None:
                cnt_qid_not_found += 1
            else:
                qids.append(qid)
                entities.append(entity)
                spans.append(span)
        linked = {
            "id": statement["id"],
            "qid": qids,
            "span": spans,
            "entity": entities
        }
        all_linked.append(linked)
    print(f"{cnt_qid_not_found} / {cnt_total_ground} grounded entities not converted to Wikidata qids")
    # check whether the folder exists
    folder_path = Path(args.output_path).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print(f"Folder {folder_path} created")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(args.output_path, all_linked)
    print(f"Entity linking result saved to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:1235', help='REL endpoint')
    parser.add_argument('--wikimapper-db', type=str, default='data/kilt/index_enwiki-latest-uncased.db', help='Wikimapper database path')
    parser.add_argument('--ground-on', type=str, default='question', help='The key to ground on, either "question" or "statement')
    parser.add_argument('--statement-path', type=str, help='Statement file path, in which the question is stored')
    parser.add_argument('--output-path', type=str, help='Output file path, in which the entity linking result is stored')
    args = parser.parse_args()
    main(args)
