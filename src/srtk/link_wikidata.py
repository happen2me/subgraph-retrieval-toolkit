"""Entity linking
This step links the entity mentions in the question to the entities in the Wikidata knowledge graph.
It inference on the REL endpoint.
"""
import argparse
import requests
import socket
from pathlib import Path

import srsly
from tqdm import tqdm

from wikimapper import WikiMapper


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
        host = args.el_endpoint.split(':')[-2].split('/')[-1]
        port = int(args.el_endpoint.split(':')[-1])
    except Exception as exc:
        raise ValueError(f"Can't parse the endpoint {args.el_endpoint}") from exc
    if not socket_reachable(host, port):
        raise RuntimeError(f"Can't reach the endpoint {args.el_endpoint}")
    mapper = WikiMapper(args.wikimapper_db)

    with open(args.input, "r", encoding="utf-8") as f:
        total_lines = len(f.readlines())
    questions = srsly.read_jsonl(args.input)
    cnt_qid_not_found = 0
    cnt_total_ground = 0
    all_linked = []
    for question in tqdm(questions, total=total_lines, desc=f"Entity linking {args.input}"):
        document = {
            'text': question[args.ground_on],
        }
        api_results = requests.post(args.el_endpoint, json=document, timeout=60).json()
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
            "question": question[args.ground_on],
            "question_entities": qids,
            "spans": spans,
            "entity_names": entities
        }
        if "id" in question:
            linked["id"] = question["id"]
        all_linked.append(linked)
    print(f"{cnt_qid_not_found} / {cnt_total_ground} grounded entities not converted to Wikidata qids")
    # check whether the folder exists
    folder_path = Path(args.output).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print(f"Folder {folder_path} created")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(args.output, all_linked)
    print(f"Entity linking result saved to {args.output}")


def _add_arguments(parser):
    """Add entity linking arguments to the parser"""
    parser.description = '''Entity linking on Wikidata.
    The input is a jsonl file. The field of interest is specified by the argument --ground-on.
    The output is a jsonl file, each line is a dict with keys: id, question_entities, spans, entity_names.
    '''
    parser.add_argument('-i', '--input', type=str, help='Input file path, in which the question is stored')
    parser.add_argument('-o', '--output', type=str, help='Output file path, in which the entity linking result is stored')
    parser.add_argument('-e', '--el-endpoint', type=str, default='http://127.0.0.1:1235', help='REL endpoint')
    parser.add_argument('--wikimapper-db', type=str, default='resources/wikimapper/index_enwiki.db', help='Wikimapper database path')
    parser.add_argument('--ground-on', type=str, default='question', help='The key to ground on, the corresponding text will be sent to the REL endpoint for entity linking')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()
    main(args)
