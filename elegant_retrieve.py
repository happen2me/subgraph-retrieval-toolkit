import argparse
import torch
import srsly
from typing import Tuple, List, Any, Dict
from collections import namedtuple
import heapq

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from scorer import Scorer

from knowledge_graph import Freebase

END_REL = "END OF HOP"

Path = namedtuple('Path', ['prev_relations', 'score'],
                  defaults=[(), 0])


def path_to_subgraph(topic_entity, path):
    """输入topic_entity, path, 得到对应的实例化子图——节点、集合三元组集合"""
    nodes, triples = set(), set()
    hop_nodes, next_hop_set = set(), set()
    hop_nodes.add(topic_entity)
    for relation in path:
        next_hop_set = set()
        if relation == END_REL:
            continue
        for node in hop_nodes:
            tail_set = set(kg.deduce_leaves(node, (relation,)))
            next_hop_set |= tail_set
            triples |= {(node, relation, tail) for tail in tail_set}
        hop_nodes = next_hop_set
        nodes = nodes | hop_nodes
    return list(nodes), list(triples)


def deduce_leaves(entity, path):
    hop_nodes, next_hop_set = set(), set()
    hop_nodes.add(entity)
    for relation in path:
        if relation == END_REL:
            continue
        next_hop_set = set()
        for node in hop_nodes:
            next_hop_set |= set(kg.deduce_leaves(node, (relation,)))
        hop_nodes = next_hop_set
    return hop_nodes


def deduce_leaves_relation_by_path(entity, path):
    relations = set()
    leaves = deduce_leaves(entity, path)
    relations = relations.union(
        *(kg.get_neighbor_relations(leave) for leave in leaves))
    return relations


def path_to_candidate_relations(topic_entity: str, path: List[str]) -> List[str]:
    """输入topic_entity, 得到叶子结点能提供的候选relation的集合"""
    new_relations = deduce_leaves_relation_by_path(topic_entity, path)
    # filter relation
    candidate_relations = [r for r in new_relations if r.split(".")[0] not in [
        "kg", "common"]]
    return tuple(candidate_relations)


# def mega_batch_score(questions, prev_relations_list, next_relations_list):
#     """A mega batch computes similarity on a batch of (question, prev_relations, next_relations)
    
#     Args:
#         questions (List[str]): questions
#         prev_relations_list (List[List[str]]): previous relations
#         next_relations_list (List[List[str]]): next relations
#     """
#     queries = [f"query: {question} [SEP] {' # '.join(prev_relations)}" 
#                for question, prev_relations in zip(questions, prev_relations_list)]
#     next_relations = list(set(sum(next_relations_list, [])))


def score_path_list_and_relation_list(question: str, candidate_paths: List[Path], next_relations_batched: List[List[str]]) -> List[Path]:
    """计算path和其对应的候选的relation的得分"""
    results = []
    sim_score = []
    for path, next_relations in zip(candidate_paths, next_relations_batched):
        scores = scorer.batch_score(question, tuple(path.prev_relations), tuple(next_relations))
        sim_score.append(scores)

    for i, (path, next_relations) in enumerate(zip(candidate_paths, next_relations_batched)):
        for j, relation in enumerate(next_relations):
            new_prev_relations = path.prev_relations + (relation,)
            score = float(sim_score[i][j]) + path.score
            results.append(Path(new_prev_relations, score))
    return results


def beam_search_path(question: str, topic_entity: str, beam_width: int, max_hop: int) -> List[Tuple[List[str], float]]:
    """从KB中进行宽为num_beams的搜索,得到num_return_paths条路径并提供对应得分"""
    candidate_paths = [Path()]  # path and its score
    result_paths = []
    n_hop = 0
    while candidate_paths and len(result_paths) < beam_width and n_hop < max_hop:
        search_list = []
        # try every possible next_hop
        next_relations_batched = []
        for path in candidate_paths:
            prev_relations = path.prev_relations
            next_relations = path_to_candidate_relations(topic_entity, prev_relations)
            next_relations = next_relations + (END_REL,)
            next_relations_batched.append(next_relations)
        search_list = score_path_list_and_relation_list(question, candidate_paths, next_relations_batched)

        search_list = heapq.nlargest(beam_width, search_list, key=lambda x: x.score)
        # Update candidate_paths and result_paths
        n_hop += 1
        candidate_paths = []
        for path in search_list:
            if path.prev_relations[-1] == END_REL:
                result_paths.append(path)
            else:
                candidate_paths.append(path)
    # Force early stop
    for prev_relations, score in candidate_paths:
        prev_relations = prev_relations + (END_REL,)
        result_paths.append(Path(prev_relations, score))
    result_paths = heapq.nlargest(beam_width, result_paths, key=lambda x: x.score)
    return result_paths


def retrieve_subgraph(sample: Dict[str, Any], beam_width):
    question = sample["question"]
    triplets = []
    for question_entity in sample["question_entities"]:
        path_score_list = beam_search_path(
            question, question_entity, beam_width, 2)
        n_nodes = 0
        threshold_ent_size = 1000
        for relations, _ in path_score_list:
            partial_nodes, partial_triples = path_to_subgraph(question_entity, relations)
            if len(partial_nodes) > 1000:
                continue
            n_nodes += len(partial_nodes)
            triplets.extend(partial_triples)

            if n_nodes > threshold_ent_size:
                break
    triplets = list(set(triplets))
    return triplets


def run(args):
    samples = list(srsly.read_jsonl(args.input))
    total = sum(1 for _ in srsly.read_jsonl(args.input))
    for sample in tqdm(samples, desc="retrieve:train", total=total):
        triplets = retrieve_subgraph(sample, args.beam_width)
        sample["triplets"] = triplets
    srsly.write_jsonl(args.output_path, samples)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparql-endpoint", type=str, default='http://localhost:3001/sparql')
    parser.add_argument("--input", type=str,
                        default="data/preprocess/ground100_webqsp.jsonl")
    parser.add_argument("--output-path", type=str,
                        default="artifacts/train_retrieved.jsonl")
    parser.add_argument("--scorer-model-path", type=str)
    parser.add_argument("--beam-width", type=int, default=10)
    args = parser.parse_args()

    kg = Freebase(args.sparql_endpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model_path)
    model = AutoModel.from_pretrained(args.scorer_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scorer = Scorer(args.scorer_model_path, device)
    run(args)
