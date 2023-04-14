import argparse
import heapq
from collections import namedtuple
from typing import List, Any, Dict

import srsly
import torch
from tqdm import tqdm

from knowledge_graph import Freebase, Wikidata, KnowledgeGraphBase
from scorer import Scorer


END_REL = 'END OF HOP'


# Path collects the information at each traversal step
# - prev_relations stores the relations that have been traversed
# - score stores the score of the relation path
Path = namedtuple('Path', ['prev_relations', 'score'], defaults=[(), 0])


class KnowledgeGraphTraverser:
    '''KnowledgeGraphTraverser is a helper class that traverses a knowledge graph'''
    def __init__(self, kg: KnowledgeGraphBase):
        self.kg = kg

    def retrive_subgraph(self, entity, path):
        '''Retrive subgraph entities and triplets by traversing from an entity following
        a relation path hop by hop.

        Args:
            entity: the source node
            path: a list of relations

        Returns:
            entities: a list of entities
            triplets: a list of triplets
        '''
        entities, triplets = set(), set()
        tracked_entities = set((entity,))
        for relation in path:
            next_hops = set()
            if relation == END_REL:
                continue
            for e in tracked_entities:
                leaves = set(self.kg.deduce_leaves(e, (relation,)))
                next_hops |= leaves
                triplets |= {(e, relation, leaf) for leaf in leaves}
            entities |= next_hops
            tracked_entities = next_hops
        return list(entities), list(triplets)

    def deduce_leaves(self, entity, path):
        """Deduce leaves from an entity following a path hop by hop.

        Args:
            entity: the source node
            path: a list of relations

        Returns:
            leaves: a list of leaves that are n-hop away from the source node,
                where n is the length of the path
        """
        leaves = set((entity,))
        for relation in path:
            if relation == END_REL:
                continue
            leaves = set().union(*(self.kg.deduce_leaves(leaf, (relation,)) for leaf in leaves))
        return leaves

    def deduce_leaf_relations(self, entity, path):
        """Deduce leaf relations from an entity following a path hop by hop.

        Args:
            entity: the source node
            path: a list of relations

        Returns:
            relations: a list of relations that are n-hop away from the source node,
                where n is the length of the path
        """
        leaves = self.deduce_leaves(entity, path)
        relations = set().union(*(self.kg.get_neighbor_relations(leaf) for leaf in leaves))
        # Special filter relation for freebase
        if self.kg.name == 'freebase':
            relations = [r for r in relations if r.split('.')[0] not in ['kg', 'common']]
        return tuple(relations)


class Retriever:
    '''Retriever is a class that retrieves subgraphs from a knowledge graph'''
    def __init__(self, kg: KnowledgeGraphBase, scorer: Scorer, beam_width: int, max_depth: int):
        self.kgh = KnowledgeGraphTraverser(kg)
        self.scorer = scorer
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.num_entity_threshold = 1000

    def retrieve_subgraph_triplets(self, sample: Dict[str, Any]):
        """Retrieve triplets as subgraphs from paths.

        Args:
            sample: a sample from the dataset, which contains at least the following fields:
                question: a string
                question_entities: a list of entities

        Returns:
            triplets: a list of triplets
        """
        question = sample['question']
        triplets = []
        for question_entity in sample['question_entities']:
            path_score_list = self.beam_search_path(question, question_entity)
            n_nodes = 0
            for relations, _ in path_score_list:
                partial_nodes, partial_triples = self.kgh.retrive_subgraph(question_entity, relations)
                if len(partial_nodes) > self.num_entity_threshold:
                    continue
                n_nodes += len(partial_nodes)
                triplets.extend(partial_triples)

                if n_nodes > self.num_entity_threshold:
                    break
        triplets = list(set(triplets))
        return triplets

    def beam_search_path(self, question: str, question_entity: str):
        '''This function reimplement RUC's paper's solution. In the search process, only the history
        paths are recorded; each new relation is looked up via looking up the end relations from the
        question entities following a history path.

        Args:
            question (str)
            question_entity (str): a grounded question entity

        Returns:
            path_score_list (list[Path]): a list of (path, score) tuples
        '''
        candidate_paths = [Path()]  # path and its score
        result_paths = []
        depth = 0

        while candidate_paths and len(result_paths) < self.beam_width and depth < self.max_depth:
            next_relations_batched = []
            for path in candidate_paths:
                prev_relations = path.prev_relations
                next_relations = self.kgh.deduce_leaf_relations(question_entity, prev_relations)
                next_relations = next_relations + (END_REL,)
                next_relations_batched.append(next_relations)

            tracked_paths = self.expand_and_score_paths(question, candidate_paths, next_relations_batched)
            tracked_paths = heapq.nlargest(self.beam_width, tracked_paths, key=lambda x: x.score)
            depth += 1
            # Update candidate_paths
            candidate_paths = []
            for path in tracked_paths:
                if  path.prev_relations and path.prev_relations[-1] == END_REL:
                    result_paths.append(path)
                else:
                    candidate_paths.append(path)
        # Merge not-yet-ended paths into the result paths 
        candidate_paths = [Path(prev_relations + (END_REL,), score) for prev_relations, score in candidate_paths]
        result_paths = heapq.nlargest(self.beam_width, result_paths + candidate_paths, key=lambda x: x.score)
        return result_paths

    def expand_and_score_paths(self, question: str, paths: List[Path], relations_batched: List[List[str]]) -> List[Path]:
        '''Expand the paths by one hop and score them by comparing the embedding similarity between
        the query (question + prev_relations) and the next relation.

        Args:
            question (str)
            paths (list[Path]): a list of current paths
            relations_batched (list[list[str]]): a list of next relations for each path

        Returns:
            scored_paths (list[Path]): a list of newly expanded and scored paths
        '''
        scored_paths = []
        score_matrix = []
        for path, next_relations in zip(paths, relations_batched):
            scores = self.scorer.batch_score(question, tuple(path.prev_relations), tuple(next_relations))
            score_matrix.append(scores)

        for i, (path, next_relations) in enumerate(zip(paths, relations_batched)):
            for j, relation in enumerate(next_relations):
                new_prev_relations = path.prev_relations + (relation,)
                score = float(score_matrix[i][j]) + path.score
                scored_paths.append(Path(new_prev_relations, score))
        return scored_paths


def main(args):
    if args.knowledge_graph == 'freebase':
        kg = Freebase(args.sparql_endpoint)
    else:
        kg = Wikidata(args.sparql_endpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scorer = Scorer(args.scorer_model_path, device)
    retriever = Retriever(kg, scorer, args.beam_width, args.max_depth)
    samples = list(srsly.read_jsonl(args.input))
    total = sum(1 for _ in srsly.read_jsonl(args.input))
    for sample in tqdm(samples, desc='Retrieving subgraphs', total=total):
        triplets = retriever.retrieve_subgraph_triplets(sample)
        sample['triplets'] = triplets
    srsly.write_jsonl(args.output_path, samples)
    print(f'Saved to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparql-endpoint', type=str, help='endpoint of the wikidata or freebase sparql service')
    parser.add_argument('-kg', '--knowledge-graph', type=str, required=True, choices=('freebase', 'wikidata'))
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input jsonl with question and grounded entities')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='path to the outputs with retrived triplets')
    parser.add_argument('--scorer-model-path', type=str, required=True, help='path to the scorer model')
    parser.add_argument('--beam-width', type=int, default=10, help='beam width for beam search')
    parser.add_argument('--max-depth', type=int, default=2, help='max depth for beam search')
    args = parser.parse_args()
    if not args.sparql_endpoint:
        if args.knowledge_graph == 'freebase':
            args.sparql_endpoint = 'http://localhost:3001/sparql'
        else:
            args.sparql_endpoint = 'http://localhost:1234/api/endpoint/sparql'
        print(f'Using default sparql endpoint for {args.knowledge_graph}: {args.sparql_endpoint}')
    main(args)
