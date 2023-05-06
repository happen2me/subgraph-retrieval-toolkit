# SRTK: Subgraph Retrieval Toolkit

[![PyPi](https://img.shields.io/pypi/v/srtk)](https://pypi.org/project/srtk/)
[![Documentation Status](https://readthedocs.org/projects/srtk/badge/?version=latest)](https://srtk.readthedocs.io/en/latest/?badge=latest)
[![PytestStatus](https://github.com/happen2me/subgraph-retrieval-wikidata/actions/workflows/pytest.yml/badge.svg)](https://github.com/happen2me/subgraph-retrieval-toolkit/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/622648166.svg)](https://zenodo.org/badge/latestdoi/622648166)


**SRTK** is a toolkit for semantic-relevant subgraph retrieval from large-scale knowledge graphs. It currently supports Wikidata, Freebase and DBPedia.

A minimum walkthrough of the retrieve process:

![retrieve example](examples/srtk_retrieve.svg)

<img width="400rem" src="https://i.imgur.com/jG7nZuo.png" alt="Visualized subgraph"/>

## Prerequisite

### Installations

```bash
pip install srtk
```

### Local Deployment of Knowledge Graphs

- [Setup Wikidata locally](https://srtk.readthedocs.io/en/latest/setup_wikidata.html)
- [Setup Freebase locally](https://srtk.readthedocs.io/en/latest/setup_freebase.html)

## Usage

There are mainly five subcommands of SRTK, which covers the whole pipeline of subgraph retrieval.

For retrieval:

- `srtk link`: Link entity mentions in texts to a knowledge graph. Currently Wikidata and DBPedia are supported out of the box.
- `srtk retrieve`: Retrieve semantic-relevant subgraphs from a knowledge graph with a trained retriever. It can also be used to evaluate a trained retriever.
- `srtk visualize`: Visualize retrieved subgraphs using a graph visualization tool.

For training a retriever:

- `srtk preprocess`: Preprocess a dataset for training a subgraph retrieval model.
- `srtk train`: Train a subgraph retrieval model on a preprocessed dataset.


Use `srtk [subcommand] --help` to see the detailed usage of each subcommand.

## Walkthrough

### Retrieve Subgraphs

#### Retrieve subgraphs with a trained scorer

```bash
srtk retrieve [-h] -i INPUT -o OUTPUT [-e SPARQL_ENDPOINT] -kg {freebase,wikidata}
              -m SCORER_MODEL_PATH [--beam-width BEAM_WIDTH] [--max-depth MAX_DEPTH]
              [--evaluate] [--include-qualifiers]
```

The `scorer-model-path` argument can be any huggingface pretrained encoder model. If it is a local
path, please ensure the tokenizer is also saved along with the model.

#### Visualize retrieved subgraph

```bash
srtk visualize [-h] -i INPUT -o OUTPUT_DIR [-e SPARQL_ENDPOINT]
               [-kg {wikidata,freebase}] [--max-output MAX_OUTPUT]
```

### Train a Retriever

A scorer is the model used to navigate the expanding path. At each expanding step, relations scored higher with scorer are picked as relations for the next hop.

The score is based on the embedding similarity of the to-be-expanded relation with the query (question + previous expanding path).

The model is trained in a distant supervised learning fashion. Given the question entities and the answer entities, the model uses the shortest paths along them as the supervision signal.

#### Preprocess a dataset

1. prepare training samples where question entities and answer entities are know.

    The training data should be saved in a jsonl file (e.g. `data/grounded.jsonl`). Each training sample should come with the following format:
    
    ```json
    {
      "id": "sample-id",
      "question": "Which universities did Barack Obama graduate from?",
      "question_entities": [
        "Q76"
      ],
      "answer_entities": [
        "Q49122",
        "Q1346110",
        "Q4569677"
      ]
    }
    ```
2. Preprocess the samples with `srtk preprocess` command.

    ```bash
    srtk preprocess [-h] -i INPUT -o OUTPUT [--intermediate-dir INTERMEDIATE_DIR]
                    -e SPARQL_ENDPOINT -kg {wikidata,freebase} [--search-path]
                    [--metric {jaccard,recall}] [--num-negative NUM_NEGATIVE]
                    [--positive-threshold POSITIVE_THRESHOLD]
    ```

    Under the hood, it does four things:

    1. Find the shortest paths between the question entities and the answer entities.
    2. Score the searched paths with Jaccard scores with the answers.
    3. Negative sampling. At each expanding step, the negative samples are those false relations connected to the tracked entities.
    4. Generate training dataset as a jsonl file.

#### Train a sentence encoder

The scorer should be initialized from a pretrained encoder model from huggingface hub. Here I used `intfloat/e5-small`, which is a checkpoint of the BERT model.

```bash
srtk train --data-file data/train.jsonl \
    --model-name-or-path intfloat/e5-small \
    --save-model-path artifacts/scorer
```

## Tutorials

- [End-to-end Subgraph Retrieval](https://github.com/happen2me/subgraph-retrieval-toolkit/blob/main/tutorials/2.end_to_end_subgraph_retrieval.ipynb)
- [Train a Retriever on Wikidata with Weak Supervision](https://github.com/happen2me/subgraph-retrieval-toolkit/blob/main/tutorials/3.weak_train_wikidata.ipynb)
- [Train a Retriever on Freebase with Weak Supervision](https://github.com/happen2me/subgraph-retrieval-toolkit/blob/main/tutorials/4.weak_train_freebase.ipynb)
- [Supervised Training with Wikidata Simple Questions](https://github.com/happen2me/subgraph-retrieval-toolkit/blob/main/tutorials/5.supervised_train_wikidata.ipynb)
- [Extend SRTK to other Knowledge Graphs](https://github.com/happen2me/subgraph-retrieval-toolkit/blob/main/tutorials/6.extend_to_new_kg.ipynb)

## License

This project is licensed under the terms of the MIT license.