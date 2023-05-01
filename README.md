# Subgraph Retrieval Toolkit

[![PyPi](https://img.shields.io/pypi/v/srtk)](https://pypi.org/project/srtk/)
[![Documentation Status](https://readthedocs.org/projects/srtk/badge/?version=latest)](https://srtk.readthedocs.io/en/latest/?badge=latest)
[![PytestStatus](https://github.com/happen2me/subgraph-retrieval-wikidata/actions/workflows/pytest.yml/badge.svg)](https://github.com/happen2me/subgraph-retrieval-toolkit/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Retrieve subgraphs on Wikidata. The method is based on this [retrieval work](https://github.com/RUCKBReasoning/SubgraphRetrievalKBQA) for Freebase.

## Prerequisite

### Install SRTK
```bash
pip install srtk
```

### Wikidata

#### Deploy a Wikidata endpoint locally
We use [qEndpoint](https://github.com/the-qa-company/qEndpoint) to spin up a Wikidata endpoint that
contains a [Wikidata Truthy](https://www.wikidata.org/wiki/Wikidata:Database_download#RDF_dumps) dump.

- Download

    ```bash
    sudo docker run -p 1234:1234 --name qendpoint-wikidata qacompany/qendpoint-wikidata
    ```

- Run

    ```bash
    sudo docker start  qendpoint-wikidata
    ```
- Add Wikidata prefixes support

    ```bash
    wget https://raw.githubusercontent.com/the-qa-company/qEndpoint/master/wikibase/prefixes.sparql
    sudo docker cp prefixes.sparql qendpoint-wikidata:/app/qendpoint && rm prefixes.sparql
    ```

Alternatively, you can also use an [online Wikidata endpoint](https://query.wikidata.org), e.g. `https://query.wikidata.org/sparql`


#### Deploy a REL endpoint for entity linking (only necessary for end-to-end inference)

Please refer to this tutorial for REL endpoint deployment: [End-to-End Entity Linking](https://rel.readthedocs.io/en/latest/tutorials/e2e_entity_linking/)

### Freebase

#### Deploy a Freebase endpoint locally

Please refer to [dki-lab/Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) for the setup.

```bash
# Download setup script
git clone https://github.com/dki-lab/Freebase-Setup.git && cd Freebase-Setup
# Download virtuoso binary
wget https://kumisystems.dl.sourceforge.net/project/virtuoso/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
tar -zxvf virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz && rm virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
# Replace the virtuoso path in virtuoso.py
sed -i 's/\/home\/dki_lab\/tools\/virtuoso\/virtuoso-opensource/\.\/virtuoso-opensource/g' virtuoso.py
# Download Freebase dump
wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
unzip virtuoso_db.zip && rm virtuoso_db.zip
# Start virtuoso
python3 virtuoso.py start 3001 -d virtuoso_db
```


## Retrieve subgraphs with a trained scorer
```bash
srtk retrieve --sparql-endpoint WIKIDATA_ENDPOINT \
    -kg wikidata
    --scorer-model-path path/to/scorer \
    --input data/ground.jsonl \
    --output-path data/subgraph.jsonl \
    --beam-width 10
```

The `scorer-model-path` argument can be any huggingface pretrained encoder model. If it is a local
path, please ensure the tokenizer is also saved along with the model.

## Visualize retrieved subgraph
```bash
srtk visualize --sparql-endpoint WIKIDATA_ENDPOINT \
    --knowledge-graph wikidata \
    --input data/subgraph.jsonl \
    --output-dir ./htmls/
```

## Train a scorer
A scorer is the model used to navigate the expanding path. At each expanding step, relations scored higher with scorer are picked as relations for the next hop.

The score is based on the embedding similarity of the to-be-expanded relation with the query (question + previous expanding path).

The model is trained in a distant supervised learning fashion. Given the question entities and the answer entities, the model uses the shortest paths along them as the supervision signal.

### Preprocess a dataset
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
    srtk preprocess --sparql-endpoint WIKIDATA_ENDPOINT \
        -kg wikidata \
        --input-file data/grounded.jsonl \
        --output-dir data/retrieved --metric jaccard
    ```

    Under the hood, it does four things:

    1. Find the shortest paths between the question entities and the answer entities.
    2. Score the searched paths with Jaccard scores with the answers.
    3. Negative sampling. At each expanding step, the negative samples are those false relations connected to the tracked entities.
    4. Generate training dataset as a jsonl file.



### Train a sentence encoder

The scorer should be initialized from a pretrained encoder model from huggingface hub. Here I used `intfloat/e5-small`, which is a checkpoint of the BERT model.

```bash
srtk train --data-file data/train.jsonl \
    --model-name-or-path intfloat/e5-small \
    --save-model-path artifacts/scorer
```
