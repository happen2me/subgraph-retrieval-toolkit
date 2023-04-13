# Subgraph Retrieval Toolkit

[![Tests](https://github.com/happen2me/subgraph-retrieval-wikidata/actions/workflows/pytest.yml/badge.svg)]()

Retrieve subgraphs on Wikidata. The method is based on this [retrieval work](https://github.com/RUCKBReasoning/SubgraphRetrievalKBQA) for Freebase.

## Prerequisite

### Install dependencies
```bash
pip install requirements.txt
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


Alternatively, you can also use an [online Wikidata endpoint](https://query.wikidata.org), e.g. `https://query.wikidata.org/sparql`


#### Deploy a REL endpoint for entity linking (only necessary for end-to-end inference)

Please refer to this tutorial for REL endpoint deployment: [End-to-End Entity Linking](https://rel.readthedocs.io/en/latest/tutorials/e2e_entity_linking/)

### Freebase

#### Deploy a Freebase endpoint locally

Please refer to [dki-lab/Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) for the setup.

```bash
# Download Freebase dump
wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
unzip virtuoso_db.zip && rm virtuoso_db.zip
# Download virtuoso binary
wget https://kumisystems.dl.sourceforge.net/project/virtuoso/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
tar -zxvf virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz && rm virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
# Replace the virtuoso path in virtuoso.py
sed -i 's/\/home\/dki_lab\/tools\/virtuoso\/virtuoso-opensource/\.\/virtuoso-opensource/g' virtuoso.py
# Start virtuoso
python3 virtuoso.py start 3001 -d virtuoso_db
```


## Retrieve subgraphs with a trained scorer
```bash
python retrieve.py --scorer-model-path path/to/scorer --input data/ground.jsonl \
    --output-path data/subgraph.jsonl --beam-width 10
```

The `scorer-model-path` argument can be any huggingface pretrained encoder model. If it is a local
path, please ensure the tokenizer is also saved along with the model.

## Visualize retrieved subgraph
```bash
python visualize.py --sparql-endpoint WIKIDATA_ENDPOINT \
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
2. Search paths between the question and the answer entities.

    ```bash
    python preprocess/search_path.py --sparql-endpoint WIKIDATA_ENDPOINT \
        --knowledge-graph wikidata \
        --ground-path data/grounded.jsonl \
        --output-path data/paths.jsonl \
        --remove-sample-without-path
    ```
3. Score the paths.

    The paths are scored with their Jaccard index with the answer. If the deduced entities is a closer set with the ground-truth answer entities, it will be assigned a higher score.
    
    ```bash
    python preprocess/score_path.py --sparql-endpoint WIKIDATA_ENDPOINT \
        --knowledge-graph wikidata \
        --paths-file data/paths.jsonl \
        --output-path data/paths_scored.jsonl
    ```
4. Negative sampling.
    
    At each expanding step, the negative samples are those false relations connected to the tracked entities. This step outputs the dataset for training as `train.jsonl`
    
    ```bash
    python preprocess/negative_sampling.py \
        --scored-path-file data/paths_scored.jsonl \
        --output-file data/train.jsonl \
        --sparql-endpoint WIKIDATA_ENDPOINT
    ```

### Train a sentence encoder

The scorer should be initialized from a pretrained encoder model from huggingface hub. Here I used `intfloat/e5-small`, which is a checkpoint of the BERT model.

```bash
python train.py --data-file data/train.jsonl \
    --model-name-or-path intfloat/e5-small \
    --save-model-path artifacts/scorer
```
