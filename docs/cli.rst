Command Line Interface
=================================


srtk preprocess
-------------------

Create the training data to train a retriever from the grounded questions.

Usage: 

.. code-block:: bash

  srtk preprocess [-h] -i INPUT -o OUTPUT_DIR -e SPARQL_ENDPOINT -kg {wikidata,freebase}
    [--metric {jaccard,recall}] [--num-negative NUM_NEGATIVE]
    [--positive-threshold POSITIVE_THRESHOLD]


optional arguments:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The grounded questions file with question, question & answer entities
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory where the training train and the data (paths, scores, train) will be saved.
  -e SPARQL_ENDPOINT, --sparql-endpoint SPARQL_ENDPOINT
                        SPARQL endpoint URL for either Wikidata or Freebase (e.g., 'http://localhost:1234/api/endpoint/sparql' for default
                        local qEndpoint)
  -kg {wikidata,freebase}, --knowledge-graph {wikidata,freebase}
                        knowledge graph name, either wikidata or freebase
  --metric {jaccard,recall}
                        The metric used to score the paths. recall will usually result in a lager size of training dataset.
  --num-negative NUM_NEGATIVE
                        The number of negative relations to sample for each positive relation. (Default: 15)
  --positive-threshold POSITIVE_THRESHOLD
                        The threshold to determine whether a path is positive or negative. The default value is 0.5. If you want to use a
                        larger training dataset, you can set this value to a smaller value.


srtk train
--------------

Train a retriever model.

Usage: 

.. code-block:: bash

  srtk train [-h] [-i INPUT] [-o OUTPUT_DIR] [--model-name-or-path MODEL_NAME_OR_PATH]
    [--batch-size BATCH_SIZE] [--max-epochs MAX_EPOCHS] [--accelerator ACCELERATOR]
    [--fast-dev-run]



optional arguments:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        training data for the scorer. It should be a JSONL file with fields: query, positive, negatives
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        output model path. the model will be saved in the format of huggingface models, which can be uploaded to the
                        huggingface hub and shared with the community.
  --model-name-or-path MODEL_NAME_OR_PATH
                        pretrained model name or path. It is fully compatible with HuggingFace models. You can specify either a local path
                        where a model is saved, or an encoder model identifier from huggingface hub, e.g. bert-base-uncased.
  --batch-size BATCH_SIZE
                        batch size
  --max-epochs MAX_EPOCHS
                        max epochs
  --accelerator ACCELERATOR
                        accelerator, can be cpu, gpu, or tpu
  --fast-dev-run        fast dev run for debugging, only use 1 batch for training and validation


srtk evaluate
-----------------

``srtk evaluate`` evaluates the trained retriever model when the answer entities are known. In the case of multiple answers, any answer
present in the entities derived from the predicted path will be considered as correct.


srtk retrieve
-----------------

Usage:

.. code-block:: bash

  srtk retrieve [-h] -i INPUT -o OUTPUT [-e SPARQL_ENDPOINT] -kg {freebase,wikidata}
    --scorer-model-path SCORER_MODEL_PATH [--beam-width BEAM_WIDTH]
    [--max-depth MAX_DEPTH] [--save-recall] [--include-qualifiers]


Optional arguments:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input jsonl file path containing questions and grounded entities.
  -o OUTPUT, --output OUTPUT
                        output file path for storing retrieved triplets.
  -e SPARQL_ENDPOINT, --sparql-endpoint SPARQL_ENDPOINT
                        SPARQL endpoint for Wikidata or Freebase services.
  -kg {freebase,wikidata}, --knowledge-graph {freebase,wikidata}
                        choose the knowledge graph: currently supports ``freebase`` and ``wikidata``.
  --scorer-model-path SCORER_MODEL_PATH
                        Path to the scorer model, containing both the saved model and its tokenizer in the Huggingface models format. Such a
                        model is saved automatically when using the ``srtk train`` command. Alternatively, provide a pre-trained model name
                        from the Hugging Face model hub. In practice it supports any Huggingface transformers encoder model, though models that
                        do not use [CLS] tokens may require modifications on similarity function.
  --beam-width BEAM_WIDTH
                        beam width for beam search (default: 10).
  --max-depth MAX_DEPTH
                        maximum depth for beam search (default: 2).
  --save-recall         save recall information for answer entities in retrieved triplets. Requires `answer_entities` field in the input jsonl.
  --include-qualifiers  Include qualifiers from the retrieved triplets. Qualifiers are informations represented in non-entity form, like date,
                        count etc. This is only relevant for Wikidata.


srtk visualize
------------------

Visualize the graph (represented as a set of triplets) using pyvis.


Usage:

.. code-block:: bash

  srtk visualize [-h] -i INPUT -o OUTPUT_DIR [-e SPARQL_ENDPOINT]
    [-kg {wikidata,freebase}] [--max-output MAX_OUTPUT]


optional arguments:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The input subgraph file path.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory path.
  -e SPARQL_ENDPOINT, --sparql-endpoint SPARQL_ENDPOINT
                        SPARQL endpoint for Wikidata or Freebase services. In this step, it is used to get the labels of entities. (Default:
                        http://localhost:1234/api/endpoint/sparql)
  -kg {wikidata,freebase}, --knowledge-graph {wikidata,freebase}
                        The knowledge graph type to use. (Default: wikidata)
  --max-output MAX_OUTPUT
                        The maximum number of graphs to output. This is useful for debugging. (Default: 1000)