Command Line Interface
=================================

SRTK command line interfaces provides an easy but powerful way to retrieve subgraphs
as well as to streamline the lifecycle of subgraph retrieval.

srtk preprocess
-------------------

Create the training data to train a retriever from the grounded questions.

Usage: 

.. code-block:: bash

 srtk preprocess [-h] -i INPUT -o OUTPUT [--intermediate-dir INTERMEDIATE_DIR]
                 -e SPARQL_ENDPOINT -kg {wikidata,freebase} [--search-path]
                 [--metric {jaccard,recall}] [--num-negative NUM_NEGATIVE]
                 [--positive-threshold POSITIVE_THRESHOLD]

Options:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The grounded questions file with question, question & answer entities
  -o OUTPUT, --output OUTPUT
                        The output path where the final training data will be saved.
  --intermediate-dir INTERMEDIATE_DIR
                        The directory to save intermediate files. If not specified, the intermediate files will be saved in the same directory as the
                        output file, with the name paths.jsonl and scores.jsonl
  -e SPARQL_ENDPOINT, --sparql-endpoint SPARQL_ENDPOINT
                        SPARQL endpoint URL for either Wikidata or Freebase (e.g., 'http://localhost:1234/api/endpoint/sparql' for default local
                        qEndpoint)
  -kg {wikidata,freebase}, --knowledge-graph {wikidata,freebase}
                        knowledge graph name, either wikidata or freebase
  --search-path         Whether to search paths between question and answer entities. If not specified, paths and scores fields must present in the
                        input file. You **have to** specify this for weakly supervised learning. (default: False)
  --metric {jaccard,recall}
                        The metric used to score the paths. recall will usually result in a lager size of training dataset. (default: jaccard))
  --num-negative NUM_NEGATIVE
                        The number of negative relations to sample for each positive relation. (default: 15)
  --positive-threshold POSITIVE_THRESHOLD
                        The threshold to determine whether a path is positive or negative. If you want to use a larger training dataset, you can set
                        this value to a smaller value. (default: 0.5)


srtk train
--------------

Train a retriever model.


Usage: 

.. code-block:: bash

  srtk train [-h] -t TRAIN_DATASET [-v VALIDATION_DATASET] [-o OUTPUT_DIR]
             [--model-name-or-path MODEL_NAME_OR_PATH] [-lr LEARNING_RATE]
             [--batch-size BATCH_SIZE] [--max-epochs MAX_EPOCHS] [--accelerator ACCELERATOR]
             [--fast-dev-run]

Options:

  -h, --help            show this help message and exit
  -t TRAIN_DATASET, --train-dataset TRAIN_DATASET
                        path to the training dataset. It should be a JSONL file with fields:
                        query, positive, negatives
  -v VALIDATION_DATASET, --validation-dataset VALIDATION_DATASET
                        path to the validation dataset. If not provided, 5 percent of the training
                        data will be used as validation data. (default: None)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        output model path. the model will be saved in the format of huggingface
                        models, which can be uploaded to the huggingface hub and shared with the
                        community. (default: artifacts/scorer)
  --model-name-or-path MODEL_NAME_OR_PATH
                        pretrained model name or path. It is fully compatible with HuggingFace
                        models. You can specify either a local path where a model is saved, or an
                        encoder model identifier from huggingface hub. (default:
                        intfloat/e5-small)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate (default: 5e-5)
  --batch-size BATCH_SIZE
                        batch size (default: 16)
  --max-epochs MAX_EPOCHS
                        max epochs (default: 10)
  --accelerator ACCELERATOR
                        accelerator, can be cpu, gpu, or tpu (default: gpu)
  --fast-dev-run        fast dev run for debugging, only use 1 batch for training and validation


srtk link
-------------------

Entity linking. The input is a jsonl file. The field of interest is specified by the argument --ground-on. The output is a jsonl file, each line is a dict with keys: id,
question_entities, spans, entity_names. Currently, only Wikidata is supported out of the box.

Usage:

.. code-block:: bash

  srtk link [-h] [-i INPUT] [-o OUTPUT] [-e EL_ENDPOINT] [-kg {wikidata}]
            [--wikimapper-db WIKIMAPPER_DB] [--ground-on GROUND_ON]


Options:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file path, in which the question is stored
  -o OUTPUT, --output OUTPUT
                        Output file path, in which the entity linking result is stored
  -e EL_ENDPOINT, --el-endpoint EL_ENDPOINT
                        REL endpoint
  -kg {wikidata}, --knowledge-graph {wikidata}
                        Knowledge graph to link to, only wikidata is supported now
  --wikimapper-db WIKIMAPPER_DB
                        Wikimapper database path
  --ground-on GROUND_ON
                        The key to ground on, the corresponding text will be sent to the REL endpoint for entity linking

srtk retrieve
-----------------

Retrieve subgraphs with a trained model on a dataset that entities are linked. This command can
also be used to evaluate a trained retriever when the answer entities are known. Provide a JSON
file as input, where each JSON object must contain at least the 'question' and 'question_entities'
fields. When ``--evaluate`` is set, the input JSON file must also contain the 'answer_entities'
field. The output JSONL file will include an added 'triplets' field, based on the input JSONL
file. This field consists of a list of triplets, with each triplet representing a (head, relation,
tail) tuple. When ``--evaluate`` is set, a metric file will also be saved to the same directory as
the output JSONL file.



Usage:

.. code-block:: bash

  srtk retrieve [-h] -i INPUT -o OUTPUT [-e SPARQL_ENDPOINT] -kg {freebase,wikidata}
                --scorer-model-path SCORER_MODEL_PATH [--beam-width BEAM_WIDTH]
                [--max-depth MAX_DEPTH] [--evaluate] [--include-qualifiers]


Options:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input jsonl file. it should contain at least ``question`` and
                        ``question_entities`` fields.
  -o OUTPUT, --output OUTPUT
                        output file path for storing retrieved triplets.
  -e SPARQL_ENDPOINT, --sparql-endpoint SPARQL_ENDPOINT
                        SPARQL endpoint for Wikidata or Freebase services.
  -kg {freebase,wikidata}, --knowledge-graph {freebase,wikidata}
                        choose the knowledge graph: currently supports ``freebase`` and
                        ``wikidata``.
  --scorer-model-path SCORER_MODEL_PATH
                        Path to the scorer model, containing both the saved model and its
                        tokenizer in the Huggingface models format. Such a model is saved
                        automatically when using the ``srtk train`` command. Alternatively,
                        provide a pre-trained model name from the Hugging Face model hub. In
                        practice it supports any Huggingface transformers encoder model, though
                        models that do not use [CLS] tokens may require modifications on
                        similarity function.
  --beam-width BEAM_WIDTH
                        beam width for beam search (default: 10).
  --max-depth MAX_DEPTH
                        maximum depth for beam search (default: 2).
  --evaluate            Evaluate the retriever model. When the answer entities are known, the
                        recall can be evluated as the number of samples that any of the answer
                        entities are retrieved in the subgraph by the number of all samples. This
                        equires `answer_entities` field in the input jsonl.
  --include-qualifiers  Include qualifiers from the retrieved triplets. Qualifiers are
                        informations represented in non-entity form, like date, count etc. This is
                        only relevant for Wikidata.


srtk visualize
------------------

Visualize the graph (represented as a set of triplets) using pyvis.


Usage:

.. code-block:: bash

  srtk visualize [-h] -i INPUT -o OUTPUT_DIR [-e SPARQL_ENDPOINT] [-kg {wikidata,freebase}]
                 [--max-output MAX_OUTPUT]



Options:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The input subgraph file path.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory path.
  -e SPARQL_ENDPOINT, --sparql-endpoint SPARQL_ENDPOINT
                        SPARQL endpoint for Wikidata or Freebase services. In this step, it is
                        used to get the labels of entities. (Default:
                        http://localhost:1234/api/endpoint/sparql)
  -kg {wikidata,freebase}, --knowledge-graph {wikidata,freebase}
                        The knowledge graph type to use. (Default: wikidata)
  --max-output MAX_OUTPUT
                        The maximum number of graphs to output. This is useful for debugging.
                        (Default: 1000)
