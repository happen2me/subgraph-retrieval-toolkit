# Setup Wikidata

## Entity Linking

Wikidata entities must first be identified in the text to retrieve
related subgraphs. You can use any available entity linking algorithms
for Wikidata. Just to name a few:

- [OpenTapioca](https://github.com/opentapioca/opentapioca)
- [ELQ](https://research.facebook.com/publications/efficient-one-pass-end-to-end-entity-linking-for-questions/)
- [TAGME](https://tagme.d4science.org/tagme/)
- [REL](https://github.com/informagi/REL)

Among them, REL (Radboud Entity Linker) can be easily set up locally for
offline inference while achieves near-SOTA performance. We show how to
use REL for entity linking on Wikidata.

### Setup Entity Linker REL

1. Install REL using pip

```bash
pip install radboud-el
```

1. Download necessary files (2019 dump)

   ```bash
   # Place them under resources/rel
   mkdir -p resources/rel && cd resources/rel
   # Download generic files
   wget http://gem.cs.ru.nl/generic.tar.gz
   # Download Wikipedia corpus (2019)
   wget http://gem.cs.ru.nl/wiki_2019.tar.gz
   # Download entity disambiguation model (2019)
   wget http://gem.cs.ru.nl/ed-wiki-2019.tar.gz

   # Unzip files
   tar -zxvf generic.tar.gz && rm generic.tar.gz
   tar -zxvf wiki_2019.tar.gz && rm iki_2019.tar.gz
   tar -zxvf ed-wiki-2019.tar.gz && rm ed-wiki-2019.tar.gz
   ```

   The unzipped folder structure should look like this. If not, please
   adjust accordingly.

   ```bash
   resources/rel
   ├── generic
   └─── wiki_2019
   |   ├── basic_data
   |      └── anchor_files
   |   └── generated
   ```

   Please refer to [REL’s
   documentation](https://rel.readthedocs.io/en/latest/) for further
   details.

### Setup Wikimapper

REL links text spans to Wikipedia article titles. We then need
[Wikimapper](https://github.com/jcklie/wikimapper) to further map
them to Wikidata IDs.

1. Install Wikimapper using pip

   ```bash
   pip install wikimapper
   ```

2. Prepare Wikimapper database

   - You can create your own database index. Please check [create
     your own
     index](https://github.com/jcklie/wikimapper#create-your-own-index).

   - You can download the precomputed indices from Wikimapper’s
     author (2019’s dump)

     ```bash
     mkdir resources/wikimapper && cd resources/wikimapper
     wget https://public.ukp.informatik.tu-darmstadt.de/wikimapper/index_enwiki-20190420.db
     ```

   - Alternatively, you can download the index computed by
     ourselves. They are newer (2023 Feb), and come with cased and
     uncased variant.

     ```bash
     mkdir resources/wikimapper && cd resources/wikimapper
     # They are hosted on google drive. gdown is a convenient gdrive download helper
     pip install gdown
     # index_enwiki-latest-cased.db
     gdown 1yMdzP4inW9CW5YbRZYVvsZYANHAERipL
     # index_enwiki-latest-uncased.db
     gdown 1hbfaaotNrWP3ecqk8B1Wnhf1ARZRakb9
     ```

## SPARQL Endpoint



```{seealso}
If you have no root access, you can also setup the qEenpoint rootlessly.

   ```{toctree}
   :maxdepth: 1

   setup_qendpoint_rootless.md
   ```




We use [qEndpoint](https://github.com/the-qa-company/qEndpoint) to
spin up a Wikidata endpoint that contains a [Wikidata
Truthy](https://www.wikidata.org/wiki/Wikidata:Database_download#RDF_dumps)
dump. If you have not installed docker yet, please check [Get
Docker](https://docs.docker.com/get-docker/).

1. Download

   ```bash
   sudo docker run -p 1234:1234 --name qendpoint-wikidata qacompany/qendpoint-wikidata
   ```

2. Run

   ```bash
   sudo docker start  qendpoint-wikidata
   ```

3. Add Wikidata prefixes support. With this, you can leave out Wikidata
   prefixes every time you send queries to the endpoint.

   ```bash
   wget https://raw.githubusercontent.com/the-qa-company/qEndpoint/master/wikibase/prefixes.sparql
   sudo docker cp prefixes.sparql qendpoint-wikidata:/app/qendpoint && rm prefixes.sparql
   ```
