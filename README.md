# subgraph-retrieval-wikidata
Retrieve subgraph on wikidata

## Spin up a Wikidata endpoint locally
We use [qEndpoint](https://github.com/the-qa-company/qEndpoint) to spin up a Wikidata endpoint that
contains a [Wikidata Truty](https://www.wikidata.org/wiki/Wikidata:Database_download#RDF_dumps) dump.

- Download
    ```bash
    sudo docker run -p 1234:1234 --name qendpoint-wikidata qacompany/qendpoint-wikidata
    ```

- Run
    ```bash
    sudo docker start  qendpoint-wikidata
    ```
