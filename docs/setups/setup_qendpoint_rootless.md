# Setup Wikidata qEndpoint Rootlessly

Running docker requires root privileges. To circumvent this, there are two solutions.

## Option1: Use Container in Rootless Mode

You can use rootless mode if you want to run docker without root privileges.

Note: for dependencies, you can download `uidmap` binary for your distribution.

## Option 2: Run qEndpoint in Native Mode

1. Create a directory for Wikidata
    
    ```
    mkdir -p ~/wikidata/qendpoint/hdt-store && cd ~/wikidata
    ```
    
2. Download Wikidata HDT file from a CDN and sparql prefix from the qEndpoint repository. The Wikidata truthy dump is around 45GB.
    
    ```
    curl -L -o qendpoint/hdt-store/index_dev.hdt <https://qanswer-svc4.univ-st-etienne.fr/wikidata_truthy.hdt>
    curl -L -o qendpoint/prefixes.sparql <https://raw.githubusercontent.com/the-qa-company/qEndpoint/master/wikibase/prefixes.sparql>
    
    ```
    
3. Download qEndpoint jar file from the latest release
    
    ```
    # Search for the latest release
    response=$(curl -s <https://api.github.com/repos/the-qa-company/qEndpoint/releases/latest>)
    download_url=$(echo "$response" | grep "browser_download_url.*qendpoint.jar" | cut -d : -f 2,3 | tr -d \\" |  sed 's/^ *//')
    curl -L -o qendpoint.jar "$download_url"
    
    ```
    
4. Run qEndpoint
    
    ```
    java -jar qendpoint.jar
    ```