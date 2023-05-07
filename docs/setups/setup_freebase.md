# Setup Freebase

## Entity Linking

Please refer to the [Entity
Linking](https://github.com/WDPS-Team/Large-Scale-Entity-Linking#25-entity-linking)
section of the repository
[WDPS-Team/Large-Scale-Entity-Linking](https://github.com/WDPS-Team/Large-Scale-Entity-Linking).

## SPARQL Endpoint

Here we show how to use virtuoso to setup a Freebase endpoint locally.
Please refer to
[dki-lab/Freebase-Setup](https://github.com/dki-lab/Freebase-Setup)
for further details.

1. Download the setup script

   ```bash
   mkdir -p resources/freebase
   git clone https://github.com/dki-lab/Freebase-Setup.git && cd Freebase-Setup
   ```

2. Download virtuoso binary

   ```bash
   wget https://kumisystems.dl.sourceforge.net/project/virtuoso/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
   tar -zxvf virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz && rm virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
   ```

3. Replace the virtuoso binary path in
   [virtuoso.py](http://virtuoso.py/) to the one that you just
   downloaded in `virtuoso.py`

   ```bash
   sed -i 's/\/home\/dki_lab\/tools\/virtuoso\/virtuoso-opensource/\.\/virtuoso-opensource/g' virtuoso.py
   ```

4. Download Freebase dump

   ```bash
   wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip
   unzip virtuoso_db.zip && rm virtuoso_db.zip
   ```

5. Start virtuoso server on port `3001`, using the Freebase dump

   ```bash
   python [virtuoso.py](http://virtuoso.py/) start 3001 -d virtuoso_db
   ```
