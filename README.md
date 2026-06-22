
# Wi-Fi Probe Request Clustering

Questo repository contiene una pipeline sperimentale per trasformare Probe Request Wi-Fi in feature numeriche e usarle per il clustering di dispositivi.

Il lavoro è pensato per valutare quanto le Information Elements dei frame 802.11 possano essere utili per distinguere dispositivi diversi, riducendo la dipendenza dal MAC address.

## Obiettivo

L'obiettivo principale è costruire una rappresentazione numerica delle Probe Request e usarla per:

- estrarre feature dai pacchetti Wi-Fi;
- costruire dataset tabellari;
- generare embedding tramite modelli encoder/autoencoder;
- applicare clustering sugli embedding;
- valutare la qualità dei cluster rispetto alle label disponibili.

## Pipeline

```text
PCAP / JSON
   ↓
estrazione Probe Request
   ↓
estrazione Information Elements
   ↓
feature engineering
   ↓
dataset numerico
   ↓
encoder / autoencoder
   ↓
embedding
   ↓
clustering
   ↓
valutazione
````

## Componenti principali

```text
src/
├── main.py                  # punto di ingresso principale
├── pcap_to_clustering.py     # pipeline da PCAP a clustering
├── ie_to_transformerIE.py    # preprocessing delle Information Elements
├── transformer_utils.py      # modelli e funzioni di training
└── dataset.py                # gestione dei dataset
```

## Dati

Il progetto può lavorare con:

* file `.pcap` contenenti traffico Wi-Fi;
* file `.json` di Probe Request già estratti

Le cartelle principali usate per i dati sono:

## Feature

Le feature sono ricavate principalmente dalle Information Elements delle Probe Request.

Tra le informazioni considerate ci sono:

* presenza o assenza delle IE;
* campi numerici;
* bitmap;
* capability flags;
* Supported Rates;
* HT/VHT capabilities;
* Extended Capabilities;
* Vendor Specific IE.

Il MAC address può essere mantenuto come metadato ma non usato come feature, per evitare leakage durante il clustering.

## Modelli

Il progetto usa modelli encoder/autoencoder per comprimere le feature originali in embedding più compatti.

Gli embedding vengono poi usati come input per algoritmi di clustering.

## Clustering e valutazione

Il clustering viene applicato sugli embedding prodotti dal modello.

Le metriche usate per la valutazione includono:

* ARI;
* NMI;
* Homogeneity;
* Completeness;
* V-measure.

Le label reali vengono usate solo per valutare i cluster, non per addestrare il modello in modo supervisionato.

## Stato del progetto

Il progetto è sperimentale e orientato alla tesi.

Le parti principali riguardano:

* preprocessing delle Probe Request;
* costruzione del dataset;
* apprendimento di embedding;
* clustering;
* analisi dei risultati.

```
```
