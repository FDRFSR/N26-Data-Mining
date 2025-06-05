# N26 Bank Personal Data Mining
I created this simple bot to log into your private N26 online banking account and harvest all your transaction data because I wasn't satisfied with the dedicated downloadable .csv from the N26 website.

I hope you'll enjoy it.

## Installation
- clone the repository
- create and activate a virtual environment in the cloned directory (for reference [venv](https://towardsdatascience.com/virtual-environments-104c62d48c54))
```bash
$ python3 -m venv venv/
$ source venv/bin/activate
```
- install requirements
```bash
$ pip install -r requirements.txt
```
- change the inputs_example.py file accordingly and rename it to inputs.py

## Usage

### Modalità Command Line
```bash
$ python3 main.py
```
than accept the 2AF request within 30 seconds. It will open a chrome driver browser and start gathering data in a file called N26_Data.csv.

### Modalità GUI (Interfaccia Grafica)
Per utilizzare l'interfaccia grafica avanzata:
```bash
$ python3 gui.py
```

La GUI offre funzionalità avanzate tra cui:
- **Dashboard Riepilogativa**: Visualizzazione immediata di saldo, spese mensili e transazioni recenti
- **Filtri Avanzati**: Filtro per data, beneficiario, categoria e importo
- **Grafici Interattivi**: Grafici a torta per categorie, grafici a barre per spese mensili e beneficiari principali
- **Esportazione Multipla**: Esportazione in CSV, Excel, JSON, PDF e PNG
- **Automazioni**: Invio report via email, notifiche Telegram, esportazioni programmate
- **Ricerca Full-Text**: Ricerca avanzata nelle descrizioni delle transazioni
- **Analisi Predittiva**: Previsioni basate sui pattern di spesa
- **Stampa Diretta**: Stampa report direttamente dall'applicazione

## Configurazione avanzata

Puoi configurare i nomi dei file CSV tramite variabili d'ambiente:

```bash
export CSV_TARGET_NAME="N26_History_With_Tags.csv"
export LABEL_CSV_NAME="labels.csv"
```

## Logging

Il programma ora utilizza il modulo logging di Python. Puoi personalizzare il livello di logging modificando la configurazione in `main.py`.

### Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors

- [@andreacannizzo](https://www.github.com/andreacannizzo) a.k.a. Andrea C.

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)