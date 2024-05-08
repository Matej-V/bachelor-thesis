Velké jazykové modely pro generování kódu se zaměřením na vestavěné systémy
=====================================================

Popis
-----
* Riešiteľ: Matej Vadovič (xvadov01)
* Začiatok: 15.8.2023
* Ukončenie: 9.5.2024

## Abstrakt

Cieľom tejto práce bola adaptácia predtrénovaného jazykového modelu pre účely generovania kódu v oblasti vstavaných systémov. V práci je predstavená nová dátová sada pre ladenie modelov generovania kódu, ktorá obsahuje 50 tisíc dvojíc zdrojového kódu a komentárov zameraných na oblasť programovania vstavaných systémov. Táto sada je zložená zo zozbieraného zdrojového kódu z platformy GitHub. Na dátach nového korpusu boli ladené dva nové jazykové modely pre generovanie kódu založené na predtrénovaných modeloch s architektúrou transformer. Model MicroCoder je založený na modeli CodeLLaMA-Instruct 7B a pri jeho ladení bola využitá technika QLoRA pre minimalizáciu výpočtových nárokov ladenia. Druhý model, MicroCoderFIM, je založený na modeli StarCoderBase 1B a podporuje vyplňovanie kódu na základe okolia (fill-in-the-middle). Jednotlivé modely boli porovnávané na základe metrík BLEU, CodeBLEU, ChrF++ a  ROUGE-L. Model MicroCoderFIM dosahuje najlepšie výsledky adaptácie na novú úlohu, pričom zaznamenal viac ako 120% zlepšenie vo všetkých meraných metrikách. Váhy modelov [1], [2] spolu s novou dátovou sadou [3] sú voľne prístupné na verejnom úložisku.


## Štruktúra repozitára

- **README.md**
- **microcoder** - priečinok so skriptami k trénovaniu a vyhodnocovaniu modelov
  - **dataset**: Súbory pre vytvorenie dátovej sady
    - **dataset-analysis.ipynb**
    - **func-desc-code-crawler.ipynb**
    - **func-desc-code-dataset-builder.ipynb**
  - **evaluate-causal-modeling.py** - Skript pre vyhodnocovanie modelov na základe kauzálneho modelovania
  - **evaluate-chatgpt.ipynb** - Vyhodnotenie modelu GPT-3.5-TURBO
  - **evaluate-fim.py** - Skript pre vyhodnocovanie modelov na základe fill-in-the-middle úlohy
  - **merge-lora.ipynb** - Zlúčeie váh modelov po ladení s QLoRA
  - **microcoderfim-inference-test** - Skripty pre testvanie parametrov inferencie modelu MicroCoderFIM
    - **run-test.py**
    - **test-analysis.ipynb**
  - **passk-evaluation** - Skripty pre vyhodnocovanie modelu MicroCoderFIM na metrike pass@k
    - **evaluate-predictions.py**
    - **generate-predictions.py**
  - **train-microcoder.py** - Skript pre trénovanie modelu MicroCoder
  - **train-microcoderfim.py** - Skript pre trénovanie modelu MicroCoderFIM
  - **traintools** - Priečinok s pomocnými skriptami pre trénovanie modelov
    - **__init__.py**
    - **lib.py** - Pomocné funkcie pre trénovanie modelov
- **poster.pdf** - Plagát s výsledkami práce
- **thesis** - Zdrojové súbory bakalárskej práce
- **thesis.pdf** - Bakalárska práca vo formáte pdf
- **requirements.txt** - Zoznam závislostí projektu


## Ako spustiť skripty

Prvý krok je nastaviť premenné prostredia. Vzor je v súbore `.env.example`.

Pre spustenie skriptov je potrebné mať nainštalované závislosti z `requirements.txt`. Následne je možné spustiť skripty v priečinku `microcoder`.

V knižnici `traintolls.lib` sú definované použité nastavenia je jednotlivé behy trénovania a evaluácie modelov. Pre zmenu nastavení je možné upraviť hodnoty v tejto knižnici.

Pre trénovanie MicroCoder modelu je možné spustiť:
```bash
python train-microcoder.py MicroCoderTrain
```

Pre trénovanie MicroCoderFIM modelu je možné spustiť:
```bash
python train-microcoderfim.py MicroCoderFIMTrain
```

Pre vyhodnocovanie modelov je možné spustiť:
```bash
python evaluate-causal-modeling.py MicroCoderEval
python evaluate-fim.py MicroCoderFIMEval
```


## Preklad práce do pdf

Pre preklad práce do pdf je potrebné mať naištalovaný balík `pdflatex`, `slovak babel` a `inkscape` pre konvertovanie svg obrázkov do pdf. Následne je možný preklad pomocou príkazu `make` v priečinku `thesis`.


## Citácia
```
@BachelorsThesis{vadovic2024,
  author = {Matej Vadovič},
  title = {Velké jazykové modely pro generování kódu se zaměřením na vestavěné systémy},
  school = {Vysoké učení technické v Brně, Fakulta informačních technologií},
  year = {2024},
  type = {Bakalářská práce},
  address = {Brno},
  supervisor = {doc. RNDr. Pavel Smrž, Ph.D.},
}
```


[1]: https://huggingface.co/datasets/xvadov01/cpp_emb_nl2pl
[2]: https://huggingface.co/datasets/xvadov01/cpp_emb_nl2pl
[3]: https://huggingface.co/datasets/xvadov01/cpp_emb_nl2pl