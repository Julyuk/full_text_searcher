# full_text_searcher
Full‑Text Search Console App

A console search engine over a provided corpus using:
- TF‑IDF with cosine similarity
- BM25 (classic, tuned)

Preprocessing: normalization, stopword removal, punctuation/URL/HTML/emoji cleanup, optional spell‑correction, optional WordNet synonyms. The app supports interactive custom queries as required by the assignment.

## What this app does
- Cleans and indexes your corpus (`id | display name | description`).
- Supports interactive search from the terminal.
- Evaluates retrieval quality with common IR metrics and saves results to `Results/`.
- Lets you toggle models and features (PRF, PMI synonyms, semantic re‑ranker) without changing code.

## Installation
Prereqs: Python 3.12+

Install packages:
```bash
pip3 install numpy pandas pyspellchecker nltk matplotlib seaborn
```

Optional (to enable WordNet synonyms):
- macOS certificates fix (python.org builds):
```bash
open "/Applications/Python 3.12/Install Certificates.command"
```
- Then download WordNet data:
```bash
python3 -m nltk.downloader wordnet omw-1.4
```


## Quick start
Run fast, open console immediately (no tuning):
```bash
python3 full_text_search_practice.py --no-tune --interactive-only --model bm25
```

In the console:
- Type your query (e.g., `wireless headphones`)
- Commands: `:model tfidf` | `:model bm25` | `:help` | `exit`

## Full evaluation run
```bash
python3 full_text_search_practice.py
```
Outputs in `Results/` with timestamp:
- `results_YYYYMMDD_HHMMSS.csv` – per‑query metrics
- `metrics_YYYYMMDD_HHMMSS.png` – bar chart of P@5, R@5, F1@5

Speed knobs:
- `--fast` – tiny grids + subset of queries for tuning
- `--no-tune` – skip tuning and use defaults
- `--eval-limit N` – use only first N labeled queries for tuning/eval
- `--interactive-only` – skip evaluation and open console

## Model details
- TF‑IDF + cosine: classic lexical model; robust, fast.
- BM25: probabilistic retrieval with term saturation and length normalization (parameters k1, b tuned by grid search unless `--no-tune`).

## Troubleshooting
- SSL/WordNet download issues on macOS: run the certificate installer (see above). The app works without WordNet.
- Slow startup: use `--no-tune` or `--interactive-only`.

## Example commands
- Fast tuning on a subset, then interactive:
```bash
python3 full_text_search_practice.py --fast --eval-limit 10
```
-- BM25 only, defaults:
```bash
python3 full_text_search_practice.py --no-tune --model bm25 --interactive-only
```



