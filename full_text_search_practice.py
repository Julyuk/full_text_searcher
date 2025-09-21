#!/usr/bin/env python3
import re, math, html, ast, string, os, time, argparse
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from spellchecker import SpellChecker
import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Text cleaning ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE)
PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
STOPWORDS = set("""a an and are as at be by for from has he in is it its of on or that the to was were will with this these those into over under above below you your yours their them we our ours she her his they i not no than then""".split())

# Check NLTK data availability (no auto-downloads to avoid SSL issues)
WN_AVAILABLE = False
def ensure_nltk_data():
    global WN_AVAILABLE
    try:
        nltk.data.find("corpora/wordnet")
        WN_AVAILABLE = True
    except LookupError:
        WN_AVAILABLE = False

ensure_nltk_data()

# No heuristic post-filters; retrieval is purely TF‑IDF or BM25

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def tokenize(text: str, use_bigrams=True):
    t = [w for w in normalize(text).split() if w and w not in STOPWORDS]
    if use_bigrams:
        bigrams = ["_".join(pair) for pair in zip(t, t[1:])]
        return t + bigrams
    return t

def get_synonyms(token: str, max_per_word: int = 3):
    if not WN_AVAILABLE:
        return []
    syns = set()
    try:
        for syn in wn.synsets(token):
            for lemma in syn.lemmas():
                name = lemma.name().replace("-", "_").lower()
                if name and name != token and name not in STOPWORDS and name.isascii():
                    syns.add(name)
            if len(syns) >= max_per_word:
                break
    except Exception:
        return []
    return list(syns)[:max_per_word]

def tokenize_query(query: str, vocab_words: set, spell: SpellChecker | None = None, use_bigrams: bool = True):
    base_tokens = [w for w in normalize(query).split() if w and w not in STOPWORDS]
    corrected_tokens = []
    if spell is not None and base_tokens:
        unknown = [w for w in base_tokens if w not in vocab_words]
        corrections = {}
        try:
            for w in unknown:
                suggestion = spell.correction(w)
                if suggestion and suggestion != w:
                    corrections[w] = suggestion
        except Exception:
            corrections = {}
        for w in base_tokens:
            corrected_tokens.append(corrections.get(w, w))
    else:
        corrected_tokens = base_tokens
    expanded = list(corrected_tokens)
    for w in corrected_tokens:
        expanded.extend(get_synonyms(w, max_per_word=2))
    if use_bigrams:
        bigrams = ["_".join(pair) for pair in zip(corrected_tokens, corrected_tokens[1:])]
        expanded.extend(bigrams)
    return expanded

# Removed heuristic post-filters

# ---------- Index building ----------
def build_indexes(corpus_df):
    titles = corpus_df["display name"].fillna("").tolist()
    descs = corpus_df["description"].fillna("").tolist()
    docs = [ti + " " + de for ti, de in zip(titles, descs)]
    tokens_title_list = [tokenize(t, use_bigrams=True) for t in titles]
    tokens_desc_list = [tokenize(t, use_bigrams=True) for t in descs]
    tokens_list = [tokenize(t) for t in docs]
    vocab = {}
    for toks in tokens_list:
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    V = len(vocab)
    N = len(tokens_list)
    tf = [Counter(toks) for toks in tokens_list]
    tf_title = [Counter(toks) for toks in tokens_title_list]
    tf_desc = [Counter(toks) for toks in tokens_desc_list]
    dfreq = Counter()
    for c in tf:
        for term in c.keys():
            dfreq[term] += 1
    def idf(term):
        return math.log(1 + (N - dfreq.get(term, 0) + 0.5) / (dfreq.get(term, 0) + 0.5))
    idf_vec = np.zeros(V, dtype=np.float32)
    for term, idx in vocab.items():
        idf_vec[idx] = idf(term)
    tfidf_docs, doc_norms = [], np.zeros(N, dtype=np.float32)
    for i, c in enumerate(tf):
        row = {}
        for term, freq in c.items():
            j = vocab[term]
            val = (1 + math.log(freq)) * idf_vec[j]
            row[j] = val
        norm = math.sqrt(sum(v*v for v in row.values())) or 1.0
        doc_norms[i] = norm
        tfidf_docs.append(row)
    avgdl = np.mean([sum(c.values()) for c in tf])
    doc_lens = np.array([sum(c.values()) for c in tf], dtype=np.float32)
    avgdl_title = np.mean([sum(c.values()) for c in tf_title])
    avgdl_desc = np.mean([sum(c.values()) for c in tf_desc])
    doc_lens_title = np.array([sum(c.values()) for c in tf_title], dtype=np.float32)
    doc_lens_desc = np.array([sum(c.values()) for c in tf_desc], dtype=np.float32)
    return {
        "vocab": vocab, "tf": tf, "df": dfreq, "idf_vec": idf_vec,
        "tfidf_docs": tfidf_docs, "doc_norms": doc_norms,
        "avgdl": float(avgdl), "doc_lens": doc_lens,
        "tf_title": tf_title, "tf_desc": tf_desc,
        "avgdl_title": float(avgdl_title), "avgdl_desc": float(avgdl_desc),
        "doc_lens_title": doc_lens_title, "doc_lens_desc": doc_lens_desc
    }

## build_pmi_synonyms removed (simplified)

# ---------- Scoring ----------
def score_tfidf(query, idx, spell: SpellChecker | None = None):
    vocab = idx["vocab"]; idf_vec = idx["idf_vec"]
    # Similarity choice: cosine similarity. Rationale: with TF-IDF vectors, cosine
    # emphasizes orientation (term importance pattern) rather than magnitude, which
    # tends to be more robust for sparse high-dimensional text representations.
    q_tokens = tokenize_query(query, set(vocab.keys()), spell)
    q_counts = Counter(q_tokens)
    q_vec = {}
    for t, fq in q_counts.items():
        if t in vocab:
            j = vocab[t]
            q_vec[j] = (1 + math.log(fq)) * idf_vec[j]
    q_norm = math.sqrt(sum(v*v for v in q_vec.values())) or 1.0
    N = len(idx["tfidf_docs"])
    scores = np.zeros(N, dtype=np.float32)
    for i, row in enumerate(idx["tfidf_docs"]):
        s = 0.0
        for j, qv in q_vec.items():
            dv = row.get(j)
            if dv is not None:
                s += qv * dv
        scores[i] = s / (idx["doc_norms"][i] * q_norm)
    return scores

def score_bm25(query, idx, k1=1.5, b=0.75, spell: SpellChecker | None = None):
    # BM25 with query preprocessing (spell-correction and synonyms expansion)
    q_tokens = tokenize_query(query, set(idx["vocab"].keys()), spell)
    q_terms = [t for t in q_tokens if t in idx["vocab"]]
    N = len(idx["tfidf_docs"])
    scores = np.zeros(N, dtype=np.float32)
    for t in set(q_terms):
        j = idx["vocab"][t]
        df_t = idx["df"][t]
        idf_t = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        for i, c in enumerate(idx["tf"]):
            f = c.get(t, 0)
            if f == 0: continue
            denom = f + k1 * (1 - b + b * (idx["doc_lens"][i] / idx["avgdl"]))
            s = idf_t * (f * (k1 + 1)) / denom
            scores[i] += s
    return scores

def score_bm25f(query, idx, k1=1.5, b_title=0.75, b_desc=0.75, w_title=2.0, w_desc=1.0, spell: SpellChecker | None = None, rm3: bool = False, rm3_docs: int = 10, rm3_terms: int = 8, rm3_alpha: float = 0.6):
    vocab = idx["vocab"]
    q_tokens = tokenize_query(query, set(vocab.keys()), spell)
    def run_once(q_terms_list):
        q_terms = [t for t in q_terms_list if t in vocab]
        N = len(idx["tf"])
        scores = np.zeros(N, dtype=np.float32)
        for t in set(q_terms):
            j = vocab[t]
            df_t = idx["df"][t]
            idf_t = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
            for i in range(N):
                ft = idx["tf_title"][i].get(t, 0)
                fd = idx["tf_desc"][i].get(t, 0)
                if ft == 0 and fd == 0: continue
                norm_t = ft / (1 - b_title + b_title * (idx["doc_lens_title"][i] / idx["avgdl_title"]))
                norm_d = fd / (1 - b_desc + b_desc * (idx["doc_lens_desc"][i] / idx["avgdl_desc"]))
                tfp = w_title * norm_t + w_desc * norm_d
                s = idf_t * (tfp * (k1 + 1)) / (tfp + k1)
                scores[i] += s
        return scores
    scores = run_once(q_tokens)
    if rm3:
        # RM3 pseudo relevance feedback: extract top terms from top docs and re-run
        top_idx = np.argsort(-scores)[:rm3_docs]
        term_counts = Counter()
        for i in top_idx:
            term_counts.update(idx["tf"][i])
        prf_terms = [t for t, _ in term_counts.most_common(50) if t not in STOPWORDS][:rm3_terms]
        expanded = q_tokens + [t for t in prf_terms for _ in range(int(3 * rm3_alpha))]
        scores = run_once(expanded)
    return scores

# ---------- Metrics ----------
def evaluate(pred_ids, gold_ids, k=10):
    gold_set = set(gold_ids)
    def dcg_at_k(rels, k):
        return sum((rel / math.log2(idx+2)) for idx, rel in enumerate(rels[:k]))
    def ndcg_at_k(pred, gold_set, k=10):
        rels = [1.0 if pid in gold_set else 0.0 for pid in pred[:k]]
        ideal = sorted(rels, reverse=True)
        idcg = dcg_at_k(ideal, k)
        return (dcg_at_k(rels, k) / idcg) if idcg > 0 else 0.0
    def precision_at_k(pred, gold_set, k):
        return sum(1 for pid in pred[:k] if pid in gold_set) / float(k)
    def recall_at_k(pred, gold_set, k):
        if not gold_set:
            return 0.0
        return sum(1 for pid in pred[:k] if pid in gold_set) / float(len(gold_set))
    def fscore_at_k(pred, gold_set, k):
        p = precision_at_k(pred, gold_set, k)
        r = recall_at_k(pred, gold_set, k)
        return (2*p*r / (p + r)) if (p + r) > 0 else 0.0
    def average_precision(pred, gold_set, k=10):
        hits, sum_prec = 0, 0.0
        for i, pid in enumerate(pred[:k], start=1):
            if pid in gold_set:
                hits += 1
                sum_prec += hits / i
        return (sum_prec / max(1, len(gold_set))) if gold_set else 0.0
    def mrr_at_k(pred, gold_set, k=10):
        for i, pid in enumerate(pred[:k], start=1):
            if pid in gold_set:
                return 1.0 / i
        return 0.0
    return {
        "P@1": precision_at_k(pred_ids, gold_set, 1),
        "P@3": precision_at_k(pred_ids, gold_set, 3),
        "P@5": precision_at_k(pred_ids, gold_set, 5),
        "P@10": precision_at_k(pred_ids, gold_set, 10),
        "R@5": recall_at_k(pred_ids, gold_set, 5),
        "F1@5": fscore_at_k(pred_ids, gold_set, 5),
        "nDCG@10": ndcg_at_k(pred_ids, gold_set, 10),
        "AP@10": average_precision(pred_ids, gold_set, 10),
        "MRR@10": mrr_at_k(pred_ids, gold_set, 10)
    }

def grid_search_bm25(queries_df, corpus_df, idx, spell: SpellChecker | None = None, k1_values=None, b_values=None, limit: int | None = None):
    # BM25 parameter optimization via exhaustive grid search.
    # We optimize mean F1@5 across the labeled queries.
    k1_values = k1_values or [1.0, 1.2, 1.5, 1.8, 2.0]
    b_values = b_values or [0.55, 0.65, 0.75, 0.85, 0.95]
    best_score, best_params = -1.0, (1.5, 0.75)
    q_iter = queries_df.iterrows() if limit is None else list(queries_df.iterrows())[:limit]
    for k1 in k1_values:
        for b in b_values:
            f1_scores = []
            for _, row in q_iter:
                scores = score_bm25(row["queries"], idx, k1=k1, b=b, spell=spell)
                topk = np.argsort(-scores)[:100]
                pred = corpus_df.loc[topk, "id"].tolist()[:5]
                gold = set(ast.literal_eval(row["expected_results"]))
                tp = len([x for x in pred if x in gold])
                p = tp / 5.0 if 5 > 0 else 0.0
                r = (tp / float(len(gold))) if gold else 0.0
                f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
                f1_scores.append(f1)
            mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
            if mean_f1 > best_score:
                best_score, best_params = mean_f1, (k1, b)
    return best_params, best_score

def grid_search_bm25f(queries_df, corpus_df, idx, spell: SpellChecker | None = None,
                      k1_values=None, b_title_values=None, b_desc_values=None,
                      w_title_values=None, w_desc_values=None,
                      use_rm3=None, rm3_alpha_values=None, limit: int | None = None):
    # Tune BM25F parameters and field weights; optimize mean F1@5
    k1_values = k1_values or [1.2, 1.5, 1.8]
    b_title_values = b_title_values or [0.6, 0.75]
    b_desc_values = b_desc_values or [0.6, 0.75]
    w_title_values = w_title_values or [1.0, 1.5, 2.0]
    w_desc_values = w_desc_values or [0.5, 1.0]
    use_rm3 = use_rm3 or [False, True]
    rm3_alpha_values = rm3_alpha_values or [0.5, 0.7]
    best, best_score = None, -1.0
    q_iter = queries_df.iterrows() if limit is None else list(queries_df.iterrows())[:limit]
    for k1 in k1_values:
        for bt in b_title_values:
            for bd in b_desc_values:
                for wt in w_title_values:
                    for wd in w_desc_values:
                        for use_prf in use_rm3:
                            for a in (rm3_alpha_values if use_prf else [0.0]):
                                f1_scores = []
                                for _, row in q_iter:
                                    scores = score_bm25f(row["queries"], idx, k1=k1, b_title=bt, b_desc=bd, w_title=wt, w_desc=wd, spell=spell, rm3=use_prf, rm3_alpha=a)
                                    topk = np.argsort(-scores)[:100]
                                    pred = corpus_df.loc[topk, "id"].tolist()[:5]
                                    gold = set(ast.literal_eval(row["expected_results"]))
                                    tp = len([x for x in pred if x in gold])
                                    p = tp / 5.0 if 5 > 0 else 0.0
                                    r = (tp / float(len(gold))) if gold else 0.0
                                    f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
                                    f1_scores.append(f1)
                                mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
                                if mean_f1 > best_score:
                                    best_score = mean_f1
                                    best = (k1, bt, bd, wt, wd, use_prf, a)
    return best, best_score

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Full-text search console app")
    parser.add_argument("--model", choices=["tfidf","bm25"], default="bm25", help="Default model for interactive mode")
    parser.add_argument("--fast", action="store_true", help="Use a tiny grid and subset of queries to speed up tuning")
    parser.add_argument("--no-tune", action="store_true", help="Skip parameter tuning; use defaults")
    parser.add_argument("--eval-limit", type=int, default=None, help="Max number of labeled queries to use for evaluation/tuning")
    parser.add_argument("--interactive-only", action="store_true", help="Skip evaluation and go straight to interactive search")
    # Simplified flags only
    args = parser.parse_args()
    corpus = pd.read_csv("corpus.csv", sep="|", engine="python")
    queries = pd.read_csv("queries.csv", sep="|", engine="python")
    t0 = time.time()
    idx = build_indexes(corpus)
    print(f"Index built in {time.time()-t0:.2f}s, {len(idx['vocab'])} terms, {len(idx['tf'])} docs")

    # Initialize spell checker and seed with corpus vocab to limit false corrections
    spell = None
    try:
        spell = SpellChecker(language="en")
        spell.word_frequency.load_words(list(idx["vocab"].keys()))
        extra_words = set()
        for t in (corpus["display name"].fillna("") + " " + corpus["description"].fillna("")).tolist():
            extra_words.update([w for w in normalize(t).split() if w])
        if extra_words:
            spell.word_frequency.load_words(list(extra_words))
    except Exception:
        spell = None

    def run(score_fn, name):
        rows = []
        for _, row in queries.iterrows():
            scores = score_fn(row["queries"], idx)
            topk = np.argsort(-scores)[:100]
            pred = corpus.loc[topk, "id"].tolist()[:10]
            gold = ast.literal_eval(row["expected_results"])
            metrics = evaluate(pred, gold, k=10)
            metrics["query"] = row["queries"]
            metrics["model"] = name
            rows.append(metrics)
        return pd.DataFrame(rows)

    # Optionally limit evaluation set
    eval_queries = queries if args.eval_limit is None else queries.head(args.eval_limit)

    # PMI synonyms removed (keep WordNet synonyms + spell-correction)

    # Optional fast grids
    if args.fast:
        bm25_k1 = [1.2, 1.5]; bm25_b = [0.65, 0.75]
        limit = args.eval_limit or min(10, len(eval_queries))
    else:
        bm25_k1 = None; bm25_b = None; limit = args.eval_limit

    # TF-IDF with cosine similarity and query preprocessing
    tfidf_df = pd.DataFrame()
    if not args.interactive_only:
        tfidf_df = run(lambda q, idx_: score_tfidf(q, idx_, spell=spell), "tfidf")

    # BM25: tune unless disabled
    if args.no_tune:
        best_k1, best_b = 1.5, 0.75
    else:
        t1 = time.time()
        (best_k1, best_b), _ = grid_search_bm25(eval_queries, corpus, idx, spell=spell, k1_values=bm25_k1, b_values=bm25_b, limit=limit)
        print(f"Tuned BM25 in {time.time()-t1:.2f}s -> k1={best_k1}, b={best_b}")

    bm25_df = pd.DataFrame() if args.interactive_only else run(lambda q, idx_: score_bm25(q, idx_, k1=best_k1, b=best_b, spell=spell), f"bm25(k1={best_k1},b={best_b})")

    # Save results to Results/ with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("Results", exist_ok=True)
    if not args.interactive_only:
        out = pd.concat([df for df in [tfidf_df, bm25_df] if not df.empty], ignore_index=True)
        results_path = os.path.join("Results", f"results_{ts}.csv")
        out.to_csv(results_path, index=False)
        print(f"Saved metrics to {results_path}")
        agg = out.groupby("model")[ ["P@5","R@5","F1@5","nDCG@10","AP@10","MRR@10"] ].mean().round(6)
        print(agg)

    # Create diagrams
    try:
        sns.set(style="whitegrid")
        agg_reset = agg.reset_index()
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        df_m = agg_reset.melt(id_vars=["model"], value_vars=["P@5","R@5","F1@5"], var_name="metric", value_name="score")
        sns.barplot(data=df_m, x="metric", y="score", hue="model", ax=ax)
        ax.set_title("Mean metrics@5 by model")
        ax.set_ylim(0, 1)
        ax.legend(title="model")
        fig.tight_layout()
        fig_path = os.path.join("Results", f"metrics_{ts}.png")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved figure to {fig_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Interactive console mode for custom queries
    sample_queries = [
        "wireless headphones",
        "bluetooth speaker waterproof",
        "gaming laptop 16gb ram",
        "running shoes men size 10",
        "smartphone with good camera",
        "usb-c hub 7 in 1",
    ]
    print("\nInteractive search. Type your query and press Enter. Type ':help' for commands. Type 'exit' to quit.")
    print("Sample queries:")
    for q in sample_queries:
        print(f"  - {q}")
    chosen_model = "bm25"  # default to tuned BM25
    print(f"Current model: {args.model.upper()}. Switch with ':model tfidf' | ':model bm25'.")
    chosen_model = args.model
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.startswith(":"):
            cmd = q[1:].strip().lower()
            if cmd in {"help", "h"}:
                print("Commands: :model tfidf | :model bm25 | :help | :exit")
                continue
            if cmd.startswith("model"):
                parts = cmd.split()
                if len(parts) >= 2 and parts[1] in {"tfidf", "bm25"}:
                    chosen_model = parts[1]
                    print(f"Switched model to {chosen_model.upper()}")
                else:
                    print("Usage: :model tfidf | :model bm25")
                continue
        if chosen_model == "tfidf":
            scores = score_tfidf(q, idx, spell=spell)
        elif chosen_model == "bm25":
            scores = score_bm25(q, idx, k1=best_k1, b=best_b, spell=spell)
        topk = np.argsort(-scores)[:10]
        results = corpus.loc[topk, ["id", "display name"]].reset_index(drop=True)
        print(results.to_string(index=False))

if __name__ == "__main__":
    main()
