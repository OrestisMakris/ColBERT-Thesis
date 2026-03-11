"""
Evaluate sparse retrieval baselines (no ColBERT, no CNN).

Baselines:
  1. BM25          — Okapi BM25 (Robertson et al., 1994)
  2. TF-IDF        — cosine similarity over TF-IDF vectors
  3. BM25 + RM3    — BM25 with pseudo-relevance feedback query expansion
  4. QL (Dirichlet)— Query Likelihood with Dirichlet smoothing (Zhai & Lafferty, 2001)

All baselines index the full corpus and rank all documents per query.
Computes MAP, MRR, P@1, P@5, P@10 at multiple top-k cutoffs.

Usage:
  python paper2/cfrun_untuned/eval_sparse_baselines.py \
    --docs_path   CF_DataSet/docs.tsv \
    --queries_path CF_DataSet/Queries.tsv \
    --qrels_path  paper2/cfrun_untuned/test_triplets_hard.jsonl \
    --topk 5,10,15,20,25,50,100,250,500,1000
"""
import os
import json
import argparse
import math
import numpy as np
from collections import defaultdict, Counter
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_docs(docs_path):
    """Load docs.tsv → dict {pid: text}."""
    docs = {}
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                docs[int(parts[0])] = parts[1]
    return docs


def load_queries(queries_path):
    """Load Queries.tsv → dict {qid: text}."""
    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[int(parts[0])] = parts[1]
            elif len(parts) == 1 and parts[0].strip():
                queries[len(queries)] = parts[0]
    return queries


def load_qrels(jsonl_path):
    """Load ground-truth qrels from JSONL."""
    qrels = defaultdict(set)
    with open(jsonl_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if len(data) >= 2:
                    qrels[int(data[0])].add(int(data[1]))
            except Exception:
                continue
    return qrels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def average_precision(retrieved, relevant, k):
    if not relevant:
        return 0.0
    hits = 0
    s = 0.0
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant:
            hits += 1
            s += hits / i
    return s / len(relevant)


def reciprocal_rank(retrieved, relevant, k):
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / k if k > 0 else 0.0


def evaluate_rankings(rankings, qrels, eval_qids, topk_values, label):
    header = f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}"
    sep = "=" * len(header)
    print(f"\n--- {label} ---")
    print(f"{sep}\n{header}\n{sep}")
    for k in topk_values:
        map_s = mrr_s = p1_s = p5_s = p10_s = 0.0
        for qid in eval_qids:
            ret = rankings.get(qid, [])
            rel = qrels[qid]
            map_s += average_precision(ret, rel, k)
            mrr_s += reciprocal_rank(ret, rel, k)
            p1_s += precision_at_k(ret, rel, 1)
            p5_s += precision_at_k(ret, rel, 5)
            p10_s += precision_at_k(ret, rel, 10)
        n = len(eval_qids)
        print(f"{k:<6} {n:<8} {map_s/n:<8.4f} {mrr_s/n:<8.4f} {p1_s/n:<8.4f} {p5_s/n:<8.4f} {p10_s/n:<8.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Tokenizer (simple whitespace + lowercase)
# ---------------------------------------------------------------------------

def tokenize(text):
    return text.lower().split()


# ---------------------------------------------------------------------------
# Baseline 1: BM25
# ---------------------------------------------------------------------------

def rank_bm25(docs, queries, eval_qids):
    """Rank documents using Okapi BM25."""
    pids = sorted(docs.keys())
    corpus = [tokenize(docs[pid]) for pid in pids]
    bm25 = BM25Okapi(corpus)

    rankings = {}
    for qid in eval_qids:
        if qid not in queries:
            continue
        q_tokens = tokenize(queries[qid])
        scores = bm25.get_scores(q_tokens)
        ranked_indices = np.argsort(scores)[::-1]
        rankings[qid] = [pids[i] for i in ranked_indices]
    return rankings


# ---------------------------------------------------------------------------
# Baseline 2: TF-IDF cosine similarity
# ---------------------------------------------------------------------------

def rank_tfidf(docs, queries, eval_qids):
    """Rank documents using TF-IDF + cosine similarity."""
    pids = sorted(docs.keys())
    corpus = [docs[pid] for pid in pids]

    vectorizer = TfidfVectorizer(lowercase=True)
    doc_matrix = vectorizer.fit_transform(corpus)

    rankings = {}
    for qid in eval_qids:
        if qid not in queries:
            continue
        q_vec = vectorizer.transform([queries[qid]])
        sims = cosine_similarity(q_vec, doc_matrix).flatten()
        ranked_indices = np.argsort(sims)[::-1]
        rankings[qid] = [pids[i] for i in ranked_indices]
    return rankings


# ---------------------------------------------------------------------------
# Baseline 3: BM25 + RM3 (pseudo-relevance feedback)
# ---------------------------------------------------------------------------

def rank_bm25_rm3(docs, queries, eval_qids, fb_docs=10, fb_terms=10, alpha=0.5):
    """
    BM25 first pass, then RM3 query expansion (Lavrenko & Croft, 2001):
      - Take top fb_docs from BM25
      - Extract fb_terms most frequent terms from those docs
      - Interpolate expanded query with original (weight alpha)
      - Re-rank with BM25 using expanded query
    """
    pids = sorted(docs.keys())
    corpus_tokenized = [tokenize(docs[pid]) for pid in pids]
    bm25 = BM25Okapi(corpus_tokenized)

    rankings = {}
    for qid in eval_qids:
        if qid not in queries:
            continue
        q_tokens = tokenize(queries[qid])

        # First pass
        scores = bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:fb_docs]

        # Collect feedback terms from top docs
        term_counts = Counter()
        for idx in top_indices:
            term_counts.update(corpus_tokenized[idx])
        # Remove query terms to get expansion terms
        expansion_terms = [t for t, _ in term_counts.most_common(fb_terms + len(q_tokens))
                          if t not in set(q_tokens)][:fb_terms]

        # Expanded query: original terms (weighted) + expansion terms
        expanded = q_tokens * max(1, int(alpha * 10)) + expansion_terms * max(1, int((1 - alpha) * 10))

        # Second pass
        scores2 = bm25.get_scores(expanded)
        ranked_indices = np.argsort(scores2)[::-1]
        rankings[qid] = [pids[i] for i in ranked_indices]
    return rankings


# ---------------------------------------------------------------------------
# Baseline 4: Query Likelihood with Dirichlet smoothing
# ---------------------------------------------------------------------------

def rank_ql_dirichlet(docs, queries, eval_qids, mu=2000):
    """
    Query Likelihood model with Dirichlet prior smoothing.
    P(q|d) = prod_i [ (tf(t_i,d) + mu * P(t_i|C)) / (|d| + mu) ]
    Score in log-space for numerical stability.
    """
    pids = sorted(docs.keys())
    corpus_tokenized = [tokenize(docs[pid]) for pid in pids]

    # Collection statistics
    total_tokens = 0
    collection_tf = Counter()
    for doc_tokens in corpus_tokenized:
        total_tokens += len(doc_tokens)
        collection_tf.update(doc_tokens)

    rankings = {}
    for qid in eval_qids:
        if qid not in queries:
            continue
        q_tokens = tokenize(queries[qid])

        scores = []
        for i, doc_tokens in enumerate(corpus_tokenized):
            doc_len = len(doc_tokens)
            doc_tf = Counter(doc_tokens)
            log_score = 0.0
            for t in q_tokens:
                p_collection = collection_tf.get(t, 0) / total_tokens if total_tokens > 0 else 0
                tf = doc_tf.get(t, 0)
                prob = (tf + mu * p_collection) / (doc_len + mu)
                if prob > 0:
                    log_score += math.log(prob)
                else:
                    log_score += -100  # very low score for unseen terms
            scores.append(log_score)

        ranked_indices = np.argsort(scores)[::-1]
        rankings[qid] = [pids[i] for i in ranked_indices]
    return rankings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate sparse retrieval baselines")
    parser.add_argument("--docs_path",    required=True, help="Path to docs.tsv")
    parser.add_argument("--queries_path", required=True, help="Path to Queries.tsv")
    parser.add_argument("--qrels_path",   required=True, help="Path to test_triplets_hard.jsonl")
    parser.add_argument("--topk",         type=str, default="5,10,15,20,25,50,100,250,500,1000",
                        help="Comma-separated K values")
    parser.add_argument("--baselines",    type=str, default="bm25,tfidf,bm25rm3,ql",
                        help="Comma-separated baselines to run (bm25,tfidf,bm25rm3,ql)")
    args = parser.parse_args()

    topk_values = sorted(set(int(k) for k in args.topk.split(",")))
    baselines = [b.strip().lower() for b in args.baselines.split(",")]

    docs    = load_docs(args.docs_path)
    queries = load_queries(args.queries_path)
    qrels   = load_qrels(args.qrels_path)

    eval_qids = sorted(qrels.keys())
    print(f"Corpus: {len(docs)} docs, {len(queries)} queries, evaluating {len(eval_qids)} test queries.")

    if "bm25" in baselines:
        print("\nComputing BM25 rankings …")
        bm25_rankings = rank_bm25(docs, queries, eval_qids)
        evaluate_rankings(bm25_rankings, qrels, eval_qids, topk_values,
                          "Baseline: BM25 (Okapi)")

    if "tfidf" in baselines:
        print("\nComputing TF-IDF rankings …")
        tfidf_rankings = rank_tfidf(docs, queries, eval_qids)
        evaluate_rankings(tfidf_rankings, qrels, eval_qids, topk_values,
                          "Baseline: TF-IDF cosine")

    if "bm25rm3" in baselines:
        print("\nComputing BM25 + RM3 rankings …")
        rm3_rankings = rank_bm25_rm3(docs, queries, eval_qids)
        evaluate_rankings(rm3_rankings, qrels, eval_qids, topk_values,
                          "Baseline: BM25 + RM3 (pseudo-relevance feedback)")

    if "ql" in baselines:
        print("\nComputing Query Likelihood (Dirichlet) rankings …")
        ql_rankings = rank_ql_dirichlet(docs, queries, eval_qids)
        evaluate_rankings(ql_rankings, qrels, eval_qids, topk_values,
                          "Baseline: QL Dirichlet (mu=2000)")


if __name__ == "__main__":
    main()
