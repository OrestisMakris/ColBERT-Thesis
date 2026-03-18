"""
Evaluate sparse retrieval baselines (no ColBERT, no CNN).

Baselines:
  1. BM25          — Okapi BM25 (Robertson et al., 1994)
  2. TF-IDF        — cosine similarity over TF-IDF vectors
  3. BM25 + RM3    — BM25 with pseudo-relevance feedback query expansion
  4. QL (Dirichlet)— Query Likelihood with Dirichlet smoothing (Zhai & Lafferty, 2001)
    5. BGE-base      — dense bi-encoder retrieval (SentenceTransformers)
    6. SPLADE        — learned sparse retrieval (masked LM term expansion)

All baselines index the full corpus and rank all documents per query.
Computes MAP, MRR, P@1, P@5, P@10 at multiple top-k cutoffs.

Usage:
python paper2/cfrun_untuned/eval_sparse_baselines.py   --docs_path CF_DataSet/docs.tsv  --queries_path CF_DataSet/Queries.tsv   --qrels_path paper2/cfrun_untuned/test_triplets_hard.jsonl  --baselines bm25,tfidf,bm25rm3,ql,bge,splade   --topk 5,10,15,20,25,40,50,80,100,150,200,250,500,1000 --batch_size 16   --device cuda
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
# Baseline 5: BGE-base dense retrieval
# ---------------------------------------------------------------------------

def rank_bge_dense(docs, queries, eval_qids,
                   model_name="BAAI/bge-base-en-v1.5",
                   batch_size=64,
                   device=None):
    """Rank documents using a dense bi-encoder (BGE family)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "BGE baseline requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from e

    pids = sorted(docs.keys())
    doc_texts = [docs[pid] for pid in pids]

    model = SentenceTransformer(model_name, device=device)

    # BGE works best with this instruction prefix for queries.
    doc_emb = model.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    rankings = {}
    for qid in eval_qids:
        if qid not in queries:
            continue
        q_text = f"Represent this sentence for searching relevant passages: {queries[qid]}"
        q_emb = model.encode(
            [q_text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        scores = doc_emb @ q_emb
        ranked_indices = np.argsort(scores)[::-1]
        rankings[qid] = [pids[i] for i in ranked_indices]
    return rankings


# ---------------------------------------------------------------------------
# Baseline 6: SPLADE learned sparse retrieval
# ---------------------------------------------------------------------------

def _splade_encode_sparse_vectors(texts, tokenizer, model, device,
                                  batch_size=16, max_length=256, top_terms=128):
    """Encode texts into sparse term-weight dictionaries using SPLADE pooling."""
    import torch

    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
            attn = batch["attention_mask"].unsqueeze(-1)
            token_weights = torch.log1p(torch.relu(logits)) * attn
            pooled = torch.max(token_weights, dim=1).values

        for row in pooled:
            if top_terms and top_terms > 0 and top_terms < row.shape[0]:
                vals, idxs = torch.topk(row, k=top_terms)
                sparse = {
                    int(term_id): float(weight)
                    for term_id, weight in zip(idxs.tolist(), vals.tolist())
                    if weight > 0
                }
            else:
                nz = torch.nonzero(row > 0, as_tuple=True)[0]
                sparse = {int(term_id): float(row[term_id]) for term_id in nz.tolist()}
            vectors.append(sparse)
    return vectors


def rank_splade(docs, queries, eval_qids,
                model_name="naver/splade-cocondenser-ensembledistil",
                batch_size=16,
                max_length=256,
                doc_top_terms=128,
                query_top_terms=64,
                device=None):
    """Rank documents with SPLADE sparse vectors and inverted-index scoring."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError as e:
        raise ImportError(
            "SPLADE baseline requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pids = sorted(docs.keys())
    doc_texts = [docs[pid] for pid in pids]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    doc_vectors = _splade_encode_sparse_vectors(
        doc_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        top_terms=doc_top_terms,
    )

    # Build inverted index: term_id -> list[(doc_index, doc_weight)].
    postings = defaultdict(list)
    for doc_idx, sparse_doc in enumerate(doc_vectors):
        for term_id, weight in sparse_doc.items():
            postings[term_id].append((doc_idx, weight))

    rankings = {}
    all_doc_indices = np.arange(len(pids))
    for qid in eval_qids:
        if qid not in queries:
            continue

        q_vec = _splade_encode_sparse_vectors(
            [queries[qid]],
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=1,
            max_length=max_length,
            top_terms=query_top_terms,
        )[0]

        scores = defaultdict(float)
        for term_id, q_weight in q_vec.items():
            for doc_idx, d_weight in postings.get(term_id, []):
                scores[doc_idx] += q_weight * d_weight

        if scores:
            scored_indices = np.array(list(scores.keys()), dtype=np.int64)
            scored_values = np.array([scores[i] for i in scored_indices], dtype=np.float32)
            order = np.argsort(scored_values)[::-1]
            ranked_scored = scored_indices[order]

            scored_set = set(ranked_scored.tolist())
            remaining = [i for i in all_doc_indices.tolist() if i not in scored_set]
            final_ranked = ranked_scored.tolist() + remaining
        else:
            final_ranked = all_doc_indices.tolist()

        rankings[qid] = [pids[i] for i in final_ranked]

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
                        help="Comma-separated baselines to run (bm25,tfidf,bm25rm3,ql,bge,splade)")
    parser.add_argument("--bge_model",    type=str, default="BAAI/bge-base-en-v1.5",
                        help="SentenceTransformers model id for BGE baseline")
    parser.add_argument("--splade_model", type=str, default="naver/splade-cocondenser-ensembledistil",
                        help="Hugging Face model id for SPLADE baseline")
    parser.add_argument("--batch_size",   type=int, default=16,
                        help="Encoding batch size for neural baselines")
    parser.add_argument("--max_length",   type=int, default=256,
                        help="Max token length for SPLADE encoding")
    parser.add_argument("--splade_doc_top_terms", type=int, default=128,
                        help="Keep top-N weighted terms per doc vector")
    parser.add_argument("--splade_query_top_terms", type=int, default=64,
                        help="Keep top-N weighted terms per query vector")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for neural baselines (e.g., cpu, cuda). Auto if not set")
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

    if "bge" in baselines:
        print("\nComputing BGE-base dense rankings …")
        bge_rankings = rank_bge_dense(
            docs,
            queries,
            eval_qids,
            model_name=args.bge_model,
            batch_size=args.batch_size,
            device=args.device,
        )
        evaluate_rankings(bge_rankings, qrels, eval_qids, topk_values,
                          f"Baseline: BGE dense ({args.bge_model})")

    if "splade" in baselines:
        print("\nComputing SPLADE learned-sparse rankings …")
        splade_rankings = rank_splade(
            docs,
            queries,
            eval_qids,
            model_name=args.splade_model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            doc_top_terms=args.splade_doc_top_terms,
            query_top_terms=args.splade_query_top_terms,
            device=args.device,
        )
        evaluate_rankings(splade_rankings, qrels, eval_qids, topk_values,
                          f"Baseline: SPLADE ({args.splade_model})")


if __name__ == "__main__":
    main()
