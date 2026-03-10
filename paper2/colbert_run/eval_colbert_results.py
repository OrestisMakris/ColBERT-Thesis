import os, json, glob
from collections import defaultdict

# Config — update COLBERT_EXPERIMENTS_DIR to match your actual run timestamp
COLBERT_EXPERIMENTS_DIR = "./experiments/CF19/paper2.colbert_run.retriever"
QRELS_PATH = "./CF_DataSet/Relevant.txt"
TOPK_VALUES = [5, 10, 20, 25, 40, 50, 80, 100, 150, 180, 200, 250, 400, 500, 1000]

# TEST_QUERIES = {15,17,70,83,86,98,102,115,170,178,210,215,222,254,287,342,364,378,387,392,394,395,399,424,
# 431,445,446,454,476,510,571,650,677,685,699,703,729,739,757,764,775,786,831,843,844,845,850,881,887,895,918,955,
# 1014,1041,1057,1097}

TEST_QUERIES = {3,4,11,13,14,17,28,31,35,54,69,75,81,86,94}

# TEST_QUERIES = {86}

def load_qrels_from_relevant(path, test_only=False):
    """Load qrels from Relevant.txt where line N = query N (0-indexed)."""
    qrels = {}
    with open(path, 'r') as f:
        for qid, line in enumerate(f, start=0):
            if test_only and qid not in TEST_QUERIES:
                continue
            relevant_docs = set()
            for doc_str in line.strip().split():
                if doc_str.isdigit():
                    relevant_docs.add(int(doc_str))
            if relevant_docs:
                qrels[qid] = relevant_docs
    return qrels


def find_latest_colbert_ranking(experiments_dir):
    """Find the most recent ColBERT ranking TSV file."""
    patterns = [
        os.path.join(experiments_dir, "**", "*.ranking.tsv"),
        os.path.join(experiments_dir, "**", "*ranking*.tsv"),
        os.path.join(experiments_dir, "**", "ranking.tsv"),
        os.path.join(experiments_dir, "**", "*.tsv"),
    ]
    ranking_files = []
    for pattern in patterns:
        ranking_files.extend(glob.glob(pattern, recursive=True))
    if not ranking_files:
        raise FileNotFoundError(f"No ranking files found in {experiments_dir}")
    ranking_files = [f for f in ranking_files if not any(
        skip in f.lower() for skip in ['config', 'metadata', 'plan', 'collection'])]
    if not ranking_files:
        raise FileNotFoundError(f"No ranking files found in {experiments_dir}")
    ranking_files.sort(key=os.path.getmtime, reverse=True)
    print(f"Found ranking files: {[os.path.basename(f) for f in ranking_files[:3]]}")
    return ranking_files[0]


def load_colbert_ranking(ranking_file, test_queries_only=False):
    """Load ColBERT ranking file: qid \\t did \\t rank [\\t score]"""
    rankings = defaultdict(list)
    print(f"Loading ColBERT ranking from: {ranking_file}")
    with open(ranking_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    qid  = int(parts[0])
                    did  = int(parts[1])
                    if test_queries_only and qid not in TEST_QUERIES:
                        continue
                    rank  = int(parts[2])   if len(parts) > 2 else len(rankings[qid]) + 1
                    score = float(parts[3]) if len(parts) > 3 else 0.0
                    rankings[qid].append((did, score, rank))
                except ValueError:
                    if line_num <= 5:
                        print(f"Warning: Could not parse line {line_num}: {line.strip()}")
    for qid in rankings:
        rankings[qid].sort(key=lambda x: x[2])
    print(f"Loaded rankings for {len(rankings)} queries")
    return rankings


def average_precision(retrieved_docs, relevant_docs):
    if not relevant_docs:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant_docs)


def reciprocal_rank(retrieved_docs, relevant_docs):
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved_docs, relevant_docs, k=1):
    if not retrieved_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    return sum(1 for doc in top_k if doc in relevant_docs) / len(top_k)


def evaluate_at_k(rankings, qrels, topk):
    total_map = total_mrr = total_p1 = total_p5 = total_p10 = 0.0
    evaluated_queries = 0
    for qid in qrels:
        if qid not in rankings:
            continue
        retrieved_docs = [did for did, _, _ in rankings[qid][:topk]]
        relevant_docs  = qrels[qid]
        total_map  += average_precision(retrieved_docs, relevant_docs)
        total_mrr  += reciprocal_rank(retrieved_docs, relevant_docs)
        total_p1   += precision_at_k(retrieved_docs, relevant_docs, 1)
        total_p5   += precision_at_k(retrieved_docs, relevant_docs, 5)
        total_p10  += precision_at_k(retrieved_docs, relevant_docs, 10)
        evaluated_queries += 1
    if evaluated_queries == 0:
        return None
    n = evaluated_queries
    return {'MAP': total_map/n, 'MRR': total_mrr/n,
            'P@1': total_p1/n, 'P@5': total_p5/n, 'P@10': total_p10/n,
            'num_queries': n, 'topk': topk}


def main():
    print("=== ColBERT Multi-TOPK Evaluation (paper2) ===")
    print(f"TOPK values: {TOPK_VALUES}")

    print("\nEvaluation options:\n1) All queries\n2) Test queries only")
    eval_mode = input("Choose [1/2]: ").strip()
    test_only = eval_mode == "2"

    qrels = load_qrels_from_relevant(QRELS_PATH, test_only=test_only)
    print(f"Loaded {len(qrels)} queries from Relevant.txt")

    ranking_file = find_latest_colbert_ranking(COLBERT_EXPERIMENTS_DIR)
    print(f"Using ranking file: {ranking_file}")

    rankings = load_colbert_ranking(ranking_file, test_queries_only=test_only)
    if not rankings:
        print("No rankings loaded!")
        return

    header = f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    all_results = []
    for topk in TOPK_VALUES:
        res = evaluate_at_k(rankings, qrels, topk)
        if res:
            all_results.append(res)
            print(f"{res['topk']:<6} {res['num_queries']:<8} "
                  f"{res['MAP']:<8.4f} {res['MRR']:<8.4f} "
                  f"{res['P@1']:<8.4f} {res['P@5']:<8.4f} {res['P@10']:<8.4f}")
        else:
            print(f"{topk:<6} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
    print(sep)

    tag = 'test' if test_only else 'all'
    output_file = f"colbert_multi_topk_results_{tag}.txt"
    with open(output_file, 'w') as f:
        f.write("ColBERT Multi-TOPK Evaluation Results (paper2)\n")
        f.write(f"{sep}\n{header}\n{sep}\n")
        for res in all_results:
            f.write(f"{res['topk']:<6} {res['num_queries']:<8} "
                    f"{res['MAP']:<8.4f} {res['MRR']:<8.4f} "
                    f"{res['P@1']:<8.4f} {res['P@5']:<8.4f} {res['P@10']:<8.4f}\n")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
