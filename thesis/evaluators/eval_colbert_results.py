import os, json, glob
from collections import defaultdict

# Config
COLBERT_EXPERIMENTS_DIR = "colbert_run/experiments/CF1/retriever/2025-04/04/10.19.19" 
QRELS_PATH = "CF_DataSet/Relevant.txt"
TOPK_VALUES = [5, 10, 20, 25, 40, 50, 80, 100, 150, 180, 200, 250]

# Hardcoded test queries from test_pairs.jsonl
#TEST_QUERIES = {1, 6, 18, 20, 23, 28, 30, 34, 35, 36, 37, 38, 45, 46, 49, 63, 75, 85, 94, 96}
TEST_QUERIES = {0, 1, 8, 10, 11, 14, 18, 22, 27, 34, 35, 36, 41, 42, 51, 57, 59, 79, 84, 94}
def load_qrels_from_relevant(path, test_only=False):
    """Load qrels from Relevant.txt where line N = query N-1 (0-indexed)"""
    qrels = {}
    with open(path, 'r') as f:
        for qid, line in enumerate(f, start=0):  # Changed: start=0
            if test_only and qid not in TEST_QUERIES:
                continue  # Skip non-test queries
            
            relevant_docs = set()
            for doc_str in line.strip().split():
                if doc_str.isdigit():
                    relevant_docs.add(int(doc_str))
            if relevant_docs:
                qrels[qid] = relevant_docs
    return qrels

def find_latest_colbert_ranking(experiments_dir):
    """Find the most recent ColBERT ranking file"""
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
    
    # Filter out obvious non-ranking files
    ranking_files = [f for f in ranking_files if not any(skip in f.lower() 
                     for skip in ['config', 'metadata', 'plan', 'collection'])]
    
    if not ranking_files:
        raise FileNotFoundError(f"No ranking files found in {experiments_dir}")
    
    # Sort by modification time, return most recent
    ranking_files.sort(key=os.path.getmtime, reverse=True)
    print(f"Found ranking files: {[os.path.basename(f) for f in ranking_files[:3]]}")
    return ranking_files[0]

def load_colbert_ranking(ranking_file, test_queries_only=False):
    """Load ColBERT ranking file format: qid \t did \t rank \t score"""
    rankings = defaultdict(list)
    
    print(f"Loading ColBERT ranking from: {ranking_file}")
    
    with open(ranking_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    qid = int(parts[0])
                    did = int(parts[1])
                    
                    # Skip if we only want test queries and this isn't one
                    if test_queries_only and qid not in TEST_QUERIES:
                        continue
                    
                    rank = int(parts[2]) if len(parts) > 2 else len(rankings[qid]) + 1
                    score = float(parts[3]) if len(parts) > 3 else 0.0
                    
                    rankings[qid].append((did, score, rank))
                    
                except ValueError:
                    if line_num <= 5:  # Only warn for first few lines
                        print(f"Warning: Could not parse line {line_num}: {line.strip()}")
                    continue
    
    # Sort by rank (don't apply topk here, we'll do it during evaluation)
    for qid in rankings:
        rankings[qid].sort(key=lambda x: x[2])  # Sort by rank
    
    print(f"Loaded rankings for {len(rankings)} queries")
    if test_queries_only:
        print(f"Test queries found: {sorted(rankings.keys())}")
        missing_test_queries = TEST_QUERIES - set(rankings.keys())
        if missing_test_queries:
            print(f"Missing test queries: {sorted(missing_test_queries)}")
    
    return rankings

def average_precision(retrieved_docs, relevant_docs):
    """Calculate Average Precision"""
    if not relevant_docs:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i
    
    return sum_precisions / len(relevant_docs)

def reciprocal_rank(retrieved_docs, relevant_docs):
    """Calculate Reciprocal Rank"""
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved_docs, relevant_docs, k=1):
    """Calculate Precision@K"""
    if not retrieved_docs:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_in_top_k / len(top_k)

def evaluate_colbert_ranking_at_k(rankings, qrels, topk):
    """Evaluate ColBERT rankings against qrels at specific topk"""
    total_map = 0.0
    total_mrr = 0.0
    total_p1 = 0.0
    total_p5 = 0.0
    total_p10 = 0.0
    
    evaluated_queries = 0
    
    for qid in qrels:
        if qid not in rankings:
            continue
        
        # Get top-k retrieved documents (ordered by rank)
        top_k_results = rankings[qid][:topk]
        retrieved_docs = [did for did, score, rank in top_k_results]
        relevant_docs = qrels[qid]
        
        # Calculate metrics
        ap = average_precision(retrieved_docs, relevant_docs)
        rr = reciprocal_rank(retrieved_docs, relevant_docs)
        p1 = precision_at_k(retrieved_docs, relevant_docs, 1)
        p5 = precision_at_k(retrieved_docs, relevant_docs, 5)
        p10 = precision_at_k(retrieved_docs, relevant_docs, 10)
        
        total_map += ap
        total_mrr += rr
        total_p1 += p1
        total_p5 += p5
        total_p10 += p10
        evaluated_queries += 1
    
    if evaluated_queries == 0:
        return None
    
    # Calculate averages
    map_score = total_map / evaluated_queries
    mrr_score = total_mrr / evaluated_queries
    p1_score = total_p1 / evaluated_queries
    p5_score = total_p5 / evaluated_queries
    p10_score = total_p10 / evaluated_queries
    
    return {
        'MAP': map_score,
        'MRR': mrr_score,
        'P@1': p1_score,
        'P@5': p5_score,
        'P@10': p10_score,
        'num_queries': evaluated_queries,
        'topk': topk
    }

def main():
    print("=== ColBERT Multi-TOPK Evaluation ===")
    print(f"TOPK values: {TOPK_VALUES}")
    print(f"Hardcoded test queries: {sorted(TEST_QUERIES)}")
    
    # Choose evaluation mode
    print("Evaluation options:")
    print("1) All queries")
    print("2) Test queries only (hardcoded)")
    eval_mode = input("Choose [1/2]: ").strip()
    
    if eval_mode == "2":
        print("Evaluating on test queries only...")
        qrels = load_qrels_from_relevant(QRELS_PATH, test_only=True)
        test_only = True
        print(f"Loaded {len(qrels)} test queries from Relevant.txt")
    else:
        print("Evaluating on all queries...")
        qrels = load_qrels_from_relevant(QRELS_PATH, test_only=False)
        test_only = False
        print(f"Loaded {len(qrels)} queries from Relevant.txt")
    
    # Find latest ColBERT ranking
    try:
        ranking_file = find_latest_colbert_ranking(COLBERT_EXPERIMENTS_DIR)
        print(f"Using ranking file: {ranking_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the experiments directory path")
        return
    
    # Load ColBERT rankings (no topk limit here)
    rankings = load_colbert_ranking(ranking_file, test_queries_only=test_only)
    
    if not rankings:
        print("No rankings loaded!")
        return
    
    # Evaluate at multiple TOPK values
    print(f"\n{'='*80}")
    print(f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}")
    print(f"{'='*80}")
    
    all_results = []
    
    for topk in TOPK_VALUES:
        results = evaluate_colbert_ranking_at_k(rankings, qrels, topk)
        
        if results:
            all_results.append(results)
            print(f"{results['topk']:<6} {results['num_queries']:<8} "
                  f"{results['MAP']:<8.4f} {results['MRR']:<8.4f} "
                  f"{results['P@1']:<8.4f} {results['P@5']:<8.4f} "
                  f"{results['P@10']:<8.4f}")
        else:
            print(f"{topk:<6} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
    
    print(f"{'='*80}")
    
    # Save results to file
    output_file = f"colbert_run/colbert_multi_topk_results_{'test' if test_only else 'all'}.txt"
    with open(output_file, 'w') as f:
        f.write("ColBERT Multi-TOPK Evaluation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}\n")
        f.write(f"{'='*80}\n")
        
        for result in all_results:
            f.write(f"{result['topk']:<6} {result['num_queries']:<8} "
                   f"{result['MAP']:<8.4f} {result['MRR']:<8.4f} "
                   f"{result['P@1']:<8.4f} {result['P@5']:<8.4f} "
                   f"{result['P@10']:<8.4f}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Compare with CNN
    print(f"\n=== For comparison, run CNN evaluation ===")
    if test_only:
        print("python colbert_run/eval_map_mrr_topk.py  # for test queries")
    else:
        print("python colbert_run/eval_all_queries.py   # for all queries")

if __name__ == "__main__":
    main()