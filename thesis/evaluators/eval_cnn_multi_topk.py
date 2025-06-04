import os, json, glob, torch
from collections import defaultdict
from model_cnn import SimpleCNN

# Config
MATS_DIR = "padded_matrices_cnn"
CNN_MODEL = "colbert_run/cnn_classifier_v2.pt"
TEST_PAIRS = "colbert_run/test_pairs.jsonl"
QRELS_PATH = "CF_DataSet/Relevant.txt"
TOPK_VALUES = [5, 10, 20, 25, 40, 50, 80, 100, 150, 180, 200, 250]

# Hardcoded test queries from test_pairs.jsonl
#TEST_QUERIES = {1, 6, 18, 20, 23, 28, 30, 34, 35, 36, 37, 38, 45, 46, 49, 63, 75, 85, 94, 96}
TEST_QUERIES = {0, 1, 8, 10, 11, 14, 18, 22, 27, 34, 35, 36, 41, 42, 51, 57, 59, 79, 84, 94}
def load_qrels_from_relevant(path, test_only=False):
    """Load qrels from Relevant.txt where line N = query N-1 (0-indexed)"""
    qrels = {}
    with open(path, 'r') as f:
        for qid, line in enumerate(f, start=0):  # 0-indexed
            if test_only and qid not in TEST_QUERIES:
                continue  # Skip non-test queries
            
            relevant_docs = set()
            for doc_str in line.strip().split():
                if doc_str.isdigit():
                    relevant_docs.add(int(doc_str))
            if relevant_docs:
                qrels[qid] = relevant_docs
    return qrels

def load_qrels_from_test_pairs(path):
    """Load qrels from test_pairs.jsonl (alternative approach)"""
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            qid, did = json.loads(line)
            qrels[qid].add(did)
    return qrels

def get_all_query_ids(mats_dir):
    """Find all unique query IDs from matrix files in mats_dir"""
    qids = set()
    for fname in glob.glob(f"{mats_dir}/q*_d*.pt"):
        basename = os.path.basename(fname)
        qid = int(basename.split("_")[0][1:])  # extract qid from q{qid}_d{did}.pt
        qids.add(qid)
    return sorted(qids)

def score_all_for_query(qid, model, device, mats_dir, topk):
    """Score all documents for a given query and return top-k"""
    files = glob.glob(f"{mats_dir}/q{qid}_d*.pt")
    scores = []
    
    for full in files:
        did = int(os.path.basename(full).split("_")[1][1:-3])
        mat = torch.load(full, map_location="cpu")
        
        # Handle dimensions
        if mat.ndim == 4 and mat.shape[0] == 1:
            mat = mat.squeeze(0)
        if mat.ndim == 2:
            mat = mat.unsqueeze(0)
        
        mat = mat.unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(mat).item()
        
        scores.append((did, score))
    
    # Return top-k sorted by score (descending)
    return sorted(scores, key=lambda x: x[1], reverse=True)[:topk]

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

def evaluate_cnn_at_k(model, device, qrels, mats_dir, topk):
    """Evaluate CNN at specific topk"""
    total_map = 0.0
    total_mrr = 0.0
    total_p1 = 0.0
    total_p5 = 0.0
    total_p10 = 0.0
    
    evaluated_queries = 0
    
    for qid in qrels:
        # Check if we have matrix files for this query
        query_files = glob.glob(f"{mats_dir}/q{qid}_d*.pt")
        if not query_files:
            continue
        
        # Get top-k retrieved documents
        ranked_results = score_all_for_query(qid, model, device, mats_dir, topk)
        retrieved_docs = [did for did, score in ranked_results]
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
    print("=== CNN Multi-TOPK Evaluation ===")
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
    
    # Get all query IDs that have matrix files
    all_qids = get_all_query_ids(MATS_DIR)
    print(f"Found matrix files for {len(all_qids)} queries in {MATS_DIR}")
    
    # Use intersection: only evaluate queries that have both qrels and matrices
    eval_qids = set(qrels.keys()) & set(all_qids)
    print(f"Will evaluate on {len(eval_qids)} queries (intersection of qrels and matrices)")
    
    if not eval_qids:
        print("No queries to evaluate! Check your qrels and matrix files.")
        return
    
    # Load CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cnn = SimpleCNN().to(device).eval()
    cnn.load_state_dict(torch.load(CNN_MODEL, map_location=device))
    print(f"Loaded CNN model from: {CNN_MODEL}")
    
    # Evaluate at multiple TOPK values
    print(f"\n{'='*80}")
    print(f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}")
    print(f"{'='*80}")
    
    all_results = []
    
    for topk in TOPK_VALUES:
        print(f"Evaluating at TOPK={topk}...", end=" ")
        results = evaluate_cnn_at_k(cnn, device, qrels, MATS_DIR, topk)
        
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
    output_file = f"colbert_run/cnn_multi_topk_results_{'test' if test_only else 'all'}.txt"
    with open(output_file, 'w') as f:
        f.write("CNN Multi-TOPK Evaluation Results\n")
        f.write(f"Model: {CNN_MODEL}\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}\n")
        f.write(f"{'='*80}\n")
        
        for result in all_results:
            f.write(f"{result['topk']:<6} {result['num_queries']:<8} "
                   f"{result['MAP']:<8.4f} {result['MRR']:<8.4f} "
                   f"{result['P@1']:<8.4f} {result['P@5']:<8.4f} "
                   f"{result['P@10']:<8.4f}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate TSV run file for the highest TOPK
    if all_results:
        max_topk = max(TOPK_VALUES)
        run_file = f"colbert_run/cnn_multi_topk_run_{'test' if test_only else 'all'}.tsv"
        print(f"Generating run file at TOPK={max_topk}: {run_file}")
        
        with open(run_file, "w") as fout:
            for qid in sorted(eval_qids):
                ranked = score_all_for_query(qid, cnn, device, MATS_DIR, max_topk)
                for rank, (did, score) in enumerate(ranked, 1):
                    fout.write(f"{qid}\t{did}\t{rank}\t{score:.4f}\n")
        
        print(f"Run file saved to: {run_file}")

if __name__ == "__main__":
    main()

# def main():
#     parser = argparse.ArgumentParser(description="Eval CNN at multiple TOPK")
#     parser.add_argument("--model",      required=True,
#                         help="path to your .pt model file")
#     parser.add_argument("--topk",       required=True,
#                         help="comma‐sep list of K values, e.g. 5,10,25")
#     parser.add_argument("--json",       dest="out_json", required=True,
#                         help="where to write JSON results")
#     parser.add_argument("--test_only",  action="store_true",
#                         help="if set, evaluate only hardcoded test queries")
#     parser.add_argument("--qrels_txt",  default="CF_DataSet/Relevant.txt",
#                         help="path to Relevant.txt")
#     parser.add_argument("--mats_dir",   default="padded_matrices_cnn",
#                         help="where your q*_d*.pt matrices live")
#     args = parser.parse_args()

# # load qrels
#     if args.test_only:
#         qrels = load_qrels_from_relevant(args.qrels_txt, test_only=True)
#     else:
#         qrels = load_qrels_from_relevant(args.qrels_txt, test_only=False)

#     # load model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cnn = SimpleCNN().to(device)
#     cnn.load_state_dict(torch.load(args.model, map_location=device))
#     cnn.eval()

#     topk_list = [int(x) for x in args.topk.split(",")]
#     results   = {}

#     for k in topk_list:
#         stats = evaluate_cnn_at_k(cnn, device, qrels, args.mats_dir, k)
#         if stats is None:
#             results[str(k)] = { "MAP": 0.0, "MRR": 0.0 }
#         else:
#             results[str(k)] = {
#                 "MAP": stats["MAP"],
#                 "MRR": stats["MRR"]
#             }
#         print(f"TOPK={k} → MAP={results[str(k)]['MAP']:.4f}, "
#               f"MRR={results[str(k)]['MRR']:.4f}")
        
#     # dump JSON
#     with open(args.out_json, "w") as out:
#         json.dump(results, out, indent=2)

# if __name__ == "__main__":
#     main()