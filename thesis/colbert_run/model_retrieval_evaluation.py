import argparse
import glob
from collections import defaultdict

def load_relevance(relevant_path):
    """
    Load relevance judgments: each line = a query's relevant doc IDs (space-separated).
    """
    qrels = {}
    with open(relevant_path, 'r') as f:
        for qid, line in enumerate(f):
            docs = list(map(int, line.strip().split()))
            qrels[qid] = set(docs)
    return qrels


def load_runs(run_path):
    """
    Load run file: qid, docid, [rank, score].
    Accepts TSV or space-delimited.
    Returns ordered list of retrieved docs per query.
    """
    runs = defaultdict(list)
    with open(run_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # assume first two columns are qid and docid
            qid, docid = int(parts[0]), int(parts[1])
            runs[qid].append(docid)
    return runs


def precision_at_k(retrieved, relevant, k):
    topk = retrieved[:k]
    return sum(1 for d in topk if d in relevant) / k if k > 0 else 0.0


def average_precision(retrieved, relevant):
    """Average precision: sum of precision at each relevant doc position"""
    hits = 0
    sum_prec = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant)


def reciprocal_rank(retrieved, relevant):
    """Reciprocal rank: 1 / rank of first relevant doc (0 if none)"""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def precision_mean(retrieved, relevant):
    """Precision over full retrieved list (all docs)"""
    if not retrieved:
        return 0.0
    return sum(1 for d in retrieved if d in relevant) / len(retrieved)


def evaluate(qrels, runs, ks=[1,5,10,20]):
    metrics = {'P@k': {k: [] for k in ks}, 'AP': [], 'RR': [], 'PrecMean': []}
    for qid, relevant in qrels.items():
        retrieved = runs.get(qid, [])
        for k in ks:
            metrics['P@k'][k].append(precision_at_k(retrieved, relevant, k))
        metrics['AP'].append(average_precision(retrieved, relevant))
        metrics['RR'].append(reciprocal_rank(retrieved, relevant))
        metrics['PrecMean'].append(precision_mean(retrieved, relevant))

    mean_p = {k: sum(v)/len(v) for k, v in metrics['P@k'].items()}
    map_score = sum(metrics['AP']) / len(metrics['AP'])
    mrr_score = sum(metrics['RR']) / len(metrics['RR'])
    mean_prec = sum(metrics['PrecMean']) / len(metrics['PrecMean'])
    return mean_p, map_score, mrr_score, mean_prec


def find_default_run():
    """
    Auto-detect run file under ./experiments/**/CF1.nbits=1.ranking.tsv
    """
    #files = glob.glob('./experiments/CF1/retriever/2025-04/29/12.54.49/*.ranking.tsv', recursive=True)
    files = glob.glob('./cnn_pairwise.ranking.tsv', recursive=True)
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(description='Evaluate IR run: P@K, MAP, MRR, mean precision')
    parser.add_argument('--qrels', default='../CF_DataSet/Relevant.txt',
                        help='Path to relevant.txt (default: ../CF_DataSet/Relevant.txt)')
    parser.add_argument('--run', default=None,
                        help='Path to run file (default: auto-detected)')
    parser.add_argument('--ks', nargs='+', type=int, default=[1,5,10,20],
                        help='Cutoff K values for Precision@K')
    parser.add_argument('--output', help='File path to write results (defaults to stdout)')
    args = parser.parse_args()

    qrels_path = args.qrels
    run_path = args.run or find_default_run()
    if not run_path:
        raise FileNotFoundError("Run file not specified and no file auto-detected under ./experiments.")

    print(f"Using qrels: {qrels_path}")
    print(f"Using run file: {run_path}")

    qrels = load_relevance(qrels_path)
    runs = load_runs(run_path)
    mean_p, map_score, mrr_score, mean_prec = evaluate(qrels, runs, ks=args.ks)

    lines = ["Precision@K:"]
    for k, v in mean_p.items():
        lines.append(f"  P@{k}: {v:.4f}")
    lines.append(f"Mean Precision (full lists): {mean_prec:.4f}")
    lines.append(f"Mean Average Precision (MAP): {map_score:.4f}")
    lines.append(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
    output_str = "\n".join(lines)

    if args.output:
        with open(args.output, 'w') as out:
            out.write(output_str)
        print(f"Results written to {args.output}")
    else:
        print(output_str)

if __name__ == '__main__':
    main()
