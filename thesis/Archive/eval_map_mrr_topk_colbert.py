import os, json, argparse
from collections import defaultdict

# <-- Adapt these imports to match your retriever.py API -->
from retriever import load_index, encode_query, search_index  

# metrics
def load_qrels(path):
    qrels = defaultdict(set)
    with open(path) as f:
        for ln, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("//") or text.startswith("#"):
                continue
            qid, did = json.loads(text)
            qrels[qid].add(did)
    return qrels

def average_precision(ret, rel):
    hits = 0; sum_p = 0.0
    for i, d in enumerate(ret, start=1):
        if d in rel:
            hits += 1
            sum_p += hits / i
    return sum_p / len(rel) if rel else 0.0

def reciprocal_rank(ret, rel):
    for i, d in enumerate(ret, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0

def precision_at_1(ret, rel):
    return 1.0 if ret and ret[0] in rel else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_pairs", default="colbert_run/test_pairs.jsonl")
    p.add_argument("--index_dir",  required=True,
                   help="path to your ColBERT index directory")
    p.add_argument("--topk",       type=int, default=25)
    args = p.parse_args()

    # 1) qrels from test_pairs.jsonl
    qrels    = load_qrels(args.test_pairs)
    test_q   = sorted(qrels.keys())

    # 2) load ColBERT index
    index = load_index(args.index_dir)

    # 3) open output run
    out_run = "colbert_run/colbert_top%d.tsv" % args.topk
    fout    = open(out_run, "w")

    # 4) for each query, retrieve topk
    for qid in test_q:
        # assume you have a mapping from qid → query text,
        # e.g. load from colbert_run/queries.jsonl (not shown here)
        text = get_query_text(qid)             # you need to implement this
        qvec = encode_query(text)              # from retriever.py
        hits = search_index(index, qvec, k=args.topk)
        # hits is list of (docid, score)
        for rank, (did, score) in enumerate(hits, start=1):
            fout.write(f"{qid}\t{did}\t{rank}\t{score:.4f}\n")
    fout.close()
    print(f"Wrote ColBERT run to {out_run}")

    # 5) evaluate
    runs = defaultdict(list)
    with open(out_run) as f:
        for line in f:
            qid, did = line.split()[:2]
            runs[int(qid)].append(int(did))

    MAP = MRR = ACC = 0.0
    for qid in test_q:
        rel = qrels[qid]
        ret = runs[qid]
        MAP += average_precision(ret, rel)
        MRR += reciprocal_rank(ret, rel)
        ACC += precision_at_1(ret, rel)

    N = len(test_q)
    print(f"\nColBERT @ top{args.topk} → P@1: {ACC/N:.4f}, MAP: {MAP/N:.4f}, MRR: {MRR/N:.4f}")

if __name__=="__main__":
    main()