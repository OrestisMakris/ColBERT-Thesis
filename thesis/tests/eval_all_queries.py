import os, json, glob, torch
from collections import defaultdict
from model_cnn import SimpleCNN

# Config
MATS_DIR   = "padded_matrices_cnn"
CNN_MODEL  = "colbert_run/cnn_classifier_026_076.pt"
OUT_RUN    = "colbert_run/cnn_all_queries_top100.tsv"
QRELS_PATH = "CF_DataSet/Relevant.txt"
TOPK       = 100

def load_qrels_from_relevant(path):
    """Load qrels from Relevant.txt where line N = query N-1 (0-indexed)"""
    qrels = {}
    with open(path, 'r') as f:
        for qid, line in enumerate(f, start=0):  # Changed: start=0 instead of start=1
            docs = {int(d) for d in line.strip().split() if d}
            if docs:  # only include queries that have relevant docs
                qrels[qid] = docs
    return qrels

def get_all_query_ids(mats_dir):
    """Find all unique query IDs from matrix files in mats_dir"""
    qids = set()
    for fname in glob.glob(f"{mats_dir}/q*_d*.pt"):
        basename = os.path.basename(fname)
        qid = int(basename.split("_")[0][1:])  # extract qid from q{qid}_d{did}.pt
        qids.add(qid)
    return sorted(qids)

def score_all(qid, model, device, mats_dir, topk):
    """Score all documents for a given query and return top-k"""
    files = glob.glob(f"{mats_dir}/q{qid}_d*.pt")
    out = []
    for full in files:
        did = int(os.path.basename(full).split("_")[1][1:-3])
        mat = torch.load(full, map_location="cpu")
        if mat.ndim==4 and mat.shape[0]==1: mat = mat.squeeze(0)
        if mat.ndim==2: mat = mat.unsqueeze(0)
        mat = mat.unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(mat).item()
        out.append((did, score))
    return sorted(out, key=lambda x: x[1], reverse=True)[:topk]

def average_precision(ret, rel):
    hits=0; sum_p=0
    for i,d in enumerate(ret,1):
        if d in rel:
            hits+=1; sum_p += hits/i
    return sum_p/len(rel) if rel else 0

def reciprocal_rank(ret, rel):
    for i,d in enumerate(ret,1):
        if d in rel: return 1/i
    return 0

def precision_at_1(ret, rel):
    return 1.0 if ret and ret[0] in rel else 0.0

def main():
    print("Evaluating CNN on ALL queries...")
    
    # 1) Load qrels from Relevant.txt
    qrels = load_qrels_from_relevant(QRELS_PATH)
    print(f"Loaded qrels for {len(qrels)} queries from {QRELS_PATH}")
    
    # 2) Get all query IDs that have matrix files
    all_qids = get_all_query_ids(MATS_DIR)
    print(f"Found matrix files for {len(all_qids)} queries in {MATS_DIR}")
    
    # 3) Use intersection: only evaluate queries that have both qrels and matrices
    eval_qids = sorted(set(qrels.keys()) & set(all_qids))
    print(f"Will evaluate on {len(eval_qids)} queries (intersection of qrels and matrices)")
    
    if not eval_qids:
        print("No queries to evaluate! Check your qrels and matrix files.")
        return
    
    # 4) Load CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cnn = SimpleCNN().to(device).eval()
    cnn.load_state_dict(torch.load(CNN_MODEL, map_location=device))
    
    # 5) Score all queries and write run file
    print(f"Scoring and writing run to {OUT_RUN}...")
    with open(OUT_RUN, "w") as fout:
        for i, qid in enumerate(eval_qids, 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(eval_qids)} queries...")
            ranked = score_all(qid, cnn, device, MATS_DIR, TOPK)
            for rank, (did, score) in enumerate(ranked, 1):
                fout.write(f"{qid}\t{did}\t{rank}\t{score:.4f}\n")
    
    print(f"Wrote {OUT_RUN}")
    
    # 6) Evaluate metrics
    print("Computing metrics...")
    MAP = MRR = ACC = 0
    for qid in eval_qids:
        rel = qrels[qid]
        ret = [d for d, _ in score_all(qid, cnn, device, MATS_DIR, TOPK)]
        MAP += average_precision(ret, rel)
        MRR += reciprocal_rank(ret, rel)
        ACC += precision_at_1(ret, rel)
    
    N = len(eval_qids)
    print(f"\n=== CNN Results on {N} queries ===")
    print(f"P@1   = {ACC/N:.4f}")
    print(f"MAP   = {MAP/N:.4f}")
    print(f"MRR   = {MRR/N:.4f}")

if __name__=="__main__":
    main()