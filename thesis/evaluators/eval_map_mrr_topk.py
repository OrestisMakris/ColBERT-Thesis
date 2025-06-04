import os, json, glob, torch
from collections import defaultdict
from model_cnn import SimpleCNN

TEST_PAIRS = "colbert_run/test_pairs.jsonl"
MATS_DIR   = "padded_matrices_cnn"
CNN_MODEL  = "colbert_run/cnn_classifierrr.pt"
OUT_RUN    = "colbert_run/cnn_full_top254.tsv"
TOPK       = 100

def load_qrels(path):
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            qid, did = json.loads(line)
            qrels[qid].add(did)
    return qrels

def score_all(qid, model, device):
    # find every q{qid}_d*.pt
    files = glob.glob(f"{MATS_DIR}/q{qid}_d*.pt")
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
    return sorted(out, key=lambda x: x[1], reverse=True)[:TOPK]

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
    # 1) load qrels (only positives)
    qrels = load_qrels(TEST_PAIRS)
    test_q = sorted(qrels.keys())

    # 2) load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = SimpleCNN().to(device).eval()
    cnn.load_state_dict(torch.load(CNN_MODEL, map_location=device))

    # 3) score & write run
    with open(OUT_RUN, "w") as fout:
        for qid in test_q:
            ranked = score_all(qid, cnn, device)
            for rank,(did,score) in enumerate(ranked,1):
                fout.write(f"{qid}\t{did}\t{rank}\t{score:.4f}\n")
    print(f"Wrote {OUT_RUN}")

    # 4) evaluate
    MAP=MRR=ACC=0
    for qid in test_q:
        rel = qrels[qid]
        ret = [d for d,_ in score_all(qid, cnn, device)]
        MAP += average_precision(ret, rel)
        MRR += reciprocal_rank(ret, rel)
        ACC += precision_at_1(ret, rel)
    N = len(test_q)
    print(f"P@1   = {ACC/N:.4f}")
    print(f"MAP   = {MAP/N:.4f}")
    print(f"MRR   = {MRR/N:.4f}")

if __name__=="__main__":
    main()