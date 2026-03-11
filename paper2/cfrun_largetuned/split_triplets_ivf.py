"""
Split triplets into train / validation / test using hard IVF negatives.

Same logic as paper/similarity_matrices_proce/split_triplets_ivf.py,
output directed to paper2/cfrun/.
"""
import os
import json
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CANDIDATES_DIR = os.path.join(_ROOT, "ivf_candidates_cf18_mediumtuned")
QRELS_FILE     = os.path.join(_ROOT, "CF_DataSet", "triplets.jsonl")

_OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_OUT = os.path.join(_OUT_DIR, "train_triplets_hard.jsonl")
VAL_OUT   = os.path.join(_OUT_DIR, "validation_triplets_hard.jsonl")
TEST_OUT  = os.path.join(_OUT_DIR, "test_triplets_hard.jsonl")

TRAIN_RATIO      = 0.8
VALIDATION_RATIO = 0.05

NEGATIVES_PER_POSITIVE = 1   # how many hard negatives per positive doc
# ---------------------------------------------------------------------------


def load_qrels(path: str):
    """Load positive documents for each query from triplets file."""
    qrels    = defaultdict(set)
    all_qids = set()
    with open(path) as f:
        for line in f:
            try:
                data   = json.loads(line)
                qid    = int(data[0])
                pos_id = int(data[1])
                qrels[qid].add(pos_id)
                all_qids.add(qid)
            except Exception:
                continue
    return qrels, list(all_qids)


def load_candidates(qid: int) -> list:
    """Load IVF candidates for a specific query."""
    path = os.path.join(CANDIDATES_DIR, f"q{qid}", f"q{qid}_ivf_candidates.txt")
    candidates = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(int(line))
    return candidates


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)

    print(f"Loading qrels from {QRELS_FILE} …")
    qrels, all_qids = load_qrels(QRELS_FILE)

    # Shuffle and split by query ID
    random.seed(42)
    random.shuffle(all_qids)

    n_train = int(len(all_qids) * TRAIN_RATIO)
    n_val   = int(len(all_qids) * VALIDATION_RATIO)

    train_qids = set(all_qids[:n_train])
    val_qids   = set(all_qids[n_train:n_train + n_val])
    test_qids  = set(all_qids[n_train + n_val:])

    print(f"Split → {len(train_qids)} train, {len(val_qids)} val, {len(test_qids)} test queries")

    stats = {"train": 0, "val": 0, "test": 0}

    # ---- Training ----
    with open(TRAIN_OUT, "w") as f:
        for qid in train_qids:
            candidates     = load_candidates(qid)
            cand_set       = set(candidates)
            positives      = [d for d in qrels[qid] if d in cand_set]
            hard_negatives = [d for d in candidates if d not in qrels[qid]]

            if not positives or not hard_negatives:
                continue

            for pos_id in positives:
                negs = random.sample(hard_negatives, min(len(hard_negatives), NEGATIVES_PER_POSITIVE))
                for neg_id in negs:
                    f.write(json.dumps([qid, pos_id, neg_id]) + "\n")
                    stats["train"] += 1

    # ---- Validation ----
    with open(VAL_OUT, "w") as f:
        for qid in val_qids:
            candidates     = load_candidates(qid)
            cand_set       = set(candidates)
            positives      = [d for d in qrels[qid] if d in cand_set]
            hard_negatives = [d for d in candidates if d not in qrels[qid]]

            if not positives or not hard_negatives:
                continue

            for pos_id in positives:
                negs = random.sample(hard_negatives, min(len(hard_negatives), NEGATIVES_PER_POSITIVE))
                for neg_id in negs:
                    f.write(json.dumps([qid, pos_id, neg_id]) + "\n")
                    stats["val"] += 1

    # ---- Test (qid + pos only — eval script handles candidates) ----
    with open(TEST_OUT, "w") as f:
        for qid in test_qids:
            for pos_id in qrels[qid]:
                f.write(json.dumps([qid, pos_id]) + "\n")
                stats["test"] += 1

    print(f"\nDone! Stats: {stats}")
    print(f"  train → {TRAIN_OUT}")
    print(f"  val   → {VAL_OUT}")
    print(f"  test  → {TEST_OUT}")


if __name__ == "__main__":
    main()
