"""
Split triplets into train / validation / test sets (random negatives).

Differences from paper/similarity_matrices_proce/split_triplets.py:
  1. random.seed(42) for reproducibility
  2. NEGATIVES_PER_POSITIVE raised to 30 (was 10) to fix the "too few negatives" problem
  3. Output directed to the paper2/cfrun/ directory
"""
import os
import json
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_TRIPLETS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),   # paper2/cfrun/
    "..", "..",                                    # project root
    "scifact_colbert_format",
    "triplets.jsonl"
)

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # write next to this script

TRAIN_OUT      = os.path.join(_OUT_DIR, "train_triplets.jsonl")
VALIDATION_OUT = os.path.join(_OUT_DIR, "validation_triplets.jsonl")
TEST_OUT       = os.path.join(_OUT_DIR, "test_triplets.jsonl")

TRAIN_RATIO      = 0.9
VALIDATION_RATIO = 0.05
# TEST_RATIO = 1 - TRAIN_RATIO - VALIDATION_RATIO

NEGATIVES_PER_POSITIVE = 30   # was 10 in paper/; increased to fix under-sampling
MAX_QUERY = 2000
# ---------------------------------------------------------------------------


def main():
    random.seed(42)   # reproducible splits

    q_pos_to_negs: dict = defaultdict(list)
    all_qids: set = set()

    input_path = os.path.normpath(INPUT_TRIPLETS)
    print(f"Reading triplets from {input_path} …")

    with open(input_path) as fin:
        for line in fin:
            try:
                qid, pos_id, neg_id = json.loads(line)
                if int(qid) > MAX_QUERY:
                    continue
                q_pos_to_negs[(qid, pos_id)].append(neg_id)
                all_qids.add(qid)
            except (json.JSONDecodeError, ValueError):
                continue

    print(f"Found {len(all_qids)} unique queries, {len(q_pos_to_negs)} (q, pos) pairs.")

    # Shuffle then split by query ID (no data leakage across splits)
    all_qids_list = list(all_qids)
    random.shuffle(all_qids_list)

    n      = len(all_qids_list)
    t_end  = int(n * TRAIN_RATIO)
    v_end  = t_end + int(n * VALIDATION_RATIO)

    train_qids = set(all_qids_list[:t_end])
    val_qids   = set(all_qids_list[t_end:v_end])
    test_qids  = set(all_qids_list[v_end:])

    assert not (train_qids & val_qids)
    assert not (train_qids & test_qids)
    assert not (val_qids   & test_qids)

    print(f"Split → {len(train_qids)} train, {len(val_qids)} val, {len(test_qids)} test queries")

    def write_split(out_path, qid_set, tag):
        count = 0
        with open(out_path, "w") as fout:
            for (qid, pos_id), neg_ids in q_pos_to_negs.items():
                if qid not in qid_set:
                    continue
                k = min(NEGATIVES_PER_POSITIVE, len(neg_ids))
                for neg_id in random.sample(neg_ids, k=k):
                    fout.write(json.dumps([qid, pos_id, neg_id]) + "\n")
                    count += 1
        print(f"  [{tag}] {count} triplets → {out_path}")

    write_split(TRAIN_OUT,      train_qids, "train")
    write_split(VALIDATION_OUT, val_qids,   "val")
    write_split(TEST_OUT,       test_qids,  "test")

    print("Done.")


if __name__ == "__main__":
    main()
