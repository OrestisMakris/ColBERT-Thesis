import os
import json
import random
from collections import defaultdict

# --- Configuration ---
INPUT_TRIPLETS = os.path.join(
    os.getcwd(),
    "CF_DataSet",
    "triplets.jsonl"
)
TRAIN_OUT = os.path.join(
    os.getcwd(),
    "colbert_run",
    "train_triplets.jsonl"
)
TEST_OUT = os.path.join(
    os.getcwd(),
    "colbert_run",
    "test_pairs.jsonl"
)
SPLIT_RATIO = 0.8


def main():
    #    q2triplets[qid] = list of (pos_id, neg_id) pairs
    q2triplets = defaultdict(list)
    with open(INPUT_TRIPLETS, "r") as fin:
        for line in fin:
            qid, pos_id, neg_id = json.loads(line)
            q2triplets[qid].append((pos_id, neg_id))


    all_qids = list(q2triplets.keys())
    random.shuffle(all_qids)
    split_point = int(len(all_qids) * SPLIT_RATIO)
    train_qids = set(all_qids[:split_point])
    test_qids  = set(all_qids[split_point:])

    #    Each line: [qid, pos_id, neg_id]
    with open(TRAIN_OUT, "w") as fout:
        for qid in train_qids:
            for pos_id, neg_id in q2triplets[qid]:
                fout.write(json.dumps([qid, pos_id, neg_id]) + "\n")


    #    Each line: [qid, pos_id]
    with open(TEST_OUT, "w") as fout:
        for qid in test_qids:
            for pos_id, _ in q2triplets[qid]:
                fout.write(json.dumps([qid, pos_id]) + "\n")

    print(f"Split {len(all_qids)} queries → "
          f"{len(train_qids)} train, {len(test_qids)} test")
    print(f"Train triplets saved to: {TRAIN_OUT}")
    print(f"Test pairs saved to:  {TEST_OUT}")

if __name__ == "__main__":
    main()