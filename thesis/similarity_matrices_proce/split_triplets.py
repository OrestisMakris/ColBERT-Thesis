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
# --- NEW: Updated output filenames ---
TRAIN_OUT = os.path.join(
    os.getcwd(),
    "thesis/cf_run_Mediumtuned",
    "train_triplets_60.jsonl"
)
VALIDATION_OUT = os.path.join(
    os.getcwd(),
    "thesis/cf_run_Mediumtuned",
    "validation_triplets_60.jsonl" # Changed from _pairs to _triplets
)
TEST_OUT = os.path.join(
    os.getcwd(),
    "thesis/cf_run_Mediumtuned",
    "test_triplets_60.jsonl" # Changed from _pairs to _triplets
)

TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.1
# TEST_RATIO is implicitly (1.0 - TRAIN_RATIO - VALIDATION_RATIO)

NEGATIVES_PER_POSITIVE = 1 # Create this many training examples for each (q, p) pair
MAX_QUERY = 140

def main():
    # Ensure the output directory exists
    output_dir = os.path.dirname(TRAIN_OUT)
    os.makedirs(output_dir, exist_ok=True)

    q_pos_to_negs = defaultdict(list)
    all_qids = set()

    print(f"Reading triplets from {INPUT_TRIPLETS}...")
    with open(INPUT_TRIPLETS, "r") as fin:
        for line in fin:
            qid, pos_id, neg_id = json.loads(line)
            if int(qid) > MAX_QUERY:
                continue
            q_pos_to_negs[(qid, pos_id)].append(neg_id)
            all_qids.add(qid)

    print(f"Found {len(all_qids)} unique queries and {len(q_pos_to_negs)} unique (query, positive_doc) pairs.")

    # --- NEW: Shuffle QIDs and split into distinct sets ---
    all_qids_list = list(all_qids)

    random.shuffle(all_qids_list)
    
    train_split_point = int(len(all_qids_list) * TRAIN_RATIO)
    validation_split_point = train_split_point + int(len(all_qids_list) * VALIDATION_RATIO)

    train_qids = set(all_qids_list[:train_split_point])
    val_qids = set(all_qids_list[train_split_point:validation_split_point])
    test_qids = set(all_qids_list[validation_split_point:])

    # --- Assert that sets are disjoint ---
    assert not (train_qids & val_qids)
    assert not (train_qids & test_qids)
    assert not (val_qids & test_qids)

    # --- Write Training Triplets ---
    train_triplets_count = 0
    with open(TRAIN_OUT, "w") as fout:
        for (qid, pos_id), neg_ids in q_pos_to_negs.items():
            if qid in train_qids:
                num_to_sample = min(NEGATIVES_PER_POSITIVE, len(neg_ids))
                for neg_id in random.sample(neg_ids, k=num_to_sample):
                    fout.write(json.dumps([qid, pos_id, neg_id]) + "\n")
                    train_triplets_count += 1

    # --- MODIFIED: Write Validation Triplets ---
    val_triplets_count = 0
    with open(VALIDATION_OUT, "w") as fout:
        for (qid, pos_id), neg_ids in q_pos_to_negs.items():
            if qid in val_qids:
                # For validation, let's also sample negatives to create a balanced set
                num_to_sample = min(NEGATIVES_PER_POSITIVE, len(neg_ids))
                for neg_id in random.sample(neg_ids, k=num_to_sample):
                    fout.write(json.dumps([qid, pos_id, neg_id]) + "\n")
                    val_triplets_count += 1

    # --- MODIFIED: Write Test Triplets ---
    test_triplets_count = 0
    with open(TEST_OUT, "w") as fout:
        for (qid, pos_id), neg_ids in q_pos_to_negs.items():
            if qid in test_qids:
                num_to_sample = min(NEGATIVES_PER_POSITIVE, len(neg_ids))
                for neg_id in random.sample(neg_ids, k=num_to_sample):
                    fout.write(json.dumps([qid, pos_id, neg_id]) + "\n")
                    test_triplets_count += 1

    print("\n--- Split Summary ---")
    print(f"Split {len(all_qids)} queries -> {len(train_qids)} train, {len(val_qids)} validation, {len(test_qids)} test")
    print(f"Wrote {train_triplets_count} triplets to training file: {TRAIN_OUT}")
    print(f"Wrote {val_triplets_count} triplets to validation file: {VALIDATION_OUT}")
    print(f"Wrote {test_triplets_count} triplets to test file:  {TEST_OUT}")

if __name__ == "__main__":
    main()