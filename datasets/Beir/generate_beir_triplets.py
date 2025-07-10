import os
import json
import random
from beir import util
import multiprocessing
from tqdm import tqdm

# --- Configuration ---
dataset_name = "fiqa"
# Download the dataset if not present
dataset_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
data_path = util.download_and_unzip(dataset_url, "fiqa")

output_dir = f"{dataset_name}_colbert_format"
os.makedirs(output_dir, exist_ok=True)

# Process all available splits for relevance information
qrels_splits_to_process = ["train", "dev", "test"]

# Global maps for worker processes
doc_to_int_id_map_global = {}
query_to_int_id_map_global = {}
all_int_doc_ids_global = []
corpus_global = {}
queries_global = {}
qrels_global = {}

def generate_triplets_for_query_worker(original_query_id):
    """
    Generates triplets for a single query using global maps.
    Input original_query_id is a string.
    Outputs triplets with integer IDs.
    """
    triplets = []
    # Use global variables within the worker
    # These globals are copies for each worker process
    relevant_docs_original_ids = qrels_global.get(original_query_id, {})
    
    int_query_id = query_to_int_id_map_global.get(original_query_id)

    if int_query_id is None or original_query_id not in queries_global:
        return triplets

    positive_int_doc_ids = []
    for original_doc_id, score in relevant_docs_original_ids.items():
        if score >= 1:
            # Check if original_doc_id is in the corpus (and thus in doc_to_int_id_map_global)
            if original_doc_id in corpus_global and original_doc_id in doc_to_int_id_map_global:
                positive_int_doc_ids.append(doc_to_int_id_map_global[original_doc_id])

    if not positive_int_doc_ids:
        return triplets

    for int_pos_doc_id in positive_int_doc_ids:
        # Ensure int_pos_doc_id is actually in the set of all known int_doc_ids
        # This check is mostly for safety, as it should be if derived from doc_to_int_id_map_global
        if int_pos_doc_id not in all_int_doc_ids_global:
            continue

        # Create a set of positive IDs for efficient lookup for this query's positives
        current_positive_set = set(positive_int_doc_ids)
        
        # Potential negatives are all integer doc IDs minus the positive integer doc IDs for THIS query
        potential_negatives = [doc_id for doc_id in all_int_doc_ids_global if doc_id not in current_positive_set]

        if not potential_negatives:
            continue

        num_negatives = min(3, len(potential_negatives)) # Generate up to 3 negatives
        # Ensure we don't sample a positive document as a negative (already handled by potential_negatives construction)
        # Also ensure negative is not same as positive
        valid_potential_negatives = [neg_id for neg_id in potential_negatives if neg_id != int_pos_doc_id]
        
        if not valid_potential_negatives:
            continue
        
        num_negatives = min(num_negatives, len(valid_potential_negatives))
        if num_negatives == 0:
            continue

        neg_int_doc_ids = random.sample(valid_potential_negatives, num_negatives)

        for int_neg_doc_id in neg_int_doc_ids:
            triplets.append([int_query_id, int_pos_doc_id, int_neg_doc_id])
            
    return triplets


print(f"Loading BEIR dataset: {dataset_name} from {data_path}")
data_path = os.path.abspath(data_path)
print(f"Debug: absolute data_path = {data_path}")

print("Attempting direct file loading...")

# Load corpus (original string IDs)
corpus_orig = {}
with open(os.path.join(data_path, "corpus.jsonl"), 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line.strip())
        corpus_orig[doc['_id']] = {"title": doc.get("title", ""), "text": doc["text"]}

# Load queries (original string IDs)
queries_orig = {}
with open(os.path.join(data_path, "queries.jsonl"), 'r', encoding='utf-8') as f:
    for line in f:
        query = json.loads(line.strip())
        queries_orig[query["_id"]] = query["text"]

# Load qrels from all specified splits (original string IDs for query and doc)
qrels_orig = {}
print(f"Loading qrels from splits: {qrels_splits_to_process}")

for split in qrels_splits_to_process:
    qrels_path = os.path.join(data_path, f"qrels/{split}.tsv")
    print(f"Debug: Loading qrels from {qrels_path}")

    try:
        with open(qrels_path, "r", encoding='utf-8') as f:
            header = next(f).strip().lower()
            if not (header.startswith("query-id") or header.startswith("query_id")):
                f.seek(0)
                print(f"Debug: No standard header found in {split}.tsv. Assuming no header.")
            else:
                print(f"Debug: Skipped header in {split}.tsv: {header}")

            for line_num, line in enumerate(f, 1):
                if line.strip():
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        # Assuming format: query-id corpus-id score (BEIR default)
                        # Or query_id Q0 doc_id score (TREC format) - need to adjust index for doc_id
                        qid_str, did_str, score_str = parts[0], parts[1], parts[2]
                        if len(parts) == 4 and parts[1].upper() == "Q0": # TREC format like
                            did_str = parts[2]
                            score_str = parts[3]
                        
                        score = int(score_str)
                        if qid_str not in qrels_orig:
                            qrels_orig[qid_str] = {}
                        # Merge relevance info. If a doc is relevant in multiple splits, the last score wins.
                        qrels_orig[qid_str][did_str] = score
                    else:
                        print(f"Warning: Skipping line {line_num} in {split}.tsv with fewer than 3 parts: {line.strip()}")
    except FileNotFoundError:
        print(f"Warning: qrels file not found at {qrels_path}. Skipping this split.")
    except Exception as e:
        print(f"Error loading qrels from {qrels_path}: {e}. Skipping this split.")

print("Finished loading all qrels splits.")

print("Direct loading successful!")

# Create integer mappings
doc_to_int_id_map = {doc_id: i for i, doc_id in enumerate(corpus_orig.keys())}
int_to_original_doc_id_map = {i: doc_id for doc_id, i in doc_to_int_id_map.items()}

query_to_int_id_map = {query_id: i for i, query_id in enumerate(queries_orig.keys())}
int_to_original_query_id_map = {i: query_id for query_id, i in query_to_int_id_map.items()}

all_int_doc_ids = list(doc_to_int_id_map.values())

# Assign to global variables for worker processes
doc_to_int_id_map_global.update(doc_to_int_id_map)
query_to_int_id_map_global.update(query_to_int_id_map)
all_int_doc_ids_global.extend(all_int_doc_ids)
corpus_global.update(corpus_orig)
queries_global.update(queries_orig)
qrels_global.update(qrels_orig)


print(
    f"Loaded {len(corpus_orig)} documents, {len(queries_orig)} queries, and qrels for {len(qrels_orig)} queries."
)
print(f"Mapped to {len(doc_to_int_id_map)} integer doc IDs and {len(query_to_int_id_map)} integer query IDs.")

# Generate CF dataset format files
print("Generating ColBERT-compatible files with integer IDs...")

#Generate docs.tsv (collection.tsv)
docs_file = os.path.join(output_dir, "docs.tsv")
with open(docs_file, "w", encoding="utf-8") as f:
    # Ensure consistent order if int_to_original_doc_id_map is used or iterate by sorted int IDs
    for int_doc_id in sorted(int_to_original_doc_id_map.keys()):
        original_doc_id = int_to_original_doc_id_map[int_doc_id]
        doc_data = corpus_orig[original_doc_id]
        full_text = f"{doc_data.get('title', '')} {doc_data['text']}".strip().replace('\t', ' ').replace('\n', ' ')
        f.write(f"{int_doc_id}\t{full_text}\n")

#Generate Queries.tsv
queries_file = os.path.join(output_dir, "Queries.tsv")
with open(queries_file, "w", encoding="utf-8") as f:
    for int_query_id in sorted(int_to_original_query_id_map.keys()):
        original_query_id = int_to_original_query_id_map[int_query_id]
        query_text = queries_orig[original_query_id].replace('\t', ' ').replace('\n', ' ')
        f.write(f"{int_query_id}\t{query_text}\n")

# Generate triplets.jsonl for training
triplets = []
print("Generating triplets (parallelized)…")
# Use original string query IDs from qrels for the worker
original_query_ids_for_triplets = [qid for qid in qrels_orig.keys() if qid in query_to_int_id_map]


if original_query_ids_for_triplets:
    # Use context manager for Pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        with tqdm(total=len(original_query_ids_for_triplets)) as pbar:
            for single_query_triplets in pool.imap_unordered(generate_triplets_for_query_worker, original_query_ids_for_triplets, chunksize=10):
                if single_query_triplets:
                    triplets.extend(single_query_triplets)
                pbar.update()
else:
    print("No query IDs from qrels found in the loaded queries. No triplets will be generated.")


print(
    f"Generated {len(triplets)} triplets using qrels from {qrels_splits_to_process} for positives."
)

# Save triplets
triplets_file = os.path.join(output_dir, "triplets.jsonl")
with open(triplets_file, "w", encoding='utf-8') as f:
    for triplet in triplets: # triplet already contains integer IDs
        f.write(json.dumps(triplet) + "\n")


relevant_file = os.path.join(output_dir, "Relevant.txt")
with open(relevant_file, "w", encoding='utf-8') as f:
    # Iterate in the order of integer query IDs
    for int_query_id in sorted(int_to_original_query_id_map.keys()):
        original_query_id = int_to_original_query_id_map[int_query_id]
        
        relevant_int_docs_for_query = []
        if original_query_id in qrels_orig:
            for original_doc_id, score in qrels_orig[original_query_id].items():
                if score >= 1 and original_doc_id in doc_to_int_id_map:
                    relevant_int_docs_for_query.append(str(doc_to_int_id_map[original_doc_id]))
        f.write(" ".join(relevant_int_docs_for_query) + "\n")

print(f"\nColBERT-compatible dataset created in '{output_dir}/':")
print(f"  - docs.tsv: {len(doc_to_int_id_map)} documents")
print(f"  - Queries.tsv: {len(query_to_int_id_map)} queries")
print(f"  - triplets.jsonl: {len(triplets)} training triplets")
print(f"  - Relevant.txt: relevance judgments (integer doc IDs, line-per-query)")

if triplets:
    print(f"\nExample triplet (integer IDs): {triplets[0]}")
    # Display original text for the example triplet
    example_int_qid, example_int_pos_id, example_int_neg_id = triplets[0]
    
    original_qid_example = int_to_original_query_id_map.get(example_int_qid)
    original_pos_doc_id_example = int_to_original_doc_id_map.get(example_int_pos_id)
    original_neg_doc_id_example = int_to_original_doc_id_map.get(example_int_neg_id)

    if original_qid_example and original_pos_doc_id_example and original_neg_doc_id_example:
        print(f"Query (Original ID: {original_qid_example}): {queries_orig[original_qid_example]}")
        print(f"Positive doc (Original ID: {original_pos_doc_id_example}): {corpus_orig[original_pos_doc_id_example]['text'][:100]}...")
        print(f"Negative doc (Original ID: {original_neg_doc_id_example}): {corpus_orig[original_neg_doc_id_example]['text'][:100]}...")
    else:
        print("Could not map example triplet IDs back to original IDs for display.")
else:
    print("No triplets generated. Check qrels, corpus, and queries.")

print(f"\nDataset is now ready for ColBERT training, indexing, and retrieval!")
print(f"Structure matches CF dataset format for compatibility.")