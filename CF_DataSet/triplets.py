import json
import random

# Paths to the input files
queries_path = './Queries.tsv'
docs_path = './docs.tsv'
relevant_path = './Relevant.txt'
output_path = './triplets.jsonl'

# Step 1: Load queries and docs
def load_tsv(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                id_, text = line.split('\t', 1)
                data[int(id_)] = text
    return data

queries = load_tsv(queries_path)
docs = load_tsv(docs_path)
all_doc_ids = set(docs.keys())

# Step 2: Load relevant pids per query
relevant_pids = []
with open(relevant_path, 'r', encoding='utf-8') as f:
    for line in f:
        relevant_pids.append([int(pid) for pid in line.strip().split()])

# Step 3: Create JSONL triples
with open(output_path, 'w', encoding='utf-8') as out_file:
    for qid, positive_pids in enumerate(relevant_pids):
        for pid_pos in positive_pids:
            # Generate a negative pid by selecting a random doc that isn't in the relevant list
            negative_candidates = list(all_doc_ids - set(positive_pids))
            pid_neg = random.choice(negative_candidates)
            # Create triple as a list and write to output file
            triple = [qid, pid_pos, pid_neg]
            out_file.write(json.dumps(triple) + "\n")

print(f"Triples file saved to {output_path}")
