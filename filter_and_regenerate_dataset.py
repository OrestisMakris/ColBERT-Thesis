import os
import json
import random
import argparse
from tqdm import tqdm

def load_qrels(relevant_path):
    """Line N in Relevant.txt → query_id = str(N) → set of doc_ids."""
    qrels = {}
    with open(relevant_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            docs = line.strip().split()
            if docs:
                qrels[str(i)] = set(docs)
    return qrels

def filter_and_reindex_dataset(input_dir, output_dir, min_rel, neg_ratio):
    """
    Filters a dataset and re-indexes all query and document IDs to be sequential.
    """
    print("--- Starting Dataset Filtering and Re-indexing Process ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- File Paths ---
    docs_path = os.path.join(input_dir, "docs.tsv")
    queries_path = os.path.join(input_dir, "Queries.tsv")
    rels_path = os.path.join(input_dir, "Relevant.txt")

    # 1. Load original data
    print("Step 1: Loading original data...")
    original_docs = {line.split('\t', 1)[0]: line.split('\t', 1)[1].rstrip('\n')
                     for line in open(docs_path, 'r', encoding='utf-8')}
    original_queries = {line.split('\t', 1)[0]: line.split('\t', 1)[1].rstrip('\n')
                        for line in open(queries_path, 'r', encoding='utf-8')}
    original_qrels = load_qrels(rels_path)

    # 2. Identify which original queries and documents to keep
    print("Step 2: Identifying data to keep...")
    # Keep only queries with more than min_rel relevant documents
    qids_to_keep = {qid for qid, docs in original_qrels.items() if len(docs) > min_rel}
    print(f"Found {len(qids_to_keep)} queries with >{min_rel} relevant documents.")

    # Collect all documents relevant to those queries
    relevant_docs_to_keep = set().union(*(original_qrels[qid] for qid in qids_to_keep))
    
    # Identify all documents that were never relevant to any query in the original set
    all_original_relevant_docs = set().union(*original_qrels.values())
    never_relevant_docs = set(original_docs) - all_original_relevant_docs

    # Sample a fraction of the never-relevant documents
    num_to_sample = int(len(never_relevant_docs) * neg_ratio)
    sampled_negatives = set(random.sample(list(never_relevant_docs), num_to_sample))
    
    # Final set of documents to keep (using original IDs)
    docs_to_keep = relevant_docs_to_keep.union(sampled_negatives)
    print(f"Keeping {len(relevant_docs_to_keep)} relevant docs + {len(sampled_negatives)} sampled negative docs = {len(docs_to_keep)} total docs.")

    # 3. Create new, sequential ID mappings
    print("Step 3: Creating new sequential ID mappings...")
    # Sort old IDs numerically to ensure deterministic mapping
    sorted_qids_to_keep = sorted(list(qids_to_keep), key=int)
    sorted_dids_to_keep = sorted(list(docs_to_keep), key=int)

    old_q_to_new_q = {old_id: str(new_id) for new_id, old_id in enumerate(sorted_qids_to_keep)}
    old_d_to_new_d = {old_id: str(new_id) for new_id, old_id in enumerate(sorted_dids_to_keep)}

    # 4. Write new files using the new sequential IDs
    print("Step 4: Writing new re-indexed dataset files...")

    # Write new docs.tsv
    with open(os.path.join(output_dir, "docs.tsv"), "w", encoding='utf-8') as f:
        for old_did in tqdm(sorted_dids_to_keep, desc="Writing new docs.tsv"):
            new_did = old_d_to_new_d[old_did]
            text = original_docs[old_did]
            f.write(f"{new_did}\t{text}\n")

    # Write new Queries.tsv
    with open(os.path.join(output_dir, "Queries.tsv"), "w", encoding='utf-8') as f:
        for old_qid in tqdm(sorted_qids_to_keep, desc="Writing new Queries.tsv"):
            new_qid = old_q_to_new_q[old_qid]
            text = original_queries[old_qid]
            f.write(f"{new_qid}\t{text}\n")

    # Write new Relevant.txt
    with open(os.path.join(output_dir, "Relevant.txt"), "w", encoding='utf-8') as f:
        # Iterate up to the number of new queries
        for new_qid_int in tqdm(range(len(sorted_qids_to_keep)), desc="Writing new Relevant.txt"):
            old_qid = sorted_qids_to_keep[new_qid_int]
            
            # Get original relevant docs for this query
            old_relevant_dids = original_qrels.get(old_qid, set())
            
            # Map them to the new doc IDs, keeping only those that are in our final doc set
            new_relevant_dids = {old_d_to_new_d[did] for did in old_relevant_dids if did in old_d_to_new_d}
            
            f.write(" ".join(sorted(new_relevant_dids, key=int)) + "\n")

    # Write new triplets.jsonl
    with open(os.path.join(output_dir, "triplets.jsonl"), "w", encoding='utf-8') as f:
        all_new_doc_ids = set(old_d_to_new_d.values())
        for old_qid in tqdm(sorted_qids_to_keep, desc="Generating Triplets"):
            new_qid = old_q_to_new_q[old_qid]
            
            # Get positive documents with new IDs
            positive_new_dids = {old_d_to_new_d[did] for did in original_qrels.get(old_qid, set()) if did in old_d_to_new_d}
            
            if not positive_new_dids:
                continue

            # Negative documents are all other documents in our new collection
            negative_new_dids = list(all_new_doc_ids - positive_new_dids)
            
            if not negative_new_dids:
                continue

            for pos_new_did in positive_new_dids:
                # Sample up to 3 negatives per positive
                num_samples = min(3, len(negative_new_dids))
                sampled_negs = random.sample(negative_new_dids, k=num_samples)
                for neg_new_did in sampled_negs:
                    f.write(json.dumps([int(new_qid), int(pos_new_did), int(neg_new_did)]) + "\n")

    print(f"\n--- Process Complete! ---")
    print(f"Filtered and re-indexed dataset saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and re-index a ColBERT dataset.")
    parser.add_argument("--input_dir", type=str, default="./fiqa_colbert_format", help="Path to the original formatted dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the new filtered and re-indexed dataset.")
    parser.add_argument("--min_relevance", type=int, default=5, help="Minimum number of relevant documents a query must have to be kept (e.g., 5 means >5).")
    parser.add_argument("--neg_ratio", type=float, default=0.5, help="Fraction of never-relevant documents to keep.")
    
    args = parser.parse_args()
    filter_and_reindex_dataset(
        args.input_dir,
        args.output_dir,
        args.min_relevance,
        args.neg_ratio
    )