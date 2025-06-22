import os

# --- Configuration ---
# Set the path to your formatted dataset directory
DATASET_DIR = "./fiqa_colbert_format"

# --- File Paths ---
docs_path = os.path.join(DATASET_DIR, "docs.tsv")
queries_path = os.path.join(DATASET_DIR, "Queries.tsv")
relevant_path = os.path.join(DATASET_DIR, "Relevant.txt")

def analyze_dataset():
    """
    Performs a statistical analysis of the ColBERT-formatted dataset.
    """
    print(f"--- Analyzing Dataset at: {DATASET_DIR} ---\n")

    # Check if all required files exist
    for path in [docs_path, queries_path, relevant_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at '{path}'. Aborting.")
            return

    # --- Analysis 1: Documents not relevant to any query ---
    all_doc_ids = set()
    with open(docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Assumes format: doc_id\tdoc_text
            doc_id = line.strip().split('\t')[0]
            all_doc_ids.add(doc_id)

    all_relevant_doc_ids = set()
    with open(relevant_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Assumes space-separated doc IDs
            doc_ids_on_line = line.strip().split()
            for doc_id in doc_ids_on_line:
                all_relevant_doc_ids.add(doc_id)

    # Calculate the difference
    non_relevant_doc_ids = all_doc_ids - all_relevant_doc_ids
    
    print("--- Document Analysis ---")
    print(f"Total documents in collection: {len(all_doc_ids)}")
    print(f"Total unique documents marked as relevant: {len(all_relevant_doc_ids)}")
    print(f"Documents not relevant for ANY query: {len(non_relevant_doc_ids)}\n")


    # --- Analysis 2 & 3: Query analysis ---
    short_queries_count = 0
    many_relevant_queries_count = 0
    total_queries = 0

    # Analyze query lengths from Queries.tsv
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_queries += 1
            # Assumes format: query_id\tquery_text
            query_text = line.strip().split('\t')[1]
            word_count = len(query_text.split())
            if word_count < 4:
                short_queries_count += 1

    # Analyze relevance counts from Relevant.txt
    with open(relevant_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Filter out empty strings that can result from splitting an empty line
            relevant_docs_for_query = [doc_id for doc_id in line.strip().split() if doc_id]
            if len(relevant_docs_for_query) >5:
                many_relevant_queries_count += 1

    print("--- Query Analysis ---")
    print(f"Total queries: {total_queries}")
    print(f"Queries with fewer than 4 words: {short_queries_count}")
    print(f"Queries with more than 5 relevant documents: {many_relevant_queries_count}\n")
    print("--- Analysis Complete ---")


if __name__ == "__main__":
    analyze_dataset()