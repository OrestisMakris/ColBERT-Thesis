import json

def extract_unique_qids(jsonl_path):
    """Extract all unique query IDs from triplets JSONL file."""
    unique_qids = set()
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                qid = int(data[0])  # First element is query ID
                unique_qids.add(qid)
            except (json.JSONDecodeError, ValueError, IndexError):
                continue
    

    sorted_qids = sorted(unique_qids)
    

    print(f"\nTotal unique queries: {len(sorted_qids)}")

    print(f"\nComma-separated: {','.join(map(str, sorted_qids))}")
    
    return sorted_qids

if __name__ == "__main__":
    jsonl_file = "./paper2/cfrun/test_triplets_hard.jsonl"
    qids = extract_unique_qids(jsonl_file)