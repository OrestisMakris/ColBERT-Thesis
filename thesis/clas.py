import json, os, random, glob
from collections import defaultdict

IN = os.path.join(os.getcwd(), "colbert_run", "train_triplets.jsonl")
RELEVANT_TXT = os.path.join(os.getcwd(), "CF_DataSet", "Relevant.txt")
MATRICES_DIR = os.path.join(os.getcwd(), "padded_matrices_cnn")
OUT = os.path.join(os.getcwd(), "colbert_run", "train_data_balanced.jsonl")

def load_qrels():
    """Load relevance judgments from Relevant.txt"""
    qrels = {}
    with open(RELEVANT_TXT, 'r') as f:
        for qid, line in enumerate(f):
            docs = {int(d) for d in line.strip().split() if d.isdigit()}
            if docs:
                qrels[qid] = docs
    return qrels

def get_available_docs():
    """Get available (qid, did) pairs from matrix files"""
    available = defaultdict(set)
    total_files = 0
    for file in glob.glob(f"{MATRICES_DIR}/q*_d*.pt"):
        total_files += 1
        parts = os.path.basename(file).replace('.pt', '').split('_')
        if len(parts) == 2:
            qid = int(parts[0][1:])
            did = int(parts[1][1:])
            available[qid].add(did)
    
    print(f"Found {total_files} matrix files across {len(available)} queries")
    return available, total_files

def create_balanced_from_triplets():
    """Create balanced dataset using existing triplets + additional negatives"""
    qrels = load_qrels()
    available_docs, total_files = get_available_docs()
    
    # First, load existing triplets
    triplet_samples = []
    with open(IN) as fin:
        for line in fin:
            qid, pos_id, neg_id = json.loads(line)
            triplet_samples.append({"matrix_file": f"q{qid}_d{pos_id}.pt", "label": 1})
            triplet_samples.append({"matrix_file": f"q{qid}_d{neg_id}.pt", "label": 0})
    
    print(f"Loaded {len(triplet_samples)} samples from existing triplets")
    
    # Count positives from triplets
    triplet_positives = sum(1 for s in triplet_samples if s["label"] == 1)
    
    # Add additional negatives to reach 1:4 ratio
    additional_negatives = []
    target_negatives = triplet_positives *1
    current_negatives = len(triplet_samples) - triplet_positives
    needed_negatives = target_negatives - current_negatives
    
    print(f"Need {needed_negatives} additional negatives for 1:4 ratio")
    
    # Get existing negative docs to avoid duplicates
    existing_negatives = {s["matrix_file"] for s in triplet_samples if s["label"] == 0}
    
    for qid in qrels:
        if needed_negatives <= 0:
            break
            
        query_docs = available_docs.get(qid, set())
        relevant = qrels[qid]
        irrelevant = query_docs - relevant
        
        # Remove already used negatives
        available_negatives = []
        for did in irrelevant:
            neg_file = f"q{qid}_d{did}.pt"
            if neg_file not in existing_negatives:
                available_negatives.append(neg_file)
        
        # Sample additional negatives
        sample_size = min(needed_negatives, len(available_negatives))
        if sample_size > 0:
            sampled = random.sample(available_negatives, sample_size)
            for neg_file in sampled:
                additional_negatives.append({"matrix_file": neg_file, "label": 0})
                existing_negatives.add(neg_file)
            needed_negatives -= sample_size
    
    # Combine all samples
    all_samples = triplet_samples + additional_negatives
    random.shuffle(all_samples)
    
    return all_samples, total_files

def main():
    print("Creating balanced dataset from existing triplets...")
    samples, total_files = create_balanced_from_triplets()
    
    positives = sum(1 for s in samples if s["label"] == 1)
    negatives = len(samples) - positives
    ratio = negatives / positives if positives > 0 else 0
    
    print(f"Total samples: {len(samples)}")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")
    print(f"Ratio: 1:{ratio:.1f}")
    
    with open(OUT, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Written to {OUT}")

if __name__ == "__main__":
    random.seed(42)
    main()