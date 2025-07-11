import os
import json
# import argparse # Removed
import time
import glob
import numpy as np
import torch # For loading .pt files
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib # For loading sklearn models

def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def count_docs_per_query(mats_dir, test_jsonl_path):
    qids = set()
    # Assuming test_jsonl_path is relative to project root
    if not os.path.exists(test_jsonl_path):
        print_message(f"Warning: Test JSONL for counting docs not found at {test_jsonl_path}")
        return

    with open(test_jsonl_path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if isinstance(rec, list) and len(rec) == 2:
                    qids.add(rec[0])
            except json.JSONDecodeError:
                continue # Skip malformed lines silently for counting
    
    # Assuming mats_dir is relative to project root
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    absolute_mats_dir = os.path.join(project_root, mats_dir)

    if not os.path.isdir(absolute_mats_dir):
        print_message(f"Warning: Matrices directory for counting not found: {absolute_mats_dir}")
        return

    print_message("Document counts per query (from available matrices):")
    for qid in sorted(list(qids)):
        pattern = os.path.join(absolute_mats_dir, f"q{qid}_d*.pt")
        files = glob.glob(pattern)
        print_message(f"  Query {qid}: {len(files)} docs")

class AllPairsLogRegTestDataset:
    def __init__(self, test_jsonl_path, mats_dir):
        # Assuming test_jsonl_path is relative to project root
        # Assuming mats_dir is relative to project root
        self.test_jsonl_path = test_jsonl_path
        self.mats_dir = mats_dir
        
        script_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(script_dir)
        self.absolute_mats_root = os.path.join(project_root, self.mats_dir)

        if not os.path.exists(self.test_jsonl_path):
            raise FileNotFoundError(f"Test JSONL file not found: {self.test_jsonl_path}")
        if not os.path.isdir(self.absolute_mats_root):
            raise FileNotFoundError(f"Matrices directory not found: {self.absolute_mats_root}")

        qrels = {}
        with open(self.test_jsonl_path, "r") as f:
            for line in f:
                try:
                    qid, did = json.loads(line)
                    qrels.setdefault(qid, set()).add(did)
                except json.JSONDecodeError:
                    print_message(f"Skipping malformed line in {self.test_jsonl_path}: {line.strip()}")
                    continue
                except ValueError:
                    print_message(f"Skipping line with unexpected format in {self.test_jsonl_path}: {line.strip()}")
                    continue

        self.samples_info = []
        print_message(f"Processing matrices for {len(qrels)} queries specified in {self.test_jsonl_path}.")
        for qid, pos_dids_set in qrels.items():
            pattern = os.path.join(self.absolute_mats_root, f"q{qid}_d*.pt")
            found_matrices_for_query = False
            for mat_path in glob.glob(pattern):
                found_matrices_for_query = True
                fname = os.path.basename(mat_path)
                try:
                    did_str = fname.split("_d")[1].split(".pt")[0]
                    did = int(did_str)
                    label = 1 if did in pos_dids_set else 0
                    self.samples_info.append({'path': mat_path, 'label': label, 'fname': fname})
                except Exception as e:
                    print_message(f"Could not parse doc ID from {fname}: {e}")
            if not found_matrices_for_query:
                 print_message(f"Warning: No matrices found for query {qid} with pattern {pattern}")
        
        if not self.samples_info:
            print_message("Warning: No samples were loaded. Check mats_dir and test_jsonl content.")
        print_message(f"Initialized dataset with {len(self.samples_info)} total q-d pairs.")

    def load_all_data_for_eval(self):
        X_eval_list = []
        y_eval_list = []
        first_mat_shape = None

        for sample_info in self.samples_info:
            try:
                mat = torch.load(sample_info['path'], map_location="cpu")
                if mat.ndim == 4 and mat.shape[0] == 1: mat = mat.squeeze(0)
                if mat.ndim == 3 and mat.shape[0] == 1: mat = mat.squeeze(0)
                
                flat_mat = mat.numpy().flatten()

                if first_mat_shape is None:
                    first_mat_shape = flat_mat.shape
                elif flat_mat.shape != first_mat_shape:
                    print_message(f"Warning: Eval matrix {sample_info['fname']} has shape {flat_mat.shape}, expected {first_mat_shape}. Skipping.")
                    continue

                X_eval_list.append(flat_mat)
                y_eval_list.append(sample_info['label'])
            except Exception as e:
                print_message(f"Error loading or processing {sample_info['fname']} for eval: {e}. Skipping.")
                continue
        
        if not X_eval_list:
            return np.array([]), np.array([])

        return np.array(X_eval_list), np.array(y_eval_list)

def main():
    # Hardcoded arguments
    # Model/Scaler paths are relative to the script's location (colbert_run)
    MODEL_PATH = "logreg_classifier.joblib"
    SCALER_PATH = "logreg_scaler.joblib"
    # Test data paths are relative to the project root (ColBERT-Thesis)
    TEST_JSONL = "colbert_run/test_pairs.jsonl" 
    MATS_DIR = "padded_matrices_cnn"

    print_message("Starting Logistic Regression evaluation with hardcoded parameters")

    script_dir = os.path.dirname(__file__)
    abs_model_path = os.path.join(script_dir, MODEL_PATH)
    abs_scaler_path = os.path.join(script_dir, SCALER_PATH)

    # Construct absolute path for TEST_JSONL if it's not already absolute
    project_root = os.path.dirname(script_dir)
    abs_test_jsonl_path = os.path.join(project_root, TEST_JSONL)


    count_docs_per_query(MATS_DIR, abs_test_jsonl_path) # MATS_DIR is relative to project root

    print_message(f"Loading scaler from {abs_scaler_path}")
    try:
        scaler = joblib.load(abs_scaler_path)
    except FileNotFoundError:
        print_message(f"Error: Scaler file not found at {abs_scaler_path}. Exiting.")
        return
        
    print_message(f"Loading model from {abs_model_path}")
    try:
        model = joblib.load(abs_model_path)
    except FileNotFoundError:
        print_message(f"Error: Model file not found at {abs_model_path}. Exiting.")
        return

    dataset = AllPairsLogRegTestDataset(abs_test_jsonl_path, MATS_DIR)
    X_test, y_true = dataset.load_all_data_for_eval()

    if X_test.shape[0] == 0:
        print_message("No test data to evaluate after processing. Exiting.")
        return

    print_message(f"Test data loaded: X_test shape {X_test.shape}, y_true shape {y_true.shape}")
    
    X_test_scaled = scaler.transform(X_test)

    print_message("Making predictions...")
    y_pred_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_labels = model.predict(X_test_scaled)

    thr = 0.5
    print_message(f"Using fixed threshold = {thr:.2f} (inherent in .predict())")

    print("\n=== Classification Metrics ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred_labels):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_labels, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred_labels, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred_labels, zero_division=0):.4f}")
    
    try:
        if len(np.unique(y_true)) > 1:
            print(f"ROC-AUC  : {roc_auc_score(y_true, y_pred_probs):.4f}")
        else:
            print("ROC-AUC  : N/A (only one class present in y_true)")
    except ValueError as e:
        print(f"ROC-AUC  : N/A ({e})")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred_labels, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred_labels)
    print("\nConfusion Matrix (rows: true, cols: pred):")
    print(cm)

if __name__ == "__main__":
    main()