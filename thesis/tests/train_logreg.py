import os
import json
# import argparse # Removed
import time
import numpy as np
import torch # For loading .pt files
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split # Removed
from sklearn.metrics import classification_report # Kept for potential future use, but not used for validation here
import joblib # For saving sklearn models

def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

class LogRegDataset:
    def __init__(self, jsonl_path, mats_dir):
        # Assuming jsonl_path is relative to the script's execution directory (ColBERT-Thesis)
        # and mats_dir is also relative to ColBERT-Thesis
        self.jsonl_path = jsonl_path
        self.mats_dir = mats_dir # This will be joined with ".." in load_all_data if script is in colbert_run

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Training JSONL file not found: {self.jsonl_path}")

        self.samples = []
        with open(self.jsonl_path) as f:
            for ln, line in enumerate(f):
                obj = json.loads(line)
                if "matrix_file" in obj and obj["label"] in (0, 1):
                    self.samples.append(obj)
                else:
                    print_message(f"Skipping malformed line {ln+1} in {self.jsonl_path}")
        print_message(f"Loaded {len(self.samples)} samples from {self.jsonl_path}")

    def load_all_data(self):
        X_list = []
        y_list = []
        
        first_mat_shape = None
        
        # Construct absolute path to mats_dir relative to the project root
        # Assuming this script (train_logreg.py) is in colbert_run
        script_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(script_dir) # Goes up one level from colbert_run to ColBERT-Thesis
        absolute_mats_dir = os.path.join(project_root, self.mats_dir)

        if not os.path.isdir(absolute_mats_dir):
            raise FileNotFoundError(f"Matrices directory not found: {absolute_mats_dir}")

        for i, rec in enumerate(self.samples):
            path = os.path.join(absolute_mats_dir, rec["matrix_file"])
            try:
                mat = torch.load(path, map_location="cpu")
                
                if mat.ndim == 4 and mat.shape[0] == 1:
                    mat = mat.squeeze(0)
                if mat.ndim == 3 and mat.shape[0] == 1:
                    mat = mat.squeeze(0)
                
                flat_mat = mat.numpy().flatten()

                if first_mat_shape is None:
                    first_mat_shape = flat_mat.shape
                elif flat_mat.shape != first_mat_shape:
                    print_message(f"Warning: Matrix {rec['matrix_file']} has shape {flat_mat.shape}, expected {first_mat_shape}. Skipping.")
                    continue
                
                X_list.append(flat_mat)
                y_list.append(int(rec["label"]))
            except Exception as e:
                print_message(f"Error loading or processing {rec['matrix_file']}: {e}. Skipping.")
                continue
        
        if not X_list:
            raise ValueError("No valid data loaded. Check matrix files and shapes.")
            
        return np.array(X_list), np.array(y_list)

def main():
    # Hardcoded arguments
    # These paths are relative to the project root (ColBERT-Thesis)
    TRAIN_JSONL = "/home/st1084516/ColBERT-Thesis/colbert_run/train_data_balanced.jsonl"
    MATS_DIR = "padded_matrices_cnn"
    # Save paths are relative to the script's location (colbert_run)
    MODEL_SAVE_PATH = "logreg_classifier.joblib"
    SCALER_SAVE_PATH = "logreg_scaler.joblib"
    RANDOM_STATE = 42
    SOLVER = "saga" 
    PENALTY = "l1"   
    C_VALUE = 1.0

    print_message("Starting Logistic Regression training with hardcoded parameters (no validation split)")
    
    # Load data
    dataset = LogRegDataset(TRAIN_JSONL, MATS_DIR)
    X_train, y_train = dataset.load_all_data() # Use all data for training

    if X_train.shape[0] == 0:
        print_message("No data loaded. Exiting.")
        return

    print_message(f"Loaded training data: X shape {X_train.shape}, y shape {y_train.shape}")

    # Scale features
    print_message("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Logistic Regression model
    print_message(f"Training Logistic Regression (solver={SOLVER}, penalty={PENALTY}, C={C_VALUE}) on full dataset...")
    model = LogisticRegression(
        solver=SOLVER,
        penalty=PENALTY,
        C=C_VALUE,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=100, 
        n_jobs=-1,
        verbose=1  # Set to 1 for more detailed output during training
    )
    model.fit(X_train_scaled, y_train)
    print_message("Model training complete.")

    # Save the scaler and the model
    script_dir = os.path.dirname(__file__) # Assumes script is in colbert_run
    scaler_path = os.path.join(script_dir, SCALER_SAVE_PATH)
    model_path = os.path.join(script_dir, MODEL_SAVE_PATH)
    
    joblib.dump(scaler, scaler_path)
    print_message(f"Scaler saved to {scaler_path}")
    joblib.dump(model, model_path)
    print_message(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()