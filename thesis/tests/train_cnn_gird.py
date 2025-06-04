import os, json, argparse, subprocess
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model_cnn_grid import SimpleCNN

TRAIN_DATA_FILE = "train_data_balanced.jsonl"
PADDED_MATRICES = "padded_matrices_cnn"

def print_message(msg):
    import time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

class ClassificationDataset(Dataset):
    def __init__(self, jsonl_path, mats_dir):
        base       = os.path.dirname(__file__)
        # JSONL file resolved relative to this script
        jsonl_full = os.path.join(base, jsonl_path)
        # mats_dir is one level up from this script
        self.dir   = os.path.join(base, "..", mats_dir)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(self.dir)

        self.samples = []
        with open(jsonl_full, "r") as f:
            for ln, line in enumerate(f):
                obj  = json.loads(line)
                path = os.path.join(self.dir, obj["matrix_file"])
                self.samples.append((path, obj["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mat = torch.load(path, map_location="cpu")
        # ensure shape [1, H, W]
        if mat.ndim == 2:
            mat = mat.unsqueeze(0)
        return mat, torch.tensor(label, dtype=torch.float)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv_blocks", type=int, choices=[1,2,3], required=True)
    parser.add_argument("--hidden",      type=int, choices=[8,12,16,24,32,38], required=True)
    parser.add_argument("--kernel",      type=int, choices=[2,3,4,5,6,7], required=True)
    parser.add_argument("--padding",     type=int, choices=[1,2], required=True)
    parser.add_argument("--dropout",     type=float, choices=[0.3,0.5,0.6], required=True)
    parser.add_argument("--mlp_head",    type=int, choices=[0,1], required=True)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=85)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--save",        type=str,   default="cnn_classifierrr.pt")
    parser.add_argument("--eval_json",   type=str,   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(
        args.conv_blocks, args.hidden,
        args.kernel, args.padding,
        args.dropout, args.mlp_head
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    ds = ClassificationDataset(TRAIN_DATA_FILE, PADDED_MATRICES)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, pin_memory=device.type=="cuda")

    model.train()
    for epoch in range(1, args.epochs+1):
        total = 0.0
        for mats, labels in dl:
            mats, labels = mats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mats)
            loss   = criterion(logits, labels.view_as(logits))
            loss.backward(); optimizer.step()
            total += loss.item()
        print_message(f"Epoch {epoch}/{args.epochs} avg_loss {total/len(dl):.4f}")

    torch.save(model.state_dict(), args.save)
    print_message(f"Model saved → {args.save}")

    if args.eval_json:
        base        = os.path.dirname(__file__)
        eval_script = os.path.join(base, "eval_cnn_multi_topk.py")
        cmd = [
            "python", eval_script,
             "--model", args.save,
             "--topk",  "5,10,25,50,100",
             "--json",  args.eval_json
         ]

        print_message(f"Running eval: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print_message(f"EVAL ERROR:\n{proc.stderr}")
        else:
            stats = json.load(open(args.eval_json))
            for k in ["5","10","25","50","100"]:
                v = stats.get(k, {})
                print_message(f"TOPK={k} → MAP={v.get('MAP',0):.4f}, MRR={v.get('MRR',0):.4f}")

if __name__=="__main__":
    main()