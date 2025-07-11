import os, subprocess, itertools, json

grid = {
    "conv_blocks": [1, 2, 3],
    "hidden":      [8, 12, 16, 24, 32, 38],
    "kernel":      [2, 3, 4, 5, 6, 7],
    "padding":     [1, 2],
    "dropout":     [0.3, 0.5, 0.6],
    "mlp_head":    [0, 1],
}

total = 1
for v in grid.values():
    total *= len(v)
print(f"Total experiments: {total}")  # 1,296

SCRIPT_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(SCRIPT_DIR, "models"), exist_ok=True)
best5 = []

def run_train_eval(params):
    name       = "_".join(f"{k}{v}" for k,v in params.items())
    model_path = os.path.join(SCRIPT_DIR, "models", f"cnn_{name}.pt")
    eval_json  = os.path.join(SCRIPT_DIR, "models", f"eval_{name}.json")

    train_script = os.path.join(SCRIPT_DIR, "train_cnn_gird.py")
    cmd = ["python", train_script] + \
           sum(([f"--{k}", str(v)] for k,v in params.items()), []) + \
           ["--save", model_path, "--eval_json", eval_json]

    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    stats = json.load(open(eval_json, "r"))
    entry = {"model": model_path, **params}
    for k in ["5","10","25","50","100"]:
        entry[f"MAP@{k}"] = stats[k]["MAP"]
        entry[f"MRR@{k}"] = stats[k]["MRR"]
    return entry

if __name__=="__main__":
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        try:
            res = run_train_eval(params)
        except subprocess.CalledProcessError:
            continue
        best5.append(res)
        best5.sort(key=lambda x: x["MRR@100"], reverse=True)
        best5 = best5[:5]

        avg_map = sum(r["MAP@100"] for r in best5)/len(best5)
        avg_mrr = sum(r["MRR@100"] for r in best5)/len(best5)
        print("\n=== Top 5 so far ===")
        print(f"Avg@100 MAP={avg_map:.4f}, MRR={avg_mrr:.4f}")
        for i,r in enumerate(best5,1):
            print(f"{i}) {os.path.basename(r['model'])}  MAP@100={r['MAP@100']:.4f}  MRR@100={r['MRR@100']:.4f}")
        print("="*40)

    with open("models/grid_results.json","w") as f:
        json.dump(best5, f, indent=2)
    print("Finished grid search. Top 5 saved to models/grid_results.json")
