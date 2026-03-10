import sys
import os

# Set GCC version BEFORE importing torch/ColBERT
os.environ['CC'] = '/usr/bin/gcc-11'
os.environ['CXX'] = '/usr/bin/g++-11'
os.environ['CUDAHOSTCXX'] = '/usr/bin/g++-11'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import torch
torch.cuda.empty_cache()

if __name__ == '__main__':
    # --- Choose experiment ---
    EXPERIMENT    = "CF19"
    QUERIES_FILE  = os.path.join(project_root, "CF_DataSet", "Queries.tsv")
    RANKING_OUT   = "CF_paper2.nbits=4.ranking.tsv"
    IVF_OUT_DIR   = os.path.join(project_root, "ivf_candidates_cf19_largetuned")

    # CF:
    # EXPERIMENT   = "CF1"
    # QUERIES_FILE = os.path.join(project_root, "CF_DataSet", "Queries.tsv")
    # RANKING_OUT  = "CF_paper2.nbits=4.ranking.tsv"
    # IVF_OUT_DIR  = os.path.join(project_root, "ivf_candidates_cf17_untuned")

    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT)):
        config = ColBERTConfig(root="../path/to/experiments")
        searcher = Searcher(index=EXPERIMENT, config=config)

        queries = Queries(QUERIES_FILE)

        # Enable IVF candidate export — feeds paper2/matrices_proce export
        searcher.configure_ivf_export(IVF_OUT_DIR)

        ranking = searcher.search_all(queries, k=1000)
        ranking.save(RANKING_OUT)

        print(f"\nIVF candidates saved to: {IVF_OUT_DIR}")
        print(f"Format: q{{qid}}/q{{qid}}_ivf_candidates.txt")
