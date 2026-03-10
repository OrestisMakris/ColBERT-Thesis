import sys
import os

# Set GCC version BEFORE importing torch/ColBERT
os.environ['CC'] = '/usr/bin/gcc-11'
os.environ['CXX'] = '/usr/bin/g++-11'
os.environ['CUDAHOSTCXX'] = '/usr/bin/g++-11'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__ == '__main__':
    # --- Choose experiment name and collection ---
    EXPERIMENT = "CF19"
    COLLECTION = os.path.join(project_root, "CF_DataSet", "docs.tsv")
    # CF 18 IS MEDIUM, TODO CF 19 IS LARGE.
    # CF:
    # EXPERIMENT = "CF1"
    # COLLECTION = os.path.join(project_root, "CF_DataSet", "docs.tsv")

    with Run().context(RunConfig(nranks=1, experiment=EXPERIMENT)):
        config = ColBERTConfig(nbits=4)
        # indexer = Indexer(checkpoint="colbert-ir/colbertv1.9", config=config)
        indexer = Indexer(checkpoint="./experiments/default/none/2026-03/10/02.46.29/checkpoints/colbert-1204", config=config)
        indexer.index(name=EXPERIMENT, collection=COLLECTION, overwrite=True)
