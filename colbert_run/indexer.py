import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Move two levels up: from colbert_run -> ColBERT-Thesis -> workspace root
project_root = os.path.join(current_dir, '..', '..', 'ColBERT')
sys.path.insert(0, project_root)

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="CF1")):
        config = ColBERTConfig(
            nbits=1,
        )
        indexer = Indexer(checkpoint="./experiments/default/none/2025-03/20/17.12.43/checkpoints/colbert", config=config)
        indexer.index(name="CF1", collection="../CF_DataSet/docs.tsv", overwrite=True)