import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="CF6")):
        config = ColBERTConfig(
            nbits=2,
        )
        indexer = Indexer(checkpoint="./experiments/default/none/2025-06/03/12.23.25/checkpoints/colbert", config=config)
        indexer.index(name="CF6", collection="./dbpedia-entity_colbert_format/docs.tsv", overwrite=True)


