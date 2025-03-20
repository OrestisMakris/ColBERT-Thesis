from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="CF20")):

        config = ColBERTConfig(
            nbits=1,
        )
        indexer = Indexer(checkpoint="./experiments/default/none/2025-03/14/16.16.13/checkpoints/colbert", config=config)
        indexer.index(name="CF20", collection="./docs.tsv", overwrite=True)
