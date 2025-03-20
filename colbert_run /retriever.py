from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="CF20")):

        config = ColBERTConfig(
            root="/path/to/experiments",
        )
        searcher = Searcher(index="CF20", config=config)
        queries = Queries("./Queries.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("CF5.nbits=1.ranking.tsv")