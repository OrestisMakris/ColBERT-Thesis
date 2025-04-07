import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Move two levels up: from colbert_run -> ColBERT-Thesis -> workspace root
project_root = os.path.join(current_dir, '..', '..', 'ColBERT')
sys.path.insert(0, project_root)


from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="CF1")):

        config = ColBERTConfig(
            root="/path/to/experiments",
        )
        searcher = Searcher(index="CF1", config=config)
        queries = Queries("../CF_DataSet/Queries.tsv")
        ranking = searcher.search_all(queries, k=1224)
        ranking.save("CF1.nbits=1.ranking.tsv")