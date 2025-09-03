import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)


from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import torch
torch.cuda.empty_cache()

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1,experiment="CF11")):
        config = ColBERTConfig(root="../path/to/experiments")
        searcher = Searcher(index="CF11", config=config)
        # Export full document embeddings before running retrieval:
        #searcher.ranker.export_all_documents(torch.zeros((1, 224 ,128)))
        queries = Queries("./CF_DataSet/Queries.tsv")
        # all_query_texts = list(queries.data.values())
        
        ranking = searcher.search_all(queries, k=1000)
        ranking.save("CF11Time.nbits=4.ranking_time.tsv")

        