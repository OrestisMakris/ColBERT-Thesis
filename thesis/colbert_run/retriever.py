# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.join(current_dir, '..', '..')
# sys.path.insert(0, project_root)


# from colbert.data import Queries
# from colbert.infra import Run, RunConfig, ColBERTConfig
# from colbert import Searcher
# import torch

# if __name__=='__main__':
#     with Run().context(RunConfig(nranks=1, experiment="CF44")):
#         config = ColBERTConfig(root="../path/to/experiments")
#         searcher = Searcher(index="CF44", config=config)
#         # Export full document embeddings before running retrieval:
#         #searcher.ranker.export_all_documents(torch.zeros((1, 224 ,128)))
#         queries = Queries("./CF_DataSet/Queries.tsv")
#         # all_query_texts = list(queries.data.values())

#         # # Encode all queries at once. The output should have shape: [num_queries, query_maxlen, emb_dim].
#         # Q_all = searcher.encode(all_query_texts)
#         # torch.save(Q_all, "exported_all_query.pt")
#         # print(f"Exported all queries with shape: {Q_all.shape}")
        
#         ranking = searcher.search_all(queries, k=1000)
#         ranking.save("CF44.nbits=8.rankinggg.tsv")



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
    with Run().context(RunConfig(nranks=1,experiment="DBpediacolbert-ir-f")):
        config = ColBERTConfig(root="../path/to/experiments")
        searcher = Searcher(index="DBpediacolbert-ir-f", config=config)
        # Export full document embeddings before running retrieval:
        #searcher.ranker.export_all_documents(torch.zeros((1, 224 ,128)))
        queries = Queries("./dbpedia-entity_colbert_format/Queries.tsv")
        # all_query_texts = list(queries.data.values())
        

        # # Encode all queries at once. The output should have shape: [num_queries, query_maxlen, emb_dim].
        # Q_all = searcher.encode(all_query_texts)
        # torch.save(Q_all, "exported_all_query.pt")
        # print(f"Exported all queries with shape: {Q_all.shape}")
        
        ranking = searcher.search_all(queries, k=500)
        ranking.save("DBpediacolbert-ir-f.nbits=4.ranking.tsv")

        