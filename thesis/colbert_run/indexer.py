import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="fiqa_colbert_tuned")):
        config = ColBERTConfig(
            nbits=8,
        )
        indexer = Indexer(checkpoint="./experiments/default/none/2025-06/23/00.34.24/checkpoints/colbert", config=config)
        indexer.index(name="fiqa_colbert_tuned", collection="./fiqa_colbert_format_gt5/docs.tsv", overwrite=True)


# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.join(current_dir, '..', '..')
# sys.path.insert(0, project_root)

# from colbert.infra import Run, RunConfig, ColBERTConfig
# from colbert import Indexer

# if __name__=='__main__':
#     with Run().context(RunConfig(nranks=1, experiment="CF44")):
#         config = ColBERTConfig(
#             nbits=8,
#         )
#         config = ColBERTConfig(bsize=4, lr=1e-03, warmup=20_000, doc_maxlen=512, dim=128, 
#                                 attend_to_mask_tokens=False, nway=2, accumsteps=1, similarity='cosine', 
#                                 use_ib_negatives=False, nbits=8)
#         indexer = Indexer(checkpoint="colbert-ir/colbertv1.9", config=config)
#         indexer.index(name="CF44", collection="./CF_DataSet/docs.tsv", overwrite=True)


