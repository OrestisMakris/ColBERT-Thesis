import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train():
    # construct absolute paths for the dataset files
    triples = os.path.join(project_root, 'dbpedia-entity_colbert_format', 'triplets.jsonl')
    queries = os.path.join(project_root, 'dbpedia-entity_colbert_format', 'Queries.tsv')
    collection = os.path.join(project_root, 'dbpedia-entity_colbert_format', 'docs.tsv')

    # triples = os.path.join(project_root, 'CF_DataSet', 'triplets.jsonl')
    # queries = os.path.join(project_root, 'CF_DataSet', 'Queries.tsv')
    # collection = os.path.join(project_root, 'CF_DataSet', 'docs.tsv')

    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=1)):
        

        config = ColBERTConfig(bsize=2, lr=1e-03, warmup=20_000, doc_maxlen=120, dim=128, 
                                attend_to_mask_tokens=False, nway=2, accumsteps=1, similarity='cosine', 
                                use_ib_negatives=False)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        best_checkpoint = trainer.train(checkpoint='colbert-ir/colbertv1.9')
        print(f"Saved checkpoint to: {best_checkpoint}")


if __name__ == '__main__':
    train()