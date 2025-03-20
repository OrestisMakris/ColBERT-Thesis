from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train():
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=1)):
        triples = './triplets.jsonl' 
        queries='./Queries.tsv'
        collection='./docs.tsv'

        config = ColBERTConfig(bsize=8, lr=1e-05, warmup=20_000, doc_maxlen=280, dim=128, attend_to_mask_tokens=False, nway=2, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        best_checkpoint = trainer.train(checkpoint='colbert-ir/colbertv1.9')
        print(f"Saved checkpoint to: {best_checkpoint}")


if __name__ == '__main__':
    train()
4