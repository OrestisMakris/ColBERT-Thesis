import faiss
import os
import torch
print(faiss.__version__)

def main():
    cwd = os.getcwd()
    query_path = os.path.join(cwd, 'query_embeddings.pt')
    document_path = os.path.join(cwd, 'document_embeddings.pt')

    try:
        query_embeddings = torch.load(query_path)
        document_embeddings = torch.load(document_path)
    except Exception as e:
        print("Error loading embeddings:", e)
        return

    output = (
        f"Query Embeddings Structure: {query_embeddings.shape}\n"
        f"Document Embeddings Structure: {document_embeddings.shape}\n"
        f"Location: {cwd}\n"
    )

    with open("embeddings_structure.txt", "w") as out_file:
        out_file.write(output)

    print("Embeddings structure information saved to embeddings_structure.txt")

if __name__ == "__main__":
    main()