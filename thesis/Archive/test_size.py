import torch

def main():
    # Load the exported query tensor
    query_tensor = torch.load("exported_all_doc_padded.pt")
    print("Exported all Docs shape:", query_tensor.shape)

if __name__ == "__main__":
    main()