import os
import torch
from src.utils.parser import load_jsonl, extract_features
from src.embeddings.embedder import CodeBERTEmbedder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/samples")
OUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/embeddings")

os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "train": "primevul_train_sample.jsonl",
    "val": "primevul_valid_sample.jsonl",
    "test": "primevul_test_sample.jsonl"
}

def process_split(name, filename, embedder):
    path = os.path.join(DATA_DIR, filename)

    df = load_jsonl(path)
    code, labels = extract_features(df)

    print(f"Processing {name}... ({len(code)} samples)")

    embeddings = embedder.embed(code)

    torch.save(embeddings, os.path.join(OUT_DIR, f"{name}_embeddings.pt"))
    torch.save(torch.tensor(labels), os.path.join(OUT_DIR, f"{name}_labels.pt"))

def main():
    embedder = CodeBERTEmbedder()

    for split, file in FILES.items():
        process_split(split, file, embedder)

if __name__ == "__main__":
    main()