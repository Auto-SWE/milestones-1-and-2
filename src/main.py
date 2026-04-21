from pathlib import Path
from typing import Final

import torch
from pandas import DataFrame

from src.embeddings.embedder import CodeBERTEmbedder
from src.utils.parser import extract_features, load_jsonl

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "samples"
OUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "embeddings"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES: Final[dict[str, str]] = {
    "train": "primevul_train_sample.jsonl",
    "val": "primevul_valid_sample.jsonl",
    "test": "primevul_test_sample.jsonl",
}


def process_split(name: str, filename: str, embedder: CodeBERTEmbedder) -> None:
    path = DATA_DIR / filename

    df: DataFrame = load_jsonl(path)
    code, labels = extract_features(df)

    print(f"Processing {name}... ({len(code)} samples)")

    embeddings: torch.Tensor = embedder.embed(code)

    torch.save(embeddings, OUT_DIR / f"{name}_embeddings.pt")
    torch.save(torch.tensor(labels), OUT_DIR / f"{name}_labels.pt")


def main() -> None:
    embedder = CodeBERTEmbedder()

    for split, file in FILES.items():
        process_split(split, file, embedder)


if __name__ == "__main__":
    main()
