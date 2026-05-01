import argparse
from pathlib import Path
from typing import Any, Final

from src.data.primevul import EXPECTED_SPLIT_FILES, find_split_files

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw" / "primevul"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CodeBERT embeddings for the downloaded PrimeVul splits."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the downloaded PrimeVul split files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where embedding and label tensors should be written.",
    )
    return parser.parse_args()


def process_split(split: str, path: Path, output_dir: Path, embedder: Any) -> None:
    import torch

    from src.utils.parser import extract_features, load_jsonl

    df = load_jsonl(path)
    code, labels = extract_features(df)

    print(f"Processing {split}... ({len(code)} samples)")

    embeddings: torch.Tensor = embedder.embed(code)

    torch.save(embeddings, output_dir / f"{split}_embeddings.pt")
    torch.save(torch.tensor(labels), output_dir / f"{split}_labels.pt")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise SystemExit(
            f"Input directory does not exist: {input_dir}\n"
            "Run `uv run download-dataset` first or pass `--input-dir`."
        )

    try:
        split_files = find_split_files(input_dir)
    except FileNotFoundError as exc:
        expected = ", ".join(EXPECTED_SPLIT_FILES.values())
        raise SystemExit(
            f"{exc}\n"
            f"Expected PrimeVul split files under {input_dir}: {expected}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    from src.embeddings.embedder import CodeBERTEmbedder

    embedder = CodeBERTEmbedder()

    for split, path in split_files.items():
        process_split(split, path, output_dir, embedder)
