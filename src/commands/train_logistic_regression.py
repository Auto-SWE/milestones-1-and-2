import argparse
from pathlib import Path
from typing import Final

from src.ml.logistic_regression import train_and_evaluate

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classifier training on generated embeddings."
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing *_embeddings.pt and *_labels.pt files.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_and_evaluate(
        embedding_dir=args.embedding_dir.resolve(),
        max_iter=args.max_iter,
    )
