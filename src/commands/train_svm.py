import argparse
from pathlib import Path
from typing import Final

from src.ml.svm import DEFAULT_C, DEFAULT_MAX_ITER, train_and_evaluate

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run linear SVM training on generated embeddings."
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
        default=DEFAULT_MAX_ITER,
        help="Maximum iterations for linear SVM.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=DEFAULT_C,
        help="Regularization parameter for linear SVM. Smaller values mean stronger regularization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_and_evaluate(
        embedding_dir=args.embedding_dir.resolve(),
        max_iter=args.max_iter,
        c=args.c,
    )
