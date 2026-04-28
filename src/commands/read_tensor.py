import argparse
from pathlib import Path
from typing import Final

import torch

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "embeddings"
VALID_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect saved embedding tensors for a dataset split."
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--train",
        action="store_true",
        help="Inspect the train split.",
    )
    split_group.add_argument(
        "--val",
        action="store_true",
        help="Inspect the validation split.",
    )
    split_group.add_argument(
        "--test",
        action="store_true",
        help="Inspect the test split.",
    )
    parser.add_argument(
        "split",
        nargs="?",
        default="train",
        choices=VALID_SPLITS,
        help="Dataset split to inspect. Defaults to train.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=5,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start printing from this row index.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full embedding vectors instead of a short preview.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory containing <split>_embeddings.pt and <split>_labels.pt files.",
    )
    return parser.parse_args()


def resolve_split(args: argparse.Namespace) -> str:
    if args.train:
        return "train"
    if args.val:
        return "val"
    if args.test:
        return "test"
    return args.split


def load_split(split: str, embeddings_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings_path = embeddings_dir / f"{split}_embeddings.pt"
    labels_path = embeddings_dir / f"{split}_labels.pt"

    if not embeddings_path.is_file() or not labels_path.is_file():
        raise SystemExit(
            f"Could not find tensor files for split '{split}' in {embeddings_dir}.\n"
            f"Expected files:\n"
            f"  {embeddings_path}\n"
            f"  {labels_path}\n"
            f"Example: uv run read-tensor {split} -n 10"
        )

    embeddings = torch.load(embeddings_path, map_location="cpu")
    labels = torch.load(labels_path, map_location="cpu")

    return embeddings, labels


def main() -> None:
    args = parse_args()

    if args.num < 1:
        raise SystemExit("--num must be at least 1.")
    if args.start < 0:
        raise SystemExit("--start must be 0 or greater.")

    split = resolve_split(args)
    embeddings_dir = args.embeddings_dir.resolve()
    embeddings, labels = load_split(split, embeddings_dir)

    if args.start >= len(embeddings):
        raise SystemExit(
            f"--start {args.start} is out of range for split '{split}' "
            f"with {len(embeddings)} rows."
        )

    stop = min(args.start + args.num, len(embeddings))

    print(f"Split: {split}")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Showing rows: {args.start}..{stop - 1}")
    print()

    for i in range(args.start, stop):
        label = labels[i].item()
        embedding = embeddings[i]

        print(f"Index {i}")
        print(f"Label: {label}")

        if args.full:
            print(f"Embedding: {embedding.tolist()}")
        else:
            preview = embedding[:10].tolist()
            print(f"Embedding first 10 values: {preview}")
            print(f"Embedding length: {embedding.numel()}")

        print()
