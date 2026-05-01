import argparse
from pathlib import Path
from typing import Final

from src.data.primevul import PRIMEVUL_DATASET_URL, download_primevul

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw" / "primevul"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the PrimeVul dataset into the local data directory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PrimeVul files should be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the target directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.output_dir.resolve()

    print(f"Downloading PrimeVul v0.1 from {PRIMEVUL_DATASET_URL}")
    print(f"Target directory: {target_dir}")

    split_files = download_primevul(target_dir=target_dir, force=args.force)

    print("Download complete.")
    for split, path in split_files.items():
        print(f"{split}: {path}")
