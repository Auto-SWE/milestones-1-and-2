import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final
from urllib.request import urlopen

PRIMEVUL_DATASET_URL: Final[str] = (
    "https://huggingface.co/datasets/colin/PrimeVul"
)
HUGGING_FACE_DATASET_BASE_URL: Final[str] = (
    "https://huggingface.co/datasets/colin/PrimeVul/resolve/main"
)
EXPECTED_SPLIT_FILES: Final[dict[str, str]] = {
    "train": "primevul_train.jsonl",
    "val": "primevul_valid.jsonl",
    "test": "primevul_test.jsonl",
}


def find_split_files(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}

    for split, filename in EXPECTED_SPLIT_FILES.items():
        matches = sorted(root.rglob(filename))

        if not matches:
            raise FileNotFoundError(f"Missing expected PrimeVul file: {filename}")

        found[split] = matches[0]

    return found


def download_primevul_from_hugging_face(target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in EXPECTED_SPLIT_FILES.values():
        file_url = f"{HUGGING_FACE_DATASET_BASE_URL}/{filename}"
        output_path = target_dir / filename

        print(f"Downloading {filename} from Hugging Face...")
        with urlopen(file_url) as response, output_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)

    return find_split_files(target_dir)


def download_primevul(target_dir: Path, force: bool = False) -> dict[str, Path]:
    if target_dir.exists():
        if any(target_dir.iterdir()) and not force:
            raise FileExistsError(
                f"{target_dir} already exists and is not empty. Re-run with --force to replace it."
            )

        if force:
            shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=target_dir.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        download_root = temp_dir / "primevul"
        split_files = download_primevul_from_hugging_face(download_root)

        shutil.move(str(download_root), str(target_dir))

    return find_split_files(target_dir)
