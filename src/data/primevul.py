import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import gdown

PRIMEVUL_V01_URL: Final[str] = (
    "https://drive.google.com/drive/folders/1cznxGme5o6A_9tT8T47JUh3MPEpRYiKK"
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
        download_root.mkdir(parents=True, exist_ok=True)

        result = gdown.download_folder(
            url=PRIMEVUL_V01_URL,
            output=str(download_root),
            quiet=False,
            resume=True,
        )

        if not result:
            raise RuntimeError("PrimeVul download returned no files.")

        split_files = find_split_files(download_root)
        shutil.move(str(download_root), str(target_dir))

    return find_split_files(target_dir)
