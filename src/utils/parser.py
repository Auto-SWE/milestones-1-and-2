import json
from os import PathLike
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

JsonlPath = str | PathLike[str]
JsonRecord = dict[str, object]
LabelArray = NDArray[np.int64]


def load_jsonl(path: JsonlPath) -> pd.DataFrame:
    data: list[JsonRecord] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path}")

            record = cast(JsonRecord, record)
            data.append(record)

    return pd.DataFrame(data)


def extract_features(
    df: pd.DataFrame,
) -> tuple[list[str], LabelArray]:
    code = [str(value) for value in df["func"].tolist()]
    labels = cast(LabelArray, df["target"].to_numpy(dtype=np.int64))

    return code, labels
