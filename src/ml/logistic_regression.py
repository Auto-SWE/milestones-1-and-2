from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

DEFAULT_C: Final[float] = 0.02
DEFAULT_MAX_ITER: Final[int] = 300
DEFAULT_CLASS_WEIGHT: Final[str] = "balanced"
DEFAULT_SOLVER: Final[str] = "liblinear"
DEFAULT_RANDOM_STATE: Final[int] = 42


class Split(StrEnum):
    TRAINING = "train"
    TEST = "test"
    VALIDATION = "val"


def load_split(split: Split, embedding_dir: Path):
    x = torch.load(embedding_dir / f"{split.value}_embeddings.pt", map_location="cpu")
    y = torch.load(embedding_dir / f"{split.value}_labels.pt", map_location="cpu")

    return x.numpy(), y.numpy()


def find_best_threshold(y_true, positive_scores) -> tuple[float, float]:
    order = np.argsort(positive_scores)[::-1]
    sorted_y = y_true[order]

    true_positives = np.cumsum(sorted_y == 1)
    false_positives = np.cumsum(sorted_y == 0)
    positives = true_positives[-1]
    false_negatives = positives - true_positives

    denominator = (2 * true_positives) + false_positives + false_negatives
    f1_scores = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros_like(true_positives, dtype=float),
        where=denominator != 0,
    )
    best_index = int(np.argmax(f1_scores))

    return float(positive_scores[order[best_index]]), float(f1_scores[best_index])


def evaluate(
    model: LogisticRegression,
    split_name: str,
    x,
    y,
    threshold: float,
) -> None:
    positive_scores = model.predict_proba(x)[:, 1]
    pred = (positive_scores >= threshold).astype(int)
    print(f"\n{split_name} F1: ", f1_score(y, pred))
    print(classification_report(y, pred))


def train_and_evaluate(
    embedding_dir: Path,
    max_iter: int = DEFAULT_MAX_ITER,
    c: float = DEFAULT_C,
) -> LogisticRegression:
    x_train, y_train = load_split(Split.TRAINING, embedding_dir)
    x_val, y_val = load_split(Split.VALIDATION, embedding_dir)
    x_test, y_test = load_split(Split.TEST, embedding_dir)

    model = LogisticRegression(
        C=c,
        max_iter=max_iter,
        class_weight=DEFAULT_CLASS_WEIGHT,
        solver=DEFAULT_SOLVER,
        random_state=DEFAULT_RANDOM_STATE,
    )

    model.fit(x_train, y_train)

    validation_scores = model.predict_proba(x_val)[:, 1]
    threshold, validation_f1 = find_best_threshold(y_val, validation_scores)
    print(f"Selected threshold from validation F1: {threshold:.6f}")
    print(f"Best validation F1 at selected threshold: {validation_f1:.6f}")

    evaluate(model, "Validation", x_val, y_val, threshold)
    evaluate(model, "Test", x_test, y_test, threshold)

    return model
