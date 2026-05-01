from enum import StrEnum
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


class Split(StrEnum):
    TRAINING = "train"
    TEST = "test"
    VALIDATION = "val"


def load_split(split: Split, embedding_dir: Path):
    x = torch.load(embedding_dir / f"{split.value}_embeddings.pt", map_location="cpu")
    y = torch.load(embedding_dir / f"{split.value}_labels.pt", map_location="cpu")

    return x.numpy(), y.numpy()


def evaluate(model: LogisticRegression, split_name: str, x, y) -> None:
    pred = model.predict(x)
    print(f"\n{split_name} F1: ", f1_score(y, pred))
    print(classification_report(y, pred))


def train_and_evaluate(embedding_dir: Path, max_iter: int = 1000) -> LogisticRegression:
    x_train, y_train = load_split(Split.TRAINING, embedding_dir)
    x_val, y_val = load_split(Split.VALIDATION, embedding_dir)
    x_test, y_test = load_split(Split.TEST, embedding_dir)

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(x_train, y_train)
    evaluate(model, "Validation", x_val, y_val)
    evaluate(model, "Test", x_test, y_test)

    return model
