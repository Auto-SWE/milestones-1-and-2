import pandas as pd
import json

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def extract_features(df):
    return df["func"].tolist(), df["target"].values