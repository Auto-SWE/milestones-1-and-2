# Milestone 1 & 2

The dataset download uses the Hugging Face mirror of PrimeVul:
https://huggingface.co/datasets/colin/PrimeVul

## Run

```bash
nix develop
uv sync
uv run download-dataset
uv run process-embeddings
uv run train-lr
uv run train-svm
uv run read-tensor --train -n 3
```
