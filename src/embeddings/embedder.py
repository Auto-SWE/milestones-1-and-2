import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/codebert-base"


class CodeBERTEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

    def embed(
        self,
        code_list,
        max_length=512,
        batch_size=16,
        show_progress=False,
        progress_desc="Embedding",
    ):
        embeddings = []
        batch_starts = range(0, len(code_list), batch_size)

        with torch.no_grad():
            for i in tqdm(
                batch_starts,
                desc=progress_desc,
                total=len(batch_starts),
                unit="batch",
                disable=not show_progress,
            ):
                batch = code_list[i : i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)

                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked_hidden_state = outputs.last_hidden_state * attention_mask
                token_counts = attention_mask.sum(dim=1).clamp(min=1)
                batch_embeds = masked_hidden_state.sum(dim=1) / token_counts
                embeddings.append(batch_embeds.cpu())

        return torch.cat(embeddings, dim=0)
