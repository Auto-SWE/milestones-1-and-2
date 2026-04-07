import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/codebert-base"

class CodeBERTEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

    def embed(self, code_list, max_length=256, batch_size=32):
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(code_list), batch_size):
                batch = code_list[i:i+batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)

                # CLS token embedding
                batch_embeds = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeds.cpu())

        return torch.cat(embeddings, dim=0)