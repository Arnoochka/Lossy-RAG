import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class BaseEncoder:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 32,
        max_length: int = 512,
        query_instruction: str = "",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_instruction = query_instruction

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        device = next(self.model.parameters()).device
        prefix = self.query_instruction if is_query else ""
        embeddings = []
        self.model.eval()

        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [prefix + text for text in texts[start:start + self.batch_size]]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                vectors = self.model(**inputs).last_hidden_state[:, 0]
                vectors = F.normalize(vectors, p=2, dim=1)
                embeddings.append(vectors.float().cpu().numpy())

        return np.concatenate(embeddings)


class DummyEncoder(BaseEncoder):
    def __init__(
        self,
        chunk_texts: list[str],
        chunk_embeddings: np.ndarray,
        query_texts: list[str],
        query_embeddings: np.ndarray,
    ) -> None:
        self.chunk_embeddings = chunk_embeddings
        self.query_embeddings = query_embeddings
        self.chunk_positions = {text: i for i, text in enumerate(chunk_texts)}
        self.query_positions = {text: i for i, text in enumerate(query_texts)}

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            positions = self.query_positions
            embeddings = self.query_embeddings
        else:
            positions = self.chunk_positions
            embeddings = self.chunk_embeddings

        indices = [positions[text] for text in texts]
        return embeddings[indices]
