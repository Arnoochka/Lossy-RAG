from collections.abc import Callable

import numpy as np
import pandas as pd

from .search.base import BaseSearch


class Database:
    def __init__(self, searcher: BaseSearch) -> None:
        self.searcher = searcher

    def fit(self, train_texts: list[str]) -> None:
        self.searcher.fit(train_texts)

    def set_data(self, texts: list[str]) -> None:
        chunk_ids = np.arange(len(texts))
        self.chunks = pd.DataFrame({"chunk_id": chunk_ids, "text": texts})
        self.searcher.add(texts, chunk_ids)

    def set_token_computer(
        self, token_computer: Callable[[str], int],
    ) -> None:
        self.token_computer = token_computer

    def compute_num_tokens(self, text: str) -> int:
        return self.token_computer(text)

    def search(
        self,
        question: str,
        k: int = 10,
        max_context_tokens: int | None = None,
    ) -> pd.DataFrame:

        top_ids, top_scores = self.searcher.search([question], k=k)

        # IVF может дополнить неполную выдачу служебными ID и оценками -inf.
        valid = np.isfinite(top_scores[0])
        result = self.chunks.loc[
            top_ids[0, valid], ["chunk_id", "text"],
        ].reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        result["score"] = top_scores[0, valid]

        if max_context_tokens is not None:
            texts = []
            for text in result["text"]:
                context = "\n\n".join([*texts, text])
                if self.compute_num_tokens(context) > max_context_tokens:
                    break
                texts.append(text)
            result = result.iloc[:len(texts)]

        return result
