import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy


class SquadDatasetBuilder:
    """Строит пять таблиц из train- и validation-частей SQuAD."""

    def __init__(
        self,
        window: int = 2,
        step: int = 1,
        batch_size: int = 256,
    ):
        """
        window     — сколько предложений в одном куске
        step       — сдвиг окна, в предложениях
        batch_size — размер пачки абзацев для spaCy

        Предложения выделяются моделью spaCy en_core_web_sm, её нужно установить заранее.
        """
        self.window = window
        self.step = step
        self.batch_size = batch_size
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

        self.tables: dict[str, pd.DataFrame] = {}
        self.excluded = pd.DataFrame()
        self.n_position_mismatch = 0

    def build(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_paraphrased: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Строит все пять таблиц и возвращает их словарём.
        """
        raw = self._merge_splits(df_train, df_valid)

        contexts = self._build_contexts(raw)
        chunks = self._build_chunks(contexts)
        contexts["n_chunks"] = (
            contexts["pid"].map(chunks.groupby("pid").size()).fillna(0).astype(int)
        )

        questions, answers = self._build_questions_and_answers(raw, contexts)
        answer_chunks = self._build_answer_chunks(questions, answers, contexts, chunks)
        if df_paraphrased is not None:
            questions = self._add_paraphrases(questions, df_paraphrased)
        questions, answers, answer_chunks = self._exclude_without_annotated_chunks(
            questions, answers, answer_chunks
        )

        self.tables = {
            "contexts": contexts,
            "chunks": chunks,
            "questions": questions,
            "answers": answers,
            "answer_chunks": answer_chunks,
        }
        self._report()
        return self.tables

    def save(self) -> None:
        if not self.tables:
            raise RuntimeError("Сначала вызовите build()")
        out = Path.cwd()
        for name, df in self.tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        print(f"Сохранено в {out}")

    @staticmethod
    def _merge_splits(df_train: pd.DataFrame, df_valid: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for role, df in (("train", df_train), ("valid", df_valid)):
            part = df[["id", "title", "context", "question", "answers"]].copy()
            part["role"] = role
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _build_contexts(raw: pd.DataFrame) -> pd.DataFrame:
        contexts = raw[["title", "context"]].drop_duplicates().reset_index(drop=True)
        contexts.insert(0, "pid", range(len(contexts)))
        return contexts

    @staticmethod
    def _sentence_spans(doc) -> list[tuple[int, int]]:
        return [(s.start_char, s.end_char) for s in doc.sents if s.text.strip()]

    def _windows(self, spans: list[tuple[int, int]], text_len: int) -> list[tuple[int, int]]:
        if len(spans) <= self.window:
            return [(0, text_len)]
        starts = list(range(0, len(spans) - self.window + 1, self.step))
        if starts[-1] + self.window < len(spans):
            starts.append(len(spans) - self.window)
        return [(spans[i][0], spans[i + self.window - 1][1]) for i in starts]

    def _build_chunks(self, contexts: pd.DataFrame) -> pd.DataFrame:
        rows = []
        pids = contexts["pid"].tolist()
        texts = contexts["context"].tolist()
        docs = self.nlp.pipe(texts, batch_size=self.batch_size)
        for pid, text, doc in zip(pids, texts, docs):
            spans = self._sentence_spans(doc)
            for chunk_no, (s, e) in enumerate(self._windows(spans, len(text))):
                rows.append((pid, chunk_no, s, e, text[s:e]))
        chunks = pd.DataFrame(rows, columns=["pid", "chunk_no", "start", "end", "text"])
        chunks.insert(0, "chunk_id", range(len(chunks)))
        return chunks

    def _build_questions_and_answers(
        self, raw: pd.DataFrame, contexts: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pid_of = {
            (title, context): pid
            for pid, title, context in contexts[["pid", "title", "context"]].itertuples(index=False)
        }
        q_rows, a_rows = [], []
        self.n_position_mismatch = 0

        for qid, row in enumerate(raw.itertuples(index=False)):
            pid = pid_of[(row.title, row.context)]
            q_rows.append((qid, row.id, qid, row.role, False, pid, row.question))

            seen = set()
            for text, start in zip(row.answers["text"], row.answers["answer_start"]):
                pair = (str(text), int(start))
                if pair in seen:
                    continue
                seen.add(pair)
                if row.context[pair[1]: pair[1] + len(pair[0])] != pair[0]:
                    self.n_position_mismatch += 1
                a_rows.append((qid, pair[0], pair[1]))

        questions = pd.DataFrame(
            q_rows,
            columns=["qid", "squad_id", "orig_qid", "role", "is_paraphrase", "pid", "question"],
        )
        answers = pd.DataFrame(a_rows, columns=["orig_qid", "text", "start"])
        return questions, answers

    @staticmethod
    def _pattern(text: str) -> re.Pattern:
        return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)")

    def _build_answer_chunks(
        self,
        questions: pd.DataFrame,
        answers: pd.DataFrame,
        contexts: pd.DataFrame,
        chunks: pd.DataFrame,
    ) -> pd.DataFrame:
        context_of = dict(zip(contexts["pid"], contexts["context"]))

        chunks_of = defaultdict(list)
        for cid, pid, s, e in chunks[["chunk_id", "pid", "start", "end"]].itertuples(index=False):
            chunks_of[pid].append((cid, s, e))

        answers_of = defaultdict(list)
        for orig_qid, text, start in answers.itertuples(index=False):
            answers_of[orig_qid].append((text, start))

        def inside(cs: int, ce: int, positions) -> bool:
            return any(cs <= s and s + l <= ce for s, l in positions)

        rows = []
        originals = questions[~questions["is_paraphrase"]]
        for orig_qid, pid in originals[["orig_qid", "pid"]].itertuples(index=False):
            context = context_of[pid]
            ans = answers_of[orig_qid]

            annotated = {(start, len(text)) for text, start in ans}
            found = set()
            for text in {t for t, _ in ans if t}:
                for m in self._pattern(text).finditer(context):
                    found.add((m.start(), len(text)))
            unannotated = found - annotated

            for cid, cs, ce in chunks_of[pid]:
                is_annotated = inside(cs, ce, annotated)
                if is_annotated or inside(cs, ce, unannotated):
                    rows.append((orig_qid, cid, is_annotated))

        return pd.DataFrame(rows, columns=["orig_qid", "chunk_id", "is_annotated"])

    @staticmethod
    def _add_paraphrases(questions: pd.DataFrame, paraphrased: pd.DataFrame) -> pd.DataFrame:
        par = paraphrased[questions.columns].copy()
        par["is_paraphrase"] = True
        pid_of = dict(zip(questions["qid"], questions["pid"]))
        if (par["orig_qid"].map(pid_of) != par["pid"]).any():
            raise ValueError(
                "Перефразы не согласуются с вопросами: orig_qid не найден или pid не совпадает"
            )
        return pd.concat([questions, par], ignore_index=True)

    def _exclude_without_annotated_chunks(
        self, questions: pd.DataFrame, answers: pd.DataFrame, answer_chunks: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        keep = set(answer_chunks.loc[answer_chunks["is_annotated"], "orig_qid"])
        bad = ~questions["orig_qid"].isin(keep)
        self.excluded = questions[bad]
        questions = questions[~bad].reset_index(drop=True)
        answers = answers[answers["orig_qid"].isin(keep)].reset_index(drop=True)
        answer_chunks = answer_chunks[answer_chunks["orig_qid"].isin(keep)].reset_index(drop=True)
        return questions, answers, answer_chunks

    def _report(self) -> None:
        t = self.tables
        q = t["questions"]
        ac = t["answer_chunks"]
        ex = self.excluded
        n_orig = int((~q["is_paraphrase"]).sum())

        print(f"contexts:      {len(t['contexts']):>8}")
        print(f"chunks:        {len(t['chunks']):>8}")
        print(f"questions:     {len(q):>8}  (исходных: {n_orig}, перефразов: {len(q) - n_orig}; "
              f"train: {(q['role'] == 'train').sum()}, valid: {(q['role'] == 'valid').sum()})")
        print(f"answers:       {len(t['answers']):>8}")
        print(f"answer_chunks: {len(ac):>8}  (is_annotated=True: {int(ac['is_annotated'].sum())})")
        print(f"исключено вопросов без размеченного куска: {int((~ex['is_paraphrase']).sum())} "
              f"(и их перефразов: {int(ex['is_paraphrase'].sum())})")
        if self.n_position_mismatch:
            print(f"внимание: у {self.n_position_mismatch} ответов текст не совпал "
                  f"с context[start:start+len]")


def build_and_save(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_paraphrased: pd.DataFrame | None = None,
    **builder_kwargs,
) -> dict[str, pd.DataFrame]:
    builder = SquadDatasetBuilder(**builder_kwargs)
    tables = builder.build(df_train, df_valid, df_paraphrased)
    builder.save()
    return tables


if __name__ == "__main__":
    splits = {
        "train": "plain_text/train-00000-of-00001.parquet",
        "valid": "plain_text/validation-00000-of-00001.parquet",
    }
    df_train = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["train"])
    df_valid = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["valid"])
    df_paraphrased = pd.read_csv("questions_paraphrased.csv")

    build_and_save(df_train, df_valid, df_paraphrased, window=2, step=1)