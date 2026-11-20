"""Прогон модели на датасете squad"""

import json
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
VERDICT_RE = re.compile(r"(?i)correct\s*:\s*(yes|no)\b")

PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your answer choice}}\n"
    "Answer: {{your chosen answer}}\n"
    "Confidence: {{your confidence score between 0% and 100% for your answer}}\n\n"
    "{question}"
)

PROMPT_WITH_CONTEXT = (
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Your response should be in the following format:\n"
    "Explanation: {{your explanation for your answer choice}}\n"
    "Answer: {{your chosen answer}}\n"
    "Confidence: {{your confidence score between 0% and 100% for your answer}}\n\n"
    "{question}"
)

JUDGE_PROMPT = (
    "You are grading an answer to a question. Compare the final answer given in [response] "
    "with [correct_answer]. Do not solve the question yourself and do not argue for any other "
    "answer: only decide whether the two answers are equivalent. Small numerical rounding "
    "differences are acceptable; a missing, ambiguous or different answer is incorrect.\n\n"
    "[question]: {question}\n\n"
    "[response]: {response}\n\n"
    "[correct_answer]: {answer}\n\n"
    "Reply in exactly this format:\n"
    "extracted_final_answer: <final answer taken from the response, or None>\n"
    "reasoning: <one or two sentences>\n"
    "correct: <yes or no>"
)


def _generate(generate_fn: Callable[..., str], prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            answer = generate_fn(prompt=prompt) or ""
            return THINK_RE.sub("", answer).strip()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] | attempt {attempt + 1}/{retries} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return ""


def _solve(
    generate_fn: Callable[..., str],
    judge_fn: Callable[..., str],
    row: dict,
    context: str | None = None,
) -> dict:
    answer = " OR ".join(dict.fromkeys(row["answer_text"]))
    if context is None:
        prompt = PROMPT.format(question=row["question"])
    else:
        prompt = PROMPT_WITH_CONTEXT.format(context=context, question=row["question"])
    response = _generate(generate_fn, prompt)
    judgement = ""
    if response:
        judgement = _generate(
            judge_fn,
            JUDGE_PROMPT.format(question=row["question"], response=response, answer=answer),
        )
    verdicts = VERDICT_RE.findall(judgement.replace("*", ""))
    return {
        "qid": row["qid"],
        "orig_qid": row["orig_qid"],
        "is_paraphrase": row["is_paraphrase"],
        "question": row["question"],
        "context": context,
        "target": answer,
        "correct": bool(verdicts) and verdicts[-1].lower() == "yes",
        "response": response,
        "judgement": judgement,
    }

def print_record(record: dict) -> None:
    print("─" * 80)
    for key, value in record.items():
        text = str(value).replace("\n", "\n" + " " * 16)
        print(f"{key:>14}: {text}")
        
def run_squad(
    generate_fn: Callable[..., str],
    df: pd.DataFrame,
    judge_fn: Callable[..., str] | None = None,
    context_column: str | None = None,
    responses_path: str | Path = "log/squad.jsonl",
    workers: int = 1,
    limit: int | None = None,
    log=None,
) -> tuple[float, float]:
    responses_path = Path(responses_path)
    judge_fn = judge_fn or generate_fn

    rows = df.to_dict("records")[:limit]

    start = time.perf_counter()

    correct = 0
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        responses_path.open("w", encoding="utf-8") as f,
        ThreadPoolExecutor(workers) as pool,
        tqdm(total=len(rows), desc=responses_path.stem, unit="req") as bar,
    ):
        futures = [
            pool.submit(_solve, generate_fn, judge_fn, row, row[context_column] if context_column else None)
            for row in rows
        ]
        for future in as_completed(futures):
            record = future.result()
            with log or nullcontext():
                correct += record["correct"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                bar.update()

    score = 100 * correct / len(rows)
    elapsed = time.perf_counter() - start

    return score, elapsed