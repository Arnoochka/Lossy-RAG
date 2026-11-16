import json
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
VERDICT_RE = re.compile(r"(?i)correct\s*:\s*(yes|no)\b")

PROMPT = (
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


def _solve(generate_fn: Callable[..., str], judge_fn: Callable[..., str], row: dict) -> dict:
    answer = " OR ".join(dict.fromkeys(row["answer_text"]))
    response = _generate(generate_fn, PROMPT.format(question=row["question"]))
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
        "target": answer,
        "correct": bool(verdicts) and verdicts[-1].lower() == "yes",
        "response": response,
        "judgement": judgement,
    }


def run_squad(
    generate_fn: Callable[..., str],
    df: pd.DataFrame,
    judge_fn: Callable[..., str] | None = None,
    responses_path: str | Path = "log/squad.jsonl",
    workers: int = 8,
    limit: int | None = None,
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
        futures = [pool.submit(_solve, generate_fn, judge_fn, row) for row in rows]
        for future in as_completed(futures):
            record = future.result()
            correct += record["correct"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            bar.update()

    score = 100 * correct / len(rows)
    elapsed = time.perf_counter() - start

    return score, elapsed