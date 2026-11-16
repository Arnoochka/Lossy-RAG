"""Прогон модели на GPQA Diamond (198 вопросов, выбор из 4 вариантов). Idavidrein/gpqa"""

import csv
import json
import random
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(r"(?i)answer\s*:\s*\$?\(?([A-D])\b")
LETTERS = "ABCD"

PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be "
    "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n"
    "{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
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


def _load_rows(data: str | Path | None) -> list[dict]:
    if data is not None:
        with open(data, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    from datasets import load_dataset

    return list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))


def _build_prompt(row: dict, rng: random.Random) -> tuple[str, str]:
    """Возвращает (промпт, буква правильного ответа)."""
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    order = rng.sample(range(4), 4)
    shuffled = dict(zip(LETTERS, (choices[i].strip() for i in order)))
    prompt = PROMPT.format(question=row["Question"].strip(), **shuffled)
    return prompt, LETTERS[order.index(0)]


def _extract_answer(response: str) -> str | None:
    matches = ANSWER_RE.findall(response.replace("*", ""))
    return matches[-1].upper() if matches else None


def run_gpqa_diamond(
    generate_fn: Callable[..., str],
    data: str | Path | None = None,
    responses_path: str | Path = "src/bench/log/gpqa_diamond.jsonl",
    workers: int = 8,
    seed: int = 0,
    limit: int | None = 50,
) -> tuple[float, float]:
    """Возвращает (accuracy в %, время в секундах). generate_fn вызывается как generate_fn(prompt=...)."""
    responses_path = Path(responses_path)

    rng = random.Random(seed)
    tasks = [_build_prompt(row, rng) for row in _load_rows(data)][:limit]

    start = time.perf_counter()

    correct = 0
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        responses_path.open("w", encoding="utf-8") as f,
        ThreadPoolExecutor(workers) as pool,
        tqdm(total=len(tasks), desc="GPQA Diamond", unit="req", ncols=80) as bar,
    ):
        futures = {pool.submit(_generate, generate_fn, prompt): i for i, (prompt, _) in enumerate(tasks)}
        for future in as_completed(futures):
            i, response = futures[future], future.result()
            prompt, target = tasks[i]
            predicted = _extract_answer(response)
            correct += predicted == target
            record = {"id": i, "target": target, "predicted": predicted, "prompt": prompt, "response": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            bar.update()

    score = 100 * correct / len(tasks)
    elapsed = time.perf_counter() - start

    return score, elapsed