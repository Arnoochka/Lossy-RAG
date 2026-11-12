"""Прогон модели на AIME 2026 (MathArena/aime_2026): ответ на каждую задачу — целое число 0..999."""
import json
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
BOXED = "\\boxed{"

PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "The answer is an integer between 0 and 999 inclusive.\n\n"
    "{problem}"
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


def _load_rows(dataset: str) -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset(dataset, split="train"))


def _extract_answer(response: str) -> int | None:
    """Целое число из последнего \\boxed{...}; None, если его нет или внутри не одно число."""
    start = response.rfind(BOXED)
    if start == -1:
        return None
    # скобки могут быть вложенными (\boxed{\textbf{42}}), поэтому ищем парную вручную
    depth, begin = 1, start + len(BOXED)
    for end in range(begin, len(response)):
        depth += {"{": 1, "}": -1}.get(response[end], 0)
        if depth == 0:
            numbers = re.findall(r"\d+", response[begin:end].replace(",", ""))
            return int(numbers[0]) if len(numbers) == 1 else None
    return None


def run_aime(
    generate_fn: Callable[..., str],
    dataset: str = "MathArena/aime_2026",
    responses_path: str | Path = "src/bench/log/aime_2026_responses.jsonl",
    workers: int = 8,
    repeats: int = 4,
    limit: int | None = None,
) -> tuple[float, float]:
    """Возвращает (accuracy в %, время в секундах). generate_fn вызывается как generate_fn(prompt=...).

    Каждая задача решается repeats раз, итог — средняя доля верных ответов по всем попыткам.
    """
    responses_path = Path(responses_path)

    rows = _load_rows(dataset)[:limit]
    tasks = [(row, attempt) for row in rows for attempt in range(repeats)]

    start = time.perf_counter()

    correct = 0
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        responses_path.open("w", encoding="utf-8") as f,
        ThreadPoolExecutor(workers) as pool,
        tqdm(total=len(tasks), desc="AIME", unit="req", ncols=80) as bar,
    ):
        futures = {
            pool.submit(_generate, generate_fn, PROMPT.format(problem=row["problem"])): (row, attempt)
            for row, attempt in tasks
        }
        for future in as_completed(futures):
            (row, attempt), response = futures[future], future.result()
            target, predicted = int(row["answer"]), _extract_answer(response)
            correct += predicted == target
            record = {
                "problem_idx": row["problem_idx"],
                "attempt": attempt,
                "target": target,
                "predicted": predicted,
                "response": response,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            bar.update()

    score = 100 * correct / len(tasks)
    elapsed = time.perf_counter() - start

    return score, elapsed