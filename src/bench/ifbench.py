"""Прогон модели на IFbench (300 вопросов)."""

import dataclasses
import json
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
 
def _generate(generate_fn: Callable[..., str], prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            answer = generate_fn(prompt=prompt) or ""
            return THINK_RE.sub("", answer).strip()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] | attempt {attempt + 1}/{retries} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return ""
 
 
def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["prompt"]: r["response"] for r in map(json.loads, f)}
 
 
def run_ifbench(
    generate_fn: Callable[..., str],
    ifbench_dir: str | Path = "IFBench",
    data: str | Path = "IFBench/data/IFBench_test.jsonl",
    responses_path: str | Path = "src/bench/log/ifbench.jsonl",
    workers: int = 8,
    strict: bool = False,
    limit: int | None = 50,
) -> tuple[float, float]:
    """Возвращает (score в %, время в секундах). generate_fn вызывается как generate_fn(prompt=...)."""
    ifbench_dir, responses_path = Path(ifbench_dir), Path(responses_path)
 
    if str(ifbench_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(ifbench_dir.resolve()))
    import evaluation_lib
 
    inputs = [
        dataclasses.replace(
            x, kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in x.kwargs]
        )
        for x in evaluation_lib.read_prompt_list(str(data))[:limit]
    ]
 
    start = time.perf_counter()
 
    responses: dict[str, str] = {}
    todo = [x.prompt for x in inputs]
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        responses_path.open("w", encoding="utf-8") as f,
        ThreadPoolExecutor(workers) as pool,
        tqdm(total=len(inputs), desc="IFBench", unit="req", ncols=80) as bar,
    ):
        futures = {pool.submit(_generate, generate_fn, prompt): prompt for prompt in todo}
        for future in as_completed(futures):
            prompt, response = futures[future], future.result()
            responses[prompt] = response
            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            f.flush()
            bar.update()
 
    check = (
        evaluation_lib.test_instruction_following_strict
        if strict
        else evaluation_lib.test_instruction_following_loose
    )
    passed = sum(check(x, responses).follow_all_instructions for x in inputs)
 
    score = 100 * passed / len(inputs)
    elapsed = time.perf_counter() - start
    
    return score, elapsed