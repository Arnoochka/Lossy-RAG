"""Сводка результатов бенчмарков из src/bench/log/*.jsonl."""
import os
import dataclasses
import functools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

# Длинные префиксы раньше коротких: имена моделей тоже содержат "_" (Q3_K_XL).
BENCHES = sorted(("aime", "gpqa_diamond", "ifbench"), key=len, reverse=True)
SCORE_KEYS = ("score", "correct", "passed", "follow_all_instructions")

IFBenchMode = Literal["loose", "strict"]


def split_name(stem: str) -> tuple[str, str]:
    for b in BENCHES:
        if stem.startswith(b + "_"):
            return b, stem[len(b) + 1:]
    bench, _, model = stem.partition("_")
    return bench, model


@functools.cache
def ifbench_judge(repo: Path, mode: IFBenchMode) -> Callable[[str, str], bool | None]:
    """Официальная проверка IFBench (evaluation_lib) по датасету из repo/data."""
    sys.path.insert(0, str(repo.resolve()))
    try:
        import evaluation_lib
    except ImportError as e:
        raise ImportError(f"IFBench: {e}. Установите зависимости: pip install -r {repo}/requirements.txt") from e

    inputs = {i.prompt.strip(): i for i in evaluation_lib.read_prompt_list(repo / "data" / "IFBench_test.jsonl")}
    test = getattr(evaluation_lib, f"test_instruction_following_{mode}")

    def judge(prompt: str, response: str) -> bool | None:
        inp = inputs.get(prompt.strip())
        if inp is None:
            return None
        # evaluation_lib мутирует kwargs и в loose-режиме не чистит None
        inp = dataclasses.replace(inp, kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in inp.kwargs])
        return test(inp, {inp.prompt: response}).follow_all_instructions

    return judge


def is_correct(rec: dict, bench: str, ifbench_repo: Path, ifbench_mode: IFBenchMode) -> bool | None:
    for k in SCORE_KEYS:
        if k in rec:
            return bool(rec[k])
    if "target" in rec and "predicted" in rec:
        return rec["predicted"] is not None and str(rec["predicted"]).strip() == str(rec["target"]).strip()
    if bench == "ifbench" and "prompt" in rec and "response" in rec:
        return ifbench_judge(ifbench_repo, ifbench_mode)(rec["prompt"], rec["response"])
    return None


def parse(path: str | Path, ifbench_repo: str | Path = "IFBench", ifbench_mode: IFBenchMode = "loose") -> dict:
    path, ifbench_repo = Path(path), Path(ifbench_repo)
    bench, model = split_name(path.stem)
    hits, total, skipped, time_spend = 0, 0, 0, None
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] {path.name}:{n} битая строка", file=sys.stderr)
                continue
            if "time_spend" in rec and len(rec) == 1:
                time_spend = rec["time_spend"]
                continue
            ok = is_correct(rec, bench, ifbench_repo, ifbench_mode)
            if ok is None:
                skipped += 1
                continue
            total += 1
            hits += ok
    if skipped:
        print(f"[WARN] {path.name}: {skipped} записей не удалось оценить", file=sys.stderr)
    # AIME с несколькими attempt: доля верных по всем попыткам = avg pass@1
    return {
        "bench": bench,
        "model": model,
        "score": 100 * hits / total if total else None,
        "n": total,
        "time": time_spend,
    }


def print_table(rows: list[dict]) -> None:
    table = [("[INFO]", "Название модели", "Результат из 100", "время выполнения sec")]
    for r in rows:
        score = f"{r['score']:.1f}" if r["score"] is not None else "n/a"
        time = str(r["time"]) if r["time"] is not None else "n/a"
        table.append((r["bench"], r["model"], score, time))

    widths = [max(len(row[i]) for row in table) for i in range(4)]
    fmt = lambda row: "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
    print(fmt(table[0]))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in table[1:]:
        print(fmt(row))

def append_time_spend(path: str | Path, elapsed: float) -> None:
    """Дописывает {"time_spend": N} в конец jsonl; прежнюю строку time_spend заменяет."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    # Перезапуск не должен плодить дубли
    lines = [l for l in lines if l.strip() and not _is_time_spend(l)]
    lines.append(json.dumps({"time_spend": round(elapsed)}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_time_spend(line: str) -> bool:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(rec, dict) and rec.keys() == {"time_spend"}

# Категориальная палитра с проверкой на дальтонизм; цвет закреплён за моделью, а не за рангом.
PALETTE = ("tab:blue","tab:orange", "tab:green", "tab:brown", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "black", "black", "#e6e5e0"

def plot(rows: list[dict]) -> None:
    time_spend = {(r["bench"], r["model"]): r["time"] for r in rows}
    # Безвредный шум Qt под Wayland при закрытии окна; задаётся до импорта Qt
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    benches = sorted({r["bench"] for r in rows})
    models = sorted({r["model"] for r in rows}, key=str.lower)
    score = {(r["bench"], r["model"]): r["score"] for r in rows}
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)

    if len(models) <= len(PALETTE):
        # Группы = бенчмарки, внутри — модели по алфавиту, цвет = модель
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(benches) * len(models) + 2), 5.2), facecolor=SURFACE)
        style(ax)
        w = 0.8 / len(models)
        for i, (m, color) in enumerate(zip(models, PALETTE)):
            xs = [b + (i - (len(models) - 1) / 2) * w for b in range(len(benches))]
            vals = [score.get((bn, m)) for bn in benches]
            bars = ax.bar([x for x, v in zip(xs, vals) if v is not None], [v for v in vals if v is not None],
                          width=w, color=color, edgecolor=SURFACE, linewidth=2, label=m)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=14, color=INK_2)
        ax.set_xticks(range(len(benches)), benches)
        ax.tick_params(axis="x", length=0, labelsize=11, labelcolor=INK)
        ax.yaxis.grid(color=GRID, linewidth=0.8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Результат из 100")
        if len(models) > 1:
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                      ncol=min(len(models), 3), fontsize=10)
        fig_time, ax_time = plt.subplots(
        figsize=(max(7, 1.1 * len(benches) * len(models) + 2), 5.2),
        facecolor=SURFACE,
    )
        style(ax_time)

        w = 0.8 / len(models)

        for i, (m, color) in enumerate(zip(models, PALETTE)):
            xs = [
                b + (i - (len(models) - 1) / 2) * w
                for b in range(len(benches))
            ]
            vals = [time_spend.get((bn, m)) for bn in benches]
            bars = ax_time.bar(
                [x for x, v in zip(xs, vals) if v is not None],
                [v for v in vals if v is not None],
                width=w,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                label=m,
            )
            ax_time.bar_label(bars, fmt="%.1f", padding=3, fontsize=14, color=INK_2)

        ax_time.set_xticks(range(len(benches)), benches)
        ax_time.tick_params(axis="x", length=0, labelsize=11, labelcolor=INK)
        ax_time.yaxis.grid(color=GRID, linewidth=0.8)
        ax_time.set_ylabel("Время выполнения (сек)")

        if len(models) > 1:
            ax_time.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(len(models),3), fontsize=10)

        fig_time.tight_layout()
    else:
        # >8 моделей: цветом их не различить — отдельная панель на бенчмарк, имена на оси
        fig, axes = plt.subplots(1, len(benches), sharex=True, facecolor=SURFACE,
                                 figsize=(4.5 * len(benches) + 3, 0.32 * len(models) + 1.5), squeeze=False)
        for j, (ax, bn) in enumerate(zip(axes[0], benches)):
            style(ax)
            vals = [score.get((bn, m)) for m in models]
            ys = list(range(len(models)))[::-1]
            bars = ax.barh([y for y, v in zip(ys, vals) if v is not None], [v for v in vals if v is not None],
                           height=0.7, color=PALETTE[0], edgecolor=SURFACE, linewidth=2)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color=INK_2)
            ax.set_yticks(ys, models if j == 0 else [""] * len(models))
            ax.tick_params(axis="y", length=0)
            ax.xaxis.grid(color=GRID, linewidth=0.8)
            ax.set_xlim(0, 110)
            ax.set_title(bn, loc="left", fontsize=11, color=INK)

    fig.tight_layout()
    plt.show()


def summary(
    log_dir: str | Path = "log/model_answers",
    ifbench_repo: str | Path = "IFBench",
    ifbench_mode: IFBenchMode = "loose",  # loose — основная метрика в статье IFBench
    show_table: bool = True,
    show_plot: bool = True,
) -> list[dict]:
    """Собирает результаты из log_dir/*.jsonl; возвращает строки {bench, model, score, n, time}."""
    log_dir = Path(log_dir)
    files = [
        p for p in sorted(log_dir.glob("*.jsonl"))
        if not p.name.startswith("squad_")
    ]

    rows = [parse(p, ifbench_repo, ifbench_mode) for p in files]
    if not rows:
        raise FileNotFoundError(f"нет *.jsonl в {log_dir}")

    rows.sort(key=lambda r: (r["bench"], -(r["score"] if r["score"] is not None else -1)))
    if show_table:
        print_table(rows)
    if show_plot:
        plot(rows)
    return rows


if __name__ == "__main__":
    summary()