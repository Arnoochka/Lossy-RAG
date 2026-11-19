
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("log/model_answers")
EXPECTED_QUESTIONS = 50

CONTEXT_TYPES = {
    "no_context": "No Context",
    "rag": "RAG",
    "full_context": "Full Context",
}

PALETTE = {
    "no_context": "tab:blue",
    "rag": "tab:orange",
    "full_context": "tab:green",
}

QUESTION_TYPES = {
    "original": "Original",
    "paraphrase": "Paraphrase",
}



def split_filename(path):
    stem = path.stem

    match = re.fullmatch(
        r"(?P<bench>.+?)_(?P<paraphrase>p_)?"
        r"(?P<context>no_context|full_context|rag)_"
        r"(?P<model>.+)",
        stem,
    )

    if not match:
        return None

    data = match.groupdict()

    return {
        "bench": data["bench"],
        "question_type": (
            "paraphrase" if data["paraphrase"] else "original"
        ),
        "context": data["context"],
        "model": data["model"],
    }


def parse_file(path):
    metadata = split_filename(path)

    if metadata is None:
        print(f"[WARN] Неизвестный формат имени: {path.name}", file=sys.stderr)
        return None

    hits = 0
    records = 0
    skipped = 0

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARN] {path.name}:{line_number} — битая строка",
                    file=sys.stderr,
                )
                continue

            if "correct" not in record or not isinstance(record["correct"], bool):
                skipped += 1
                continue

            records += 1
            hits += int(record["correct"])

    if skipped:
        print(
            f"[WARN] {path.name}: пропущено {skipped} записей",
            file=sys.stderr,
        )

    if records != EXPECTED_QUESTIONS:
        print(
            f"[WARN] {path.name}: найдено {records} оцениваемых записей "
            f"вместо {EXPECTED_QUESTIONS}",
            file=sys.stderr,
        )

    metadata.update({
        "score": hits / EXPECTED_QUESTIONS * 100,
        "hits": hits,
        "total": records,
        "skipped": skipped,
        "file": path.name,
    })

    return metadata


def collect_results(log_dir=LOG_DIR):
    files = sorted(log_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"Нет JSONL-файлов в {log_dir}")

    results = []

    for path in files:
        result = parse_file(path)

        if result is not None:
            results.append(result)

    if not results:
        raise ValueError("Не удалось обработать ни одного файла")

    return results


def print_table(results):
    headers = (
        "Benchmark",
        "Questions",
        "Context",
        "Model",
        "Correct",
        "Score",
    )

    rows = []

    for result in sorted(
        results,
        key=lambda r: (
            r["bench"],
            r["question_type"],
            r["model"],
            r["context"],
        ),
    ):
        rows.append((
            result["bench"],
            QUESTION_TYPES[result["question_type"]],
            CONTEXT_TYPES[result["context"]],
            result["model"],
            f'{result["hits"]}/{EXPECTED_QUESTIONS}',
            f'{result["score"]:.1f}',
        ))

    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]

    def format_row(row):
        return "| " + " | ".join(
            str(value).ljust(width)
            for value, width in zip(row, widths)
        ) + " |"

    print(format_row(headers))
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")

    for row in rows:
        print(format_row(row))



def plot(results):
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 18,
    })
    benches = sorted({r["bench"] for r in results})

    question_types = (
        "original",
        "paraphrase",
    )

    for bench in benches:
        for question_type in question_types:
            subset = [
                r for r in results
                if r["bench"] == bench
                and r["question_type"] == question_type
            ]

            if not subset:
                continue

            models = sorted(
                {r["model"] for r in subset},
                key=str.lower,
            )

            scores = {
                (r["model"], r["context"]): r["score"]
                for r in subset
            }

            fig, ax = plt.subplots(
                figsize=(max(9, len(models) * 2.2), 6),
                facecolor="#fcfcfb",
            )

            ax.set_facecolor("#fcfcfb")

            x = np.arange(len(models))
            width = 0.24

            for i, context in enumerate(CONTEXT_TYPES):
                positions = x + (i - 1) * width

                values = [
                    scores.get((model, context))
                    for model in models
                ]

                valid = [
                    (position, value)
                    for position, value in zip(positions, values)
                    if value is not None
                ]

                if not valid:
                    continue

                bars = ax.bar(
                    [p for p, _ in valid],
                    [v for _, v in valid],
                    width=width,
                    color=PALETTE[context],
                    label=CONTEXT_TYPES[context],
                    edgecolor="#fcfcfb",
                    linewidth=1.5,
                )

                ax.bar_label(
                    bars,
                    fmt="%.1f",
                    padding=3,
                    fontsize=14,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                models,
                rotation=0,
                ha="center",
                fontweight="bold",
                fontsize=11,
            )

            ax.set_ylim(0, 110)
            ax.set_ylabel("Точность (%)")

            title_type = QUESTION_TYPES[question_type]


            ax.yaxis.grid(
                color="#e6e5e0",
                linewidth=0.8,
            )

            ax.set_axisbelow(True)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", length=0)

            ax.legend(
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=3,
            )

            for i in range(len(models) - 1):
                ax.axvline(
                    i + 0.5,
                    color="#e6e5e0",
                    linewidth=0.8,
                    linestyle="--",
                )

            fig.tight_layout()
            plt.show()



def summary(log_dir=LOG_DIR, show_table=True, show_plot=True):
    results = collect_results(log_dir)

    if show_table:
        print_table(results)

    if show_plot:
        plot(results)

    return results


if __name__ == "__main__":
    summary()
