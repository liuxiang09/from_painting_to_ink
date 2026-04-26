from __future__ import annotations

import csv
from pathlib import Path


def save_history_csv(history: list[dict[str, float]], out_path: Path) -> None:
    if not history:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_metric_groups(
    history: list[dict[str, float]],
    out_dir: Path,
    plots: list[tuple[str, str, list[str]]],
) -> None:
    if not history:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    for file_name, title, keys in plots:
        plt.figure(figsize=(10, 6))
        for key in keys:
            values = [row[key] for row in history]
            plt.plot(epochs, values, marker="o", linewidth=2, markersize=4, label=key)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / file_name, dpi=160)
        plt.close()


def mean_metric_dict(metric_sums: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        raise ValueError("count must be positive")
    return {key: value / count for key, value in metric_sums.items()}
