"""Run reproducible DWA simulations and export poster-ready results."""

import argparse
import csv
from collections import Counter
from pathlib import Path
import time

import matplotlib.pyplot as plt

from dwa_simulation import main as run_simulation


RESULT_FIELDS = [
    "seed",
    "status",
    "success",
    "obstacle_count",
    "steps",
    "completion_time_s",
    "path_length_m",
    "minimum_clearance_m",
    "final_distance_to_goal_m",
    "wall_time_s",
]

INTEGER_FIELDS = {"seed", "success", "obstacle_count", "steps"}
FLOAT_FIELDS = set(RESULT_FIELDS) - INTEGER_FIELDS - {"status"}


def _read_checkpoint(csv_path):
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        for saved in csv.DictReader(csv_file):
            row = dict(saved)
            for field in INTEGER_FIELDS:
                row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def _save_outcome_plot(rows, output_path):
    counts = Counter(row["status"] for row in rows)
    statuses = ["goal_reached", "max_steps", "collision"]
    values = [counts.get(status, 0) for status in statuses]
    percentages = [100.0 * value / len(rows) for value in values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ["Goal reached", "Max steps", "Collision"],
        percentages,
        color=["tab:green", "tab:orange", "tab:red"],
    )
    ax.set_ylabel("Percentage of runs (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title(f"DWA outcomes across {len(rows)} random seeds")
    ax.grid(axis="y", alpha=0.3)
    for bar, percentage, count in zip(bars, percentages, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.0,
            f"{percentage:.1f}%\n(n={count})",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_success_boxplot(rows, field, ylabel, title, output_path):
    values = [row[field] for row in rows if row["success"] == 1]
    if not values:
        return False

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(values, tick_labels=["Successful runs"], showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def run_batch(
    seed_start, run_count, obstacle_count, output_directory, resume=False
):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "dwa_seed_results.csv"
    rows = _read_checkpoint(csv_path) if resume else []
    batch_start = time.perf_counter()
    requested_seeds = list(range(seed_start, seed_start + run_count))
    rows = [row for row in rows if row["seed"] in requested_seeds]
    completed_seeds = {row["seed"] for row in rows}
    pending_seeds = [seed for seed in requested_seeds if seed not in completed_seeds]

    if resume:
        print(
            f"Resuming with {len(completed_seeds)} completed seeds; "
            f"{len(pending_seeds)} remain"
        )

    file_mode = "a" if resume and csv_path.exists() else "w"
    with csv_path.open(file_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if file_mode == "w":
            writer.writeheader()
        csv_file.flush()

        for seed in pending_seeds:
            run_start = time.perf_counter()
            result = run_simulation(
                seed=seed,
                obstacle_count=obstacle_count,
                show_animation=False,
            )
            wall_time = time.perf_counter() - run_start
            row = {
                "seed": seed,
                "status": result["status"],
                "success": int(result["status"] == "goal_reached"),
                "obstacle_count": len(result["obstacles"]),
                "steps": result["steps"],
                "completion_time_s": result["completion_time_s"],
                "path_length_m": result["path_length_m"],
                "minimum_clearance_m": result["minimum_clearance"],
                "final_distance_to_goal_m": result["distance_to_goal"],
                "wall_time_s": wall_time,
            }
            rows.append(row)
            writer.writerow(row)
            # Preserve every completed seed if a later seed fails or the
            # process is interrupted.
            csv_file.flush()
            print(
                f"[{len(rows)}/{run_count}] seed={seed} "
                f"status={row['status']} "
                f"path={row['path_length_m']:.3f} m "
                f"wall={wall_time:.2f} s"
            )

    rows.sort(key=lambda row: row["seed"])

    _save_outcome_plot(rows, output_directory / "dwa_outcomes.png")
    _save_success_boxplot(
        rows,
        "completion_time_s",
        "Simulated completion time (s)",
        "Completion time for successful DWA runs",
        output_directory / "dwa_completion_time_boxplot.png",
    )
    _save_success_boxplot(
        rows,
        "path_length_m",
        "Path length (m)",
        "Path length for successful DWA runs",
        output_directory / "dwa_path_length_boxplot.png",
    )

    counts = Counter(row["status"] for row in rows)
    success_rate = 100.0 * counts.get("goal_reached", 0) / len(rows)
    elapsed = time.perf_counter() - batch_start
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Outcome counts: {dict(counts)}")
    print(f"Total wall time: {elapsed:.1f} seconds")
    print(f"Results directory: {output_directory.resolve()}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--obstacles", type=int, default=5)
    parser.add_argument("--output", default="dwa_batch_results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.obstacles < 0:
        parser.error("--obstacles cannot be negative")
    run_batch(
        args.seed_start,
        args.runs,
        args.obstacles,
        Path(args.output),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
