"""Collect the best validation loss of every run in a sweep folder into one table.

Usage:
    uv run python scripts/generative/summarize_sweeps.py results/train/sweeps/mlp
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_value(value: str):
    """Turn '50' into 50 and '0.001' into 0.001; leave other strings (e.g. 'adam') as they are."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def collect_runs(sweep_dir: Path) -> list[dict]:
    """One row per run: its overrides as columns, plus its best validation loss."""
    rows = []

    # Every Hydra run has .hydra/overrides.yaml, even runs that crashed before writing metrics.json
    for overrides_file in sorted(sweep_dir.rglob(".hydra/overrides.yaml")):
        run_dir = overrides_file.parent.parent
        row = {}

        # overrides.yaml "model=mlp", "optimizer=adam", "optimizer.lr=0.001"
        overrides = yaml.safe_load(overrides_file.read_text()) or []
        for item in overrides:
            key, _, value = item.partition("=")
            row[key] = parse_value(value)

        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            row["best_val_loss"] = metrics["best_val_loss"]
            row["status"] = "done"
        else:
            row["best_val_loss"] = float("nan")
            row["status"] = "missing metrics.json"   # crashed or killed (e.g. HPC time limit)

        row["run"] = str(run_dir.relative_to(sweep_dir))
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("sweep_dir", type=Path,help="folder to search, e.g. results/train/sweeps/mlp")
    
    parser.add_argument("--out", type=Path, default=None, help="where to save the CSV (default: <sweep_dir>/summary.csv)")
    
    args = parser.parse_args()

    rows = collect_runs(args.sweep_dir)
    
    if not rows:
        
        print(f"No runs found under {args.sweep_dir}")
        
        return

    table = pd.DataFrame(rows)

    # Column order: best_val_loss first, settings in the middle, status and path at the end
    
    first, last = ["best_val_loss"], ["status", "run"]
    
    middle = [c for c in table.columns if c not in first + last]
    
    table = table[first + middle + last]

    # Best run on top; runs without metrics (NaN) at the bottom
    
    table = table.sort_values("best_val_loss", na_position="last")

    print(table.to_string(index=False))

    out = args.out or args.sweep_dir / "summary.csv"
    
    table.to_csv(out, index=False)
    
    print(f"\nSaved table to {out}")


if __name__ == "__main__":
    main()