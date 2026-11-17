"""Precompute result sets for the presentation (run before the talk).

    python -m gui.precompute                       # course budget, seed 42
    python -m gui.precompute --budget extended --seeds 42 43 44
    python -m gui.precompute --all                 # the three standard sets

Result sets are written to ``results/gui/`` and appear in the app's selector.
"""
from __future__ import annotations

import argparse
import sys
import time

from gui.core.catalog import CONTROLLERS, OPTIMIZERS, PLANT_NAMES
from gui.core.engine import BUDGETS, execute_batch, save_result_set

PRESETS = {
    "course": ("Course default (≈18 evals)", "Course budget · seed 42", [42]),
    "extended": ("Extended (≈60 evals)", "Extended budget · 3 seeds", [42, 43, 44]),
    "thorough": ("Thorough (≈150 evals)", "Thorough budget · 3 seeds", [42, 43, 44]),
}


def run_preset(key: str, seeds=None, workers=None, name=None) -> None:
    budget_label, default_name, default_seeds = PRESETS[key]
    seeds = seeds or default_seeds
    name = name or (default_name if seeds == default_seeds else f"{budget_label.split(' (')[0]} · seeds {seeds}")
    pop, iters = BUDGETS[budget_label]
    started = time.perf_counter()

    def progress(done, total, rec):
        status = "ok " if rec.get("status") == "success" else "ERR"
        sys.stdout.write(f"\r[{key}] {done:>3}/{total}  {status} {rec['optimizer']:<4} "
                         f"{rec['controller']:<15} {rec['plant']:<13} seed {rec['seed']}   ")
        sys.stdout.flush()

    rs = execute_batch(name, list(OPTIMIZERS), list(CONTROLLERS), PLANT_NAMES, seeds, pop, iters,
                       workers=workers, on_progress=progress, budget_label=budget_label)
    path = save_result_set(rs)
    print(f"\n[{key}] saved {len(rs.records)} runs to {path} in {time.perf_counter() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", choices=list(PRESETS), default="course")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--all", action="store_true", help="build course, extended and thorough sets")
    args = parser.parse_args()
    if args.all:
        for key in PRESETS:
            run_preset(key, workers=args.workers)
    else:
        run_preset(args.budget, seeds=args.seeds, workers=args.workers, name=args.name)


if __name__ == "__main__":
    main()
