"""Eksperimen tanpa UI: python benchmark.py --all-fitness --presets."""
import argparse
from datetime import datetime
from pathlib import Path

from klinik.experiments import PRESETS, export_batch, run_experiments
from klinik.model import MODES, generate_problem


def main():
    parser = argparse.ArgumentParser(description="Perbandingan GA, PSO, ACO pada antrean klinik")
    parser.add_argument("--patients", type=int, default=20)
    parser.add_argument("--doctors", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fitness", choices=list(MODES), default="weighted")
    parser.add_argument("--weights", nargs=3, type=float, default=[0.5, 0.3, 0.2])
    parser.add_argument("--all-fitness", action="store_true")
    parser.add_argument("--presets", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Folder baru untuk ekspor; tidak boleh sudah ada")
    args = parser.parse_args()
    try:
        problems = ({name: generate_problem(n, d, h, s) for name, n, d, h, s in PRESETS} if args.presets else
                    {"Kustom": generate_problem(args.patients, args.doctors, args.horizon, args.seed)})
        batch = run_experiments(problems, list(MODES) if args.all_fitness else [args.fitness],
                                tuple(args.weights), args.repeats, args.budget,
                                progress=lambda done, total, label: print(f"[{done}/{total}] {label.replace(' · ', ' | ')}", flush=True))
        for summary in batch.summaries():
            print(f"{summary['scenario']:8} {summary['fitness']:10} {summary['algorithm']:3} "
                  f"peringkat={summary['rank']} skor={summary['mean_score']:.4f} +/- {summary['std_score']:.4f} "
                  f"waktu={summary['mean_seconds']:.3f}s")
        output = args.output or Path("exports") / datetime.now().strftime("benchmark-%Y%m%d-%H%M%S-%f")
        for path in export_batch(batch, output):
            print(path.resolve())
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Gagal: {exc}\n")


if __name__ == "__main__":
    main()
