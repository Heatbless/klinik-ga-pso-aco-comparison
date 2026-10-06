from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from .algorithms import Cancelled, RunResult, optimize
from .model import Evaluator, FitnessSpec, MODES, Problem, fcfs_order, generate_problem, ratio_order


PRESETS = (("Kecil", 20, 2, 60, 42), ("Sedang", 50, 3, 120, 43), ("Besar", 100, 5, 180, 44))


@dataclass
class ExperimentRun:
    scenario: str
    repeat: int
    result: RunResult


@dataclass
class Baseline:
    scenario: str
    fitness: FitnessSpec
    name: str
    score: float
    metrics: dict
    visits: list[dict]


@dataclass
class Batch:
    problems: dict[str, Problem]
    runs: list[ExperimentRun] = field(default_factory=list)
    baselines: list[Baseline] = field(default_factory=list)
    complete: bool = False

    def groups(self):
        return sorted({(r.scenario, r.result.fitness.mode) for r in self.runs})

    def summaries(self):
        summaries = []
        for scenario, mode in self.groups():
            for algorithm in ("GA", "PSO", "ACO"):
                results = [r.result for r in self.runs if
                           (r.scenario, r.result.fitness.mode, r.result.algorithm) == (scenario, mode, algorithm)]
                if not results:
                    continue
                scores = [r.best_score for r in results]
                summaries.append({
                    "scenario": scenario, "fitness": mode, "algorithm": algorithm,
                    "runs": len(scores), "mean_score": statistics.mean(scores),
                    "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "best_score": min(scores),
                    "mean_seconds": statistics.mean(r.elapsed for r in results),
                    "mean_weighted_wait": statistics.mean(r.metrics.weighted for r in results),
                    "mean_max_wait": statistics.mean(r.metrics.maximum for r in results),
                    "mean_completion": statistics.mean(r.metrics.completion for r in results),
                })
        for scenario, mode in self.groups():
            group = sorted((s for s in summaries if (s["scenario"], s["fitness"]) == (scenario, mode)),
                           key=lambda s: (s["mean_score"], s["algorithm"]))
            previous = None
            rank = 0
            for index, summary in enumerate(group, 1):
                if previous is None or not math.isclose(summary["mean_score"], previous, rel_tol=1e-9, abs_tol=1e-9):
                    rank = index
                summary["rank"] = rank
                previous = summary["mean_score"]
        return sorted(summaries, key=lambda s: (s["scenario"], s["fitness"], s["rank"], s["algorithm"]))


def run_experiments(problems: dict[str, Problem], modes: list[str], weights=(0.5, 0.3, 0.2),
                    repeats=10, budget=2000, progress=None, cancel_event=None) -> Batch:
    if not 1 <= repeats <= 100 or not modes or len(set(modes)) != len(modes):
        raise ValueError("Gunakan 1–100 pengulangan dan mode fitness yang unik.")
    specs = [FitnessSpec(mode, tuple(weights)) for mode in modes]
    batch = Batch(dict(problems))
    total = len(problems) * len(specs) * repeats * 3
    completed = 0
    for scenario, problem in problems.items():
        for spec in specs:
            evaluator = Evaluator(problem, spec)
            for name, order in (("FCFS", fcfs_order(problem)), ("Prioritas/Durasi", ratio_order(problem))):
                score, visits, metrics = evaluator.evaluate(order)
                batch.baselines.append(Baseline(scenario, spec, name, score, asdict(metrics), [asdict(v) for v in visits]))
            for repeat in range(repeats):
                algorithms = ["GA", "PSO", "ACO"]
                rotation = repeat % 3
                algorithms = algorithms[rotation:] + algorithms[:rotation]
                for algorithm in algorithms:
                    if cancel_event is not None and cancel_event.is_set():
                        raise Cancelled()
                    result = optimize(problem, algorithm, spec, repeat, budget, cancel_event=cancel_event)
                    batch.runs.append(ExperimentRun(scenario, repeat + 1, result))
                    completed += 1
                    if progress:
                        progress(completed, total, f"{scenario} · {MODES[spec.mode]} · {algorithm} · ulangan {repeat + 1}")
    batch.complete = True
    return batch


def demo_batch(problem, results) -> Batch:
    batch = Batch({"Demo": problem}, [ExperimentRun("Demo", 1, result) for result in results], complete=True)
    if results:
        evaluator = Evaluator(problem, results[0].fitness)
        for name, order in (("FCFS", fcfs_order(problem)), ("Prioritas/Durasi", ratio_order(problem))):
            score, visits, metrics = evaluator.evaluate(order)
            batch.baselines.append(Baseline("Demo", results[0].fitness, name, score, asdict(metrics), [asdict(v) for v in visits]))
    return batch


def export_batch(batch: Batch, directory: str | Path) -> list[Path]:
    if not batch.complete or not batch.runs:
        raise ValueError("Hanya hasil yang telah selesai yang dapat diekspor.")
    directory = Path(directory)
    # Exclusive directory creation prevents accidental overwriting of previous exports.
    directory.mkdir(parents=True, exist_ok=False)
    paths = [directory / name for name in ("hasil.csv", "konvergensi.csv", "ringkasan.csv", "manifest.json")]
    with paths[0].open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["skenario", "fitness", "algoritma", "pengulangan", "seed_data", "seed_algoritma",
                         "evaluasi", "skor", "waktu_komputasi_detik", "tunggu_berbobot_menit",
                         "tunggu_rata_rata_menit", "tunggu_maksimum_menit", "selesai_menit",
                         "tunggu_prioritas_1", "tunggu_prioritas_2", "tunggu_prioritas_4"])
        for run in batch.runs:
            r, m = run.result, run.result.metrics
            writer.writerow([run.scenario, r.fitness.mode, r.algorithm, run.repeat, batch.problems[run.scenario].seed,
                             r.seed, r.evaluations, r.best_score, r.elapsed, m.weighted, m.average, m.maximum, m.completion,
                             *[m.per_priority.get(w, "") for w in (1, 2, 4)]])
    with paths[1].open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["skenario", "fitness", "algoritma", "pengulangan", "evaluasi", "skor_terbaik"])
        for run in batch.runs:
            for count, score in run.result.history:
                writer.writerow([run.scenario, run.result.fitness.mode, run.result.algorithm, run.repeat, count, score])
    summaries = batch.summaries()
    with paths[2].open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "complete": batch.complete,
        "runtime_note": "Waktu komputasi tidak mencakup rendering, snapshot visual, atau ekspor.",
        "problems": {name: asdict(problem) for name, problem in batch.problems.items()},
        "baselines": [asdict(b) for b in batch.baselines],
        "runs": [{"scenario": run.scenario, "repeat": run.repeat,
                  "result": {k: v for k, v in run.result.to_dict().items() if k != "snapshots"}}
                 for run in batch.runs],
    }
    with paths[3].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False)
    return paths
