from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import time
from typing import Callable

from .model import Evaluator, FitnessSpec, Metrics, Problem, Visit


PARAMETERS = {
    "GA": {"population": 20, "tournament": 3, "crossover": 0.9, "mutation": 0.2, "elite": 2},
    "PSO": {"population": 20, "inertia": 0.7, "c1": 1.5, "c2": 1.5, "velocity_limit": 0.2},
    "ACO": {"population": 20, "alpha": 1.0, "beta": 2.0, "evaporation": 0.2, "initial_pheromone": 1.0},
}


@dataclass
class Snapshot:
    evaluations: int
    iteration: int
    best_score: float
    best_order: tuple[str, ...]
    details: dict


@dataclass
class RunResult:
    algorithm: str
    seed: int
    fitness: FitnessSpec
    budget: int
    evaluations: int
    best_score: float
    best_order: tuple[str, ...]
    visits: tuple[Visit, ...]
    metrics: Metrics
    elapsed: float
    history: list[tuple[int, float]]
    denominators: tuple[float, float, float]
    parameters: dict
    snapshots: list[Snapshot] = field(default_factory=list)

    def to_dict(self):
        data = asdict(self)
        data["fitness"]["normalized_weights"] = self.fitness.normalized_weights
        return data


class Cancelled(Exception):
    pass


def decode_keys(ids, keys) -> tuple[str, ...]:
    return tuple(ids[i] for i in sorted(range(len(ids)), key=lambda i: (keys[i], ids[i])))


def order_crossover(a, b, rng):
    if len(a) < 2:
        return tuple(a)
    left, right = sorted(rng.sample(range(len(a)), 2))
    right += 1
    child = [None] * len(a)
    child[left:right] = a[left:right]
    chosen = set(a[left:right])
    candidates = [b[(right + k) % len(b)] for k in range(len(b)) if b[(right + k) % len(b)] not in chosen]
    positions = [(right + k) % len(a) for k in range(len(a)) if child[(right + k) % len(a)] is None]
    for position, pid in zip(positions, candidates):
        child[position] = pid
    return tuple(child)


class _Recorder:
    def __init__(self, evaluator, budget, observer, cancel_event):
        self.evaluator = evaluator
        self.budget = budget
        self.observer = observer
        self.cancel_event = cancel_event
        self.count = 0
        self.best_score = math.inf
        self.best_order = ()
        self.best_visits = ()
        self.best_metrics = None
        self.history = []
        self.snapshots = []
        self.observer_elapsed = 0.0

    def check(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise Cancelled("Perhitungan dibatalkan.")

    def evaluate(self, order):
        self.check()
        score, visits, metrics = self.evaluator.evaluate(order)
        self.count += 1
        if score < self.best_score:
            self.best_score, self.best_order = score, tuple(order)
            self.best_visits, self.best_metrics = visits, metrics
        self.history.append((self.count, self.best_score))
        return score

    def snapshot(self, iteration, details_factory):
        self.check()
        if self.observer is None:
            return
        started = time.perf_counter()
        snapshot = Snapshot(self.count, iteration, self.best_score, self.best_order, details_factory())
        self.snapshots.append(snapshot)
        self.observer(snapshot)
        self.observer_elapsed += time.perf_counter() - started


def _ga(ids, rng, rec):
    cfg = PARAMETERS["GA"]
    population = [tuple(rng.sample(ids, len(ids))) for _ in range(cfg["population"])]
    scores = [rec.evaluate(p) for p in population]
    iteration = 0
    detail = {"parents": [], "before_mutation": [], "child": [], "swapped": []}
    rec.snapshot(iteration, lambda: {**detail, "scores": list(scores)})
    while rec.count < rec.budget:
        ranked = sorted(range(len(population)), key=lambda i: (scores[i], population[i]))
        next_population = [population[i] for i in ranked[:cfg["elite"]]]
        next_scores = [scores[i] for i in ranked[:cfg["elite"]]]

        def select():
            contenders = rng.sample(range(len(population)), cfg["tournament"])
            return population[min(contenders, key=lambda i: scores[i])]

        while len(next_population) < cfg["population"] and rec.count < rec.budget:
            a, b = select(), select()
            child = order_crossover(a, b, rng) if rng.random() < cfg["crossover"] else tuple(a)
            before = child
            swapped = []
            if len(ids) > 1 and rng.random() < cfg["mutation"]:
                i, j = rng.sample(range(len(ids)), 2)
                mutable = list(child)
                mutable[i], mutable[j] = mutable[j], mutable[i]
                child, swapped = tuple(mutable), [i, j]
            next_population.append(child)
            next_scores.append(rec.evaluate(child))
            detail = {"parents": [a, b], "before_mutation": before, "child": child, "swapped": swapped}
        population, scores = next_population, next_scores
        iteration += 1
        rec.snapshot(iteration, lambda: {**detail, "scores": list(scores)})


def _pso(ids, rng, rec):
    cfg = PARAMETERS["PSO"]
    positions = [[rng.random() for _ in ids] for _ in range(cfg["population"])]
    velocities = [[rng.uniform(-0.2, 0.2) for _ in ids] for _ in positions]
    pbest = [list(x) for x in positions]
    scores = [rec.evaluate(decode_keys(ids, x)) for x in positions]
    best_index = min(range(len(scores)), key=lambda i: scores[i])
    gbest, global_score = list(pbest[best_index]), scores[best_index]
    iteration = 0

    def details():
        return {"points": [[x[0], x[1] if len(ids) > 1 else 0.5] for x in positions],
                "velocities": [[v[0], v[1] if len(ids) > 1 else 0.0] for v in velocities],
                "personal_best": [[x[0], x[1] if len(ids) > 1 else 0.5] for x in pbest],
                "global_best": [gbest[0], gbest[1] if len(ids) > 1 else 0.5],
                "dimensions": [ids[0], ids[1] if len(ids) > 1 else "—"]}

    rec.snapshot(iteration, details)
    while rec.count < rec.budget:
        for i, position in enumerate(positions):
            if rec.count >= rec.budget:
                break
            for j in range(len(ids)):
                velocity = (cfg["inertia"] * velocities[i][j]
                            + cfg["c1"] * rng.random() * (pbest[i][j] - position[j])
                            + cfg["c2"] * rng.random() * (gbest[j] - position[j]))
                velocity = max(-cfg["velocity_limit"], min(cfg["velocity_limit"], velocity))
                raw = position[j] + velocity
                position[j] = max(0.0, min(1.0, raw))
                velocities[i][j] = velocity if 0 <= raw <= 1 else 0.0
            score = rec.evaluate(decode_keys(ids, position))
            if score < scores[i]:
                pbest[i], scores[i] = list(position), score
            if score < global_score:
                gbest, global_score = list(position), score
        iteration += 1
        rec.snapshot(iteration, details)


def _aco(ids, rng, rec):
    cfg = PARAMETERS["ACO"]
    n = len(ids)
    tau = [[cfg["initial_pheromone"]] * n for _ in range(n + 1)]
    patients = {p.id: p for p in rec.evaluator.problem.patients}
    eta = [patients[pid].weight / patients[pid].duration for pid in ids]
    iteration = 0
    while rec.count < rec.budget:
        colony = []
        for _ in range(min(cfg["population"], rec.budget - rec.count)):
            remaining = list(range(n))
            path = []
            previous = n
            while remaining:
                probabilities = [tau[previous][j] ** cfg["alpha"] * eta[j] ** cfg["beta"] for j in remaining]
                selected = rng.choices(remaining, weights=probabilities)[0]
                path.append(selected)
                remaining.remove(selected)
                previous = selected
            order = tuple(ids[j] for j in path)
            colony.append((path, rec.evaluate(order)))
        for row in tau:
            for j in range(n):
                row[j] = max(1e-12, row[j] * (1 - cfg["evaporation"]))
        for path, score in colony:
            previous = n
            for j in path:
                tau[previous][j] += 1 / (1 + score)
                previous = j
        iteration += 1
        best_path, _ = min(colony, key=lambda item: item[1])
        rec.snapshot(iteration, lambda: {
            "pheromone": [row[:20] for row in tau[:min(n, 20)]],
            "labels": list(ids[:20]), "patient_count": n,
            "ant_order": [ids[j] for j in best_path],
        })


def optimize(problem: Problem, algorithm: str, fitness: FitnessSpec | None = None,
             seed: int = 0, evaluation_budget: int = 2000,
             observer: Callable[[Snapshot], None] | None = None, cancel_event=None) -> RunResult:
    if algorithm not in PARAMETERS:
        raise ValueError("Algoritma harus GA, PSO, atau ACO.")
    if not isinstance(evaluation_budget, int) or evaluation_budget < 20:
        raise ValueError("Anggaran evaluasi harus berupa bilangan bulat minimal 20.")
    evaluator = Evaluator(problem, fitness or FitnessSpec())
    rec = _Recorder(evaluator, evaluation_budget, observer, cancel_event)
    rec.check()
    ids = tuple(sorted(p.id for p in problem.patients))
    started = time.perf_counter()
    {"GA": _ga, "PSO": _pso, "ACO": _aco}[algorithm](ids, random.Random(seed), rec)
    elapsed = max(0.0, time.perf_counter() - started - rec.observer_elapsed)
    return RunResult(algorithm, seed, evaluator.spec, evaluation_budget, rec.count, rec.best_score,
                     rec.best_order, rec.best_visits, rec.best_metrics, elapsed, rec.history,
                     evaluator.denominators, dict(PARAMETERS[algorithm]), rec.snapshots)
