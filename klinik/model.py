from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import random
from typing import Iterable


MODES = {
    "weighted": "Waktu tunggu berbobot",
    "maximum": "Waktu tunggu maksimum",
    "completion": "Waktu seluruh pelayanan selesai",
    "combined": "Gabungan",
}


@dataclass(frozen=True)
class Patient:
    id: str
    arrival: int
    duration: int
    weight: int


@dataclass(frozen=True)
class Problem:
    patients: tuple[Patient, ...]
    doctors: int
    seed: int = 42
    horizon: int = 60

    def __post_init__(self):
        if not self.patients or self.doctors < 1:
            raise ValueError("Pasien dan dokter harus berjumlah minimal satu.")
        if len({p.id for p in self.patients}) != len(self.patients):
            raise ValueError("ID pasien harus unik.")
        if any(p.arrival < 0 or p.duration <= 0 or p.weight <= 0 for p in self.patients):
            raise ValueError("Kedatangan tidak boleh negatif; durasi dan bobot harus positif.")


@dataclass(frozen=True)
class Visit:
    patient_id: str
    doctor_id: int
    arrival: int
    start: int
    end: int
    weight: int

    @property
    def wait(self) -> int:
        return self.start - self.arrival


@dataclass(frozen=True)
class Metrics:
    weighted: float
    maximum: float
    completion: float
    average: float
    per_priority: dict[int, float]


def generate_problem(count=20, doctors=2, horizon=60, seed=42) -> Problem:
    if not 1 <= count <= 100 or not 1 <= doctors <= 10 or not 0 <= horizon <= 480:
        raise ValueError("Gunakan 1–100 pasien, 1–10 dokter, dan rentang kedatangan 0–480 menit.")
    rng = random.Random(seed)
    patients = tuple(
        Patient(f"P{i + 1:03}", rng.randint(0, horizon), rng.randint(5, 20),
                rng.choices((1, 2, 4), weights=(70, 20, 10))[0])
        for i in range(count)
    )
    return Problem(patients, doctors, seed, horizon)


def fcfs_order(problem: Problem) -> tuple[str, ...]:
    return tuple(p.id for p in sorted(problem.patients, key=lambda p: (p.arrival, p.id)))


def ratio_order(problem: Problem) -> tuple[str, ...]:
    return tuple(p.id for p in sorted(problem.patients, key=lambda p: (-p.weight / p.duration, p.arrival, p.id)))


def simulate(problem: Problem, order: Iterable[str]) -> tuple[tuple[Visit, ...], Metrics]:
    """Dispatch available patients by permutation rank without intentional idling."""
    order = tuple(order)
    if len(order) != len(problem.patients) or set(order) != {p.id for p in problem.patients}:
        raise ValueError("Solusi harus berupa permutasi seluruh ID pasien.")
    rank = {pid: i for i, pid in enumerate(order)}
    arrivals = sorted(problem.patients, key=lambda p: (p.arrival, p.id))
    by_id = {p.id: p for p in arrivals}
    waiting: list[tuple[int, str]] = []
    busy: list[tuple[int, int]] = []
    free = list(range(1, problem.doctors + 1))
    visits: list[Visit] = []
    cursor = 0
    now = 0
    while len(visits) < len(arrivals):
        while cursor < len(arrivals) and arrivals[cursor].arrival <= now:
            p = arrivals[cursor]
            heapq.heappush(waiting, (rank[p.id], p.id))
            cursor += 1
        while busy and busy[0][0] <= now:
            _, doctor = heapq.heappop(busy)
            free.append(doctor)
        free.sort()
        while free and waiting:
            doctor = free.pop(0)
            _, pid = heapq.heappop(waiting)
            p = by_id[pid]
            end = now + p.duration
            visits.append(Visit(pid, doctor, p.arrival, now, end, p.weight))
            heapq.heappush(busy, (end, doctor))
        if len(visits) < len(arrivals):
            events = ([arrivals[cursor].arrival] if cursor < len(arrivals) else [])
            if busy:
                events.append(busy[0][0])
            now = min(events)
    metrics = Metrics(
        weighted=sum(v.weight * v.wait for v in visits) / sum(v.weight for v in visits),
        maximum=float(max(v.wait for v in visits)),
        completion=float(max(v.end for v in visits)),
        average=sum(v.wait for v in visits) / len(visits),
        per_priority={w: sum(v.wait for v in visits if v.weight == w) / sum(v.weight == w for v in visits)
                      for w in (1, 2, 4) if any(v.weight == w for v in visits)},
    )
    return tuple(visits), metrics


@dataclass(frozen=True)
class FitnessSpec:
    mode: str = "weighted"
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)

    def __post_init__(self):
        import math
        if self.mode not in MODES:
            raise ValueError("Mode fitness tidak dikenal.")
        if (len(self.weights) != 3 or any(not math.isfinite(w) or w < 0 for w in self.weights)
                or not math.isfinite(sum(self.weights)) or sum(self.weights) <= 0):
            raise ValueError("Bobot harus berupa tiga angka nonnegatif; jumlahnya harus lebih dari nol.")

    @property
    def normalized_weights(self) -> tuple[float, float, float]:
        total = sum(self.weights)
        return tuple(w / total for w in self.weights)


class Evaluator:
    def __init__(self, problem: Problem, spec: FitnessSpec):
        self.problem = problem
        self.spec = spec
        self.baseline_visits, self.baseline_metrics = simulate(problem, fcfs_order(problem))
        self.denominators = tuple(max(getattr(self.baseline_metrics, k), 1.0)
                                  for k in ("weighted", "maximum", "completion"))

    def score(self, metrics: Metrics) -> float:
        if self.spec.mode != "combined":
            return getattr(metrics, self.spec.mode)
        values = (metrics.weighted, metrics.maximum, metrics.completion)
        return sum(w * value / denominator for w, value, denominator in
                   zip(self.spec.normalized_weights, values, self.denominators))

    def evaluate(self, order):
        visits, metrics = simulate(self.problem, order)
        return self.score(metrics), visits, metrics


def problem_dict(problem: Problem) -> dict:
    return asdict(problem)
