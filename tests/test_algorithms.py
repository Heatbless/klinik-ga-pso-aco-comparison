import itertools
import random
import threading
import unittest

from klinik.algorithms import Cancelled, decode_keys, optimize, order_crossover
from klinik.model import Evaluator, FitnessSpec, MODES, generate_problem


class AlgorithmTests(unittest.TestCase):
    def test_permutation_operators(self):
        ids = tuple(f"P{i}" for i in range(10))
        rng = random.Random(7)
        for _ in range(100):
            a, b = rng.sample(ids, 10), rng.sample(ids, 10)
            child = order_crossover(a, b, rng)
            self.assertEqual(set(child), set(ids))
            self.assertEqual(len(child), len(ids))
        self.assertEqual(order_crossover(("A",), ("A",), rng), ("A",))
        self.assertEqual(decode_keys(("C", "B", "A"), (0.5, 0.5, 0.1)), ("A", "B", "C"))

    def test_budgets_fitness_and_reproducibility(self):
        problem = generate_problem(8, 2, 15)
        for algorithm in ("GA", "PSO", "ACO"):
            for mode in MODES:
                for budget in (20, 21, 57):
                    with self.subTest(algorithm=algorithm, mode=mode, budget=budget):
                        first = optimize(problem, algorithm, FitnessSpec(mode), 7, budget)
                        second = optimize(problem, algorithm, FitnessSpec(mode), 7, budget)
                        self.assertEqual(first.evaluations, budget)
                        self.assertEqual(len(first.history), budget)
                        self.assertEqual(first.history, second.history)
                        self.assertEqual(first.best_order, second.best_order)
                        self.assertEqual(set(first.best_order), {p.id for p in problem.patients})
                        self.assertTrue(all(a[1] >= b[1] for a, b in zip(first.history, first.history[1:])))
                        score, visits, metrics = Evaluator(problem, FitnessSpec(mode)).evaluate(first.best_order)
                        self.assertEqual(first.best_score, score)
                        self.assertEqual(first.visits, visits)
                        self.assertEqual(first.metrics, metrics)
                        self.assertEqual(first.snapshots, [])

    def test_single_patient_all_algorithms(self):
        for algorithm in ("GA", "PSO", "ACO"):
            result = optimize(generate_problem(1, 5, 0), algorithm, FitnessSpec("combined"), 0, 21, observer=lambda _: None)
            self.assertEqual(result.best_order, ("P001",))
            self.assertEqual(result.metrics.weighted, 0)
            self.assertEqual(result.snapshots[-1].evaluations, 21)

    def test_small_exhaustive_oracle(self):
        problem = generate_problem(5, 1, 0)
        evaluator = Evaluator(problem, FitnessSpec())
        optimum = min(evaluator.evaluate(p)[0] for p in itertools.permutations(p.id for p in problem.patients))
        for algorithm in ("GA", "PSO", "ACO"):
            result = optimize(problem, algorithm, evaluation_budget=80)
            self.assertGreaterEqual(result.best_score + 1e-10, optimum)

    def test_cancellation_before_and_during_run(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            optimize(generate_problem(), "GA", cancel_event=event)
        for algorithm in ("GA", "PSO", "ACO"):
            event.clear()
            with self.assertRaises(Cancelled):
                optimize(generate_problem(), algorithm, evaluation_budget=200,
                         observer=lambda _: event.set(), cancel_event=event)

    def test_snapshot_observer_does_not_change_search(self):
        for algorithm in ("GA", "PSO", "ACO"):
            problem = generate_problem()
            plain = optimize(problem, algorithm, seed=9, evaluation_budget=100)
            snapshots = []
            visual = optimize(problem, algorithm, seed=9, evaluation_budget=100, observer=snapshots.append)
            self.assertEqual(plain.history, visual.history)
            self.assertEqual(plain.best_order, visual.best_order)
            self.assertEqual(snapshots, visual.snapshots)
            self.assertEqual(snapshots[-1].evaluations, 100)


if __name__ == "__main__":
    unittest.main()
