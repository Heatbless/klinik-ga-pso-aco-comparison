import itertools
import math
import unittest

from klinik.model import Evaluator, FitnessSpec, Patient, Problem, fcfs_order, generate_problem, simulate


class ModelTests(unittest.TestCase):
    def test_hand_calculated_priority_example(self):
        problem = Problem((Patient("P001", 0, 10, 1), Patient("P002", 0, 5, 2), Patient("P003", 0, 5, 4)), 1)
        visits, metrics = simulate(problem, ("P003", "P002", "P001"))
        self.assertEqual([v.start for v in visits], [0, 5, 10])
        self.assertAlmostEqual(metrics.weighted, 20 / 7)
        self.assertEqual(metrics.maximum, 10)
        self.assertEqual(metrics.completion, 20)
        self.assertEqual(metrics.average, 5)
        self.assertEqual(metrics.per_priority, {1: 10, 2: 5, 4: 0})

    def test_available_patient_is_not_blocked_by_future_priority(self):
        problem = Problem((Patient("now", 0, 5, 1), Patient("future", 20, 5, 4)), 1)
        visits, metrics = simulate(problem, ("future", "now"))
        self.assertEqual([(v.patient_id, v.start) for v in visits], [("now", 0), ("future", 20)])
        self.assertEqual(metrics.weighted, 0)

    def test_simultaneous_arrivals_and_completions(self):
        problem = Problem((Patient("A", 0, 5, 1), Patient("B", 0, 5, 1),
                           Patient("C", 5, 2, 4), Patient("D", 5, 2, 1)), 2)
        visits, _ = simulate(problem, ("C", "D", "B", "A"))
        self.assertEqual([(v.patient_id, v.doctor_id, v.start) for v in visits],
                         [("B", 1, 0), ("A", 2, 0), ("C", 1, 5), ("D", 2, 5)])

    def test_single_patient_and_excess_doctors(self):
        visits, metrics = simulate(Problem((Patient("A", 15, 8, 2),), 4), ("A",))
        self.assertEqual((visits[0].doctor_id, visits[0].start, metrics.completion), (1, 15, 23))
        self.assertEqual(metrics.maximum, 0)

    def test_schedule_invariants_and_work_conservation(self):
        for seed in range(10):
            problem = generate_problem(30, 3, 60, seed)
            visits, _ = simulate(problem, reversed(fcfs_order(problem)))
            self.assertEqual(len({v.patient_id for v in visits}), 30)
            for v in visits:
                self.assertGreaterEqual(v.start, v.arrival)
                p = next(p for p in problem.patients if p.id == v.patient_id)
                self.assertEqual(v.end - v.start, p.duration)
            for doctor in range(1, 4):
                assigned = sorted((v for v in visits if v.doctor_id == doctor), key=lambda v: v.start)
                for a, b in zip(assigned, assigned[1:]):
                    self.assertLessEqual(a.end, b.start)
            for minute in range(int(max(v.end for v in visits))):
                waiting = any(v.arrival <= minute < v.start for v in visits)
                occupied = sum(v.start <= minute < v.end for v in visits)
                if waiting:
                    self.assertEqual(occupied, problem.doctors)

    def test_all_objectives_and_fcfs_normalization(self):
        problem = generate_problem()
        _, metrics = simulate(problem, fcfs_order(problem))
        for mode in ("weighted", "maximum", "completion"):
            self.assertEqual(Evaluator(problem, FitnessSpec(mode)).score(metrics), getattr(metrics, mode))
        evaluator = Evaluator(problem, FitnessSpec("combined", (5, 3, 2)))
        self.assertAlmostEqual(evaluator.score(metrics), 1)
        order = tuple(reversed(fcfs_order(problem)))
        _, candidate = simulate(problem, order)
        expected = sum(w * value / denominator for w, value, denominator in zip(
            (0.5, 0.3, 0.2), (candidate.weighted, candidate.maximum, candidate.completion), evaluator.denominators))
        self.assertAlmostEqual(evaluator.score(candidate), expected)

    def test_zero_wait_baseline_is_finite(self):
        problem = Problem((Patient("A", 0, 5, 1), Patient("B", 15, 5, 4)), 2)
        score, _, _ = Evaluator(problem, FitnessSpec("combined")).evaluate(("A", "B"))
        self.assertTrue(math.isfinite(score))
        self.assertAlmostEqual(score, 0.2)

    def test_exhaustive_small_case(self):
        problem = Problem((Patient("A", 0, 6, 1), Patient("B", 0, 2, 4), Patient("C", 0, 3, 2)), 1)
        evaluator = Evaluator(problem, FitnessSpec())
        scores = {order: evaluator.evaluate(order)[0] for order in itertools.permutations(("A", "B", "C"))}
        self.assertEqual(min(scores, key=scores.get), ("B", "C", "A"))
        self.assertAlmostEqual(min(scores.values()), 9 / 7)

    def test_invalid_inputs(self):
        problem = generate_problem(3)
        for order in (("P001",) * 3, ("P001",), ("X", "Y", "Z")):
            with self.assertRaises(ValueError):
                simulate(problem, order)
        for weights in ((0, 0, 0), (-1, 1, 1), (float("nan"), 1, 1), (float("inf"), 1, 1), (1e308, 1e308, 1e308)):
            with self.assertRaises(ValueError):
                FitnessSpec("combined", weights)
        for count, doctors, horizon in ((0, 2, 60), (101, 2, 60), (2, 0, 60), (2, 2, -1)):
            with self.assertRaises(ValueError):
                generate_problem(count, doctors, horizon)

    def test_generator_reproducibility(self):
        self.assertEqual(generate_problem(seed=42), generate_problem(seed=42))
        self.assertNotEqual(generate_problem(seed=42).patients, generate_problem(seed=43).patients)


if __name__ == "__main__":
    unittest.main()
