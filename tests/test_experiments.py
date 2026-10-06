import csv
import json
from pathlib import Path
import tempfile
import unittest

from klinik.experiments import export_batch, run_experiments
from klinik.model import MODES, generate_problem


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = run_experiments({"Uji": generate_problem(8, 2, 15)}, list(MODES), repeats=2, budget=40)

    def test_grouping_and_paired_runs(self):
        self.assertTrue(self.batch.complete)
        self.assertEqual(len(self.batch.runs), 24)
        self.assertEqual(len(self.batch.baselines), 8)
        summaries = self.batch.summaries()
        self.assertEqual(len(summaries), 12)
        self.assertTrue(all(s["runs"] == 2 for s in summaries))
        for mode in MODES:
            group = [s for s in summaries if s["fitness"] == mode]
            self.assertEqual(group[0]["rank"], 1)
            self.assertEqual([s["mean_score"] for s in group], sorted(s["mean_score"] for s in group))
        self.assertTrue(all(r.result.evaluations == 40 for r in self.batch.runs))
        for mode in MODES:
            results = [r for r in self.batch.runs if r.result.fitness.mode == mode]
            self.assertEqual([r.result.algorithm for r in results], ["GA", "PSO", "ACO", "PSO", "ACO", "GA"])
            self.assertEqual([r.result.seed for r in results], [0, 0, 0, 1, 1, 1])

    def test_export_matches_results_and_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "hasil"
            paths = export_batch(self.batch, target)
            self.assertTrue(all(p.is_file() for p in paths))
            with (target / "hasil.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 24)
            self.assertEqual(float(rows[0]["skor"]), self.batch.runs[0].result.best_score)
            with (target / "konvergensi.csv").open(encoding="utf-8-sig", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 24 * 40)
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["problems"]["Uji"]["seed"], 42)
            self.assertEqual(manifest["runs"][0]["result"]["best_order"], list(self.batch.runs[0].result.best_order))
            self.assertEqual(manifest["runs"][0]["result"]["fitness"]["normalized_weights"], [0.5, 0.3, 0.2])
            self.assertIn("denominators", manifest["runs"][0]["result"])
            self.assertNotIn("snapshots", manifest["runs"][0]["result"])
            before = (target / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                export_batch(self.batch, target)
            self.assertEqual(before, (target / "manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
