"""Integration checks exercise Tk's event loop, worker queue and replay."""
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from klinik.model import MODES
from klinik.ui import ClinicApp


class UITests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display tidak tersedia: {exc}")
        self.root.withdraw()
        self.errors = []
        self.root.report_callback_exception = lambda kind, value, tb: self.errors.append(str(value))
        self.app = ClinicApp(self.root)
        self.app.count_var.set("8")
        self.app.horizon_var.set("15")
        self.app.budget_var.set("40")
        self.app.generate()
        self.root.update()

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.close()

    def finish_worker(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.root.update()
        self.assertFalse(self.app.busy, "Worker tidak selesai dalam 15 detik")
        self.assertEqual(self.errors, [])

    @patch("klinik.ui.messagebox.showerror")
    def test_demo_replays_every_algorithm_and_fitness(self, showerror):
        for mode in MODES:
            self.app.fitness_var.set(MODES[mode])
            self.app._fitness_changed()
            self.app.run_demo()
            self.finish_worker()
            self.assertEqual(set(self.app.demo_results), {"GA", "PSO", "ACO"})
            self.assertEqual(self.app.demo_results["GA"].fitness.mode, mode)
            for replay in ("Proses optimasi", "Pelayanan klinik"):
                self.app.replay_mode.set(replay)
                for fraction in (0, 0.5, 1):
                    self.app.cursor = self.app._limit() * fraction
                    self.app.render_demo()
                    self.root.update()
                    for canvas in self.app.panel_canvases.values():
                        self.assertTrue(canvas.find_all())
            self.app.reset_replay()
            self.app.step()
            self.assertGreater(self.app.cursor, 0)
            self.app.play()
            self.app.pause()
            self.assertIsNone(self.app.play_after)
        showerror.assert_not_called()

    @patch("klinik.ui.messagebox.showerror")
    def test_experiment_worker_all_modes_and_group_graphs(self, showerror):
        self.app.repeats_var.set("2")
        self.app.experiment_modes.set("Keempat fitness")
        self.app.run_experiment()
        self.finish_worker()
        self.assertEqual(len(self.app.batch.runs), 24)
        self.assertEqual(len(self.app.summary_tree.get_children()), 12)
        self.assertEqual(len(self.app.raw_tree.get_children()), 24)
        for group in self.app.group_lookup:
            self.app.group_var.set(group)
            self.app.render_experiment()
            self.assertTrue(self.app.experiment_chart.find_all())
        self.assertEqual(self.errors, [])
        showerror.assert_not_called()

    @patch("klinik.ui.messagebox.showerror")
    def test_cancel_restores_controls_and_keeps_previous_results(self, showerror):
        self.app.run_demo()
        self.finish_worker()
        previous = self.app.demo_results
        self.app.budget_var.set("100000")
        self.app.run_demo()
        self.app.cancel()
        self.finish_worker()
        self.assertIs(self.app.demo_results, previous)
        self.assertEqual(str(self.app.demo_run_button.cget("state")), "normal")
        self.assertIn("dibatalkan", self.app.status.get())
        showerror.assert_not_called()


if __name__ == "__main__":
    unittest.main()
