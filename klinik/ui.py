from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
from pathlib import Path
import queue
import statistics
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .algorithms import Cancelled, PARAMETERS, optimize
from .experiments import PRESETS, demo_batch, export_batch, run_experiments
from .model import Evaluator, FitnessSpec, MODES, fcfs_order, generate_problem, ratio_order


BG = "#f1f5f9"
INK = "#0f172a"
MUTED = "#64748b"
COLORS = {"GA": "#2563eb", "PSO": "#7c3aed", "ACO": "#059669"}
PRIORITY_COLORS = {1: "#bfdbfe", 2: "#fcd34d", 4: "#fca5a5"}


def draw_chart(canvas, series, baselines=(), title="Konvergensi — lebih kecil lebih baik", unit="Skor"):
    canvas.delete("all")
    width, height = max(canvas.winfo_width(), 350), max(canvas.winfo_height(), 170)
    canvas.create_text(14, 14, text=title, anchor="nw", fill=INK, font=("Segoe UI", 10, "bold"))
    if not series:
        canvas.create_text(width / 2, height / 2, text="Jalankan optimasi untuk menampilkan grafik.", fill=MUTED)
        return
    left, top, right, bottom = 68, 45, width - 20, height - 38
    values = [y for _, points in series for _, y in points] + [v for _, v in baselines]
    low, high = min(values), max(values)
    margin = max((high - low) * 0.08, abs(high) * 0.03, 0.01)
    low, high = max(0, low - margin), high + margin
    xmax = max(x for _, points in series for x, _ in points)

    def coord(x, y):
        return left + x / max(xmax, 1) * (right - left), bottom - (y - low) / (high - low) * (bottom - top)

    for k in range(5):
        y = low + k / 4 * (high - low)
        _, py = coord(0, y)
        canvas.create_line(left, py, right, py, fill="#e2e8f0")
        canvas.create_text(left - 7, py, text=f"{y:.2f}", anchor="e", fill=MUTED, font=("Segoe UI", 8))
    for k in range(5):
        x = round(k / 4 * xmax)
        px, _ = coord(x, low)
        canvas.create_text(px, bottom + 11, text=str(x), fill=MUTED, font=("Segoe UI", 8))
    canvas.create_text((left + right) / 2, height - 9, text="Jumlah evaluasi", fill=MUTED, font=("Segoe UI", 8))
    canvas.create_text(7, top - 12, text=unit, anchor="w", fill=MUTED, font=("Segoe UI", 8))
    legend_x = left
    for name, points in series:
        # Limit rendering cost while retaining the first and last evaluation.
        stride = max(1, len(points) // 400)
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled = sampled + [points[-1]]
        coordinates = [value for x, y in sampled for value in coord(x, y)]
        if len(coordinates) >= 4:
            canvas.create_line(*coordinates, fill=COLORS.get(name, INK), width=2)
        else:
            px, py = coordinates
            canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=COLORS.get(name, INK))
        canvas.create_text(legend_x, 28, text=name, anchor="w", fill=COLORS.get(name, INK), font=("Segoe UI", 9, "bold"))
        legend_x += 55
    for index, (name, value) in enumerate(baselines):
        _, py = coord(0, value)
        color = "#94a3b8" if index == 0 else "#c2410c"
        canvas.create_line(left, py, right, py, fill=color, dash=(5, 4))
        canvas.create_text(legend_x, 28, text=f"{name}: {value:.2f}", anchor="w", fill=color, font=("Segoe UI", 8))
        legend_x += 165


class ClinicApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Klinik Optima · GA / PSO / ACO")
        root.geometry("1320x920")
        root.minsize(1080, 780)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", foreground=MUTED, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 5))
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TLabelframe", background=BG)
        style.configure("TLabelframe.Label", background=BG, font=("Segoe UI", 10, "bold"))
        self.messages = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.locked_widgets = []
        self.restore_states = []
        self.demo_results = {}
        self.demo_problem = None
        self.batch = None
        self.problem = None
        self.playing = False
        self.play_after = None
        self.queue_after = None
        self.cursor = 0.0
        self.closing = False
        self._build_header()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(8, 10))
        self.scenario_tab = ttk.Frame(self.notebook, padding=14)
        self.demo_tab = ttk.Frame(self.notebook, padding=12)
        self.experiment_tab = ttk.Frame(self.notebook, padding=12)
        for tab, text in ((self.scenario_tab, "1 · Skenario"), (self.demo_tab, "2 · Demo"),
                          (self.experiment_tab, "3 · Eksperimen")):
            self.notebook.add(tab, text=text)
        self._build_scenario()
        self._build_demo()
        self._build_experiments()
        footer = ttk.Frame(root)
        footer.pack(fill="x", padx=18, pady=(0, 12))
        self.status = tk.StringVar(value="Siap")
        ttk.Label(footer, textvariable=self.status, style="Subtitle.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(footer, length=230, mode="determinate")
        self.progress.pack(side="right", padx=(12, 0))
        self.cancel_button = ttk.Button(footer, text="Batalkan komputasi", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="right")
        self.generate()
        self.queue_after = root.after(80, self._poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_header(self):
        header = ttk.Frame(self.root, padding=(20, 16, 20, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Klinik Optima", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Antrean pasien · Genetic Algorithm · Particle Swarm Optimization · Ant Colony Optimization",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(3, 0))

    def _field(self, frame, column, title, variable, start, end, width=10):
        box = ttk.Frame(frame)
        box.grid(row=0, column=column, padx=(0, 15), sticky="w")
        ttk.Label(box, text=title).pack(anchor="w", pady=(0, 4))
        widget = ttk.Spinbox(box, from_=start, to=end, textvariable=variable, width=width)
        widget.pack(anchor="w")
        self.locked_widgets.append(widget)
        return widget

    def _tree(self, parent, columns, headings, height=8):
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
        for col, title in zip(columns, headings):
            tree.heading(col, text=title)
            tree.column(col, width=110, minwidth=75, anchor="center", stretch=True)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        return frame, tree

    def _build_scenario(self):
        controls = ttk.LabelFrame(self.scenario_tab, text="Data sintetis", padding=14)
        controls.pack(fill="x")
        self.count_var, self.doctors_var = tk.StringVar(value="20"), tk.StringVar(value="2")
        self.horizon_var, self.seed_var = tk.StringVar(value="60"), tk.StringVar(value="42")
        self.budget_var = tk.StringVar(value="2000")
        for column, args in enumerate((
            ("Pasien", self.count_var, 1, 100), ("Dokter", self.doctors_var, 1, 10),
            ("Kedatangan (menit)", self.horizon_var, 0, 480), ("Seed data", self.seed_var, 0, 999999),
            ("Evaluasi / algoritma", self.budget_var, 20, 100000),
        )):
            self._field(controls, column, *args)
        self.generate_button = ttk.Button(controls, text="Bangkitkan data", command=self.generate)
        self.generate_button.grid(row=0, column=5, sticky="s", padx=(5, 0))
        self.locked_widgets.append(self.generate_button)
        fitness_box = ttk.LabelFrame(self.scenario_tab, text="Fungsi fitness — lebih kecil lebih baik", padding=14)
        fitness_box.pack(fill="x", pady=12)
        self.fitness_var = tk.StringVar(value=MODES["weighted"])
        self.fitness_combo = ttk.Combobox(fitness_box, textvariable=self.fitness_var, values=list(MODES.values()),
                                         state="readonly", width=37)
        self.fitness_combo.grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.fitness_combo.bind("<<ComboboxSelected>>", lambda _: self._fitness_changed())
        self.locked_widgets.append(self.fitness_combo)
        self.weight_vars = [tk.StringVar(value=v) for v in ("0.50", "0.30", "0.20")]
        self.weight_widgets = []
        for column, (label, var) in enumerate(zip(("Tunggu berbobot", "Tunggu maksimum", "Selesai"), self.weight_vars), 1):
            box = ttk.Frame(fitness_box)
            box.grid(row=0, column=column, padx=8)
            ttk.Label(box, text=label).pack(anchor="w")
            widget = ttk.Spinbox(box, from_=0, to=1, increment=0.05, textvariable=var, width=7)
            widget.pack(anchor="w")
            self.weight_widgets.append(widget)
            self.locked_widgets.append(widget)
            var.trace_add("write", lambda *_: self._update_formula())
        self.formula_var = tk.StringVar()
        ttk.Label(fitness_box, textvariable=self.formula_var, style="Subtitle.TLabel", wraplength=1120).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(12, 0))
        self.data_info = tk.StringVar()
        ttk.Label(self.scenario_tab, textvariable=self.data_info).pack(anchor="w", pady=(0, 8))
        table, self.patient_tree = self._tree(self.scenario_tab, ("id", "arrival", "duration", "priority"),
                                             ("ID pasien", "Kedatangan (menit)", "Pemeriksaan (menit)", "Bobot prioritas"), 12)
        table.pack(fill="both", expand=True)
        parameters = "   |   ".join(f"{algo}: " + ", ".join(f"{k}={v}" for k, v in config.items())
                                    for algo, config in PARAMETERS.items())
        ttk.Label(self.scenario_tab, text=parameters, wraplength=1180, style="Subtitle.TLabel").pack(anchor="w", pady=(12, 4))
        ttk.Label(self.scenario_tab, text="Jam simulasi dimulai 08:00. Dokter setara; satu pemeriksaan per pasien. "
                  "Prioritas 1 / 2 / 4 adalah atribut simulasi; seluruh kedatangan diketahui sejak awal.",
                  style="Subtitle.TLabel", wraplength=1150).pack(anchor="w")
        self._fitness_changed()

    def _fitness_mode(self):
        return next(key for key, value in MODES.items() if value == self.fitness_var.get())

    def _spec(self):
        # Disabled weights do not affect standalone objectives.
        weights = tuple(float(var.get()) for var in self.weight_vars) if self._fitness_mode() == "combined" else (0.5, 0.3, 0.2)
        return FitnessSpec(self._fitness_mode(), weights)

    def _fitness_changed(self):
        for widget in self.weight_widgets:
            widget.configure(state="normal" if self._fitness_mode() == "combined" else "disabled")
        self._update_formula()

    def _update_formula(self):
        mode = self._fitness_mode()
        formulas = {
            "weighted": "F₁ = Σ(bobot × waktu tunggu) / Σbobot. Satuan: menit; prioritas lebih tinggi meningkatkan biaya menunggu.",
            "maximum": "F₂ = waktu tunggu pasien terlama. Satuan: menit; fokus pada pasien yang menunggu paling lama.",
            "completion": "F₃ = waktu selesai pasien terakhir sejak klinik dibuka. Satuan: menit; fokus pada penyelesaian pelayanan.",
        }
        try:
            if mode == "combined":
                spec = FitnessSpec(mode, tuple(float(v.get()) for v in self.weight_vars))
                a, b, c = spec.normalized_weights
                text = f"F₄ = {a:.3f} × F̂₁ + {b:.3f} × F̂₂ + {c:.3f} × F̂₃. F̂ₖ = Fₖ / max(FCFSₖ, 1 menit). Skor tanpa satuan."
            else:
                text = formulas[mode]
        except ValueError:
            text = "Bobot belum valid: gunakan angka nonnegatif dengan jumlah lebih dari nol."
        self.formula_var.set(text)

    def _budget(self):
        value = int(self.budget_var.get())
        if not 20 <= value <= 100000:
            raise ValueError("Anggaran evaluasi harus 20–100.000.")
        return value

    def generate(self):
        if self.busy:
            return
        try:
            problem = generate_problem(int(self.count_var.get()), int(self.doctors_var.get()),
                                       int(self.horizon_var.get()), int(self.seed_var.get()))
        except ValueError as exc:
            messagebox.showerror("Data tidak valid", str(exc), parent=self.root)
            return
        self.problem = problem
        self.patient_tree.delete(*self.patient_tree.get_children())
        for p in sorted(problem.patients, key=lambda p: (p.arrival, p.id)):
            self.patient_tree.insert("", "end", values=(p.id, p.arrival, p.duration, p.weight))
        metrics = Evaluator(problem, FitnessSpec()).baseline_metrics
        self.data_info.set(f"{len(problem.patients)} pasien · {problem.doctors} dokter · seed {problem.seed} · "
                           f"FCFS: tunggu berbobot {metrics.weighted:.2f}, maksimum {metrics.maximum:.0f}, "
                           f"selesai {metrics.completion:.0f} menit")
        if hasattr(self, "status"):
            self.status.set("Data dibangkitkan. Klik Jalankan demo untuk memakai data ini; hasil lama tetap memakai skenario asalnya.")

    def _build_demo(self):
        toolbar = ttk.Frame(self.demo_tab)
        toolbar.pack(fill="x")
        self.demo_run_button = ttk.Button(toolbar, text="Jalankan demo", command=self.run_demo)
        self.demo_run_button.pack(side="left", padx=(0, 12))
        self.locked_widgets.append(self.demo_run_button)
        self.replay_mode = tk.StringVar(value="Proses optimasi")
        combo = ttk.Combobox(toolbar, textvariable=self.replay_mode, values=("Proses optimasi", "Pelayanan klinik"),
                             state="readonly", width=19)
        combo.pack(side="left", padx=(0, 8))
        combo.bind("<<ComboboxSelected>>", lambda _: self.reset_replay())
        ttk.Button(toolbar, text="▶ Putar", command=self.play).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Ⅱ Jeda", command=self.pause).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Langkah berikutnya", command=self.step).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Ulangi", command=self.reset_replay).pack(side="left", padx=2)
        ttk.Label(toolbar, text="Kecepatan").pack(side="left", padx=(14, 5))
        self.speed_var = tk.StringVar(value="1×")
        ttk.Combobox(toolbar, textvariable=self.speed_var, values=("0.25×", "0.5×", "1×", "2×", "4×"),
                     state="readonly", width=6).pack(side="left")
        self.demo_info = tk.StringVar(value="Bangkitkan data di tab Skenario, lalu jalankan demo. Perhitungan selesai sebelum replay dimulai.")
        ttk.Label(self.demo_tab, textvariable=self.demo_info, style="Subtitle.TLabel", wraplength=1150).pack(anchor="w", pady=(10, 5))
        self.clock_var = tk.StringVar(value="Evaluasi: 0")
        ttk.Label(self.demo_tab, textvariable=self.clock_var, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        panels = ttk.Frame(self.demo_tab)
        panels.pack(fill="both", expand=True)
        self.panel_canvases, self.panel_labels, self.panel_orders = {}, {}, {}
        for i, algorithm in enumerate(("GA", "PSO", "ACO")):
            panels.columnconfigure(i, weight=1, uniform="algo")
            frame = ttk.LabelFrame(panels, text=algorithm, padding=7)
            frame.grid(row=0, column=i, sticky="nsew", padx=4)
            canvas_frame = ttk.Frame(frame)
            canvas_frame.pack(fill="both", expand=True)
            canvas = tk.Canvas(canvas_frame, bg="white", height=300, highlightthickness=0)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            canvas.bind("<Configure>", lambda _: self.render_demo())
            text = tk.StringVar(value="Belum ada hasil")
            ttk.Label(frame, textvariable=text, wraplength=345, font=("Segoe UI", 9)).pack(anchor="w", pady=(5, 0))
            order = tk.StringVar(value="")
            ttk.Label(frame, textvariable=order, wraplength=345, style="Subtitle.TLabel", font=("Segoe UI", 8)).pack(anchor="w", pady=(3, 0))
            self.panel_canvases[algorithm], self.panel_labels[algorithm], self.panel_orders[algorithm] = canvas, text, order
        panels.rowconfigure(0, weight=1)
        self.demo_chart = tk.Canvas(self.demo_tab, bg="white", height=190, highlightthickness=0)
        self.demo_chart.pack(fill="x", pady=(10, 6))
        self.demo_chart.bind("<Configure>", lambda _: self.render_demo())
        self.demo_export_button = ttk.Button(self.demo_tab, text="Ekspor hasil demo", command=lambda: self.export(True))
        self.demo_export_button.pack(anchor="e")
        self.locked_widgets.append(self.demo_export_button)

    def _build_experiments(self):
        toolbar = ttk.Frame(self.experiment_tab)
        toolbar.pack(fill="x")
        self.experiment_scenarios = tk.StringVar(value="Skenario aktif")
        self.experiment_modes = tk.StringVar(value="Fitness terpilih")
        for variable, values, width in ((self.experiment_scenarios, ("Skenario aktif", "Ketiga preset"), 17),
                                         (self.experiment_modes, ("Fitness terpilih", "Keempat fitness"), 19)):
            widget = ttk.Combobox(toolbar, textvariable=variable, values=values, state="readonly", width=width)
            widget.pack(side="left", padx=(0, 10))
            self.locked_widgets.append(widget)
        ttk.Label(toolbar, text="Pengulangan").pack(side="left", padx=(0, 5))
        self.repeats_var = tk.StringVar(value="10")
        repeats = ttk.Spinbox(toolbar, from_=1, to=100, textvariable=self.repeats_var, width=5)
        repeats.pack(side="left", padx=(0, 12))
        self.locked_widgets.append(repeats)
        button = ttk.Button(toolbar, text="Jalankan eksperimen", command=self.run_experiment)
        button.pack(side="left")
        self.locked_widgets.append(button)
        ttk.Label(self.experiment_tab, text="Dataset tetap per skenario; seed optimizer 0…n−1. Urutan eksekusi dirotasi. "
                  "Waktu diukur tanpa animasi. Rata-rata dan simpangan baku dihitung per skenario dan mode fitness.",
                  style="Subtitle.TLabel", wraplength=1150).pack(anchor="w", pady=10)
        frame, self.summary_tree = self._tree(self.experiment_tab,
            ("scenario", "fitness", "rank", "algo", "n", "mean", "std", "best", "seconds"),
            ("Skenario", "Fitness", "Peringkat", "Algoritma", "n", "Skor rata-rata", "Simp. baku", "Terbaik", "Waktu (detik)"), 7)
        frame.pack(fill="x")
        graph_toolbar = ttk.Frame(self.experiment_tab)
        graph_toolbar.pack(fill="x", pady=(10, 5))
        ttk.Label(graph_toolbar, text="Grafik kelompok").pack(side="left", padx=(0, 10))
        self.group_var = tk.StringVar()
        self.group_combo = ttk.Combobox(graph_toolbar, textvariable=self.group_var, state="readonly", width=48)
        self.group_combo.pack(side="left")
        self.group_combo.bind("<<ComboboxSelected>>", lambda _: self.render_experiment())
        self.group_lookup = {}
        self.experiment_chart = tk.Canvas(self.experiment_tab, bg="white", height=210, highlightthickness=0)
        self.experiment_chart.pack(fill="both", expand=True)
        self.experiment_chart.bind("<Configure>", lambda _: self.render_experiment())
        self.baseline_var = tk.StringVar(value="Baseline FCFS dan Prioritas/Durasi akan muncul setelah eksperimen selesai.")
        ttk.Label(self.experiment_tab, textvariable=self.baseline_var, style="Subtitle.TLabel", wraplength=1150).pack(anchor="w", pady=6)
        frame, self.raw_tree = self._tree(self.experiment_tab,
            ("scenario", "mode", "algorithm", "repeat", "score", "weighted", "maximum", "completion", "seconds"),
            ("Skenario", "Fitness", "Algoritma", "Ulangan", "Skor", "Tunggu berbobot", "Tunggu maks.", "Selesai", "Detik"), 5)
        frame.pack(fill="x")
        self.raw_tree.bind("<<TreeviewSelect>>", self._show_run_details)
        self.run_lookup = {}
        self.run_detail_var = tk.StringVar(value="Pilih satu hasil pengulangan untuk melihat waktu tunggu rata-rata dan per prioritas.")
        ttk.Label(self.experiment_tab, textvariable=self.run_detail_var, style="Subtitle.TLabel", wraplength=1150).pack(anchor="w", pady=(4, 0))
        button = ttk.Button(self.experiment_tab, text="Ekspor CSV dan JSON", command=self.export)
        button.pack(anchor="e", pady=(8, 0))
        self.locked_widgets.append(button)

    def _launch(self, target):
        if self.busy:
            return
        self.pause()
        self.busy = True
        self.cancel_event = threading.Event()
        self.restore_states = [(w, w.cget("state")) for w in self.locked_widgets]
        for widget, _ in self.restore_states:
            widget.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.configure(value=0, maximum=100)
        self.status.set("Menghitung…")

        def worker():
            try:
                target()
            except Cancelled:
                self.messages.put(("cancelled", None))
            except Exception as exc:
                self.messages.put(("error", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=worker, daemon=True).start()

    def _unlock(self):
        self.busy = False
        for widget, state in self.restore_states:
            widget.configure(state=state)
        self.cancel_button.configure(state="disabled")
        self._fitness_changed()

    def run_demo(self):
        if self.busy:
            return
        try:
            problem, spec, budget = self.problem, self._spec(), self._budget()
        except ValueError as exc:
            messagebox.showerror("Konfigurasi tidak valid", str(exc), parent=self.root)
            return
        self.notebook.select(self.demo_tab)

        def worker():
            results = []
            for index, algorithm in enumerate(("GA", "PSO", "ACO")):
                def observer(snapshot, index=index, algorithm=algorithm):
                    self.messages.put(("progress", ((index + snapshot.evaluations / budget) / 3 * 100,
                                                    f"Menghitung {algorithm}: {snapshot.evaluations}/{budget} evaluasi")))
                results.append(optimize(problem, algorithm, spec, 0, budget, observer, self.cancel_event))
            self.messages.put(("demo_done", (problem, results)))
        self._launch(worker)

    def run_experiment(self):
        if self.busy:
            return
        try:
            spec, budget, repeats = self._spec(), self._budget(), int(self.repeats_var.get())
            if not 1 <= repeats <= 100:
                raise ValueError("Pengulangan harus 1–100.")
            problems = ({name: generate_problem(n, d, h, s) for name, n, d, h, s in PRESETS}
                        if self.experiment_scenarios.get() == "Ketiga preset" else {"Kustom": self.problem})
            modes = list(MODES) if self.experiment_modes.get() == "Keempat fitness" else [spec.mode]
            weights = tuple(float(v.get()) for v in self.weight_vars) if "combined" in modes else spec.weights
            FitnessSpec("combined", weights)
        except ValueError as exc:
            messagebox.showerror("Konfigurasi tidak valid", str(exc), parent=self.root)
            return

        def worker():
            batch = run_experiments(problems, modes, weights, repeats, budget,
                                    lambda done, total, label: self.messages.put(("progress", (100 * done / total, label))),
                                    self.cancel_event)
            self.messages.put(("batch_done", batch))
        self._launch(worker)

    def cancel(self):
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Membatalkan setelah evaluasi yang sedang berjalan selesai…")

    def _poll(self):
        if self.closing:
            return
        latest_progress = None
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                latest_progress = payload
            elif kind == "demo_done":
                self._unlock()
                self.demo_problem, results = payload
                self.demo_results = {r.algorithm: r for r in results}
                spec = results[0].fitness
                weights = " · bobot " + "/".join(f"{w:.2f}" for w in spec.normalized_weights) if spec.mode == "combined" else ""
                self.demo_info.set(f"{len(self.demo_problem.patients)} pasien · {self.demo_problem.doctors} dokter · "
                                   f"seed data {self.demo_problem.seed} · {MODES[spec.mode]}{weights} · "
                                   f"{results[0].budget} evaluasi / algoritma. Replay memakai hasil yang telah dihitung.")
                self.reset_replay()
                self.status.set("Demo siap. Pilih proses optimasi atau pelayanan klinik, lalu Putar.")
                self.progress.configure(value=100)
                latest_progress = None
            elif kind == "batch_done":
                self._unlock()
                self.batch = payload
                self._populate_results()
                self.status.set(f"Eksperimen selesai: {len(payload.runs)} run. Hasil dapat diekspor.")
                self.progress.configure(value=100)
                latest_progress = None
            elif kind == "export_done":
                self._unlock()
                self.status.set(f"Ekspor selesai: {payload}")
                self.progress.configure(value=100)
                latest_progress = None
            elif kind in ("error", "cancelled"):
                self._unlock()
                self.status.set("Komputasi dibatalkan; hasil sebelumnya dipertahankan." if kind == "cancelled" else "Proses gagal.")
                if kind == "error":
                    messagebox.showerror("Proses gagal", payload, parent=self.root)
                latest_progress = None
        if latest_progress and self.busy:
            value, label = latest_progress
            self.progress.configure(value=value)
            self.status.set(label)
        self.queue_after = self.root.after(80, self._poll)

    def _populate_results(self):
        self.summary_tree.delete(*self.summary_tree.get_children())
        for s in self.batch.summaries():
            self.summary_tree.insert("", "end", values=(s["scenario"], MODES[s["fitness"]], s["rank"], s["algorithm"], s["runs"],
                f"{s['mean_score']:.4f}", f"{s['std_score']:.4f}", f"{s['best_score']:.4f}", f"{s['mean_seconds']:.4f}"))
        self.raw_tree.delete(*self.raw_tree.get_children())
        self.run_lookup = {}
        self.run_detail_var.set("Pilih satu pengulangan untuk melihat rincian metrik; seluruh waktu pelayanan dalam menit.")
        for run in self.batch.runs:
            r = run.result
            item = self.raw_tree.insert("", "end", values=(run.scenario, MODES[r.fitness.mode], r.algorithm, run.repeat,
                f"{r.best_score:.4f}", f"{r.metrics.weighted:.2f}", f"{r.metrics.maximum:.0f}",
                f"{r.metrics.completion:.0f}", f"{r.elapsed:.4f}"))
            self.run_lookup[item] = r
        self.group_lookup = {f"{scenario} · {MODES[mode]}": (scenario, mode) for scenario, mode in self.batch.groups()}
        self.group_combo.configure(values=list(self.group_lookup))
        self.group_var.set(next(iter(self.group_lookup), ""))
        self.render_experiment()

    def _show_run_details(self, _=None):
        selected = self.raw_tree.selection()
        result = self.run_lookup.get(selected[0]) if selected else None
        if result is None:
            return
        classes = " / ".join(f"bobot {w}: {result.metrics.per_priority[w]:.2f}" for w in (1, 2, 4)
                             if w in result.metrics.per_priority)
        self.run_detail_var.set(f"Seed optimizer {result.seed} · {result.evaluations} evaluasi · "
                                f"tunggu rata-rata {result.metrics.average:.2f} menit · tunggu per prioritas: {classes}")

    def render_experiment(self):
        if not hasattr(self, "experiment_chart"):
            return
        group = self.group_lookup.get(self.group_var.get())
        if self.batch is None or group is None:
            draw_chart(self.experiment_chart, [])
            return
        scenario, mode = group
        series = []
        for algorithm in ("GA", "PSO", "ACO"):
            results = [run.result for run in self.batch.runs if
                       (run.scenario, run.result.fitness.mode, run.result.algorithm) == (scenario, mode, algorithm)]
            if results:
                budget = results[0].budget
                checkpoints = sorted(set([1, budget] + list(range(20, budget + 1, 20))))
                series.append((algorithm, [(x, statistics.mean(r.history[x - 1][1] for r in results)) for x in checkpoints]))
        baselines = [(b.name, b.score) for b in self.batch.baselines if (b.scenario, b.fitness.mode) == group]
        draw_chart(self.experiment_chart, series, baselines, "Konvergensi rata-rata pengulangan — lebih kecil lebih baik",
                   "Skor" if mode == "combined" else "Menit")
        self.baseline_var.set(" · ".join(f"{name}: {score:.4f}" for name, score in baselines) +
                              " · Peringkat hanya berlaku pada skenario, fitness, dan konfigurasi ini.")

    def export(self, demo=False):
        if self.busy:
            return
        batch = demo_batch(self.demo_problem, list(self.demo_results.values())) if demo and self.demo_results else (None if demo else self.batch)
        if batch is None:
            messagebox.showinfo("Belum ada hasil", "Jalankan demo atau eksperimen terlebih dahulu.", parent=self.root)
            return
        parent = filedialog.askdirectory(title="Pilih folder induk ekspor", parent=self.root)
        if not parent:
            return
        destination = Path(parent) / datetime.now().strftime("klinik-%Y%m%d-%H%M%S-%f")

        def worker():
            export_batch(batch, destination)
            self.messages.put(("export_done", str(destination)))
        self._launch(worker)
        self.cancel_button.configure(state="disabled")
        self.status.set("Mengekspor CSV dan JSON…")

    def _limit(self):
        if not self.demo_results:
            return 0
        if self.replay_mode.get() == "Proses optimasi":
            return max(r.budget for r in self.demo_results.values())
        return max(r.metrics.completion for r in self.demo_results.values())

    def play(self):
        if not self.demo_results or self.busy or self.playing:
            return
        if self.cursor >= self._limit():
            self.cursor = 0
        self.playing = True
        self._tick()

    def pause(self):
        self.playing = False
        if self.play_after is not None:
            self.root.after_cancel(self.play_after)
            self.play_after = None

    def reset_replay(self):
        self.pause()
        self.cursor = 0
        self.render_demo()

    def step(self):
        self.pause()
        if not self.demo_results:
            return
        self.cursor = min(self._limit(), self.cursor + (20 if self.replay_mode.get() == "Proses optimasi" else 1))
        self.render_demo()

    def _tick(self):
        self.play_after = None
        if not self.playing or self.closing:
            return
        self.cursor = min(self._limit(), self.cursor + (20 if self.replay_mode.get() == "Proses optimasi" else 0.5))
        self.render_demo()
        if self.cursor >= self._limit():
            self.playing = False
            return
        speed = float(self.speed_var.get().replace("×", ""))
        self.play_after = self.root.after(round(100 / speed), self._tick)

    def render_demo(self):
        if not hasattr(self, "demo_chart"):
            return
        optimization = self.replay_mode.get() == "Proses optimasi"
        self.clock_var.set(f"Evaluasi: {int(self.cursor)} / {int(self._limit())}" if optimization else
                           f"Waktu klinik: {self._clock(self.cursor)} · menit {self.cursor:.1f}")
        series = []
        for algorithm, canvas in self.panel_canvases.items():
            result = self.demo_results.get(algorithm)
            canvas.delete("all")
            if result is None:
                canvas.create_text(20, 25, text="Belum ada hasil", anchor="nw", fill=MUTED)
                continue
            if optimization:
                snapshots = result.snapshots
                index = bisect_right([s.evaluations for s in snapshots], self.cursor) - 1
                if index < 0:
                    canvas.create_text(18, 24, text="Menunggu evaluasi awal…", anchor="nw", fill=MUTED)
                    self.panel_labels[algorithm].set("Inisialisasi belum selesai pada titik replay ini.")
                    self.panel_orders[algorithm].set("")
                else:
                    snapshot = snapshots[index]
                    self._draw_algorithm(canvas, algorithm, snapshot)
                    self.panel_labels[algorithm].set(f"Iterasi {snapshot.iteration} · snapshot evaluasi {snapshot.evaluations}\n"
                                                      f"Skor terbaik: {snapshot.best_score:.4f}")
                    self.panel_orders[algorithm].set("Peringkat: " + " → ".join(snapshot.best_order[:14]) +
                                                     (" …" if len(snapshot.best_order) > 14 else ""))
                points = [p for p in result.history if p[0] <= self.cursor]
            else:
                self._draw_clinic(canvas, result)
                m = result.metrics
                per_class = " / ".join(f"{w}: {m.per_priority[w]:.1f}" for w in (1, 2, 4) if w in m.per_priority)
                self.panel_labels[algorithm].set(f"Hasil akhir · skor {result.best_score:.4f}\n"
                    f"Tunggu berbobot {m.weighted:.2f} · rata-rata {m.average:.2f}\n"
                    f"Maksimum {m.maximum:.0f} · selesai {m.completion:.0f} menit")
                self.panel_orders[algorithm].set(f"Tunggu per prioritas (menit): {per_class}")
                points = result.history
            if points:
                series.append((algorithm, points))
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=(0, 0, max(canvas.winfo_width(), bbox[2] + 8),
                                               max(canvas.winfo_height(), bbox[3] + 8)))
        baselines = []
        if self.demo_results and series:
            first = next(iter(self.demo_results.values()))
            evaluator = Evaluator(self.demo_problem, first.fitness)
            baselines = [(name, evaluator.evaluate(order)[0]) for name, order in
                         (("FCFS", fcfs_order(self.demo_problem)), ("Prioritas/Durasi", ratio_order(self.demo_problem)))]
        draw_chart(self.demo_chart, series, baselines, unit="Skor" if self.demo_results and
                   next(iter(self.demo_results.values())).fitness.mode == "combined" else "Menit")

    @staticmethod
    def _clock(minutes):
        total = 8 * 60 + int(minutes)
        return f"{total // 60:02}:{total % 60:02}"

    def _draw_algorithm(self, canvas, algorithm, snapshot):
        width, height = max(canvas.winfo_width(), 310), max(canvas.winfo_height(), 260)
        detail = snapshot.details
        if algorithm == "GA":
            canvas.create_text(12, 14, text="Populasi · skor tiap calon urutan", anchor="nw", fill=INK, font=("Segoe UI", 10, "bold"))
            scores = detail["scores"]
            high = max(max(scores), 0.01)
            step = (width - 35) / len(scores)
            for i, score in enumerate(scores):
                x = 18 + i * step
                h = score / high * 75
                canvas.create_rectangle(x, 112 - h, x + max(3, step - 3), 112,
                                        fill=COLORS[algorithm], outline="")
            canvas.create_text(12, 124, text="Batang lebih pendek = skor lebih baik", anchor="nw", fill=MUTED, font=("Segoe UI", 8))
            rows = [("Induk A", detail["parents"][0] if detail["parents"] else ()),
                    ("Induk B", detail["parents"][1] if detail["parents"] else ()),
                    ("Setelah crossover", detail["before_mutation"]), ("Setelah mutasi", detail["child"])]
            for i, (title, order) in enumerate(rows):
                y = 136 + i * 35
                canvas.create_text(12, y, text=title, anchor="nw", fill=INK, font=("Segoe UI", 8, "bold"))
                for j, pid in enumerate(order[:10]):
                    x = 12 + j * ((width - 24) / 10)
                    changed = title == "Setelah mutasi" and j in detail["swapped"]
                    canvas.create_rectangle(x, y + 15, x + (width - 24) / 10 - 2, y + 32,
                                            fill="#fcd34d" if changed else "#dbeafe", outline="")
                    canvas.create_text(x + (width - 24) / 20 - 1, y + 23, text=pid[1:], fill=INK, font=("Segoe UI", 7))
            canvas.create_text(12, min(height - 22, 315), text="Kromosom: cuplikan 10 pasien pertama; kuning = mutasi.",
                               anchor="nw", fill=MUTED, font=("Segoe UI", 8))
        elif algorithm == "PSO":
            left, top, right, bottom = 42, 42, width - 22, max(180, min(height - 70, 265))
            canvas.create_text(12, 12, text="Proyeksi dua dimensi random keys", anchor="nw", fill=INK, font=("Segoe UI", 10, "bold"))
            for k in range(5):
                x = left + k / 4 * (right - left)
                y = top + k / 4 * (bottom - top)
                canvas.create_line(x, top, x, bottom, fill="#e2e8f0")
                canvas.create_line(left, y, right, y, fill="#e2e8f0")

            def point(xy):
                return left + xy[0] * (right - left), bottom - xy[1] * (bottom - top)

            for xy, velocity, personal in zip(detail["points"], detail["velocities"], detail["personal_best"]):
                x, y = point(xy)
                px, py = point(personal)
                canvas.create_oval(px - 2, py - 2, px + 2, py + 2, outline="#a78bfa")
                ex, ey = point([min(1, max(0, xy[0] + velocity[0])), min(1, max(0, xy[1] + velocity[1]))])
                canvas.create_line(x, y, ex, ey, arrow="last", fill="#c4b5fd")
                canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=COLORS[algorithm], outline="")
            x, y = point(detail["global_best"])
            canvas.create_line(x - 7, y, x + 7, y, fill="#ef4444", width=3)
            canvas.create_line(x, y - 7, x, y + 7, fill="#ef4444", width=3)
            canvas.create_text((left + right) / 2, bottom + 15, text=f"Key {detail['dimensions'][0]} (0–1)", fill=MUTED, font=("Segoe UI", 8))
            canvas.create_text(left, top - 10, text=f"Key {detail['dimensions'][1]} (0–1)", anchor="w", fill=MUTED, font=("Segoe UI", 8))
            canvas.create_text(12, bottom + 35, text="● partikel   ○ personal best   + global best\n"
                               "Ini ruang solusi, bukan posisi fisik pasien.", anchor="nw", fill=MUTED, font=("Segoe UI", 8))
        else:
            matrix = detail["pheromone"]
            n = len(matrix)
            cell = min((width - 62) / max(n, 1), max(120, height - 135) / max(n, 1))
            high = max(max(row) for row in matrix)
            low = min(min(row) for row in matrix)
            canvas.create_text(12, 12, text="Feromon transisi · gelap = lebih kuat", anchor="nw", fill=INK, font=("Segoe UI", 10, "bold"))
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    fraction = (value - low) / max(high - low, 1e-12)
                    color = f"#{int(220 - 210 * fraction):02x}{int(252 - 95 * fraction):02x}{int(231 - 125 * fraction):02x}"
                    canvas.create_rectangle(40 + j * cell, 45 + i * cell, 40 + (j + 1) * cell, 45 + (i + 1) * cell,
                                            fill=color, outline="white")
            for i, label in enumerate(detail["labels"]):
                if n <= 10 or i % 2 == 0:
                    canvas.create_text(35, 45 + (i + 0.5) * cell, text=label[1:], anchor="e", fill=MUTED, font=("Segoe UI", 7))
                    canvas.create_text(40 + (i + 0.5) * cell, 36, text=label[1:], fill=MUTED, font=("Segoe UI", 7))
            y = 55 + n * cell
            canvas.create_text(12, y, text=f"Baris: asal · kolom: tujuan · {n}/{detail['patient_count']} pasien\n"
                               "Semut terbaik pada iterasi ini:", anchor="nw", fill=MUTED, font=("Segoe UI", 8))
            canvas.create_text(12, y + 35, text=" → ".join(detail["ant_order"][:10]) + (" …" if len(detail["ant_order"]) > 10 else ""),
                               anchor="nw", fill=COLORS[algorithm], width=width - 24, font=("Segoe UI", 8))

    def _draw_clinic(self, canvas, result):
        width = max(canvas.winfo_width(), 310)
        now = self.cursor
        visits = result.visits
        waiting = sorted([v for v in visits if v.arrival <= now < v.start], key=lambda v: result.best_order.index(v.patient_id))
        done = sorted([v for v in visits if v.end <= now], key=lambda v: (v.end, v.patient_id))
        future = sum(v.arrival > now for v in visits)
        active = {v.doctor_id: v for v in visits if v.start <= now < v.end}
        canvas.create_text(12, 12, text=f"Belum datang {future} · Menunggu {len(waiting)}\n"
                           f"Diperiksa {len(active)} · Selesai {len(done)}", anchor="nw", fill=INK, font=("Segoe UI", 10, "bold"))
        canvas.create_text(12, 56, text="ANTREAN", anchor="nw", fill=MUTED, font=("Segoe UI", 8, "bold"))
        canvas.create_text(width - 92, 56, text="SELESAI", anchor="nw", fill=MUTED, font=("Segoe UI", 8, "bold"))
        doctor_x = width * 0.5
        for doctor in range(1, self.demo_problem.doctors + 1):
            y = 80 + (doctor - 1) * 25
            canvas.create_rectangle(doctor_x - 40, y, doctor_x + 40, y + 21,
                                    fill="#f0fdf4" if doctor in active else "#f8fafc", outline="#cbd5e1")
            canvas.create_text(doctor_x, y + 10, text=f"D{doctor}" + (f" · {active[doctor].patient_id}" if doctor in active else " · kosong"),
                               fill=INK, font=("Segoe UI", 8))

        def tile(v, x, y):
            canvas.create_oval(x - 13, y - 9, x + 13, y + 9, fill=PRIORITY_COLORS[v.weight], outline="white")
            canvas.create_text(x, y, text=v.patient_id[1:], fill=INK, font=("Segoe UI", 7))

        for i, v in enumerate(waiting[:24]):
            x, y = 25 + (i % 3) * 29, 86 + (i // 3) * 23
            # Interpolate arrival into the queue for a visible movement.
            fraction = min(1.0, (now - v.arrival) / 0.8)
            tile(v, x, 55 + (y - 55) * fraction)
        for i, v in enumerate(done[-24:]):
            x, y = width - 80 + (i % 3) * 28, 86 + (i // 3) * 23
            fraction = min(1.0, (now - v.end) / 0.8)
            origin_y = 90 + (v.doctor_id - 1) * 25
            tile(v, doctor_x + (x - doctor_x) * fraction, origin_y + (y - origin_y) * fraction)
        for doctor, v in active.items():
            fraction = min(1.0, (now - v.start) / 0.8)
            tile(v, 25 + (doctor_x - 70 - 25) * fraction, 86 + (doctor - 1) * 25 * fraction)
        extra_y = max(244, 92 + self.demo_problem.doctors * 25)
        if len(waiting) > 24 or len(done) > 24:
            canvas.create_text(12, extra_y, text=f"Antrean +{max(0, len(waiting) - 24)} · selesai +{max(0, len(done) - 24)} ikon tersembunyi",
                               anchor="nw", fill=MUTED, font=("Segoe UI", 8))
            extra_y += 17
        canvas.create_text(12, extra_y + 3, text="Biru: bobot 1 · kuning: 2 · merah: 4", anchor="nw", fill=MUTED, font=("Segoe UI", 8))
        observed = [min(now, v.start) - v.arrival for v in visits if v.arrival <= now]
        current_average = statistics.mean(observed) if observed else 0
        canvas.create_text(12, extra_y + 22, text=f"Tunggu rata-rata teramati: {current_average:.2f} menit", anchor="nw", fill=INK, font=("Segoe UI", 9))

    def close(self):
        self.closing = True
        self.cancel_event.set()
        self.pause()
        if self.queue_after is not None:
            self.root.after_cancel(self.queue_after)
        self.root.destroy()


def main():
    root = tk.Tk()
    ClinicApp(root)
    root.mainloop()
