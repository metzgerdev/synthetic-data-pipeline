"""
Analysis report for DIY repair LLM judge evaluation.

AnalysisReport takes:
  - entries       : list[dict]        from repair_dataset_balanced.json
  - human_reviews : list[dict]        from review_results.json
  - llm_judgments : list[DimJudgment] from LLMJudge.baseline().evaluate_dims_dataset()

Produces:
  - Per-dim pass rate overall and per category segment
  - Human / LLM agreement per dim and per segment
  - Five charts: dim pass rates, segment heatmap, agreement heatmap,
                 overall confusion matrix, per-segment agreement bars
  - JSON + CSV export
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

from judge import DIM_KEYS, DimJudgment


# ─── AnalysisReport ──────────────────────────────────────────────────────────

class AnalysisReport:

    def __init__(
        self,
        entries:       list[dict],
        human_reviews: list[dict],
        llm_judgments: list[DimJudgment],
    ):
        review_by_index = {r["index"]: r for r in human_reviews}

        self.rows: list[dict] = []
        for i, (entry, llm) in enumerate(zip(entries, llm_judgments)):
            review = review_by_index.get(i)
            if review is None:
                continue
            self.rows.append({
                "index":    i,
                "category": entry.get("category", "unknown"),
                "question": entry.get("question", ""),
                "human":    review["labels"],          # dict[dim, bool]
                "llm":      llm.dims,                  # dict[dim, bool]
                "human_overall": "RESOLVED" if review["all_pass"] else "NOT_RESOLVED",
                "llm_overall":   llm.overall,
                "llm_reasoning": llm.reasoning,
            })

        self.segments = sorted({r["category"] for r in self.rows})

    # ── Stat helpers ─────────────────────────────────────────────────────────

    def _pass_rate(self, subset: list[dict], source: str) -> dict[str, float]:
        """Pass rate per dim for a list of rows, source = 'human' | 'llm'."""
        totals: dict[str, int] = defaultdict(int)
        passes: dict[str, int] = defaultdict(int)
        for row in subset:
            for dim, val in row[source].items():
                totals[dim] += 1
                passes[dim] += int(val)
        return {d: passes[d] / totals[d] if totals[d] else 0.0 for d in DIM_KEYS}

    def _agreement_rate(self, subset: list[dict]) -> dict[str, float]:
        """Per-dim agreement (human == llm) rate."""
        totals: dict[str, int] = defaultdict(int)
        agrees: dict[str, int] = defaultdict(int)
        for row in subset:
            for dim in DIM_KEYS:
                totals[dim] += 1
                agrees[dim] += int(row["human"].get(dim) == row["llm"].get(dim))
        return {d: agrees[d] / totals[d] if totals[d] else 0.0 for d in DIM_KEYS}

    # ── Summary dict ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        out = {
            "total": len(self.rows),
            "overall": {
                "human_pass_rate": sum(1 for r in self.rows if r["human_overall"] == "RESOLVED") / len(self.rows),
                "llm_pass_rate":   sum(1 for r in self.rows if r["llm_overall"]   == "RESOLVED") / len(self.rows),
                "agreement":       sum(1 for r in self.rows if r["human_overall"] == r["llm_overall"]) / len(self.rows),
                "human_dim_pass_rates": self._pass_rate(self.rows, "human"),
                "llm_dim_pass_rates":   self._pass_rate(self.rows, "llm"),
                "dim_agreement":        self._agreement_rate(self.rows),
            },
            "by_segment": {},
        }
        for seg in self.segments:
            subset = [r for r in self.rows if r["category"] == seg]
            out["by_segment"][seg] = {
                "n": len(subset),
                "human_dim_pass_rates": self._pass_rate(subset, "human"),
                "llm_dim_pass_rates":   self._pass_rate(subset, "llm"),
                "dim_agreement":        self._agreement_rate(subset),
            }
        return out

    def print_summary(self) -> "AnalysisReport":
        s = self.summary()
        print(f"\n{'─'*64}")
        print(f"  Analysis Report  ({s['total']} entries)")
        print(f"{'─'*64}")
        print(f"  Overall human RESOLVED : {s['overall']['human_pass_rate']:.1%}")
        print(f"  Overall LLM   RESOLVED : {s['overall']['llm_pass_rate']:.1%}")
        print(f"  Overall agreement      : {s['overall']['agreement']:.1%}\n")
        print(f"  {'Dimension':<30s}  {'Human':>6}  {'LLM':>6}  {'Agree':>6}")
        print(f"  {'─'*56}")
        for dim in DIM_KEYS:
            h = s["overall"]["human_dim_pass_rates"][dim]
            l = s["overall"]["llm_dim_pass_rates"][dim]
            a = s["overall"]["dim_agreement"][dim]
            print(f"  {dim:<30s}  {h:>5.1%}  {l:>5.1%}  {a:>5.1%}")
        print(f"{'─'*64}\n")
        return self

    # ── Charts ────────────────────────────────────────────────────────────────

    def plot_dim_pass_rates(self) -> "AnalysisReport":
        """Grouped bar chart: human vs LLM pass rate per dimension."""
        import matplotlib.pyplot as plt
        import numpy as np

        s = self.summary()["overall"]
        labels  = [d.replace("D", "D\n") for d in DIM_KEYS]
        human_v = [s["human_dim_pass_rates"][d] for d in DIM_KEYS]
        llm_v   = [s["llm_dim_pass_rates"][d]   for d in DIM_KEYS]

        x = np.arange(len(DIM_KEYS))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w/2, human_v, w, label="Human", color="#1565C0", alpha=0.85)
        ax.bar(x + w/2, llm_v,   w, label="LLM",   color="#C62828", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Pass rate")
        ax.set_title("Pass Rate per Dimension — Human vs LLM Judge")
        ax.axhline(0.8, color="grey", linestyle="--", linewidth=1, label="0.8 target")
        ax.legend(); plt.tight_layout(); plt.show()
        return self

    def plot_segment_heatmap(self) -> "AnalysisReport":
        """Heatmap: segment × dimension, colour = LLM pass rate."""
        import matplotlib.pyplot as plt
        import numpy as np

        s = self.summary()
        data = np.array([
            [s["by_segment"][seg]["llm_dim_pass_rates"][d] for d in DIM_KEYS]
            for seg in self.segments
        ])
        fig, ax = plt.subplots(figsize=(13, len(self.segments) * 0.9 + 1.5))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DIM_KEYS)));     ax.set_xticklabels(DIM_KEYS, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(self.segments))); ax.set_yticklabels(self.segments, fontsize=10)
        ax.set_title("LLM Judge Pass Rate — Segment × Dimension")
        for i, seg in enumerate(self.segments):
            for j, dim in enumerate(DIM_KEYS):
                ax.text(j, i, f"{data[i,j]:.0%}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.02); plt.tight_layout(); plt.show()
        return self

    def plot_agreement_heatmap(self) -> "AnalysisReport":
        """Heatmap: segment × dimension, colour = human/LLM agreement rate."""
        import matplotlib.pyplot as plt
        import numpy as np

        s = self.summary()
        data = np.array([
            [s["by_segment"][seg]["dim_agreement"][d] for d in DIM_KEYS]
            for seg in self.segments
        ])
        fig, ax = plt.subplots(figsize=(13, len(self.segments) * 0.9 + 1.5))
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DIM_KEYS)));      ax.set_xticklabels(DIM_KEYS, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(self.segments)));  ax.set_yticklabels(self.segments, fontsize=10)
        ax.set_title("Human / LLM Agreement Rate — Segment × Dimension")
        for i, seg in enumerate(self.segments):
            for j, dim in enumerate(DIM_KEYS):
                ax.text(j, i, f"{data[i,j]:.0%}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.02); plt.tight_layout(); plt.show()
        return self

    def plot_overall_confusion(self) -> "AnalysisReport":
        """Confusion matrix: human overall vs LLM overall label."""
        import matplotlib.pyplot as plt

        labels = ["RESOLVED", "NOT_RESOLVED"]
        cm = [[0, 0], [0, 0]]
        for row in self.rows:
            i = 0 if row["human_overall"] == "RESOLVED" else 1
            j = 0 if row["llm_overall"]   == "RESOLVED" else 1
            cm[i][j] += 1

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("LLM Judge"); ax.set_ylabel("Human")
        ax.set_title("Overall Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=18, fontweight="bold")
        plt.tight_layout(); plt.show()
        return self

    def plot_segment_agreement_bars(self) -> "AnalysisReport":
        """Horizontal bar chart: overall human/LLM agreement rate per segment."""
        import matplotlib.pyplot as plt

        rates = {}
        for seg in self.segments:
            subset = [r for r in self.rows if r["category"] == seg]
            rates[seg] = sum(1 for r in subset if r["human_overall"] == r["llm_overall"]) / len(subset)

        segs   = list(rates.keys())
        values = list(rates.values())
        colors = ["#2E7D32" if v >= 0.8 else "#F57F17" if v >= 0.6 else "#C62828" for v in values]

        fig, ax = plt.subplots(figsize=(8, len(segs) * 0.7 + 1))
        bars = ax.barh(segs, values, color=colors, alpha=0.85)
        ax.set_xlim(0, 1.15); ax.set_xlabel("Agreement rate")
        ax.set_title("Human / LLM Overall Agreement per Segment")
        ax.axvline(0.8, color="grey", linestyle="--", linewidth=1)
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", fontsize=10)
        plt.tight_layout(); plt.show()
        return self

    def plot_all(self) -> "AnalysisReport":
        """Run all five charts in sequence."""
        return (self
            .plot_dim_pass_rates()
            .plot_segment_heatmap()
            .plot_agreement_heatmap()
            .plot_overall_confusion()
            .plot_segment_agreement_bars())

    def save_charts(self, out_dir, variant: str = "") -> "AnalysisReport":
        """Save all five charts as PNGs to out_dir without displaying them."""
        import matplotlib.pyplot as plt
        out_dir = Path(out_dir)
        tag     = f"_{variant}" if variant else ""

        _orig_show = plt.show
        saved: list[str] = []

        def _patched_show():
            name = f"chart{tag}_{len(saved)+1}.png"
            path = out_dir / name
            plt.savefig(path, dpi=150, bbox_inches="tight")
            saved.append(name)
            plt.close()

        plt.show = _patched_show
        try:
            self.plot_all()
        finally:
            plt.show = _orig_show

        # rename saved files to meaningful names
        chart_names = [
            f"chart{tag}_dim_pass_rates.png",
            f"chart{tag}_segment_heatmap.png",
            f"chart{tag}_agreement_heatmap.png",
            f"chart{tag}_overall_confusion.png",
            f"chart{tag}_segment_bars.png",
        ]
        for tmp, final in zip(saved, chart_names):
            (out_dir / tmp).rename(out_dir / final)
            print(f"  Saved → {final}")

        return self

    # ── Export ────────────────────────────────────────────────────────────────

    def save(self, path: str = "analysis_results") -> "AnalysisReport":
        """Save labels as JSON and CSV, and summary stats as JSON."""
        base = Path(path)

        # per-item labels JSON
        records = []
        for row in self.rows:
            records.append({
                "index":         row["index"],
                "category":      row["category"],
                "human_overall": row["human_overall"],
                "llm_overall":   row["llm_overall"],
                "llm_reasoning": row["llm_reasoning"],
                **{f"human_{d}": row["human"].get(d) for d in DIM_KEYS},
                **{f"llm_{d}":   row["llm"].get(d)   for d in DIM_KEYS},
            })

        json_path = Path(str(base) + "_labels.json")
        json_path.write_text(json.dumps(records, indent=2))

        csv_path = Path(str(base) + "_labels.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        summary_path = Path(str(base) + "_summary.json")
        summary_path.write_text(json.dumps(self.summary(), indent=2))

        print(f"Saved: {json_path.name}  {csv_path.name}  {summary_path.name}")
        return self
