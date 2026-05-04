"""
LLM-as-Judge
  - Scores all 6 dimensions for every entry using structured output (instructor/Pydantic)
  - Temperature 0.1 (lower than generator's 0.9)
  - Assigns a trace_id per item: {variant}_{index:04d}
  - Saves labels to JSON + CSV

Analysis & Visualisation
  - Segment-level per-dim pass rates (human and LLM)
  - Per-dim human/LLM agreement overall and per segment
  - Items grouped by category segment and prompt variant
  - All charts saved to --out-dir

"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
from pathlib import Path

import dotenv
dotenv.load_dotenv()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Judge evaluation pipeline")
    p.add_argument("--dataset", required=True,
                   help="Dataset JSON file (list of repair entries)")
    p.add_argument("--human",   default=None,
                   help="Human review JSON (review_results.json). Optional — enables human/LLM comparison.")
    p.add_argument("--prompt",  default="calibrate", choices=["baseline", "calibrate"],
                   help="Judge prompt variant: baseline or calibrate (default: calibrate)")
    p.add_argument("--out-dir", default="eval_results",
                   help="Directory to write all output files and charts")
    p.add_argument("--delay",   type=float, default=3.0,
                   help="Seconds to wait between API calls (default: 3.0)")
    p.add_argument("--limit",   type=int,   default=None,
                   help="Only evaluate the first N entries (for quick smoke tests)")
    return p.parse_args()


#LLM Judge ──────────────────────────────────────────────────────

def run_judge(
    dataset:  list[dict],
    variant:  str,
    out_dir:  Path,
    delay:    float,
) -> list[dict]:
    """
    Score every entry on all 6 dimensions.
    Returns a list of label records, each with a trace_id.
    Saves labels_{variant}.json and labels_{variant}.csv to out_dir.
    """
    from judge import LLMJudge, Sample, Trace, DIM_KEYS, BaselinePromptBuilder, ImprovedPromptBuilder

    prompt_builder = BaselinePromptBuilder() if variant == "baseline" else ImprovedPromptBuilder()
    judge = LLMJudge.create(prompt=prompt_builder, api_key=os.getenv("GROQ_API_KEY"))
    trace = Trace()

    print(f"\n── LLM Judge ({variant}, temperature=0.1) ──────────────────")
    print(f"  Dataset : {len(dataset)} entries")
    print(f"  Output  : {out_dir}\n")

    records: list[dict] = []

    for i, entry in enumerate(dataset):
        trace_id = f"{variant}_{i:04d}"
        sample   = Sample(
            user_query   = entry["question"],
            bot_response = entry["answer"],
            issue_type   = entry.get("category", ""),
        )

        judgment = judge.evaluate_dims(sample, trace=trace)

        record = {
            "trace_id":    trace_id,
            "index":       i,
            "category":    entry.get("category", "unknown"),
            "prompt_variant": variant,
            "question":    entry["question"],
            "llm_overall": judgment.overall,
            "llm_reasoning": judgment.reasoning,
            **{f"llm_{d}": judgment.dims[d] for d in DIM_KEYS},
        }
        records.append(record)

        flags = " ".join("✓" if judgment.dims[d] else "✗" for d in DIM_KEYS)
        print(f"  [{i+1:>3}/{len(dataset)}] {judgment.overall:<14s} {flags}  {entry['question'][:55]}")

        if i < len(dataset) - 1:
            time.sleep(delay)

    # ── Save labels ───────────────────────────────────────────────────────────
    json_path = out_dir / f"labels_{variant}.json"
    csv_path  = out_dir / f"labels_{variant}.csv"

    json_path.write_text(json.dumps(records, indent=2))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    resolved = sum(1 for r in records if r["llm_overall"] == "RESOLVED")
    print(f"\n  RESOLVED    : {resolved}/{len(records)}")
    print(f"  NOT_RESOLVED: {len(records) - resolved}/{len(records)}")
    print(f"  Saved → {json_path.name}  {csv_path.name}")

    # ── Trace summary ─────────────────────────────────────────────────────────
    trace.summary()

    return records


# ─── Analysis ────────────────────────────────────────────────────────

def _dim_pass_rates(rows: list[dict], source: str, dim_keys: list[str]) -> dict[str, float]:
    totals = {d: 0 for d in dim_keys}
    passes = {d: 0 for d in dim_keys}
    for row in rows:
        for d in dim_keys:
            totals[d] += 1
            passes[d] += int(row[f"{source}_{d}"])
    return {d: passes[d] / totals[d] if totals[d] else 0.0 for d in dim_keys}


def _agreement_rates(rows: list[dict], dim_keys: list[str]) -> dict[str, float]:
    totals = {d: 0 for d in dim_keys}
    agrees = {d: 0 for d in dim_keys}
    for row in rows:
        for d in dim_keys:
            totals[d] += 1
            agrees[d] += int(row[f"human_{d}"] == row[f"llm_{d}"])
    return {d: agrees[d] / totals[d] if totals[d] else 0.0 for d in dim_keys}


def run_analysis_with_human(
    label_records: list[dict],
    human_reviews: list[dict],
    dataset:       list[dict],
    variant:       str,
    out_dir:       Path,
) -> None:
    """Full Step 5 when human reviews are available."""
    from judge import DimJudgment, DIM_KEYS
    from analysis import AnalysisReport

    # Build DimJudgment objects so AnalysisReport can consume them
    judgments = []
    for rec in label_records:
        judgments.append(DimJudgment(
            **{d: rec[f"llm_{d}"] for d in DIM_KEYS},
            overall=rec["llm_overall"],
            reasoning=rec.get("llm_reasoning", ""),
        ))

    report = AnalysisReport(
        entries       = dataset,
        human_reviews = human_reviews,
        llm_judgments = judgments,
    )

    print(f"\n── Step 5: Analysis Report ({variant}) ────────────────────────────")
    report.print_summary()
    report.save(str(out_dir / f"analysis_{variant}"))
    report.save_charts(out_dir, variant)


def run_analysis_llm_only(
    label_records: list[dict],
    variant:       str,
    out_dir:       Path,
) -> None:
    """LLM-only segment analysis + charts."""
    import matplotlib.pyplot as plt
    import numpy as np
    from judge import DIM_KEYS

    segments  = sorted({r["category"] for r in label_records})
    n_total   = len(label_records)
    resolved  = sum(1 for r in label_records if r["llm_overall"] == "RESOLVED")

    print(f"\n── LLM-Only Analysis ({variant}) ──────────────────────────")
    print(f"  {'─'*60}")
    print(f"  Total entries : {n_total}")
    print(f"  RESOLVED      : {resolved}  ({resolved/n_total:.1%})")
    print(f"  NOT_RESOLVED  : {n_total - resolved}  ({(n_total - resolved)/n_total:.1%})\n")

    overall_rates = _dim_pass_rates(label_records, "llm", DIM_KEYS)
    print(f"  {'Dimension':<30s}  {'Pass rate':>9}")
    print(f"  {'─'*42}")
    for d in DIM_KEYS:
        print(f"  {d:<30s}  {overall_rates[d]:>8.1%}")

    by_seg: dict[str, dict] = {}
    for seg in segments:
        subset = [r for r in label_records if r["category"] == seg]
        by_seg[seg] = {
            "n":          len(subset),
            "resolved":   sum(1 for r in subset if r["llm_overall"] == "RESOLVED"),
            "pass_rates": _dim_pass_rates(subset, "llm", DIM_KEYS),
        }

    print("\n  Segment breakdown:")
    print(f"  {'─'*60}")
    for seg, stats in by_seg.items():
        pct = stats["resolved"] / stats["n"]
        print(f"  {seg:<22s}  {stats['resolved']}/{stats['n']} RESOLVED ({pct:.1%})")

    # ── Chart 1: overall dim pass rates ──────────────────────────────────────
    labels = [d.split("_", 1)[1].replace("_", " ") for d in DIM_KEYS]
    values = [overall_rates[d] for d in DIM_KEYS]
    colors = ["#2E7D32" if v >= 0.8 else "#F57F17" if v >= 0.6 else "#C62828" for v in values]

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pass rate")
    ax.set_title(f"LLM Judge Pass Rate per Dimension  [{variant}]")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=1, label="0.8 target")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.0%}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    p = out_dir / f"chart_{variant}_dim_pass_rates.png"
    plt.savefig(p, dpi=150)
    plt.show()
    print(f"\n  Saved → {p.name}")

    # ── Chart 2: segment × dimension heatmap ─────────────────────────────────
    matrix = np.array([
        [by_seg[seg]["pass_rates"][d] for d in DIM_KEYS]
        for seg in segments
    ])
    fig, ax = plt.subplots(figsize=(13, len(segments) * 0.9 + 1.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DIM_KEYS)))
    ax.set_xticklabels(DIM_KEYS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels(segments, fontsize=10)
    ax.set_title(f"LLM Pass Rate — Segment × Dimension  [{variant}]")
    for i in range(len(segments)):
        for j in range(len(DIM_KEYS)):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    p = out_dir / f"chart_{variant}_segment_heatmap.png"
    plt.savefig(p, dpi=150)
    plt.show()
    print(f"  Saved → {p.name}")

    # ── Chart 3: per-segment RESOLVED rate bars ───────────────────────────────
    seg_labels = list(by_seg.keys())
    seg_rates  = [by_seg[s]["resolved"] / by_seg[s]["n"] for s in seg_labels]
    seg_colors = ["#2E7D32" if v >= 0.8 else "#F57F17" if v >= 0.6 else "#C62828" for v in seg_rates]

    fig, ax = plt.subplots(figsize=(8, len(seg_labels) * 0.7 + 1.2))
    bars = ax.barh(seg_labels, seg_rates, color=seg_colors, alpha=0.85)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("RESOLVED rate")
    ax.set_title(f"RESOLVED Rate per Segment  [{variant}]")
    ax.axvline(0.8, color="grey", linestyle="--", linewidth=1)
    for bar, val in zip(bars, seg_rates):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=10)
    plt.tight_layout()
    p = out_dir / f"chart_{variant}_segment_bars.png"
    plt.savefig(p, dpi=150)
    plt.show()
    print(f"  Saved → {p.name}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset: list[dict] = json.loads(Path(args.dataset).read_text())
    if args.limit:
        dataset = dataset[:args.limit]
    print(f"Loaded {len(dataset)} entries from {args.dataset}")

    # ── Load human reviews (optional) ─────────────────────────────────────────
    human_reviews: list[dict] | None = None
    if args.human:
        human_reviews = json.loads(Path(args.human).read_text())
        print(f"Loaded {len(human_reviews)} human reviews from {args.human}")

    # ── LLM Judge ───────────────────────────────────────────────────────────────
    label_records = run_judge(
        dataset  = dataset,
        variant  = args.prompt,
        out_dir  = out_dir,
        delay    = args.delay,
    )

    # ── Analysis────────────────────────────────────────────────────────────────
    if human_reviews:
        run_analysis_with_human(
            label_records = label_records,
            human_reviews = human_reviews,
            dataset       = dataset,
            variant       = args.prompt,
            out_dir       = out_dir,
        )
    else:
        run_analysis_llm_only(
            label_records = label_records,
            variant       = args.prompt,
            out_dir       = out_dir,
        )

    print(f"\nDone. All output in: {out_dir}/\n")


if __name__ == "__main__":
    main()
