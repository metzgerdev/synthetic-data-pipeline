"""
CLI reviewer for the DIY repair dataset.

Walks through each entry and collects binary pass/fail labels on
6 quality dimensions. Results are saved to review_results.json with
a trace_id per entry.

Usage:
    python review.py                          # reviews all entries
    python review.py --dataset repair_dataset.json
    python review.py --start 10              # resume from entry 10
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ─── Quality dimensions ───────────────────────────────────────────────────────

DIMENSIONS = [
    (
        "D1_answer_completeness",
        "Does the answer include tools, concrete steps, safety, and a useful tip — "
        "enough for a homeowner to complete the repair end to end?",
    ),
    (
        "D2_safety_specificity",
        "Does safety_info name the SPECIFIC hazard and precaution for THIS repair? "
        "Fail if it only says 'be careful', 'use caution', or 'stay safe'.",
    ),
    (
        "D3_tool_realism",
        "Is every tool something a typical homeowner owns or could buy at a hardware "
        "store for under $50? Fail if any tool is professional, specialty, or trade-only.",
    ),
    (
        "D4_scope_appropriateness",
        "Is the repair within realistic DIY capability? If professional help is needed "
        "(gas lines, panel work, etc.) does the answer say so clearly?",
    ),
    (
        "D5_context_clarity",
        "Do the question and answer contain enough context to understand the problem, "
        "and does the answer directly address the specific repair described?",
    ),
    (
        "D6_tip_usefulness",
        "Do the tips provide non-obvious, task-specific advice beyond the steps? "
        "Fail if tips merely restate a step or offer generic encouragement.",
    ),
]

# ─── Terminal helpers ─────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
DIM    = "\033[2m"

def _h(text: str, color: str = CYAN) -> str:
    return f"{BOLD}{color}{text}{RESET}"

def _dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"

def _clear() -> None:
    print("\033[2J\033[H", end="")

def _rule() -> None:
    print(_dim("─" * 72))

def _ask_binary(prompt: str) -> bool | None:
    """Returns True=pass, False=fail, None=skip entry."""
    while True:
        raw = input(f"  {prompt} [p/f/s=skip] ").strip().lower()
        if raw in ("p", "pass", "y", "yes", "1"):
            return True
        if raw in ("f", "fail", "n", "no", "0"):
            return False
        if raw in ("s", "skip"):
            return None
        print("  Please enter p (pass), f (fail), or s (skip).")

# ─── Display ──────────────────────────────────────────────────────────────────

def _display_entry(idx: int, total: int, entry: dict) -> None:
    _clear()
    print(_h(f"\n  Entry {idx + 1} / {total}  ·  {entry.get('category', '').upper()}", CYAN))
    _rule()

    print(_h("\n  QUESTION", YELLOW))
    print(f"  {entry.get('question', '')}\n")

    print(_h("  ANSWER", YELLOW))
    for line in entry.get("answer", "").splitlines():
        print(f"  {line}")

    print(_h("\n  TOOLS", YELLOW))
    for tool in entry.get("tools_required", []):
        print(f"    • {tool}")

    print(_h("\n  STEPS", YELLOW))
    for i, step in enumerate(entry.get("steps", []), 1):
        print(f"    {i}. {step}")

    print(_h("\n  SAFETY", YELLOW))
    print(f"  {entry.get('safety_info', '')}")

    print(_h("\n  TIPS", YELLOW))
    for tip in entry.get("tips", []):
        print(f"    • {tip}")

    _rule()

# ─── Core review loop ─────────────────────────────────────────────────────────

def review(dataset_path: Path, output_path: Path, start: int) -> None:
    with open(dataset_path) as f:
        entries: list[dict] = json.load(f)

    # load existing results so we can resume
    existing: dict[str, dict] = {}
    if output_path.exists():
        for row in json.loads(output_path.read_text()):
            existing[row["trace_id"]] = row

    results: list[dict] = list(existing.values())
    reviewed_indices = {r["index"] for r in results}

    total = len(entries)
    saved = 0

    print(_h("\n  DIY Repair Dataset Reviewer", CYAN))
    print(_dim(f"  Dataset : {dataset_path.name}  ({total} entries)"))
    print(_dim(f"  Output  : {output_path.name}"))
    print(_dim(f"  Already reviewed: {len(reviewed_indices)}"))
    print(_dim("\n  Controls: p=pass  f=fail  s=skip entry  q=quit\n"))
    input("  Press Enter to begin...")

    for idx, entry in enumerate(entries):
        if idx < start or idx in reviewed_indices:
            continue

        _display_entry(idx, total, entry)

        print(_h("\n  QUALITY REVIEW", GREEN))
        labels: dict[str, bool] = {}
        skipped = False

        for key, question in DIMENSIONS:
            result = _ask_binary(f"{key:<26s}  {question}")
            if result is None:
                skipped = True
                print(_dim("  Entry skipped.\n"))
                break
            labels[key] = result

        if skipped:
            continue

        # check for quit after all dimensions answered
        quit_after = input(_dim("\n  Continue? [Enter / q=quit] ")).strip().lower()

        row = {
            "trace_id":  str(uuid.uuid4()),
            "index":     idx,
            "category":  entry.get("category", ""),
            "question":  entry.get("question", ""),
            "labels":    labels,
            "pass_count": sum(labels.values()),
            "fail_count": sum(1 for v in labels.values() if not v),
            "all_pass":  all(labels.values()),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

        results.append(row)
        saved += 1

        output_path.write_text(json.dumps(results, indent=2))
        print(_dim(f"  Saved ({saved} new this session, {len(results)} total)"))

        if quit_after == "q":
            break

    _clear()
    _print_summary(results)

# ─── Summary ─────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No results yet.")
        return

    print(_h("\n  REVIEW SUMMARY", CYAN))
    _rule()
    print(f"  Total reviewed : {len(results)}")
    print(f"  All-pass       : {sum(r['all_pass'] for r in results)}")
    print()

    dim_totals: dict[str, list[int]] = {k: [0, 0] for k, _ in DIMENSIONS}
    for row in results:
        for key, val in row["labels"].items():
            dim_totals[key][0] += int(val)   # passes
            dim_totals[key][1] += 1          # total

    print(_h("  Pass rate per dimension:", YELLOW))
    for key, (passes, total) in dim_totals.items():
        if total == 0:
            continue
        pct  = passes / total * 100
        bar  = "█" * int(pct / 5)
        color = GREEN if pct >= 80 else (YELLOW if pct >= 60 else RED)
        print(f"  {key:<26s}  {_h(f'{pct:5.1f}%', color)}  {_dim(bar)}")

    _rule()

# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DIY repair dataset reviewer")
    parser.add_argument(
        "--dataset", default="repair_dataset_v2_deduped.json",
        help="Dataset JSON file (default: repair_dataset_v2_deduped.json)",
    )
    parser.add_argument(
        "--output", default="review_results.json",
        help="Output file for labels (default: review_results.json)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Entry index to start from (default: 0)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary of existing results and exit",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    dataset_path = base / args.dataset
    output_path  = base / args.output

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    if args.summary:
        results = json.loads(output_path.read_text()) if output_path.exists() else []
        _print_summary(results)
        return

    review(dataset_path, output_path, start=args.start)


if __name__ == "__main__":
    main()
