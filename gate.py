"""
Quality gate: per-item checks applied to generated repair entries.

Each check returns {"passed": bool, "reason": str}.
run_quality_gate(item) returns {"passed": bool, "checks": dict[str, check_result]}.
"""
from __future__ import annotations

import re
from collections import Counter

from generate import RepairEntry

# ── Blocklists ────────────────────────────────────────────────────────────────

_D2_GENERIC_PHRASES = [
    "be careful", "use caution", "stay safe", "be cautious",
    "exercise caution", "take care", "be safe", "caution is advised",
]

_D3_TRADE_TOOLS = [
    "oscilloscope", "pipe threading machine", "pipe threader",
    "refrigerant recovery machine", "manifold gauge set", "hvac gauges",
    "pipe bending machine", "pipe bender", "wire stripper set",  # ok at hw store but let's keep
    "arc welder", "mig welder", "tig welder", "plasma cutter",
    "hydraulic press", "hydraulic jack set",
    "borescope", "gas leak detector",           # >$50 threshold edge cases
    "thermal imaging camera", "infrared camera",
    "digital multimeter clamp",                 # fine; clamp meter is trade-grade
    "conduit bender", "knockout punch set",
]

_D6_GENERIC_TIP_PHRASES = [
    "if in doubt, call a professional",
    "when in doubt call",
    "call a professional",
    "contact a professional",
    "hire a professional",
    "consult a professional",
    "consult an expert",
    "if you're not sure",
    "if you are not sure",
    "be sure to",
    "make sure to",
    "always remember",
    "remember to",
    "don't forget",
    "do not forget",
]


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_schema(item: dict) -> dict:
    """Pydantic structural validation."""
    try:
        RepairEntry.model_validate(item)
        return {"passed": True, "reason": "All required fields present and valid"}
    except Exception as e:
        return {"passed": False, "reason": str(e)[:200]}


def _check_step_count(item: dict) -> dict:
    steps = item.get("steps", [])
    n = len(steps)
    if n < 3:
        return {"passed": False, "reason": f"Only {n} steps — minimum is 3"}
    if n > 9:
        return {"passed": False, "reason": f"{n} steps — maximum is 9"}
    return {"passed": True, "reason": f"{n} steps"}


def _check_d2_safety(item: dict) -> dict:
    """D2 pre-check: safety_info must not be purely generic."""
    safety = (item.get("safety_info") or "").strip().lower()
    if len(safety) < 20:
        return {"passed": False, "reason": f"safety_info too short ({len(safety)} chars)"}
    for phrase in _D2_GENERIC_PHRASES:
        if phrase in safety and len(safety) < 60:
            return {"passed": False, "reason": f'safety_info contains generic phrase "{phrase}" with no additional content'}
    return {"passed": True, "reason": "Safety info appears specific"}


def _check_d3_tools(item: dict) -> dict:
    """D3 pre-check: no professional/trade-only tools."""
    tools_raw = " ".join(item.get("tools_required", [])).lower()
    for tool in _D3_TRADE_TOOLS:
        if tool in tools_raw:
            return {"passed": False, "reason": f'Professional/trade tool detected: "{tool}"'}
    return {"passed": True, "reason": "No trade-only tools detected"}


def _check_d6_tips(item: dict) -> dict:
    """D6 pre-check: tip must not be a generic call-a-pro phrase."""
    tips_text = " ".join(item.get("tips", [])).lower()
    if not tips_text.strip():
        return {"passed": False, "reason": "No tips provided"}
    for phrase in _D6_GENERIC_TIP_PHRASES:
        if phrase in tips_text and len(tips_text) < 80:
            return {"passed": False, "reason": f'Tip contains generic phrase "{phrase}" with no additional insight'}
    return {"passed": True, "reason": "Tip appears specific"}


def _check_answer_format(item: dict) -> dict:
    """Answer must contain the preamble + numbered steps. Safety is validated separately via safety_info."""
    answer = item.get("answer", "")
    has_preamble = "before you start" in answer.lower()
    has_steps    = bool(re.search(r"\n\s*\d+\.", answer))
    if not has_preamble:
        return {"passed": False, "reason": "Answer missing 'Before you start' preamble"}
    if not has_steps:
        return {"passed": False, "reason": "Answer missing numbered steps"}
    return {"passed": True, "reason": "Answer format looks correct"}


def _check_tools_in_answer(item: dict) -> dict:
    """Every tool listed in tools_required should appear somewhere in the answer."""
    answer = item.get("answer", "").lower()
    tools  = item.get("tools_required", [])
    missing = []
    for tool in tools:
        # check each significant word; fall back to full tool string for short names (e.g. "saw", "awl")
        words = [w for w in tool.lower().split() if len(w) > 3]
        if not words:
            words = [tool.lower()]
        if not any(w in answer for w in words):
            missing.append(tool)
    if missing:
        return {"passed": False, "reason": f"Tools not mentioned in answer: {', '.join(missing)}"}
    return {"passed": True, "reason": "All tools referenced in answer"}


# ── Per-item gate ─────────────────────────────────────────────────────────────

_CHECKS = [
    ("schema_validation",  _check_schema),
    ("step_count",         _check_step_count),
    ("answer_format",      _check_answer_format),
    ("d2_safety",          _check_d2_safety),
    ("d3_tools",           _check_d3_tools),
    ("d6_tips",            _check_d6_tips),
    ("tools_in_answer",    _check_tools_in_answer),
]


def run_quality_gate(item: dict) -> dict:
    """
    Run all per-item checks. Returns:
      {"passed": bool, "checks": {check_name: {"passed": bool, "reason": str}}}
    """
    results = {}
    all_passed = True
    for name, fn in _CHECKS:
        result = fn(item)
        results[name] = result
        if not result["passed"]:
            all_passed = False
    return {"passed": all_passed, "checks": results}


# ── Batch-level checks ────────────────────────────────────────────────────────

def check_distribution(items: list[dict], expected_categories: list[str]) -> dict:
    """
    Returns per-category counts and flags categories under 15% of total.
    """
    total = len(items)
    counts = Counter(i.get("category", "unknown") for i in items)
    result = {}
    all_ok = True
    for cat in expected_categories:
        count = counts.get(cat, 0)
        pct   = count / total if total else 0.0
        ok    = pct >= 0.15
        if not ok:
            all_ok = False
        result[cat] = {"count": count, "pct": round(pct, 3), "ok": ok}
    return {"passed": all_ok, "categories": result, "total": total}


def check_deduplication(items: list[dict]) -> dict:
    """
    Exact dedup via normalized question text. Returns list of duplicate index pairs.
    """
    def _norm(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9 ]", "", text.lower().strip())
        return normalized if normalized else text.lower().strip()

    seen: dict[str, int] = {}
    dupes: list[tuple[int, int]] = []
    for i, item in enumerate(items):
        key = _norm(item.get("question", ""))
        if key in seen:
            dupes.append((seen[key], i))
        else:
            seen[key] = i
    return {"passed": len(dupes) == 0, "duplicate_pairs": dupes, "count": len(dupes)}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import json
    from pathlib import Path
    from generate import CATEGORIES

    p = argparse.ArgumentParser(description="Run quality gate on a repair dataset")
    p.add_argument("--dataset", required=True, help="Dataset JSON file (list of repair entries)")
    p.add_argument("--out-dir", default=None,  help="Save gate_results.json here (optional)")
    args = p.parse_args()

    dataset: list[dict] = json.loads(Path(args.dataset).read_text())
    print(f"Loaded {len(dataset)} entries from {args.dataset}\n")

    gate_results = [run_quality_gate(entry) for entry in dataset]

    for i, (entry, result) in enumerate(zip(dataset, gate_results)):
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{i+1:>3}] {status}  {entry.get('category', 'unknown'):<22s}  {entry.get('question', '')[:60]}")
        if not result["passed"]:
            for name, check in result["checks"].items():
                if not check["passed"]:
                    print(f"         ✗ {name}: {check['reason']}")

    if not dataset:
        print("  (empty dataset — nothing to gate)")
        return

    pass_count = sum(1 for r in gate_results if r["passed"])
    total = len(dataset)
    print("\n── Gate Summary ─────────────────────────────────────────────")
    print(f"  Passed : {pass_count}/{total} ({pass_count/total:.1%})")
    print(f"  Failed : {total - pass_count}/{total}")

    dist = check_distribution(dataset, CATEGORIES)
    print("\n── Category Distribution ────────────────────────────────────")
    for cat, info in dist["categories"].items():
        marker = "✓" if info["ok"] else "✗"
        print(f"  {marker} {cat:<22s}  {info['count']:>3}  ({info['pct']:.1%})")

    dedup = check_deduplication(dataset)
    if dedup["count"]:
        print(f"\n── Duplicate Pairs ({dedup['count']}) ─────────────────────────────")
        for i, j in dedup["duplicate_pairs"]:
            print(f"  [{i+1}] ↔ [{j+1}]")
    else:
        print("\n  No exact duplicates found.")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        records = [{"index": i, **result} for i, result in enumerate(gate_results)]
        out_path = out / "gate_results.json"
        out_path.write_text(json.dumps(records, indent=2))
        print(f"\n  Saved → {out_path}")

    print()


if __name__ == "__main__":
    main()
