"""Tests for gate.py — all per-item checks and batch-level checks."""
import pytest
from gate import (
    _check_schema,
    _check_step_count,
    _check_d2_safety,
    _check_d3_tools,
    _check_d6_tips,
    _check_answer_format,
    _check_tools_in_answer,
    run_quality_gate,
    check_distribution,
    check_deduplication,
)

VALID_ENTRY = {
    "question": "My kitchen faucet is dripping — how do I fix it?",
    "equipment_problem": "Dripping kitchen faucet",
    "answer": (
        "Before you start, here are some useful things to know: "
        "A worn washer causes most dripping faucets.\n\n"
        "Now, gather adjustable wrench, replacement washer, and screwdriver "
        "and follow these steps:\n\n"
        "1. Turn off the water supply valve under the sink\n"
        "2. Remove the faucet handle by unscrewing the set screw\n"
        "3. Replace the worn washer inside the valve seat\n"
        "4. Reassemble and test"
    ),
    "category": "plumbing",
    "tools_required": ["adjustable wrench", "replacement washer", "screwdriver"],
    "steps": [
        "Turn off the water supply valve under the sink",
        "Remove the faucet handle by unscrewing the set screw",
        "Replace the worn washer inside the valve seat",
        "Reassemble and test",
    ],
    "safety_info": "Turn off the water supply valve before starting to prevent flooding.",
    "tips": ["A worn washer is the most common cause — check it first before replacing other parts."],
}

CATEGORIES = ["plumbing", "electrical", "HVAC", "appliances", "general home repair"]


# ── _check_schema ──────────────────────────────────────────────────────────────

class TestCheckSchema:
    def test_valid_entry_passes(self):
        assert _check_schema(VALID_ENTRY)["passed"] is True

    def test_missing_question_fails(self):
        entry = {k: v for k, v in VALID_ENTRY.items() if k != "question"}
        assert _check_schema(entry)["passed"] is False

    def test_missing_safety_info_fails(self):
        entry = {k: v for k, v in VALID_ENTRY.items() if k != "safety_info"}
        assert _check_schema(entry)["passed"] is False

    def test_missing_answer_fails(self):
        entry = {k: v for k, v in VALID_ENTRY.items() if k != "answer"}
        assert _check_schema(entry)["passed"] is False

    def test_empty_steps_list_fails(self):
        assert _check_schema({**VALID_ENTRY, "steps": []})["passed"] is False

    def test_two_steps_fails(self):
        assert _check_schema({**VALID_ENTRY, "steps": ["s1", "s2"]})["passed"] is False

    def test_empty_tools_list_fails(self):
        assert _check_schema({**VALID_ENTRY, "tools_required": []})["passed"] is False

    def test_empty_tips_list_fails(self):
        assert _check_schema({**VALID_ENTRY, "tips": []})["passed"] is False

    def test_reason_on_failure(self):
        result = _check_schema({**VALID_ENTRY, "steps": []})
        assert len(result["reason"]) > 0


# ── _check_step_count ──────────────────────────────────────────────────────────

class TestCheckStepCount:
    def test_valid_count_passes(self):
        assert _check_step_count(VALID_ENTRY)["passed"] is True

    def test_two_steps_fails(self):
        result = _check_step_count({**VALID_ENTRY, "steps": ["s1", "s2"]})
        assert result["passed"] is False
        assert "2" in result["reason"]

    def test_ten_steps_fails(self):
        assert _check_step_count({**VALID_ENTRY, "steps": [f"s{i}" for i in range(10)]})["passed"] is False

    def test_boundary_min_three_passes(self):
        assert _check_step_count({**VALID_ENTRY, "steps": ["s1", "s2", "s3"]})["passed"] is True

    def test_boundary_max_nine_passes(self):
        assert _check_step_count({**VALID_ENTRY, "steps": [f"s{i}" for i in range(9)]})["passed"] is True

    def test_reason_contains_count(self):
        result = _check_step_count(VALID_ENTRY)
        assert "4" in result["reason"]


# ── _check_d2_safety ───────────────────────────────────────────────────────────

class TestCheckD2Safety:
    def test_specific_safety_passes(self):
        assert _check_d2_safety(VALID_ENTRY)["passed"] is True

    def test_generic_short_phrase_fails(self):
        entry = {**VALID_ENTRY, "safety_info": "Be careful when working."}
        assert _check_d2_safety(entry)["passed"] is False

    def test_too_short_fails(self):
        assert _check_d2_safety({**VALID_ENTRY, "safety_info": "careful"})["passed"] is False

    def test_empty_fails(self):
        assert _check_d2_safety({**VALID_ENTRY, "safety_info": ""})["passed"] is False

    def test_generic_phrase_in_long_string_passes(self):
        # "be careful" is in a long, specific sentence — should pass
        long_specific = (
            "Be careful to turn off the circuit breaker before touching any wiring "
            "to avoid electric shock — use a non-contact voltage tester to confirm power is off."
        )
        assert _check_d2_safety({**VALID_ENTRY, "safety_info": long_specific})["passed"] is True

    @pytest.mark.parametrize("phrase", [
        "use caution", "stay safe", "be cautious", "exercise caution",
    ])
    def test_various_generic_phrases_fail_when_short(self, phrase):
        assert _check_d2_safety({**VALID_ENTRY, "safety_info": f"{phrase}."})["passed"] is False


# ── _check_d3_tools ────────────────────────────────────────────────────────────

class TestCheckD3Tools:
    def test_normal_tools_pass(self):
        assert _check_d3_tools(VALID_ENTRY)["passed"] is True

    def test_oscilloscope_fails(self):
        result = _check_d3_tools({**VALID_ENTRY, "tools_required": ["screwdriver", "oscilloscope"]})
        assert result["passed"] is False
        assert "oscilloscope" in result["reason"]

    def test_manifold_gauge_set_fails(self):
        assert _check_d3_tools({**VALID_ENTRY, "tools_required": ["manifold gauge set"]})["passed"] is False

    def test_pipe_threading_machine_fails(self):
        assert _check_d3_tools({**VALID_ENTRY, "tools_required": ["pipe threading machine"]})["passed"] is False

    def test_arc_welder_fails(self):
        assert _check_d3_tools({**VALID_ENTRY, "tools_required": ["arc welder"]})["passed"] is False

    def test_empty_tools_passes(self):
        assert _check_d3_tools({**VALID_ENTRY, "tools_required": []})["passed"] is True

    @pytest.mark.parametrize("tool", [
        "screwdriver", "adjustable wrench", "pliers", "level", "tape measure",
        "drill", "hammer", "utility knife", "voltage tester",
    ])
    def test_common_tools_pass(self, tool):
        assert _check_d3_tools({**VALID_ENTRY, "tools_required": [tool]})["passed"] is True


# ── _check_d6_tips ─────────────────────────────────────────────────────────────

class TestCheckD6Tips:
    def test_specific_tip_passes(self):
        assert _check_d6_tips(VALID_ENTRY)["passed"] is True

    def test_empty_list_fails(self):
        assert _check_d6_tips({**VALID_ENTRY, "tips": []})["passed"] is False

    def test_generic_call_pro_short_fails(self):
        assert _check_d6_tips({**VALID_ENTRY, "tips": ["Call a professional."]})["passed"] is False

    def test_dont_forget_short_fails(self):
        assert _check_d6_tips({**VALID_ENTRY, "tips": ["Don't forget to be safe."]})["passed"] is False

    def test_specific_tip_mentioning_pro_passes(self):
        long_tip = (
            "Call a professional if you find corrosion on the supply lines, "
            "as this indicates the pipes may need full replacement rather than just washer repair."
        )
        assert _check_d6_tips({**VALID_ENTRY, "tips": [long_tip]})["passed"] is True

    def test_whitespace_only_tips_fail(self):
        assert _check_d6_tips({**VALID_ENTRY, "tips": ["   "]})["passed"] is False


# ── _check_answer_format ───────────────────────────────────────────────────────

class TestCheckAnswerFormat:
    def test_valid_format_passes(self):
        assert _check_answer_format(VALID_ENTRY)["passed"] is True

    def test_missing_preamble_fails(self):
        no_preamble = "1. Turn off water\n2. Replace washer\n3. Reassemble"
        result = _check_answer_format({**VALID_ENTRY, "answer": no_preamble})
        assert result["passed"] is False
        assert "preamble" in result["reason"].lower()

    def test_missing_numbered_steps_fails(self):
        no_steps = "Before you start, here are some useful things to know: check the washer. Just do the repair."
        result = _check_answer_format({**VALID_ENTRY, "answer": no_steps})
        assert result["passed"] is False
        assert "steps" in result["reason"].lower()

    def test_preamble_case_insensitive(self):
        upper = VALID_ENTRY["answer"].replace("Before you start", "BEFORE YOU START")
        assert _check_answer_format({**VALID_ENTRY, "answer": upper})["passed"] is True


# ── _check_tools_in_answer ─────────────────────────────────────────────────────

class TestCheckToolsInAnswer:
    def test_all_tools_present_passes(self):
        assert _check_tools_in_answer(VALID_ENTRY)["passed"] is True

    def test_missing_tool_fails(self):
        entry = {**VALID_ENTRY, "tools_required": ["adjustable wrench", "pipe cutter"]}
        result = _check_tools_in_answer(entry)
        assert result["passed"] is False
        assert "pipe cutter" in result["reason"]

    def test_multi_word_tool_partial_match(self):
        # "screwdriver set" — word "screwdriver" appears in answer
        entry = {**VALID_ENTRY, "tools_required": ["screwdriver set"]}
        assert _check_tools_in_answer(entry)["passed"] is True

    def test_empty_tools_passes(self):
        assert _check_tools_in_answer({**VALID_ENTRY, "tools_required": []})["passed"] is True

    def test_reason_lists_missing_tools(self):
        entry = {**VALID_ENTRY, "tools_required": ["xenon spectrometer"]}
        result = _check_tools_in_answer(entry)
        assert "xenon spectrometer" in result["reason"]


# ── run_quality_gate ───────────────────────────────────────────────────────────

class TestRunQualityGate:
    def test_valid_entry_fully_passes(self):
        result = run_quality_gate(VALID_ENTRY)
        assert result["passed"] is True
        assert all(v["passed"] for v in result["checks"].values())

    def test_returns_all_seven_checks(self):
        result = run_quality_gate(VALID_ENTRY)
        expected = {
            "schema_validation", "step_count", "answer_format",
            "d2_safety", "d3_tools", "d6_tips", "tools_in_answer",
        }
        assert set(result["checks"].keys()) == expected

    def test_trade_tool_fails_gate(self):
        entry = {**VALID_ENTRY, "tools_required": ["oscilloscope"]}
        result = run_quality_gate(entry)
        assert result["passed"] is False
        assert result["checks"]["d3_tools"]["passed"] is False

    def test_failed_check_sets_overall_false(self):
        entry = {**VALID_ENTRY, "steps": ["only one step"]}
        assert run_quality_gate(entry)["passed"] is False

    def test_each_check_has_passed_and_reason(self):
        result = run_quality_gate(VALID_ENTRY)
        for name, check in result["checks"].items():
            assert "passed" in check, f"{name} missing 'passed'"
            assert "reason" in check, f"{name} missing 'reason'"


# ── check_distribution ─────────────────────────────────────────────────────────

class TestCheckDistribution:
    def test_balanced_passes(self):
        items = [{"category": c} for c in CATEGORIES * 4]  # 20 items, 4 each
        result = check_distribution(items, CATEGORIES)
        assert result["passed"] is True
        for cat in CATEGORIES:
            assert result["categories"][cat]["ok"] is True

    def test_missing_category_fails(self):
        items = [{"category": "plumbing"}] * 20
        result = check_distribution(items, CATEGORIES)
        assert result["passed"] is False
        assert result["categories"]["electrical"]["ok"] is False

    def test_total_count_correct(self):
        items = [{"category": "plumbing"}] * 5
        assert check_distribution(items, CATEGORIES)["total"] == 5

    def test_percentages_sum_to_one(self):
        items = [{"category": c} for c in CATEGORIES * 2]
        result = check_distribution(items, CATEGORIES)
        total_pct = sum(result["categories"][c]["pct"] for c in CATEGORIES)
        assert abs(total_pct - 1.0) < 0.01

    def test_empty_dataset(self):
        result = check_distribution([], CATEGORIES)
        assert result["total"] == 0


# ── check_deduplication ────────────────────────────────────────────────────────

class TestCheckDeduplication:
    def test_no_duplicates_passes(self):
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "How do I fix a flickering light?"},
        ]
        result = check_deduplication(items)
        assert result["passed"] is True
        assert result["count"] == 0

    def test_exact_duplicate_fails(self):
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "How do I fix a dripping faucet?"},
        ]
        result = check_deduplication(items)
        assert result["passed"] is False
        assert result["count"] == 1
        assert (0, 1) in result["duplicate_pairs"]

    def test_normalizes_punctuation(self):
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "How do I fix a dripping faucet"},
        ]
        assert check_deduplication(items)["passed"] is False

    def test_normalizes_case(self):
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "HOW DO I FIX A DRIPPING FAUCET?"},
        ]
        assert check_deduplication(items)["passed"] is False

    def test_empty_list_passes(self):
        result = check_deduplication([])
        assert result["passed"] is True
        assert result["count"] == 0

    def test_multiple_duplicates_detected(self):
        items = [
            {"question": "Q1"},
            {"question": "Q2"},
            {"question": "Q1"},
            {"question": "Q2"},
        ]
        result = check_deduplication(items)
        assert result["count"] == 2
