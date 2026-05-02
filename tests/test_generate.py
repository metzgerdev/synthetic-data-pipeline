"""Tests for generate.py — schema validation, dedup, distribution."""
import pytest
from pydantic import ValidationError
from generate import (
    CATEGORIES,
    DataGenerator,
    DistributionResult,
    RepairEntry,
)

VALID_DICT = {
    "question": "My kitchen faucet is dripping — how do I fix it?",
    "equipment_problem": "Dripping kitchen faucet",
    "answer": (
        "Before you start, here are some useful things to know: "
        "A worn washer causes most dripping faucets.\n\n"
        "Now, gather adjustable wrench and follow these steps:\n\n"
        "1. Turn off the water supply\n"
        "2. Remove the handle\n"
        "3. Replace the washer"
    ),
    "category": "plumbing",
    "tools_required": ["adjustable wrench"],
    "steps": ["Turn off the water supply", "Remove the handle", "Replace the washer"],
    "safety_info": "Turn off the supply valve before starting to prevent flooding.",
    "tips": ["A worn washer is the most common cause — check it first."],
}


def _gen() -> DataGenerator:
    """Create a DataGenerator without calling __init__ (avoids Groq import)."""
    return object.__new__(DataGenerator)


# ── RepairEntry schema ─────────────────────────────────────────────────────────

class TestRepairEntrySchema:
    def test_valid_entry_accepted(self):
        entry = RepairEntry.model_validate(VALID_DICT)
        assert entry.category == "plumbing"
        assert len(entry.steps) == 3

    def test_missing_question_rejected(self):
        data = {k: v for k, v in VALID_DICT.items() if k != "question"}
        with pytest.raises(ValidationError):
            RepairEntry.model_validate(data)

    def test_missing_answer_rejected(self):
        data = {k: v for k, v in VALID_DICT.items() if k != "answer"}
        with pytest.raises(ValidationError):
            RepairEntry.model_validate(data)

    def test_missing_safety_info_rejected(self):
        data = {k: v for k, v in VALID_DICT.items() if k != "safety_info"}
        with pytest.raises(ValidationError):
            RepairEntry.model_validate(data)

    def test_empty_steps_rejected(self):
        with pytest.raises(ValidationError):
            RepairEntry.model_validate({**VALID_DICT, "steps": []})

    def test_two_steps_rejected(self):
        with pytest.raises(ValidationError):
            RepairEntry.model_validate({**VALID_DICT, "steps": ["s1", "s2"]})

    def test_three_steps_accepted(self):
        entry = RepairEntry.model_validate({**VALID_DICT, "steps": ["s1", "s2", "s3"]})
        assert len(entry.steps) == 3

    def test_empty_tools_rejected(self):
        with pytest.raises(ValidationError):
            RepairEntry.model_validate({**VALID_DICT, "tools_required": []})

    def test_empty_tips_rejected(self):
        with pytest.raises(ValidationError):
            RepairEntry.model_validate({**VALID_DICT, "tips": []})

    def test_to_dict_round_trips_fields(self):
        entry = RepairEntry.model_validate(VALID_DICT)
        d = entry.to_dict()
        assert d["question"] == VALID_DICT["question"]
        assert d["steps"] == VALID_DICT["steps"]
        assert d["category"] == VALID_DICT["category"]

    def test_to_dict_includes_all_fields(self):
        entry = RepairEntry.model_validate(VALID_DICT)
        d = entry.to_dict()
        for key in VALID_DICT:
            assert key in d


# ── DataGenerator.find_duplicates_exact ───────────────────────────────────────

class TestFindDuplicatesExact:
    def test_no_duplicates_returns_empty(self):
        gen = _gen()
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "Why is my outlet not working?"},
        ]
        assert gen.find_duplicates_exact(items) == []

    def test_exact_duplicate_detected(self):
        gen = _gen()
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "How do I fix a dripping faucet?"},
        ]
        dupes = gen.find_duplicates_exact(items)
        assert dupes == [(0, 1)]

    def test_normalizes_to_lowercase(self):
        gen = _gen()
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "HOW DO I FIX A DRIPPING FAUCET?"},
        ]
        assert gen.find_duplicates_exact(items) == [(0, 1)]

    def test_normalizes_punctuation(self):
        gen = _gen()
        items = [
            {"question": "How do I fix a dripping faucet?"},
            {"question": "How do I fix a dripping faucet"},
        ]
        assert gen.find_duplicates_exact(items) == [(0, 1)]

    def test_first_occurrence_wins(self):
        gen = _gen()
        items = [
            {"question": "Q1"},
            {"question": "Q2"},
            {"question": "Q1"},
        ]
        dupes = gen.find_duplicates_exact(items)
        assert (0, 2) in dupes

    def test_multiple_duplicate_pairs(self):
        gen = _gen()
        items = [
            {"question": "Q1"},
            {"question": "Q2"},
            {"question": "Q1"},
            {"question": "Q2"},
        ]
        dupes = gen.find_duplicates_exact(items)
        assert len(dupes) == 2
        assert (0, 2) in dupes
        assert (1, 3) in dupes

    def test_empty_list_returns_empty(self):
        assert _gen().find_duplicates_exact([]) == []

    def test_single_item_returns_empty(self):
        assert _gen().find_duplicates_exact([{"question": "Q"}]) == []

    def test_accepts_repair_entry_objects(self):
        entry = RepairEntry.model_validate(VALID_DICT)
        gen = _gen()
        dupes = gen.find_duplicates_exact([entry, entry])
        assert len(dupes) == 1


# ── DataGenerator.distribution ────────────────────────────────────────────────

class TestDistribution:
    def test_counts_per_category(self):
        gen = _gen()
        items = [
            {"category": "plumbing"},
            {"category": "plumbing"},
            {"category": "electrical"},
        ]
        dist = gen.distribution(items)
        assert dist["plumbing"] == 2
        assert dist["electrical"] == 1
        assert dist.total == 3

    def test_total_matches_item_count(self):
        gen = _gen()
        items = [{"category": "HVAC"}] * 7
        assert gen.distribution(items).total == 7

    def test_empty_list(self):
        gen = _gen()
        dist = gen.distribution([])
        assert dist.total == 0

    def test_iterable(self):
        d = DistributionResult({"plumbing": 3, "electrical": 2}, total=5)
        assert set(d) == {"plumbing", "electrical"}

    def test_repr_contains_keys(self):
        d = DistributionResult({"plumbing": 3}, total=3)
        assert "plumbing" in repr(d)

    def test_accepts_repair_entry_objects(self):
        gen = _gen()
        entry = RepairEntry.model_validate(VALID_DICT)
        dist = gen.distribution([entry, entry])
        assert dist["plumbing"] == 2


# ── CATEGORIES constant ────────────────────────────────────────────────────────

class TestCategories:
    def test_five_categories(self):
        assert len(CATEGORIES) == 5

    @pytest.mark.parametrize("cat", [
        "plumbing", "electrical", "HVAC", "appliances", "general home repair",
    ])
    def test_expected_category_present(self, cat):
        assert cat in CATEGORIES
