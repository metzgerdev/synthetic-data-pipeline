"""Tests for judge.py — DimJudgment, parser, and all prompt builders."""
import json
import pytest
from judge import (
    DIM_KEYS,
    DimJudgment,
    DimJudgmentParser,
    Sample,
    BaselinePromptBuilder,
    ImprovedPromptBuilder,
    SoftenPromptBuilder,
    PermissivePromptBuilder,
    DynamicPromptBuilder,
)

ALL_BUILDERS = [
    BaselinePromptBuilder,
    ImprovedPromptBuilder,
    SoftenPromptBuilder,
    PermissivePromptBuilder,
]

SAMPLE = Sample(
    user_query="My kitchen faucet is dripping — how do I fix it?",
    bot_response=(
        "Before you start: worn washers cause most drips.\n"
        "1. Turn off the water.\n2. Replace washer.\n3. Reassemble."
    ),
    issue_type="plumbing",
)

FULL_JSON = {
    "D1_answer_completeness": True,
    "D2_safety_specificity": True,
    "D3_tool_realism": True,
    "D4_scope_appropriateness": True,
    "D5_context_clarity": False,
    "D6_tip_usefulness": True,
    "overall": "NOT_RESOLVED",
    "reasoning": "D5 context clarity failed.",
}


# ── DimJudgment ────────────────────────────────────────────────────────────────

def _judgment(**overrides) -> DimJudgment:
    defaults = {k: True for k in DIM_KEYS}
    defaults.update(overrides)
    return DimJudgment(**defaults, overall="RESOLVED")


class TestDimJudgment:
    def test_dims_property_returns_all_keys(self):
        j = _judgment()
        assert set(j.dims.keys()) == set(DIM_KEYS)

    def test_dims_property_reflects_values(self):
        j = _judgment(D1_answer_completeness=False)
        assert j.dims["D1_answer_completeness"] is False
        assert j.dims["D2_safety_specificity"] is True

    def test_pass_count_all_pass(self):
        assert _judgment().pass_count == 6

    def test_pass_count_none_pass(self):
        j = _judgment(**{k: False for k in DIM_KEYS})
        assert j.pass_count == 0

    def test_pass_count_partial(self):
        j = _judgment(D1_answer_completeness=False, D3_tool_realism=False)
        assert j.pass_count == 4

    def test_to_dict_contains_all_dim_keys(self):
        d = _judgment().to_dict()
        for key in DIM_KEYS:
            assert key in d

    def test_to_dict_contains_overall_and_reasoning(self):
        d = _judgment().to_dict()
        assert "overall" in d
        assert "reasoning" in d

    def test_to_dict_does_not_contain_raw(self):
        d = _judgment().to_dict()
        assert "raw" not in d

    def test_resolved_overall(self):
        j = DimJudgment(**{k: True for k in DIM_KEYS}, overall="RESOLVED")
        assert j.overall == "RESOLVED"

    def test_not_resolved_overall(self):
        j = DimJudgment(**{k: False for k in DIM_KEYS}, overall="NOT_RESOLVED")
        assert j.overall == "NOT_RESOLVED"


# ── DIM_KEYS ──────────────────────────────────────────────────────────────────

class TestDimKeys:
    def test_exactly_six(self):
        assert len(DIM_KEYS) == 6

    def test_all_d_prefixed(self):
        for key in DIM_KEYS:
            assert key.startswith("D"), f"{key} does not start with 'D'"

    def test_expected_names(self):
        expected = {
            "D1_answer_completeness",
            "D2_safety_specificity",
            "D3_tool_realism",
            "D4_scope_appropriateness",
            "D5_context_clarity",
            "D6_tip_usefulness",
        }
        assert set(DIM_KEYS) == expected


# ── DimJudgmentParser ──────────────────────────────────────────────────────────

class TestDimJudgmentParser:
    def setup_method(self):
        self.parser = DimJudgmentParser()

    def test_parses_valid_json(self):
        j = self.parser.parse(json.dumps(FULL_JSON))
        assert j.overall == "NOT_RESOLVED"
        assert j.D5_context_clarity is False
        assert j.D1_answer_completeness is True

    def test_parses_json_embedded_in_text(self):
        raw = f"Here is my evaluation:\n{json.dumps(FULL_JSON)}\nThat is all."
        j = self.parser.parse(raw)
        assert j.overall == "NOT_RESOLVED"

    def test_overall_resolved_case_insensitive(self):
        data = {**FULL_JSON, "overall": "resolved"}
        j = self.parser.parse(json.dumps(data))
        assert j.overall == "RESOLVED"

    def test_overall_not_resolved_case_insensitive(self):
        data = {**FULL_JSON, "overall": "NOT_resolved"}
        j = self.parser.parse(json.dumps(data))
        assert j.overall == "NOT_RESOLVED"

    def test_missing_dims_default_to_false(self):
        minimal = json.dumps({"overall": "NOT_RESOLVED", "reasoning": "test"})
        j = self.parser.parse(minimal)
        for key in DIM_KEYS:
            assert getattr(j, key) is False

    def test_reasoning_extracted(self):
        j = self.parser.parse(json.dumps(FULL_JSON))
        assert j.reasoning == "D5 context clarity failed."

    def test_raw_stored_verbatim(self):
        raw = json.dumps(FULL_JSON)
        j = self.parser.parse(raw)
        assert j.raw == raw

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="No JSON found"):
            self.parser.parse("This response has no JSON at all.")

    def test_malformed_json_raises(self):
        with pytest.raises(Exception):
            self.parser.parse("{not valid json}")

    def test_all_true_parsed_correctly(self):
        data = {k: True for k in DIM_KEYS}
        data["overall"] = "RESOLVED"
        data["reasoning"] = "All passed."
        j = self.parser.parse(json.dumps(data))
        assert j.pass_count == 6

    def test_all_false_parsed_correctly(self):
        data = {k: False for k in DIM_KEYS}
        data["overall"] = "NOT_RESOLVED"
        data["reasoning"] = "All failed."
        j = self.parser.parse(json.dumps(data))
        assert j.pass_count == 0


# ── Prompt builders ────────────────────────────────────────────────────────────

class TestPromptBuilders:
    @pytest.mark.parametrize("BuilderClass", ALL_BUILDERS)
    def test_contains_user_query(self, BuilderClass):
        prompt = BuilderClass().build(SAMPLE)
        assert SAMPLE.user_query in prompt

    @pytest.mark.parametrize("BuilderClass", ALL_BUILDERS)
    def test_contains_bot_response(self, BuilderClass):
        prompt = BuilderClass().build(SAMPLE)
        assert SAMPLE.bot_response in prompt

    @pytest.mark.parametrize("BuilderClass", ALL_BUILDERS)
    def test_mentions_json(self, BuilderClass):
        prompt = BuilderClass().build(SAMPLE)
        assert "JSON" in prompt or "json" in prompt

    @pytest.mark.parametrize("BuilderClass", ALL_BUILDERS)
    def test_all_six_dim_keys_present(self, BuilderClass):
        prompt = BuilderClass().build(SAMPLE)
        for key in DIM_KEYS:
            assert key in prompt, f"{key} missing from {BuilderClass.__name__}"

    @pytest.mark.parametrize("BuilderClass", ALL_BUILDERS)
    def test_mentions_resolved(self, BuilderClass):
        prompt = BuilderClass().build(SAMPLE)
        assert "RESOLVED" in prompt

    @pytest.mark.parametrize("BuilderClass", ALL_BUILDERS)
    def test_has_template_attribute(self, BuilderClass):
        assert hasattr(BuilderClass, "TEMPLATE")
        assert len(BuilderClass.TEMPLATE) > 100

    def test_four_builders_produce_distinct_prompts(self):
        prompts = [cls().build(SAMPLE) for cls in ALL_BUILDERS]
        assert len(set(prompts)) == 4, "All builders should produce unique prompts"

    def test_soften_includes_calibration_example(self):
        prompt = SoftenPromptBuilder().build(SAMPLE)
        assert "RESOLVED EXAMPLE" in prompt

    def test_permissive_includes_two_examples(self):
        prompt = PermissivePromptBuilder().build(SAMPLE)
        assert "EXAMPLE 1" in prompt
        assert "EXAMPLE 2" in prompt


# ── DynamicPromptBuilder ──────────────────────────────────────────────────────

class TestDynamicPromptBuilder:
    def test_uses_provided_template(self):
        template = "Query: {user_query}\nAnswer: {bot_response}\nJSON please."
        builder = DynamicPromptBuilder(template)
        prompt = builder.build(SAMPLE)
        assert SAMPLE.user_query in prompt
        assert SAMPLE.bot_response in prompt

    def test_stores_template(self):
        template = "test {user_query} {bot_response}"
        builder = DynamicPromptBuilder(template)
        assert builder.TEMPLATE == template


# ── Sample ────────────────────────────────────────────────────────────────────

class TestSample:
    def test_defaults(self):
        s = Sample(user_query="Q", bot_response="A")
        assert s.issue_type == ""

    def test_all_fields(self):
        s = Sample(user_query="Q", bot_response="A", issue_type="plumbing")
        assert s.issue_type == "plumbing"
