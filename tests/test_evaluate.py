"""Tests for evaluate.py — _dim_pass_rates and _agreement_rates helpers."""
from evaluate import _dim_pass_rates, _agreement_rates
from judge import DIM_KEYS


def _llm_row(*vals: bool) -> dict:
    """Build a row dict with llm_{dim} keys from a sequence of 6 booleans."""
    assert len(vals) == 6
    return {f"llm_{k}": v for k, v in zip(DIM_KEYS, vals)}


def _agreement_row(human_vals: list[bool], llm_vals: list[bool]) -> dict:
    """Build a row dict with both human_{dim} and llm_{dim} keys."""
    assert len(human_vals) == len(llm_vals) == 6
    row = {}
    for k, h, l in zip(DIM_KEYS, human_vals, llm_vals):
        row[f"human_{k}"] = h
        row[f"llm_{k}"] = l
    return row


# ── _dim_pass_rates ────────────────────────────────────────────────────────────

class TestDimPassRates:
    def test_all_pass_returns_one(self):
        rows = [_llm_row(*([True] * 6))] * 4
        rates = _dim_pass_rates(rows, "llm", DIM_KEYS)
        assert all(v == 1.0 for v in rates.values())

    def test_all_fail_returns_zero(self):
        rows = [_llm_row(*([False] * 6))] * 4
        rates = _dim_pass_rates(rows, "llm", DIM_KEYS)
        assert all(v == 0.0 for v in rates.values())

    def test_half_pass_returns_half(self):
        rows = [
            _llm_row(*([True] * 6)),
            _llm_row(*([False] * 6)),
        ]
        rates = _dim_pass_rates(rows, "llm", DIM_KEYS)
        assert all(v == 0.5 for v in rates.values())

    def test_empty_rows_returns_zero(self):
        rates = _dim_pass_rates([], "llm", DIM_KEYS)
        assert all(v == 0.0 for v in rates.values())

    def test_returns_all_dim_keys(self):
        rows = [_llm_row(*([True] * 6))]
        rates = _dim_pass_rates(rows, "llm", DIM_KEYS)
        assert set(rates.keys()) == set(DIM_KEYS)

    def test_single_row_single_dim_fail(self):
        row = {f"llm_{k}": True for k in DIM_KEYS}
        row[f"llm_{DIM_KEYS[0]}"] = False
        rates = _dim_pass_rates([row], "llm", DIM_KEYS)
        assert rates[DIM_KEYS[0]] == 0.0
        assert rates[DIM_KEYS[1]] == 1.0

    def test_three_rows_one_fail(self):
        rows = [
            _llm_row(*([True] * 6)),
            _llm_row(*([True] * 6)),
            _llm_row(*([False] * 6)),
        ]
        rates = _dim_pass_rates(rows, "llm", DIM_KEYS)
        assert abs(rates[DIM_KEYS[0]] - 2 / 3) < 1e-9

    def test_human_source_key(self):
        row = {f"human_{k}": True for k in DIM_KEYS}
        row[f"human_{DIM_KEYS[2]}"] = False
        rates = _dim_pass_rates([row], "human", DIM_KEYS)
        assert rates[DIM_KEYS[2]] == 0.0
        assert rates[DIM_KEYS[0]] == 1.0


# ── _agreement_rates ───────────────────────────────────────────────────────────

class TestAgreementRates:
    def test_perfect_agreement_returns_one(self):
        rows = [_agreement_row([True] * 6, [True] * 6)] * 3
        rates = _agreement_rates(rows, DIM_KEYS)
        assert all(v == 1.0 for v in rates.values())

    def test_zero_agreement_returns_zero(self):
        rows = [_agreement_row([True] * 6, [False] * 6)] * 3
        rates = _agreement_rates(rows, DIM_KEYS)
        assert all(v == 0.0 for v in rates.values())

    def test_half_agreement(self):
        rows = [
            _agreement_row([True] * 6, [True] * 6),
            _agreement_row([True] * 6, [False] * 6),
        ]
        rates = _agreement_rates(rows, DIM_KEYS)
        assert all(v == 0.5 for v in rates.values())

    def test_empty_rows_returns_zero(self):
        rates = _agreement_rates([], DIM_KEYS)
        assert all(v == 0.0 for v in rates.values())

    def test_returns_all_dim_keys(self):
        rows = [_agreement_row([True] * 6, [True] * 6)]
        rates = _agreement_rates(rows, DIM_KEYS)
        assert set(rates.keys()) == set(DIM_KEYS)

    def test_both_false_counts_as_agreement(self):
        rows = [_agreement_row([False] * 6, [False] * 6)]
        rates = _agreement_rates(rows, DIM_KEYS)
        assert all(v == 1.0 for v in rates.values())

    def test_per_dim_agreement_independent(self):
        # D1 agrees (both True), D2 disagrees (True vs False), rest match
        human = [True, True, True, True, True, True]
        llm   = [True, False, True, True, True, True]
        rates = _agreement_rates([_agreement_row(human, llm)], DIM_KEYS)
        assert rates[DIM_KEYS[0]] == 1.0   # D1: agree
        assert rates[DIM_KEYS[1]] == 0.0   # D2: disagree
        assert rates[DIM_KEYS[2]] == 1.0   # D3: agree
