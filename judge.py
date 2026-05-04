"""
LLM-as-Judge: Reusable evaluation framework for customer support chatbots.
Facade pattern with Dependency Injection for swappable components.
"""

from __future__ import annotations
import functools
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, runtime_checkable


from generate import (  # noqa: F401  — re-exported for backwards compat
    CATEGORIES,
    DataGenerator,
    DistributionResult,
    RepairEntry,
)


# ─── Logging ─────────────────────────────────────────────────────────────────

def _setup_logger(path: str = "llm_judge.log") -> logging.Logger:
    logger = logging.getLogger("llm_judge")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    logger.addHandler(handler)
    return logger

logger = _setup_logger()


def trace_step(name: str):
    """Decorator that records a Span on the active Trace and writes a JSON log line."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, trace: Trace | None = None, **kwargs):
            t0 = time.perf_counter()
            try:
                result  = fn(*args, **kwargs)
                latency = time.perf_counter() - t0
                _log_and_record(trace, name, args, kwargs, str(result)[:300], latency, "ok")
                return result
            except Exception as exc:
                latency = time.perf_counter() - t0
                _log_and_record(trace, name, args, kwargs, str(exc), latency, "error")
                raise
        return wrapper
    return decorator


def _log_and_record(
    trace: "Trace | None",
    name: str,
    args: tuple,
    kwargs: dict,
    output: str,
    latency: float,
    status: str,
) -> None:
    entry = {
        "step":       name,
        "status":     status,
        "latency_ms": round(latency * 1000, 2),
        "input":      str(args[1:])[:200],   # skip self
        "output":     output[:200],
    }
    log_fn = logger.error if status == "error" else logger.info
    log_fn(json.dumps(entry))

    if trace is not None:
        trace.record(
            name    = name,
            input   = entry["input"],
            output  = output,
            latency = latency,
            status  = status,
        )


# ─── Domain types ────────────────────────────────────────────────────────────

@dataclass
class Sample:
    user_query:   str
    bot_response: str
    issue_type:   str = ""


DIM_KEYS = [
    "D1_answer_completeness",
    "D2_safety_specificity",
    "D3_tool_realism",
    "D4_scope_appropriateness",
    "D5_context_clarity",
    "D6_tip_usefulness",
]

@dataclass
class DimJudgment:
    D1_answer_completeness:   bool
    D2_safety_specificity:    bool
    D3_tool_realism:          bool
    D4_scope_appropriateness: bool
    D5_context_clarity:       bool
    D6_tip_usefulness:        bool
    overall:   Literal["RESOLVED", "NOT_RESOLVED"]
    reasoning: str = ""
    raw:       str = ""

    @property
    def dims(self) -> dict[str, bool]:
        return {k: getattr(self, k) for k in DIM_KEYS}

    @property
    def pass_count(self) -> int:
        return sum(self.dims.values())

    def to_dict(self) -> dict:
        return {**self.dims, "overall": self.overall, "reasoning": self.reasoning}


# ─── Tracing ─────────────────────────────────────────────────────────────────

@dataclass
class Span:
    name:    str
    input:   str
    output:  str
    latency: float          # seconds
    meta:    dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.name}] ({self.latency*1000:.0f}ms)\n"
            f"  IN:  {self.input[:120]}\n"
            f"  OUT: {self.output[:120]}"
            + (f"\n  META: {self.meta}" if self.meta else "")
        )


class Trace:
    def __init__(self):
        self.spans: list[Span] = []

    def record(self, name: str, input: str, output: str, latency: float, **meta) -> Span:
        span = Span(name=name, input=input, output=output, latency=latency, meta=meta)
        self.spans.append(span)
        return span

    def summary(self) -> None:
        total = sum(s.latency for s in self.spans)
        print(f"── Trace ({len(self.spans)} spans, {total*1000:.0f}ms total) ──")
        for i, span in enumerate(self.spans, 1):
            print(f"  [{i}] {span}")

    def to_dict(self) -> list[dict]:
        return [
            {"name": s.name, "input": s.input, "output": s.output,
             "latency_ms": round(s.latency * 1000, 2), "meta": s.meta}
            for s in self.spans
        ]


# ─── Injectable interfaces ────────────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, temperature: float, max_tokens: int) -> str: ...


@runtime_checkable
class PromptBuilder(Protocol):
    def build(self, sample: Sample) -> str: ...


# ─── Default implementations ──────────────────────────────────────────────────

class GroqClient:
    MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str | None = None):
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def complete(
        self,
        prompt:      str,
        temperature: float = 0.1,
        max_tokens:  int   = 300,
        trace:       Trace | None = None,
    ) -> str:
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - t0
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty response")
        if trace is not None:
            trace.record(
                name    = "llm_call",
                input   = prompt,
                output  = content,
                latency = latency,
                model   = self.MODEL,
                temperature = temperature,
                tokens  = response.usage.total_tokens if response.usage else None,
            )
        return content

class BaselinePromptBuilder:
    """Domain-specific prompt grounded in the 6 DIY repair quality dimensions."""

    TEMPLATE = """\
You are a quality reviewer for a DIY home repair Q&A dataset.
Evaluate the answer below on exactly 6 dimensions. Return ONLY valid JSON.
Question: {user_query}
Answer: {bot_response}

Score each dimension true (pass) or false (fail):

D1_answer_completeness   — Contains tools, concrete steps, safety info, and a useful tip. Answers that stop short or omit key stages FAIL.
D2_safety_specificity    — Names the SPECIFIC hazard and precaution for THIS repair. Generic phrases ("be careful", "use caution", "stay safe") FAIL.
D3_tool_realism          — Every tool is something a typical homeowner owns or can buy at a hardware store for under $50. Professional or trade-only tools FAIL.
D4_scope_appropriateness — Within realistic DIY capability. If professional help is genuinely needed (gas, panel work, structural), the answer says so clearly.
D5_context_clarity       — Question and answer contain enough context to understand the problem; answer directly addresses it.
D6_tip_usefulness        — Tips are non-obvious and task-specific. Tips that restate a step or offer generic encouragement FAIL.

Derive overall: "RESOLVED" if ALL 6 dimensions pass, otherwise "NOT_RESOLVED".

Respond in exactly this JSON format:
{{
  "D1_answer_completeness": boolean,
  "D2_safety_specificity": boolean,
  "D3_tool_realism": boolean,
  "D4_scope_appropriateness": boolean,
  "D5_context_clarity": boolean,
  "D6_tip_usefulness": boolean,
  "overall": "NOT_RESOLVED | RESOLVED",
  "reasoning": "one sentence explaining the overall verdict"
}}"""

    def build(self, sample: Sample) -> str:
        return self.TEMPLATE.format(
            user_query=sample.user_query,
            bot_response=sample.bot_response,
        )

class ImprovedPromptBuilder:
    """
    Each criterion is tightened to the specific failure modes seen in the
    human-labelled dataset:
      D1 — answers that stop at diagnosis without a fix, or skip basic checks
      D2 — safety that describes hazards from a different repair type (mismatched, not just generic)
      D3 — tools listed that are not actually used in the steps, or are trade-only
      D5 — answer doesn't address the most likely cause given the symptom
    """

    TEMPLATE = """\
You are a quality reviewer for a DIY home repair Q&A dataset.
Evaluate the answer below on exactly 6 dimensions. Return ONLY valid JSON — no explanation, no markdown.

Question: {user_query}
Answer: {bot_response}

Score each dimension true (pass) or false (fail):

D1_answer_completeness   — The steps must form a complete repair path from diagnosis to resolution. FAIL if the steps stop at a diagnostic measurement or check without providing a fix, or if the answer dives into advanced components (compressor, control board) before covering simpler checks first.
D2_safety_specificity    — The safety warning must address a hazard that can actually occur in THIS specific repair. FAIL if the warning describes hazards from a different repair type (e.g. gas leak warnings on a toilet repair, electrical shock on a job with no electrical components, underground pipe warnings on a faucet fix). PASS even with conventional phrasing, as long as the hazard is genuinely relevant to this repair.
D3_tool_realism          — Every tool listed must be (a) actually used in the steps described, and (b) something a typical homeowner can buy at a hardware store for under $50. FAIL if any tool is irrelevant to the steps, is a specialist/trade-only tool, or costs more than $50.
D4_scope_appropriateness — Within realistic DIY capability. If professional help is genuinely needed (gas, panel work, structural), the answer says so clearly.
D5_context_clarity       — The answer must address the most likely cause of the specific symptom in the question. FAIL if the answer skips obvious simple causes and jumps straight to advanced internal components, or if the steps would not help with the described symptom.
D6_tip_usefulness        — Tips are non-obvious and task-specific. Tips that only say "call a professional if it gets complicated" with no additional insight FAIL. Tips that restate a step or offer generic encouragement FAIL.

Derive overall: "RESOLVED" if ALL 6 dimensions pass, otherwise "NOT_RESOLVED".

Respond in exactly this JSON format:
{{
  "D1_answer_completeness": boolean,
  "D2_safety_specificity": boolean,
  "D3_tool_realism": boolean,
  "D4_scope_appropriateness": boolean,
  "D5_context_clarity": boolean,
  "D6_tip_usefulness": boolean,
  "overall": "RESOLVED | NOT_RESOLVED",
  "reasoning": "one sentence explaining the overall verdict"
}}"""

    def build(self, sample: Sample) -> str:
        return self.TEMPLATE.format(
            user_query=sample.user_query,
            bot_response=sample.bot_response,
        )


class SoftenPromptBuilder:
    """
    Generous scoring — "not like a lawyer."
    Default to PASS; only fail on clear, obvious problems.
    One calibration example anchors all rubric items.
    """

    TEMPLATE = """\
You are a quality reviewer for a DIY home repair Q&A dataset.
Your job is to score answers the same way a knowledgeable homeowner would — generously, not like a lawyer.
Default to PASS. Only fail a dimension when there is a clear, obvious problem.

Score each of the 6 dimensions true (pass) or false (fail), then derive overall.

--- RESOLVED EXAMPLE ---

Question: My front-loading washing machine is making a loud banging noise during the spin cycle — how do I fix it?

Answer: Before you start, here are some useful things to know: An unbalanced washing machine or a loose item in the drum causes the vast majority of loud banging noises during the spin cycle.

Now, gather screwdriver set, pliers, level tool, and a socket wrench and follow these steps:

1. Check the washing machine's balance by verifying it's level on the floor — an unbalanced machine will vibrate and cause loud noises
2. Inspect the drum for any loose items that might be causing the noise, such as coins or a broken item
3. Check the suspension system — many modern washers have a self-adjusting suspension, but some may require manual adjustment
4. Verify the washer is properly installed and has enough space around it for airflow and vibration
5. Check the belt or direct drive system for wear or misalignment, as this can cause loud noises
6. If the issue persists, consider calling a professional to diagnose and repair any internal mechanical problems

Safety note: Always unplug the washing machine before performing any maintenance or repairs to avoid electric shock. Be cautious when working with sharp objects and heavy machinery, such as the washing machine's moving parts.

{{
  "D1_answer_completeness": true,
  "D2_safety_specificity": true,
  "D3_tool_realism": true,
  "D4_scope_appropriateness": true,
  "D5_context_clarity": true,
  "D6_tip_usefulness": true,
  "overall": "RESOLVED",
  "reasoning": "Answer starts with the two most likely simple causes, escalates through more complex checks, and correctly defers unresolved issues to a professional."
}}

--- END EXAMPLE ---

Scoring rubric (use the example above as your calibration anchor):

D1_answer_completeness   — Does the answer give the user actionable steps from simple to more involved? PASS if it starts with easy checks and escalates. Ending with "call a professional if it persists" counts as a complete path. Only FAIL if every single step is a dead-end diagnostic with no action to take.
D2_safety_specificity    — Is the safety warning relevant to this type of repair? PASS if the hazard mentioned (electric shock, water damage, sharp objects, etc.) could actually occur in this repair. Only FAIL if the warning is about a completely different repair type (e.g. gas warnings for a toilet fix).
D3_tool_realism          — Are the tools basic hardware store items a homeowner would own? PASS for common tools like screwdrivers, wrenches, pliers, testers, tape. Only FAIL if a professional trade tool is listed (oscilloscope, manifold gauge set, pipe threading machine).
D4_scope_appropriateness — Is this repair within DIY reach? PASS if the answer attempts the repair or correctly tells the user when to call a pro. Only FAIL if the answer encourages dangerous DIY work (gas lines, main electrical panel) without any safety caveat.
D5_context_clarity       — Does the answer address the symptom in the question? PASS if the preamble or any step names the likely cause of the stated symptom. Only FAIL if the answer is completely off-topic for the stated problem.
D6_tip_usefulness        — Does the "Before you start" preamble add value? PASS if it names a common root cause, gives a shortcut, or sets expectations. Only FAIL if the preamble is missing entirely or says nothing beyond "this is a common problem."

Derive overall: "RESOLVED" if ALL 6 dimensions pass, otherwise "NOT_RESOLVED".

Now evaluate:

Question: {user_query}
Answer: {bot_response}

Respond in exactly this JSON format:
{{
  "D1_answer_completeness": boolean,
  "D2_safety_specificity": boolean,
  "D3_tool_realism": boolean,
  "D4_scope_appropriateness": boolean,
  "D5_context_clarity": boolean,
  "D6_tip_usefulness": boolean,
  "overall": "RESOLVED | NOT_RESOLVED",
  "reasoning": "one sentence explaining the overall verdict"
}}"""

    def build(self, sample: Sample) -> str:
        return self.TEMPLATE.format(
            user_query=sample.user_query,
            bot_response=sample.bot_response,
        )


class PermissivePromptBuilder:
    """
    Loosest scoring — RESOLVED = "gives the homeowner something useful to try."
    Two calibration examples; FAIL only for truly unhelpful/unsafe/off-topic answers.
    """

    TEMPLATE = """\
You are evaluating answers from a DIY home repair chatbot whose purpose is to give homeowners a helpful starting point — not a complete repair manual.

A real repair guide requires photos, measurements, and hands-on diagnosis that a chatbot cannot provide. RESOLVED simply means: "this answer gives the homeowner something useful to try." NOT_RESOLVED means: "this answer is unhelpful, unsafe, or completely off-topic."

Below are two examples of RESOLVED answers. Use them as your calibration standard.

=== RESOLVED EXAMPLE 1 ===

Question: My front-loading washing machine is making a loud banging noise during the spin cycle — how do I fix it?

Answer: Before you start, here are some useful things to know: An unbalanced washing machine or a loose item in the drum causes the vast majority of loud banging noises during the spin cycle.

Now, gather screwdriver set, pliers, level tool, and a socket wrench and follow these steps:

1. Check the washing machine's balance by verifying it's level on the floor — an unbalanced machine will vibrate and cause loud noises
2. Inspect the drum for any loose items that might be causing the noise, such as coins or a broken item
3. Check the suspension system — many modern washers have a self-adjusting suspension, but some may require manual adjustment
4. Verify the washer is properly installed and has enough space around it for airflow and vibration
5. Check the belt or direct drive system for wear or misalignment, as this can cause loud noises
6. If the issue persists, consider calling a professional to diagnose and repair any internal mechanical problems

Safety note: Always unplug the washing machine before performing any maintenance or repairs to avoid electric shock.

{{
  "D1_answer_completeness": true,
  "D2_safety_specificity": true,
  "D3_tool_realism": true,
  "D4_scope_appropriateness": true,
  "D5_context_clarity": true,
  "D6_tip_usefulness": true,
  "overall": "RESOLVED",
  "reasoning": "Gives the homeowner clear first steps to try; escalates appropriately to a professional if self-repair fails."
}}

=== RESOLVED EXAMPLE 2 ===

Question: My refrigerator is making a weird humming noise and the freezer isn't keeping things frozen — how do I fix it?

Answer: Before you start, here are some useful things to know: A faulty or blocked defrost drain is a common cause of freezer temperature issues and unusual noises.

Now, gather a wrench set, a screwdriver, a drain cleaning brush, and follow these steps:

1. Unplug the refrigerator to prevent any electrical shock or damage
2. Check the drain at the bottom of the freezer compartment — make sure it is not blocked with ice or debris
3. If the drain is blocked, use a drain cleaning brush to clear the blockage
4. Next, inspect the defrost drain hose for any kinks or blockages and repair or replace as needed
5. Check the refrigerator's temperature settings to ensure they are set correctly
6. If issues persist, check the compressor and fan for any signs of malfunction or blockage

Safety note: Always unplug the refrigerator before attempting any repairs to avoid risk of electric shock.

{{
  "D1_answer_completeness": true,
  "D2_safety_specificity": true,
  "D3_tool_realism": true,
  "D4_scope_appropriateness": true,
  "D5_context_clarity": true,
  "D6_tip_usefulness": true,
  "overall": "RESOLVED",
  "reasoning": "Points the homeowner at the most likely cause and gives them actionable steps to try before calling a professional."
}}

=== SCORING RULES ===

Remember: you are not grading a repair manual. You are checking whether this response gives a homeowner a useful starting point.

D1_answer_completeness   — PASS if the answer provides at least a few steps the homeowner can actually attempt. It does not need to cover every possible outcome — pointing the user toward the most likely cause and telling them to call a professional if it does not resolve is perfectly acceptable. FAIL only if the answer provides no actionable steps at all.
D2_safety_specificity    — PASS if the safety note mentions any hazard that is plausible for this type of repair. FAIL only if the safety note describes a hazard that is impossible for this repair (e.g. gas leak warning on a toilet fix, structural warning on changing a lightbulb).
D3_tool_realism          — PASS for any tool available at a hardware store. FAIL only for specialist trade equipment a homeowner would never own (oscilloscope, manifold gauge set, pipe threading machine).
D4_scope_appropriateness — PASS if the answer helps the homeowner take a first step or correctly advises them to call a professional. FAIL only if it encourages genuinely dangerous unlicensed work (opening a gas line, rewiring an electrical panel) without any safety warning.
D5_context_clarity       — PASS if the answer is about the same type of problem described in the question. FAIL only if the answer is about a completely different appliance or repair type.
D6_tip_usefulness        — PASS if the "Before you start" preamble contains any information specific to this repair — a likely cause, what to check first, or what to expect. FAIL only if the preamble is absent or contains nothing specific to this repair at all.

Derive overall: "RESOLVED" if ALL 6 pass, otherwise "NOT_RESOLVED".

Now evaluate:

Question: {user_query}
Answer: {bot_response}

Respond in exactly this JSON format:
{{
  "D1_answer_completeness": boolean,
  "D2_safety_specificity": boolean,
  "D3_tool_realism": boolean,
  "D4_scope_appropriateness": boolean,
  "D5_context_clarity": boolean,
  "D6_tip_usefulness": boolean,
  "overall": "RESOLVED | NOT_RESOLVED",
  "reasoning": "one sentence explaining the overall verdict"
}}"""

    def build(self, sample: Sample) -> str:
        return self.TEMPLATE.format(
            user_query=sample.user_query,
            bot_response=sample.bot_response,
        )


class DynamicPromptBuilder:
    """Prompt builder that uses a template string loaded at runtime (e.g. from the DB)."""

    def __init__(self, template: str) -> None:
        self.TEMPLATE = template

    def build(self, sample: Sample) -> str:
        return self.TEMPLATE.format(
            user_query=sample.user_query,
            bot_response=sample.bot_response,
        )


class DimJudgmentParser:

    def parse(self, raw: str) -> DimJudgment:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")
        data = json.loads(match.group())
        return DimJudgment(
            D1_answer_completeness   = bool(data.get("D1_answer_completeness", False)),
            D2_safety_specificity    = bool(data.get("D2_safety_specificity", False)),
            D3_tool_realism          = bool(data.get("D3_tool_realism", False)),
            D4_scope_appropriateness = bool(data.get("D4_scope_appropriateness", False)),
            D5_context_clarity       = bool(data.get("D5_context_clarity", False)),
            D6_tip_usefulness        = bool(data.get("D6_tip_usefulness", False)),
            overall   = "RESOLVED" if data.get("overall", "").upper() == "RESOLVED" else "NOT_RESOLVED",
            reasoning = data.get("reasoning", ""),
            raw       = raw,
        )


class LLMJudge:
    """Facade for automated chatbot evaluation. Dependencies are injected."""

    def __init__(
        self,
        client:      LLMClient,
        prompt:      PromptBuilder,
        parser:      DimJudgmentParser,
        temperature: float = 0.1,
        max_tokens:  int   = 300,
    ):
        self._client      = client
        self._prompt      = prompt
        self._parser      = parser
        self._temperature = temperature
        self._max_tokens  = max_tokens

    # ── factory helper ───────────────────────────────────────────────────────

    @classmethod
    def create(cls, prompt: PromptBuilder, api_key: str | None = None) -> "LLMJudge":
        """Create a judge with the given prompt builder."""
        return cls(
            client=GroqClient(api_key),
            prompt=prompt,
            parser=DimJudgmentParser(),
            max_tokens=400,
        )

    # ── pipeline steps ───────────────────────────────────────────────────────

    @trace_step("prompt_build")
    def _build_prompt(self, sample: Sample, **_) -> str:
        return self._prompt.build(sample)

    @trace_step("llm_call")
    def _call_llm(self, prompt: str, **_) -> str:
        return self._client.complete(prompt, self._temperature, self._max_tokens)

    @trace_step("parse_response")
    def _parse_response(self, raw: str, **_) -> DimJudgment:
        return self._parser.parse(raw)

    # ── public API ───────────────────────────────────────────────────────────

    def evaluate_dims(self, sample: Sample, trace: Trace | None = None) -> DimJudgment:
        """Score a single sample across all 6 dimensions."""
        prompt = self._build_prompt(sample, trace=trace)
        raw    = self._call_llm(prompt,    trace=trace)
        return self._parse_response(raw,   trace=trace)

    def evaluate_dims_dataset(
        self,
        dataset:     list[Sample],
        on_progress: Callable[[int, int, DimJudgment], None] | None = None,
        trace:       Trace | None = None,
    ) -> list[DimJudgment]:
        judgments = []
        for i, sample in enumerate(dataset, 1):
            j = self.evaluate_dims(sample, trace=trace)
            judgments.append(j)
            if on_progress:
                on_progress(i, len(dataset), j)
        return judgments



