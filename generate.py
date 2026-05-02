"""
Synthetic DIY repair Q&A dataset generation.

"""

from __future__ import annotations
import json
import math
import re
import time
from collections import Counter
from typing import Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError("pip install matplotlib") from None


_T = TypeVar("_T")
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)

def _with_backoff(fn: Callable[[], _T], retries: int = 5, base: float = 2.0) -> _T:
    """Call fn(), retrying on rate-limit errors.

    Sleeps for the exact 'try again in Xs' duration returned by the API when
    available, falling back to exponential backoff otherwise.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = any(
                kw in msg.lower()
                for kw in ("rate limit", "rate_limit", "429", "too many requests")
            )
            if not is_rate_limit or attempt == retries - 1:
                raise
            match = _RETRY_AFTER_RE.search(msg)
            wait = float(match.group(1)) + 1.0 if match else base ** attempt
            print(f"  rate limit — waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ─── Constants ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "plumbing", "electrical", "HVAC", "appliances",
    "general home repair",
]

IMPROVED_GENERATION_PROMPT = """\
Generate a DIY home repair Q&A entry for a chatbot training dataset. Output valid JSON only — no explanation, no markdown, no trailing text.

The entry must follow this exact schema:
{{
  "question": "<natural language question a homeowner would ask, e.g. 'My X is doing Y — how do I fix it?'>",
  "equipment_problem": "<concise factual description of the problem, e.g. 'Ceiling light fixture flickering intermittently'>",
  "answer": "<full answer in this exact format: 'Before you start, here are some useful things to know: <most likely cause or key diagnostic insight>\\n\\nNow, gather <comma-separated tool list> and follow these steps:\\n\\n1. <step>\\n2. <step>\\n...'>",
  "category": "<repair category, e.g. plumbing, electrical, HVAC, appliances, general home repair>",
  "tools_required": ["<tool1>", "<tool2>"],
  "steps": [
    "<step 1 — imperative sentence, specific and actionable>",
    "<step 2>"
  ],
  "safety_info": "<one or two sentences. specific to the hazards of THIS repair. Name the exact risk — electric shock, flooding, carbon monoxide, structural load. Never say only 'be careful' or 'use caution'.>",
  "tips": [
    "<the most likely root cause or a non-obvious diagnostic insight — what a seasoned professional would tell you before you start>"
  ]
}}

Rules:
- question must sound like a real homeowner wrote it — informal, first person, describes the symptom
- equipment_problem is a clean 3–8 word factual label for the issue, not a question
- answer preamble must state the most common cause FIRST — this is the key insight that saves time
- steps must be 4–7 items, start simple (check the obvious) before going deeper into the repair
- tools_required must match exactly the tools listed in the answer
- safety_info must name a hazard specific to this repair type — circuit breaker for electrical, shutoff valve for plumbing, refrigerant warning for HVAC
- tips must contain a diagnostic insight a professional would share — not a restatement of a step
- the problem you generate must be meaningfully different from every entry in "Already generated problems" — different symptom, different root cause, or different equipment type within the category

Here is a high-quality example of the output to aim for:
{{
  "id": "qa_01384",
  "category": "electrical",
  "question": "My ceiling light fixture is flickering intermittently. How do I fix this?",
  "equipment_problem": "Ceiling light fixture flickering intermittently",
  "answer": "Before you start, here are some useful things to know: A loose bulb or loose wire nut connection causes the vast majority of flickering light fixtures.\\n\\nNow, gather replacement bulb, voltage tester, stable ladder, wire nuts, and screwdriver set and follow these steps:\\n\\n1. Turn off the light and check that the bulb is screwed in firmly — a loose bulb is the most common cause of flickering\\n2. Replace the bulb with a known working one to rule out a faulty bulb\\n3. If flickering continues, turn off the breaker and verify power is off with a voltage tester\\n4. Set up a stable ladder and remove the fixture mounting screws to lower the fixture canopy\\n5. Check the wire nut connections — disconnect each one, re-twist the bare wire ends together tightly, and apply a new wire nut\\n6. Remount the fixture, restore power, and monitor for continued flickering",
  "tools_required": ["replacement bulb", "voltage tester", "stable ladder", "wire nuts", "screwdriver set"],
  "steps": [
    "Turn off the light and check that the bulb is screwed in firmly — a loose bulb is the most common cause of flickering",
    "Replace the bulb with a known working one to rule out a faulty bulb",
    "If flickering continues, turn off the breaker and verify power is off with a voltage tester",
    "Set up a stable ladder and remove the fixture mounting screws to lower the fixture canopy",
    "Check the wire nut connections — disconnect each one, re-twist the bare wire ends together tightly, and apply a new wire nut",
    "Remount the fixture, restore power, and monitor for continued flickering"
  ],
  "safety_info": "Turn off the circuit breaker before accessing fixture wiring. Use a stable ladder on a flat surface — never stand on the top step.",
  "tips": ["A loose bulb or loose wire nut connection causes the vast majority of flickering light fixtures"]
}}

Already generated problems: {already_generated}

Generate one entry now for category: {category}"""


BASELINE_GENERATION_PROMPT = """\
Generate a DIY home repair Q&A entry for a chatbot training dataset. Output valid JSON only.

The entry must follow this schema:
{{
  "question": "<a question a homeowner might ask>",
  "equipment_problem": "<description of the problem>",
  "answer": "<answer with preamble and steps only — no safety section>",
  "category": "<repair category>",
  "tools_required": ["<tool1>", "<tool2>"],
  "steps": ["<step 1>", "<step 2>"],
  "safety_info": "<safety advice>",
  "tips": ["<a helpful tip>"]
}}

Already generated problems: {already_generated}

Generate one entry now for category: {category}"""


# ─── Schema ───────────────────────────────────────────────────────────────────

class RepairEntry(BaseModel):
    question:          str       = Field(description="Informal homeowner question describing the symptom")
    equipment_problem: str       = Field(description="Concise factual label for the problem, 3–8 words, e.g. 'Ceiling light fixture flickering intermittently'")
    answer:            str       = Field(description="Full answer: preamble with root cause, tools line, numbered steps, safety note")
    category:          str       = Field(description="Repair category, e.g. plumbing, electrical, HVAC, appliances, general home repair")
    tools_required:    list[str] = Field(description="Tools matching the tools line in answer, 2–8 items", min_length=1)
    steps:             list[str] = Field(min_length=3, description="Ordered imperative steps")
    safety_info:       str       = Field(description="Specific hazard for this repair — electric shock, flooding, carbon monoxide, structural load")
    tips:              list[str] = Field(min_length=1, description="Diagnostic insight a professional would share — not a restatement of a step")

    def to_dict(self) -> dict:
        return self.model_dump()


# ─── Distribution result ──────────────────────────────────────────────────────

class DistributionResult:
    """Returned by DataGenerator.distribution(). Supports .print().plot() chaining."""

    _COLORS = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A']

    def __init__(self, counts: dict, total: int):
        self.counts = dict(counts)
        self.total  = total

    def __getitem__(self, key):  return self.counts[key]
    def __iter__(self):          return iter(self.counts)
    def __repr__(self):          return f"DistributionResult({self.counts})"

    def print(self) -> "DistributionResult":
        print(f"── Category Distribution ({self.total} entries) ──")
        for category, count in sorted(self.counts.items(), key=lambda x: -x[1]):
            print(f"  {category:<22s} {'█' * count} {count}")
        return self

    def plot(self) -> "DistributionResult":
        _require_matplotlib()
        import matplotlib.pyplot as plt

        labels = list(self.counts.keys())
        values = [self.counts[l] for l in labels]
        colors = (self._COLORS * math.ceil(len(labels) / len(self._COLORS)))[:len(labels)]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values, color=colors, alpha=0.85)
        ax.set_title("Repair Dataset — Category Distribution", fontsize=13)
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        if len(labels) > 1:
            ax.axhline(self.total / len(labels), color="grey",
                       linestyle="--", linewidth=1, label="expected even split")
            ax.legend(fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                    str(val), ha="center", fontsize=10)
        plt.tight_layout()
        plt.show()
        return self


# ─── Base generator ───────────────────────────────────────────────────────────

class DataGenerator:
    """Generates synthetic DIY repair Q&A entries using instructor for structured output."""

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str | None = None):
        import instructor
        from groq import Groq
        self._client = instructor.from_groq(
            Groq(api_key=api_key),
            mode=instructor.Mode.JSON,
        )

    _embedder = None  # lazy-loaded, shared across calls

    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("pip install sentence-transformers") from None
            cls._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._embedder

    def _is_too_similar(self, question: str, seen_questions: list[str], threshold: float = 0.80) -> bool:
        if not seen_questions:
            return False
        import numpy as np
        model = self._get_embedder()
        vecs = model.encode([question] + seen_questions, normalize_embeddings=True)
        sims = vecs[0] @ vecs[1:].T
        return bool(np.any(sims >= threshold))

    def generate_one(self, category: str, already_generated: list[str] | None = None) -> RepairEntry:
        prompt = IMPROVED_GENERATION_PROMPT.format(
            category=category,
            already_generated=json.dumps(already_generated or []),
        )
        return _with_backoff(lambda: self._client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_model=RepairEntry,
            temperature=0.9,
            max_tokens=800,
        ))

    def generate_dataset(
        self,
        n: int = 50,
        categories: list[str] | None = None,
        on_progress: Callable[[int, int, RepairEntry], None] | None = None,
        output_path: str = "repair_dataset.json",
        request_delay: float = 15.0,
    ) -> list[RepairEntry]:
        categories = categories or CATEGORIES
        entries: list[RepairEntry] = []
        seen_problems: list[str] = []   # equipment_problem labels fed back into prompt
        seen_questions: list[str] = []  # full questions for similarity gating
        SIMILARITY_THRESHOLD = 0.80
        SIMILARITY_RETRIES   = 3

        for i in range(n):
            category = categories[i % len(categories)]
            entry = None
            for attempt in range(SIMILARITY_RETRIES):
                candidate = _with_backoff(
                    lambda: self.generate_one(category, already_generated=seen_problems)
                )
                if self._is_too_similar(candidate.question, seen_questions, SIMILARITY_THRESHOLD):
                    print(f"  [{i+1}/{n}] similarity reject (attempt {attempt+1}) — {candidate.question[:60]}")
                else:
                    entry = candidate
                    break
            if entry is None:
                print(f"  [{i+1}/{n}] WARNING: kept best candidate after {SIMILARITY_RETRIES} retries")
                entry = candidate  # use last candidate rather than stalling the run

            entries.append(entry)
            seen_problems.append(entry.equipment_problem)
            seen_questions.append(entry.question)
            print(f"  [{i+1}/{n}] {entry.category:<22s} {entry.question[:60]}")
            if on_progress:
                on_progress(i + 1, n, entry)
            if i < n - 1:
                time.sleep(request_delay)

        self.save(entries, output_path)
        return entries

    # ── Schema validation ─────────────────────────────────────────────────────

    def validate(self, entries: list, semantic: bool = False) -> dict:
        passed, failed = [], []

        for i, entry in enumerate(entries):
            data     = entry if isinstance(entry, dict) else entry.model_dump()
            category = entry.get("category", "") if isinstance(entry, dict) else entry.category
            try:
                RepairEntry.model_validate(data)
                passed.append(i)
            except ValidationError as e:
                failed.append({"index": i, "category": category, "errors": e.errors()})

        report = {
            "total":     len(entries),
            "passed":    len(passed),
            "failed":    len(failed),
            "pass_rate": round(len(passed) / len(entries) * 100, 1) if entries else 0.0,
            "failures":  failed,
        }

        print(f"Validation: {report['passed']}/{report['total']} passed ({report['pass_rate']}%)")
        for f in failed:
            print(f"  [#{f['index']}] {f['category']}")
            for err in f["errors"]:
                print(f"      {err['loc']} — {err['msg']}")

        exact  = self.find_duplicates_exact(entries)
        report["exact_duplicates"] = exact
        if exact:
            print(f"\nExact duplicates: {len(exact)}")
            for i, j in exact:
                print(f"  [{i}] ↔ [{j}]")
        else:
            print("\nNo exact duplicates found.")

        if semantic:
            sem = self.find_duplicates_semantic(entries)
            report["semantic_duplicates"] = sem
            if sem:
                print(f"\nSemantic duplicates (≥0.85): {len(sem)}")
                raw = [e if isinstance(e, dict) else e.model_dump() for e in entries]
                for i, j, score in sem:
                    print(f"  [{i}] ↔ [{j}]  score={score:.3f}")
                    print(f"    {raw[i]['question'][:70]}")
                    print(f"    {raw[j]['question'][:70]}")
            else:
                print("No semantic duplicates found.")

        return report

    # ── Deduplication ─────────────────────────────────────────────────────────

    def find_duplicates_exact(self, entries: list) -> list[tuple[int, int]]:
        """Flag pairs whose questions match after normalisation (lowercase, strip punctuation)."""
        def _norm(text: str) -> str:
            return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())

        raw = [e if isinstance(e, dict) else e.model_dump() for e in entries]
        seen: dict[str, int] = {}
        dupes: list[tuple[int, int]] = []
        for i, entry in enumerate(raw):
            key = _norm(entry.get("question", ""))
            if key in seen:
                dupes.append((seen[key], i))
            else:
                seen[key] = i
        return dupes

    def find_duplicates_semantic(
        self,
        entries: list,
        threshold: float = 0.85,
    ) -> list[tuple[int, int, float]]:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np  # noqa: F401  — availability check; required by sentence-transformers
        except ImportError:
            raise ImportError("pip install sentence-transformers numpy") from None

        raw = [e if isinstance(e, dict) else e.model_dump() for e in entries]
        questions = [e.get("question", "") for e in raw]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(questions, normalize_embeddings=True)
        sim = embeddings @ embeddings.T

        dupes: list[tuple[int, int, float]] = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                if sim[i][j] >= threshold:
                    dupes.append((i, j, float(sim[i][j])))
        return sorted(dupes, key=lambda x: -x[2])

    def distribution(self, entries: list) -> DistributionResult:
        def _get_category(e):
            return e.get("category", "") if isinstance(e, dict) else e.category

        counts = Counter(_get_category(e) for e in entries)
        return DistributionResult(counts, total=len(entries))

    def save(self, entries: list[RepairEntry], path: str) -> None:
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)


class BaselineDataGenerator(DataGenerator):
    """Weaker data generator using a minimal prompt with no worked example, no
    step-ordering rules, no tool-matching requirement, and no safety specificity
    requirement. Produces lower-quality training data for baseline comparison."""

    def generate_one(self, category: str, already_generated: list[str] | None = None) -> RepairEntry:
        prompt = BASELINE_GENERATION_PROMPT.format(
            category=category,
            already_generated=json.dumps(already_generated or []),
        )
        return _with_backoff(lambda: self._client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_model=RepairEntry,
            temperature=0.9,
            max_tokens=800,
        ))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(description="Generate synthetic DIY repair Q&A dataset")
    p.add_argument("--n",       type=int,   default=50,
                   help="Number of entries to generate (default: 50)")
    p.add_argument("--output",  default="repair_dataset.json",
                   help="Output JSON path (default: repair_dataset.json)")
    p.add_argument("--variant", default="improved", choices=["improved", "baseline"],
                   help="Generator prompt variant (default: improved)")
    p.add_argument("--delay",   type=float, default=15.0,
                   help="Seconds between API calls (default: 15.0)")
    args = p.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: GROQ_API_KEY is not set. Add it to your environment or a .env file.")

    GeneratorClass = BaselineDataGenerator if args.variant == "baseline" else DataGenerator
    gen = GeneratorClass(api_key=api_key)
    entries = gen.generate_dataset(n=args.n, output_path=args.output, request_delay=args.delay)
    gen.distribution(entries).print()
    print(f"\nDone. {len(entries)} entries saved to {args.output}")


if __name__ == "__main__":
    main()

