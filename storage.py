"""
SQLite persistence layer for the LLM Judge Pipeline app.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "llm_judge_app.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id                    TEXT PRIMARY KEY,
            created_at            TEXT,
            generator_variant     TEXT,
            judge_variant         TEXT,
            status                TEXT DEFAULT 'pending',
            n_requested           INTEGER,
            generator_prompt_name TEXT,
            judge_prompt_name     TEXT,
            notes                 TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS items (
            id                TEXT PRIMARY KEY,
            run_id            TEXT,
            trace_id          TEXT UNIQUE,
            category          TEXT,
            question          TEXT,
            equipment_problem TEXT,
            answer            TEXT,
            tools_required    TEXT,
            steps             TEXT,
            safety_info       TEXT,
            tips              TEXT,
            generated_at      TEXT,
            model             TEXT,
            raw_response      TEXT DEFAULT '',
            gate_passed       INTEGER DEFAULT 0,
            gate_results      TEXT DEFAULT '{}',
            status            TEXT DEFAULT 'generated'
        );

        CREATE TABLE IF NOT EXISTS human_labels (
            id                        TEXT PRIMARY KEY,
            item_id                   TEXT,
            trace_id                  TEXT,
            D1_answer_completeness    INTEGER,
            D2_safety_specificity     INTEGER,
            D3_tool_realism           INTEGER,
            D4_scope_appropriateness  INTEGER,
            D5_context_clarity        INTEGER,
            D6_tip_usefulness         INTEGER,
            overall                   TEXT,
            notes                     TEXT DEFAULT '',
            labeled_at                TEXT
        );

        CREATE TABLE IF NOT EXISTS llm_labels (
            id                        TEXT PRIMARY KEY,
            item_id                   TEXT,
            trace_id                  TEXT,
            judge_prompt_name         TEXT,
            D1_answer_completeness    INTEGER,
            D2_safety_specificity     INTEGER,
            D3_tool_realism           INTEGER,
            D4_scope_appropriateness  INTEGER,
            D5_context_clarity        INTEGER,
            D6_tip_usefulness         INTEGER,
            overall                   TEXT,
            reasoning                 TEXT DEFAULT '',
            raw_response              TEXT DEFAULT '',
            latency_ms                REAL DEFAULT 0.0,
            judged_at                 TEXT
        );

        CREATE TABLE IF NOT EXISTS prompts (
            name       TEXT PRIMARY KEY,
            content    TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS iteration_log (
            id              TEXT PRIMARY KEY,
            run_id          TEXT,
            phase           TEXT,
            component       TEXT,
            hypothesis      TEXT,
            prompt_before   TEXT DEFAULT '',
            prompt_after    TEXT DEFAULT '',
            metrics_before  TEXT DEFAULT '{}',
            metrics_after   TEXT DEFAULT '{}',
            decision        TEXT DEFAULT '',
            created_at      TEXT
        );
        """)
    # migrate existing DBs that pre-date the notes column
    with get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(human_labels)").fetchall()}
        if "notes" not in cols:
            conn.execute("ALTER TABLE human_labels ADD COLUMN notes TEXT DEFAULT ''")
    _seed_default_prompts()


def _seed_default_prompts() -> None:
    from judge import (
        BaselinePromptBuilder, ImprovedPromptBuilder,
        SoftenPromptBuilder, PermissivePromptBuilder,
    )

    defaults = {
        "judge_baseline":   BaselinePromptBuilder.TEMPLATE,
        "judge_calibrate":  ImprovedPromptBuilder.TEMPLATE,
        "judge_soften":     SoftenPromptBuilder.TEMPLATE,
        "judge_permissive": PermissivePromptBuilder.TEMPLATE,
    }
    with get_conn() as conn:
        for name, content in defaults.items():
            exists = conn.execute(
                "SELECT 1 FROM prompts WHERE name=?", (name,)
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO prompts (name, content, updated_at) VALUES (?,?,?)",
                    (name, content, _now()),
                )


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _uid() -> str:
    return str(uuid.uuid4())

def _json_loads_list(v) -> list:
    try:
        return json.loads(v or "[]")
    except Exception:
        return []

def _json_loads_dict(v) -> dict:
    try:
        return json.loads(v or "{}")
    except Exception:
        return {}


# ── runs ──────────────────────────────────────────────────────────────────────

def create_run(
    generator_variant: str,
    judge_variant: str,
    n_requested: int,
    generator_prompt_name: str,
    judge_prompt_name: str,
    notes: str = "",
) -> str:
    run_id = _uid()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, _now(), generator_variant, judge_variant,
             "pending", n_requested, generator_prompt_name, judge_prompt_name, notes),
        )
    return run_id


def update_run_status(run_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))


def delete_run(run_id: str) -> None:
    with get_conn() as conn:
        conn.execute("""
            DELETE FROM llm_labels WHERE item_id IN
                (SELECT id FROM items WHERE run_id=?)
        """, (run_id,))
        conn.execute("""
            DELETE FROM human_labels WHERE item_id IN
                (SELECT id FROM items WHERE run_id=?)
        """, (run_id,))
        conn.execute("DELETE FROM items          WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM iteration_log  WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM runs           WHERE id=?",     (run_id,))


def get_runs() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()]


def get_run(run_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


# ── items ─────────────────────────────────────────────────────────────────────

def save_item(
    run_id: str,
    entry: dict,
    trace_id: str,
    model: str = "",
    raw_response: str = "",
) -> str:
    item_id = _uid()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO items
               (id, run_id, trace_id, category, question, equipment_problem,
                answer, tools_required, steps, safety_info, tips,
                generated_at, model, raw_response, gate_passed, gate_results, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item_id, run_id, trace_id,
                entry.get("category", ""),
                entry.get("question", ""),
                entry.get("equipment_problem", ""),
                entry.get("answer", ""),
                json.dumps(entry.get("tools_required", [])),
                json.dumps(entry.get("steps", [])),
                entry.get("safety_info", ""),
                json.dumps(entry.get("tips", [])),
                _now(), model, raw_response,
                0, "{}", "generated",
            ),
        )
    return item_id


def update_item_gate(item_id: str, passed: bool, gate_results: dict) -> None:
    status = "gated" if passed else "generated"
    with get_conn() as conn:
        conn.execute(
            "UPDATE items SET gate_passed=?, gate_results=?, status=? WHERE id=?",
            (1 if passed else 0, json.dumps(gate_results), status, item_id),
        )


def get_items(run_id: str | None = None, gate_passed: bool | None = None) -> list[dict]:
    with get_conn() as conn:
        q, params = "SELECT * FROM items", []
        conds: list[str] = []
        if run_id is not None:
            conds.append("run_id=?"); params.append(run_id)
        if gate_passed is not None:
            conds.append("gate_passed=?"); params.append(1 if gate_passed else 0)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY generated_at"
        rows = conn.execute(q, params).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        d["tools_required"] = _json_loads_list(d.get("tools_required"))
        d["steps"]          = _json_loads_list(d.get("steps"))
        d["tips"]           = _json_loads_list(d.get("tips"))
        d["gate_results"]   = _json_loads_dict(d.get("gate_results"))
        items.append(d)
    return items


# ── human labels ──────────────────────────────────────────────────────────────

DIM_KEYS = [
    "D1_answer_completeness",
    "D2_safety_specificity",
    "D3_tool_realism",
    "D4_scope_appropriateness",
    "D5_context_clarity",
    "D6_tip_usefulness",
]


def save_human_label(item_id: str, trace_id: str, labels: dict, notes: str = "") -> str:
    label_id = _uid()
    overall = "RESOLVED" if all(labels.get(k, False) for k in DIM_KEYS) else "NOT_RESOLVED"
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO human_labels
               (id, item_id, trace_id,
                D1_answer_completeness, D2_safety_specificity, D3_tool_realism,
                D4_scope_appropriateness, D5_context_clarity, D6_tip_usefulness,
                overall, notes, labeled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                label_id, item_id, trace_id,
                int(bool(labels.get("D1_answer_completeness"))),
                int(bool(labels.get("D2_safety_specificity"))),
                int(bool(labels.get("D3_tool_realism"))),
                int(bool(labels.get("D4_scope_appropriateness"))),
                int(bool(labels.get("D5_context_clarity"))),
                int(bool(labels.get("D6_tip_usefulness"))),
                overall, notes, _now(),
            ),
        )
        conn.execute("UPDATE items SET status='labeled' WHERE id=?", (item_id,))
    return label_id


def get_human_labels(run_id: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if run_id:
            rows = conn.execute(
                """SELECT hl.* FROM human_labels hl
                   JOIN items i ON hl.item_id = i.id
                   WHERE i.run_id=?""",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM human_labels").fetchall()
        return [dict(r) for r in rows]


def get_human_label_for_item(item_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM human_labels WHERE item_id=? ORDER BY labeled_at DESC LIMIT 1",
            (item_id,),
        ).fetchone()
        return dict(row) if row else None


# ── LLM labels ────────────────────────────────────────────────────────────────

def save_llm_label(
    item_id: str,
    trace_id: str,
    judge_prompt_name: str,
    dim_judgment,
    latency_ms: float = 0.0,
) -> str:
    label_id = _uid()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO llm_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                label_id, item_id, trace_id, judge_prompt_name,
                int(bool(dim_judgment.D1_answer_completeness)),
                int(bool(dim_judgment.D2_safety_specificity)),
                int(bool(dim_judgment.D3_tool_realism)),
                int(bool(dim_judgment.D4_scope_appropriateness)),
                int(bool(dim_judgment.D5_context_clarity)),
                int(bool(dim_judgment.D6_tip_usefulness)),
                dim_judgment.overall,
                getattr(dim_judgment, "reasoning", ""),
                getattr(dim_judgment, "raw", ""),
                latency_ms,
                _now(),
            ),
        )
        conn.execute("UPDATE items SET status='judged' WHERE id=?", (item_id,))
    return label_id


def get_llm_labels(run_id: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if run_id:
            rows = conn.execute(
                """SELECT ll.* FROM llm_labels ll
                   JOIN items i ON ll.item_id = i.id
                   WHERE i.run_id=?""",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM llm_labels").fetchall()
        return [dict(r) for r in rows]


def get_judge_run_summaries() -> list[dict]:
    """One row per (run, judge_prompt_name) with aggregate pass rates."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                r.id                                          AS run_id,
                r.created_at                                  AS created_at,
                r.generator_variant                           AS generator_variant,
                r.status                                      AS run_status,
                r.notes                                       AS notes,
                ll.judge_prompt_name                          AS judge_prompt_name,
                COUNT(ll.id)                                  AS n_judged,
                SUM(ll.overall = 'RESOLVED')                  AS n_resolved,
                AVG(ll.D1_answer_completeness)                AS D1,
                AVG(ll.D2_safety_specificity)                 AS D2,
                AVG(ll.D3_tool_realism)                       AS D3,
                AVG(ll.D4_scope_appropriateness)              AS D4,
                AVG(ll.D5_context_clarity)                    AS D5,
                AVG(ll.D6_tip_usefulness)                     AS D6,
                AVG(ll.latency_ms)                            AS avg_latency_ms
            FROM llm_labels ll
            JOIN items i ON ll.item_id = i.id
            JOIN runs r  ON i.run_id   = r.id
            GROUP BY r.id, ll.judge_prompt_name
            ORDER BY r.created_at DESC, ll.judge_prompt_name
        """).fetchall()
        return [dict(r) for r in rows]


# ── prompts ───────────────────────────────────────────────────────────────────

def get_prompt(name: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT content FROM prompts WHERE name=?", (name,)
        ).fetchone()
        return row["content"] if row else ""


def save_prompt(name: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO prompts (name, content, updated_at) VALUES (?,?,?)",
            (name, content, _now()),
        )


def get_all_prompts() -> dict[str, dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM prompts ORDER BY name").fetchall()
        return {r["name"]: dict(r) for r in rows}


# ── iteration log ─────────────────────────────────────────────────────────────

def save_iteration(
    run_id: str,
    phase: str,
    component: str,
    hypothesis: str,
    prompt_before: str = "",
    prompt_after: str = "",
    metrics_before: dict | None = None,
    metrics_after: dict | None = None,
    decision: str = "",
) -> str:
    it_id = _uid()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO iteration_log VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                it_id, run_id, phase, component,
                hypothesis, prompt_before, prompt_after,
                json.dumps(metrics_before or {}),
                json.dumps(metrics_after or {}),
                decision, _now(),
            ),
        )
    return it_id


def get_pending_judge_reruns(run_id: str) -> list[dict]:
    """Iterations for this run that changed a judge prompt but haven't captured metrics_after yet."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM iteration_log
               WHERE run_id=?
                 AND component LIKE 'judge_%'
                 AND prompt_after != ''
                 AND (metrics_after IS NULL OR metrics_after = '{}')
               ORDER BY created_at DESC""",
            (run_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["metrics_before"] = _json_loads_dict(d.get("metrics_before"))
        d["metrics_after"]  = _json_loads_dict(d.get("metrics_after"))
        result.append(d)
    return result


def update_iteration_metrics_after(iteration_id: str, metrics_after: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE iteration_log SET metrics_after=? WHERE id=?",
            (json.dumps(metrics_after), iteration_id),
        )


def get_iterations(run_id: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if run_id:
            rows = conn.execute(
                "SELECT * FROM iteration_log WHERE run_id=? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM iteration_log ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metrics_before"] = _json_loads_dict(d.get("metrics_before"))
            d["metrics_after"]  = _json_loads_dict(d.get("metrics_after"))
            result.append(d)
        return result
