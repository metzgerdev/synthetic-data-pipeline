from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import streamlit as st

import storage
storage.init_db()

from judge import DIM_KEYS
from generate import CATEGORIES

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Judge Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────

DIM_SHORT = {
    "D1_answer_completeness":   "D1",
    "D2_safety_specificity":    "D2",
    "D3_tool_realism":          "D3",
    "D4_scope_appropriateness": "D4",
    "D5_context_clarity":       "D5",
    "D6_tip_usefulness":        "D6",
}
DIM_LABEL = {
    "D1_answer_completeness":   "D1\nAnswer Completeness",
    "D2_safety_specificity":    "D2\nSafety Specificity",
    "D3_tool_realism":          "D3\nTool Realism",
    "D4_scope_appropriateness": "D4\nScope Appropriateness",
    "D5_context_clarity":       "D5\nContext Clarity",
    "D6_tip_usefulness":        "D6\nTip Usefulness",
}
DIM_DESC = {
    "D1_answer_completeness":   "Complete repair path (tools, steps, safety, tip)?",
    "D2_safety_specificity":    "Safety names SPECIFIC hazard for THIS repair?",
    "D3_tool_realism":          "All tools homeowner-accessible (< $50)?",
    "D4_scope_appropriateness": "Within realistic DIY capability?",
    "D5_context_clarity":       "Addresses the most likely cause of the symptom?",
    "D6_tip_usefulness":        "Tip is non-obvious and task-specific?",
}
PROMPT_DISPLAY = {
    "judge_baseline":   "Judge — Baseline",
    "judge_calibrate":  "Judge — Calibrate",
    "judge_soften":     "Judge — Soften",
    "judge_permissive": "Judge — Permissive",
}

# Cause → intended effect for each known judge prompt version.
# Shown in the cross-run summary table.
PROMPT_CHANGES: dict[str, str] = {
    "judge_baseline":    "Reference baseline — all 6 dims strict; D2 requires hazard-specific safety; D6 restatements fail",
    "judge_calibrate":   "D2: safety must match repair type, not just any plausible hazard; D5: must address most likely cause → ↓ false positives on mismatched safety/context",
    "judge_soften":      "Framing shifted to 'score like a homeowner, default PASS'; D5/D6 only fail if completely off-topic → ↑↑ RESOLVED, reduced false negatives",
    "judge_permissive":  "Reframed as 'helpful starting point, not a repair manual'; D5 only fails if wrong appliance entirely; D1 passes if any actionable steps exist → near-maximum RESOLVED",
}

# ── selected run ─────────────────────────────────────────────────────────────

_stored_rid = st.session_state.get("selected_run_id")
if _stored_rid and storage.get_run(_stored_rid):
    selected_run_id = _stored_rid
else:
    selected_run_id = None
    st.session_state.pop("selected_run_id", None)


# ── helpers ───────────────────────────────────────────────────────────────────

DIM_CONTENT_FIELD = {
    "D1_answer_completeness":   ("steps",         "Steps"),
    "D2_safety_specificity":    ("safety_info",   "Safety info"),
    "D3_tool_realism":          ("tools_required","Tools"),
    "D4_scope_appropriateness": ("answer",        "Answer"),
    "D5_context_clarity":       ("question",      "Question"),
    "D6_tip_usefulness":        ("tips",          "Tips"),
}


def render_dim_detail(item: dict, labels: dict, source_label: str = "Label") -> None:
    st.markdown(f"**{item.get('trace_id','')}** · {item.get('category','')} · {labels.get('overall','')}")
    st.caption(item.get("question", ""))
    st.divider()
    cols = st.columns(2)
    for idx, dim in enumerate(DIM_KEYS):
        field, field_label = DIM_CONTENT_FIELD[dim]
        val     = labels.get(dim)
        icon    = "✓" if val else "✗"
        color   = "green" if val else "red"
        content = item.get(field, "")
        if isinstance(content, list):
            content = "\n".join(f"• {x}" for x in content)
        with cols[idx % 2]:
            st.markdown(
                f"<span style='color:{color};font-weight:700'>{icon} {DIM_SHORT[dim]}</span>"
                f" — {DIM_DESC[dim]}",
                unsafe_allow_html=True,
            )
            st.caption(f"**{field_label}:** {str(content)[:300]}")



def compute_agreement_metrics(run_id: str) -> dict:
    human_labels = storage.get_human_labels(run_id)
    llm_labels   = storage.get_llm_labels(run_id)
    items        = storage.get_items(run_id)
    item_by_trace = {i["trace_id"]: i for i in items}
    human_by_item = {l["item_id"]: l for l in human_labels}

    paired = []
    for lbl in llm_labels:
        item = item_by_trace.get(lbl["trace_id"])
        if not item:
            continue
        human = human_by_item.get(item["id"])
        if human:
            paired.append({"llm": lbl, "human": human})

    if not paired:
        return {}

    metrics: dict = {"n": len(paired)}
    for dim in DIM_KEYS:
        agree = sum(1 for p in paired if bool(p["llm"][dim]) == bool(p["human"][dim]))
        metrics[dim] = round(agree / len(paired), 3)
    agree_ov = sum(1 for p in paired if p["llm"]["overall"] == p["human"]["overall"])
    metrics["overall"] = round(agree_ov / len(paired), 3)
    return metrics


def run_judge_on_items(
    items: list[dict],
    judge_variant: str,
    judge_prompt_name: str,
    delay: float = 1.0,
    progress_ph=None,
    status_ph=None,
) -> None:
    from judge import LLMJudge, Sample, DynamicPromptBuilder, DimJudgmentParser, BaselinePromptBuilder, ImprovedPromptBuilder
    from judge import GroqClient

    prompt_content = storage.get_prompt(judge_prompt_name)
    if prompt_content:
        judge = LLMJudge(
            client=GroqClient(os.getenv("GROQ_API_KEY")),
            prompt=DynamicPromptBuilder(prompt_content),
            parser=DimJudgmentParser(),
            max_tokens=400,
        )
    else:
        prompt_builder = ImprovedPromptBuilder() if judge_variant == "improved" else BaselinePromptBuilder()
        judge = LLMJudge.create(prompt=prompt_builder, api_key=os.getenv("GROQ_API_KEY"))
    for k, item in enumerate(items):
        safety = item.get("safety_info", "").strip()
        bot_response = item["answer"]
        if safety:
            bot_response += f"\n\nSafety note: {safety}"
        sample = Sample(
            user_query=item["question"],
            bot_response=bot_response,
            issue_type=item["category"],
        )
        t0  = time.perf_counter()
        dj  = judge.evaluate_dims(sample)
        lat = (time.perf_counter() - t0) * 1000
        storage.save_llm_label(
            item_id=item["id"],
            trace_id=item["trace_id"],
            judge_prompt_name=judge_prompt_name,
            dim_judgment=dj,
            latency_ms=lat,
        )
        if progress_ph:
            progress_ph.progress((k + 1) / len(items))
        if status_ph:
            flags = " ".join("✓" if getattr(dj, d) else "✗" for d in DIM_KEYS)
            status_ph.info(f"[{k+1}/{len(items)}] {item['trace_id']}  {dj.overall}  {flags}")
        if k < len(items) - 1:
            time.sleep(delay)



# ── tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["Runs", "Generate", "Label", "Judge", "Analyze", "Logs", "Report"])

# ══════════════════════════════════════════════════════════════════════════════
# Runs
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Runs")

    _all_runs = storage.get_runs()

    # ── New Run form ──────────────────────────────────────────────────────────
    with st.expander("New Run", expanded=not _all_runs):
        nf1, nf2 = st.columns(2)
        _gen_var   = nf1.selectbox("Generator", ["improved", "baseline"], key="new_gen")
        _judge_var = nf2.selectbox("Judge",     ["improved", "baseline"], key="new_judge")
        _n_items   = st.number_input("Items to generate", 5, 200, 20, 5, key="new_n")
        _notes_run = st.text_input("Notes (optional)", key="new_notes")
        if st.button("Create Run", type="primary", key="create_run_btn"):
            _rid = storage.create_run(
                generator_variant=_gen_var,
                judge_variant=_judge_var,
                n_requested=_n_items,
                generator_prompt_name="",
                judge_prompt_name={"improved": "judge_calibrate", "baseline": "judge_baseline"}[_judge_var],
                notes=_notes_run,
            )
            st.session_state["selected_run_id"] = _rid
            st.success(f"Run {_rid[:8]} created and set as active.")
            st.rerun()

    # ── Runs table ────────────────────────────────────────────────────────────
    if not _all_runs:
        st.info("No runs yet — create one above.")
    else:
        _pending_del = st.session_state.get("pending_delete")

        _hcols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 2, 1.2, 1.2])
        for _hc, _lbl in zip(_hcols, ["Created", "ID", "Generator", "Judge", "Status", "Notes", "", ""]):
            if _lbl:
                _hc.markdown(f"**{_lbl}**")
        st.divider()

        for _run in _all_runs:
            _is_active = _run["id"] == selected_run_id
            _rcols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 2, 1.2, 1.2])
            _rcols[0].write(("▶  " if _is_active else "") + _run["created_at"][5:16])
            _rcols[1].code(_run["id"][:8])
            _rcols[2].write(_run["generator_variant"])
            _rcols[3].write(_run["judge_variant"])
            _rcols[4].write(_run["status"])
            _rcols[5].write(_run["notes"] or "—")
            with _rcols[6]:
                if st.button(
                    "✓ Active" if _is_active else "Activate",
                    key=f"act_{_run['id']}",
                    type="primary" if _is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["selected_run_id"] = _run["id"]
                    st.rerun()
            with _rcols[7]:
                if st.button("Delete", key=f"del_{_run['id']}", use_container_width=True):
                    st.session_state["pending_delete"] = _run["id"]
                    st.rerun()

        if _pending_del:
            _pdrun = next((r for r in _all_runs if r["id"] == _pending_del), None)
            if _pdrun:
                with st.container(border=True):
                    st.warning(
                        f"Delete run `{_pending_del[:8]}` "
                        f"({_pdrun['created_at'][5:16]}  {_pdrun['status']})? "
                        f"This cannot be undone."
                    )
                    _dc1, _dc2, _ = st.columns([1, 1, 4])
                    with _dc1:
                        if st.button("Confirm Delete", type="primary", key="confirm_del_btn"):
                            storage.delete_run(_pending_del)
                            if selected_run_id == _pending_del:
                                st.session_state.pop("selected_run_id", None)
                            st.session_state.pop("pending_delete", None)
                            st.rerun()
                    with _dc2:
                        if st.button("Cancel", key="cancel_del_btn"):
                            st.session_state.pop("pending_delete", None)
                            st.rerun()




# ══════════════════════════════════════════════════════════════════════════════
# Generate  (data generation + quality gate inline)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Generate")

    if not selected_run_id:
        st.warning("Create a run first (sidebar).")
    else:
        run   = storage.get_run(selected_run_id)
        items = storage.get_items(selected_run_id)
        gated = [i for i in items if i["gate_passed"]]

        # ── Generation ────────────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Data Generation")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Requested",  run["n_requested"])
            c2.metric("Generated",  len(items))
            c3.metric("Generator",  run["generator_variant"])
            c4.metric("Model",      "llama-3.3-70b-versatile")

            cats  = st.multiselect("Categories", CATEGORIES, default=CATEGORIES, key="gen_cats")
            delay = st.slider("Delay between API calls (s)", 0.0, 30.0, 1.0, 0.5, key="gen_delay")

            col_btn, col_warn = st.columns([1, 3])
            with col_btn:
                gen_clicked = st.button(
                    "Run Generation", type="primary",
                    disabled=bool(items),
                    key="run_gen_btn",
                )
            with col_warn:
                if items:
                    st.caption("Items already generated. Create a new run to re-generate.")

            if gen_clicked and not items:
                from generate import DataGenerator, BaselineDataGenerator

                api_key = os.getenv("GROQ_API_KEY")
                gen_cls = BaselineDataGenerator if run["generator_variant"] == "baseline" else DataGenerator
                gen     = gen_cls(api_key=api_key)
                n       = run["n_requested"]
                width   = len(str(n))

                progress  = st.progress(0, text="Starting…")
                log_box   = st.empty()
                log_lines: list[str] = []

                def on_progress(i: int, total: int, entry) -> None:
                    line = f"  [{i:>{width}}/{total}] {entry.category:<22s} {entry.question[:60]}"
                    log_lines.append(line)
                    progress.progress(i / total, text=f"Generating {i}/{total}…")
                    log_box.code("\n".join(log_lines), language=None)
                    storage.save_item(
                        run_id=selected_run_id,
                        entry=entry.model_dump(),
                        trace_id=f"{selected_run_id[:8]}_{run['generator_variant']}_{(i-1):04d}",
                        model=gen.MODEL,
                    )

                output_path = f"repair_dataset_{selected_run_id[:8]}.json"
                gen.generate_dataset(
                    n=n,
                    categories=cats,
                    on_progress=on_progress,
                    output_path=output_path,
                    request_delay=delay,
                )

                progress.progress(1.0, text=f"Done — {n} entries saved to {output_path}")
                storage.update_run_status(selected_run_id, "generated")
                # auto-run gate immediately after generation
                st.session_state["_auto_run_gate"] = True
                st.rerun()

        # ── Quality Gate ──────────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Quality Gate")

            if not items:
                st.info("Generate items first.")
            else:
                ungated_count = len([i for i in items if not i["gate_passed"] and i["status"] == "generated"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Total",     len(items))
                c2.metric("Passed",    len(gated))
                c3.metric("Pass rate", f"{len(gated)/len(items)*100:.0f}%" if items else "—")

                col_gate, col_note = st.columns([1, 3])
                with col_gate:
                    auto_gate = st.session_state.pop("_auto_run_gate", False)
                    gate_clicked = st.button("Run Gate", type="primary", key="run_gate_btn") or auto_gate
                with col_note:
                    if not ungated_count and gated:
                        st.caption("Gate already run — click to re-run.")

                if gate_clicked:
                    from gate import run_quality_gate
                    prog = st.progress(0, text="Running gate checks…")
                    passed_count = 0
                    fresh_items = storage.get_items(selected_run_id)
                    for k, item in enumerate(fresh_items):
                        result = run_quality_gate(item)
                        storage.update_item_gate(item["id"], result["passed"], result)
                        if result["passed"]:
                            passed_count += 1
                        prog.progress((k + 1) / len(fresh_items))
                    storage.update_run_status(selected_run_id, "gated")
                    st.success(f"Gate complete — {passed_count}/{len(fresh_items)} passed")
                    st.rerun()

                # Batch checks
                if gated:
                    from gate import check_distribution, check_deduplication
                    dedup = check_deduplication(gated)
                    dist  = check_distribution(gated, CATEGORIES)

                    dc1, dc2 = st.columns(2)
                    with dc1:
                        icon = "✅" if dedup["passed"] else "⚠️"
                        st.metric(f"{icon} Exact duplicates", dedup["count"])
                        for a, b in dedup["duplicate_pairs"]:
                            st.caption(f"  items #{a} ↔ #{b}")
                    with dc2:
                        st.markdown("**Category distribution**")
                        dist_rows = []
                        for cat, info in dist["categories"].items():
                            flag = "✅" if info["ok"] else "⚠️"
                            dist_rows.append({"": flag, "Category": cat,
                                              "Count": info["count"], "Pct": f"{info['pct']*100:.0f}%"})
                        st.table(dist_rows)

                # Per-item results
                st.subheader("Per-item results")
                filt = st.radio("Filter", ["All", "Passed", "Failed"], horizontal=True, key="gate_filter")
                show = items
                if filt == "Passed":
                    show = [i for i in items if i["gate_passed"]]
                elif filt == "Failed":
                    show = [i for i in items if not i["gate_passed"]]

                for item in show[:100]:
                    icon   = "✅" if item["gate_passed"] else "❌"
                    checks = item.get("gate_results", {}).get("checks", {})
                    failed = [n for n, r in checks.items() if not r.get("passed")]
                    with st.expander(
                        f"{icon} `{item['trace_id']}` · {item['category']} · {item['question'][:65]}"
                    ):
                        if checks:
                            for cname, cres in checks.items():
                                ci = "✅" if cres.get("passed") else "❌"
                                st.caption(f"{ci} **{cname}**: {cres.get('reason', '')}")
                        else:
                            st.caption("Gate not yet run.")

        # ── Generated items table ─────────────────────────────────────────────
        if items:
            with st.expander(f"Generated items ({len(items)})", expanded=False):
                table = [
                    {"trace_id": i["trace_id"], "category": i["category"], "question": i["question"][:90]}
                    for i in items
                ]
                st.dataframe(table, use_container_width=True, hide_index=True)

                idx = st.number_input("Inspect index", 0, len(items) - 1, 0, key="gen_inspect_idx")
                it  = items[idx]
                col_l, col_r = st.columns([1, 2])
                with col_l:
                    st.markdown(f"**Category:** {it['category']}")
                    st.markdown(f"**trace_id:** `{it['trace_id']}`")
                    st.markdown(f"**Problem:** {it['equipment_problem']}")
                    st.markdown("**Tools:**")
                    for t in it["tools_required"]:
                        st.caption(f"• {t}")
                    st.markdown("**Steps:**")
                    for j, s in enumerate(it["steps"], 1):
                        st.caption(f"{j}. {s}")
                with col_r:
                    st.markdown(f"**Question:** {it['question']}")
                    st.markdown(f"**Answer:**\n\n{it['answer']}")
                    st.markdown(f"**Safety:** {it['safety_info']}")
                    st.markdown("**Tips:**")
                    for tip in it["tips"]:
                        st.caption(f"• {tip}")


# ══════════════════════════════════════════════════════════════════════════════
# Label
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Label")

    if not selected_run_id:
        st.warning("Create a run first (sidebar).")
    else:
        gated_items   = storage.get_items(selected_run_id, gate_passed=True)
        labeled_items = [i for i in gated_items if i["status"] in ("labeled", "judged")]
        unlabeled     = [i for i in gated_items if i["status"] == "gated"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Gated",     len(gated_items))
        c2.metric("Labeled",   len(labeled_items))
        c3.metric("Remaining", len(unlabeled))

        if not gated_items:
            st.info("Run the quality gate first (Generate tab).")
        else:
            st.divider()

            show_pool  = unlabeled if unlabeled else gated_items
            pool_label = "Unlabeled items" if unlabeled else "All gated items (re-labeling)"

            col_nav, col_item = st.columns([1, 3])
            with col_nav:
                st.caption(pool_label)
                label_idx = st.number_input(
                    "Item index", 0, max(0, len(show_pool) - 1), 0, key="label_nav"
                )
                item = show_pool[label_idx]
                st.caption(f"`{item['trace_id']}`")
                st.caption(f"Category: {item['category']}")
                st.caption(f"Status: {item['status']}")

                existing = storage.get_human_label_for_item(item["id"])
                if existing:
                    st.caption("Existing labels:")
                    for dim in DIM_KEYS:
                        v = existing.get(dim, 0)
                        st.caption(f"  {'✅' if v else '❌'} {DIM_SHORT[dim]}")

            with col_item:
                st.markdown(f"**Q:** {item['question']}")
                st.markdown("---")
                st.markdown(item["answer"])
                if item.get("safety_info"):
                    st.info(f"Safety: {item['safety_info']}")

            st.divider()
            st.subheader("Labels")

            st.markdown(
                "<style>input[type='checkbox'] { accent-color: #2E7D32; }</style>",
                unsafe_allow_html=True,
            )

            existing_vals = (
                {dim: bool(existing.get(dim, 0)) for dim in DIM_KEYS}
                if existing else {dim: False for dim in DIM_KEYS}
            )
            item_id = item["id"]

            def _apply_select_all():
                val = st.session_state[f"select_all_{item_id}"]
                for dim in DIM_KEYS:
                    st.session_state[f"lbl_{dim}_{item_id}"] = val

            st.checkbox(
                "Select all",
                value=all(existing_vals.values()),
                key=f"select_all_{item_id}",
                on_change=_apply_select_all,
            )

            st.markdown("<div style='margin-top: 0.5rem'></div>", unsafe_allow_html=True)
            label_cols  = st.columns(3)
            new_labels: dict[str, bool] = {}
            for i, dim in enumerate(DIM_KEYS):
                with label_cols[i % 3]:
                    new_labels[dim] = st.checkbox(
                        f"**{DIM_SHORT[dim]}** — {DIM_DESC[dim]}",
                        value=existing_vals[dim],
                        key=f"lbl_{dim}_{item_id}",
                    )

            st.markdown("<div style='margin-top: 0.75rem'></div>", unsafe_allow_html=True)
            notes_lbl = st.text_area(
                "Notes",
                value=existing.get("notes", "") if existing else "",
                placeholder="Optional annotation — edge cases, ambiguities, reviewer observations…",
                height=90,
                key=f"notes_{item_id}",
            )

            if st.button("Save Labels & Next", type="primary", key="save_labels_btn"):
                storage.save_human_label(item["id"], item["trace_id"], new_labels, notes=notes_lbl)
                st.success(f"Labels saved for `{item['trace_id']}`")
                time.sleep(0.3)
                st.rerun()

        # Results table
        human_labels_all = storage.get_human_labels(selected_run_id)
        if human_labels_all:
            item_by_id = {i["id"]: i for i in storage.get_items(selected_run_id)}
            st.subheader(f"Human label results ({len(human_labels_all)})")
            rows = []
            for lbl in human_labels_all:
                item_meta = item_by_id.get(lbl["item_id"], {})
                row: dict = {
                    "trace_id": lbl["trace_id"],
                    "category": item_meta.get("category", ""),
                    "overall":  lbl["overall"],
                }
                for dim in DIM_KEYS:
                    row[DIM_SHORT[dim]] = "✓" if lbl[dim] else "✗"
                row["notes"] = (lbl.get("notes") or "")[:60]
                rows.append(row)
            sel = st.dataframe(
                rows, use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row",
                key="human_label_table",
            )
            selected_rows = sel.selection.rows
            if selected_rows:
                lbl      = human_labels_all[selected_rows[0]]
                item_sel = item_by_id.get(lbl["item_id"], {})
                item_sel["trace_id"] = lbl["trace_id"]
                with st.container(border=True):
                    render_dim_detail(item_sel, lbl)
                    if lbl.get("notes"):
                        st.caption(f"**Notes:** {lbl['notes']}")


# ══════════════════════════════════════════════════════════════════════════════
# Judge
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Judge")

    if not selected_run_id:
        st.warning("Create a run first (sidebar).")
    else:
        run         = storage.get_run(selected_run_id)
        gated_items = storage.get_items(selected_run_id, gate_passed=True)

        # Prompt name (controls which saved prompt is loaded into the judge)
        all_judge_prompts = [k for k in storage.get_all_prompts() if k.startswith("judge_")]
        default_prompt = {"improved": "judge_calibrate", "baseline": "judge_baseline"}.get(run["judge_variant"], "judge_baseline")
        prompt_idx = all_judge_prompts.index(default_prompt) if default_prompt in all_judge_prompts else 0
        judge_prompt_sel = st.selectbox(
            "Judge prompt",
            all_judge_prompts,
            index=prompt_idx,
            format_func=lambda k: PROMPT_DISPLAY.get(k, k),
            key="judge_prompt_sel",
        )
        judge_prompt_content = storage.get_prompt(judge_prompt_sel)
        if judge_prompt_content:
            with st.expander("Prompt", expanded=False):
                edited_judge_prompt = st.text_area(
                    "Edit prompt",
                    value=judge_prompt_content,
                    height=300,
                    key=f"judge_prompt_text_{judge_prompt_sel}",
                    label_visibility="collapsed",
                )
                if st.button("Save prompt", key="save_judge_prompt_btn"):
                    storage.save_prompt(judge_prompt_sel, edited_judge_prompt)
                    st.success("Prompt saved.")

        # All metrics and results are scoped to the selected prompt
        llm_labels  = [l for l in storage.get_llm_labels(selected_run_id)
                       if l["judge_prompt_name"] == judge_prompt_sel]
        judged_ids  = {l["item_id"] for l in llm_labels}

        n_resolved     = sum(1 for l in llm_labels if l["overall"] == "RESOLVED")
        n_not_resolved = sum(1 for l in llm_labels if l["overall"] == "NOT_RESOLVED")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gated items", len(gated_items))
        c2.metric("Judged (this prompt)", len(judged_ids))
        c3.metric("RESOLVED", n_resolved)
        c4.metric("NOT_RESOLVED", n_not_resolved)

        delay_j  = st.slider("Delay (s)", 0.0, 10.0, 1.0, 0.5, key="judge_delay")
        only_new = st.checkbox("Only judge items not yet judged", value=True, key="judge_only_new")

        to_judge = [i for i in gated_items if i["id"] not in judged_ids] if only_new else gated_items

        col_b, col_i = st.columns([1, 3])
        with col_b:
            run_judge_btn = st.button(
                "Run Judge", type="primary",
                disabled=not to_judge,
                key="run_judge_btn",
            )
        with col_i:
            st.caption(f"{len(to_judge)} items to judge · prompt: {PROMPT_DISPLAY.get(judge_prompt_sel, judge_prompt_sel)}")

        if run_judge_btn and to_judge:
            from judge import LLMJudge, Sample, DynamicPromptBuilder, DimJudgmentParser
            from judge import GroqClient, BaselinePromptBuilder, ImprovedPromptBuilder

            judge_prompt_name = judge_prompt_sel
            prompt_content    = storage.get_prompt(judge_prompt_name)
            if prompt_content:
                judge = LLMJudge(
                    client=GroqClient(os.getenv("GROQ_API_KEY")),
                    prompt=DynamicPromptBuilder(prompt_content),
                    parser=DimJudgmentParser(),
                    max_tokens=400,
                )
            else:
                prompt_builder = BaselinePromptBuilder() if "baseline" in judge_prompt_sel else ImprovedPromptBuilder()
                judge = LLMJudge.create(prompt=prompt_builder, api_key=os.getenv("GROQ_API_KEY"))

            prog   = st.progress(0, text="Running judge…")
            status = st.empty()

            for k, item in enumerate(to_judge):
                safety = item.get("safety_info", "").strip()
                bot_response = item["answer"]
                if safety:
                    bot_response += f"\n\nSafety note: {safety}"
                sample = Sample(
                    user_query   = item["question"],
                    bot_response = bot_response,
                    issue_type   = item["category"],
                )
                t0 = time.perf_counter()
                try:
                    dj  = judge.evaluate_dims(sample)
                    lat = (time.perf_counter() - t0) * 1000
                    storage.save_llm_label(
                        item_id=item["id"],
                        trace_id=item["trace_id"],
                        judge_prompt_name=judge_prompt_name,
                        dim_judgment=dj,
                        latency_ms=lat,
                    )
                    flags = " ".join("✓" if getattr(dj, d) else "✗" for d in DIM_KEYS)
                    status.info(f"[{k+1}/{len(to_judge)}] {item['trace_id']}  {dj.overall}  {flags}")
                except Exception as exc:
                    status.error(f"[{k+1}/{len(to_judge)}] {item['trace_id']}: {exc}")

                prog.progress((k + 1) / len(to_judge))
                if k < len(to_judge) - 1:
                    time.sleep(delay_j)

            storage.update_run_status(selected_run_id, "judged")
            st.success("Judging complete!")
            st.rerun()

        # Results table — already filtered to judge_prompt_sel above
        if llm_labels:
            item_by_trace_j = {i["trace_id"]: i for i in gated_items}
            st.subheader(f"Judge results ({len(llm_labels)})")
            rows = []
            for lbl in llm_labels:
                row: dict = {"trace_id": lbl["trace_id"], "overall": lbl["overall"]}
                for dim in DIM_KEYS:
                    row[DIM_SHORT[dim]] = "✓" if lbl[dim] else "✗"
                row["reasoning"] = (lbl.get("reasoning") or "")[:80]
                row["lat_ms"]    = f"{lbl.get('latency_ms', 0):.0f}"
                rows.append(row)
            sel_j = st.dataframe(
                rows, use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row",
                key="llm_judge_table",
            )
            selected_rows_j = sel_j.selection.rows
            if selected_rows_j:
                lbl      = llm_labels[selected_rows_j[0]]
                item_sel = dict(item_by_trace_j.get(lbl["trace_id"], {}))
                with st.container(border=True):
                    render_dim_detail(item_sel, lbl, source_label="LLM")
                    if lbl.get("reasoning"):
                        st.caption(f"**Reasoning:** {lbl['reasoning']}")


# ══════════════════════════════════════════════════════════════════════════════
# Analyze
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("Analyze")

    # ── Cross-run summary ─────────────────────────────────────────────────────
    all_summaries = storage.get_judge_run_summaries()
    if all_summaries:
        with st.expander("All runs — judge summary", expanded=True):
            # Build a per-run baseline RESOLVED rate (judge_baseline prompt, or first result)
            # so each row can show Δ RESOLVED vs that reference.
            run_baseline: dict[str, float] = {}
            for s in all_summaries:
                rid = s["run_id"]
                n   = s["n_judged"] or 0
                if n == 0:
                    continue
                rate = (s["n_resolved"] or 0) / n
                # Prefer judge_baseline as the reference; otherwise take the first seen
                if s["judge_prompt_name"] == "judge_baseline" or rid not in run_baseline:
                    run_baseline[rid] = rate

            # Pull iteration hypotheses for custom prompts saved via iteration log.
            all_iterations = storage.get_iterations()
            iter_hypothesis: dict[str, str] = {}
            for it in all_iterations:
                comp = it.get("component", "")
                hyp  = (it.get("hypothesis") or "").strip()
                if comp and hyp and comp not in iter_hypothesis:
                    iter_hypothesis[comp] = hyp[:120]

            summary_rows = []
            for s in all_summaries:
                n        = s["n_judged"] or 0
                resolved = s["n_resolved"] or 0
                pname    = s["judge_prompt_name"] or ""
                rid      = s["run_id"]

                resolved_pct = resolved / n if n else None
                base_pct     = run_baseline.get(rid)
                if resolved_pct is not None and base_pct is not None and pname != "judge_baseline":
                    delta = resolved_pct - base_pct
                    delta_str = f"{delta:+.0%}"
                else:
                    delta_str = "—"

                # Cause description: static dict → iteration log → fallback
                cause = (
                    PROMPT_CHANGES.get(pname)
                    or iter_hypothesis.get(pname)
                    or "—"
                )

                summary_rows.append({
                    "Run":        s["run_id"][:8],
                    "Date":       (s["created_at"] or "")[:16],
                    "Gen":        s["generator_variant"] or "—",
                    "Prompt":     PROMPT_DISPLAY.get(pname, pname),
                    "Judged":     n,
                    "RESOLVED %": f"{resolved_pct*100:.0f}%" if resolved_pct is not None else "—",
                    "Δ vs ref":   delta_str,
                    "D1": f"{(s['D1'] or 0)*100:.0f}%",
                    "D2": f"{(s['D2'] or 0)*100:.0f}%",
                    "D3": f"{(s['D3'] or 0)*100:.0f}%",
                    "D4": f"{(s['D4'] or 0)*100:.0f}%",
                    "D5": f"{(s['D5'] or 0)*100:.0f}%",
                    "D6": f"{(s['D6'] or 0)*100:.0f}%",
                    "Avg ms":     f"{s['avg_latency_ms']:.0f}" if s["avg_latency_ms"] else "—",
                    "Changes":    cause,
                    "Notes":      (s["notes"] or "")[:40],
                })
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)
            st.caption("Δ vs ref = RESOLVED % change relative to the judge_baseline result for the same run (or first prompt judged if no baseline).")

    if not selected_run_id:
        st.warning("Create a run first (sidebar).")
    else:
        llm_labels_raw   = storage.get_llm_labels(selected_run_id)
        human_labels_raw = storage.get_human_labels(selected_run_id)
        items            = storage.get_items(selected_run_id)

        if not llm_labels_raw:
            st.info("Run the LLM judge first (Judge tab).")
        else:
            import matplotlib.pyplot as plt
            import numpy as np

            item_by_trace  = {i["trace_id"]: i for i in items}
            human_by_item  = {l["item_id"]: l for l in human_labels_raw}
            has_human      = bool(human_labels_raw)

            rows = []
            for lbl in llm_labels_raw:
                item = item_by_trace.get(lbl["trace_id"])
                if not item:
                    continue
                row: dict = {
                    "trace_id":    lbl["trace_id"],
                    "category":    item["category"],
                    "llm_overall": lbl["overall"],
                }
                for dim in DIM_KEYS:
                    row[f"llm_{dim}"] = bool(lbl[dim])
                human = human_by_item.get(item["id"])
                if human:
                    row["human_overall"] = human["overall"]
                    for dim in DIM_KEYS:
                        row[f"human_{dim}"] = bool(human[dim])
                rows.append(row)

            if not rows:
                st.warning("No rows to analyze.")
            else:
                cats_present = sorted({r["category"] for r in rows})
                dim_labels   = [DIM_LABEL[d] for d in DIM_KEYS]
                human_rows   = [r for r in rows if "human_overall" in r]

                resolved_llm = sum(1 for r in rows if r["llm_overall"] == "RESOLVED")
                st.subheader("Summary")
                if has_human and len(human_rows) < len(rows):
                    st.caption(
                        f"{len(human_rows)} of {len(rows)} items have human labels — "
                        "human metrics are computed over labeled items only."
                    )
                kc = st.columns(4 if has_human else 2)
                kc[0].metric("Items analyzed", len(rows))
                kc[1].metric("LLM RESOLVED",   f"{resolved_llm}/{len(rows)}",
                             f"{resolved_llm/len(rows)*100:.0f}%")
                if has_human:
                    resolved_h = sum(1 for r in human_rows if r.get("human_overall") == "RESOLVED")
                    kc[2].metric("Human RESOLVED", f"{resolved_h}/{len(human_rows)}",
                                 f"{resolved_h/len(human_rows)*100:.0f}%" if human_rows else "—")
                    agree_ov = sum(1 for r in human_rows if r.get("llm_overall") == r.get("human_overall"))
                    kc[3].metric("Overall agreement", f"{agree_ov/len(human_rows)*100:.0f}%" if human_rows else "—")

                st.subheader("Per-dimension pass rates")
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                x     = np.arange(len(DIM_KEYS))
                width = 0.35 if has_human else 0.6
                llm_r = [sum(r.get(f"llm_{d}", False) for r in rows) / len(rows) for d in DIM_KEYS]

                offset = -width / 2 if has_human else 0
                b1 = ax1.bar(x + offset, llm_r, width, label="LLM", color="#1565C0", alpha=0.85)

                if has_human and human_rows:
                    hum_r = [sum(r.get(f"human_{d}", False) for r in human_rows) / len(human_rows) for d in DIM_KEYS]
                    b2 = ax1.bar(x + width / 2, hum_r, width, label="Human", color="#C62828", alpha=0.85)
                    for bar in b2:
                        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                                 f"{bar.get_height():.0%}", ha="center", fontsize=8)

                for bar in b1:
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                             f"{bar.get_height():.0%}", ha="center", fontsize=8)

                ax1.set_xticks(x); ax1.set_xticklabels(dim_labels, fontsize=8)
                ax1.set_ylim(0, 1.18); ax1.set_ylabel("Pass rate")
                ax1.axhline(0.8, color="grey", linestyle="--", linewidth=1, label="0.8 target")
                ax1.legend(); ax1.set_title("Pass rate per dimension")
                plt.tight_layout()
                st.pyplot(fig1); plt.close(fig1)

                if len(cats_present) > 1:
                    st.subheader("LLM pass rate heatmap — category × dimension")
                    fig2, ax2 = plt.subplots(figsize=(10, max(3, len(cats_present))))
                    heat = np.zeros((len(cats_present), len(DIM_KEYS)))
                    for ci, cat in enumerate(cats_present):
                        cat_rows = [r for r in rows if r["category"] == cat]
                        for di, dim in enumerate(DIM_KEYS):
                            heat[ci, di] = (sum(r.get(f"llm_{dim}", False) for r in cat_rows)
                                            / len(cat_rows) if cat_rows else 0.0)
                    im = ax2.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
                    ax2.set_xticks(range(len(DIM_KEYS))); ax2.set_xticklabels(dim_labels, fontsize=8)
                    ax2.set_yticks(range(len(cats_present))); ax2.set_yticklabels(cats_present)
                    plt.colorbar(im, ax=ax2)
                    for ci in range(len(cats_present)):
                        for di in range(len(DIM_KEYS)):
                            ax2.text(di, ci, f"{heat[ci, di]:.0%}",
                                     ha="center", va="center", fontsize=9)
                    ax2.set_title("LLM pass rate by category × dimension")
                    plt.tight_layout()
                    st.pyplot(fig2); plt.close(fig2)

                if has_human and human_rows:
                    st.subheader("Human / LLM agreement per dimension")
                    agree_r = [
                        sum(1 for r in human_rows if r.get(f"llm_{d}") == r.get(f"human_{d}")) / len(human_rows)
                        for d in DIM_KEYS
                    ]
                    colors = ["#2E7D32" if v >= 0.8 else "#C62828" for v in agree_r]
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    bars = ax3.bar(dim_labels, agree_r, color=colors, alpha=0.85)
                    ax3.axhline(0.8, color="grey", linestyle="--", linewidth=1, label="80% threshold")
                    ax3.set_ylim(0, 1.15); ax3.set_ylabel("Agreement")
                    ax3.set_title("Human / LLM agreement per dimension (green ≥ 80%)")
                    ax3.tick_params(axis="x", labelsize=8)
                    ax3.legend()
                    for bar, val in zip(bars, agree_r):
                        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                                 f"{val:.0%}", ha="center", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig3); plt.close(fig3)

                    if len(cats_present) > 1:
                        st.subheader("Agreement heatmap — category × dimension")
                        fig4, ax4 = plt.subplots(figsize=(10, max(3, len(cats_present))))
                        agree_heat = np.zeros((len(cats_present), len(DIM_KEYS)))
                        for ci, cat in enumerate(cats_present):
                            cat_rows = [r for r in rows
                                        if r["category"] == cat and f"human_{DIM_KEYS[0]}" in r]
                            for di, dim in enumerate(DIM_KEYS):
                                if cat_rows:
                                    agree_heat[ci, di] = sum(
                                        1 for r in cat_rows
                                        if r.get(f"llm_{dim}") == r.get(f"human_{dim}")
                                    ) / len(cat_rows)
                        im4 = ax4.imshow(agree_heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
                        ax4.set_xticks(range(len(DIM_KEYS))); ax4.set_xticklabels(dim_labels, fontsize=8)
                        ax4.set_yticks(range(len(cats_present))); ax4.set_yticklabels(cats_present)
                        plt.colorbar(im4, ax=ax4)
                        for ci in range(len(cats_present)):
                            for di in range(len(DIM_KEYS)):
                                ax4.text(di, ci, f"{agree_heat[ci, di]:.0%}",
                                         ha="center", va="center", fontsize=9)
                        ax4.set_title("Agreement by category × dimension")
                        plt.tight_layout()
                        st.pyplot(fig4); plt.close(fig4)

                st.subheader("Export")
                st.download_button(
                    "Download analysis JSON",
                    data=json.dumps(rows, indent=2, default=str),
                    file_name=f"analysis_{selected_run_id[:8]}.json",
                    mime="application/json",
                )


# ══════════════════════════════════════════════════════════════════════════════
# Logs
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("Logs & Traces")

    log_view = st.radio(
        "View", ["Run Audit", "Log File", "Trace Detail"],
        horizontal=True, key="log_view"
    )

    if log_view == "Run Audit":
        if not selected_run_id:
            st.warning("Select a run.")
        else:
            items     = storage.get_items(selected_run_id)
            llm_map   = {l["item_id"]: l for l in storage.get_llm_labels(selected_run_id)}
            human_map = {l["item_id"]: l for l in storage.get_human_labels(selected_run_id)}

            filt_audit = st.radio(
                "Filter", ["All", "Gate passed", "Gate failed"],
                horizontal=True, key="audit_filter"
            )
            audit_items = items
            if filt_audit == "Gate passed":
                audit_items = [i for i in items if i["gate_passed"]]
            elif filt_audit == "Gate failed":
                audit_items = [i for i in items if not i["gate_passed"]]

            st.caption(f"Showing {len(audit_items)} items")
            for item in audit_items[:200]:
                gate_icon = "✅" if item["gate_passed"] else "❌"
                llm       = llm_map.get(item["id"])
                human     = human_map.get(item["id"])
                with st.expander(
                    f"{gate_icon} `{item['trace_id']}`  ·  {item['category']}  ·  "
                    f"{item['question'][:60]}"
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.caption(f"Generated: {item['generated_at'][5:16]}")
                    c2.caption(f"Gate: {'PASS' if item['gate_passed'] else 'FAIL'}")
                    c3.caption(f"Human: {human['overall'] if human else '—'}")
                    c4.caption(f"LLM: {llm['overall'] if llm else '—'}")

                    checks = item.get("gate_results", {}).get("checks", {})
                    if checks:
                        for cname, cres in checks.items():
                            ci = "✅" if cres.get("passed") else "❌"
                            st.caption(f"  {ci} {cname}: {cres.get('reason', '')}")

                    if human:
                        dim_flags = "  ".join(
                            f"{'✓' if human[d] else '✗'}{DIM_SHORT[d]}" for d in DIM_KEYS
                        )
                        st.caption(f"Human dims: {dim_flags}")
                        if human.get("notes"):
                            st.caption(f"Notes: {human['notes']}")

                    if llm:
                        dim_flags = "  ".join(
                            f"{'✓' if llm[d] else '✗'}{DIM_SHORT[d]}" for d in DIM_KEYS
                        )
                        st.caption(f"LLM dims:   {dim_flags}")
                        if llm.get("reasoning"):
                            st.caption(f"Reasoning: {llm['reasoning']}")

    elif log_view == "Log File":
        log_path = Path(__file__).parent / "llm_judge.log"
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            st.caption(f"{len(lines)} log lines in {log_path.name}")
            filt_status = st.selectbox("Filter by status", ["all", "ok", "error"], key="log_status")
            shown = [l for l in lines if filt_status == "all" or f'"status": "{filt_status}"' in l]
            st.text_area("Log (last 300 lines)", "\n".join(shown[-300:]), height=500)
        else:
            st.info(f"No log file found at {log_path}")

    else:  # Trace Detail
        if not selected_run_id:
            st.warning("Select a run.")
        else:
            items = storage.get_items(selected_run_id)
            if not items:
                st.info("No items in this run.")
            else:
                trace_ids = [i["trace_id"] for i in items]
                sel_trace = st.selectbox("trace_id", trace_ids, key="trace_sel")
                item      = next((i for i in items if i["trace_id"] == sel_trace), None)

                if item:
                    st.subheader("Item record")
                    st.json({k: v for k, v in item.items() if k not in ("run_id", "raw_response")})

                    llm_labels_run = storage.get_llm_labels(selected_run_id)
                    llm = next((l for l in llm_labels_run if l["trace_id"] == sel_trace), None)
                    if llm:
                        st.subheader("LLM Judge record")
                        st.json({k: v for k, v in llm.items() if k not in ("id", "item_id", "raw_response")})
                        if llm.get("raw_response"):
                            with st.expander("Raw LLM response"):
                                st.code(llm["raw_response"])

                    human = storage.get_human_label_for_item(item["id"])
                    if human:
                        st.subheader("Human label record")
                        st.json({k: v for k, v in human.items() if k != "id"})


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.header("Report")

    if not selected_run_id:
        st.warning("Select a run to generate a report.")
    else:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        _run   = storage.get_run(selected_run_id)
        _items = storage.get_items(selected_run_id)
        _gated = [i for i in _items if i["gate_passed"]]
        _human_lbls = storage.get_human_labels(selected_run_id)
        _llm_lbls   = storage.get_llm_labels(selected_run_id)
        _iterations = storage.get_iterations(selected_run_id)

        # Group LLM labels by prompt name
        _llm_by_prompt: dict[str, list[dict]] = defaultdict(list)
        for _l in _llm_lbls:
            _llm_by_prompt[_l["judge_prompt_name"]].append(_l)

        _item_by_id    = {i["id"]: i for i in _items}
        _item_by_trace = {i["trace_id"]: i for i in _items}
        _human_by_item = {l["item_id"]: l for l in _human_lbls}

        # Project thresholds from spec
        _DIM_TARGETS = {
            "D1_answer_completeness":   0.85,
            "D2_safety_specificity":    0.90,
            "D3_tool_realism":          0.95,
            "D4_scope_appropriateness": 0.95,
            "D5_context_clarity":       0.90,
            "D6_tip_usefulness":        0.85,
        }
        _OVERALL_TARGET    = 0.80
        _AGREEMENT_TARGET  = 0.80
        _SCHEMA_TARGET     = 0.95
        _IMPROVEMENT_TARGET = 0.80

        # Pre-compute per-prompt aggregate stats (used across sections)
        _prompt_stats: dict[str, dict] = {}
        for _pname, _lbls in _llm_by_prompt.items():
            _n = len(_lbls)
            if not _n:
                continue
            _n_res = sum(1 for l in _lbls if l["overall"] == "RESOLVED")
            _prompt_stats[_pname] = {
                "n":            _n,
                "resolved_pct": _n_res / _n,
                "dim_rates":    {d: sum(1 for l in _lbls if l.get(d)) / _n for d in DIM_KEYS},
            }

        _sc_rows: list[dict] = []  # populated in scorecard section, used by export
        _improvement: float = 0.0  # set in section 6 if multiple prompts exist

        # ── 1. Run Overview ───────────────────────────────────────────────────
        st.subheader("1 · Run Overview")
        with st.container(border=True):
            oc1, oc2, oc3, oc4, oc5 = st.columns(5)
            oc1.metric("Generated",     len(_items))
            oc2.metric("Gate passed",   len(_gated))
            oc3.metric("Human labeled", len(_human_lbls))
            oc4.metric("Prompts judged", len(_prompt_stats))
            oc5.metric("Iterations",    len(_iterations))
            st.caption(
                f"**Run:** `{selected_run_id[:8]}`  ·  "
                f"**Created:** {(_run.get('created_at') or '')[:16]}  ·  "
                f"**Generator:** {_run.get('generator_variant')}  ·  "
                f"**Status:** {_run.get('status')}"
            )
            if _run.get("notes"):
                st.caption(f"**Notes:** {_run['notes']}")

        # ── 2. Data Quality Gate (Step 2) ─────────────────────────────────────
        st.subheader("2 · Data Quality Gate (Step 2)")
        with st.container(border=True):
            _n_total  = len(_items)
            _n_passed = len(_gated)
            _gate_rate = _n_passed / _n_total if _n_total else 0
            _gate_ok   = _gate_rate >= _SCHEMA_TARGET

            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Gate pass rate", f"{_gate_rate*100:.0f}%",
                       help=f"Target ≥{_SCHEMA_TARGET*100:.0f}%")
            gc1.caption("✅ Meets target" if _gate_ok else "❌ Below target")
            gc2.metric("Items passed",  _n_passed)
            gc3.metric("Items dropped", _n_total - _n_passed)

            if _gated:
                _cat_counts = Counter(i["category"] for i in _gated)
                _total_gated = len(_gated)
                _cats_sorted = sorted(_cat_counts.keys())

                st.markdown("**Category distribution vs. benchmark (20% per category)**")
                _fig_dist, _ax_dist = plt.subplots(figsize=(10, 3.5))
                _gen_pcts   = [_cat_counts.get(c, 0) / _total_gated * 100 for c in _cats_sorted]
                _bench_pcts = [20.0] * len(_cats_sorted)
                _x = np.arange(len(_cats_sorted))
                _w = 0.35
                _b1 = _ax_dist.bar(_x - _w/2, _gen_pcts,  _w, label="Generated", color="#1565C0", alpha=0.85)
                _b2 = _ax_dist.bar(_x + _w/2, _bench_pcts, _w, label="Benchmark (20%)", color="#43A047", alpha=0.5)
                _ax_dist.axhline(18, color="orange", linestyle="--", linewidth=1, label="18% floor (tolerance)")
                _ax_dist.set_xticks(_x)
                _ax_dist.set_xticklabels([c[:22] for c in _cats_sorted], fontsize=8)
                _ax_dist.set_ylim(0, 40)
                _ax_dist.set_ylabel("% of items")
                _ax_dist.set_title("Category distribution — generated vs. benchmark")
                _ax_dist.legend(fontsize=8)
                for _bar, _val in zip(_b1, _gen_pcts):
                    _ax_dist.text(_bar.get_x() + _bar.get_width()/2, _val + 0.5,
                                  f"{_val:.0f}%", ha="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(_fig_dist); plt.close(_fig_dist)

                _dist_rows = []
                for _cat in _cats_sorted:
                    _cnt = _cat_counts.get(_cat, 0)
                    _pct = _cnt / _total_gated * 100
                    _dist_rows.append({
                        "Category": _cat,
                        "Count":    _cnt,
                        "Pct":      f"{_pct:.0f}%",
                        "≥18% (benchmark-aligned)": "✅" if _pct >= 18 else "❌",
                    })
                st.table(_dist_rows)

        # ── 3. Human Labels (Step 3) ──────────────────────────────────────────
        st.subheader("3 · Human Labels (Step 3)")
        with st.container(border=True):
            if not _human_lbls:
                st.info("No human labels yet — use the Label tab.")
            else:
                _n_h = len(_human_lbls)
                _n_h_res = sum(1 for l in _human_lbls if l["overall"] == "RESOLVED")
                h1, h2 = st.columns(2)
                h1.metric("Items labeled",   _n_h)
                h2.metric("Human RESOLVED",  f"{_n_h_res}/{_n_h}  ({_n_h_res/_n_h*100:.0f}%)")

                _h_rows = []
                for _dim in DIM_KEYS:
                    _rate = sum(1 for l in _human_lbls if l.get(_dim)) / _n_h
                    _tgt  = _DIM_TARGETS[_dim]
                    _h_rows.append({
                        "Dimension":  f"{DIM_SHORT[_dim]} — {_dim.replace('_',' ').title()}",
                        "Pass rate":  f"{_rate*100:.0f}%",
                        "Target":     f"≥{_tgt*100:.0f}%",
                        "Status":     "✅" if _rate >= _tgt else "❌",
                    })
                st.table(_h_rows)

        # ── 4. LLM Judge Results — All Prompts (Step 4) ───────────────────────
        st.subheader("4 · LLM Judge Results — All Prompts (Step 4)")

        if not _prompt_stats:
            st.info("No LLM judge results yet — use the Judge tab.")
        else:
            # Cross-prompt summary table
            _sum_rows = []
            for _pname, _stats in _prompt_stats.items():
                _row = {
                    "Prompt":     PROMPT_DISPLAY.get(_pname, _pname),
                    "Judged":     _stats["n"],
                    "RESOLVED %": f"{_stats['resolved_pct']*100:.0f}%",
                    "≥80%":       "✅" if _stats["resolved_pct"] >= _OVERALL_TARGET else "❌",
                }
                for _dim in DIM_KEYS:
                    _r = _stats["dim_rates"][_dim]
                    _t = _DIM_TARGETS[_dim]
                    _row[DIM_SHORT[_dim]] = f"{_r*100:.0f}% {'✅' if _r >= _t else '❌'}"
                _sum_rows.append(_row)
            st.dataframe(_sum_rows, use_container_width=True, hide_index=True)

            # Per-prompt detail: bar chart with per-dim thresholds
            for _pname, _stats in _prompt_stats.items():
                with st.expander(
                    f"{PROMPT_DISPLAY.get(_pname, _pname)}  ·  "
                    f"RESOLVED {_stats['resolved_pct']*100:.0f}%  ·  n={_stats['n']}",
                    expanded=len(_prompt_stats) == 1,
                ):
                    _rates = [_stats["dim_rates"][d] for d in DIM_KEYS]
                    _colors = [
                        "#2E7D32" if _rates[i] >= _DIM_TARGETS[DIM_KEYS[i]] else "#C62828"
                        for i in range(len(DIM_KEYS))
                    ]
                    _fig_p, _ax_p = plt.subplots(figsize=(10, 4))
                    _bars_p = _ax_p.bar(range(len(DIM_KEYS)), _rates, 0.6, color=_colors, alpha=0.85)
                    for _xi, _dim in enumerate(DIM_KEYS):
                        _t = _DIM_TARGETS[_dim]
                        _ax_p.plot([_xi - 0.3, _xi + 0.3], [_t, _t],
                                   color="black", linewidth=1.5, linestyle="--")
                    _ax_p.set_xticks(range(len(DIM_KEYS)))
                    _ax_p.set_xticklabels([DIM_LABEL[d] for d in DIM_KEYS], fontsize=8)
                    _ax_p.set_ylim(0, 1.18)
                    _ax_p.set_ylabel("Pass rate")
                    _ax_p.set_title(f"Per-dimension pass rate — {PROMPT_DISPLAY.get(_pname, _pname)}")
                    _ax_p.legend(handles=[
                        mpatches.Patch(color="#2E7D32", label="Meets target"),
                        mpatches.Patch(color="#C62828", label="Below target"),
                    ], fontsize=9)
                    for _bar, _val in zip(_bars_p, _rates):
                        _ax_p.text(_bar.get_x() + _bar.get_width()/2, _val + 0.02,
                                   f"{_val:.0%}", ha="center", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(_fig_p); plt.close(_fig_p)

                    # Change description
                    _change = PROMPT_CHANGES.get(_pname, "—")
                    st.caption(f"**Changes:** {_change}")

        # ── 5. Analysis (Step 5) ──────────────────────────────────────────────
        st.subheader("5 · Analysis (Step 5)")

        if _prompt_stats and _gated:
            _cats_all = sorted({i["category"] for i in _gated})

            # 5a — Segment heatmap per judge prompt
            for _pname, _lbls in _llm_by_prompt.items():
                if not _lbls:
                    continue
                _lbl_by_trace = {l["trace_id"]: l for l in _lbls}
                _heat = np.zeros((len(_cats_all), len(DIM_KEYS)))
                for _ci, _cat in enumerate(_cats_all):
                    _cat_items = [i for i in _gated if i["category"] == _cat]
                    for _di, _dim in enumerate(DIM_KEYS):
                        _cat_lbls = [_lbl_by_trace[i["trace_id"]]
                                     for i in _cat_items if i["trace_id"] in _lbl_by_trace]
                        _heat[_ci, _di] = (
                            sum(bool(l.get(_dim)) for l in _cat_lbls) / len(_cat_lbls)
                            if _cat_lbls else 0.0
                        )

                st.markdown(f"**Segment heatmap — {PROMPT_DISPLAY.get(_pname, _pname)}**")
                _fig_h, _ax_h = plt.subplots(figsize=(10, max(3, len(_cats_all) * 0.9)))
                _im = _ax_h.imshow(_heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
                _ax_h.set_xticks(range(len(DIM_KEYS)))
                _ax_h.set_xticklabels([DIM_LABEL[d] for d in DIM_KEYS], fontsize=8)
                _ax_h.set_yticks(range(len(_cats_all)))
                _ax_h.set_yticklabels(_cats_all, fontsize=9)
                plt.colorbar(_im, ax=_ax_h)
                for _ci in range(len(_cats_all)):
                    for _di in range(len(DIM_KEYS)):
                        _ax_h.text(_di, _ci, f"{_heat[_ci, _di]:.0%}",
                                   ha="center", va="center", fontsize=9)
                _ax_h.set_title(f"Pass rate: category × dimension — {PROMPT_DISPLAY.get(_pname, _pname)}")
                plt.tight_layout()
                st.pyplot(_fig_h); plt.close(_fig_h)

            # 5b — Human / LLM agreement
            if _human_lbls:
                st.markdown("**Human / LLM agreement per dimension (paired items)**")
                for _pname, _lbls in _llm_by_prompt.items():
                    if not _lbls:
                        continue
                    _lbl_by_item = {l["item_id"]: l for l in _lbls}
                    _paired = [
                        (_human_by_item[_iid], _lbl_by_item[_iid])
                        for _iid in _human_by_item if _iid in _lbl_by_item
                    ]
                    if not _paired:
                        continue
                    _n_paired = len(_paired)
                    _agree_rates = [
                        sum(1 for _h, _l in _paired if bool(_h.get(_d)) == bool(_l.get(_d))) / _n_paired
                        for _d in DIM_KEYS
                    ]
                    _colors_a = ["#2E7D32" if r >= _AGREEMENT_TARGET else "#C62828" for r in _agree_rates]
                    _fig_a, _ax_a = plt.subplots(figsize=(10, 3.5))
                    _bars_a = _ax_a.bar([DIM_LABEL[d] for d in DIM_KEYS], _agree_rates,
                                        color=_colors_a, alpha=0.85)
                    _ax_a.axhline(_AGREEMENT_TARGET, color="grey", linestyle="--", linewidth=1,
                                  label=f"≥{_AGREEMENT_TARGET*100:.0f}% target")
                    _ax_a.set_ylim(0, 1.18)
                    _ax_a.set_ylabel("Agreement")
                    _ax_a.set_title(
                        f"Human / LLM agreement — {PROMPT_DISPLAY.get(_pname, _pname)}"
                        f"  (n={_n_paired} paired)"
                    )
                    _ax_a.tick_params(axis="x", labelsize=8)
                    _ax_a.legend()
                    for _bar, _val in zip(_bars_a, _agree_rates):
                        _ax_a.text(_bar.get_x() + _bar.get_width()/2, _val + 0.02,
                                   f"{_val:.0%}", ha="center", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(_fig_a); plt.close(_fig_a)

        # ── 6. Iteration Log + Before/After (Step 6) ─────────────────────────
        st.subheader("6 · Iteration Log & Before/After (Step 6)")
        with st.container(border=True):
            if not _iterations:
                st.info("No iterations logged yet.")
            else:
                _it_rows = []
                for _idx, _it in enumerate(_iterations, 1):
                    _mb = _it.get("metrics_before") or {}
                    _ma = _it.get("metrics_after")  or {}
                    _it_rows.append({
                        "#":                _idx,
                        "Phase":            _it["phase"],
                        "Component":        PROMPT_DISPLAY.get(_it["component"], _it["component"]),
                        "Hypothesis":       (_it.get("hypothesis") or "")[:90],
                        "Agreement before": f"{_mb.get('overall', 0):.0%}" if _mb else "—",
                        "Agreement after":  f"{_ma.get('overall', 0):.0%}"
                                            if _ma and not _ma.get("skipped") else "—",
                        "Decision":         (_it.get("decision") or "")[:50],
                    })
                st.dataframe(_it_rows, use_container_width=True, hide_index=True)

            # Before/after RESOLVED % across all judge prompts on this run
            if len(_prompt_stats) > 1:
                st.markdown("**RESOLVED % progression across all prompts judged on this run**")
                _p_names  = list(_prompt_stats.keys())
                _p_labels = [PROMPT_DISPLAY.get(p, p) for p in _p_names]
                _p_rates  = [_prompt_stats[p]["resolved_pct"] for p in _p_names]

                _base_pct = (
                    _prompt_stats.get("judge_baseline", {}).get("resolved_pct")
                    or _p_rates[0]
                )
                _best_pct = max(_p_rates)
                _base_fail = 1 - _base_pct
                _best_fail = 1 - _best_pct
                _improvement = (_base_fail - _best_fail) / _base_fail if _base_fail else 0

                _ic1, _ic2, _ic3 = st.columns(3)
                _ic1.metric("Baseline fail rate", f"{_base_fail*100:.0f}%")
                _ic2.metric("Best fail rate",      f"{_best_fail*100:.0f}%")
                _impr_ok = _improvement >= _IMPROVEMENT_TARGET
                _ic3.metric("Improvement ratio",   f"{_improvement*100:.0f}%",
                            help=f"Target ≥{_IMPROVEMENT_TARGET*100:.0f}%")
                _ic3.caption("✅ Meets target" if _impr_ok else "❌ Below target")

                _fig_ba, _ax_ba = plt.subplots(figsize=(max(7, len(_p_names) * 1.8), 4))
                _colors_ba = ["#B0BEC5" if p == "judge_baseline" else "#1565C0" for p in _p_names]
                _bars_ba   = _ax_ba.bar(_p_labels, _p_rates, color=_colors_ba, alpha=0.85)
                _ax_ba.axhline(_OVERALL_TARGET, color="grey", linestyle="--", linewidth=1,
                               label=f"≥{_OVERALL_TARGET*100:.0f}% target")
                _ax_ba.set_ylim(0, 1.18)
                _ax_ba.set_ylabel("RESOLVED %")
                _ax_ba.set_title("RESOLVED rate — baseline vs. prompt iterations")
                _ax_ba.tick_params(axis="x", rotation=15, labelsize=8)
                _ax_ba.legend()
                for _bar, _val in zip(_bars_ba, _p_rates):
                    _ax_ba.text(_bar.get_x() + _bar.get_width()/2, _val + 0.02,
                                f"{_val:.0%}", ha="center", fontsize=9)
                plt.tight_layout()
                st.pyplot(_fig_ba); plt.close(_fig_ba)

        # ── 7. Quality Scorecard ──────────────────────────────────────────────
        st.subheader("7 · Quality Scorecard")
        with st.container(border=True):
            st.caption(
                "All project thresholds evaluated against the best-performing judge prompt on this run."
            )

            _best_pname = (
                max(_prompt_stats, key=lambda p: _prompt_stats[p]["resolved_pct"])
                if _prompt_stats else None
            )
            _best_stats = _prompt_stats.get(_best_pname, {})

            # Schema / gate
            _sc_rows.append({
                "Metric":  "Schema validation pass rate (Step 2)",
                "Target":  f"≥{_SCHEMA_TARGET*100:.0f}%",
                "Actual":  f"{_gate_rate*100:.0f}%",
                "Status":  "✅" if _gate_ok else "❌",
            })

            # Dataset size
            _sc_rows.append({
                "Metric": "Items generated",
                "Target": "≥50",
                "Actual": str(len(_items)),
                "Status": "✅" if len(_items) >= 50 else "❌",
            })
            _sc_rows.append({
                "Metric": "Human labels collected (Step 3)",
                "Target": "≥20",
                "Actual": str(len(_human_lbls)),
                "Status": "✅" if len(_human_lbls) >= 20 else "❌",
            })

            # Per-dim pass rates
            if _best_stats:
                for _dim in DIM_KEYS:
                    _rate = _best_stats["dim_rates"][_dim]
                    _tgt  = _DIM_TARGETS[_dim]
                    _sc_rows.append({
                        "Metric":  f"{DIM_SHORT[_dim]} {_dim.replace('_',' ').title()} pass rate",
                        "Target":  f"≥{_tgt*100:.0f}%",
                        "Actual":  f"{_rate*100:.0f}%",
                        "Status":  "✅" if _rate >= _tgt else "❌",
                    })

                # Overall RESOLVED
                _res_rate = _best_stats["resolved_pct"]
                _sc_rows.append({
                    "Metric": "Overall RESOLVED (all 6 dims pass)",
                    "Target": f"≥{_OVERALL_TARGET*100:.0f}%",
                    "Actual": f"{_res_rate*100:.0f}%",
                    "Status": "✅" if _res_rate >= _OVERALL_TARGET else "❌",
                })

            # Human / LLM agreement — minimum across dims
            if _human_lbls and _best_pname and _best_pname in _llm_by_prompt:
                _lbl_by_item2 = {l["item_id"]: l for l in _llm_by_prompt[_best_pname]}
                _paired2      = [
                    (_human_by_item[_iid], _lbl_by_item2[_iid])
                    for _iid in _human_by_item if _iid in _lbl_by_item2
                ]
                if _paired2:
                    _n_p2 = len(_paired2)
                    _agree_by_dim = {
                        _d: sum(1 for _h, _l in _paired2
                                if bool(_h.get(_d)) == bool(_l.get(_d))) / _n_p2
                        for _d in DIM_KEYS
                    }
                    _min_dim  = min(_agree_by_dim, key=_agree_by_dim.get)
                    _min_rate = _agree_by_dim[_min_dim]
                    _sc_rows.append({
                        "Metric": f"Human/LLM agreement — worst dim ({DIM_SHORT[_min_dim]})",
                        "Target": f"≥{_AGREEMENT_TARGET*100:.0f}% per dim",
                        "Actual": f"{_min_rate*100:.0f}%",
                        "Status": "✅" if _min_rate >= _AGREEMENT_TARGET else "❌",
                    })

            # Improvement ratio (only if multiple prompts exist)
            if len(_prompt_stats) > 1:
                _sc_rows.append({
                    "Metric": "Improvement ratio (Phase B)",
                    "Target": f"≥{_IMPROVEMENT_TARGET*100:.0f}% failure reduction",
                    "Actual": f"{_improvement*100:.0f}%",
                    "Status": "✅" if _improvement >= _IMPROVEMENT_TARGET else "❌",
                })

            st.dataframe(_sc_rows, use_container_width=True, hide_index=True)

            _passed = sum(1 for r in _sc_rows if r["Status"] == "✅")
            _total  = len(_sc_rows)
            if _total:
                if _passed == _total:
                    st.success(f"All {_total} criteria met — run is submission-ready.")
                else:
                    st.warning(f"{_passed}/{_total} criteria met — {_total - _passed} outstanding.")

        # ── Export ────────────────────────────────────────────────────────────
        st.subheader("Export")
        _report_data = {
            "run_id":            selected_run_id,
            "created_at":        _run.get("created_at"),
            "generator_variant": _run.get("generator_variant"),
            "notes":             _run.get("notes"),
            "gate": {
                "total":   len(_items),
                "passed":  len(_gated),
                "rate":    round(_gate_rate, 3),
            },
            "human_labels_count": len(_human_lbls),
            "judge_summaries": {
                _pname: {
                    "n_judged":     _s["n"],
                    "resolved_pct": round(_s["resolved_pct"], 3),
                    "dim_rates":    {d: round(_s["dim_rates"][d], 3) for d in DIM_KEYS},
                }
                for _pname, _s in _prompt_stats.items()
            },
            "iterations": [
                {k: v for k, v in _it.items() if k not in ("prompt_before", "prompt_after")}
                for _it in _iterations
            ],
            "scorecard": _sc_rows,
        }
        st.download_button(
            "Download report JSON",
            data=json.dumps(_report_data, indent=2, default=str),
            file_name=f"report_{selected_run_id[:8]}.json",
            mime="application/json",
        )
