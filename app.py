"""Exercise 5 — run with `uv run streamlit run app.py`."""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph


def snapshot_result(snapshot) -> dict:
    """Read saved state without executing nodes or repeating side effects."""
    result = dict(snapshot.values)
    interrupts = tuple(item for task in snapshot.tasks for item in task.interrupts)
    if interrupts:
        result["__interrupt__"] = interrupts
    return result


async def load_session(thread_id: str) -> dict:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.values:
            raise ValueError("No saved review found for this thread")
        return snapshot_result(snapshot)


async def recent_sessions() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute("""
            SELECT thread_id, pr_url,
                   CASE MAX(CASE risk_level
                       WHEN 'high' THEN 2 WHEN 'med' THEN 1 ELSE 0 END)
                       WHEN 2 THEN 'high' WHEN 1 THEN 'med' ELSE 'low'
                   END AS worst_risk,
                   MAX(timestamp) AS last_event
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
        """) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Start or resume once, opening a fresh connection for each Streamlit run."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = await app.aget_state(cfg)
        if snapshot.values:
            if snapshot.values.get("pr_url") != pr_url:
                raise ValueError("This thread belongs to a different PR")
            saved = snapshot_result(snapshot)
            # A repeated click must not start a completed review or post twice.
            if not snapshot.next:
                return saved
            if resume_value is None and "__interrupt__" in saved:
                return saved
        elif resume_value is not None:
            raise ValueError("No saved review is waiting for this response")

        if resume_value is not None:
            return await app.ainvoke(Command(resume=resume_value), cfg)
        if snapshot.values:
            return await app.ainvoke(None, cfg)
        return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


def store_result(result: dict) -> None:
    st.session_state.pr_url = result.get("pr_url", st.session_state.pr_url)
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def open_session(thread_id: str) -> None:
    result = asyncio.run(load_session(thread_id))
    st.session_state.thread_id = thread_id
    st.query_params["thread"] = thread_id
    store_result(result)


def render_approval_card(payload: dict) -> dict | None:
    st.subheader(f"Approval requested — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** `{comment['file']}:{comment.get('line') or '?'}` — {comment['body']}"
        )
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    thread = st.session_state.thread_id
    feedback = st.text_input("Feedback (optional)", key=f"approval_feedback_{thread}")
    st.caption("Approve posts the review to GitHub. Reject and Edit skip posting in this exercise.")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks:\n" + "\n".join(f"- {risk}" for risk in payload["risk_factors"]))
    st.markdown(payload["summary"])
    st.caption("Submit answers refines the review and posts it to GitHub.")
    thread = st.session_state.thread_id
    with st.form(f"escalation_{thread}"):
        answers = {
            question: st.text_input(question, key=f"answer_{thread}_{index}").strip()
            for index, question in enumerate(payload["questions"])
        }
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        if not answers or not all(answers.values()):
            st.warning("Please answer every question before submitting.")
            return None
        return answers
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action")
    if action in {"committed", "auto_committed", "auto_approved", "committed_after_escalation"}:
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        st.link_button("View comments on GitHub", st.session_state.pr_url + "#discussion_bucket")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif action and "failed" in action:
        st.error("The comment could not be posted. See the audit trail for this session.")
    else:
        st.info("This review has unfinished work. Continue to resume its saved state.")
        if st.button("Continue review"):
            store_result(asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id)))
            st.rerun()
    analysis = final.get("analysis")
    if analysis is not None:
        st.metric("Confidence", f"{analysis.confidence:.0%}")
        st.markdown(analysis.summary)
        st.caption(analysis.confidence_reasoning)
        with st.expander("Review comments"):
            for comment in analysis.comments:
                st.markdown(f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` — {comment.body}")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="HITL PR Review", layout="wide")
    for key, value in {"thread_id": None, "pr_url": "", "interrupt_payload": None, "final": None}.items():
        st.session_state.setdefault(key, value)
    st.title("HITL PR Review Agent")

    # A URL survives a browser refresh, while Streamlit session_state does not.
    saved_thread = st.query_params.get("thread")
    if saved_thread and saved_thread != st.session_state.thread_id:
        try:
            open_session(saved_thread)
        except Exception as exc:
            st.error(f"Unable to restore review: {exc}")

    with st.sidebar:
        st.header("Recent sessions")
        try:
            sessions = asyncio.run(recent_sessions())
            if sessions:
                selection = st.dataframe(
                    sessions, hide_index=True, on_select="rerun", selection_mode="single-row",
                    key=f"recent_sessions_{st.session_state.thread_id}",
                    column_config={"pr_url": st.column_config.LinkColumn("PR URL")},
                )
                if selection.selection.rows:
                    selected = sessions[selection.selection.rows[0]]
                    if selected["thread_id"] != st.session_state.thread_id:
                        open_session(selected["thread_id"])
                        st.rerun()
            else:
                st.caption("No review sessions yet.")
        except Exception as exc:
            st.error(f"Unable to load sessions: {exc}")

    with st.form("start"):
        pr_url = st.text_input(
            "PR URL", value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
        submitted = st.form_submit_button("Run review")

    if submitted:
        try:
            pr_url = pr_url.strip()
            owner, repo, number = parse_pr_url(pr_url)
            pr_url = f"https://github.com/{owner}/{repo}/pull/{number}"
            thread = str(uuid.uuid4())
            st.session_state.update(thread_id=thread, pr_url=pr_url, interrupt_payload=None, final=None)
            st.query_params["thread"] = thread
            with st.spinner("Fetching PR + asking the LLM..."):
                store_result(asyncio.run(run_graph(pr_url, thread)))
            st.rerun()
        except Exception as exc:
            st.error(f"Unable to run review: {exc}")

    payload = st.session_state.interrupt_payload
    if payload is not None:
        kind = payload["kind"]
        if kind == "approval_request":
            answer = render_approval_card(payload)
        elif kind == "escalation":
            answer = render_escalation_card(payload)
        else:
            st.error(f"Unknown review request: {kind}")
            answer = None
        if answer is not None:
            try:
                with st.spinner("Resuming..."):
                    result = asyncio.run(run_graph(
                        st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                    ))
                store_result(result)
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to resume review: {exc}")

    if st.session_state.final is not None:
        render_final(st.session_state.final)
    if st.session_state.thread_id:
        st.caption(f"thread_id = {st.session_state.thread_id}")
        st.code(f"uv run python -m audit.replay --thread {st.session_state.thread_id}", language="shell")


if __name__ == "__main__":
    main()
