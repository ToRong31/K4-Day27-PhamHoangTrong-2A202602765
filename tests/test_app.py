"""Exercise 5 UI checks with real graphs/SQLite and mocked GitHub/LLM."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

import app
from common.db import replay_events
from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_4_audit as ex


APP_FILE = str(Path(__file__).resolve().parents[1] / "app.py")
PR_URL = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {
            "HITL_DB_PATH": os.path.join(folder, "audit.db"),
            "GITHUB_USER": "test-reviewer",
        }))
        self.fetch = self.enterContext(patch.object(ex, "fetch_pr", return_value=SimpleNamespace(
            title="Test PR", diff="diff --git a/auth.py b/auth.py", files_changed=["auth.py"], head_sha="abcdef12",
        )))
        self.factory = self.enterContext(patch.object(ex, "get_llm"))
        self.llm = self.factory.return_value.with_structured_output.return_value
        self.llm.ainvoke = AsyncMock()
        self.post = self.enterContext(patch.object(ex, "post_review_comment"))

    def analysis(self, confidence=0.65, **kwargs):
        return PRAnalysis(
            summary="Review summary", confidence=confidence, confidence_reasoning="Missing migration context",
            comments=[ReviewComment(file="auth.py", line=8, severity="issue", body="Check existing users")],
            **kwargs,
        )

    def button(self, page, label):
        return next(button for button in page.button if button.label == label)

    def assert_clean(self, page):
        self.assertEqual(len(page.exception), 0, [e.message for e in page.exception])

    def start(self, analysis):
        self.llm.ainvoke.return_value = analysis
        page = AppTest.from_file(APP_FILE, default_timeout=20).run()
        self.assert_clean(page)
        page.text_input[0].set_value(PR_URL)
        self.button(page, "Run review").click().run()
        self.assert_clean(page)
        return page

    def refresh(self, thread):
        page = AppTest.from_file(APP_FILE, default_timeout=20)
        page.query_params["thread"] = thread
        page.run()
        self.assert_clean(page)
        return page

    def test_approve_and_refresh_do_not_post_twice(self):
        page = self.start(self.analysis())
        self.assertIn("Approval requested", page.subheader[0].value)
        self.assertEqual(page.expander[0].label, "Diff")
        self.post.assert_not_called()
        thread = page.session_state.thread_id
        page = self.refresh(thread)
        self.fetch.assert_called_once()
        self.assertEqual(self.llm.ainvoke.await_count, 1)
        page.text_input(key=f"approval_feedback_{thread}").set_value("Migration checked")
        self.button(page, "Approve").click().run()
        self.assert_clean(page)
        self.assertEqual(page.session_state.final["final_action"], "committed")
        self.assertTrue(page.success)
        self.assertEqual(page.metric[0].value, "65%")
        self.assertTrue(page.get("link_button"))
        self.post.assert_called_once()
        self.assertIn("Migration checked", self.post.call_args.args[1])
        page.run()
        self.refresh(thread)
        asyncio.run(app.run_graph(PR_URL, thread, {"choice": "approve"}))
        self.post.assert_called_once()
        rows = asyncio.run(replay_events(thread))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[-1]["reviewer_id"], "test-reviewer")
        self.assertEqual(rows[-1]["decision"], "approve")
        sessions = asyncio.run(app.recent_sessions())
        self.assertEqual(sessions[0]["thread_id"], thread)
        self.assertEqual(sessions[0]["worst_risk"], "med")
        self.assertEqual(set(page.dataframe[0].value.columns), {"thread_id", "pr_url", "worst_risk", "last_event"})
        with self.assertRaises(ValueError):
            asyncio.run(app.run_graph("https://github.com/example/other/pull/2", thread))

    def test_reject(self):
        page = self.start(self.analysis())
        self.button(page, "Reject").click().run()
        self.assert_clean(page)
        self.assertEqual(page.session_state.final["final_action"], "rejected")
        self.assertTrue(page.warning)
        self.post.assert_not_called()

    def test_sidebar_selection_restores_another_thread(self):
        older = self.start(self.analysis())
        older_thread = older.session_state.thread_id
        page = self.start(self.analysis())
        current = page.session_state.thread_id
        self.assertNotEqual(current, older_thread)
        rows = page.dataframe[0].value
        selected = rows.index[rows["thread_id"] == older_thread][0]
        page.session_state[f"recent_sessions_{current}"] = {
            "selection": {"rows": [int(selected)], "columns": [], "cells": []},
        }
        page.run()
        self.assert_clean(page)
        self.assertEqual(page.session_state.thread_id, older_thread)
        self.assertEqual(page.query_params["thread"], older_thread)
        self.assertIn("Approval requested", page.subheader[0].value)
        self.assertEqual(self.fetch.call_count, 2)
        self.post.assert_not_called()

    def test_invalid_url_does_not_start_a_review(self):
        page = AppTest.from_file(APP_FILE, default_timeout=20).run()
        page.text_input[0].set_value("not a PR URL")
        self.button(page, "Run review").click().run()
        self.assert_clean(page)
        self.assertTrue(page.error)
        self.fetch.assert_not_called()
        self.post.assert_not_called()

    def test_edit_skips_posting(self):
        page = self.start(self.analysis())
        thread = page.session_state.thread_id
        page.text_input(key=f"approval_feedback_{thread}").set_value("Please revise")
        self.button(page, "Edit").click().run()
        self.assert_clean(page)
        self.assertEqual(page.session_state.final["human_choice"], "edit")
        self.assertEqual(page.session_state.final["human_feedback"], "Please revise")
        self.assertEqual(page.session_state.final["final_action"], "rejected")
        self.post.assert_not_called()

    def test_escalation_form_and_refined_review(self):
        questions = ["auth.py: Why MD5?", "storage.py: Is SQL parameterized?"]
        initial = self.analysis(0.5, escalation_questions=questions, risk_factors=["SQL injection", "Weak password hashing"])
        refined = self.analysis(0.7)
        self.llm.ainvoke.side_effect = [initial, refined]
        page = self.start(initial)
        self.assertIn("Strong escalation", page.subheader[0].value)
        self.assertIn("SQL injection", page.error[0].value)
        thread = page.session_state.thread_id
        self.button(page, "Submit answers").click().run()
        self.assert_clean(page)
        self.post.assert_not_called()
        self.assertTrue(page.warning)
        page.text_input(key=f"answer_{thread}_0").set_value("Legacy implementation; needs replacement")
        page.text_input(key=f"answer_{thread}_1").set_value("Not yet; parameterize before merging")
        self.button(page, "Submit answers").click().run()
        self.assert_clean(page)
        self.assertTrue(page.success)
        self.assertEqual(page.metric[0].value, "70%")
        self.post.assert_called_once()
        body = self.post.call_args.args[1]
        self.assertIn("Reviewer answered escalation questions", body)
        self.assertIn("Not yet; parameterize before merging", body)
        rows = asyncio.run(replay_events(thread))
        self.assertEqual(rows[-2]["action"], "synthesize")
        self.assertEqual(rows[-2]["confidence"], 0.7)
        self.assertEqual(asyncio.run(app.recent_sessions())[0]["worst_risk"], "high")
        self.refresh(thread)
        self.post.assert_called_once()

    def test_auto_success(self):
        page = self.start(self.analysis(0.9))
        self.assertTrue(page.success)
        self.assertEqual(page.metric[0].value, "90%")
        self.assertTrue(page.get("link_button"))
        self.assertFalse(page.subheader)
        self.post.assert_called_once()

    def test_auto_post_failure_is_an_error(self):
        self.post.side_effect = RuntimeError("GitHub unavailable")
        page = self.start(self.analysis(0.9))
        self.assertFalse(page.success)
        self.assertTrue(page.error)
        self.assertEqual(page.session_state.final["final_action"], "auto_commit_failed")


if __name__ == "__main__":
    unittest.main()
