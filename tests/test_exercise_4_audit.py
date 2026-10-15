"""Behavior checks using real SQLite checkpoints and mocked external services."""

import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console

from audit.replay import list_threads
from common.db import db_conn, replay_events
from common.schemas import PRAnalysis, risk_level_for
from exercises import exercise_4_audit as ex


class AuditFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_and_durable_resume(self):
        for choice in ("approve", "reject", "edit", "auto", "escalate", "post_failure"):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as folder:
                await self.check_flow(choice, os.path.join(folder, "audit.db"))

    async def check_flow(self, choice, database):
        confidence = 0.9 if choice == "auto" else 0.5 if choice == "escalate" else 0.65
        initial = PRAnalysis(
            summary="Initial review", confidence=confidence,
            confidence_reasoning="Initial uncertainty",
            escalation_questions=["auth.py: Why MD5?"],
        )
        # Refinement is allowed to LOWER confidence; the audit must reflect it.
        refined = PRAnalysis(
            summary="Refined review", confidence=0.4,
            confidence_reasoning="Reviewer confirmed unresolved security gaps",
        )
        pr_url = "https://example.com/pr/1"
        cfg = {"configurable": {"thread_id": choice}}
        pr = SimpleNamespace(title="Test PR", diff="original diff", files_changed=["auth.py"], head_sha="abcdef12")
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"HITL_DB_PATH": database, "GITHUB_USER": "test-reviewer"}),
            patch.object(ex, "fetch_pr", return_value=pr) as fetch,
            patch.object(ex, "get_llm") as factory,
            patch.object(ex, "post_review_comment") as post,
            patch.object(ex, "console", Console(file=output, width=160)),
        ):
            model = factory.return_value.with_structured_output.return_value
            model.ainvoke = AsyncMock(side_effect=[initial, refined])
            if choice == "post_failure":
                post.side_effect = RuntimeError("Simulated GitHub failure")
            async with AsyncSqliteSaver.from_conn_string(database) as cp:
                app = ex.build_graph(cp)
                result = await app.ainvoke({"pr_url": pr_url, "thread_id": choice}, cfg)
                if choice != "auto":
                    self.assertIn("__interrupt__", result)
                    post.assert_not_called()
                    self.assertEqual(model.ainvoke.await_count, 1)

            # Reopen the database/checkpointer as a fresh process would.
            if choice == "reject":
                with patch.object(ex, "handle_interrupt", return_value={"choice": "reject", "feedback": "Needs work"}):
                    await ex.run(pr_url, choice)
            elif choice != "auto":
                async with AsyncSqliteSaver.from_conn_string(database) as cp:
                    app = ex.build_graph(cp)
                    answer = {"auth.py: Why MD5?": "Legacy code; still unsafe"} if choice == "escalate" else {
                        "choice": "approve" if choice == "post_failure" else choice,
                        "feedback": "Needs work",
                    }
                    result = await app.ainvoke(Command(resume=answer), cfg)
                    self.assertNotIn("__interrupt__", result)

            events = await replay_events(choice)
            expected = ["fetch_pr", "analyze", "route"]
            if choice == "auto":
                expected += ["auto_approve"]
            else:
                expected += ["escalate" if choice == "escalate" else "human_approval"] * 3
                if choice == "escalate":
                    expected += ["synthesize"]
                expected += ["commit"]
            self.assertEqual([e["action"] for e in events], expected)
            self.assertEqual(events[0]["confidence"], 0.0)
            self.assertEqual(events[0]["risk_level"], "med")
            self.assertEqual(events[1]["decision"], "pending")
            self.assertEqual(events[2]["decision"], "auto" if choice == "auto" else "escalate" if choice == "escalate" else "pending")
            for event in events:
                self.assertIn(event["decision"], {"auto", "approve", "reject", "edit", "escalate", "pending"})
                self.assertGreaterEqual(event["execution_time_ms"], 0)
                self.assertTrue(event["timestamp"])
                self.assertEqual(event["agent_id"], ex.AGENT_ID)
                if event["action"] != "fetch_pr":
                    self.assertEqual(event["risk_level"], risk_level_for(event["confidence"]))
            if choice != "auto":
                self.assertEqual([e["decision"] for e in events[3:5]], ["pending", "pending"])
                self.assertTrue(all(e["reviewer_id"] is None for e in events[:5]))
                self.assertEqual(events[5]["reviewer_id"], "test-reviewer")
                self.assertEqual(events[-1]["reviewer_id"], "test-reviewer")
            else:
                self.assertTrue(all(e["reviewer_id"] is None for e in events))
            self.assertIn("commit_failed" if choice == "post_failure" else "rejected" if choice in {"reject", "edit"} else "committed", events[-1]["reason"])
            self.assertEqual(post.call_count, int(choice not in {"reject", "edit"}))
            fetch.assert_called_once()
            self.assertEqual(model.ainvoke.await_count, 2 if choice == "escalate" else 1)
            if choice == "escalate":
                self.assertEqual(events[-2]["confidence"], refined.confidence)
                self.assertEqual(events[-2]["risk_level"], "high")
                self.assertIn("Legacy code; still unsafe", events[5]["reason"])
                self.assertIn("Reviewer answered escalation questions", post.call_args.args[1])
                table = io.StringIO()
                with patch("audit.replay.Console", return_value=Console(file=table, width=250)):
                    await list_threads()
                self.assertIn("high", table.getvalue())
            async with db_conn() as conn:
                async with conn.execute("SELECT COUNT(*) FROM audit_events WHERE risk_level = 'high' AND decision = 'auto'") as cursor:
                    self.assertEqual((await cursor.fetchone())[0], 0)
            # Reopening a completed session must not repeat work or posting.
            count = len(events)
            await ex.run(pr_url, choice)
            self.assertEqual(len(await replay_events(choice)), count)
            fetch.assert_called_once()
            self.assertEqual(post.call_count, int(choice not in {"reject", "edit"}))
            with self.assertRaises(ValueError):
                await ex.run("https://example.com/pr/2", choice)


if __name__ == "__main__":
    unittest.main()
