"""Re-runs, rescue rates and the decisive step, with fakes. No network, no cost."""
import asyncio

import pytest

from rootstep.attribute import StepScore, attribute, decisive_step, last_command_before, wilson
from rootstep.llm import Reply
from rootstep.record import Run, Step
from rootstep.sandbox import CommandResult
from rootstep.tasks import TEXTSTATS


def test_wilson_range():
    low, high = wilson(6, 12)
    assert 0.2 < low < 0.3 and 0.7 < high < 0.8   # 6/12 could really be ~25%..75%
    assert wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)              # no tries = we know nothing


def score(step, successes, tries):
    low, high = wilson(successes, tries)
    return StepScore(step, successes, tries, successes / tries, low, high)


def test_decisive_step_is_biggest_drop():
    scores = [score(0, 9, 10), score(1, 9, 10), score(2, 8, 10), score(3, 1, 10), score(4, 0, 10)]
    d = decisive_step(scores)
    assert d["step"] == 2 and d["next_scored_step"] == 3
    assert d["confident"]                           # 8/10 vs 1/10: ranges don't overlap


def test_small_samples_are_not_confident():
    d = decisive_step([score(0, 2, 3), score(1, 1, 3)])
    assert d["step"] == 0 and not d["confident"]


def make_step(i, action, snap):
    return Step(i, f"s{i}", snap, "/repo", "/repo", [{"role": "user", "content": f"before step {i}"}],
                "", None, action, "", "", 0, 0, 0, 0.0)


def test_last_command_before():
    run = Run("r", "textstats", {}, "s0", steps=[make_step(0, "ls", "s1"), make_step(1, None, "s1"),
                                               make_step(2, "cat x", "s2")])
    assert last_command_before(run, 0) is None
    assert last_command_before(run, 2) == "ls"      # step 1 ran nothing, so "ls" is the memory


class FakeLLM:
    spent_usd = 0.0

    def __init__(self):
        self.first_messages = []

    async def chat(self, messages, **settings):
        if len(self.first_messages) < 50 and messages[-1]["content"].startswith("before step"):
            self.first_messages.append(messages[-1]["content"])
        return Reply("FINISH", None, 10, 1, 0.0)


class FakeSandbox:
    """The check passes only if the re-run started at or before step 1."""

    async def run(self, snapshot, command, cwd):
        ok = snapshot in ("s0", "s1")
        return CommandResult("", 0 if ok else 1, snapshot, cwd)


def test_attribute_end_to_end():
    original = Run("orig", "textstats", {"model": "m", "temperature": 0.7, "thinking": False, "max_steps": 30},
                   "s0", steps=[make_step(0, "ls", "s1"), make_step(1, "rm -rf x", "s2"),
                                make_step(2, "cat y", "s3")], success=False)
    llm = FakeLLM()
    report = asyncio.run(attribute(TEXTSTATS, original, llm, FakeSandbox(), steps=[0, 1, 2], tries=4))

    rates = {s["step"]: s["rate"] for s in report["scores"]}
    assert rates == {0: 1.0, 1: 1.0, 2: 0.0}        # step 1's save point was the last good one
    assert report["decisive"]["step"] == 1           # what the agent did at step 1 sealed its fate
    assert "before step 2" in llm.first_messages     # re-runs got the original conversation of step k


@pytest.mark.parametrize("k", [0, 2])
def test_rerun_ids_point_back(k):
    original = Run("orig", "textstats", {"model": "m", "temperature": 0.7, "thinking": False, "max_steps": 30},
                   "s0", steps=[make_step(0, "ls", "s1"), make_step(1, "a", "s2"), make_step(2, "b", "s3")])
    report = asyncio.run(attribute(TEXTSTATS, original, FakeLLM(), FakeSandbox(), steps=[k], tries=2))
    assert all(rid.startswith(f"orig-k{k}-") for rid in report["scores"][0]["rerun_ids"])
