"""Re-runs, rescue rates and the decisive step, with fakes. No network, no cost."""
import asyncio

import pytest
from dataclasses import asdict

from rootstep.attribute import (StepScore, attribute, biggest_drop, build_report, last_command_before,
                                merge_scores, point_of_no_return, wilson)
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


def test_biggest_drop():
    scores = [score(0, 9, 10), score(1, 9, 10), score(2, 8, 10), score(3, 1, 10), score(4, 0, 10)]
    d = biggest_drop(scores)
    assert d["step"] == 2 and d["next_scored_step"] == 3
    assert d["confident"]                           # 8/10 vs 1/10: ranges don't overlap


def test_small_samples_are_not_confident():
    d = biggest_drop([score(0, 2, 3), score(1, 1, 3)])
    assert d["step"] == 0 and not d["confident"]


def test_point_of_no_return_ignores_an_unlucky_dip():
    # the real curve of run 382be4fa-k0-e8ca (4 tries per step)
    curve = [score(0, 3, 4), score(3, 0, 4), score(6, 4, 4), score(9, 2, 4), score(12, 0, 4),
             score(15, 1, 4), score(18, 0, 4), score(21, 0, 4), score(24, 0, 4), score(27, 0, 4)]
    p = point_of_no_return(curve)
    assert (p["step"], p["next_scored_step"], p["later_tries"]) == (15, 18, 16)
    assert not p["confident"]                       # 1/4 is too few to be sure
    assert biggest_drop(curve)["step"] == 0         # the naive rule is fooled by the dip at step 3


def test_no_point_of_no_return_when_still_saveable_at_the_end():
    assert point_of_no_return([score(0, 1, 4), score(5, 2, 4)]) is None
    assert point_of_no_return([score(0, 0, 4), score(5, 0, 4)]) is None


def test_merge_adds_tries():
    old = [asdict(score(15, 1, 4)), asdict(score(18, 0, 4))]
    merged = merge_scores(old, [score(15, 5, 6), score(16, 2, 6)])
    by_step = {s.step: s for s in merged}
    assert (by_step[15].successes, by_step[15].tries) == (6, 10)
    assert by_step[16].tries == 6 and by_step[18].tries == 4
    assert [s.step for s in merged] == [15, 16, 18]


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
    scores = asyncio.run(attribute(TEXTSTATS, original, llm, FakeSandbox(), positions=[0, 1, 2], tries=4))
    report = build_report(original, scores, 0.0)

    rates = {s["step"]: s["rate"] for s in report["scores"]}
    assert rates == {0: 1.0, 1: 1.0, 2: 0.0}        # step 1's save point was the last good one
    assert report["point_of_no_return"]["step"] == 1  # what the agent did at step 1 sealed its fate
    assert "before step 2" in llm.first_messages     # re-runs got the original conversation of step k


@pytest.mark.parametrize("k", [0, 2])
def test_rerun_ids_point_back(k):
    original = Run("orig", "textstats", {"model": "m", "temperature": 0.7, "thinking": False, "max_steps": 30},
                   "s0", steps=[make_step(0, "ls", "s1"), make_step(1, "a", "s2"), make_step(2, "b", "s3")])
    scores = asyncio.run(attribute(TEXTSTATS, original, FakeLLM(), FakeSandbox(), positions=[k], tries=2))
    assert all(rid.startswith(f"orig-k{k}-") for rid in scores[0].rerun_ids)


def test_rates_only_rise_means_no_drop():
    assert biggest_drop([score(0, 1, 3), score(5, 2, 3), score(14, 3, 3)]) is None


def test_rerun_of_a_rerun_keeps_real_step_numbers():
    # a re-run that started at step 11: its first diary entry is step 11, at position 0
    rerun = Run("orig-k11-abcd", "textstats", {"model": "m", "temperature": 0.7, "thinking": False, "max_steps": 30},
                "s0", steps=[make_step(11, "sed -i x", "s12"), make_step(12, None, "s12")],
                parent_run_id="orig", fork_step=11)
    scores = asyncio.run(attribute(TEXTSTATS, rerun, FakeLLM(), FakeSandbox(), positions=[0, 1], tries=2))
    assert [s.step for s in scores] == [11, 12]
    assert all(rid.startswith("orig-k11-abcd-k11-") for rid in scores[0].rerun_ids)
