"""The whole agent loop with a fake model and a fake sandbox. No network, no cost."""
import asyncio

from rootstep.agent import FORMAT_REMINDER, REPEAT_WARNING, run_agent
from rootstep.llm import Reply
from rootstep.record import Run
from rootstep.sandbox import CommandResult
from rootstep.tasks import TEXTSTATS


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, **settings):
        self.calls.append((list(messages), settings))
        return Reply(self.replies.pop(0), "hidden thoughts", 100, 10, 0.001)


class FakeSandbox:
    def __init__(self):
        self.count = 0

    async def run(self, snapshot, command, cwd):
        self.count += 1
        new_cwd = f"{cwd}/{command[3:]}" if command.startswith("cd ") else cwd
        return CommandResult(f"output of {command}", 0, f"snap{self.count}", new_cwd)


def bash(cmd):
    return f"THOUGHT: do it\n```bash\n{cmd}\n```"


def new_run():
    cfg = {"model": "m", "temperature": 0.7, "thinking": False, "max_steps": 30}
    return Run(run_id="r1", task="textstats", config=cfg, start_snapshot="snap0")


def test_full_loop():
    llm = FakeLLM([bash("ls"), bash("ls"), "no format here", bash("cd textstats"), "THOUGHT: done\nFINISH"])
    seen = []
    run = asyncio.run(run_agent(TEXTSTATS, llm, FakeSandbox(), new_run(), on_step=seen.append))

    assert [s.action for s in run.steps] == ["ls", None, None, "cd textstats", None]
    assert seen == run.steps  # live progress callback sees every step
    assert run.steps[1].observation_shown == REPEAT_WARNING
    assert run.steps[2].observation_shown == FORMAT_REMINDER
    assert run.steps[3].cwd_after == "/repo/textstats"
    assert run.steps[0].snapshot_before == "snap0" and run.steps[0].snapshot_after == "snap1"
    assert run.finished and run.final_snapshot == "snap2"
    assert abs(run.total_cost_usd - 0.005) < 1e-9
    # settings come from run.config, so re-runs use the same agent
    assert llm.calls[0][1] == {"model": "m", "temperature": 0.7, "thinking": False, "max_tokens": 2048}
    # the hidden reasoning is never sent back to the model
    assert all("hidden thoughts" not in m["content"] for m in run.steps[-1].messages_sent)


def test_rerun_from_step_k():
    first = asyncio.run(run_agent(
        TEXTSTATS, FakeLLM([bash("ls"), bash("cd textstats"), bash("cat core.py"), "FINISH"]),
        FakeSandbox(), new_run()))
    k = first.steps[2]

    rerun = Run(run_id="r2", task="textstats", config=first.config, start_snapshot=first.start_snapshot,
                parent_run_id="r1", fork_step=2)
    llm = FakeLLM([bash("cat cli.py"), "FINISH"])
    asyncio.run(run_agent(TEXTSTATS, llm, FakeSandbox(), rerun, snapshot=k.snapshot_before,
                          cwd=k.cwd_before, messages=k.messages_sent, start_index=2))

    assert rerun.steps[0].index == 2
    assert rerun.steps[0].cwd_before == "/repo/textstats"
    assert llm.calls[0][0] == k.messages_sent      # same conversation as the original step 2
    assert rerun.steps[0].action == "cat cli.py"   # but a fresh decision


def test_save_and_load(tmp_path):
    run = asyncio.run(run_agent(TEXTSTATS, FakeLLM([bash("ls"), "FINISH"]), FakeSandbox(), new_run()))
    loaded = Run.load(run.save(tmp_path))
    assert loaded == run


def test_tidy_history_for_new_runs():
    # harness 0.3+: leaked reasoning dropped, invented outputs never reach the next step
    rambling = ("I should look around.</think>THOUGHT: list the files\n```bash\nls\n```\n"
                "```bash\nfake output that never happened\n```\nTHOUGHT: now this\n```bash\nrm -rf /\n```")
    run = new_run()
    run.config["harness_version"] = "0.3"
    llm = FakeLLM([rambling, "THOUGHT: done\nFINISH"])
    asyncio.run(run_agent(TEXTSTATS, llm, FakeSandbox(), run))

    assert run.steps[0].action == "ls"
    assert run.steps[0].raw_response == rambling                 # the diary keeps the full evidence
    kept = run.steps[1].messages_sent[2]["content"]
    assert kept == "THOUGHT: list the files\n```bash\nls\n```"  # the conversation keeps only what ran
    assert llm.calls[0][1]["max_tokens"] == 600


def test_old_runs_keep_old_behaviour():
    llm = FakeLLM(["THOUGHT: a\n```bash\nls\n```\nextra text", "FINISH"])
    run = asyncio.run(run_agent(TEXTSTATS, llm, FakeSandbox(), new_run()))
    assert run.steps[1].messages_sent[2]["content"].endswith("extra text")
