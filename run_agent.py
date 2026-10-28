"""Run the agent once on a task and save the recording.

Usage (from the workspace folder):
    python rootstep/run_agent.py textstats

Each step is printed as it happens. If you stop it with Ctrl+C, the diary
of the steps so far is still saved.
"""
import asyncio
import sys
import time
import uuid
from pathlib import Path

from rootstep.agent import run_agent
from rootstep.llm import LIGHTNING, LLM
from rootstep.record import Run, Step
from rootstep.sandbox import Sandbox
from rootstep.tasks import TASKS

HARNESS_VERSION = "0.3"  # 0.2: THOUGHT-in-box parsing fix; 0.3: strip leaked </think> from content
RUNS_DIR = Path(__file__).parent / "runs"


def describe(step: Step) -> str:
    if step.action:
        return f"exit {step.exit_code}  {step.action.splitlines()[0][:80]}"
    if step.observation_shown == "":
        return "FINISH"
    return step.observation_shown.split(".")[0]


async def main(task_name: str) -> None:
    task = TASKS[task_name]
    sandbox = Sandbox()
    llm = LLM(budget_usd=0.25)
    t0 = time.monotonic()

    def show(step: Step) -> None:
        print(f"  step {step.index:>2}  {time.monotonic() - t0:4.0f}s  {describe(step)}", flush=True)

    print(f"Preparing task '{task.name}'...", flush=True)
    start = await sandbox.prepare(task.image, task.setup, task.files)
    print(f"  ready in {time.monotonic() - t0:.0f}s", flush=True)

    run = Run(
        run_id=uuid.uuid4().hex[:8],
        task=task.name,
        config={"model": LIGHTNING, "temperature": 0.7, "thinking": False,
                "max_steps": 30, "harness_version": HARNESS_VERSION},
        start_snapshot=start,
    )
    print(f"Running agent (run {run.run_id}):", flush=True)
    try:
        await run_agent(task, llm, sandbox, run, on_step=show)
        result = await sandbox.run(run.final_snapshot, task.check, task.workdir)
        run.success = result.exit_code == 0
    finally:
        path = run.save(RUNS_DIR)
        print(f"steps={len(run.steps)} finished={run.finished} success={run.success} "
              f"cost=${run.total_cost_usd:.4f}")
        print(f"saved {path}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "textstats"))
