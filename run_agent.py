"""Run the agent once on a task and save the recording.

Usage (from the workspace folder):
    python rootstep/run_agent.py textstats
    python rootstep/run_agent.py textstats --plant 3   # test run: a known mistake at step 3

Each step is printed as it happens. If you stop it with Ctrl+C, the diary
of the steps so far is still saved.
"""
import argparse
import asyncio
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
    if step.planted:
        return f"PLANTED MISTAKE  {step.action}"
    if step.action:
        return f"exit {step.exit_code}  {step.action.splitlines()[0][:80]}"
    if step.observation_shown == "":
        return "FINISH"
    return step.observation_shown.split(".")[0]


async def main(task_name: str, plant_step: int | None) -> None:
    task = TASKS[task_name]
    plant = None
    if plant_step is not None:
        if not task.planted_mistake:
            raise SystemExit(f"Task '{task.name}' has no planted mistake defined")
        plant = {"step": plant_step, **task.planted_mistake}
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
        planted_step=plant_step,
    )
    note = f", known mistake planted at step {plant_step}" if plant else ""
    print(f"Running agent (run {run.run_id}{note}):", flush=True)
    try:
        await run_agent(task, llm, sandbox, run, on_step=show, plant=plant)
        result = await sandbox.run(run.final_snapshot, task.check, task.workdir)
        run.success = result.exit_code == 0
    finally:
        path = run.save(RUNS_DIR)
        print(f"steps={len(run.steps)} finished={run.finished} success={run.success} "
              f"cost=${run.total_cost_usd:.4f}")
        print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", nargs="?", default="textstats")
    p.add_argument("--plant", type=int, metavar="STEP",
                   help="replace the agent's decision at STEP with the task's known mistake")
    args = p.parse_args()
    asyncio.run(main(args.task, args.plant))
