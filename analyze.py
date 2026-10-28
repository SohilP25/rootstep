"""Go back into a saved run, re-run the agent from its steps, and find the decisive step.

Usage (from the workspace folder):
    python rootstep/analyze.py 382be4fa                  # every step, 5 re-runs each
    python rootstep/analyze.py 382be4fa --tries 3 --steps 0,4,8,12

Saves runs/<run_id>.analysis.json and every re-run's diary in runs/reruns/.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from rootstep.attribute import attribute
from rootstep.llm import LLM
from rootstep.record import Run
from rootstep.sandbox import Sandbox
from rootstep.tasks import TASKS

RUNS_DIR = Path(__file__).parent / "runs"


def bar(rate: float, width: int = 20) -> str:
    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled)


async def main(args: argparse.Namespace) -> None:
    original = Run.load(RUNS_DIR / f"{args.run_id}.json")
    task = TASKS[original.task]
    steps = [int(s) for s in args.steps.split(",")] if args.steps else list(range(len(original.steps)))
    llm = LLM(budget_usd=args.budget)

    print(f"Run {original.run_id}: {len(original.steps)} steps, success={original.success}")
    print(f"Re-running {args.tries}x from each of {len(steps)} steps "
          f"({args.tries * len(steps)} re-runs, budget ${args.budget:.2f})...", flush=True)
    t0 = time.monotonic()
    report = await attribute(task, original, llm, Sandbox(), steps=steps, tries=args.tries,
                             reruns_dir=RUNS_DIR / "reruns")

    print()
    for s in report["scores"]:
        cmd = (original.steps[s["step"]].action or "(no command)").splitlines()[0][:50]
        errors = f"  errors: {len(s['errors'])}" if s["errors"] else ""
        print(f"step {s['step']:>2}  {bar(s['rate'])}  {s['successes']}/{s['tries']}  "
              f"95%: {s['low']:.2f}-{s['high']:.2f}  {cmd}{errors}")
    d = report["decisive"]
    if d:
        sure = "confident" if d["confident"] else "not yet confident: try more re-runs"
        print(f"\nDecisive step: {d['step']}  (rescue rate {d['rate_before']:.0%} -> "
              f"{d['rate_after']:.0%} at step {d['next_scored_step']}; {sure})")

    out = RUNS_DIR / f"{original.run_id}.analysis.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"cost ${report['cost_usd']:.4f}, {time.monotonic() - t0:.0f}s, saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_id")
    p.add_argument("--tries", type=int, default=5, help="re-runs per step (default 5)")
    p.add_argument("--steps", help="comma-separated step numbers (default: every step)")
    p.add_argument("--budget", type=float, default=1.0, help="max $ for this analysis (default 1.00)")
    asyncio.run(main(p.parse_args()))
