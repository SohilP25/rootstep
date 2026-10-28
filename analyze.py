"""Go back into a saved run, re-run the agent from its steps, and find the decisive step.

Usage (from the workspace folder):
    python rootstep/analyze.py 382be4fa                     # every step, 5 re-runs each
    python rootstep/analyze.py 382be4fa-k0-e8ca --every 3   # quick first pass: every 3rd step
    python rootstep/analyze.py 382be4fa --tries 3 --steps 0,4,8,12

Works on runs in runs/ and on re-runs in runs/reruns/. Saves
runs/<run_id>.analysis.json and every re-run's diary in runs/reruns/.
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


def find_run(run_id: str) -> Path:
    for folder in (RUNS_DIR, RUNS_DIR / "reruns"):
        path = folder / f"{run_id}.json"
        if path.exists():
            return path
    raise SystemExit(f"No diary called {run_id}.json in {RUNS_DIR} or {RUNS_DIR / 'reruns'}")


def bar(rate: float, width: int = 20) -> str:
    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled)


def pick_positions(run: Run, steps: str | None, every: int | None) -> list[int]:
    position_of = {s.index: pos for pos, s in enumerate(run.steps)}
    if steps:
        return [position_of[int(n)] for n in steps.split(",")]
    positions = list(range(len(run.steps)))
    return positions[::every] if every else positions


async def main(args: argparse.Namespace) -> None:
    original = Run.load(find_run(args.run_id))
    task = TASKS[original.task]
    positions = pick_positions(original, args.steps, args.every)
    llm = LLM(budget_usd=args.budget)

    first, last = original.steps[0].index, original.steps[-1].index
    print(f"Run {original.run_id}: steps {first}-{last}, success={original.success}")
    print(f"Re-running {args.tries}x from each of {len(positions)} steps "
          f"({args.tries * len(positions)} re-runs, budget ${args.budget:.2f})...", flush=True)
    t0 = time.monotonic()
    report = await attribute(task, original, llm, Sandbox(), positions=positions, tries=args.tries,
                             reruns_dir=RUNS_DIR / "reruns")

    print()
    for pos, s in zip(positions, report["scores"]):
        cmd = (original.steps[pos].action or "(no command)").splitlines()[0][:50]
        errors = f"  errors: {len(s['errors'])}" if s["errors"] else ""
        print(f"step {s['step']:>2}  {bar(s['rate'])}  {s['successes']}/{s['tries']}  "
              f"95%: {s['low']:.2f}-{s['high']:.2f}  {cmd}{errors}")

    d = report["decisive"]
    if d is None:
        print("\nNo step made things worse: the rescue rate never drops.")
    else:
        sure = "confident" if d["confident"] else "not yet confident: try more re-runs"
        where = (f"step {d['step']}" if d["next_scored_step"] == d["step"] + 1
                 else f"between step {d['step']} and step {d['next_scored_step'] - 1}")
        print(f"\nDecisive step: {where}  (rescue rate {d['rate_before']:.0%} -> "
              f"{d['rate_after']:.0%}; {sure})")

    out = RUNS_DIR / f"{original.run_id}.analysis.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"cost ${report['cost_usd']:.4f}, {time.monotonic() - t0:.0f}s, saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_id")
    p.add_argument("--tries", type=int, default=5, help="re-runs per step (default 5)")
    p.add_argument("--steps", help="comma-separated step numbers (default: every step)")
    p.add_argument("--every", type=int, help="score only every Nth step (quick first pass)")
    p.add_argument("--budget", type=float, default=1.0, help="max $ for this analysis (default 1.00)")
    asyncio.run(main(p.parse_args()))
