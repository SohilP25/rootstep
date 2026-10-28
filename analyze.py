"""Go back into a saved run, re-run the agent from its steps, and find the decisive step.

Usage (from the workspace folder):
    python rootstep/analyze.py 382be4fa                     # every step, 5 re-runs each
    python rootstep/analyze.py 382be4fa-k0-e8ca --every 3   # quick first pass: every 3rd step
    python rootstep/analyze.py 382be4fa --tries 3 --steps 0,4,8,12

Works on runs in runs/ and on re-runs in runs/reruns/. Saves
runs/<run_id>.analysis.json and every re-run's diary in runs/reruns/.
Running it again on the same run ADDS the new re-runs to the old ones
(use --fresh to start over), so you can zoom in on a few steps later.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from rootstep.attribute import attribute, build_report, merge_scores
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
    new = await attribute(task, original, llm, Sandbox(), positions=positions, tries=args.tries,
                          reruns_dir=RUNS_DIR / "reruns")

    out = RUNS_DIR / f"{original.run_id}.analysis.json"
    old = json.loads(out.read_text()) if out.exists() and not args.fresh else None
    scores = merge_scores(old["scores"], new) if old else new
    report = build_report(original, scores, llm.spent_usd + (old["cost_usd"] if old else 0.0))
    if old:
        print("\n(added to the earlier re-runs of this run)")

    print()
    position_of = {st.index: pos for pos, st in enumerate(original.steps)}
    for s in report["scores"]:
        cmd = (original.steps[position_of[s["step"]]].action or "(no command)").splitlines()[0][:50]
        errors = f"  errors: {len(s['errors'])}" if s["errors"] else ""
        print(f"step {s['step']:>2}  {bar(s['rate'])}  {s['successes']}/{s['tries']}  "
              f"95%: {s['low']:.2f}-{s['high']:.2f}  {cmd}{errors}")

    def where(d: dict) -> str:
        if d["next_scored_step"] == d["step"] + 1:
            return f"step {d['step']}"
        return f"between step {d['step']} and step {d['next_scored_step'] - 1}"

    p = report["point_of_no_return"]
    if p:
        sure = "confident" if p["confident"] else "not yet confident: zoom in with more tries"
        print(f"\nPoint of no return: {where(p)}  (after it, 0 of {p['later_tries']} re-runs succeed; {sure})")
    else:
        print("\nPoint of no return: none found (the run can still be saved at the last scored step, or never could)")
    d = report["biggest_drop"]
    if d:
        sure = "confident" if d["confident"] else "could be luck"
        print(f"Biggest drop:       {where(d)}  ({d['rate_before']:.0%} -> {d['rate_after']:.0%}; {sure})")

    out.write_text(json.dumps(report, indent=2))
    print(f"this pass: ${llm.spent_usd:.4f}, {time.monotonic() - t0:.0f}s · total for this run: "
          f"${report['cost_usd']:.4f} · saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_id")
    p.add_argument("--tries", type=int, default=5, help="re-runs per step (default 5)")
    p.add_argument("--steps", help="comma-separated step numbers (default: every step)")
    p.add_argument("--every", type=int, help="score only every Nth step (quick first pass)")
    p.add_argument("--budget", type=float, default=1.0, help="max $ for this pass (default 1.00)")
    p.add_argument("--fresh", action="store_true", help="ignore earlier re-runs of this run")
    asyncio.run(main(p.parse_args()))
