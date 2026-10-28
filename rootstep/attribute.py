"""Find the step where a run went wrong, by going back and trying again.

For a step k we load the save point from just before step k, give the agent
the same conversation it had at that moment, and let it decide step k (and
everything after) afresh. We do that N times. The share of re-runs that
succeed is step k's **rescue rate**.

    high rescue rate at k  -> the run could still be saved at step k
    big drop from k to k+1 -> what the agent did at step k sealed its fate
"""
import asyncio
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import run_agent
from .llm import LLM
from .record import Run
from .sandbox import Sandbox
from .tasks import Task


@dataclass
class StepScore:
    step: int
    successes: int
    tries: int
    rate: float
    low: float                    # 95% confidence range (Wilson interval)
    high: float
    rerun_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def wilson(successes: int, tries: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% range for the true success rate, honest even for few tries.

    6/12 doesn't mean "exactly 50%": with only 12 tries the truth could be
    anywhere from about 25% to 75%. Wilson's formula gives that range."""
    if tries == 0:
        return 0.0, 1.0
    p = successes / tries
    denom = 1 + z * z / tries
    centre = (p + z * z / (2 * tries)) / denom
    half = z * math.sqrt(p * (1 - p) / tries + z * z / (4 * tries * tries)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def last_command_before(run: Run, pos: int) -> str | None:
    """The repeat guard's memory at a step: the last command that actually ran before it."""
    return next((s.action for s in reversed(run.steps[:pos]) if s.action), None)


async def rerun_from(task: Task, original: Run, pos: int, llm: LLM, sandbox: Sandbox) -> Run:
    """Go back to just before the step at position `pos` in `original` and carry on afresh.

    `pos` is a position in the steps list. For a normal run it equals the step
    number; a re-run that started at step 11 has step 11 at position 0."""
    step = original.steps[pos]
    rerun = Run(
        run_id=f"{original.run_id}-k{step.index}-{uuid.uuid4().hex[:4]}",
        task=original.task,
        config=original.config,               # same agent, same settings
        start_snapshot=original.start_snapshot,
        parent_run_id=original.run_id,
        fork_step=step.index,
    )
    await run_agent(task, llm, sandbox, rerun,
                    snapshot=step.snapshot_before, cwd=step.cwd_before,
                    messages=step.messages_sent, start_index=step.index,
                    last_action=last_command_before(original, pos))
    result = await sandbox.run(rerun.final_snapshot, task.check, task.workdir)
    rerun.success = result.exit_code == 0
    return rerun


async def score_step(task: Task, original: Run, pos: int, tries: int, llm: LLM, sandbox: Sandbox,
                     gate: asyncio.Semaphore, reruns_dir: Path | None = None) -> StepScore:
    async def one() -> Run:
        async with gate:
            rerun = await rerun_from(task, original, pos, llm, sandbox)
        if reruns_dir:
            rerun.save(reruns_dir)
        return rerun

    results = await asyncio.gather(*(one() for _ in range(tries)), return_exceptions=True)
    done = [r for r in results if isinstance(r, Run)]
    errors = [f"{type(r).__name__}: {r}"[:200] for r in results if isinstance(r, BaseException)]
    successes = sum(bool(r.success) for r in done)
    low, high = wilson(successes, len(done))
    return StepScore(original.steps[pos].index, successes, len(done),
                     successes / len(done) if done else 0.0, low, high, [r.run_id for r in done], errors)


def biggest_drop(scores: list[StepScore]) -> dict | None:
    """The step after which the rescue rate falls the most, or None if it never falls.

    Easily fooled by one unlucky dip when tries are few, so it is reported as
    extra information. `confident` is True when the two 95% ranges don't overlap."""
    scores = sorted((s for s in scores if s.tries), key=lambda s: s.step)
    best = None
    for here, after in zip(scores, scores[1:]):
        drop = here.rate - after.rate
        if best is None or drop > best["drop"]:
            best = {"step": here.step, "next_scored_step": after.step, "drop": round(drop, 3),
                    "rate_before": here.rate, "rate_after": after.rate,
                    "confident": here.low > after.high}
    return best if best and best["drop"] > 0 else None


def point_of_no_return(scores: list[StepScore]) -> dict | None:
    """The last step where re-deciding can still save the run.

    It is the last scored step where at least one re-run succeeded, when every
    scored step after it failed every time. The agent's original action at
    that step (or between it and the next scored step) sealed the run's fate.
    None if the run can still be saved at the last scored step, or never could.
    `confident` is True when its 95% range sits above that of all later re-runs."""
    scored = sorted((s for s in scores if s.tries), key=lambda s: s.step)
    alive = [i for i, s in enumerate(scored) if s.successes > 0]
    if not alive or alive[-1] == len(scored) - 1:
        return None
    here, later = scored[alive[-1]], scored[alive[-1] + 1:]
    later_tries = sum(s.tries for s in later)
    _, later_high = wilson(0, later_tries)
    return {"step": here.step, "next_scored_step": later[0].step,
            "rate_before": here.rate, "later_tries": later_tries,
            "confident": here.low > later_high}


def merge_scores(old: list[dict], new: list[StepScore]) -> list[StepScore]:
    """Add new re-runs to earlier ones for the same steps, so repeated analyses sharpen the curve."""
    by_step = {d["step"]: StepScore(**d) for d in old}
    for s in new:
        o = by_step.get(s.step)
        if o is None:
            by_step[s.step] = s
            continue
        successes, tries = o.successes + s.successes, o.tries + s.tries
        low, high = wilson(successes, tries)
        by_step[s.step] = StepScore(s.step, successes, tries, successes / tries if tries else 0.0,
                                    low, high, o.rerun_ids + s.rerun_ids, o.errors + s.errors)
    return sorted(by_step.values(), key=lambda s: s.step)


def build_report(original: Run, scores: list[StepScore], cost_usd: float) -> dict:
    return {
        "run_id": original.run_id,
        "task": original.task,
        "original_success": original.success,
        "scores": [asdict(s) for s in scores],
        "point_of_no_return": point_of_no_return(scores),
        "biggest_drop": biggest_drop(scores),
        "cost_usd": round(cost_usd, 5),
    }


async def attribute(task: Task, original: Run, llm: LLM, sandbox: Sandbox, *, positions: list[int],
                    tries: int = 5, parallel: int = 10, reruns_dir: Path | None = None) -> list[StepScore]:
    """Score the steps at `positions` (indexes into original.steps)."""
    gate = asyncio.Semaphore(parallel)
    return list(await asyncio.gather(*(score_step(task, original, pos, tries, llm, sandbox, gate, reruns_dir)
                                       for pos in positions)))
