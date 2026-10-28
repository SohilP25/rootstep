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


def last_command_before(run: Run, k: int) -> str | None:
    """The repeat guard's memory at step k: the last command that actually ran."""
    return next((s.action for s in reversed(run.steps[:k]) if s.action), None)


async def rerun_from(task: Task, original: Run, k: int, llm: LLM, sandbox: Sandbox) -> Run:
    """Go back to just before step k of `original` and let the agent carry on afresh."""
    step = original.steps[k]
    rerun = Run(
        run_id=f"{original.run_id}-k{k}-{uuid.uuid4().hex[:4]}",
        task=original.task,
        config=original.config,               # same agent, same settings
        start_snapshot=original.start_snapshot,
        parent_run_id=original.run_id,
        fork_step=k,
    )
    await run_agent(task, llm, sandbox, rerun,
                    snapshot=step.snapshot_before, cwd=step.cwd_before,
                    messages=step.messages_sent, start_index=k,
                    last_action=last_command_before(original, k))
    result = await sandbox.run(rerun.final_snapshot, task.check, task.workdir)
    rerun.success = result.exit_code == 0
    return rerun


async def score_step(task: Task, original: Run, k: int, tries: int, llm: LLM, sandbox: Sandbox,
                     gate: asyncio.Semaphore, reruns_dir: Path | None = None) -> StepScore:
    async def one() -> Run:
        async with gate:
            rerun = await rerun_from(task, original, k, llm, sandbox)
        if reruns_dir:
            rerun.save(reruns_dir)
        return rerun

    results = await asyncio.gather(*(one() for _ in range(tries)), return_exceptions=True)
    done = [r for r in results if isinstance(r, Run)]
    errors = [f"{type(r).__name__}: {r}"[:200] for r in results if isinstance(r, BaseException)]
    successes = sum(bool(r.success) for r in done)
    low, high = wilson(successes, len(done))
    return StepScore(k, successes, len(done), successes / len(done) if done else 0.0,
                     low, high, [r.run_id for r in done], errors)


def decisive_step(scores: list[StepScore]) -> dict | None:
    """The step after which the rescue rate falls the most.

    `confident` is True when the two 95% ranges don't overlap, i.e. the drop
    is very unlikely to be luck."""
    scores = sorted((s for s in scores if s.tries), key=lambda s: s.step)
    best = None
    for here, after in zip(scores, scores[1:]):
        drop = here.rate - after.rate
        if best is None or drop > best["drop"]:
            best = {"step": here.step, "next_scored_step": after.step, "drop": round(drop, 3),
                    "rate_before": here.rate, "rate_after": after.rate,
                    "confident": here.low > after.high}
    return best


async def attribute(task: Task, original: Run, llm: LLM, sandbox: Sandbox, *, steps: list[int],
                    tries: int = 5, parallel: int = 10, reruns_dir: Path | None = None) -> dict:
    gate = asyncio.Semaphore(parallel)
    scores = await asyncio.gather(*(score_step(task, original, k, tries, llm, sandbox, gate, reruns_dir)
                                    for k in steps))
    return {
        "run_id": original.run_id,
        "task": original.task,
        "original_success": original.success,
        "tries_per_step": tries,
        "scores": [asdict(s) for s in scores],
        "decisive": decisive_step(list(scores)),
        "cost_usd": round(llm.spent_usd, 5),
    }
