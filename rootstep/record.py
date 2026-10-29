"""The recorder: what we save about every step of a run.

Rootstep can only go back to step k if we saved enough to rebuild step k
exactly: the files (snapshot id), the folder (cwd) and the conversation
(messages). Everything else is for explaining failures and counting cost.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Step:
    index: int
    snapshot_before: str        # save point before this step ran
    snapshot_after: str         # save point after the command (same as before if nothing ran)
    cwd_before: str
    cwd_after: str
    messages_sent: list[dict]   # exactly what the model saw at this step
    raw_response: str           # the model's visible reply
    reasoning: str | None       # hidden reasoning: kept for explanations, never re-sent
    action: str | None          # the command that ran, or None (FINISH / bad format / repeat)
    observation_raw: str        # full command output
    observation_shown: str      # the shortened version the model saw
    exit_code: int | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    planted: bool = False       # this step was a planted mistake (a test with a known answer)


@dataclass
class Run:
    run_id: str
    task: str
    config: dict                  # model, temperature, thinking, max_steps, harness_version
    start_snapshot: str
    steps: list[Step] = field(default_factory=list)
    final_snapshot: str | None = None
    finished: bool = False        # the agent said FINISH
    success: bool | None = None   # result of the task's check command
    total_cost_usd: float = 0.0
    parent_run_id: str | None = None  # set on re-runs: the run we went back into
    fork_step: int | None = None      # set on re-runs: the step we re-decided from
    planted_step: int | None = None   # set when a known mistake was planted at this step

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "Run":
        data = json.loads(Path(path).read_text())
        data["steps"] = [Step(**s) for s in data["steps"]]
        return cls(**data)
