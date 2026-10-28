"""The agent: a minimal terminal agent (the ReAct loop).

    ask the model -> read one bash command -> run it in the sandbox
    -> show the (shortened) output to the model -> repeat

It stops when the model says FINISH or after max_steps. Every step is
recorded, so Rootstep can later go back to any step and re-run from there.
"""
import re
from collections.abc import Callable

from .llm import LLM
from .record import Run, Step
from .sandbox import Sandbox
from .tasks import Task

MAX_STEPS = 30
OBS_HEAD = 1000   # characters kept from the start of long output
OBS_TAIL = 1000   # characters kept from the end of long output

SYSTEM_PROMPT = """You are an autonomous software engineer working in a Linux terminal.
You solve the task by running shell commands, one per reply. You see each command's output.

Reply in exactly this format:
THOUGHT: <one or two sentences about what you will do and why>
```bash
<exactly one shell command>
```

When the task is fully done and verified, reply with:
THOUGHT: <why you are sure it is done>
FINISH

Rules:
- One command per reply. Put the THOUGHT line outside the ```bash box; the box holds only the command.
- Long output is shortened, so prefer focused commands (grep, head, sed -n).
- Edit files with commands like `cat > file <<'EOF' ... EOF` or `sed -i`.
- Never edit the tests unless the task says so.
"""

FORMAT_REMINDER = "Your reply had no ```bash block and no FINISH. Reply again in the required format."
REPEAT_WARNING = "You just ran that exact command. It was not run again. Try something different."


def parse_action(text: str) -> str | None:
    """Return the bash command inside the reply, or None if the agent said FINISH.
    Raise ValueError if the reply has neither.

    Models sometimes put their THOUGHT line inside the ```bash box too, so we
    skip any box that starts with "THOUGHT:" and take the first real command."""
    if any(line.strip() == "FINISH" for line in text.splitlines()):
        return None
    for opener in re.finditer(r"```bash\n", text):
        end = text.find("```", opener.end())
        command = text[opener.end():end if end != -1 else len(text)].strip()
        if command and not command.startswith("THOUGHT:"):
            return command
    raise ValueError("no bash block")


def shorten(output: str) -> str:
    """What the model sees: long output keeps its start and end, the middle is cut."""
    if not output.strip():
        return "(no output)"
    if len(output) <= OBS_HEAD + OBS_TAIL:
        return output
    cut = len(output) - OBS_HEAD - OBS_TAIL
    return output[:OBS_HEAD] + f"\n... [{cut} characters omitted] ...\n" + output[-OBS_TAIL:]


def first_messages(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.instruction},
    ]


async def run_agent(task: Task, llm: LLM, sandbox: Sandbox, run: Run, *,
                    snapshot: str | None = None, cwd: str | None = None,
                    messages: list[dict] | None = None, start_index: int = 0,
                    last_action: str | None = None,
                    on_step: Callable[[Step], None] | None = None) -> Run:
    """Run the agent and record every step into `run`.

    Normally it starts at the beginning of the task. For a re-run, pass the
    state of step k (snapshot, cwd, messages, start_index=k) and it carries
    on from there, deciding step k afresh.
    """
    snapshot = snapshot or run.start_snapshot
    cwd = cwd or task.workdir
    messages = list(messages) if messages is not None else first_messages(task)
    cfg = run.config

    for index in range(start_index, cfg["max_steps"]):
        messages_sent = list(messages)
        reply = await llm.chat(messages, model=cfg["model"], temperature=cfg["temperature"],
                               thinking=cfg["thinking"])
        snapshot_before, cwd_before = snapshot, cwd
        exit_code, raw = None, ""

        try:
            action = parse_action(reply.content)
            finished = action is None
            observation = ""
        except ValueError:
            action, finished, observation = None, False, FORMAT_REMINDER

        if not finished and action is not None:
            if action == last_action:
                action, observation = None, REPEAT_WARNING
            else:
                result = await sandbox.run(snapshot, action, cwd)
                snapshot, cwd, last_action = result.snapshot, result.cwd, action
                exit_code, raw = result.exit_code, result.output
                observation = f"exit code: {result.exit_code}\n{shorten(result.output)}"

        run.steps.append(Step(
            index=index, snapshot_before=snapshot_before, snapshot_after=snapshot,
            cwd_before=cwd_before, cwd_after=cwd, messages_sent=messages_sent,
            raw_response=reply.content, reasoning=reply.reasoning, action=action,
            observation_raw=raw, observation_shown=observation, exit_code=exit_code,
            prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
            cost_usd=reply.cost_usd,
        ))
        run.total_cost_usd += reply.cost_usd
        if on_step:
            on_step(run.steps[-1])

        if finished:
            run.finished = True
            break
        messages.append({"role": "assistant", "content": reply.content})
        messages.append({"role": "user", "content": observation})

    run.final_snapshot = snapshot
    return run
