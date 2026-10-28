"""A terminal for the agent, on top of Nebius Sandboxes (ConTree).

Two things happen here:
1. Every command returns a new snapshot id — the "save point" Rootstep
   can later load and replay from.
2. Every command runs in a brand-new shell, so `cd` would be forgotten.
   We fix that: each command starts in the last known folder, and we read
   back the folder it finished in.
"""
import os
import shlex
from dataclasses import dataclass

from contree_sdk import Contree
from contree_sdk.sdk.exceptions import OperationTimedOutError
from dotenv import load_dotenv

load_dotenv()
# The CLI docs use NEBIUS_AI_PROJECT; the SDK reads NEBIUS_PROJECT_ID.
if "NEBIUS_PROJECT_ID" not in os.environ and "NEBIUS_AI_PROJECT" in os.environ:
    os.environ["NEBIUS_PROJECT_ID"] = os.environ["NEBIUS_AI_PROJECT"]

CWD_MARKER = "__ROOTSTEP_CWD__="


@dataclass
class CommandResult:
    output: str      # stdout + stderr of the command
    exit_code: int
    snapshot: str    # snapshot id after the command ran
    cwd: str         # folder the command finished in


def wrap(command: str, cwd: str) -> str:
    """Shell script: start in `cwd`, run `command`, then print the folder it ended in."""
    script = (
        f"cd {shlex.quote(cwd)} 2>/dev/null || cd /\n"
        f"{{ {command}\n}} 2>&1\n"
        "__rc=$?\n"
        f'printf "\\n{CWD_MARKER}%s\\n" "$(pwd)"\n'
        "exit $__rc"
    )
    return f"bash -c {shlex.quote(script)}"


def unwrap(stdout: str, fallback_cwd: str) -> tuple[str, str]:
    """Split the command's own output from the folder marker that `wrap` added."""
    head, sep, tail = stdout.rpartition("\n" + CWD_MARKER)
    if not sep:
        return stdout, fallback_cwd
    return head, tail.strip() or fallback_cwd


class Sandbox:
    def __init__(self):
        self.sdk = Contree()

    async def prepare(self, image_ref: str, setup: str, files: dict[str, str]) -> str:
        """Build a task's starting snapshot: base image + task files + setup command."""
        base = await self.sdk.images.oci(image_ref)
        img = await base.run(
            shell=f"bash -c {shlex.quote(setup)}",
            files={path: content.encode() for path, content in files.items()},
            disposable=False,
            timeout=600,
        )
        if img.exit_code != 0:
            raise RuntimeError(f"task setup failed ({img.exit_code}): {(img.stderr or '')[-500:]}")
        return str(img.uuid)

    async def run(self, snapshot: str, command: str, cwd: str, timeout: float = 60) -> CommandResult:
        """Run one command on top of `snapshot` and return the new snapshot."""
        image = await self.sdk.images.use(snapshot)
        try:
            img = await image.run(shell=wrap(command, cwd), disposable=False,
                                  timeout=timeout, truncate_output_at=200_000)
        except OperationTimedOutError:
            return CommandResult(f"[command timed out after {timeout:.0f}s]", 124, snapshot, cwd)
        output, new_cwd = unwrap(img.stdout or "", cwd)
        return CommandResult(output, img.exit_code, str(img.uuid), new_cwd)
