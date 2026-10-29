"""Tasks the agent can be asked to do.

A task is: some files, an instruction, and a `check` command whose exit
code says whether the agent succeeded (0 = solved). The check also makes
sure the agent didn't "fix" the tests by editing them.
"""
from dataclasses import dataclass


@dataclass
class Task:
    name: str
    instruction: str
    files: dict[str, str]          # path inside the sandbox -> file content
    check: str                     # shell command; exit code 0 = solved
    setup: str = ("pip install -q --no-cache-dir --root-user-action=ignore pytest"
                  " && mkdir -p /opt/rootstep && cd /repo && sha256sum tests/*.py > /opt/rootstep/tests.sha256")
    image: str = "docker.io/library/python:3.12-slim"
    workdir: str = "/repo"
    # A harmless-looking action that silently makes the task impossible, used
    # to test Rootstep against a known answer (run_agent.py --plant STEP).
    planted_mistake: dict | None = None


CHECK_TESTS = "cd /repo && sha256sum --quiet -c /opt/rootstep/tests.sha256 && python -m pytest -q"

TEXTSTATS = Task(
    name="textstats",
    instruction=(
        "The repository in /repo has failing tests. Fix the source code (do not edit the tests) "
        "so that `python -m pytest -q` passes when run from /repo."
    ),
    files={
        "/repo/textstats/__init__.py": "",
        "/repo/textstats/core.py": (
            "def word_counts(text):\n"
            "    counts = {}\n"
            "    for word in text.split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n"
        ),
        "/repo/textstats/cli.py": (
            "import sys\n"
            "\n"
            "from textstats.core import word_counts\n"
            "\n"
            "\n"
            "def format_plain(counts):\n"
            '    return "\\n".join(f"{word}={n}" for word, n in sorted(counts.items()))\n'
            "\n"
            "\n"
            "def main(argv=None):\n"
            "    argv = sys.argv[1:] if argv is None else argv\n"
            '    text = argv[-1] if argv else ""\n'
            "    return format_plain(word_counts(text))\n"
        ),
        "/repo/tests/test_cli.py": (
            "import json\n"
            "\n"
            "from textstats.cli import main\n"
            "\n"
            "\n"
            "def test_plain():\n"
            '    assert main(["b a b"]) == "a=1\\nb=2"\n'
            "\n"
            "\n"
            "def test_json_flag():\n"
            '    assert json.loads(main(["--json", "b a b"])) == {"a": 1, "b": 2}\n'
            "\n"
            "\n"
            "def test_counts_ignore_case():\n"
            '    assert main(["A a"]) == "a=2"\n'
        ),
    },
    check=CHECK_TESTS,
    # Adding one blank line to the test file looks like tidying, and the tests
    # still pass. But it breaks the rule "don't edit the tests" (the fingerprint
    # check fails), and the agent can't notice or undo it.
    planted_mistake={
        "command": "printf '\\n' >> /repo/tests/test_cli.py",
        "thought": "Tidy up: make sure the test file ends with a clean newline.",
    },
)

TASKS = {t.name: t for t in [TEXTSTATS]}
