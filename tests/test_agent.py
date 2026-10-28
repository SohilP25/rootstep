"""Checks for the parts you write in agent.py and record.py. No network, no cost."""
import json

import pytest

from rootstep.agent import OBS_HEAD, OBS_TAIL, parse_action, shorten
from rootstep.record import Run


def test_parse_bash_block():
    assert parse_action("THOUGHT: look around\n```bash\nls -la\n```") == "ls -la"


def test_parse_multiline_block():
    reply = "THOUGHT: write a file\n```bash\ncat > a.txt <<'EOF'\nhi\nEOF\n```"
    assert parse_action(reply) == "cat > a.txt <<'EOF'\nhi\nEOF"


def test_parse_thought_inside_box():
    # seen in a real run (382be4fa, steps 5 and 8): the THOUGHT line was put inside the box
    reply = ("```bash\nTHOUGHT: The tests are now passing! Let me verify.\n```\n"
             "```bash\npython -m pytest -q 2>&1\n```")
    assert parse_action(reply) == "python -m pytest -q 2>&1"


def test_parse_takes_first_command_when_many():
    # seen at step 8: the model planned several commands in one reply
    reply = ("```bash\nTHOUGHT: check files\n```\n```bash\ncat textstats/cli.py\n```\n"
             "```bash\nTHOUGHT: next\n```\n```bash\ncat textstats/core.py\n```")
    assert parse_action(reply) == "cat textstats/cli.py"


def test_parse_only_thought_in_box_is_bad_format():
    with pytest.raises(ValueError):
        parse_action("```bash\nTHOUGHT: I am done thinking\n```")


def test_parse_finish():
    assert parse_action("THOUGHT: the tests pass now\nFINISH") is None


def test_parse_missing_block():
    with pytest.raises(ValueError):
        parse_action("I think we should run ls")


def test_shorten_empty():
    assert shorten("") == "(no output)"


def test_shorten_short_output_unchanged():
    assert shorten("hello") == "hello"


def test_shorten_long_output():
    text = "a" * OBS_HEAD + "b" * 5000 + "c" * OBS_TAIL
    out = shorten(text)
    assert out.startswith("a" * OBS_HEAD)
    assert out.endswith("c" * OBS_TAIL)
    assert "5000" in out          # the note says how much was cut
    assert "b" * 100 not in out   # the middle is gone


def test_run_save(tmp_path):
    run = Run(run_id="abc", task="demo", config={}, start_snapshot="s0")
    path = run.save(tmp_path / "runs")
    assert path.name == "abc.json"
    assert json.loads(path.read_text())["task"] == "demo"
