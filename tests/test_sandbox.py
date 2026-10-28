"""The folder-tracking trick in sandbox.py, tested with local bash. No network."""
import subprocess

from rootstep.sandbox import unwrap, wrap


def run_local(command: str, cwd: str) -> tuple[str, int]:
    p = subprocess.run(["sh", "-c", wrap(command, cwd)], capture_output=True, text=True)
    return p.stdout, p.returncode


def test_cd_is_remembered(tmp_path):
    (tmp_path / "src").mkdir()
    stdout, code = run_local("cd src && echo hi", str(tmp_path))
    output, cwd = unwrap(stdout, "/")
    assert code == 0
    assert output.strip() == "hi"
    assert cwd.endswith("/src")


def test_exit_code_and_stderr_kept(tmp_path):
    stdout, code = run_local("echo oops >&2; exit 3", str(tmp_path))
    output, cwd = unwrap(stdout, "/fallback")
    assert code == 3
    assert "oops" in output
    assert cwd == "/fallback"      # `exit` skips the marker, so we keep the old folder


def test_heredoc_command(tmp_path):
    stdout, code = run_local("cat > note.txt <<'EOF'\nhello\nEOF\ncat note.txt", str(tmp_path))
    output, _ = unwrap(stdout, "/")
    assert code == 0
    assert output.strip() == "hello"


def test_missing_folder_falls_back_to_root():
    stdout, _ = run_local("pwd", "/definitely/not/here")
    output, _ = unwrap(stdout, "/x")
    assert output.strip() == "/"
