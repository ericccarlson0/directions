"""scripts/stages.py: several stages as one job, each its own process, stopping at the first failure."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stages.py"
sys.path.insert(0, str(SCRIPT.parent))
import stages  # noqa: E402


def test_stage_argv_runs_uv_commands_without_syncing_and_others_as_given():
    assert stages.stage_argv("uv run directions pilot --config a.yaml --run-id 'x y'") == \
        ["uv", "run", "--no-sync", "directions", "pilot", "--config", "a.yaml", "--run-id", "x y"]
    assert stages.stage_argv("uv run --no-sync python s.py") == ["uv", "run", "--no-sync", "python", "s.py"]
    assert stages.stage_argv("python3 -c 'print(1)'") == ["python3", "-c", "print(1)"]


def test_stages_run_in_order_and_stop_at_the_first_failure(tmp_path):
    py = sys.executable
    marks = tmp_path / "marks.txt"
    ok = f"{py} -c \"open(r'{marks}', 'a').write('a\\n')\""
    bad = f"{py} -c \"import sys; open(r'{marks}', 'a').write('b\\n'); sys.exit(3)\""
    never = f"{py} -c \"open(r'{marks}', 'a').write('c\\n')\""
    r = subprocess.run([py, str(SCRIPT), ok, bad, never], capture_output=True, text=True)
    assert r.returncode == 3 and marks.read_text() == "a\nb\n"
    assert "[stages] 1/3" in r.stdout and "exit 3" in r.stdout and "3/3" not in r.stdout
    assert subprocess.run([py, str(SCRIPT)], capture_output=True).returncode == 2
    assert subprocess.run([py, str(SCRIPT), ok, ok], capture_output=True).returncode == 0
