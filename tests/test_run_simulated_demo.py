from pathlib import Path
import subprocess


def test_run_simulated_demo(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "run_simulated_demo.py"
    python = repo_root / ".venv" / "bin" / "python"

    result = subprocess.run(
        [
            str(python),
            str(script),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wrote demo .snpdat files to" in result.stdout
