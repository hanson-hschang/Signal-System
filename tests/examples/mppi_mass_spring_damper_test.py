import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
MODULE = "examples.control.mppi_mass_spring_damper"


def run_example(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", MODULE, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class TestMPPIMassSpringDamperExample:
    def test_runs_with_multiple_batches_without_writing(self) -> None:
        result = run_example(
            "--duration",
            "0.04",
            "--time-step",
            "0.02",
            "--horizon",
            "4",
            "--num-rollouts",
            "8",
            "--batch-size",
            "2",
        )

        assert result.returncode == 0, result.stderr
        assert "final_state=" in result.stdout
        assert "total_running_cost=" in result.stdout

    def test_saves_multibatch_plot_and_animation(self, tmp_path: Path) -> None:
        result = run_example(
            "--duration",
            "0.04",
            "--time-step",
            "0.02",
            "--horizon",
            "4",
            "--num-rollouts",
            "8",
            "--batch-size",
            "2",
            "--save-dir",
            str(tmp_path),
        )
        plot_path = tmp_path / "mppi_mass_spring_damper_plot.png"
        animation_path = tmp_path / "mppi_mass_spring_damper.mp4"

        assert result.returncode == 0, result.stderr
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0
        assert animation_path.is_file()
        assert animation_path.stat().st_size > 0
