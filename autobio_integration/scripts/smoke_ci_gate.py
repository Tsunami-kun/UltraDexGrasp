from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts import assert_run_consistency


DEFAULT_GOOD_RUN = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "smoke_ci_fixture"
    / "centrifuge_lid_open_mini"
    / "fixture_run"
)


def _find_input_root(run_dir: Path) -> Path:
    if run_dir.parent.parent.name == "artifacts":
        return run_dir.parent
    return run_dir.parent.parent


def _apply_failure_mode(broken_dir: Path, mode: str) -> str:
    if mode == "missing_trajectory":
        broken_path = broken_dir / "episodes" / "episode_000000" / "trajectory.npz"
        if not broken_path.exists():
            raise FileNotFoundError(f"Expected trajectory artifact to delete is missing: {broken_path}")
        broken_path.unlink()
        return f"deleted {broken_path.name}"

    if mode == "metadata_seed_mismatch":
        metadata_path = broken_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, list) or not metadata:
            raise ValueError(f"metadata.json must be a non-empty list: {metadata_path}")
        metadata[0]["seed"] = int(metadata[0].get("seed", 0)) + 999
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return "mutated metadata seed for episode 0"

    raise ValueError(f"Unsupported failure mode: {mode}")


def _expect_failure(run_dir: Path, expected_episodes: int, mode: str) -> int:
    with tempfile.TemporaryDirectory(prefix=f"autobio_smoke_ci_{mode}_") as tmpdir:
        broken_dir = Path(tmpdir) / run_dir.name
        shutil.copytree(run_dir, broken_dir)
        mutation = _apply_failure_mode(broken_dir, mode)
        print(f"Smoke/CI gate failure fixture ({mode}): {broken_dir} [{mutation}]")
        rc = assert_run_consistency.main(
            ["--run-dir", str(broken_dir), "--expected-episodes", str(expected_episodes)]
        )
        if rc == 0:
            print(f"Smoke/CI gate failure fixture unexpectedly passed for mode={mode}")
            return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a lightweight smoke/CI gate over a known-good run plus one intentional failure case."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_GOOD_RUN),
        help="Path to a known-good run directory with summary.json/metadata.json/episodes",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=5,
        help="Expected episode count for the successful run assertion.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip checking that a summary/evaluate-style root exists for the run.",
    )
    parser.add_argument(
        "--failure-modes",
        nargs="+",
        default=["missing_trajectory", "metadata_seed_mismatch"],
        help="Failure injection modes that must fail during the gate.",
    )
    parser.add_argument(
        "--force-gate-failure-mode",
        type=str,
        default=None,
        choices=["missing_trajectory", "metadata_seed_mismatch"],
        help="Force the overall gate to fail after the normal checks using one diagnostic failure mode.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Smoke/CI gate fixture run not found: {run_dir}")
        return 1

    if not args.skip_evaluate:
        input_root = _find_input_root(run_dir)
        if not input_root.exists():
            print(f"Derived evaluate input root does not exist: {input_root}")
            return 1
        print(f"Smoke/CI gate evaluate root: {input_root}")
        evaluate_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "cli.py"),
            "evaluate",
            "--input-dir",
            str(input_root),
            "--task",
            run_dir.parent.name,
        ]
        evaluate_rc = subprocess.run(evaluate_cmd, check=False).returncode
        if evaluate_rc != 0:
            print(f"Smoke/CI gate evaluate command failed with exit code {evaluate_rc}")
            return evaluate_rc

    print(f"Smoke/CI gate success fixture: {run_dir}")
    ok = assert_run_consistency.main(
        ["--run-dir", str(run_dir), "--expected-episodes", str(args.expected_episodes)]
    )
    if ok != 0:
        print("Smoke/CI gate failed on success fixture")
        return ok

    for mode in args.failure_modes:
        rc = _expect_failure(run_dir, args.expected_episodes, mode)
        if rc != 0:
            return 1

    if args.force_gate_failure_mode:
        print(f"Forcing overall gate failure with mode={args.force_gate_failure_mode}")
        rc = _expect_failure(run_dir, args.expected_episodes, args.force_gate_failure_mode)
        if rc != 0:
            return rc
        return 1

    print("Smoke/CI gate passed: success and failure assertions behaved as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
