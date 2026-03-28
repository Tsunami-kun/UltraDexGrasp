from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert generate/evaluate/artifact consistency for one task run directory."
    )
    parser.add_argument("--run-dir", required=True, help="Path to one run directory containing summary.json")
    parser.add_argument("--expected-episodes", type=int, default=None, help="Optional expected episode count")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    config_path = run_dir / "config.json"

    for required in (summary_path, metadata_path, config_path):
        if not required.exists():
            print(f"Missing required run file: {required}")
            return 1

    summary = _load_json(summary_path)
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, list):
        print(f"metadata.json must be a list: {metadata_path}")
        return 1

    total = summary.get("total_episodes")
    successful = summary.get("successful")
    success_rate = summary.get("success_rate")

    if not isinstance(total, int) or total < 0:
        print(f"Invalid total_episodes in {summary_path}: {total}")
        return 1
    if args.expected_episodes is not None and total != args.expected_episodes:
        print(
            f"Episode count mismatch: expected {args.expected_episodes}, "
            f"summary reports {total}"
        )
        return 1
    if total != len(metadata):
        print(
            f"Metadata length mismatch: summary reports {total}, "
            f"metadata has {len(metadata)} entries"
        )
        return 1

    computed_successes = sum(1 for item in metadata if item.get("success"))
    if successful != computed_successes:
        print(
            f"Success count mismatch: summary reports {successful}, "
            f"metadata implies {computed_successes}"
        )
        return 1

    computed_rate = (computed_successes / total) if total else 0.0
    if abs(float(success_rate) - computed_rate) > 1e-9:
        print(
            f"Success rate mismatch: summary reports {success_rate}, "
            f"computed rate is {computed_rate}"
        )
        return 1

    for index, item in enumerate(metadata):
        episode_dir = run_dir / "episodes" / f"episode_{index:06d}"
        info_path = episode_dir / "info.json"
        episode_path = episode_dir / "episode.json"
        trajectory_path = episode_dir / "trajectory.npz"
        for required in (info_path, episode_path, trajectory_path):
            if not required.exists():
                print(f"Missing required episode artifact: {required}")
                return 1

        info = _load_json(info_path)
        if info.get("task") != item.get("task"):
            print(
                f"Task mismatch for episode {index}: "
                f"info={info.get('task')} metadata={item.get('task')}"
            )
            return 1
        if info.get("seed") != item.get("seed"):
            print(
                f"Seed mismatch for episode {index}: "
                f"info={info.get('seed')} metadata={item.get('seed')}"
            )
            return 1
        if bool(info.get("success")) != bool(item.get("success")):
            print(
                f"Success mismatch for episode {index}: "
                f"info={info.get('success')} metadata={item.get('success')}"
            )
            return 1
        if info.get("num_steps") != item.get("num_steps"):
            print(
                f"Step mismatch for episode {index}: "
                f"info={info.get('num_steps')} metadata={item.get('num_steps')}"
            )
            return 1

    print(f"Run consistency check passed for {run_dir}")
    print(f"episodes={total} successes={computed_successes} success_rate={computed_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
