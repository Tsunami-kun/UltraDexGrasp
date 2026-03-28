from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import assert_run_consistency


class AssertRunConsistencyTests(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _build_run(self, root: Path, *, episodes: int = 2, include_last_trajectory: bool = True) -> Path:
        run_dir = root / "run"
        summary = {
            "total_episodes": episodes,
            "successful": episodes,
            "success_rate": 1.0,
        }
        metadata = []
        for idx in range(episodes):
            metadata.append(
                {
                    "task": "fixture_task",
                    "seed": idx,
                    "success": True,
                    "num_steps": 10,
                }
            )
        self._write_json(run_dir / "summary.json", summary)
        self._write_json(run_dir / "metadata.json", metadata)
        self._write_json(run_dir / "config.json", {"task": "fixture_task"})

        for idx in range(episodes):
            episode_dir = run_dir / "episodes" / f"episode_{idx:06d}"
            self._write_json(
                episode_dir / "info.json",
                {
                    "task": "fixture_task",
                    "seed": idx,
                    "success": True,
                    "num_steps": 10,
                },
            )
            self._write_json(episode_dir / "episode.json", {"index": idx})
            if include_last_trajectory or idx < episodes - 1:
                (episode_dir / "trajectory.npz").write_bytes(b"fixture")
        return run_dir

    def test_successful_run_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._build_run(Path(tmpdir), episodes=2, include_last_trajectory=True)
            rc = assert_run_consistency.main(
                ["--run-dir", str(run_dir), "--expected-episodes", "2"]
            )
            self.assertEqual(rc, 0)

    def test_missing_episode_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._build_run(Path(tmpdir), episodes=2, include_last_trajectory=False)
            rc = assert_run_consistency.main(
                ["--run-dir", str(run_dir), "--expected-episodes", "2"]
            )
            self.assertEqual(rc, 1)

    def test_metadata_seed_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._build_run(Path(tmpdir), episodes=2, include_last_trajectory=True)
            metadata_path = run_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata[0]["seed"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            rc = assert_run_consistency.main(
                ["--run-dir", str(run_dir), "--expected-episodes", "2"]
            )
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
