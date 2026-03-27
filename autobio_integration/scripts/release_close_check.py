from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate release ledger/catalog state and one canonical-path consumer invocation"
    )
    parser.add_argument("--repo-root", type=str, default=None, help="Repository root; defaults to script parent")
    parser.add_argument("--release-tag", type=str, default=None, help="Override the release tag instead of reading artifacts/releases/LATEST")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file for the downstream consumer output")
    parser.add_argument("--config", type=str, default=None, help="Optional config path forwarded to cli.py")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    releases_root = repo_root / "artifacts" / "releases"
    latest_path = releases_root / "LATEST"
    catalog_path = releases_root / "catalog.json"

    if not latest_path.exists():
        print(f"Missing LATEST file: {latest_path}")
        return 1
    if not catalog_path.exists():
        print(f"Missing release catalog: {catalog_path}")
        return 1

    latest_tag = args.release_tag or latest_path.read_text(encoding="utf-8").strip()
    if not latest_tag:
        print("LATEST file is empty")
        return 1

    catalog = _load_json(catalog_path)
    if catalog.get("latest_release_tag") != latest_tag:
        print(
            f"Catalog latest_release_tag mismatch: expected {latest_tag}, "
            f"found {catalog.get('latest_release_tag')}"
        )
        return 1

    entry = None
    for candidate in catalog.get("releases", []):
        if candidate.get("release_tag") == latest_tag:
            entry = candidate
            break
    if entry is None:
        print(f"Release tag {latest_tag} not found in {catalog_path}")
        return 1

    canonical_root = releases_root / latest_tag
    expected_root = str(canonical_root.relative_to(repo_root))
    if entry.get("canonical_storage_root") != expected_root:
        print(
            f"Catalog canonical root mismatch: expected {expected_root}, "
            f"found {entry.get('canonical_storage_root')}"
        )
        return 1

    manifest_path = repo_root / entry["manifest"]
    retrieval_path = repo_root / entry["retrieval_verification"]
    approval_path = repo_root / entry["approval"]
    for required in (manifest_path, retrieval_path, approval_path):
        if not required.exists():
            print(f"Missing required release metadata file: {required}")
            return 1

    retrieval = _load_json(retrieval_path)
    if not retrieval.get("all_retrievable"):
        print(f"Retrieval verification failed in {retrieval_path}")
        return 1
    if not retrieval.get("all_sha256_match"):
        print(f"SHA-256 verification failed in {retrieval_path}")
        return 1

    approval = _load_json(approval_path)
    if approval.get("approval_status") != "approved":
        print(f"Release approval status is not approved in {approval_path}")
        return 1

    demo_root = canonical_root / "demos"
    if not demo_root.exists():
        print(f"Missing canonical demo root: {demo_root}")
        return 1

    cmd = [sys.executable, str(repo_root / "cli.py")]
    if args.config:
        cmd += ["--config", str(Path(args.config).resolve())]
    cmd += ["evaluate", "--input-dir", str(demo_root)]

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = canonical_root / "release_close_check.log"
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")

    if completed.returncode != 0:
        print(f"Downstream consumer failed; see {log_path}")
        return completed.returncode
    if "No evaluation summaries found" in (completed.stdout or ""):
        print(f"Downstream consumer returned no summaries; see {log_path}")
        return 1

    print(f"Release close check passed for {latest_tag}")
    print(f"Catalog: {catalog_path}")
    print(f"Canonical root: {canonical_root}")
    print(f"Consumer log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
