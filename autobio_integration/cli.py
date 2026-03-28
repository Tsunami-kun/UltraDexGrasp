#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import load_config, setup_python_path, get_runtime_paths


def _common_parent(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")


def _evaluate_runs(root: Path, task: str | None) -> int:
    if task:
        task_roots = [root / task]
    else:
        task_roots = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []

    summaries = []
    for task_root in task_roots:
        if not task_root.exists():
            continue
        for run_dir in sorted([p for p in task_root.iterdir() if p.is_dir()]):
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            with summary_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            summaries.append((task_root.name, run_dir.name, data))

    if not summaries:
        print(f"No evaluation summaries found under {root}")
        return 1

    print(f"Evaluation root: {root}")
    for task_name, run_name, data in summaries:
        print(
            f"{task_name}/{run_name}: total={data.get('total_episodes', 0)} "
            f"success={data.get('successful', 0)} "
            f"rate={data.get('success_rate', 0):.3f}"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Single entry CLI for AutoBio integration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process AutoBio meshes")
    _common_parent(process_parser)
    process_parser.add_argument("--autobio-root", type=str, default=None)
    process_parser.add_argument("--output-root", type=str, default=None)
    process_parser.add_argument("--max-workers", type=int, default=8)
    process_parser.add_argument("--manifest-file", type=str, default=None)
    process_parser.add_argument("--checksum-report", type=str, default=None)

    validate_parser = subparsers.add_parser("validate", help="Validate imports and asset roots")
    _common_parent(validate_parser)

    smoke_parser = subparsers.add_parser("smoke-test", help="Run physics smoke tests")
    _common_parent(smoke_parser)

    task_contract_parser = subparsers.add_parser("task-contract-smoke", help="Run deterministic task contract smoke tests")
    _common_parent(task_contract_parser)
    task_contract_parser.add_argument("--seed", type=int, default=7)
    task_contract_parser.add_argument("--tasks", nargs="+", default=None)

    workflow_parser = subparsers.add_parser("workflow", help="Run a scripted multi-step workflow")
    _common_parent(workflow_parser)
    workflow_parser.add_argument("--workflow", type=str, default="sample_stage_and_cycle")
    workflow_parser.add_argument("--seed", type=int, default=0)
    workflow_parser.add_argument("--retries", type=int, default=1)
    workflow_parser.add_argument("--output-dir", type=str, default=None)
    workflow_parser.add_argument("--num-runs", type=int, default=1)
    workflow_parser.add_argument("--resume-dir", type=str, default=None)

    generate_parser = subparsers.add_parser("generate", help="Generate demonstrations")
    _common_parent(generate_parser)
    generate_parser.add_argument("--task", type=str, default="pickup_tube")
    generate_parser.add_argument("--num-episodes", type=int, default=None)
    generate_parser.add_argument("--num-cpu-workers", type=int, default=None)
    generate_parser.add_argument("--gpu-id", type=int, default=None)
    generate_parser.add_argument("--output-dir", type=str, default=None)
    generate_parser.add_argument("--batch-grasp-size", type=int, default=64)
    generate_parser.add_argument("--max-grasps-per-object", type=int, default=500)
    generate_parser.add_argument("--seed-start", type=int, default=0)
    generate_parser.add_argument("--controller-mode", type=str, choices=["direct", "grasp_planned", "reactive"], default="direct")

    eval_parser = subparsers.add_parser("evaluate", help="Summarize generated runs")
    _common_parent(eval_parser)
    eval_parser.add_argument("--task", type=str, default=None)
    eval_parser.add_argument("--input-dir", type=str, default=None)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark sweeps with aggregate metrics")
    _common_parent(benchmark_parser)
    benchmark_parser.add_argument("--tasks", nargs="+", default=None)
    benchmark_parser.add_argument("--episodes-per-task", type=int, default=5)
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--output-dir", type=str, default=None)
    benchmark_parser.add_argument("--save-episodes", action="store_true")
    benchmark_parser.add_argument("--resume-dir", type=str, default=None)
    benchmark_parser.add_argument("--repair-existing-tasks", action="store_true")

    export_parser = subparsers.add_parser("export-dataset", help="Export generated episodes into a standard dataset manifest")
    _common_parent(export_parser)
    export_parser.add_argument("--input-dir", type=str, required=True)
    export_parser.add_argument("--output-dir", type=str, required=True)

    train_parser = subparsers.add_parser("train-baseline", help="Train a simple baseline from exported dataset artifacts")
    _common_parent(train_parser)
    train_parser.add_argument("--dataset-dir", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)

    serve_parser = subparsers.add_parser("serve-policy", help="Serve a trained baseline model over HTTP")
    _common_parent(serve_parser)
    serve_parser.add_argument("--model", type=str, required=True)
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    eval_policy_parser = subparsers.add_parser("eval-policy", help="Evaluate local or remote policy predictions against exported episodes")
    _common_parent(eval_policy_parser)
    eval_policy_parser.add_argument("--dataset-dir", type=str, required=True)
    eval_policy_parser.add_argument("--model", type=str, default=None)
    eval_policy_parser.add_argument("--server-url", type=str, default=None)
    eval_policy_parser.add_argument("--output", type=str, required=True)

    dashboard_parser = subparsers.add_parser("dashboard", help="Render a static dashboard from benchmark/workflow outputs")
    _common_parent(dashboard_parser)
    dashboard_parser.add_argument("--benchmark-report", type=str, default=None)
    dashboard_parser.add_argument("--workflow-summary", type=str, default=None)
    dashboard_parser.add_argument("--output", type=str, required=True)

    ablation_parser = subparsers.add_parser("controller-ablation", help="Run controller ablations for supported task/controller combinations")
    _common_parent(ablation_parser)
    ablation_parser.add_argument("--task", type=str, default="pickup_tube")
    ablation_parser.add_argument("--modes", nargs="+", default=["scripted", "grasp_planned", "reactive"])
    ablation_parser.add_argument("--scenarios", nargs="+", default=["seen", "perturbed"])
    ablation_parser.add_argument("--num-episodes", type=int, default=3)
    ablation_parser.add_argument("--seed", type=int, default=0)
    ablation_parser.add_argument("--generate-timeout-sec", type=int, default=180)
    ablation_parser.add_argument("--output-dir", type=str, default=None)

    release_close_parser = subparsers.add_parser(
        "release-close-check",
        help="Validate release ledger/catalog state and one canonical-path consumer invocation",
    )
    _common_parent(release_close_parser)
    release_close_parser.add_argument("--release-tag", type=str, default=None)
    release_close_parser.add_argument("--log-file", type=str, default=None)

    smoke_ci_parser = subparsers.add_parser(
        "smoke-ci-gate",
        help="Run the lightweight smoke/CI quality gate over a known-good run and one forced failure case",
    )
    _common_parent(smoke_ci_parser)
    smoke_ci_parser.add_argument("--run-dir", type=str, default=None)
    smoke_ci_parser.add_argument("--expected-episodes", type=int, default=2)
    smoke_ci_parser.add_argument("--skip-evaluate", action="store_true")
    smoke_ci_parser.add_argument(
        "--failure-modes",
        nargs="+",
        default=None,
        choices=["missing_trajectory", "metadata_seed_mismatch"],
    )
    smoke_ci_parser.add_argument(
        "--force-gate-failure-mode",
        type=str,
        default=None,
        choices=["missing_trajectory", "metadata_seed_mismatch"],
    )

    run_consistency_parser = subparsers.add_parser(
        "assert-run-consistency",
        help="Assert one generated task run has consistent summary, metadata, and episode artifacts",
    )
    _common_parent(run_consistency_parser)
    run_consistency_parser.add_argument("--run-dir", type=str, required=True)
    run_consistency_parser.add_argument("--expected-episodes", type=int, default=None)

    args = parser.parse_args(argv)
    if args.config:
        os.environ["AUTOBIO_INTEGRATION_CONFIG"] = str(Path(args.config).resolve())
    paths = setup_python_path(args.config)
    config = load_config(args.config)
    os.environ.setdefault("DISPLAY", config["devices"].get("display", ""))

    if args.command == "process":
        from scripts import process_autobio_meshes

        cmd = [
            "--config", str(paths.config_path),
            "--autobio_root", args.autobio_root or str(paths.autobio_root),
            "--output_root", args.output_root or str(paths.processed_assets),
            "--max_workers", str(args.max_workers),
        ]
        if args.manifest_file:
            cmd += ["--manifest_file", args.manifest_file]
        if args.checksum_report:
            cmd += ["--checksum_report", args.checksum_report]
        process_autobio_meshes.main(cmd)
        return 0

    if args.command == "validate":
        from scripts import validate_setup

        validate_setup.main(["--config", str(paths.config_path)])
        return 0

    if args.command == "smoke-test":
        from scripts import smoke_test

        smoke_test.main(["--config", str(paths.config_path)])
        return 0

    if args.command == "task-contract-smoke":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "task_contract_smoke.py"),
            "--config",
            str(paths.config_path),
            "--seed",
            str(args.seed),
        ]
        if args.tasks:
            cmd += ["--tasks", *args.tasks]
        return subprocess.run(cmd, check=False).returncode

    if args.command == "workflow":
        from scripts import run_workflow

        cmd = [
            "--config", str(paths.config_path),
            "--workflow", args.workflow,
            "--seed", str(args.seed),
            "--retries", str(args.retries),
            "--num-runs", str(args.num_runs),
        ]
        if args.output_dir:
            cmd += ["--output-dir", args.output_dir]
        if args.resume_dir:
            cmd += ["--resume-dir", args.resume_dir]
        return run_workflow.main(cmd)

    if args.command == "generate":
        from scripts import parallel_demo_synthesis

        cmd = ["--config", str(paths.config_path), "--task", args.task]
        if args.num_episodes is not None:
            cmd += ["--num_episodes", str(args.num_episodes)]
        if args.num_cpu_workers is not None:
            cmd += ["--num_cpu_workers", str(args.num_cpu_workers)]
        if args.gpu_id is not None:
            cmd += ["--gpu_id", str(args.gpu_id)]
        if args.output_dir is not None:
            cmd += ["--output_dir", args.output_dir]
        cmd += [
            "--batch_grasp_size", str(args.batch_grasp_size),
            "--max_grasps_per_object", str(args.max_grasps_per_object),
            "--seed_start", str(args.seed_start),
            "--controller_mode", str(args.controller_mode),
        ]
        parallel_demo_synthesis.main(cmd)
        return 0

    if args.command == "evaluate":
        input_root = Path(args.input_dir).resolve() if args.input_dir else paths.demo_output
        return _evaluate_runs(input_root, args.task)

    if args.command == "benchmark":
        from scripts import benchmark_tasks

        cmd = ["--config", str(paths.config_path), "--episodes-per-task", str(args.episodes_per_task), "--seed", str(args.seed)]
        if args.tasks:
            cmd += ["--tasks", *args.tasks]
        if args.output_dir:
            cmd += ["--output-dir", args.output_dir]
        if args.save_episodes:
            cmd += ["--save-episodes"]
        if args.resume_dir:
            cmd += ["--resume-dir", args.resume_dir]
        if args.repair_existing_tasks:
            cmd += ["--repair-existing-tasks"]
        return benchmark_tasks.main(cmd)

    if args.command == "export-dataset":
        from scripts import export_dataset

        return export_dataset.main(["--input-dir", args.input_dir, "--output-dir", args.output_dir])

    if args.command == "train-baseline":
        from scripts import train_baseline

        return train_baseline.main(["--dataset-dir", args.dataset_dir, "--output-dir", args.output_dir])

    if args.command == "serve-policy":
        from scripts import serve_policy

        return serve_policy.main(["--model", args.model, "--host", args.host, "--port", str(args.port)])

    if args.command == "eval-policy":
        from scripts import evaluate_policy

        cmd = ["--dataset-dir", args.dataset_dir, "--output", args.output]
        if args.model:
            cmd += ["--model", args.model]
        if args.server_url:
            cmd += ["--server-url", args.server_url]
        return evaluate_policy.main(cmd)

    if args.command == "dashboard":
        from scripts import render_dashboard

        cmd = ["--output", args.output]
        if args.benchmark_report:
            cmd += ["--benchmark-report", args.benchmark_report]
        if args.workflow_summary:
            cmd += ["--workflow-summary", args.workflow_summary]
        return render_dashboard.main(cmd)

    if args.command == "controller-ablation":
        from scripts import run_controller_ablations

        cmd = [
            "--task", args.task,
            "--num-episodes", str(args.num_episodes),
            "--seed", str(args.seed),
            "--generate-timeout-sec", str(args.generate_timeout_sec),
            "--modes", *args.modes,
            "--scenarios", *args.scenarios,
        ]
        if args.output_dir:
            cmd += ["--output-dir", args.output_dir]
        return run_controller_ablations.main(cmd)

    if args.command == "release-close-check":
        from scripts import release_close_check

        cmd = ["--repo-root", str(paths.repo_root)]
        if args.release_tag:
            cmd += ["--release-tag", args.release_tag]
        if args.log_file:
            cmd += ["--log-file", args.log_file]
        if args.config:
            cmd += ["--config", str(Path(args.config).resolve())]
        return release_close_check.main(cmd)

    if args.command == "smoke-ci-gate":
        from scripts import smoke_ci_gate

        cmd = []
        if args.run_dir:
            cmd += ["--run-dir", args.run_dir]
        if args.expected_episodes is not None:
            cmd += ["--expected-episodes", str(args.expected_episodes)]
        if args.skip_evaluate:
            cmd += ["--skip-evaluate"]
        if args.failure_modes:
            cmd += ["--failure-modes", *args.failure_modes]
        if args.force_gate_failure_mode:
            cmd += ["--force-gate-failure-mode", args.force_gate_failure_mode]
        return smoke_ci_gate.main(cmd)

    if args.command == "assert-run-consistency":
        from scripts import assert_run_consistency

        cmd = ["--run-dir", args.run_dir]
        if args.expected_episodes is not None:
            cmd += ["--expected-episodes", str(args.expected_episodes)]
        return assert_run_consistency.main(cmd)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
