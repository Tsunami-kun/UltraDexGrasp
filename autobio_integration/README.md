# AutoBio Integration

This repository now exposes one config-driven CLI: `python cli.py`.

## Supported Happy Path

The supported layout is:

- this repo at `.../UltraDexGrasp/autobio_integration`
- UltraDexGrasp root at `..`
- AutoBio root at `../../AutoBio`

If your layout differs, set `AUTOBIO_ROOT` and `ULTRADEX_ROOT` explicitly.

Current supported deliverable:
- clean bootstrap via `scripts/bootstrap_clean_env.sh`
- asset processing via `python cli.py process`
- environment validation via `python cli.py validate`
- smoke testing via `python cli.py smoke-test`
- demo generation/evaluation via `python cli.py generate` and `python cli.py evaluate`
- six supported benchmark tasks:
  - `tube_transfer`
  - `thermal_cycler_open`
  - `thermal_cycler_close`
  - `centrifuge_lid_open_mini`
  - `pickup_tube`
  - `centrifuge_lid_open_5430`
- one supported multi-step workflow:
  - `supported_cycle_sequence`
- supported controller-ablation tasks:
  - `pickup_tube`
  - `tube_transfer`
  - `thermal_cycler_open`
  - `thermal_cycler_close`
  - `centrifuge_lid_open_mini`
  - `centrifuge_lid_open_5430`
  - `scripted`
  - `grasp_planned`
  - `reactive`
  - `seen`
  - `perturbed`
- dataset export, baseline training/evaluation, and static dashboard generation

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export AUTOBIO_ROOT="$(realpath ../../AutoBio)"
export ULTRADEX_ROOT="$(realpath ..)"
```

Or bootstrap a clean validation environment in one script:

```bash
bash scripts/bootstrap_clean_env.sh
```

For the non-fallback dexterous stack, use the dedicated installer:

```bash
bash scripts/install_dexterous_stack.sh
```

## Commands

Validate the environment and required roots:

```bash
python cli.py validate
```

Process raw AutoBio meshes into SAPIEN-ready assets:

```bash
python cli.py process --max-workers 8
```

Run a physics smoke test against the processed assets:

```bash
python cli.py smoke-test
```

Generate a small demo batch for the validated scripted task:

```bash
python cli.py generate --task thermal_cycler_close --num-episodes 2 --num-cpu-workers 1
```

Summarize generated runs:

```bash
python cli.py evaluate --task thermal_cycler_close
```

Run the common task-contract smoke test:

```bash
python cli.py task-contract-smoke --seed 11
```

Run a benchmark sweep with aggregate success, contact, collision, and diversity metrics:

```bash
AUTOBIO_DISABLE_RENDER=1 AUTOBIO_FORCE_HEADLESS_TASKS=1 \
python cli.py benchmark --tasks tube_transfer thermal_cycler_open thermal_cycler_close --episodes-per-task 10
```

Run a scripted multi-step workflow:

```bash
AUTOBIO_FORCE_HEADLESS_TASKS=1 python cli.py workflow --workflow sample_stage_and_cycle
```

If your current shell `python` differs from the validated runtime used for task
subprocesses, pin the nested interpreter explicitly:

```bash
AUTOBIO_PYTHON_EXECUTABLE="$(pwd)/artifacts/bootstrap_env_readme/bin/python" \
AUTOBIO_FORCE_HEADLESS_TASKS=1 python cli.py workflow --workflow supported_cycle_sequence
```

Export generated episodes into a standard dataset manifest, train a simple baseline
hook, evaluate it locally or via a remote HTTP server, and render a static dashboard:

```bash
python cli.py export-dataset --input-dir artifacts/nonfallback_acceptance --output-dir artifacts/exports/nonfallback_acceptance_v1
python cli.py train-baseline --dataset-dir artifacts/exports/nonfallback_acceptance_v1 --output-dir artifacts/training/nonfallback_baseline_v1
python cli.py eval-policy --dataset-dir artifacts/exports/nonfallback_acceptance_v1 --model artifacts/training/nonfallback_baseline_v1/baseline_model.json --output artifacts/training/nonfallback_baseline_v1/eval_summary.json
python cli.py serve-policy --model artifacts/training/nonfallback_baseline_v1/baseline_model.json
python cli.py dashboard --workflow-summary artifacts/nonfallback_workflow/supported_cycle_sequence/20260327_160841/workflow_summary.json --output artifacts/dashboards/nonfallback_dashboard.html
```

Run the current supported controller-ablation probe for `pickup_tube`:

```bash
python cli.py controller-ablation --task pickup_tube --modes scripted grasp_planned --scenarios seen perturbed --num-episodes 1 --generate-timeout-sec 120
```

The current probe has validated successful `scripted` and `grasp_planned`
modes for `pickup_tube` when run in the full `ultradexgrasp` stack, including a
single public-CLI report covering `seen` and `perturbed` settings:
[ablation_report.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/ablations/pickup_tube/20260327_220839/ablation_report.json).
An updated all-modes public report is also checked in at
[ablation_report.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/ablations/pickup_tube/20260327_221733/ablation_report.json),
where `scripted`, `grasp_planned`, and `reactive` all succeed for `seen` and
`perturbed` `pickup_tube` probes. Additional successful public reports now exist
for [tube_transfer](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/ablations/tube_transfer/20260327_222724/ablation_report.json)
and [thermal_cycler_open](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/ablations/thermal_cycler_open/20260327_223655/ablation_report.json),
plus [thermal_cycler_close](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/ablations/thermal_cycler_close/20260327_224610/ablation_report.json),
and [centrifuge_lid_open_mini](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/ablations/centrifuge_lid_open_mini/20260327_224834/ablation_report.json),
plus [centrifuge_lid_open_5430](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/ablations/centrifuge_lid_open_5430/20260327_225142/ablation_report.json).
The checked-in supported controller-ablation reports now span the current six-task
benchmark suite.

Run the currently supported non-fallback acceptance workflow in the `ultradexgrasp`
environment:

```bash
source /root/miniconda3/bin/activate ultradexgrasp
export AUTOBIO_ROOT="$(realpath ../../AutoBio)"
export ULTRADEX_ROOT="$(realpath ..)"
export AUTOBIO_PYTHON_EXECUTABLE="$(pwd)/artifacts/bootstrap_env_readme/bin/python"
python cli.py workflow --workflow supported_cycle_sequence
```

## Config And Env

Default config lives at [configs/autobio_config.yaml](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/configs/autobio_config.yaml).

Supported overrides:

- `AUTOBIO_INTEGRATION_CONFIG`
- `AUTOBIO_ROOT`
- `ULTRADEX_ROOT`
- `AUTOBIO_PROCESSED_ASSETS`
- `AUTOBIO_DEMO_OUTPUT`
- `AUTOBIO_GPU_ID`
- `AUTOBIO_NUM_CPU_WORKERS`
- `AUTOBIO_FORCE_HEADLESS_TASKS`
- `DISPLAY`

## Supported Validation Scope

The documented happy path validated in this repository is:

1. `python cli.py process --max-workers 1`
2. `python cli.py validate`
3. `python cli.py smoke-test`
4. `python cli.py generate --task thermal_cycler_close --num-episodes 1 --num-cpu-workers 1`
5. `python cli.py evaluate --task thermal_cycler_close`

`pickup_tube` generation still depends on the wider UltraDexGrasp stack (`torch`, `curobo`, BODex) and is not part of the minimal validated path yet.

For deterministic task-API validation without the full robot stack, set `AUTOBIO_FORCE_HEADLESS_TASKS=1`.
This forces the current tasks onto their scripted validation path while still using the normal `generate` pipeline and episode export format.

The broader non-fallback path validated in this repository uses the existing
`ultradexgrasp` conda environment and the normal CLI without
`AUTOBIO_FORCE_HEADLESS_TASKS=1` for:

1. `python cli.py validate`
2. `python cli.py generate --task tube_transfer --num-episodes 1 --num-cpu-workers 1`
3. `python cli.py generate --task thermal_cycler_open --num-episodes 1 --num-cpu-workers 1`
4. `python cli.py generate --task thermal_cycler_close --num-episodes 1 --num-cpu-workers 1`
5. `python cli.py generate --task centrifuge_lid_open_mini --num-episodes 1 --num-cpu-workers 1`
6. `python cli.py generate --task centrifuge_lid_open_5430 --num-episodes 1 --num-cpu-workers 1`
7. `python cli.py evaluate --input-dir artifacts/nonfallback_acceptance`
8. `python cli.py workflow --workflow supported_cycle_sequence`

Current task surface validated in this repository includes:

- `pickup_tube`
- `tube_transfer`
- `thermal_cycler_open`
- `thermal_cycler_close`
- `centrifuge_lid_open_mini`
- `centrifuge_lid_open_5430`

The current supported `pickup_tube` CLI generate path writes complete artifacts
through the direct pickup pipeline in `scripts/parallel_demo_synthesis.py`. It is
reproducible, machine-readable, and validated in the clean bootstrap environment,
but it currently uses the deterministic direct pickup pipeline rather than a fully
validated end-to-end dexterous BODex/cuRobo rollout.

## Asset Reproducibility

The repository now includes:

- [manifests/validated_asset_manifest.yaml](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/manifests/validated_asset_manifest.yaml)
- [manifests/full_asset_manifest.yaml](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/manifests/full_asset_manifest.yaml)
- [schemas/asset.schema.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/schemas/asset.schema.json)

To emit a checksum report for a manifest-driven processing run:

```bash
python cli.py process \
  --manifest-file manifests/full_asset_manifest.yaml \
  --checksum-report assets/processed/full_manifest.checksums.json
```

To compare two checksum reports:

```bash
python scripts/compare_asset_checksums.py reference.json candidate.json
```

Validated checksum-equivalence evidence already checked into this repository includes:

- [artifacts/full_repro_a.checksums.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/full_repro_a.checksums.json)
- [artifacts/full_repro_b.checksums.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/full_repro_b.checksums.json)
- [artifacts/repro_ref3b.checksums.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/repro_ref3b.checksums.json)
- [artifacts/repro_scratch3b.checksums.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/repro_scratch3b.checksums.json)

The paired reports above are byte-for-byte identical, which is the current
repository evidence that manifest-driven asset processing reproduced identical
processed outputs across independent reruns.

## Release Candidate Freeze

An independent README-and-CLI replay has been frozen under:

- [artifacts/release_candidates/rc_20260327_readme_cli/release_validation.md](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/release_candidates/rc_20260327_readme_cli/release_validation.md)
- [artifacts/release_candidates/rc_20260327_readme_cli/release_manifest.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/release_candidates/rc_20260327_readme_cli/release_manifest.json)

That tagged bundle captures:

- a fresh clean-bootstrap validation run from `scripts/bootstrap_clean_env.sh`
- a manifest-driven `process` replay with checksum-equivalence against `assets/processed/manifest.checksums.json`
- a fresh `generate` / `evaluate` replay for `thermal_cycler_close`
- frozen environment lockfiles from the clean venv and the checked-in conda specs

The published final release tag is now:

- [artifacts/releases/release_20260327_readme_cli/release_manifest.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/releases/release_20260327_readme_cli/release_manifest.json)
- [artifacts/releases/release_20260327_readme_cli/retrieval_verification.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/releases/release_20260327_readme_cli/retrieval_verification.json)
- [artifacts/releases/release_20260327_readme_cli/release_approval.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/releases/release_20260327_readme_cli/release_approval.json)

Release catalog / ledger:

- [artifacts/releases/catalog.json](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/releases/catalog.json)
- [artifacts/releases/catalog.md](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/releases/catalog.md)
- [artifacts/releases/LATEST](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/artifacts/releases/LATEST)

Lightweight release-close CI check:

```bash
python cli.py release-close-check
```

This validates:

- `artifacts/releases/LATEST`
- `artifacts/releases/catalog.json`
- the published release metadata referenced by the catalog entry
- one canonical-path downstream consumer invocation via `python cli.py evaluate --input-dir <canonical-release>/demos`

Actual release-close workflow gate:

```bash
bash scripts/release_close_ci.sh
```

This wrapper uses `artifacts/releases/LATEST` by default, runs `python cli.py release-close-check`,
and prints the canonical consumer log to stderr if the gate fails.

Repository CI workflow:

- [release-close.yml](/cephfs/shaoyanming/00000_dexbio/UltraDexGrasp/autobio_integration/.github/workflows/release-close.yml)

The workflow runs the release-close gate on release-related changes and publishes
the canonical consumer log on failure.
