from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "autobio_config.yaml"


@dataclass(frozen=True)
class RuntimePaths:
    repo_root: Path
    config_path: Path
    ultradex_root: Path
    autobio_root: Path
    autobio_assets_root: Path
    processed_assets: Path
    demo_output: Path
    robot_asset_root: Path
    python_executable: Path


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must deserialize to a mapping")
    return data


def _default_data(repo_root: Path) -> Dict[str, Any]:
    ultradex_root = repo_root.parent
    autobio_root = ultradex_root.parent / "AutoBio"
    return {
        "paths": {
            "repo_root": str(repo_root),
            "ultradex_root": str(ultradex_root),
            "autobio_root": str(autobio_root),
            "processed_assets": str(repo_root / "assets" / "processed"),
            "demo_output": str(repo_root / "demos"),
        },
        "devices": {
            "gpu_id": 0,
            "display": "",
            "headless": True,
            "num_cpu_workers": 4,
        },
        "execution": {
            "python_executable": sys.executable,
        },
        "robot": {
            "asset_root": str(ultradex_root / "asset"),
            "left_arm": {
                "urdf": "ur5e_with_xhand_urdf_offset_sim2real/ur5e_with_xhand_left_limited_joint_sapien.urdf",
                "curobo_config": "ur5e_with_xhand_urdf_offset_sim2real/ur5e_with_xhand_left.yaml",
                "ik_config": "ur5e_with_xhand_urdf_offset_sim2real/ur5e_with_base_left.yaml",
                "pos": [0.0, 0.45, 0.714],
            },
            "right_arm": {
                "urdf": "ur5e_with_xhand_urdf_offset_sim2real/ur5e_with_xhand_right_limited_joint_sapien.urdf",
                "curobo_config": "ur5e_with_xhand_urdf_offset_sim2real/ur5e_with_xhand_right.yaml",
                "ik_config": "ur5e_with_xhand_urdf_offset_sim2real/ur5e_with_base_right.yaml",
                "pos": [0.0, -0.45, 0.714],
            },
        },
    }


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    resolved_config = Path(
        config_path
        or os.environ.get("AUTOBIO_INTEGRATION_CONFIG")
        or DEFAULT_CONFIG_PATH
    ).resolve()
    data = _default_data(REPO_ROOT)
    data = _deep_update(data, _read_yaml(resolved_config))

    env_overrides: Dict[str, Any] = {"paths": {}, "devices": {}, "execution": {}}
    for env_name, target in [
        ("AUTOBIO_ROOT", ("paths", "autobio_root")),
        ("ULTRADEX_ROOT", ("paths", "ultradex_root")),
        ("AUTOBIO_PROCESSED_ASSETS", ("paths", "processed_assets")),
        ("AUTOBIO_DEMO_OUTPUT", ("paths", "demo_output")),
        ("AUTOBIO_GPU_ID", ("devices", "gpu_id")),
        ("AUTOBIO_NUM_CPU_WORKERS", ("devices", "num_cpu_workers")),
        ("DISPLAY", ("devices", "display")),
        ("AUTOBIO_PYTHON_EXECUTABLE", ("execution", "python_executable")),
    ]:
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        section, key = target
        env_overrides[section][key] = int(value) if key in {"gpu_id", "num_cpu_workers"} else value

    data = _deep_update(data, env_overrides)

    base_dir = resolved_config.parent

    def _resolve_path(value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        return (base_dir / path).resolve()

    def _resolve_executable(value: str | Path) -> str:
        path = Path(value).expanduser()
        if path.is_absolute():
            return os.path.abspath(str(path))
        return os.path.abspath(str(base_dir / path))

    paths = data.setdefault("paths", {})
    repo_root = _resolve_path(paths.get("repo_root", REPO_ROOT))
    ultradex_root = _resolve_path(paths["ultradex_root"])
    autobio_root = _resolve_path(paths["autobio_root"])
    robot = data.setdefault("robot", {})
    robot["asset_root"] = str(_resolve_path(robot.get("asset_root", ultradex_root / "asset")))
    paths["repo_root"] = str(repo_root)
    paths["ultradex_root"] = str(ultradex_root)
    paths["autobio_root"] = str(autobio_root)
    paths["autobio_assets_root"] = str((autobio_root / "autobio" / "assets").resolve())
    paths["processed_assets"] = str(_resolve_path(paths["processed_assets"]))
    paths["demo_output"] = str(_resolve_path(paths["demo_output"]))
    execution = data.setdefault("execution", {})
    python_override = os.environ.get("AUTOBIO_PYTHON_EXECUTABLE")
    execution["python_executable"] = _resolve_executable(
        python_override or execution.get("python_executable", sys.executable)
    )
    data["resolved_config_path"] = str(resolved_config)
    return data


def get_runtime_paths(config_path: str | Path | None = None) -> RuntimePaths:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    return RuntimePaths(
        repo_root=Path(paths["repo_root"]),
        config_path=Path(cfg["resolved_config_path"]),
        ultradex_root=Path(paths["ultradex_root"]),
        autobio_root=Path(paths["autobio_root"]),
        autobio_assets_root=Path(paths["autobio_assets_root"]),
        processed_assets=Path(paths["processed_assets"]),
        demo_output=Path(paths["demo_output"]),
        robot_asset_root=Path(cfg["robot"]["asset_root"]),
        python_executable=Path(cfg["execution"]["python_executable"]),
    )


def setup_python_path(config_path: str | Path | None = None) -> RuntimePaths:
    paths = get_runtime_paths(config_path)
    for entry in [paths.ultradex_root, paths.repo_root, paths.repo_root / "assets"]:
        entry_str = str(entry)
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)
    return paths


def resolve_robot_path(relative_or_abs: str, config_path: str | Path | None = None) -> str:
    if not relative_or_abs:
        return relative_or_abs
    path = Path(relative_or_abs)
    if path.is_absolute():
        return str(path)
    paths = get_runtime_paths(config_path)
    return str((paths.robot_asset_root / path).resolve())


def resolve_python_executable(config_path: str | Path | None = None) -> str:
    return str(get_runtime_paths(config_path).python_executable)
