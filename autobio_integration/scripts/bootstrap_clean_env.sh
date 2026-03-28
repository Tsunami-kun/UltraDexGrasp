#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-clean-bootstrap}"
AUTOBIO_ROOT="${AUTOBIO_ROOT:-$(realpath "$ROOT_DIR/../../AutoBio")}"
ULTRADEX_ROOT="${ULTRADEX_ROOT:-$(realpath "$ROOT_DIR/..")}"

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export AUTOBIO_ROOT
export ULTRADEX_ROOT

python cli.py validate
python cli.py smoke-test
python cli.py smoke-ci-gate

echo "Bootstrap complete: VENV_DIR=$VENV_DIR"
