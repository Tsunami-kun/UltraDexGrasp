#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LATEST_FILE="${ROOT_DIR}/artifacts/releases/LATEST"

if [[ $# -ge 1 ]]; then
  RELEASE_TAG="$1"
elif [[ -f "$LATEST_FILE" ]]; then
  RELEASE_TAG="$(tr -d '\r\n' < "$LATEST_FILE")"
else
  echo "Missing release pointer: $LATEST_FILE" >&2
  exit 1
fi

if [[ -z "$RELEASE_TAG" ]]; then
  echo "Release tag is empty" >&2
  exit 1
fi

CANONICAL_ROOT="${ROOT_DIR}/artifacts/releases/${RELEASE_TAG}"
LOG_FILE="${LOG_FILE:-${CANONICAL_ROOT}/release_close_check.log}"

echo "============================================"
echo "Release-Close CI Gate"
echo "============================================"
echo "Release tag: ${RELEASE_TAG}"
echo "Canonical root: ${CANONICAL_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Consumer log: ${LOG_FILE}"
echo ""

if ! "${PYTHON_BIN}" cli.py release-close-check --release-tag "${RELEASE_TAG}" --log-file "${LOG_FILE}"; then
  echo "" >&2
  echo "Release-close gate failed for ${RELEASE_TAG}" >&2
  if [[ -f "${LOG_FILE}" ]]; then
    echo "--- begin consumer log: ${LOG_FILE} ---" >&2
    cat "${LOG_FILE}" >&2
    echo "--- end consumer log ---" >&2
  else
    echo "Consumer log not found: ${LOG_FILE}" >&2
  fi
  exit 1
fi

if [[ "${RELEASE_CLOSE_FORCE_FAIL_AFTER_CHECK:-0}" == "1" ]]; then
  echo "" >&2
  echo "Release-close gate intentionally failed after check for ${RELEASE_TAG}" >&2
  if [[ -f "${LOG_FILE}" ]]; then
    echo "--- begin consumer log: ${LOG_FILE} ---" >&2
    cat "${LOG_FILE}" >&2
    echo "--- end consumer log ---" >&2
  else
    echo "Consumer log not found: ${LOG_FILE}" >&2
  fi
  exit 1
fi

echo ""
echo "Release-close gate passed for ${RELEASE_TAG}"
