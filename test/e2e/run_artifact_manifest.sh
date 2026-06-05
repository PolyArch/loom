#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "${ROOT}/test/artifacts/intermediate_artifacts.py" write-json artifact_manifest "$@"
