#!/usr/bin/env bash

load_cmsis_sim_default_batch_stems() {
    local root="$1"
    local manifest="$2"
    python3 - "${root}" "${manifest}" <<'PY'
import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
sys.path.insert(0, str(root / "test" / "e2e"))
import run_cmsis_dfg_sim_attempts as attempts  # noqa: E402

data = json.loads(manifest.read_text())
if data.get("schema_version") != 1:
    raise SystemExit(f"{manifest} schema_version must be 1")
stems = data.get("attempt_stems")
if not isinstance(stems, list) or not stems:
    raise SystemExit(f"{manifest} attempt_stems must be a non-empty list")
seen: set[str] = set()
for stem in stems:
    if not isinstance(stem, str) or not stem.strip():
        raise SystemExit(f"{manifest} attempt_stems entries must be non-empty strings")
    if stem in seen:
        raise SystemExit(f"{manifest} contains duplicate CMSIS attempt stem {stem}")
    if not any(attempts.attempt_matches_stem(attempt, stem) for attempt in attempts.ATTEMPTS):
        raise SystemExit(f"{manifest} contains unknown CMSIS attempt stem {stem}")
    seen.add(stem)
    print(stem)
PY
}

dedupe_array() {
    python3 - "$@" <<'PY'
import sys

seen: set[str] = set()
for item in sys.argv[1:]:
    if item in seen:
        continue
    seen.add(item)
    print(item)
PY
}

cmsis_status_default_jobs() {
    local explicit="${1:-}"
    local value="${explicit:-${LOOM_TEST_JOBS:-${JOBS:-}}}"
    if [[ -z "${value}" ]]; then
        value="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "invalid --jobs value: ${value}" >&2
        exit 2
    fi
    printf '%s\n' "${value}"
}

clean_cmsis_sim_evidence() {
    local root="$1"
    local evidence_dir="$2"
    local comparison_dir="$3"
    python3 - "${root}" "${evidence_dir}" "${comparison_dir}" <<'PY'
import shutil
import sys
from pathlib import Path


root = Path(sys.argv[1])
evidence_dir = Path(sys.argv[2])
comparison_dir = Path(sys.argv[3])
sys.path.insert(0, str(root / "test" / "e2e"))
import run_cmsis_dfg_sim_attempts as attempts  # noqa: E402

evidence_dir.mkdir(parents=True, exist_ok=True)

labels: set[str] = set()
cases: set[str] = set()
for attempt in attempts.ATTEMPTS:
    cases.add(attempt.case)
    for label in (attempt.stem, attempt.artifact_stem, attempt.aggregate_stem):
        if label:
            labels.add(label)

suffixes = (
    ".dfg.report.json",
    ".mapping.csv",
    ".mapping.json",
    ".cgra.report.json",
    ".lowered.dfg.mlir",
)
for label in labels:
    for suffix in suffixes:
        path = evidence_dir / f"{label}{suffix}"
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

for case in cases:
    comparison = comparison_dir / f"{case}.sim-comparison-report.json"
    if comparison.is_file() or comparison.is_symlink():
        comparison.unlink()

for directory in sorted(comparison_dir.glob("**/*"), reverse=True):
    if directory.is_dir():
        try:
            directory.rmdir()
        except OSError:
            pass
PY
}
