#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

LOOM_BIN=${1:-"${ROOT_DIR}/build/bin/loom"}
shift || true

HANDSHAKE_SCRIPT="${ROOT_DIR}/tools/loom/loom_handshake.sh"
if [[ ! -x "${HANDSHAKE_SCRIPT}" ]]; then
  echo "error: handshake script not found: ${HANDSHAKE_SCRIPT}" >&2
  exit 1
fi

APPS_DIR="${ROOT_DIR}/tests/app"
if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

levels=(0 1 2 3)
EXPECTED_ARITH=26
EXPECTED_DATAFLOW=4
EXPECTED_HANDSHAKE=11
EXPECTED_MATH=7
EXPECTED_UB=1

check_dialect_counts() {
  local path="$1"
  declare -A expected=(
    [arith]="${EXPECTED_ARITH}"
    [dataflow]="${EXPECTED_DATAFLOW}"
    [handshake]="${EXPECTED_HANDSHAKE}"
    [math]="${EXPECTED_MATH}"
    [ub]="${EXPECTED_UB}"
  )
  declare -A found=()
  local line
  while IFS= read -r line; do
    if [[ "${line}" =~ ^dialect:\ ([^[:space:]]+)\ \(([0-9]+)\)$ ]]; then
      found["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
    fi
  done < "${path}"

  local errors=0
  local dialect expected_count actual
  for dialect in "${!expected[@]}"; do
    expected_count="${expected[${dialect}]}"
    actual="${found[${dialect}]:-}"
    if [[ -z "${actual}" ]]; then
      echo "error: missing dialect ${dialect} in ${path}" >&2
      errors=1
      continue
    fi
    if [[ "${actual}" != "${expected_count}" ]]; then
      echo "error: dialect ${dialect} count mismatch in ${path}: expected ${expected_count}, got ${actual}" >&2
      errors=1
    fi
  done

  return "${errors}"
}

for level in "${levels[@]}"; do
  tag="O${level}"
  output_ops="${APPS_DIR}/full-ops-${tag}.stat"

  rm -f "${output_ops}"

  mapfile -t handshake_outputs < <(
    find "${APPS_DIR}" -mindepth 2 -maxdepth 2 -type f \
      \( -name "*.${tag}.handshake.mlir" -o -name "*.${tag}.handshake.log" \) \
      | sort
  )
  if [[ ${#handshake_outputs[@]} -gt 0 ]]; then
    rm -f "${handshake_outputs[@]}"
  fi

  LOOM_SUMMARY_PREFIX="loom_handshake ${tag}" LOOM_HANDSHAKE_TAG="${tag}" "${HANDSHAKE_SCRIPT}" "${LOOM_BIN}" "-O${level}" "$@"

  python3 - "${APPS_DIR}" "${tag}" "${output_ops}" <<'PY'
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

apps_dir = Path(sys.argv[1])
tag = sys.argv[2]
out_path = Path(sys.argv[3])

files = sorted(apps_dir.glob(f"*/Output/*.{tag}.handshake.mlir"))
if not files:
    sys.stderr.write(f"error: no handshake files for {tag}\n")
    sys.exit(1)

counts = Counter()

op_re = re.compile(r"^([A-Za-z_][\w.]*)")
assign_re = re.compile(r"=\s*([A-Za-z_][\w.]*)")
quoted_re = re.compile(r'"([A-Za-z_][\w.]*)"')

for path in files:
    text = path.read_text()
    in_func = False
    for line in text.splitlines():
        if not in_func and "handshake.func" in line:
            in_func = True
            continue
        if not in_func:
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("^"):
            continue
        if "handshake.func" in stripped:
            continue
        if stripped.startswith("}"):
            in_func = False
            continue

        op = None
        if "=" in stripped:
            match = assign_re.search(stripped)
            if match:
                op = match.group(1)
            else:
                match = quoted_re.search(stripped)
                if match:
                    op = match.group(1)
        else:
            match = op_re.match(stripped)
            if match:
                op = match.group(1)
            else:
                match = quoted_re.match(stripped)
                if match:
                    op = match.group(1)

        if op:
            counts[op] += 1

by_dialect = defaultdict(list)
for op, count in counts.items():
    dialect = op.split(".", 1)[0] if "." in op else "builtin"
    by_dialect[dialect].append((op, count))

lines = []
for dialect in sorted(by_dialect):
    ops = sorted(by_dialect[dialect])
    lines.append(f"dialect: {dialect} ({len(ops)})")
    for op, count in ops:
        lines.append(f"  {op}: {count}")

out_path.write_text("\n".join(lines) + "\n")
PY

  check_dialect_counts "${output_ops}"

  mapfile -t llvm_outputs < <(
    find "${APPS_DIR}" -mindepth 2 -maxdepth 2 -type f -name "*.${tag}.llvm.ll" | sort
  )
  if [[ ${#llvm_outputs[@]} -gt 0 ]]; then
    rm -f "${llvm_outputs[@]}"
  fi

done

python3 - "${APPS_DIR}" <<'PY'
import sys
from pathlib import Path

apps_dir = Path(sys.argv[1])
levels = [0, 1, 2, 3]

baseline = None
baseline_level = None

for level in levels:
    path = apps_dir / f"full-ops-O{level}.stat"
    if not path.exists():
        sys.stderr.write(f"error: missing ops file: {path}\n")
        sys.exit(1)
    ops = []
    for line in path.read_text().splitlines():
        if line.startswith("  "):
            op = line.strip().split(":", 1)[0]
            ops.append(op)
    ops_set = set(ops)
    if baseline is None:
        baseline = ops_set
        baseline_level = level
    elif ops_set != baseline:
        missing = sorted(baseline - ops_set)
        extra = sorted(ops_set - baseline)
        sys.stderr.write(
            f"error: op set mismatch between O{baseline_level} and O{level}\n"
        )
        if missing:
            sys.stderr.write(f"  missing in O{level}: {', '.join(missing)}\n")
        if extra:
            sys.stderr.write(f"  extra in O{level}: {', '.join(extra)}\n")
        sys.exit(1)
PY
