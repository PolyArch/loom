#!/usr/bin/env bash
# Handshake Ops Statistics Test
# Runs handshake generation at O0-O3, generates per-level ops stat files.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
APPS_DIR="${ROOT_DIR}/tests/app"

LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

any_failure=false

for level in 0 1 2 3; do
  tag="O${level}"
  output_ops="${APPS_DIR}/full-ops-${tag}.stat"

  rm -f "${output_ops}"

  # Clean previous tagged output
  find "${APPS_DIR}" -mindepth 2 -maxdepth 2 -type f \
    \( -name "*.${tag}.handshake.mlir" -o -name "*.${tag}.handshake.log" \) \
    -delete 2>/dev/null || true

  LOOM_SUMMARY_PREFIX="Handshake ${tag}" LOOM_HANDSHAKE_TAG="${tag}" \
    "${SCRIPT_DIR}/handshake.sh" "${LOOM_BIN}" "-O${level}" "$@" || any_failure=true

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

  # Clean intermediate .llvm.ll files
  find "${APPS_DIR}" -mindepth 2 -maxdepth 2 -type f \
    -name "*.${tag}.llvm.ll" -delete 2>/dev/null || true
done

if [[ "${any_failure}" == "true" ]]; then
  exit 1
fi
