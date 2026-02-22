#!/usr/bin/env bash
# Verify that all declared PE types in CGRA templates have matching instantiations.
# Parses the actual generated template MLIR files and checks that every
# fabric.pe declaration has at least one fabric.instance referencing it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/../../mapper-app/templates"
exit_code=0

if [[ ! -d "${TEMPLATE_DIR}" ]]; then
  echo "error: templates directory not found: ${TEMPLATE_DIR}" >&2
  exit 1
fi

found=0
for template in "${TEMPLATE_DIR}"/loom_cgra_*.fabric.mlir; do
  [[ -f "$template" ]] || continue
  found=1
  name=$(basename "$template")

  # Extract declared PE types (fabric.pe @name...)
  declared=$(grep -oP 'fabric\.pe @\K[a-zA-Z0-9_]+' "$template" | sort -u)

  # Extract instantiated PE types (fabric.instance @name...)
  instantiated=$(grep -oP 'fabric\.instance @\K[a-zA-Z0-9_]+' "$template" | sort -u)

  decl_count=$(echo "$declared" | wc -w)
  inst_count=$(echo "$instantiated" | wc -w)

  # Check each declared type has at least one instantiation.
  for pe in $declared; do
    if ! echo "$instantiated" | grep -qx "$pe"; then
      echo "FAIL: ${name}: PE @${pe} declared but never instantiated"
      exit_code=1
    fi
  done

  echo "OK: ${name}: ${decl_count} declared, ${inst_count} instantiated"
done

if (( found == 0 )); then
  echo "error: no loom_cgra_*.fabric.mlir template files found" >&2
  exit 1
fi

exit $exit_code
