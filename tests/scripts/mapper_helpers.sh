#!/usr/bin/env bash
# mapper_helpers.sh - Shared functions for mapper template selection.
#
# Provides count_ops() and select_template() used by both the production
# mapper_app_test.sh runner and the template_selection_regression.sh test.
#
# Source this file; do not execute directly.

# Counts non-sentinel operations in the first handshake.func body with loom.accel.
count_ops() {
  local mlir_file="$1"
  local count
  count=$(awk '
    /handshake\.func.*loom\.accel/ { if (done==0) inside=1; next }
    inside && /^[[:space:]]*\}/ { inside=0; done=1 }
    inside && /=/ { ops++ }
    END { print ops+0 }
  ' "$mlir_file")
  echo "$count"
}

# Selects the smallest adequate template for an app based on operation count.
select_template() {
  local op_count="$1"
  if (( op_count <= 30 )); then
    echo "loom_cgra_small"
  elif (( op_count <= 120 )); then
    echo "loom_cgra_medium"
  else
    echo "loom_cgra_large"
  fi
}
