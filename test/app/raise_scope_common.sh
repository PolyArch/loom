#!/usr/bin/env bash
# Shared helpers for checking raised MLIR inside one function scope.

awk_function_scope_pattern() {
    local func_name="$1"
    printf '%s\n' "/func\\.func @${func_name}\\(/ { in_func = 1; next }"
    printf '%s\n' 'in_func && /func\.func / { in_func = 0 }'
}
