#!/usr/bin/env bash
# Shared raise checks for trigonometric window app cases.

RAISE_SCOPE_HELPER="${REPO}/test/app/raise_scope_common.sh"
. "${RAISE_SCOPE_HELPER}"

raise_window_one() {
    local variant="$1"
    local has_call="$2"

    local src="${HERE}/${variant}.cpp"
    local ll="${BUILD_DIR}/${variant}.ll"
    local mlir="${BUILD_DIR}/${variant}.scf.mlir"

    "${LOOM_CXX}" -emit-llvm -O1 -S "${src}" -o "${ll}"
    "${LOOM_RAISE}" "${ll}" -o "${mlir}"

    if [[ ! -s "${mlir}" ]]; then
        echo "[${KERNEL}/${variant}] raised MLIR is empty" >&2
        return 1
    fi
    local scope_name
    if [[ "${has_call}" == "yes" ]]; then
        scope_name="${KERNEL_FN}"
    else
        scope_name="main"
    fi
    local scope_pattern
    scope_pattern="$(awk_function_scope_pattern "${scope_name}")"
    if ! awk -v min_cos="${WINDOW_MIN_COS}" -v min_mul="${WINDOW_MIN_MUL}" \
        -v allow_addf="${WINDOW_ALLOW_ADDF}" "${scope_pattern}"'
        in_func && /scf\.(forall|for) / {
            in_loop = 1
            has_load = 0
            has_store = 0
            cos_count = 0
            mul_count = 0
            has_window = 0
            next
        }
        in_func && in_loop {
            if ($0 ~ /llvm\.load/) {
                has_load = 1
            }
            if ($0 ~ /llvm\.store/) {
                has_store = 1
            }
            if ($0 ~ /llvm\.intr\.cos|math\.cos|llvm\.call.*@cosf/) {
                cos_count += 1
            }
            if ($0 ~ /arith\.mulf/) {
                mul_count += 1
            }
            if ($0 ~ /arith\.subf|llvm\.intr\.fmuladd/ ||
                (allow_addf == "yes" && $0 ~ /arith\.addf/)) {
                has_window = 1
            }
            if (has_load && has_store && cos_count >= min_cos &&
                mul_count >= min_mul && has_window) {
                found = 1
            }
        }
        END { exit found ? 0 : 1 }
    ' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no ${WINDOW_LABEL} loop in @${scope_name}: ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'func\.func @main' "${mlir}"; then
        echo "[${KERNEL}/${variant}] main was not raised: ${mlir}" >&2
        return 1
    fi
    if [[ "${has_call}" == "yes" ]]; then
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL_FN}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected call to @${KERNEL_FN}" >&2
            return 1
        fi
    else
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL_FN}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] unexpected call to @${KERNEL_FN}" >&2
            return 1
        fi
    fi
}

raise_window_variants() {
    mkdir -p "${BUILD_DIR}"

    raise_window_one "main_func" "yes"
    raise_window_one "main_inline" "no"

    echo "[${KERNEL}] PASS"
}
