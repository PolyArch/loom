#!/usr/bin/env bash
# Shared helper for the per-kernel dfg_check.sh scripts under test/app.
# Each kernel sources this file with the following shell variables set:
#
#   KERNEL          -- short kernel name (vecadd, gemm, ...)
#   EXPECT_GRAPH    -- "yes" if the kernel must carry at least one
#                      dataflow.graph.func + dataflow.graph.launch
#                      (e.g., kernels with iter_args reductions);
#                      "no" otherwise.
#   EXPECT_GRAPH_SYMBOL_<VARIANT>
#                    -- optional exact dataflow.graph.func symbol that
#                      this variant must emit. When set, graph-body op
#                      checks are scoped to that symbol instead of any
#                      graph in the lowered file.
#   HERE            -- absolute path of the kernel's directory.
#   REPO            -- absolute path of the repository root.
#   LOOM_CC         -- absolute path of loom-cc.
#   LOOM_RAISE      -- absolute path of loom-raise.
#   LOOM_LOWER      -- absolute path of loom-lower.
#   LOOM_RAISE_OPT  -- absolute path of loom-raise-opt.
#
# The exported `dfg_one $variant $source_ext` runs:
#   loom-cc -> .ll
#   loom-raise -> .scf.mlir
#   loom-lower -> .dfg.mlir
#   loom-raise-opt -o /dev/null < .dfg.mlir   (round-trip parse)
# And asserts that .dfg.mlir contains the expected dataflow.thread /
# dataflow.graph.func / *.launch symbols.

BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-dfg}"
mkdir -p "${BUILD_DIR}"

require_kernel_graph() {
    local variant="$1"
    local kernel_fn="$2"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

    if ! grep -E -q "dataflow\\.thread (private )?@t_${kernel_fn}_[A-Za-z0-9_]+" "${dfg}"; then
        echo "[${KERNEL}/${variant}] no ${kernel_fn} dataflow.thread in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q "dataflow\\.graph\\.launch @g_t_${kernel_fn}_[A-Za-z0-9_]+" "${dfg}"; then
        echo "[${KERNEL}/${variant}] no ${kernel_fn} graph launch in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q "dataflow\\.graph\\.func (private )?@g_t_${kernel_fn}_[A-Za-z0-9_]+" "${dfg}"; then
        echo "[${KERNEL}/${variant}] no ${kernel_fn} graph func in ${dfg}" >&2
        return 1
    fi
}

graph_symbol_for_variant() {
    local variant="$1"
    local key="${variant^^}"
    key="${key//[^A-Z0-9_]/_}"
    local var_name="EXPECT_GRAPH_SYMBOL_${key}"
    printf '%s' "${!var_name:-${EXPECT_GRAPH_SYMBOL:-}}"
}

require_exact_graph_symbol() {
    local variant="$1"
    local symbol="$2"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

    if ! grep -E -q "dataflow\\.graph\\.launch @${symbol}(\\(|\\b)" "${dfg}"; then
        echo "[${KERNEL}/${variant}] no dataflow.graph.launch @${symbol} in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q "dataflow\\.graph\\.func (private )?@${symbol}(\\(|\\b)" "${dfg}"; then
        echo "[${KERNEL}/${variant}] no dataflow.graph.func @${symbol} in ${dfg}" >&2
        return 1
    fi
}

require_graph_body_op() {
    local variant="$1"
    local symbol="$2"
    local needle="$3"
    local label="$4"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"
    local status

    set +e
    python3 - "${dfg}" "${symbol}" "${needle}" <<'PY'
import sys

path, symbol, needle = sys.argv[1:]
lines = open(path, encoding="utf-8").read().splitlines()
header = "@" + symbol

for index, line in enumerate(lines):
    if "dataflow.graph.func" not in line or header not in line:
        continue
    depth = line.count("{") - line.count("}")
    body = [line]
    for nested in lines[index + 1:]:
        body.append(nested)
        depth += nested.count("{") - nested.count("}")
        if depth <= 0:
            break
    sys.exit(0 if needle in "\n".join(body) else 2)

sys.exit(1)
PY
    status="$?"
    set -e
    case "${status}" in
        0)
            return 0
            ;;
        1)
            echo "[${KERNEL}/${variant}] no graph body for @${symbol} in ${dfg}" >&2
            ;;
        *)
            echo "[${KERNEL}/${variant}] no ${label} in graph @${symbol} in ${dfg}" >&2
            ;;
    esac
    return 1
}

dfg_one() {
    local variant="$1"
    local source_ext="$2"

    local src="${HERE}/${variant}.${source_ext}"
    local ll="${BUILD_DIR}/${variant}.ll"
    local scf="${BUILD_DIR}/${variant}.scf.mlir"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

    if [[ ! -f "${src}" ]]; then
        echo "[${KERNEL}/${variant}] missing source: ${src}" >&2
        return 1
    fi

    "${LOOM_CC}" -emit-llvm -O1 -S "${src}" -o "${ll}"
    "${LOOM_RAISE}" "${ll}" -o "${scf}"
    "${LOOM_LOWER}" "${scf}" -o "${dfg}"

    if [[ ! -s "${dfg}" ]]; then
        echo "[${KERNEL}/${variant}] lowered MLIR is empty: ${dfg}" >&2
        return 1
    fi

    # Round-trip parse: feed .dfg.mlir back through loom-raise-opt
    # (which has the dataflow dialect registered) and discard the
    # output. Failure to parse means the dataflow ops we emitted are
    # not structurally valid.
    if ! "${LOOM_RAISE_OPT}" "${dfg}" -o /dev/null >/dev/null 2>&1; then
        echo "[${KERNEL}/${variant}] dfg.mlir failed round-trip parse" >&2
        return 1
    fi

    # Required: at least one `dataflow.thread @t_<sym>` definition and
    # at least one `dataflow.thread.launch @t_<sym>` reference.
    if ! grep -E -q 'dataflow\.thread (private )?@t_[A-Za-z0-9_]+' \
            "${dfg}"; then
        echo "[${KERNEL}/${variant}] no dataflow.thread @t_ symbol in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q 'dataflow\.thread\.launch @t_[A-Za-z0-9_]+' "${dfg}"; then
        echo "[${KERNEL}/${variant}] no dataflow.thread.launch @t_ in ${dfg}" >&2
        return 1
    fi

    if [[ "${EXPECT_GRAPH}" == "yes" ]]; then
        local expected_graph_symbol
        expected_graph_symbol="$(graph_symbol_for_variant "${variant}")"
        if ! grep -E -q 'dataflow\.graph\.func (private )?@g_[A-Za-z0-9_]+' \
                "${dfg}"; then
            echo "[${KERNEL}/${variant}] no dataflow.graph.func @g_ symbol in ${dfg}" >&2
            return 1
        fi
        if ! grep -E -q 'dataflow\.graph\.launch @g_[A-Za-z0-9_]+' \
                "${dfg}"; then
            echo "[${KERNEL}/${variant}] no dataflow.graph.launch @g_ in ${dfg}" >&2
            return 1
        fi
        if [[ -n "${expected_graph_symbol}" ]]; then
            require_exact_graph_symbol "${variant}" "${expected_graph_symbol}" || return 1
        fi
        # Streaming primitives appear inside the graph.func body once
        # the simple-reduction shape has been lowered. EXPECT_STREAM
        # defaults to "yes" so every reduction kernel asserts that
        # dataflow.stream + dataflow.carry are present; kernels that
        # carry only nested or call-bearing reductions may set
        # EXPECT_STREAM=no to opt out.
        if [[ "${EXPECT_STREAM:-yes}" == "yes" ]]; then
            if [[ -n "${expected_graph_symbol}" ]]; then
                require_graph_body_op "${variant}" "${expected_graph_symbol}" "dataflow.stream " \
                    "dataflow.stream" || return 1
            elif ! grep -E -q 'dataflow\.stream ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.stream in ${dfg}" >&2
                return 1
            fi
            if [[ -n "${expected_graph_symbol}" ]]; then
                require_graph_body_op "${variant}" "${expected_graph_symbol}" "dataflow.carry " \
                    "dataflow.carry" || return 1
            elif ! grep -E -q 'dataflow\.carry ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.carry in ${dfg}" >&2
                return 1
            fi
        fi
        # Memory tokenization: the graph-memory pass replaces residual
        # llvm.{load, store} ops in the graph.func body with the
        # dataflow.{load, store} streaming primitives. Default to
        # asserting at least one dataflow.load (every reduction body
        # in test/app reads from an input array). Set EXPECT_LOAD=no
        # to opt out.
        if [[ "${EXPECT_LOAD:-yes}" == "yes" ]]; then
            if [[ -n "${expected_graph_symbol}" ]]; then
                require_graph_body_op "${variant}" "${expected_graph_symbol}" "dataflow.load " \
                    "dataflow.load" || return 1
            elif ! grep -E -q 'dataflow\.load ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.load in ${dfg}" >&2
                return 1
            fi
        fi
        # Stores in graph.func bodies only appear when the kernel's
        # reduction loop also writes back into a buffer. The five
        # current test/app kernels are pure-reduction (read-only inside
        # the reduction body), so EXPECT_STORE defaults to "no". The
        # cmsis-dsp / cmsis-nn corpora exercise the store path (e.g.
        # arm_offset_f32, arm_relu_q7).
        if [[ "${EXPECT_STORE:-no}" == "yes" ]]; then
            if [[ -n "${expected_graph_symbol}" ]]; then
                require_graph_body_op "${variant}" "${expected_graph_symbol}" "dataflow.store " \
                    "dataflow.store" || return 1
            elif ! grep -E -q 'dataflow\.store ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.store in ${dfg}" >&2
                return 1
            fi
        fi
        # Loop-invariant block-arg scalars consumed inside the body
        # get wrapped in dataflow.invariant by the graph-invariant
        # pass. Loop-carried initializer operands stay as raw one-shot
        # carry init tokens, so kernels that require an invariant
        # hyperparameter should opt in explicitly.
        if [[ "${EXPECT_INVARIANT:-no}" == "yes" ]]; then
            if [[ -n "${expected_graph_symbol}" ]]; then
                require_graph_body_op "${variant}" "${expected_graph_symbol}" "dataflow.invariant " \
                    "dataflow.invariant" || return 1
            elif ! grep -E -q 'dataflow\.invariant ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.invariant in ${dfg}" >&2
                return 1
            fi
        fi
    fi
}
