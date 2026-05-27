#!/usr/bin/env bash
# Shared helper for the per-kernel dfg_check.sh scripts under test/app.
# Each kernel sources this file with the following shell variables set:
#
#   KERNEL          -- short kernel name (vecadd, gemm, ...)
#   EXPECT_GRAPH    -- "yes" if the kernel must carry at least one
#                      dataflow.graph.func + dataflow.graph.launch
#                      (e.g., kernels with iter_args reductions);
#                      "no" otherwise.
#   HERE            -- absolute path of the kernel's directory.
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

BUILD_DIR="${BUILD_DIR:-${HERE}/build}"
mkdir -p "${BUILD_DIR}"

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
        # Streaming primitives appear inside the graph.func body once
        # the simple-reduction shape has been lowered. EXPECT_STREAM
        # defaults to "yes" so every reduction kernel asserts that
        # dataflow.stream + dataflow.carry are present; kernels that
        # carry only nested or call-bearing reductions may set
        # EXPECT_STREAM=no to opt out.
        if [[ "${EXPECT_STREAM:-yes}" == "yes" ]]; then
            if ! grep -E -q 'dataflow\.stream ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.stream in ${dfg}" >&2
                return 1
            fi
            if ! grep -E -q 'dataflow\.carry ' "${dfg}"; then
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
            if ! grep -E -q 'dataflow\.load ' "${dfg}"; then
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
            if ! grep -E -q 'dataflow\.store ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.store in ${dfg}" >&2
                return 1
            fi
        fi
        # Loop-invariant block-arg scalars consumed inside the body
        # get wrapped in dataflow.invariant by the graph-invariant
        # pass. Every test/app reduction body has at least one (the
        # accumulator initial value), so EXPECT_INVARIANT defaults to
        # "yes".
        if [[ "${EXPECT_INVARIANT:-yes}" == "yes" ]]; then
            if ! grep -E -q 'dataflow\.invariant ' "${dfg}"; then
                echo "[${KERNEL}/${variant}] no dataflow.invariant in ${dfg}" >&2
                return 1
            fi
        fi
    fi
}
