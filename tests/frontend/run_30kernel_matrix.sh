#!/usr/bin/env bash
# 30-kernel / 6-domain frontend validation matrix.
# Runs each kernel through the loom frontend pipeline (C -> LLVM -> CF -> SCF -> DFG)
# WITHOUT the mapper (no --adg flag) and reports pass/fail per kernel plus an
# overall pass-rate summary table.
#
# Domains (5 kernels each):
#   ai_llm, arvr_stereo, dsp_ofdm, graph_analytics, robotics_vio, zk_stark
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks/tapestry"
COMMON_INC="$BENCH_DIR/common"

# ---------------------------------------------------------------------------
# Locate the loom binary
# ---------------------------------------------------------------------------
LOOM=""
if [ -x "$REPO_ROOT/build/bin/loom" ]; then
  LOOM="$REPO_ROOT/build/bin/loom"
elif [ -f "$REPO_ROOT/.git" ]; then
  MAIN_WORKTREE="$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null | sed 's|/\.git$||')"
  if [ -x "$MAIN_WORKTREE/build/bin/loom" ]; then
    LOOM="$MAIN_WORKTREE/build/bin/loom"
  fi
fi

if [ -z "$LOOM" ] || [ ! -x "$LOOM" ]; then
  echo "SKIP: loom binary not found (looked in $REPO_ROOT/build/bin/loom)"
  exit 0
fi

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR="$REPO_ROOT/out/30kernel-matrix"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Kernel matrix: 30 kernels, 5 per domain
# Each entry is  domain|kernel_name|source_path|extra_include_dirs
# ---------------------------------------------------------------------------
KERNELS=(
  # ai_llm (5 of 19 -- representative mix of attention, FFN, normalization)
  "ai_llm|activation|$BENCH_DIR/ai_llm/kernels/activation.c|"
  "ai_llm|layernorm|$BENCH_DIR/ai_llm/kernels/layernorm.c|"
  "ai_llm|softmax|$BENCH_DIR/ai_llm/kernels/softmax.c|"
  "ai_llm|qkv_proj|$BENCH_DIR/ai_llm/kernels/qkv_proj.c|"
  "ai_llm|ffn_up|$BENCH_DIR/ai_llm/kernels/ffn_up.c|"

  # arvr_stereo (all 5)
  "arvr_stereo|harris_corner|$BENCH_DIR/arvr_stereo/harris_corner.c|"
  "arvr_stereo|image_warp|$BENCH_DIR/arvr_stereo/image_warp.c|"
  "arvr_stereo|post_filter|$BENCH_DIR/arvr_stereo/post_filter.c|"
  "arvr_stereo|sad_matching|$BENCH_DIR/arvr_stereo/sad_matching.c|"
  "arvr_stereo|stereo_disparity|$BENCH_DIR/arvr_stereo/stereo_disparity.c|"

  # dsp_ofdm (5 of 6 -- skip crc_check which is purely bitwise)
  "dsp_ofdm|fft_butterfly|$BENCH_DIR/dsp_ofdm/fft_butterfly.c|"
  "dsp_ofdm|channel_est|$BENCH_DIR/dsp_ofdm/channel_est.c|"
  "dsp_ofdm|equalizer|$BENCH_DIR/dsp_ofdm/equalizer.c|"
  "dsp_ofdm|qam_demod|$BENCH_DIR/dsp_ofdm/qam_demod.c|"
  "dsp_ofdm|viterbi|$BENCH_DIR/dsp_ofdm/viterbi.c|"

  # graph_analytics (4 standalone + pipeline entry)
  "graph_analytics|bfs_traversal|$BENCH_DIR/graph_analytics/bfs_traversal.c|"
  "graph_analytics|pagerank_spmv|$BENCH_DIR/graph_analytics/pagerank_spmv.c|"
  "graph_analytics|triangle_count|$BENCH_DIR/graph_analytics/triangle_count.c|"
  "graph_analytics|label_prop|$BENCH_DIR/graph_analytics/label_prop.c|"
  "graph_analytics|graph_pipeline|$BENCH_DIR/graph_analytics/e02_pipeline/graph_pipeline.c|"

  # robotics_vio (all 5)
  "robotics_vio|fast_detect|$BENCH_DIR/robotics_vio/fast_detect.c|"
  "robotics_vio|feature_match|$BENCH_DIR/robotics_vio/feature_match.c|"
  "robotics_vio|imu_integration|$BENCH_DIR/robotics_vio/imu_integration.c|"
  "robotics_vio|orb_descriptor|$BENCH_DIR/robotics_vio/orb_descriptor.c|"
  "robotics_vio|pose_estimate|$BENCH_DIR/robotics_vio/pose_estimate.c|"

  # zk_stark (all 5)
  "zk_stark|ntt|$BENCH_DIR/zk_stark/ntt.c|$BENCH_DIR/zk_stark"
  "zk_stark|msm|$BENCH_DIR/zk_stark/msm.c|$BENCH_DIR/zk_stark"
  "zk_stark|poseidon_hash|$BENCH_DIR/zk_stark/poseidon_hash.c|$BENCH_DIR/zk_stark"
  "zk_stark|poly_eval|$BENCH_DIR/zk_stark/poly_eval.c|$BENCH_DIR/zk_stark"
  "zk_stark|proof_compose|$BENCH_DIR/zk_stark/proof_compose.c|$BENCH_DIR/zk_stark"
)

# ---------------------------------------------------------------------------
# Per-domain counters
# ---------------------------------------------------------------------------
declare -A DOMAIN_TOTAL
declare -A DOMAIN_PASS
declare -A DOMAIN_FAIL
declare -A DOMAIN_FAIL_NAMES

TOTAL=0
PASSED=0
FAILED=0

# ---------------------------------------------------------------------------
# Run a single kernel through the frontend pipeline
# ---------------------------------------------------------------------------
run_kernel() {
  local domain="$1"
  local name="$2"
  local src="$3"
  local extra_inc="$4"

  local tag="${domain}/${name}"
  local kout="$OUT_DIR/$domain/$name"
  mkdir -p "$kout"

  TOTAL=$((TOTAL + 1))
  DOMAIN_TOTAL[$domain]=$(( ${DOMAIN_TOTAL[$domain]:-0} + 1 ))

  # Build loom command with include paths
  local cmd=("$LOOM" "$src" -o "$kout" -I"$COMMON_INC")
  if [ -n "$extra_inc" ]; then
    cmd+=(-I"$extra_inc")
  fi

  # Run frontend (no --adg => frontend only)
  if "${cmd[@]}" >"$kout/stdout.log" 2>"$kout/stderr.log"; then
    if grep -q "error:" "$kout/stderr.log"; then
      printf "  FAIL  %-40s  (frontend emitted verifier errors)\n" "$tag"
      FAILED=$((FAILED + 1))
      DOMAIN_FAIL[$domain]=$(( ${DOMAIN_FAIL[$domain]:-0} + 1 ))
      DOMAIN_FAIL_NAMES[$domain]="${DOMAIN_FAIL_NAMES[$domain]:-} $name"
      return
    fi

    # Look for a DFG MLIR output
    local dfg_file
    dfg_file="$(find "$kout" -name '*.dfg.mlir' -print -quit 2>/dev/null || true)"

    if [ -n "$dfg_file" ] && [ -f "$dfg_file" ]; then
      if grep -q "handshake.func" "$dfg_file" 2>/dev/null; then
        printf "  PASS  %-40s  (DFG generated)\n" "$tag"
        PASSED=$((PASSED + 1))
        DOMAIN_PASS[$domain]=$(( ${DOMAIN_PASS[$domain]:-0} + 1 ))
        return
      fi
    fi
    # Pipeline succeeded but no valid DFG
    printf "  FAIL  %-40s  (no valid DFG output)\n" "$tag"
  else
    local rc=$?
    printf "  FAIL  %-40s  (exit code %d)\n" "$tag" "$rc"
    if [ -f "$kout/stderr.log" ]; then
      head -3 "$kout/stderr.log" | sed 's/^/        /'
    fi
  fi

  FAILED=$((FAILED + 1))
  DOMAIN_FAIL[$domain]=$(( ${DOMAIN_FAIL[$domain]:-0} + 1 ))
  DOMAIN_FAIL_NAMES[$domain]="${DOMAIN_FAIL_NAMES[$domain]:-} $name"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "======================================================================="
echo "  30-Kernel / 6-Domain Frontend Validation Matrix"
echo "  loom binary: $LOOM"
echo "  timestamp:   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "======================================================================="
echo ""

for entry in "${KERNELS[@]}"; do
  IFS='|' read -r domain kname ksrc kinc <<< "$entry"
  run_kernel "$domain" "$kname" "$ksrc" "$kinc"
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================="
echo "  Summary"
echo "======================================================================="
printf "  %-20s  %5s  %5s  %5s  %7s\n" "Domain" "Total" "Pass" "Fail" "Rate"
printf "  %-20s  %5s  %5s  %5s  %7s\n" "--------------------" "-----" "-----" "-----" "-------"

DOMAINS_ORDERED=(ai_llm arvr_stereo dsp_ofdm graph_analytics robotics_vio zk_stark)
for d in "${DOMAINS_ORDERED[@]}"; do
  dt=${DOMAIN_TOTAL[$d]:-0}
  dp=${DOMAIN_PASS[$d]:-0}
  df=${DOMAIN_FAIL[$d]:-0}
  if [ "$dt" -gt 0 ]; then
    rate=$(( dp * 100 / dt ))
  else
    rate=0
  fi
  printf "  %-20s  %5d  %5d  %5d  %6d%%\n" "$d" "$dt" "$dp" "$df" "$rate"
done

printf "  %-20s  %5s  %5s  %5s  %7s\n" "--------------------" "-----" "-----" "-----" "-------"
if [ "$TOTAL" -gt 0 ]; then
  OVERALL_RATE=$(( PASSED * 100 / TOTAL ))
else
  OVERALL_RATE=0
fi
printf "  %-20s  %5d  %5d  %5d  %6d%%\n" "TOTAL" "$TOTAL" "$PASSED" "$FAILED" "$OVERALL_RATE"
echo ""

# Print failed kernels if any
if [ "$FAILED" -gt 0 ]; then
  echo "  Failed kernels:"
  for d in "${DOMAINS_ORDERED[@]}"; do
    if [ -n "${DOMAIN_FAIL_NAMES[$d]:-}" ]; then
      echo "    $d:${DOMAIN_FAIL_NAMES[$d]}"
    fi
  done
  echo ""
fi

echo "  Logs: $OUT_DIR"
echo ""

# Exit with the number of failures (0 = all pass)
exit "$FAILED"
