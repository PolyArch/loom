// TaskGraph definition for Application 2: Mixture-of-Experts (MoE) Layer.
//
// Topology: Diamond scatter-gather with parameterized expert count.
//   - 4 + K kernel nodes: input source, router, topk_select, K expert_ffn
//     instances, and combine.
//   - 3 + 2K edges: input->router, router->topk, topk->combine (gate
//     weights), plus K topk->expert and K expert->combine edges.
//
// Reference: spec-bench-llm.md Section 3

#include "llm_taskgraph_common.h"

#include <iostream>
#include <string>
#include <vector>

// Stub kernel function declarations.
static void input_source_stub() {}
static void router_stub(const float *, const float *, float *, int, int,
                        int) {}
static void topk_select_stub(const float *, int *, float *, int, int, int) {}
static void expert_ffn_stub(const float *, const float *, float *, int, int,
                            int) {}
static void combine_stub(const float *, const float *, const int *, float *,
                         int, int) {}

namespace llm_bench {

// ============================================================================
// MoE Layer TaskGraph builder
// ============================================================================

tapestry::TaskGraph buildMoELayerTaskGraph(const LLMDims &dims) {
  tapestry::TaskGraph tg("moe_layer");

  // --- Kernel nodes (4 + K total) ---

  // Source node representing the external layer input.
  auto input = tg.kernel("input", input_source_stub);

  auto rtr = tg.kernel("router", router_stub);
  auto topk = tg.kernel("topk_select", topk_select_stub);
  auto comb = tg.kernel("combine", combine_stub);

  // --- Edge: input -> router (token activations [B, S, D]) ---
  tg.connect(input, rtr)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // --- Edge: router -> topk_select (gating logits [B, S, E]) ---
  tg.connect(rtr, topk)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volGatingLogits(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // --- Parameterized expert instances and their edges ---
  std::vector<tapestry::KernelHandle> experts;
  for (uint64_t i = 0; i < dims.K; ++i) {
    std::string expertName = "expert_ffn_" + std::to_string(i);
    auto exp_i = tg.kernel(expertName, expert_ffn_stub);
    experts.push_back(exp_i);

    // topk_select -> expert_ffn_i (gathered token subset, UNORDERED
    // because token routing is data-dependent)
    tg.connect(topk, exp_i)
        .ordering(tapestry::Ordering::UNORDERED)
        .data_type<float>()
        .data_volume(volRoutedTokens(dims))
        .placement(tapestry::Placement::SHARED_L2);

    // expert_ffn_i -> combine (expert output, UNORDERED because
    // results are scattered back by position, not arrival order)
    tg.connect(exp_i, comb)
        .ordering(tapestry::Ordering::UNORDERED)
        .data_type<float>()
        .data_volume(volRoutedTokens(dims))
        .placement(tapestry::Placement::SHARED_L2);
  }

  // --- Edge: topk_select -> combine (gate weights + expert indices) ---
  // This control-data edge carries the routing metadata needed for
  // weighted aggregation: [B, S, K] weights + [B, S, K] indices.
  tg.connect(topk, comb)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volGateWeightsIndices(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  return tg;
}

// ============================================================================
// Variant registration for MoE Layer kernels
// ============================================================================

void registerMoELayerVariants(tapestry::TaskGraph &tg, const LLMDims &dims) {
  // Placeholder -- see buildMoELayerWithVariants().
  (void)tg;
  (void)dims;
}

// Full builder with variant registration.
//
// Variant counts per kernel:
//   router: 4 (dense, noisy_topk, sigmoid, hash)
//   topk_select: 3 (linear_scan, heap, threshold)
//   expert_ffn_i: 4 each (full, narrow, shared_up, tile64)
//   combine: 3 (weighted_sum, renormalized, top1_passthrough)
tapestry::TaskGraph buildMoELayerWithVariants(const LLMDims &dims) {
  tapestry::TaskGraph tg("moe_layer");

  auto input = tg.kernel("input", input_source_stub);
  auto rtr = tg.kernel("router", router_stub);
  auto topk = tg.kernel("topk_select", topk_select_stub);
  auto comb = tg.kernel("combine", combine_stub);

  tg.connect(input, rtr)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));

  tg.connect(rtr, topk)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volGatingLogits(dims));

  std::vector<tapestry::KernelHandle> experts;
  for (uint64_t i = 0; i < dims.K; ++i) {
    std::string expertName = "expert_ffn_" + std::to_string(i);
    auto exp_i = tg.kernel(expertName, expert_ffn_stub);
    experts.push_back(exp_i);

    tg.connect(topk, exp_i)
        .ordering(tapestry::Ordering::UNORDERED)
        .data_type<float>()
        .data_volume(volRoutedTokens(dims));

    tg.connect(exp_i, comb)
        .ordering(tapestry::Ordering::UNORDERED)
        .data_type<float>()
        .data_volume(volRoutedTokens(dims));
  }

  tg.connect(topk, comb)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volGateWeightsIndices(dims));

  // --- Variant registration ---

  // router variants
  tg.addVariant(rtr, "router_dense", {});
  tg.addVariant(rtr, "router_noisy_topk", {});
  tg.addVariant(rtr, "router_sigmoid", {});
  tg.addVariant(rtr, "router_hash", {});

  // topk_select variants
  tg.addVariant(topk, "topk_linear_scan", {});
  tg.addVariant(topk, "topk_heap", {});
  tg.addVariant(topk, "topk_threshold", {});

  // expert_ffn variants (per expert instance)
  for (uint64_t i = 0; i < dims.K; ++i) {
    std::string prefix = "expert_ffn_" + std::to_string(i) + "_";
    tg.addVariant(experts[i], prefix + "full", {});
    tg.addVariant(experts[i], prefix + "narrow", {});
    tg.addVariant(experts[i], prefix + "shared_up", {});
    tg.addVariant(experts[i], prefix + "tile64", {.unrollFactor = 64});
  }

  // combine variants
  tg.addVariant(comb, "combine_weighted_sum", {});
  tg.addVariant(comb, "combine_renormalized", {});
  tg.addVariant(comb, "combine_top1_passthrough", {});

  return tg;
}

} // namespace llm_bench

// ============================================================================
// main: construct and dump the MoE Layer TaskGraph
// ============================================================================

int main(int argc, char *argv[]) {
  bool withVariants = false;
  llm_bench::LLMDims dims;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--variants") {
      withVariants = true;
    } else if (arg == "--K" && i + 1 < argc) {
      dims.K = std::stoull(argv[++i]);
    }
  }

  if (withVariants) {
    auto tg = llm_bench::buildMoELayerWithVariants(dims);
    tg.dump();
  } else {
    auto tg = llm_bench::buildMoELayerTaskGraph(dims);
    tg.dump();
  }

  return 0;
}
