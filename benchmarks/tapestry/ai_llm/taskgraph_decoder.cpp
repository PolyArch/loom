// TaskGraph definition for Application 1: Transformer Decoder Layer.
//
// Topology: 11 computational kernels + 1 "input" source node, connected
// by 14 edges including residual skip connections around the attention
// and FFN sub-blocks.
//
// Reference: spec-bench-llm.md Section 2

#include "llm_taskgraph_common.h"

#include <iostream>
#include <string>

// Stub kernel function declarations.
// These are placeholders that satisfy the function-pointer API;
// the actual C kernel implementations live in the per-kernel .c files.
static void layernorm_stub(const float *, float *, int, int) {}
static void qkv_proj_stub(const float *, const float *, float *, int, int,
                           int) {}
static void attn_score_stub(const float *, const float *, float *, int, int,
                            int) {}
static void softmax_stub(float *, int, int) {}
static void attn_apply_stub(const float *, const float *, float *, int, int,
                            int) {}
static void add_residual_stub(const float *, const float *, float *, int) {}
static void ffn_up_stub(const float *, const float *, float *, int, int,
                        int) {}
static void activation_stub(float *, int) {}
static void ffn_down_stub(const float *, const float *, float *, int, int,
                          int) {}
static void input_source_stub() {}

namespace llm_bench {

// ============================================================================
// Decoder Layer TaskGraph builder
// ============================================================================

tapestry::TaskGraph buildDecoderLayerTaskGraph(const LLMDims &dims) {
  tapestry::TaskGraph tg("transformer_decoder_layer");

  // --- Kernel nodes ---

  // Source node representing the external layer input. This is not a
  // computational kernel but models the data source for the first
  // layernorm and the residual skip to add_res_1.
  auto input = tg.kernel("input", input_source_stub);

  auto ln1 = tg.kernel("layernorm_1", layernorm_stub);
  auto qkv = tg.kernel("qkv_proj", qkv_proj_stub);
  auto attn_sc = tg.kernel("attn_score", attn_score_stub);
  auto smax = tg.kernel("softmax", softmax_stub);
  auto attn_ap = tg.kernel("attn_apply", attn_apply_stub);
  auto add_r1 = tg.kernel("add_res_1", add_residual_stub);
  auto ln2 = tg.kernel("layernorm_2", layernorm_stub);
  auto ffn_u = tg.kernel("ffn_up", ffn_up_stub);
  auto act = tg.kernel("activation", activation_stub);
  auto ffn_d = tg.kernel("ffn_down", ffn_down_stub);
  auto add_r2 = tg.kernel("add_res_2", add_residual_stub);

  // --- Edges (14 total) ---

  // E1: input -> layernorm_1 (token activations enter the layer)
  tg.connect(input, ln1)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E2: layernorm_1 -> qkv_proj (normalized activations)
  tg.connect(ln1, qkv)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E3: qkv_proj -> attn_score (Q and K tensors, 2 * B * S * D)
  tg.connect(qkv, attn_sc)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volQK(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E4: qkv_proj -> attn_apply (V tensor, second outgoing edge)
  tg.connect(qkv, attn_ap)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volV(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E5: attn_score -> softmax (raw attention scores [B, H, S, S])
  tg.connect(attn_sc, smax)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volAttnScore(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E6: softmax -> attn_apply (normalized attention weights)
  tg.connect(smax, attn_ap)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volAttnScore(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E7: attn_apply -> add_res_1 (attention output [B, S, D])
  tg.connect(attn_ap, add_r1)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E8: input -> add_res_1 (residual skip around the attention block)
  tg.connect(input, add_r1)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E9: add_res_1 -> layernorm_2 (post-attention activations)
  tg.connect(add_r1, ln2)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E10: layernorm_2 -> ffn_up (normalized activations)
  tg.connect(ln2, ffn_u)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E11: ffn_up -> activation (expanded [B, S, D_ff])
  tg.connect(ffn_u, act)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volFFNIntermediate(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E12: activation -> ffn_down (activated values [B, S, D_ff])
  tg.connect(act, ffn_d)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volFFNIntermediate(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E13: ffn_down -> add_res_2 (contracted [B, S, D])
  tg.connect(ffn_d, add_r2)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E14: add_res_1 -> add_res_2 (residual skip around the FFN block)
  tg.connect(add_r1, add_r2)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  return tg;
}

// ============================================================================
// Variant registration for Decoder Layer kernels
// ============================================================================

void registerDecoderLayerVariants(tapestry::TaskGraph &tg) {
  // The addVariant API requires KernelHandles, which are only available
  // during graph construction. This standalone registration function
  // cannot reconstruct handles from a completed graph.
  //
  // Use buildDecoderLayerWithVariants() instead, which creates the
  // topology and registers variants in a single construction pass.
  (void)tg;
}

// Full builder that includes variant registration. Returns the graph and
// also registers all specified variants per spec Section 2.2.
//
// Variant counts per kernel:
//   layernorm_1, layernorm_2: 4 each (fused_2pass, online_welford,
//                                      tiled_32, tiled_64)
//   qkv_proj: 4 (tile32, tile64, tile128, separate)
//   attn_score: 4 (tile32, tile64, causal_masked, flash_tiled)
//   softmax: 4 (3pass, online_2pass, tiled_32, fused_with_score)
//   attn_apply: 3 (tile32, tile64, tile128)
//   add_res_1, add_res_2: 2 each (inplace, separate)
//   ffn_up: 4 (tile32, tile64, tile128, gated)
//   activation: 4 (gelu_poly, gelu_lut, silu, relu)
//   ffn_down: 3 (tile32, tile64, tile128)
tapestry::TaskGraph buildDecoderLayerWithVariants(const LLMDims &dims) {
  tapestry::TaskGraph tg("transformer_decoder_layer");

  // --- Kernel nodes (12 = 1 input + 11 computational) ---
  auto input = tg.kernel("input", input_source_stub);
  auto ln1 = tg.kernel("layernorm_1", layernorm_stub);
  auto qkv = tg.kernel("qkv_proj", qkv_proj_stub);
  auto attn_sc = tg.kernel("attn_score", attn_score_stub);
  auto smax = tg.kernel("softmax", softmax_stub);
  auto attn_ap = tg.kernel("attn_apply", attn_apply_stub);
  auto add_r1 = tg.kernel("add_res_1", add_residual_stub);
  auto ln2 = tg.kernel("layernorm_2", layernorm_stub);
  auto ffn_u = tg.kernel("ffn_up", ffn_up_stub);
  auto act = tg.kernel("activation", activation_stub);
  auto ffn_d = tg.kernel("ffn_down", ffn_down_stub);
  auto add_r2 = tg.kernel("add_res_2", add_residual_stub);

  // --- Edges (14 total, same as buildDecoderLayerTaskGraph) ---
  tg.connect(input, ln1)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));
  tg.connect(ln1, qkv)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));
  tg.connect(qkv, attn_sc)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volQK(dims));
  tg.connect(qkv, attn_ap)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volV(dims));
  tg.connect(attn_sc, smax)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volAttnScore(dims));
  tg.connect(smax, attn_ap)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volAttnScore(dims));
  tg.connect(attn_ap, add_r1)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));
  tg.connect(input, add_r1)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));
  tg.connect(add_r1, ln2)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));
  tg.connect(ln2, ffn_u)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));
  tg.connect(ffn_u, act)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volFFNIntermediate(dims));
  tg.connect(act, ffn_d)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volFFNIntermediate(dims));
  tg.connect(ffn_d, add_r2)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));
  tg.connect(add_r1, add_r2)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volActivation(dims));

  // --- Variant registration ---

  // layernorm_1 variants
  tg.addVariant(ln1, "layernorm_1_fused_2pass", {});
  tg.addVariant(ln1, "layernorm_1_online_welford", {});
  tg.addVariant(ln1, "layernorm_1_tiled_32", {.unrollFactor = 32});
  tg.addVariant(ln1, "layernorm_1_tiled_64", {.unrollFactor = 64});

  // qkv_proj variants
  tg.addVariant(qkv, "qkv_proj_tile32", {.unrollFactor = 32});
  tg.addVariant(qkv, "qkv_proj_tile64", {.unrollFactor = 64});
  tg.addVariant(qkv, "qkv_proj_tile128", {.unrollFactor = 128});
  tg.addVariant(qkv, "qkv_proj_separate", {});

  // attn_score variants
  tg.addVariant(attn_sc, "attn_score_tile32", {.unrollFactor = 32});
  tg.addVariant(attn_sc, "attn_score_tile64", {.unrollFactor = 64});
  tg.addVariant(attn_sc, "attn_score_causal_masked", {});
  tg.addVariant(attn_sc, "attn_score_flash_tiled", {});

  // softmax variants
  tg.addVariant(smax, "softmax_3pass", {});
  tg.addVariant(smax, "softmax_online_2pass", {});
  tg.addVariant(smax, "softmax_tiled_32", {.unrollFactor = 32});
  tg.addVariant(smax, "softmax_fused_with_score", {});

  // attn_apply variants
  tg.addVariant(attn_ap, "attn_apply_tile32", {.unrollFactor = 32});
  tg.addVariant(attn_ap, "attn_apply_tile64", {.unrollFactor = 64});
  tg.addVariant(attn_ap, "attn_apply_tile128", {.unrollFactor = 128});

  // add_res_1 variants
  tg.addVariant(add_r1, "add_res_1_inplace", {});
  tg.addVariant(add_r1, "add_res_1_separate", {});

  // layernorm_2 variants
  tg.addVariant(ln2, "layernorm_2_fused_2pass", {});
  tg.addVariant(ln2, "layernorm_2_online_welford", {});
  tg.addVariant(ln2, "layernorm_2_tiled_32", {.unrollFactor = 32});
  tg.addVariant(ln2, "layernorm_2_tiled_64", {.unrollFactor = 64});

  // ffn_up variants
  tg.addVariant(ffn_u, "ffn_up_tile32", {.unrollFactor = 32});
  tg.addVariant(ffn_u, "ffn_up_tile64", {.unrollFactor = 64});
  tg.addVariant(ffn_u, "ffn_up_tile128", {.unrollFactor = 128});
  tg.addVariant(ffn_u, "ffn_up_gated", {});

  // activation variants
  tg.addVariant(act, "activation_gelu_poly", {});
  tg.addVariant(act, "activation_gelu_lut", {});
  tg.addVariant(act, "activation_silu", {});
  tg.addVariant(act, "activation_relu", {});

  // ffn_down variants
  tg.addVariant(ffn_d, "ffn_down_tile32", {.unrollFactor = 32});
  tg.addVariant(ffn_d, "ffn_down_tile64", {.unrollFactor = 64});
  tg.addVariant(ffn_d, "ffn_down_tile128", {.unrollFactor = 128});

  // add_res_2 variants
  tg.addVariant(add_r2, "add_res_2_inplace", {});
  tg.addVariant(add_r2, "add_res_2_separate", {});

  return tg;
}

} // namespace llm_bench

// ============================================================================
// main: construct and dump the Decoder Layer TaskGraph
// ============================================================================

int main(int argc, char *argv[]) {
  bool withVariants = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--variants") {
      withVariants = true;
    }
  }

  llm_bench::LLMDims dims;

  if (withVariants) {
    auto tg = llm_bench::buildDecoderLayerWithVariants(dims);
    tg.dump();
  } else {
    auto tg = llm_bench::buildDecoderLayerTaskGraph(dims);
    tg.dump();
  }

  return 0;
}
