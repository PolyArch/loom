// TaskGraph definition for Application 3: Multi-Head Attention with
// KV Cache (Decode Phase).
//
// Topology: Fan-out from input to 3 parallel projections (Q, K, V),
// cache update, attention computation with full cache read, and
// output projection.
//   - 9 computational kernels + 1 "input" source node (10 total).
//   - 11 edges including 2 UNORDERED cache edges that carry the
//     dominant data volume.
//
// Reference: spec-bench-llm.md Section 4

#include "llm_taskgraph_common.h"

#include <iostream>
#include <string>

// Stub kernel function declarations.
static void q_proj_stub(const float *, const float *, float *, int, int) {}
static void k_proj_stub(const float *, const float *, float *, int, int) {}
static void v_proj_stub(const float *, const float *, float *, int, int) {}
static void kv_cache_update_stub(const float *, float *, int, int, int) {}
static void attn_score_cached_stub(const float *, const float *, float *, int,
                                   int, int) {}
static void softmax_stub(float *, int, int) {}
static void attn_apply_cached_stub(const float *, const float *, float *, int,
                                   int, int) {}
static void out_proj_stub(const float *, const float *, float *, int, int) {}
static void input_source_stub() {}

namespace llm_bench {

// ============================================================================
// KV-Cache Decode TaskGraph builder
// ============================================================================

tapestry::TaskGraph buildKVCacheDecodeTaskGraph(const LLMDims &dims) {
  tapestry::TaskGraph tg("mha_kv_cache_decode");

  // --- Kernel nodes (10 = 1 input + 9 computational) ---

  // Source node representing the new token embedding entering the layer.
  auto input = tg.kernel("input", input_source_stub);

  auto q_p = tg.kernel("q_proj", q_proj_stub);
  auto k_p = tg.kernel("k_proj", k_proj_stub);
  auto v_p = tg.kernel("v_proj", v_proj_stub);
  auto kv_k = tg.kernel("kv_cache_update_k", kv_cache_update_stub);
  auto kv_v = tg.kernel("kv_cache_update_v", kv_cache_update_stub);
  auto attn_sc = tg.kernel("attn_score_cached", attn_score_cached_stub);
  auto smax = tg.kernel("softmax", softmax_stub);
  auto attn_ap = tg.kernel("attn_apply_cached", attn_apply_cached_stub);
  auto out_p = tg.kernel("out_proj", out_proj_stub);

  // --- Edges (11 total) ---

  // E1: input -> q_proj (new token embedding [B, 1, D])
  tg.connect(input, q_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(fp32Bytes(dims.B * dims.D))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E2: input -> k_proj (new token embedding [B, 1, D])
  tg.connect(input, k_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(fp32Bytes(dims.B * dims.D))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E3: input -> v_proj (new token embedding [B, 1, D])
  tg.connect(input, v_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(fp32Bytes(dims.B * dims.D))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E4: q_proj -> attn_score_cached (query vector [B, H, 1, D_h])
  tg.connect(q_p, attn_sc)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volQueryVec(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E5: k_proj -> kv_cache_update_k (new key vector [B, H_kv, 1, D_h])
  tg.connect(k_p, kv_k)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volNewKV(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E6: v_proj -> kv_cache_update_v (new value vector [B, H_kv, 1, D_h])
  tg.connect(v_p, kv_v)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volNewKV(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E7: kv_cache_update_k -> attn_score_cached (full K cache)
  // UNORDERED: full cache read with random/sequential access depending
  // on cache management strategy. Dominant data volume.
  tg.connect(kv_k, attn_sc)
      .ordering(tapestry::Ordering::UNORDERED)
      .data_type<float>()
      .data_volume(volKVCache(dims))
      .placement(tapestry::Placement::SHARED_L2);

  // E8: kv_cache_update_v -> attn_apply_cached (full V cache)
  // UNORDERED: same reasoning as E7.
  tg.connect(kv_v, attn_ap)
      .ordering(tapestry::Ordering::UNORDERED)
      .data_type<float>()
      .data_volume(volKVCache(dims))
      .placement(tapestry::Placement::SHARED_L2);

  // E9: attn_score_cached -> softmax (scores [B, H, 1, C])
  tg.connect(attn_sc, smax)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volDecodeAttnScore(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E10: softmax -> attn_apply_cached (attention weights [B, H, 1, C])
  tg.connect(smax, attn_ap)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volDecodeAttnScore(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  // E11: attn_apply_cached -> out_proj (weighted value sum [B, H, 1, D_h])
  tg.connect(attn_ap, out_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volDecodeAttnOutput(dims))
      .placement(tapestry::Placement::LOCAL_SPM);

  return tg;
}

// ============================================================================
// Variant registration for KV-Cache Decode kernels
// ============================================================================

void registerKVCacheDecodeVariants(tapestry::TaskGraph &tg) {
  // Placeholder -- see buildKVCacheDecodeWithVariants().
  (void)tg;
}

// Full builder with variant registration.
//
// Variant counts per kernel (from spec Section 4.2):
//   q_proj: 3 (full, int8_dequant, grouped)
//   k_proj: 4 (mha, gqa, mqa, int8)
//   v_proj: 4 (mha, gqa, mqa, int8)
//   kv_cache_update_k, kv_cache_update_v: 3 each (linear, ring, paged)
//   attn_score_cached: 3 (full, blocked, sparse)
//   softmax: 4 (3pass, online_2pass, tiled_32, fused_with_score)
//   attn_apply_cached: 3 (full, blocked, fused)
//   out_proj: 3 (full, int8_dequant, low_rank)
tapestry::TaskGraph buildKVCacheDecodeWithVariants(const LLMDims &dims) {
  tapestry::TaskGraph tg("mha_kv_cache_decode");

  // --- Kernel nodes ---
  auto input = tg.kernel("input", input_source_stub);
  auto q_p = tg.kernel("q_proj", q_proj_stub);
  auto k_p = tg.kernel("k_proj", k_proj_stub);
  auto v_p = tg.kernel("v_proj", v_proj_stub);
  auto kv_k = tg.kernel("kv_cache_update_k", kv_cache_update_stub);
  auto kv_v = tg.kernel("kv_cache_update_v", kv_cache_update_stub);
  auto attn_sc = tg.kernel("attn_score_cached", attn_score_cached_stub);
  auto smax = tg.kernel("softmax", softmax_stub);
  auto attn_ap = tg.kernel("attn_apply_cached", attn_apply_cached_stub);
  auto out_p = tg.kernel("out_proj", out_proj_stub);

  // --- Edges (same 11 as buildKVCacheDecodeTaskGraph) ---
  tg.connect(input, q_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(fp32Bytes(dims.B * dims.D));
  tg.connect(input, k_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(fp32Bytes(dims.B * dims.D));
  tg.connect(input, v_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(fp32Bytes(dims.B * dims.D));
  tg.connect(q_p, attn_sc)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volQueryVec(dims));
  tg.connect(k_p, kv_k)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volNewKV(dims));
  tg.connect(v_p, kv_v)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volNewKV(dims));
  tg.connect(kv_k, attn_sc)
      .ordering(tapestry::Ordering::UNORDERED)
      .data_type<float>()
      .data_volume(volKVCache(dims));
  tg.connect(kv_v, attn_ap)
      .ordering(tapestry::Ordering::UNORDERED)
      .data_type<float>()
      .data_volume(volKVCache(dims));
  tg.connect(attn_sc, smax)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volDecodeAttnScore(dims));
  tg.connect(smax, attn_ap)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volDecodeAttnScore(dims));
  tg.connect(attn_ap, out_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(volDecodeAttnOutput(dims));

  // --- Variant registration ---

  // q_proj variants
  tg.addVariant(q_p, "q_proj_full", {});
  tg.addVariant(q_p, "q_proj_int8_dequant", {});
  tg.addVariant(q_p, "q_proj_grouped", {});

  // k_proj variants
  tg.addVariant(k_p, "kv_proj_mha", {});
  tg.addVariant(k_p, "kv_proj_gqa", {});
  tg.addVariant(k_p, "kv_proj_mqa", {});
  tg.addVariant(k_p, "kv_proj_int8", {});

  // v_proj variants
  tg.addVariant(v_p, "v_proj_mha", {});
  tg.addVariant(v_p, "v_proj_gqa", {});
  tg.addVariant(v_p, "v_proj_mqa", {});
  tg.addVariant(v_p, "v_proj_int8", {});

  // kv_cache_update_k variants
  tg.addVariant(kv_k, "kv_cache_k_linear", {});
  tg.addVariant(kv_k, "kv_cache_k_ring", {});
  tg.addVariant(kv_k, "kv_cache_k_paged", {});

  // kv_cache_update_v variants
  tg.addVariant(kv_v, "kv_cache_v_linear", {});
  tg.addVariant(kv_v, "kv_cache_v_ring", {});
  tg.addVariant(kv_v, "kv_cache_v_paged", {});

  // attn_score_cached variants
  tg.addVariant(attn_sc, "attn_score_cached_full", {});
  tg.addVariant(attn_sc, "attn_score_cached_blocked", {});
  tg.addVariant(attn_sc, "attn_score_cached_sparse", {});

  // softmax variants (same set as decoder layer)
  tg.addVariant(smax, "softmax_3pass", {});
  tg.addVariant(smax, "softmax_online_2pass", {});
  tg.addVariant(smax, "softmax_tiled_32", {.unrollFactor = 32});
  tg.addVariant(smax, "softmax_fused_with_score", {});

  // attn_apply_cached variants
  tg.addVariant(attn_ap, "attn_apply_cached_full", {});
  tg.addVariant(attn_ap, "attn_apply_cached_blocked", {});
  tg.addVariant(attn_ap, "attn_apply_cached_fused", {});

  // out_proj variants
  tg.addVariant(out_p, "out_proj_full", {});
  tg.addVariant(out_p, "out_proj_int8_dequant", {});
  tg.addVariant(out_p, "out_proj_low_rank", {});

  return tg;
}

} // namespace llm_bench

// ============================================================================
// main: construct and dump the KV-Cache Decode TaskGraph
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
    auto tg = llm_bench::buildKVCacheDecodeWithVariants(dims);
    tg.dump();
  } else {
    auto tg = llm_bench::buildKVCacheDecodeTaskGraph(dims);
    tg.dump();
  }

  return 0;
}
