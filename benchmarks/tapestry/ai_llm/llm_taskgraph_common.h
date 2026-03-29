// Common symbolic dimension constants and data volume helpers for
// LLM benchmark TaskGraph definitions.

#ifndef BENCHMARKS_TAPESTRY_AI_LLM_TASKGRAPH_COMMON_H
#define BENCHMARKS_TAPESTRY_AI_LLM_TASKGRAPH_COMMON_H

#include "tapestry/task_graph.h"

#include <cstdint>
#include <string>

namespace llm_bench {

// ============================================================================
// Symbolic dimension parameters with sensible defaults.
// Can be overridden via compile-time defines (e.g., -DLLM_B=4).
// ============================================================================

#ifndef LLM_B
#define LLM_B 1
#endif
#ifndef LLM_S
#define LLM_S 128
#endif
#ifndef LLM_D
#define LLM_D 512
#endif
#ifndef LLM_D_FF
#define LLM_D_FF 2048
#endif
#ifndef LLM_H
#define LLM_H 8
#endif
#ifndef LLM_D_H
#define LLM_D_H 64
#endif
#ifndef LLM_E
#define LLM_E 8
#endif
#ifndef LLM_K
#define LLM_K 2
#endif
#ifndef LLM_C
#define LLM_C 4096
#endif
#ifndef LLM_H_KV
#define LLM_H_KV 8
#endif

struct LLMDims {
  uint64_t B = LLM_B;
  uint64_t S = LLM_S;
  uint64_t D = LLM_D;
  uint64_t D_ff = LLM_D_FF;
  uint64_t H = LLM_H;
  uint64_t D_h = LLM_D_H;
  uint64_t E = LLM_E;
  uint64_t K = LLM_K;
  uint64_t C = LLM_C;
  uint64_t H_kv = LLM_H_KV;
};

// ============================================================================
// FP32 data volume helpers (all return bytes)
// ============================================================================

inline uint64_t fp32Bytes(uint64_t numElements) {
  return 4 * numElements;
}

// --- Decoder Layer edge volumes ---

// [B, S, D] activation volume.
inline uint64_t volActivation(const LLMDims &d) {
  return fp32Bytes(d.B * d.S * d.D);
}

// Q and K tensors combined: [B, H, S, D_h] x 2 = 2 * B * S * D elements.
inline uint64_t volQK(const LLMDims &d) {
  return fp32Bytes(2 * d.B * d.H * d.S * d.D_h);
}

// V tensor: [B, H, S, D_h] = B * S * D elements.
inline uint64_t volV(const LLMDims &d) {
  return fp32Bytes(d.B * d.H * d.S * d.D_h);
}

// Attention score matrix: [B, H, S, S].
inline uint64_t volAttnScore(const LLMDims &d) {
  return fp32Bytes(d.B * d.H * d.S * d.S);
}

// FFN intermediate: [B, S, D_ff].
inline uint64_t volFFNIntermediate(const LLMDims &d) {
  return fp32Bytes(d.B * d.S * d.D_ff);
}

// --- MoE Layer edge volumes ---

// Router gating logits: [B, S, E].
inline uint64_t volGatingLogits(const LLMDims &d) {
  return fp32Bytes(d.B * d.S * d.E);
}

// Tokens routed to one expert (approximate): B_e * S * D where
// B_e = B * K / E (average tokens per expert).
inline uint64_t volRoutedTokens(const LLMDims &d) {
  return fp32Bytes((d.B * d.S * d.K / d.E) * d.D);
}

// Gate weights and indices for combine: [B, S, K] weights + [B, S, K]
// indices = 2 * B * S * K elements.
inline uint64_t volGateWeightsIndices(const LLMDims &d) {
  return fp32Bytes(2 * d.B * d.S * d.K);
}

// --- KV-Cache Decode edge volumes (S=1 for decode) ---

// Query vector: [B, H, 1, D_h] = B * H * D_h elements.
inline uint64_t volQueryVec(const LLMDims &d) {
  return fp32Bytes(d.B * d.H * d.D_h);
}

// New key or value vector: [B, H_kv, 1, D_h] = B * H_kv * D_h elements.
inline uint64_t volNewKV(const LLMDims &d) {
  return fp32Bytes(d.B * d.H_kv * d.D_h);
}

// Full KV cache read: [B, H_kv, C, D_h].
inline uint64_t volKVCache(const LLMDims &d) {
  return fp32Bytes(d.B * d.H_kv * d.C * d.D_h);
}

// Attention score vector for decode: [B, H, 1, C] = B * H * C elements.
inline uint64_t volDecodeAttnScore(const LLMDims &d) {
  return fp32Bytes(d.B * d.H * d.C);
}

// Attention output per head for decode: [B, H, 1, D_h] = B * H * D_h.
inline uint64_t volDecodeAttnOutput(const LLMDims &d) {
  return fp32Bytes(d.B * d.H * d.D_h);
}

// ============================================================================
// Builder function declarations
// ============================================================================

// Build the Transformer Decoder Layer TaskGraph.
// Topology: 12 kernel nodes (11 computational + 1 "input" source),
// connected by 14 edges including 2 residual skip connections.
// Does NOT register variants.
tapestry::TaskGraph buildDecoderLayerTaskGraph(const LLMDims &dims = {});

// Build Decoder Layer TaskGraph WITH variant registration.
tapestry::TaskGraph buildDecoderLayerWithVariants(const LLMDims &dims = {});

// Register all kernel variants on a Decoder Layer TaskGraph.
void registerDecoderLayerVariants(tapestry::TaskGraph &tg);

// Build the MoE Layer TaskGraph.
// Topology: 4 + K kernel nodes (3 fixed + 1 "input" source + K experts),
// connected by 3 + 2K edges.
// Does NOT register variants.
tapestry::TaskGraph buildMoELayerTaskGraph(const LLMDims &dims = {});

// Build MoE Layer TaskGraph WITH variant registration.
tapestry::TaskGraph buildMoELayerWithVariants(const LLMDims &dims = {});

// Register all kernel variants on a MoE Layer TaskGraph.
void registerMoELayerVariants(tapestry::TaskGraph &tg,
                              const LLMDims &dims = {});

// Build the KV-Cache Decode TaskGraph.
// Topology: 10 kernel nodes (9 computational + 1 "input" source),
// connected by 11 edges including 2 UNORDERED cache edges.
// Does NOT register variants.
tapestry::TaskGraph buildKVCacheDecodeTaskGraph(const LLMDims &dims = {});

// Build KV-Cache Decode TaskGraph WITH variant registration.
tapestry::TaskGraph buildKVCacheDecodeWithVariants(const LLMDims &dims = {});

// Register all kernel variants on a KV-Cache Decode TaskGraph.
void registerKVCacheDecodeVariants(tapestry::TaskGraph &tg);

} // namespace llm_bench

#endif // BENCHMARKS_TAPESTRY_AI_LLM_TASKGRAPH_COMMON_H
