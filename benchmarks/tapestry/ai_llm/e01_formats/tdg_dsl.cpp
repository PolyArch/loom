// TaskGraph C++ DSL -- Transformer Layer pipeline (AI/LLM domain)
// E01 Productivity Comparison: Tier 1 DSL format

#include "tapestry/task_graph.h"

// Forward declarations for kernel functions
extern "C" {
void qkv_proj(const float *, const float *, float *, int, int, int);
void attn_score(const float *, const float *, float *, int, int, int);
void softmax(float *, int, int);
void attn_output(const float *, const float *, float *, int, int, int);
void ffn1(const float *, const float *, float *, int, int, int);
void gelu(float *, int);
void ffn2(const float *, const float *, float *, int, int, int);
void layernorm(float *, int, int);
}

tapestry::TaskGraph buildTransformerTDG() {
  tapestry::TaskGraph tg("transformer_layer");

  auto k_qkv = tg.kernel("qkv_proj", qkv_proj);
  auto k_attn = tg.kernel("attn_score", attn_score);
  auto k_soft = tg.kernel("softmax", softmax);
  auto k_out = tg.kernel("attn_output", attn_output);
  auto k_ffn1 = tg.kernel("ffn1", ffn1);
  auto k_gelu = tg.kernel("gelu", gelu);
  auto k_ffn2 = tg.kernel("ffn2", ffn2);
  auto k_ln = tg.kernel("layernorm", layernorm);

  tg.connect(k_qkv, k_attn)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("32x64")
      .data_volume(2048);

  tg.connect(k_attn, k_soft)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("32x128")
      .data_volume(4096);

  tg.connect(k_soft, k_out)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("32x128")
      .data_volume(4096);

  tg.connect(k_out, k_ffn1)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("32x512")
      .data_volume(16384);

  tg.connect(k_ffn1, k_gelu)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("32x2048")
      .data_volume(65536);

  tg.connect(k_gelu, k_ffn2)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("32x2048")
      .data_volume(65536);

  tg.connect(k_ffn2, k_ln)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("32x512")
      .data_volume(16384);

  return tg;
}
