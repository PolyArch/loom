// TaskGraph C++ DSL -- STARK Proof pipeline (ZK domain)
// E01 Productivity Comparison: Tier 1 DSL format

#include "tapestry/task_graph.h"

typedef unsigned int m31_t;
extern "C" {
void ntt_forward_tiled(m31_t *, int, int, const m31_t *);
void msm(const m31_t *, const m31_t *, m31_t *, int);
void poseidon_hash(const m31_t *, m31_t *, int, int);
void poly_eval(const m31_t *, const m31_t *, m31_t *, int, int);
void proof_compose(const m31_t *, const m31_t *, const m31_t *,
                   m31_t *, int);
}

tapestry::TaskGraph buildSTARKTDG() {
  tapestry::TaskGraph tg("stark_proof");

  auto k_ntt = tg.kernel("ntt", ntt_forward_tiled);
  auto k_msm = tg.kernel("msm", msm);
  auto k_hash = tg.kernel("poseidon_hash", poseidon_hash);
  auto k_poly = tg.kernel("poly_eval", poly_eval);
  auto k_comp = tg.kernel("proof_compose", proof_compose);

  tg.connect(k_ntt, k_poly)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<uint32_t>()
      .shape("1024")
      .data_volume(1024);

  tg.connect(k_poly, k_comp)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<uint32_t>()
      .shape("256")
      .data_volume(256);

  tg.connect(k_hash, k_comp)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<uint32_t>()
      .shape("4")
      .data_volume(4);

  tg.connect(k_msm, k_comp)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<uint32_t>()
      .shape("3")
      .data_volume(3);

  tg.connect(k_ntt, k_hash)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<uint32_t>()
      .shape("8")
      .data_volume(8);

  return tg;
}
