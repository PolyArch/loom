// TaskGraph C++ DSL -- OFDM Receiver Chain (DSP domain)
// E01 Productivity Comparison: Tier 1 DSL format

#include "tapestry/task_graph.h"

// Forward declarations
typedef struct { float re, im; } cmplx_t;
extern "C" {
void fft_butterfly(cmplx_t *, int);
void channel_est(const cmplx_t *, cmplx_t *, int, int);
void equalizer(const cmplx_t *, const cmplx_t *, cmplx_t *, int);
void qam_demod(const cmplx_t *, int *, int, int);
void viterbi(const int *, int *, int, int);
void crc_check(const int *, int *, int);
}

tapestry::TaskGraph buildOFDMTDG() {
  tapestry::TaskGraph tg("ofdm_receiver");

  auto k_fft = tg.kernel("fft_butterfly", fft_butterfly);
  auto k_ch = tg.kernel("channel_est", channel_est);
  auto k_eq = tg.kernel("equalizer", equalizer);
  auto k_qam = tg.kernel("qam_demod", qam_demod);
  auto k_vit = tg.kernel("viterbi", viterbi);
  auto k_crc = tg.kernel("crc_check", crc_check);

  tg.connect(k_fft, k_ch)
      .ordering(tapestry::Ordering::FIFO)
      .data_type("complex64")
      .shape("4096")
      .data_volume(4096);

  tg.connect(k_ch, k_eq)
      .ordering(tapestry::Ordering::FIFO)
      .data_type("complex64")
      .shape("1200")
      .data_volume(1200);

  tg.connect(k_eq, k_qam)
      .ordering(tapestry::Ordering::FIFO)
      .data_type("complex64")
      .shape("128")
      .data_volume(1200);

  tg.connect(k_qam, k_vit)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<int32_t>()
      .shape("7200")
      .data_volume(7200);

  tg.connect(k_vit, k_crc)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<int32_t>()
      .shape("3600")
      .data_volume(1800);

  return tg;
}
