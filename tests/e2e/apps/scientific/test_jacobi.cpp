#include "reference/jacobi_ref.h"
#include "scicomp_params.h"
#include "test_scicomp_utils.h"

#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>
#include <vector>

extern "C" {
void halo_row_only(float *tile, int rows, int cols, int halo_w);
void stencil_5pt(const float *in, float *out, int rows, int cols, int halo_w,
                 float factor);
void boundary_dirichlet(float *tile, int rows, int cols, int halo_w,
                        float value);
float residual_l2(const float *curr, const float *prev, int rows, int cols);
}

namespace {

using namespace scicomp;
using scicomp_test::JacobiReferenceResult;
using scicomp_test::grids_close;
using scicomp_test::init_jacobi_problem;
using scicomp_test::make_jacobi_grid;
using scicomp_test::nearly_equal;
using scicomp_test::run_jacobi_reference;

static void halo_exchange_stub(float *, const float *, float *, unsigned,
                               unsigned, unsigned) {}
static void stencil_compute_stub(const float *, float *, unsigned, unsigned,
                                 unsigned) {}
static void boundary_apply_stub(float *, unsigned, unsigned, float) {}
static void residual_check_stub(const float *, const float *, unsigned,
                                unsigned) {}

static tapestry::TaskGraph buildJacobiTaskGraph(const JacobiParams &params) {
  tapestry::TaskGraph tg("jacobi_2d");
  auto halo = tg.kernel("halo_exchange", halo_exchange_stub);
  auto stencil = tg.kernel("stencil_compute", stencil_compute_stub);
  auto boundary = tg.kernel("boundary_apply", boundary_apply_stub);
  auto residual = tg.kernel("residual_check", residual_check_stub);

  tg.connect(halo, stencil)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(haloBytes(params))
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(stencil, boundary)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(tileBytes(params))
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(boundary, residual)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(2 * tileBytes(params))
      .placement(tapestry::Placement::LOCAL_SPM);
  return tg;
}

static bool check_taskgraph_emits_tdg() {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  JacobiParams params;
  auto tg = buildJacobiTaskGraph(params);
  auto module = tapestry::emitTDG(tg, ctx);
  return static_cast<bool>(module);
}

static bool run_jacobi_smoke() {
  JacobiParams params;
  params.tileRows = 8;
  params.tileCols = 8;
  params.haloWidth = 1;

  const int rows = static_cast<int>(params.tileRows);
  const int cols = static_cast<int>(params.tileCols);
  const int halo_w = static_cast<int>(params.haloWidth);
  const int total_rows = rows + 2 * halo_w;
  const int total_cols = cols + 2 * halo_w;
  const float factor = 0.25f;
  const float stop_eps = -1.0f;
  const int max_iters = 3;

  std::vector<float> initial = make_jacobi_grid(rows, cols, halo_w);
  init_jacobi_problem(initial, rows, cols, halo_w);

  JacobiReferenceResult ref =
      run_jacobi_reference(initial, rows, cols, halo_w, factor, max_iters,
                           stop_eps);

  std::vector<float> curr = initial;
  std::vector<float> next = initial;
  std::vector<float> residuals;

  for (int iter = 0; iter < max_iters; ++iter) {
    halo_row_only(curr.data(), rows, cols, halo_w);
    stencil_5pt(curr.data(), next.data(), rows, cols, halo_w, factor);
    boundary_dirichlet(next.data(), rows, cols, halo_w, 0.0f);
    const float residual =
        residual_l2(next.data(), curr.data(), total_rows, total_cols);
    residuals.push_back(residual);
    curr.swap(next);
    if (residual < stop_eps)
      break;
  }

  if (!grids_close(curr, ref.grid, 1e-4f)) {
    for (size_t i = 0; i < curr.size(); ++i) {
      if (!nearly_equal(curr[i], ref.grid[i], 1e-4f)) {
        const size_t row = i / static_cast<size_t>(total_cols);
        const size_t col = i % static_cast<size_t>(total_cols);
        std::printf("FAIL jacobi grid mismatch at (%zu,%zu): %f vs %f\n",
                    row, col, curr[i], ref.grid[i]);
        break;
      }
    }
    return false;
  }

  if (residuals.empty()) {
    std::printf("FAIL jacobi residuals empty\n");
    return false;
  }

  if (!nearly_equal(residuals.back(), ref.residual, 1e-3f)) {
    std::printf("FAIL jacobi residual mismatch %f %f\n", residuals.back(),
                ref.residual);
    return false;
  }

  return true;
}

} // namespace

int main() {
  bool ok = check_taskgraph_emits_tdg() && run_jacobi_smoke();
  std::puts(ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
