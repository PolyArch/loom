#include "reference/cg_ref.h"
#include "scicomp_params.h"
#include "test_scicomp_utils.h"

#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" {
void spmv_csr(const int *row_ptr, const int *col_idx, const float *values,
              const float *x, float *y, int nrows);
void spmv_ell(const int *col_idx, const float *values, const float *x, float *y,
              int nrows, int max_nnz);
void spmv_csr_unroll2(const int *row_ptr, const int *col_idx,
                      const float *values, const float *x, float *y,
                      int nrows);
float dot_sequential(const float *a, const float *b, int n);
float dot_tree_reduce(const float *a, const float *b, int n);
float dot_sequential_unroll2(const float *a, const float *b, int n);
void axpy_basic(float alpha, const float *x, float *y, int n);
void axpy_unroll4(float alpha, const float *x, float *y, int n);
void precond_diag(const float *r, const float *m_diag, float *z, int n);
void precond_block4(const float *block4, const float *r, float *z, int n);
float conv_l2(const float *r, int n);
float conv_max(const float *r, int n);
}

namespace {

using namespace scicomp;
using namespace scicomp_test;

static void spmv_stub(const float *, const int32_t *, const float *, float *,
                      unsigned, unsigned) {}
static void dot_stub(const float *, const float *, float *, unsigned) {}
static void axpy_stub(float *, const float *, const float *, float, unsigned) {}
static void precond_stub(float *, const float *, const float *, unsigned,
                         unsigned) {}
static void convergence_stub(const float *, unsigned, float *) {}

static tapestry::TaskGraph buildCGTaskGraph(const CGParams &params) {
  tapestry::TaskGraph tg("conjugate_gradient");

  auto spmv = tg.kernel("spmv", spmv_stub);
  auto dot_pq = tg.kernel("dot_pq", dot_stub);
  auto axpy_x = tg.kernel("axpy_x", axpy_stub);
  auto axpy_r = tg.kernel("axpy_r", axpy_stub);
  auto precondition = tg.kernel("precondition", precond_stub);
  auto dot_rz = tg.kernel("dot_rz", dot_stub);
  auto axpy_p = tg.kernel("axpy_p", axpy_stub);
  auto convergence = tg.kernel("convergence_check", convergence_stub);

  tg.addVariant(spmv, "spmv_ell", tapestry::VariantOptions{1, 1});
  tg.addVariant(spmv, "spmv_csr_unroll2", tapestry::VariantOptions{2, 0});
  tg.addVariant(dot_pq, "dot_tree_reduce", tapestry::VariantOptions{1, 1});
  tg.addVariant(dot_pq, "dot_sequential_unroll2",
                tapestry::VariantOptions{2, 0});
  tg.addVariant(axpy_x, "axpy_unroll4", tapestry::VariantOptions{4, 0});
  tg.addVariant(axpy_r, "axpy_unroll4", tapestry::VariantOptions{4, 0});
  tg.addVariant(precondition, "precond_block4",
                tapestry::VariantOptions{1, 1});
  tg.addVariant(dot_rz, "dot_tree_reduce", tapestry::VariantOptions{1, 1});
  tg.addVariant(dot_rz, "dot_sequential_unroll2",
                tapestry::VariantOptions{2, 0});
  tg.addVariant(axpy_p, "axpy_unroll4", tapestry::VariantOptions{4, 0});
  tg.addVariant(convergence, "conv_max", tapestry::VariantOptions{1, 1});

  const uint64_t vec_bytes = cgVectorBytes(params);
  const uint64_t scalar_bytes = cgScalarBytes();

  tg.connect(spmv, dot_pq)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(2 * vec_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(spmv, axpy_r)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(vec_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(dot_pq, axpy_x)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(scalar_bytes + vec_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(dot_pq, axpy_r)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(scalar_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(axpy_r, precondition)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(vec_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(precondition, dot_rz)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(2 * vec_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(dot_rz, axpy_p)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(scalar_bytes + vec_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);
  tg.connect(axpy_r, convergence)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .data_volume(vec_bytes)
      .placement(tapestry::Placement::LOCAL_SPM);

  return tg;
}

static bool emit_cg_tdg() {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();
  CGParams params;
  params.n = 16;
  auto module = tapestry::emitTDG(buildCGTaskGraph(params), ctx);
  if (!module) {
    std::puts("FAIL cg tdg emission");
    return false;
  }
  return true;
}

static std::vector<float> extract_diagonal(const CsrMatrixF32 &matrix) {
  std::vector<float> diag(static_cast<size_t>(matrix.rows), 0.0f);
  for (int row = 0; row < matrix.rows; ++row) {
    const int begin = matrix.row_ptr[static_cast<size_t>(row)];
    const int end = matrix.row_ptr[static_cast<size_t>(row + 1)];
    for (int idx = begin; idx < end; ++idx) {
      if (matrix.col_idx[static_cast<size_t>(idx)] == row) {
        diag[static_cast<size_t>(row)] =
            matrix.values[static_cast<size_t>(idx)];
        break;
      }
    }
  }
  return diag;
}

static std::vector<float> spmv_csr_ref(const CsrMatrixF32 &matrix,
                                       const std::vector<float> &x) {
  std::vector<float> y(static_cast<size_t>(matrix.rows), 0.0f);
  spmv_csr(matrix.row_ptr.data(), matrix.col_idx.data(), matrix.values.data(),
           x.data(), y.data(), matrix.rows);
  return y;
}

static CgReferenceResult run_cg_kernels(const CsrMatrixF32 &matrix,
                                        const std::vector<float> &b,
                                        const std::vector<float> &diag,
                                        int max_iters, float tol) {
  CgReferenceResult result;
  const int n = matrix.rows;
  result.x.assign(static_cast<size_t>(n), 0.0f);

  std::vector<float> r = b;
  std::vector<float> z(static_cast<size_t>(n), 0.0f);
  std::vector<float> p(static_cast<size_t>(n), 0.0f);
  std::vector<float> q(static_cast<size_t>(n), 0.0f);

  result.residual_history.push_back(conv_l2(r.data(), n));
  if (result.residual_history.back() < tol) {
    result.residual = result.residual_history.back();
    return result;
  }

  precond_diag(r.data(), diag.data(), z.data(), n);
  p = z;
  float rz = dot_sequential(r.data(), z.data(), n);

  for (int iter = 0; iter < max_iters; ++iter) {
    spmv_csr(matrix.row_ptr.data(), matrix.col_idx.data(), matrix.values.data(),
             p.data(), q.data(), n);
    const float pq = dot_sequential(p.data(), q.data(), n);
    if (std::fabs(pq) <= 1e-20f)
      break;

    const float alpha = rz / pq;
    axpy_basic(alpha, p.data(), result.x.data(), n);
    axpy_basic(-alpha, q.data(), r.data(), n);

    const float residual = conv_l2(r.data(), n);
    result.residual_history.push_back(residual);
    result.residual = residual;
    result.iterations = iter + 1;
    if (residual < tol)
      return result;

    precond_diag(r.data(), diag.data(), z.data(), n);
    const float rz_new = dot_sequential(r.data(), z.data(), n);
    const float beta = rz_new / rz;
    const std::vector<float> p_prev = p;
    p = z;
    axpy_basic(beta, p_prev.data(), p.data(), n);
    rz = rz_new;
  }

  return result;
}

static bool check_spmv_variants(const CsrMatrixF32 &matrix,
                                const EllMatrixF32 &ell) {
  const std::vector<float> x = make_random_vector(matrix.cols, 7, -0.5f, 0.5f);
  std::vector<float> y_csr(static_cast<size_t>(matrix.rows), 0.0f);
  std::vector<float> y_ell(static_cast<size_t>(matrix.rows), 0.0f);
  std::vector<float> y_unroll(static_cast<size_t>(matrix.rows), 0.0f);

  spmv_csr(matrix.row_ptr.data(), matrix.col_idx.data(), matrix.values.data(),
           x.data(), y_csr.data(), matrix.rows);
  spmv_ell(ell.col_idx.data(), ell.values.data(), x.data(), y_ell.data(),
           ell.rows, ell.max_nnz);
  spmv_csr_unroll2(matrix.row_ptr.data(), matrix.col_idx.data(),
                   matrix.values.data(), x.data(), y_unroll.data(),
                   matrix.rows);

  if (!vectors_close(y_csr, y_ell, 1e-6f, 1e-6f)) {
    std::puts("FAIL cg spmv csr vs ell");
    return false;
  }
  if (!vectors_close(y_csr, y_unroll, 1e-6f, 1e-6f)) {
    std::puts("FAIL cg spmv csr vs unroll2");
    return false;
  }
  return true;
}

static bool check_dot_variants() {
  const std::vector<float> a = make_random_vector(17, 13, -1.0f, 1.0f);
  const std::vector<float> b = make_random_vector(17, 17, -1.0f, 1.0f);
  const float seq =
      dot_sequential(a.data(), b.data(), static_cast<int>(a.size()));
  const float tree =
      dot_tree_reduce(a.data(), b.data(), static_cast<int>(a.size()));
  const float unroll =
      dot_sequential_unroll2(a.data(), b.data(), static_cast<int>(a.size()));
  if (!nearly_equal(seq, tree, 1e-5f, 1e-5f)) {
    std::printf("FAIL cg dot mismatch seq=%f tree=%f\n", seq, tree);
    return false;
  }
  if (!nearly_equal(seq, unroll, 1e-5f, 1e-5f)) {
    std::printf("FAIL cg dot mismatch seq=%f unroll=%f\n", seq, unroll);
    return false;
  }
  return true;
}

static bool check_conv_variants() {
  const std::vector<float> values = make_random_vector(9, 31, -2.0f, 2.0f);
  const float l2 = conv_l2(values.data(), static_cast<int>(values.size()));
  const float max_abs = conv_max(values.data(), static_cast<int>(values.size()));
  float manual_max = 0.0f;
  for (float value : values)
    manual_max = std::max(manual_max, std::fabs(value));
  if (!nearly_equal(max_abs, manual_max, 1e-6f, 1e-6f)) {
    std::printf("FAIL cg conv max mismatch %f %f\n", max_abs, manual_max);
    return false;
  }
  if (l2 < max_abs) {
    std::printf("FAIL cg conv ordering l2=%f max=%f\n", l2, max_abs);
    return false;
  }
  return true;
}

static bool run_tridiagonal_cg_case() {
  const int n = 16;
  const CsrMatrixF32 matrix = make_tridiagonal_laplacian_csr(n);
  const EllMatrixF32 ell = csr_to_ell(matrix);
  const std::vector<float> x_exact = [] {
    std::vector<float> values(16);
    for (int i = 0; i < 16; ++i)
      values[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    return values;
  }();
  const std::vector<float> b = spmv_csr_ref(matrix, x_exact);
  const std::vector<float> diag = extract_diagonal(matrix);

  CgReferenceResult ref = run_cg_reference(matrix, b, diag, 64, 1e-4f);
  CgReferenceResult run = run_cg_kernels(matrix, b, diag, 64, 1e-4f);

  if (!vectors_close(ref.x, x_exact, 1e-3f, 1e-3f)) {
    std::puts("FAIL cg reference solution mismatch");
    return false;
  }
  if (!vectors_close(run.x, x_exact, 1e-3f, 1e-3f)) {
    std::puts("FAIL cg kernel solution mismatch");
    return false;
  }
  if (!vectors_close(ref.x, run.x, 1e-4f, 1e-4f)) {
    std::puts("FAIL cg kernel vs reference mismatch");
    return false;
  }
  if (ref.residual_history.size() < 2 || run.residual_history.size() < 2) {
    std::puts("FAIL cg residual history too short");
    return false;
  }
  for (size_t i = 1; i < ref.residual_history.size(); ++i) {
    if (ref.residual_history[i] > ref.residual_history[i - 1] + 1e-5f) {
      std::puts("FAIL cg reference residual not monotonic");
      return false;
    }
  }
  for (size_t i = 1; i < run.residual_history.size(); ++i) {
    if (run.residual_history[i] > run.residual_history[i - 1] + 1e-5f) {
      std::puts("FAIL cg kernel residual not monotonic");
      return false;
    }
  }
  if (run.residual > 1e-4f) {
    std::printf("FAIL cg kernel residual too large %f\n", run.residual);
    return false;
  }
  (void)ell;
  return true;
}

static bool run_2d_laplacian_stress() {
  const int side = 10;
  const CsrMatrixF32 matrix = make_2d_laplacian_csr(side);
  const std::vector<float> x_true =
      make_random_vector(matrix.cols, 42, -0.2f, 0.8f);
  const std::vector<float> b = spmv_csr_ref(matrix, x_true);
  const std::vector<float> diag = extract_diagonal(matrix);
  const CgReferenceResult run = run_cg_kernels(matrix, b, diag, 200, 1e-4f);

  if (!vectors_close(run.x, x_true, 1e-3f, 1e-3f)) {
    std::puts("FAIL cg stress solution mismatch");
    return false;
  }

  const std::vector<float> ax = spmv_csr_ref(matrix, run.x);
  std::vector<float> residual(ax.size());
  for (size_t i = 0; i < ax.size(); ++i)
    residual[i] = ax[i] - b[i];
  const float rel =
      conv_l2(residual.data(), static_cast<int>(residual.size())) /
      std::max(conv_l2(b.data(), static_cast<int>(b.size())), 1e-6f);
  if (rel > 1e-4f) {
    std::printf("FAIL cg stress relative residual %f\n", rel);
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  (void)initLLVM;

  bool ok = emit_cg_tdg();
  ok = ok && check_spmv_variants(make_tridiagonal_laplacian_csr(5),
                                 csr_to_ell(make_tridiagonal_laplacian_csr(5)));
  ok = ok && check_dot_variants();
  ok = ok && check_conv_variants();
  ok = ok && run_tridiagonal_cg_case();
  ok = ok && run_2d_laplacian_stress();

  std::puts(ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
