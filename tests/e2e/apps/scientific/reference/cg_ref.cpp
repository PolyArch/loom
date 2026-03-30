#include "cg_ref.h"

#include <cmath>

namespace scicomp_test {

namespace {

static float dot_product(const std::vector<float> &lhs,
                         const std::vector<float> &rhs) {
  float sum = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i)
    sum += lhs[i] * rhs[i];
  return sum;
}

static float l2_norm(const std::vector<float> &values) {
  float sum = 0.0f;
  for (float value : values)
    sum += value * value;
  return std::sqrt(sum);
}

static void spmv_csr_ref(const CsrMatrixF32 &matrix,
                         const std::vector<float> &x,
                         std::vector<float> &y) {
  const int rows = matrix.rows;
  y.assign(static_cast<size_t>(rows), 0.0f);
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    const int begin = matrix.row_ptr[static_cast<size_t>(row)];
    const int end = matrix.row_ptr[static_cast<size_t>(row + 1)];
    for (int idx = begin; idx < end; ++idx)
      sum += matrix.values[static_cast<size_t>(idx)] *
             x[static_cast<size_t>(matrix.col_idx[static_cast<size_t>(idx)])];
    y[static_cast<size_t>(row)] = sum;
  }
}

static void precondition_diag_ref(const std::vector<float> &r,
                                  const std::vector<float> &diag,
                                  std::vector<float> &z) {
  z.resize(r.size());
  for (size_t i = 0; i < r.size(); ++i) {
    const float denom = i < diag.size() ? diag[i] : 1.0f;
    z[i] = denom != 0.0f ? r[i] / denom : r[i];
  }
}

static void axpy_ref(float alpha, const std::vector<float> &x,
                     std::vector<float> &y) {
  for (size_t i = 0; i < x.size(); ++i)
    y[i] += alpha * x[i];
}

} // namespace

CgReferenceResult run_cg_reference(const CsrMatrixF32 &matrix,
                                   const std::vector<float> &b,
                                   const std::vector<float> &diag,
                                   int max_iters, float tol) {
  CgReferenceResult result;
  const int n = matrix.rows;
  if (n <= 0 || static_cast<int>(b.size()) != n) {
    return result;
  }

  result.x.assign(static_cast<size_t>(n), 0.0f);
  std::vector<float> r = b;
  std::vector<float> z(n, 0.0f);
  std::vector<float> p(n, 0.0f);
  std::vector<float> q(n, 0.0f);

  const float initial_residual = l2_norm(r);
  result.residual_history.push_back(initial_residual);
  if (initial_residual < tol) {
    result.residual = initial_residual;
    return result;
  }

  precondition_diag_ref(r, diag, z);
  p = z;
  float rz = dot_product(r, z);

  for (int iter = 0; iter < max_iters; ++iter) {
    spmv_csr_ref(matrix, p, q);
    const float pq = dot_product(p, q);
    if (std::fabs(pq) <= 1e-20f)
      break;

    const float alpha = rz / pq;
    axpy_ref(alpha, p, result.x);
    axpy_ref(-alpha, q, r);

    const float residual = l2_norm(r);
    result.residual_history.push_back(residual);
    result.iterations = iter + 1;
    result.residual = residual;
    if (residual < tol)
      return result;

    precondition_diag_ref(r, diag, z);
    const float rz_new = dot_product(r, z);
    const float beta = rz_new / rz;
    const std::vector<float> p_prev = p;
    p = z;
    axpy_ref(beta, p_prev, p);
    rz = rz_new;
  }

  if (!result.residual_history.empty())
    result.residual = result.residual_history.back();
  return result;
}

} // namespace scicomp_test
