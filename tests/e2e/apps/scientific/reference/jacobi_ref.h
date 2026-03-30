#ifndef TAPESTRY_SCICOMP_JACOBI_REF_H
#define TAPESTRY_SCICOMP_JACOBI_REF_H

#include <vector>

namespace scicomp_test {

struct JacobiReferenceResult {
  std::vector<float> grid;
  float residual = 0.0f;
  int iterations = 0;
};

JacobiReferenceResult run_jacobi_reference(const std::vector<float> &initial,
                                           int rows, int cols, int halo_w,
                                           float factor, int max_iters,
                                           float stop_eps);

} // namespace scicomp_test

#endif // TAPESTRY_SCICOMP_JACOBI_REF_H
