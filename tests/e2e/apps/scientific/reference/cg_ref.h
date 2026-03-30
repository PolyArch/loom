#ifndef TAPESTRY_SCICOMP_CG_REF_H
#define TAPESTRY_SCICOMP_CG_REF_H

#include "test_scicomp_utils.h"

#include <vector>

namespace scicomp_test {

struct CgReferenceResult {
  std::vector<float> x;
  float residual = 0.0f;
  int iterations = 0;
  std::vector<float> residual_history;
};

CgReferenceResult run_cg_reference(const CsrMatrixF32 &matrix,
                                   const std::vector<float> &b,
                                   const std::vector<float> &diag,
                                   int max_iters, float tol);

} // namespace scicomp_test

#endif // TAPESTRY_SCICOMP_CG_REF_H
