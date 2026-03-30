#ifndef TAPESTRY_SCICOMP_NBODY_REF_H
#define TAPESTRY_SCICOMP_NBODY_REF_H

#include "test_scicomp_utils.h"

#include <vector>

namespace scicomp_test {

struct NBodyReferenceResult {
  NBodyState state;
  std::vector<float> energy_history;
};

float compute_nbody_energy(const NBodyState &state, float g, float softening);

NBodyReferenceResult run_nbody_reference(const NBodyState &initial, int steps,
                                         float dt, float g, float softening);

} // namespace scicomp_test

#endif // TAPESTRY_SCICOMP_NBODY_REF_H
