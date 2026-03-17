//===-- DomainADGs.h - Reusable e2e ADG builders ---------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#ifndef FCC_TESTS_E2E_COMMON_DOMAINADGS_H
#define FCC_TESTS_E2E_COMMON_DOMAINADGS_H

#include <string>

namespace fcc {
namespace e2e {

struct SpatialVectorDomainOptions {
  std::string moduleName;
  unsigned numPEs = 24;
  unsigned numExtMems = 4;
  unsigned dataWidth = 64;
  unsigned numScalarInputs = 4;
  unsigned numScalarOutputs = 2;
};

void buildSpatialVectorDomain(const std::string &outputPath,
                              const SpatialVectorDomainOptions &opts);

} // namespace e2e
} // namespace fcc

#endif // FCC_TESTS_E2E_COMMON_DOMAINADGS_H
