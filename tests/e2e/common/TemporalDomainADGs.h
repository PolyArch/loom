//===-- TemporalDomainADGs.h - Temporal e2e ADG builders -------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#ifndef FCC_TESTS_E2E_COMMON_TEMPORALDOMAINADGS_H
#define FCC_TESTS_E2E_COMMON_TEMPORALDOMAINADGS_H

#include <string>

namespace fcc {
namespace e2e {

struct TemporalReductionDomainOptions {
  std::string moduleName;
  unsigned dataWidth = 64;
  std::string memrefType = "memref<?xi32>";
  unsigned numRegister = 0;
  unsigned numInstruction = 16;
  unsigned regFifoDepth = 0;
};

void buildTemporalReductionDomain(const std::string &outputPath,
                                  const TemporalReductionDomainOptions &opts);

} // namespace e2e
} // namespace fcc

#endif // FCC_TESTS_E2E_COMMON_TEMPORALDOMAINADGS_H
