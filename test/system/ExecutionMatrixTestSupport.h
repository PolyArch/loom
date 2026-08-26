#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXTESTSUPPORT_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXTESTSUPPORT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::system_test {

enum class ExecutionMatrixCell : std::uint8_t {
  SpatialDfg,
  SpatialCgra,
  SpatialRtl,
  SystemDfg,
  SystemCgra,
  SystemRtl,
  PairedSpatialCgra,
  PairedSystemCgra,
};

llvm::Expected<ExecutionMatrixCell>
parseExecutionMatrixCell(llvm::StringRef spelling);

void runExecutionMatrixCell(ExecutionMatrixCell cell,
                            llvm::StringRef gem5ReadinessPath);

void runPairedSpatialCgraBatch(std::uint64_t warmupRuns,
                               std::uint64_t measurementRuns);

void verifyDeterministicSystemReplay(llvm::StringRef gem5ReadinessPath);

void verifyHeterogeneousSystemAnchor();

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXTESTSUPPORT_H
