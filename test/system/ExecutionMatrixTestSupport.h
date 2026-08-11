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
};

llvm::Expected<ExecutionMatrixCell>
parseExecutionMatrixCell(llvm::StringRef spelling);

void runExecutionMatrixCell(ExecutionMatrixCell cell,
                            llvm::StringRef gem5ReadinessPath);

void verifyDeterministicSystemReplay(llvm::StringRef gem5ReadinessPath);

void verifyHeterogeneousSystemAnchor();

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXTESTSUPPORT_H
