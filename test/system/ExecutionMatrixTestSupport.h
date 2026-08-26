#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXTESTSUPPORT_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXTESTSUPPORT_H

#include "ExecutionMatrixInvocation.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace loom::system_test {

void runExecutionMatrixCell(ExecutionMatrixInvocation invocation,
                            llvm::StringRef gem5ReadinessPath);

void runPairedSpatialCgraBatch(std::uint64_t warmupRuns,
                               std::uint64_t measurementRuns);

void verifyDeterministicSystemReplay(llvm::StringRef gem5ReadinessPath);

void verifyHeterogeneousSystemAnchor();

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXTESTSUPPORT_H
