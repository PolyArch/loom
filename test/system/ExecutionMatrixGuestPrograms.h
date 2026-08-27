#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXGUESTPROGRAMS_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXGUESTPROGRAMS_H

#include "DeploymentTestSupport.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::system_test {

inline constexpr std::uint64_t pairedMeasurementInvocationCount = 512;
static_assert(pairedMeasurementInvocationCount > 0);

std::vector<std::uint8_t> compileGuestProgram(
    llvm::StringRef test, const deployment::test::TemporaryTree &tree,
    llvm::StringRef stem, llvm::StringRef source, std::uint64_t loadAddress,
    llvm::StringRef entrySymbol, bool includeM5Ops);

llvm::StringRef orderedChannelHostProgramSource();
llvm::StringRef singleInvocationHostProgramSource();
std::string pairedInvocationHostProgramSource();
llvm::StringRef spatialInstructionProgramSource();

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXGUESTPROGRAMS_H
