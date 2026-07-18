#ifndef LOOM_TEST_MAPPING_MAPPINGCORETESTSUPPORT_H
#define LOOM_TEST_MAPPING_MAPPINGCORETESTSUPPORT_H

#include "Mapping/Artifact.h"
#include "Mapping/Verifier.h"
#include "PnR/FrozenRealizationGraph.h"
#include "PnR/PnrIndex.h"
#include "PnR/PnrProblemInputs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>

namespace loom::mapping::test {

using namespace loom::mapping;
using namespace loom::pnr;

struct TestCase {
  DataflowProgramView dataflow;
  FabricHardwareView fabric;
  TechMappingDraft mapping;
};

ArtifactIdentity artifact(std::uint8_t value);
TypeKey type(std::uint64_t value);
PortRoleKey role(std::uint64_t value);
SemanticKey semantic(std::uint8_t value);
PortDescriptor port(PortKind kind, TypeKey typeKey,
                    std::uint32_t payloadWidthBits = 0,
                    std::uint32_t tagWidthBits = 0,
                    PortRoleKey roleKey = role(0));
ComputeOccurrenceDescriptor makeSpatialComputeOccurrence(
    const ArtifactIdentity &fabric, ComputeOccurrenceId occurrence,
    const FuDescriptor &fu, std::uint64_t endpointBase);

[[noreturn]] void fail(const char *test, const char *message);
MappingErrorCode takeCode(llvm::Error error);

template <typename T>
void expectError(const char *test, llvm::Expected<T> result,
                 MappingErrorCode expected) {
  if (result)
    fail(test, "expected validation failure");
  if (takeCode(result.takeError()) != expected)
    fail(test, "received a different validation failure");
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()).c_str());
  return std::move(*value);
}

template <typename T>
void expectAnyError(const char *test, llvm::Expected<T> result) {
  if (result)
    fail(test, "expected failure");
  llvm::consumeError(result.takeError());
}

ValidatedTechMapping validateCase(const char *test, const TestCase &testCase);
PnrProblemInputs makePnrProblemInputs(const TestCase &testCase,
                                      const ValidatedTechMapping &mapping,
                                      const ResolvedPnrConfigView &config);
FrozenRealizationGraph validateAndFreeze(const char *test, TestCase &testCase);
void expectMapError(const char *test, const TestCase &testCase,
                    MappingErrorCode expected);

TestCase makeValidCase();
TestCase makeWideSyncCase();
void selectWideSyncLanes(TestCase &testCase,
                         llvm::ArrayRef<std::uint32_t> laneIndices);
TestCase makeMemoryAnchorCase();
void selectInternalMemoryGraph(TestCase &testCase);

void runComputeFreezeTests();
void runMemoryMappingTests();
void runMappingVerifierTests();
void runPnrProblemInputsTests();

} // namespace loom::mapping::test

#endif // LOOM_TEST_MAPPING_MAPPINGCORETESTSUPPORT_H
