#ifndef LOOM_TEST_FRONTEND_STRUCTUREDOWNERSHIPEVALUATIONTESTSUPPORT_H
#define LOOM_TEST_FRONTEND_STRUCTUREDOWNERSHIPEVALUATIONTESTSUPPORT_H

#include "Simulator/NativeSimulationOracle.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>
#include <utility>

namespace loom::dse {
struct CompletedPreMappingSelection;
}

namespace loom::test::structured_ownership {

[[noreturn]] void fail(const std::string &message);

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

struct SourceSimulationInputs final {
  loom::sim::CanonicalSimulationWorkload workload;
  loom::sim::CanonicalSimulationRuntimeInput runtimeInput;
  loom::ArtifactRootReference workloadReference;
  loom::ArtifactRootReference runtimeInputReference;
  loom::sim::NativeStructuredProgramObservations observations;
};

bool hasGeneratorInvocation(
    const loom::dse::CompletedPreMappingSelection &selection,
    llvm::StringRef spelling);

loom::frontend::StructuredEntityRef
findCallable(const loom::frontend::StructuredProgramCandidate &candidate,
             llvm::StringRef name);

loom::sim::RuntimeMemoryObject zeroedMemory(std::size_t byteCount);

void centralPlanEvaluatesScheduleChildren();

void exactUniformCallArgumentsAreCandidateLocal();

void ownershipLineageRejectsAnOutOfRangeScope();

} // namespace loom::test::structured_ownership

#endif // LOOM_TEST_FRONTEND_STRUCTUREDOWNERSHIPEVALUATIONTESTSUPPORT_H
