#ifndef LOOM_TEST_FRONTEND_STRUCTUREDSCHEDULETESTSUPPORT_H
#define LOOM_TEST_FRONTEND_STRUCTUREDSCHEDULETESTSUPPORT_H

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Frontend/Compilation/StructuredSchedule.h"

#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <vector>

namespace loom::frontend::schedule_test {

[[noreturn]] void fail(const std::string &message);

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

StructuredProgramCandidate parseProgram(llvm::StringRef text);
StructuredEntityRef structuredLoopReference(
    const StructuredProgramCandidate &candidate, llvm::StringRef functionName);
std::vector<ArtifactRootReference> generated(
    const StructuredProgramCandidate &program,
    const fabric::FinalizedFabricRoot &fabric, const ArtifactStore &store,
    const BlobStore &blobs);

} // namespace loom::frontend::schedule_test

#endif // LOOM_TEST_FRONTEND_STRUCTUREDSCHEDULETESTSUPPORT_H
