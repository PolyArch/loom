#pragma once

#include "Common/Artifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
class MLIRContext;
}

namespace loom {
class ArtifactStore;

namespace test {

struct PublishedStructuredSimulationInputs final {
  sim::CanonicalSimulationWorkload workload;
  sim::CanonicalSimulationRuntimeInput runtimeInput;
  ArtifactRootReference workloadReference;
  ArtifactRootReference runtimeInputReference;
};

frontend::StructuredProgramCandidate
buildWideVectorStructuredSource(mlir::MLIRContext &context);

frontend::StructuredEntityRef
findStructuredCallable(const frontend::StructuredProgramCandidate &candidate,
                       llvm::StringRef name);

PublishedStructuredSimulationInputs publishWideVectorStructuredInputs(
    const frontend::StructuredProgramCandidate &source, ArtifactStore &store);

} // namespace test
} // namespace loom
