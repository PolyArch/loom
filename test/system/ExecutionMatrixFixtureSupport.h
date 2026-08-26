#ifndef LOOM_TEST_SYSTEM_EXECUTIONMATRIXFIXTURESUPPORT_H
#define LOOM_TEST_SYSTEM_EXECUTIONMATRIXFIXTURESUPPORT_H

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "DeploymentTestSupport.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <utility>

namespace loom::system_test {

std::unique_ptr<mlir::MLIRContext> makeContext();

dataflow::CanonicalDataflowArtifact buildCanonicalApplication(
    llvm::StringRef test, mlir::MLIRContext &context, bool paired);

std::pair<ArtifactRootReference, ArtifactRootReference> publishSpatialInputs(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    ArtifactStore &artifacts, bool paired);

} // namespace loom::system_test

#endif // LOOM_TEST_SYSTEM_EXECUTIONMATRIXFIXTURESUPPORT_H
