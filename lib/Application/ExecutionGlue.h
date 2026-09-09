#ifndef LOOM_LIB_APPLICATION_EXECUTIONGLUE_H
#define LOOM_LIB_APPLICATION_EXECUTIONGLUE_H

#include "Application/Build.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Runtime/SpatialInvocationWire.h"
#include "Simulator/SimulationInputCapture.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class Module;
}

namespace loom::application::detail {

inline constexpr llvm::StringLiteral applicationHostEntrySymbol{
    "__loom_host_entry"};

struct ApplicationSpatialInvocationPlan final {
  std::shared_ptr<const dataflow::CanonicalDataflowArtifact> dataflow;

  struct MemoryObjectSource final {
    std::uint64_t dispatchArgumentOrdinal = 0;
    std::uint64_t byteOffset = 0;
    mlir::Value base;
  };

  struct MemoryRootSource final {
    std::uint64_t dispatchArgumentOrdinal = 0;
    std::uint64_t objectIndex = 0;
  };

  struct Site final {
    sim::OperationSimulationInputCapturePlan capture;
    std::vector<MemoryObjectSource> memoryObjectSources;
    std::vector<MemoryRootSource> memoryRootSources;
    runtime::SpatialInvocationWireLayout wireLayout;
  };

  struct Launch final {
    struct Point final {
      std::uint64_t dispatchTargetOrdinal = 0;
      std::vector<std::uint64_t> denseCoordinates;
    };

    dataflow::RootThreadLaunchRef root;
    dataflow::RootedGraphLaunchRef graph;
    std::vector<Point> points;
    std::vector<std::uint64_t> dispatchRootOperandOrdinals;
    std::vector<std::uint32_t> valueBitCounts;
    std::vector<std::uint32_t> resultBitCounts;
    std::vector<std::uint64_t> resultRootOperandOrdinals;
    std::vector<Site> sites;

    llvm::Expected<std::uint64_t>
    dispatchOperandOrdinal(std::uint64_t rootOperandOrdinal) const;
  };

  struct Callable final {
    std::string sourceCallableSymbol;
    std::vector<std::uint64_t> launchOrdinals;
  };

  std::vector<Launch> launches;
  std::vector<Callable> callables;
};

/// Retry a statically unsupported memory relation using exact source-backed
/// finite-object provenance. The plan retains the exact Dataflow owner so its
/// capture handles remain valid throughout Mapping and Deployment.
llvm::Expected<ApplicationSpatialInvocationPlan>
deriveApplicationSpatialInvocationPlan(
    const ArtifactRootReference &dataflow, llvm::StringRef entrySymbol,
    const ArtifactRootReference &selectedProgram,
    const ArtifactRootReference &sourceWorkload,
    const ArtifactRootReference &sourceRuntimeInput,
    const ArtifactStore &artifacts, std::uint64_t maxRetainedCaptureBytes,
    llvm::ArrayRef<dataflow::DataflowRewriteDerivation> derivations);

/// Preserves the source callable closure and shares the System entry ABI;
/// no source invocation is rewritten into accelerator dispatch.
llvm::Expected<std::unique_ptr<llvm::Module>> materializeHostOnlyModule(
    const llvm::Module &finalLinkedModule,
    const ApplicationSourceInvocation &sourceInvocation);

llvm::Expected<std::unique_ptr<llvm::Module>> materializeHostDispatchModule(
    const llvm::Module &finalLinkedModule,
    const ApplicationSourceInvocation &sourceInvocation,
    const ApplicationSpatialInvocationPlan &plan);

llvm::Expected<std::unique_ptr<llvm::Module>>
materializeInstructionDispatchModule(const llvm::Module &finalLinkedModule,
                                     std::uint64_t entryCount);

} // namespace loom::application::detail

#endif // LOOM_LIB_APPLICATION_EXECUTIONGLUE_H
