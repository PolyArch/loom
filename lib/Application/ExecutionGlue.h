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
    std::vector<runtime::SpatialInvocationWireLayout> pointWireLayouts;
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
  };

  struct Callable final {
    std::string sourceCallableSymbol;
    std::vector<std::uint64_t> launchOrdinals;
  };

  std::vector<Launch> launches;
  std::vector<Callable> callables;
};

llvm::Expected<ApplicationSpatialInvocationPlan>
deriveApplicationSpatialInvocationPlan(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::StringRef entrySymbol);

llvm::Expected<std::unique_ptr<llvm::Module>> materializeHostDispatchModule(
    const llvm::Module &finalLinkedModule,
    const dataflow::CanonicalDataflowArtifact &dataflow,
    const ApplicationSourceInvocation &sourceInvocation,
    const ApplicationSpatialInvocationPlan &plan);

llvm::Expected<std::unique_ptr<llvm::Module>>
materializeInstructionDispatchModule(const llvm::Module &finalLinkedModule,
                                     std::uint64_t entryCount);

} // namespace loom::application::detail

#endif // LOOM_LIB_APPLICATION_EXECUTIONGLUE_H
