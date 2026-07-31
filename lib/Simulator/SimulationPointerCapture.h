#ifndef LOOM_LIB_SIMULATOR_SIMULATIONPOINTERCAPTURE_H
#define LOOM_LIB_SIMULATOR_SIMULATIONPOINTERCAPTURE_H

#include "SimulationWireInternal.h"

#include "Simulator/SimulationInputCapture.h"

#include "llvm/ADT/SmallVector.h"

namespace loom::sim::capture_detail {

llvm::Expected<mlir::Value>
threadMemorySourceForRoot(const dataflow::CanonicalLogicalMemoryRootView &root,
                          detail::ResolvedLaunchContext &context);

struct PointerValueTargetProjection final {
  SimulationPointerValueTargetCapture target;
  llvm::SmallVector<std::uint64_t, 2> equivalentMemoryRootOrdinals;
};

llvm::Expected<std::optional<PointerValueTargetProjection>>
pointerValueTargetForInput(
    const dataflow::CanonicalDataflowProgramView &program,
    detail::ResolvedLaunchContext &context, std::uint64_t valueInputOrdinal);

llvm::Error
attachPointerValueTargets(const dataflow::CanonicalDataflowProgramView &program,
                          detail::ResolvedLaunchContext &context,
                          SimulationInputCapturePlan &plan);

} // namespace loom::sim::capture_detail

#endif // LOOM_LIB_SIMULATOR_SIMULATIONPOINTERCAPTURE_H
