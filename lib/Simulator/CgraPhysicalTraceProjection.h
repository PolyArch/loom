#ifndef LOOM_LIB_SIMULATOR_CGRAPHYSICALTRACEPROJECTION_H
#define LOOM_LIB_SIMULATOR_CGRAPHYSICALTRACEPROJECTION_H

#include "Simulator/SpatialTrace.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::sim::detail {

struct CgraFrozenExecutionPlan;

struct CgraPhysicalTraceBinding final {
  PhysicalActionOccurrenceRef occurrence;
  PhysicalActionTarget target;
};

llvm::Expected<PhysicalActionTarget>
projectPhysicalUseTarget(const CgraFrozenExecutionPlan &plan,
                         std::uint64_t actionOrdinal);

llvm::Expected<PhysicalActionTarget> projectPhysicalTransferTarget(
    const CgraFrozenExecutionPlan &plan, std::uint64_t actionOrdinal,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef> traversals);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAPHYSICALTRACEPROJECTION_H
