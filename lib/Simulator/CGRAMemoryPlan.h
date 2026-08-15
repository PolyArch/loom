#ifndef LOOM_LIB_SIMULATOR_CGRAMEMORYPLAN_H
#define LOOM_LIB_SIMULATOR_CGRAMEMORYPLAN_H

#include "CGRATransportPlan.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/IR/MemoryPortTransaction.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::sim::detail {

struct CgraMemoryChildTransactionPlan final {
  ::fabric::MemoryChildActivationKind activation =
      ::fabric::MemoryChildActivationKind::Always;
  std::optional<std::uint64_t> activationLane;
  ::fabric::MemoryChildProjectionKind projection =
      ::fabric::MemoryChildProjectionKind::ParentRequest;
  std::optional<std::uint64_t> projectionLane;
};

struct CgraMemoryResultAssemblyPlan final {
  ::dataflow::semantics::ServiceValueRole role =
      ::dataflow::semantics::ServiceValueRole::Data;
  ::fabric::MemoryResultAssemblyStrategy strategy =
      ::fabric::MemoryResultAssemblyStrategy::PassThroughParent;
  std::optional<std::uint64_t> laneCount;
  std::optional<::fabric::MemoryInactiveAssemblyValue> inactiveValue;
};

using CgraMemoryServiceTarget =
    std::variant<::loom::fabric::LocalMemoryServiceRef,
                 ::loom::fabric::MemoryConsistencyDomainRef,
                 ::loom::fabric::ManagerEndpointRef>;

struct CgraMemoryRootedUsePlan final {
  ::dataflow::RootedGraphLaunchRef launch;
  std::optional<std::uint64_t> bindingEntityId;
  CgraMemoryServiceTarget target;
  std::optional<std::uint64_t> localServicePhysicalUseOrdinal;
};

struct CgraMemoryActorPlan final {
  ::dataflow::ActorRef actor;
  ::dataflow::GraphRef graph;
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  ::loom::mapping::SpatialMemoryOperationPlacementView placement;
  ::loom::fabric::FabricMemoryCapabilityAlternativeRef capability;
  std::uint64_t operationPhysicalUseOrdinal = 0;
  std::uint64_t rootedUseOffset = 0;
  std::uint32_t rootedUseCount = 0;
  std::uint64_t childTransactionOffset = 0;
  std::uint32_t childTransactionCount = 0;
  std::uint64_t resultAssemblyOffset = 0;
  std::uint32_t resultAssemblyCount = 0;
  std::vector<std::optional<::loom::fabric::FabricMemoryHandshakeRoleSource>>
      roleSources;
  std::vector<
      std::optional<::loom::fabric::FabricMemoryHandshakeRoleDestination>>
      roleDestinations;
};

/// One exact occurrence-local engine connection selected by TechMapping.
/// Runtime transport consumes this relation directly; it is neither a routed
/// residual net nor a second Mapping choice.
struct CgraMemoryInternalConnectionPlan final {
  ::loom::fabric::FabricMemoryOccurrenceRef occurrence;
  ::loom::fabric::FabricOrdinal connection = 0;
  ::dataflow::ActorTokenResultRef producer;
  ::dataflow::ActorTokenOperandRef consumer;
};

struct CgraMemoryBindingPlan final {
  std::uint64_t entityId = 0;
  ::dataflow::LogicalMemoryRootOrViewRef logicalMemory;
  ::loom::mapping::SpatialMemoryIntervalView interval;
  ::loom::mapping::SpatialMemoryBindingTargetView target;
};

struct CgraMemoryPlan final {
  std::vector<CgraMemoryActorPlan> actors;
  std::vector<CgraMemoryRootedUsePlan> rootedUses;
  std::vector<CgraMemoryChildTransactionPlan> childTransactions;
  std::vector<CgraMemoryResultAssemblyPlan> resultAssemblies;
  std::vector<CgraMemoryBindingPlan> bindings;
  std::vector<CgraMemoryInternalConnectionPlan> internalConnections;
};

llvm::Expected<CgraMemoryPlan> freezeCgraMemoryPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients);

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAMEMORYPLAN_H
