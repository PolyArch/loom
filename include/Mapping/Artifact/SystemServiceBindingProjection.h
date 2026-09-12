#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMSERVICEBINDINGPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMSERVICEBINDINGPROJECTION_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricMemoryServiceClosure.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::mapping {

struct SystemBoundMemoryEndpointPairView final {
  ::loom::fabric::SystemServiceEndpointRef systemEndpoint;
  ::loom::fabric::FabricMemoryEndpointRef occurrenceEndpoint;
};

struct SystemSpatialMemoryBindingProjection final {
  std::vector<SystemBoundMemoryEndpointPairView> endpointPairs;
  std::optional<SpatialMemoryIntervalView> interval;
  std::optional<::loom::fabric::SubordinateEndpointRef> exposureTerminal;
};

struct SystemMemoryUsePatternDomainView final {
  ::loom::fabric::FabricMemoryServiceRegionRef region;
  std::vector<::loom::fabric::FabricUsePatternRef> patterns;
};

struct SystemOperationCompletionProjection final {
  bool admitted = false;
  std::optional<std::uint64_t> maxIssueToRetireCycles;
};

using SystemMessageExecutionOwner =
    std::variant<::loom::fabric::HostCoreOccurrenceRef,
                 ::loom::fabric::AccCoreOccurrenceRef>;

struct SystemMessageTerminalEndpointDomainView final {
  ::loom::fabric::FabricTransportEndpointRef endpoint;
  bool payloadCompatible = false;
};

/// Derives every mechanically bindable transport endpoint for one exact
/// message terminal and execution owner. A role-compatible endpoint whose
/// payload domain excludes the message remains present with an empty domain;
/// callers may apply constraints but cannot rebuild Fabric capability rules.
llvm::Expected<std::vector<SystemMessageTerminalEndpointDomainView>>
projectSystemMessageTerminalEndpointDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemTransferTerminalKey &terminal, mlir::Type payload,
    const SystemMessageExecutionOwner &owner);

/// Rebuilds the exact Module-manager to occurrence-qualified System attachment
/// relation selected by one immutable SpatialMapping. An exact AccCore narrows
/// the projection to one execution occurrence; omission returns the complete
/// finite relation used while constructing a search domain.
llvm::Expected<SystemSpatialMemoryBindingProjection>
projectSystemSpatialMemoryBinding(
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SpatialMappingView &mapping, std::uint64_t moduleDependencyOrdinal,
    const ServicePlanSelectionAnchor &anchor,
    std::optional<::loom::fabric::AccCoreOccurrenceRef> exactAccCore =
        std::nullopt);

/// Derives the operation service kind from the canonical actor schema.
/// MessageTransfer and MemoryExposure anchors are outside this projection.
llvm::Expected<::dataflow::semantics::ServiceKind>
resolveSystemOperationServiceKind(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::dataflow::ServiceMemberRef &member);

/// Derives the endpoint-compatible target domain for one operation member.
/// An incompatible endpoint yields an empty domain rather than selecting an
/// alternative endpoint.
llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
projectSystemOperationTargetRegions(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    const ::dataflow::ServiceMemberRef &member);

llvm::Expected<std::vector<::loom::fabric::MemoryConsistencyDomainRef>>
projectSystemFenceTargetDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    const ::dataflow::ServiceMemberRef &member);

/// Projects whether one endpoint admits the operation and, when admitted,
/// converts its progress guarantee into cycles of the exact issuing
/// SpatialCore occurrence. Fair-eventual service has no numeric bound.
llvm::Expected<SystemOperationCompletionProjection>
projectSystemOperationCompletion(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    ::loom::fabric::AccCoreOccurrenceRef accCore,
    const ::dataflow::ServiceMemberRef &member);

/// Derives the complete Fabric target-plan domain for one logical-memory
/// interval. A finite interval is checked exactly. A dynamically unbounded
/// whole interval selects a structural service envelope whose concrete range
/// remains subject to Runtime ABI admission.
llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceTargetPlan>>
projectSystemMemoryTargetPlans(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    ::dataflow::LogicalMemoryRootOrViewRef logicalMemory,
    const SpatialMemoryIntervalView &interval);

llvm::Expected<std::vector<SystemMemoryUsePatternDomainView>>
projectSystemMemoryUsePatternDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::dataflow::ServiceMemberRef &member,
    llvm::ArrayRef<::loom::fabric::FabricMemoryServiceTargetPlan> plans);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMSERVICEBINDINGPROJECTION_H
