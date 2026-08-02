#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

inline constexpr ArtifactSchemaDescriptor mappingArtifactSchema{
    "loom.mapping", SchemaVersion{2, 0}};

/// Canonicalizes one complete in-memory Mapping root for final verification.
/// This syntax layer normalizes schema-owned record order and Mapping-local
/// IDs. Exact upstream import and profile completeness are enforced by the
/// finalizer that publishes a MappingArtifact.
llvm::Expected<CanonicalSemanticBytes>
writeCanonicalMappingAssembly(::mapping::TechOp root);

struct TechComputeActorView final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricFuTemplateNodeRef fabricOperation;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
};

struct TechComputeBoundaryView final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricPortDirection direction;
  std::uint64_t portOrdinal;
  ::loom::fabric::FabricFuTemplatePortRef fabricPort;
};

struct TechComputeRealizationView final {
  std::uint64_t entityId;
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  std::vector<TechComputeActorView> actors;
  std::vector<TechComputeBoundaryView> boundaries;
};

using TechMemoryGraphEndpointRef =
    std::variant<::dataflow::CanonicalGraphProducerEndpointRef,
                 ::dataflow::CanonicalGraphConsumerEndpointRef>;

struct TechMemoryActorView final {
  ::dataflow::ActorRef actor;
  ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef operationPort;
  ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef capability;
  std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
      operandPorts;
  std::vector<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
      resultPorts;
};

struct TechMemoryGraphBoundaryView final {
  TechMemoryGraphEndpointRef terminal;
  ::loom::fabric::FabricMemoryEngineTemplateEndpointRef endpoint;
};

struct TechMemoryInternalEdgeView final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::dataflow::CanonicalGraphConsumerEndpointRef consumer;
  ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef connection;
};

struct TechMemoryRealizationView final {
  std::uint64_t entityId;
  ::loom::fabric::FabricMemoryEngineTemplateRef engine;
  std::vector<TechMemoryActorView> actors;
  std::vector<TechMemoryGraphBoundaryView> graphBoundaries;
  std::vector<TechMemoryInternalEdgeView> internalEdges;
};

/// One exact residual graph-local transfer obligation derived from D/T after
/// realization-internal sinks have been removed. The producer remains the
/// persistent SpatialLogicalNetKey; this view is a removable import cache and
/// introduces no Mapping-local identity.
struct TechResidualLogicalNetView final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::vector<::dataflow::CanonicalGraphConsumerEndpointRef> sinks;
};

/// Verifies the exact realization-wide topology and correspondence relation
/// after each actor's typed operation capability has been resolved. Generator
/// and strict importer share these owners so candidate pruning cannot diverge
/// from persistent-artifact admission.
llvm::Error verifyTechComputeRealizationClosure(
    const TechComputeRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric);
llvm::Error verifyTechMemoryRealizationClosure(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric);

/// Canonical prospective persistent-payload keys used by Mapping-local
/// identity assignment and TechMapping seed enumeration. These encoders are
/// the sole owner of row ordering; callers must not duplicate their formula.
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchActorKey(const TechComputeActorView &actor,
                           const ArtifactIdentity &dataflowOwner);
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchActorKey(const TechMemoryActorView &actor,
                           const ArtifactIdentity &dataflowOwner);
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchRowKey(const TechComputeRealizationView &realization,
                         const ArtifactIdentity &dataflowOwner);
llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMatchRowKey(const TechMemoryRealizationView &realization,
                         const ArtifactIdentity &dataflowOwner);

/// Immutable read-only projection of one independently verified mapping.tech
/// object. Every member is a typed reference into the exact bound Dataflow or
/// Fabric artifact; copied semantic descriptions and authoring handles are
/// deliberately absent.
class TechMappingView final {
public:
  static llvm::Expected<TechMappingView>
  import(const ArtifactIdentity &mappingIdentity, ::mapping::TechOp root,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const ::loom::fabric::FabricArtifactView &fabric);

  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  llvm::ArrayRef<::dataflow::GraphRef> covers() const { return covers_; }
  llvm::ArrayRef<TechComputeRealizationView> computeRealizations() const {
    return computeRealizations_;
  }
  llvm::ArrayRef<TechMemoryRealizationView> memoryRealizations() const {
    return memoryRealizations_;
  }
  llvm::ArrayRef<TechResidualLogicalNetView> residualLogicalNets() const {
    return residualLogicalNets_;
  }
  const TechResidualLogicalNetView *residualLogicalNet(
      const ::dataflow::CanonicalGraphProducerEndpointRef &producer) const;

private:
  TechMappingView(ArtifactIdentity identity, ArtifactIdentity dataflowIdentity,
                  ArtifactIdentity fabricIdentity,
                  std::vector<::dataflow::GraphRef> covers,
                  std::vector<TechComputeRealizationView> computeRealizations,
                  std::vector<TechMemoryRealizationView> memoryRealizations,
                  std::vector<TechResidualLogicalNetView> residualLogicalNets)
      : identity_(std::move(identity)),
        dataflowIdentity_(std::move(dataflowIdentity)),
        fabricIdentity_(std::move(fabricIdentity)), covers_(std::move(covers)),
        computeRealizations_(std::move(computeRealizations)),
        memoryRealizations_(std::move(memoryRealizations)),
        residualLogicalNets_(std::move(residualLogicalNets)) {}

  ArtifactIdentity identity_;
  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity fabricIdentity_;
  std::vector<::dataflow::GraphRef> covers_;
  std::vector<TechComputeRealizationView> computeRealizations_;
  std::vector<TechMemoryRealizationView> memoryRealizations_;
  std::vector<TechResidualLogicalNetView> residualLogicalNets_;
};

/// The immutable result of failure-atomic publication or strict import of one
/// exact mapping.tech 2.0 object.
class FinalizedTechMapping final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const TechMappingView &view() const { return view_; }

private:
  FinalizedTechMapping(ArtifactRootReference reference,
                       CanonicalSemanticBytes canonicalBytes,
                       TechMappingView view)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), view_(std::move(view)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  TechMappingView view_;

  friend llvm::Expected<FinalizedTechMapping>
  finalizeTechMapping(::mapping::TechOp source, const ArtifactStore &store);
  friend llvm::Expected<FinalizedTechMapping>
  finalizeTechMapping(::mapping::TechOp source,
                      const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const ArtifactStore &store);
  friend llvm::Expected<FinalizedTechMapping>
  importTechMapping(const ArtifactRootReference &reference,
                    const ArtifactStore &store);
};

llvm::Expected<FinalizedTechMapping>
finalizeTechMapping(::mapping::TechOp source, const ArtifactStore &store);

/// Finalizes against exact upstream views already sealed by their family
/// finalizers or strict importers. The corresponding objects must still be
/// durably present in `store`; this overload only avoids reparsing the same
/// immutable upstream artifacts for every candidate in one invocation.
llvm::Expected<FinalizedTechMapping>
finalizeTechMapping(::mapping::TechOp source,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const ::loom::fabric::FabricArtifactView &fabric,
                    const ArtifactStore &store);

llvm::Expected<FinalizedTechMapping>
importTechMapping(const ArtifactRootReference &reference,
                  const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGARTIFACT_H
