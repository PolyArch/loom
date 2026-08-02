#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSET_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSET_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

inline constexpr ArtifactSchemaDescriptor mappingConstraintSetSchema{
    "loom.mapping_constraints", SchemaVersion{1, 0}};

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSpatialConstraintAssembly(::mapping::ConstraintsSpatialOp root);

struct TechComputeRealizationRef final {
  std::uint64_t entity = 0;

  friend bool operator==(TechComputeRealizationRef lhs,
                         TechComputeRealizationRef rhs) {
    return lhs.entity == rhs.entity;
  }
};

struct TechMemoryRealizationRef final {
  std::uint64_t entity = 0;

  friend bool operator==(TechMemoryRealizationRef lhs,
                         TechMemoryRealizationRef rhs) {
    return lhs.entity == rhs.entity;
  }
};

struct SpatialConstraintTransferTerminal final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumer;

  friend bool operator==(const SpatialConstraintTransferTerminal &lhs,
                         const SpatialConstraintTransferTerminal &rhs) {
    return lhs.producer == rhs.producer && lhs.consumer == rhs.consumer;
  }
};

using SpatialConstraintSubject =
    std::variant<TechComputeRealizationRef, TechMemoryRealizationRef,
                 ::dataflow::CanonicalGraphProducerEndpointRef,
                 SpatialConstraintTransferTerminal, ::dataflow::ActorRef,
                 ::dataflow::LogicalMemoryRootRef>;

struct SpatialConstraintFuContext final {
  ::loom::fabric::FabricFuOccurrenceRef fu;
  ::loom::fabric::InstructionContextRef instructionContext;

  friend bool operator==(const SpatialConstraintFuContext &lhs,
                         const SpatialConstraintFuContext &rhs) {
    return lhs.fu == rhs.fu && lhs.instructionContext == rhs.instructionContext;
  }
};

struct SpatialConstraintUnsignedInterval final {
  llvm::APInt lower;
  llvm::APInt upper;

  friend bool operator==(const SpatialConstraintUnsignedInterval &lhs,
                         const SpatialConstraintUnsignedInterval &rhs) {
    return lhs.lower == rhs.lower && lhs.upper == rhs.upper;
  }
};

struct SpatialConstraintAddressRegion final {
  ::loom::fabric::FabricMemoryServiceRef service;
  std::vector<SpatialConstraintUnsignedInterval> intervals;

  friend bool operator==(const SpatialConstraintAddressRegion &lhs,
                         const SpatialConstraintAddressRegion &rhs) {
    return lhs.service == rhs.service && lhs.intervals == rhs.intervals;
  }
};

using SpatialConstraintDomainValue = std::variant<
    ::loom::fabric::FabricFuOccurrenceRef,
    ::loom::fabric::FabricPeOccurrenceRef,
    ::loom::fabric::InstructionContextRef, SpatialConstraintFuContext,
    ::loom::fabric::FabricMemoryOccurrenceRef,
    SpatialConstraintUnsignedInterval,
    ::loom::fabric::FabricPhysicalTraversalRef,
    ::loom::fabric::FabricResourceStateRef,
    ::loom::fabric::FabricTransportEndpointRef,
    ::loom::fabric::FabricMemoryOperationPortRef,
    ::loom::fabric::FabricMemoryServiceRef, SpatialConstraintAddressRegion>;

struct SpatialDomainRestrictionView final {
  ::mapping::SpatialConstraintProjection projection;
  SpatialConstraintSubject subject;
  std::vector<SpatialConstraintDomainValue> admissibleDomain;
};

struct SpatialEqualView final {
  ::mapping::SpatialConstraintProjection projection;
  std::vector<SpatialConstraintSubject> subjects;
};

struct SpatialDisjointView final {
  ::mapping::SpatialConstraintProjection projection;
  std::vector<SpatialConstraintSubject> subjects;
};

using SpatialConstraintClauseView =
    std::variant<SpatialDomainRestrictionView, SpatialEqualView,
                 SpatialDisjointView>;

class SpatialMappingConstraintSetView final {
public:
  static llvm::Expected<SpatialMappingConstraintSetView>
  import(const ArtifactIdentity &identity, ::mapping::ConstraintsSpatialOp root,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const TechMappingView &techMapping,
         const ::loom::fabric::FabricArtifactView &fabric);

  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &techMappingIdentity() const {
    return techMappingIdentity_;
  }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  llvm::ArrayRef<SpatialConstraintClauseView> clauses() const {
    return clauses_;
  }

private:
  SpatialMappingConstraintSetView(
      ArtifactIdentity identity, ArtifactIdentity dataflowIdentity,
      ArtifactIdentity techMappingIdentity, ArtifactIdentity fabricIdentity,
      std::vector<SpatialConstraintClauseView> clauses)
      : identity_(std::move(identity)),
        dataflowIdentity_(std::move(dataflowIdentity)),
        techMappingIdentity_(std::move(techMappingIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        clauses_(std::move(clauses)) {}

  ArtifactIdentity identity_;
  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity techMappingIdentity_;
  ArtifactIdentity fabricIdentity_;
  std::vector<SpatialConstraintClauseView> clauses_;
};

class FinalizedSpatialMappingConstraintSet final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const SpatialMappingConstraintSetView &view() const { return view_; }

private:
  FinalizedSpatialMappingConstraintSet(ArtifactRootReference reference,
                                       CanonicalSemanticBytes canonicalBytes,
                                       SpatialMappingConstraintSetView view)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), view_(std::move(view)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  SpatialMappingConstraintSetView view_;

  friend llvm::Expected<FinalizedSpatialMappingConstraintSet>
  finalizeSpatialMappingConstraintSet(::mapping::ConstraintsSpatialOp source,
                                      const ArtifactStore &store);
  friend llvm::Expected<FinalizedSpatialMappingConstraintSet>
  finalizeSpatialMappingConstraintSet(
      ::mapping::ConstraintsSpatialOp source,
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const TechMappingView &techMapping,
      const ::loom::fabric::FabricArtifactView &fabric,
      const ArtifactStore &store);
  friend llvm::Expected<FinalizedSpatialMappingConstraintSet>
  importSpatialMappingConstraintSet(const ArtifactRootReference &reference,
                                    const ArtifactStore &store);
};

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialMappingConstraintSet(::mapping::ConstraintsSpatialOp source,
                                    const ArtifactStore &store);

llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialMappingConstraintSet(
    ::mapping::ConstraintsSpatialOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store);

llvm::Expected<FinalizedSpatialMappingConstraintSet>
importSpatialMappingConstraintSet(const ArtifactRootReference &reference,
                                  const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSET_H
