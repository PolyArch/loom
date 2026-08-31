#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSET_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSET_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {

/// One family, two roots. 1.1 adds an optional Spatial-only clause kind, 1.2
/// appends the route-segment Physical Tag literal required for tag-local wait
/// causality, and 1.3 appends an exact parent SpatialMapping identity literal
/// for conservative complete-assignment counterexamples. These are compatible
/// semantic extensions: no existing carrier changed meaning or wire encoding.
/// Artifact identity nevertheless hashes the version, so references from
/// different minor versions are deliberately not interchangeable. Superseded
/// descriptors and explicit migration owners live in
/// `Mapping/Artifact/MappingConstraintSetMigration.h`.
inline constexpr ArtifactSchemaDescriptor mappingConstraintSetSchema{
    "loom.mapping_constraints", SchemaVersion{1, 3}};

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

/// One exact Spatial Mapping choice named by a no-good literal: the RouteTree
/// of `producer` selects `traversal`. An engaged `consumer` narrows the claim
/// to the branch from the route root to that exact sink.
struct SpatialNetUsesTraversalLiteral final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> consumer;
  ::loom::fabric::FabricPhysicalTraversalRef traversal;

  friend bool operator==(const SpatialNetUsesTraversalLiteral &lhs,
                         const SpatialNetUsesTraversalLiteral &rhs) {
    return lhs.producer == rhs.producer && lhs.consumer == rhs.consumer &&
           lhs.traversal == rhs.traversal;
  }
};

/// One exact Spatial Mapping choice named by a no-good literal: `terminal`
/// attaches at `endpoint`.
struct SpatialTransferAttachmentEqualsLiteral final {
  SpatialConstraintTransferTerminal terminal;
  ::loom::fabric::FabricTransportEndpointRef endpoint;

  friend bool operator==(const SpatialTransferAttachmentEqualsLiteral &lhs,
                         const SpatialTransferAttachmentEqualsLiteral &rhs) {
    return lhs.terminal == rhs.terminal && lhs.endpoint == rhs.endpoint;
  }
};

/// One exact Mapping-owned route Physical Tag segment value. Segment ordinals
/// are canonical within the RouteTree of `producer`; execution-plan tag ranks
/// and virtual-channel cache keys are never semantic identities.
struct SpatialNetTagEqualsLiteral final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::uint64_t segmentOrdinal = 0;
  llvm::APInt value = llvm::APInt(1, 0);

  friend bool operator==(const SpatialNetTagEqualsLiteral &lhs,
                         const SpatialNetTagEqualsLiteral &rhs) {
    return lhs.producer == rhs.producer &&
           lhs.segmentOrdinal == rhs.segmentOrdinal &&
           ::fabric::comparePhysicalTagValues(lhs.value, rhs.value) == 0;
  }
};

/// One exact sealed SpatialMapping selection. This is the conservative full
/// assignment literal for a replay-verified runtime counterexample: equality
/// to the referenced Mapping mechanically entails every persistent compute,
/// memory, route, local-disposition, ResourceUse, configured-hardware, and
/// handshake selection. Candidate-side expansion is a removable cache; the
/// referenced Mapping remains the sole semantic owner.
struct SpatialMappingIdentityEqualsLiteral final {
  ArtifactRootReference mapping;
  /// Removable strict-import cache of `mapping`. It never participates in
  /// equality or canonical encoding and is rebuilt from ArtifactStore bytes.
  std::shared_ptr<const FinalizedSpatialMapping> importedMapping;

  friend bool operator==(const SpatialMappingIdentityEqualsLiteral &lhs,
                         const SpatialMappingIdentityEqualsLiteral &rhs) {
    return lhs.mapping == rhs.mapping;
  }
};

enum class SpatialNoGoodLiteralKind : std::uint32_t {
  NetUsesTraversal = 0,
  TransferAttachmentEquals = 1,
  NetTagEquals = 2,
  SpatialMappingIdentityEquals = 3,
};

/// The closed Spatial no-good literal catalog. Only kinds a current production
/// admission consumer can independently verify against a sealed Mapping appear
/// here; no kind is pre-added for a future consumer.
using SpatialNoGoodLiteral =
    std::variant<SpatialNetUsesTraversalLiteral,
                 SpatialTransferAttachmentEqualsLiteral,
                 SpatialNetTagEqualsLiteral,
                 SpatialMappingIdentityEqualsLiteral>;

/// One disjunctive runtime-counterexample clause: the listed exact Mapping
/// choices may not all hold at once, so at least one literal must change. The
/// literal sequence is canonically sorted and duplicate-free, and never empty.
struct SpatialRuntimeCounterexampleNoGoodView final {
  std::vector<SpatialNoGoodLiteral> literals;
  struct Lineage final {
    ArtifactRootReference parentMapping;
    ArtifactRootReference runtimeEvidence;
    ArtifactRootReference evaluationRequest;
    ArtifactRootReference runtimeExecution;
    ComponentViewDigest certificateDigest;

    friend bool operator==(const Lineage &lhs, const Lineage &rhs) {
      return lhs.parentMapping == rhs.parentMapping &&
             lhs.runtimeEvidence == rhs.runtimeEvidence &&
             lhs.evaluationRequest == rhs.evaluationRequest &&
             lhs.runtimeExecution == rhs.runtimeExecution &&
             lhs.certificateDigest == rhs.certificateDigest;
    }
  };
  std::optional<Lineage> lineage;
};

using SpatialConstraintClauseView =
    std::variant<SpatialDomainRestrictionView, SpatialEqualView,
                 SpatialDisjointView, SpatialRuntimeCounterexampleNoGoodView>;

class SpatialMappingConstraintSetView final {
public:
  static llvm::Expected<SpatialMappingConstraintSetView>
  import(const ArtifactIdentity &identity, ::mapping::ConstraintsSpatialOp root,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const TechMappingView &techMapping,
         const ::loom::fabric::FabricArtifactView &fabric,
         const ArtifactStore &store);

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

/// Typed negative result of applying one exact Spatial MappingConstraintSet to
/// an independently base-verified SpatialMapping. Rejection is invocation
/// state, not intrinsic Mapping invalidity and not persistent artifact content.
class SpatialMappingConstraintRejection final
    : public llvm::ErrorInfo<SpatialMappingConstraintRejection> {
public:
  static char ID;

  using Owner = std::variant<::mapping::SpatialConstraintProjection,
                             SpatialNoGoodLiteralKind>;

  SpatialMappingConstraintRejection(
      ::mapping::SpatialConstraintProjection projection,
      std::uint64_t clauseOrdinal, std::string message)
      : owner_(projection), clauseOrdinal_(clauseOrdinal),
        message_(std::move(message)) {}
  SpatialMappingConstraintRejection(SpatialNoGoodLiteralKind literalKind,
                                    std::uint64_t clauseOrdinal,
                                    std::string message)
      : owner_(literalKind), clauseOrdinal_(clauseOrdinal),
        message_(std::move(message)) {}

  const Owner &owner() const { return owner_; }
  std::uint64_t clauseOrdinal() const { return clauseOrdinal_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Owner owner_;
  std::uint64_t clauseOrdinal_ = 0;
  std::string message_;
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

/// Materializes and finalizes the unique empty clause sequence for one exact
/// D/T/F closure. Absence of a constraint Artifact is never interpreted as an
/// empty set.
llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeEmptySpatialMappingConstraintSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store);

/// Publishes one exact domain restriction for a logical net's selected
/// physical traversals. The caller supplies the complete admissible domain;
/// finalization remains the sole owner of reference and projection legality.
llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialNetTraversalDomainConstraintSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef>
        admissibleTraversals,
    const ArtifactStore &store);

/// Publishes the canonical union of `parent`'s clauses with one additional
/// runtime-counterexample no-good over the same exact D/T/F closure. The clause
/// must be non-empty; its literals are canonically sorted and deduplicated, and
/// a clause equal to one already present is idempotent. Re-publishing the same
/// counterexample therefore yields the same Artifact identity.
llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizeSpatialRuntimeCounterexampleConstraintSet(
    const ArtifactRootReference &parent,
    llvm::ArrayRef<SpatialNoGoodLiteral> literals, const ArtifactStore &store);

/// Canonicalizes a replay-promoted runtime clause with its required durable
/// lineage. This is a codec/finalization boundary, not a proof constructor:
/// the DSE promotion owner must first consume VerifiedCgraClosedWaitEvidence
/// and establish the complete causal core. Explicitly authored no-goods use
/// `finalizeSpatialRuntimeCounterexampleConstraintSet` and carry no lineage.
llvm::Expected<FinalizedSpatialMappingConstraintSet>
finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
    const ArtifactRootReference &parent,
    llvm::ArrayRef<SpatialNoGoodLiteral> literals,
    const SpatialRuntimeCounterexampleNoGoodView::Lineage &lineage,
    const ArtifactStore &store);

llvm::Expected<FinalizedSpatialMappingConstraintSet>
importSpatialMappingConstraintSet(const ArtifactRootReference &reference,
                                  const ArtifactStore &store);

/// Independently evaluates one closed no-good literal against a sealed,
/// base-valid SpatialMapping. This is the shared truth owner for causal-core
/// promotion and constraint admission; it never consults CandidateState,
/// solver state, or the constraint set that supplied the literal.
llvm::Expected<bool> spatialMappingHoldsNoGoodLiteral(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialMappingView &spatialMapping,
    const SpatialNoGoodLiteral &literal);

/// Independently projects every closed Spatial constraint carrier from a
/// sealed, base-valid Mapping and checks the exact canonical K. The routine
/// builds only invocation-local read indexes and never consumes PnR caches,
/// CandidateState, solver assignments, or search history.
llvm::Error admitSpatialMappingConstraints(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const SpatialMappingConstraintSetView &constraints,
    const SpatialMappingView &spatialMapping);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGCONSTRAINTSET_H
