#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/StructuredScop.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::frontend {

enum class StructuredScheduleDecisionKind : std::uint32_t {
  Tile = 0,
  Unroll = 1,
  Interchange = 2,
  UnrollAndJam = 3,
  Parallelize = 4,
  ParallelizeNest = 5,
  Vectorize = 6,
  PolyhedralSchedule = 7,
};

enum class StructuredVectorTailPolicy : std::uint32_t {
  Exact = 0,
  ReductionMask = 1,
};

enum class StructuredVectorAliasPolicy : std::uint32_t {
  ProviderProvenNoAlias = 0,
};

inline constexpr std::uint64_t maximumCanonicalStructuredScheduleFactor = 64;

struct StructuredVectorScheduleCoordinate final {
  std::vector<std::uint64_t> shape;
  StructuredVectorTailPolicy tailPolicy = StructuredVectorTailPolicy::Exact;
  std::uint64_t requiredAlignmentBytes = 0;
  StructuredVectorAliasPolicy aliasPolicy =
      StructuredVectorAliasPolicy::ProviderProvenNoAlias;
  StructuredReductionSchedule reductionSchedule =
      StructuredReductionSchedule::None;

  friend bool operator==(const StructuredVectorScheduleCoordinate &lhs,
                         const StructuredVectorScheduleCoordinate &rhs) {
    return lhs.shape == rhs.shape && lhs.tailPolicy == rhs.tailPolicy &&
           lhs.requiredAlignmentBytes == rhs.requiredAlignmentBytes &&
           lhs.aliasPolicy == rhs.aliasPolicy &&
           lhs.reductionSchedule == rhs.reductionSchedule;
  }
};

/// One atomic schedule decision over an exact parent-local loop. A zero factor
/// is canonical for decisions without a factor; replication decisions carry a
/// factor greater than one.
struct StructuredScheduleDecision final {
  StructuredEntityRef loop;
  StructuredScheduleDecisionKind kind;
  std::uint64_t factor = 0;
  std::optional<StructuredVectorScheduleCoordinate> vector;

  friend bool operator==(const StructuredScheduleDecision &lhs,
                         const StructuredScheduleDecision &rhs) {
    return lhs.loop == rhs.loop && lhs.kind == rhs.kind &&
           lhs.factor == rhs.factor && lhs.vector == rhs.vector;
  }
};

struct MaterializedStructuredScheduleCandidate final {
  StructuredProgramCandidate structuredProgram;
  std::optional<StructuredEntityRef> trackedSpatialRegion;
  /// Invocation-local roots introduced by a reconstructed polyhedral
  /// schedule. They let the finite schedule search compose a second atomic
  /// decision with exactly the transformed loop rather than an unrelated
  /// sibling. The decision lineage remains the persistent owner.
  std::vector<StructuredEntityRef> transformedScheduleRoots;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

/// An expected local refusal after a proposal reaches its provider or exact
/// Fabric gate. Malformed decisions and implementation failures remain errors.
class StructuredScheduleProposalRefusal final
    : public llvm::ErrorInfo<StructuredScheduleProposalRefusal> {
public:
  static char ID;

  StructuredScheduleProposalRefusal(StructuredEntityRef loop,
                                    StructuredScopRefusalKind kind)
      : loop_(std::move(loop)), kind_(kind) {}

  const StructuredEntityRef &loop() const { return loop_; }
  StructuredScopRefusalKind kind() const { return kind_; }
  std::string message() const override;
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  StructuredEntityRef loop_;
  StructuredScopRefusalKind kind_;
};

struct StructuredScheduleDecisionDomain;

/// One canonical atomic decision bound to the exact Fabric and any removable
/// exact-SCoP view used to construct it. Construction remains owned by the
/// enumerator, so a selected materializer can reuse the frozen provider proof
/// without accepting caller-authored dependence or capacity summaries.
/// Provider/Fabric refusal remains local.
class StructuredScheduleProposal final {
public:
  StructuredScheduleProposal(const StructuredScheduleProposal &) = default;
  StructuredScheduleProposal(StructuredScheduleProposal &&) = default;
  StructuredScheduleProposal &
  operator=(const StructuredScheduleProposal &) = default;
  StructuredScheduleProposal &
  operator=(StructuredScheduleProposal &&) = default;

  const StructuredScheduleDecision &decision() const { return decision_; }

private:
  StructuredScheduleProposal(
      StructuredScheduleDecision decision,
      std::shared_ptr<const ExactStructuredScopView> exactScop,
      std::shared_ptr<const StructuredPolyhedralScopView> polyhedralScop,
      ArtifactRootReference fabric)
      : decision_(std::move(decision)), exactScop_(std::move(exactScop)),
        polyhedralScop_(std::move(polyhedralScop)), fabric_(std::move(fabric)) {
  }

  StructuredScheduleDecision decision_;
  std::shared_ptr<const ExactStructuredScopView> exactScop_;
  std::shared_ptr<const StructuredPolyhedralScopView> polyhedralScop_;
  ArtifactRootReference fabric_;

  friend llvm::Expected<StructuredScheduleDecisionDomain>
  enumerateStructuredScheduleDecisions(const StructuredProgramCandidate &,
                                       const fabric::FinalizedFabricRoot &,
                                       std::uint64_t,
                                       std::optional<StructuredEntityRef>);
  friend llvm::Expected<MaterializedStructuredScheduleCandidate>
  materializeStructuredScheduleProposal(
      const StructuredProgramCandidate &, const StructuredScheduleProposal &,
      const fabric::FinalizedFabricRoot &, std::optional<StructuredEntityRef>,
      llvm::ArrayRef<StructuredOperationSourceProvenance>);
};

struct StructuredScheduleDecisionDomain final {
  std::vector<StructuredScheduleProposal> proposals;
  /// Exact general schedules, including source-realized, materializable, and
  /// typed-unavailable forms. These removable views are analysis results, not
  /// candidate identities or persistent Schedule artifacts.
  std::vector<StructuredPolyhedralScopView> polyhedralScops;
  std::vector<StructuredScopRefusal> refusals;
  std::uint64_t inspectedLoopScopes = 0;
  std::uint64_t inspectedDecisionCoordinates = 0;
  std::uint64_t inspectedPolyhedralDependenceQueries = 0;
};

llvm::ArrayRef<std::uint8_t> structuredScheduleDecisionSchemaBytes();
/// Canonical diagnostic spelling of one decision kind.
llvm::StringRef
structuredScheduleDecisionKindSpelling(StructuredScheduleDecisionKind kind);
llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredScheduleDecision(const StructuredScheduleDecision &decision);
llvm::Expected<StructuredScheduleDecision>
adoptStructuredScheduleDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes);

/// Independently verifies that an upstream-vectorized child realizes exactly
/// one coordinate over the provider-proven source SCoP.
llvm::Error verifyStructuredVectorScheduleMaterialization(
    const ExactStructuredScopView &source,
    const StructuredVectorScheduleCoordinate &coordinate,
    mlir::ModuleOp materialized);

/// Enumerates the finite canonical proposal domain in loop order. Scalar
/// replication uses proved aggregate Fabric-capacity pruning; vector provider
/// and Fabric gates run only for selected proposals. No proposal is a Mapping
/// feasibility claim. A scope restricts work to that operation and descendants.
llvm::Expected<StructuredScheduleDecisionDomain>
enumerateStructuredScheduleDecisions(
    const StructuredProgramCandidate &parent,
    const fabric::FinalizedFabricRoot &fabric,
    std::uint64_t scopeExpansionLimit,
    std::optional<StructuredEntityRef> schedulingScope = std::nullopt);

/// Applies exactly one typed decision to a private clone and finalizes the
/// complete immutable child. Failure publishes no partial candidate.
llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleDecision(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision,
    std::optional<StructuredEntityRef> trackedSpatialRegion = std::nullopt,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance = {});

/// Applies one enumerated proposal while reusing its frozen exact-SCoP proof.
/// A provider or exact-Fabric negative is a typed proposal refusal.
llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleProposal(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleProposal &proposal,
    const fabric::FinalizedFabricRoot &fabric,
    std::optional<StructuredEntityRef> trackedSpatialRegion = std::nullopt,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance = {});

/// Re-enumerates the exact parent/Fabric domain, replays the production
/// materializer, and requires the immutable child. Downstream Dataflow and
/// event owners derive their own identity and invalidation facts.
llvm::Error
verifyStructuredScheduleDerivation(const StructuredProgramCandidate &parent,
                                   const fabric::FinalizedFabricRoot &fabric,
                                   const StructuredScheduleDecision &decision,
                                   const StructuredProgramCandidate &child);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
