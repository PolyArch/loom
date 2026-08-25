#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/StructuredScop.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
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
};

enum class StructuredVectorTailPolicy : std::uint32_t {
  Exact = 0,
  ReductionMask = 1,
};

enum class StructuredVectorAliasPolicy : std::uint32_t {
  ProviderProvenNoAlias = 0,
};

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
  std::optional<StructuredEntityRef> transformedLoop;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

struct StructuredScheduleDecisionDomain final {
  std::vector<StructuredScheduleDecision> decisions;
  std::vector<StructuredScopRefusal> refusals;
  std::uint64_t inspectedLoopScopes = 0;
};

llvm::ArrayRef<std::uint8_t> structuredScheduleDecisionSchemaBytes();
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

/// Enumerates the finite legal schedule domain in canonical loop order. The
/// Fabric is consumed only for proved aggregate-capacity pruning; a surviving
/// decision is not a Mapping feasibility claim. When `schedulingScope` is
/// present, only loops nested in that exact operation consume the scope bound.
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

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
