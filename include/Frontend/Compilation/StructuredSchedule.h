#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H

#include "Fabric/Artifact/FabricArtifact.h"
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
};

/// One atomic schedule decision over an exact parent-local loop. A zero factor
/// is canonical for decisions without a factor; replication decisions carry a
/// factor greater than one.
struct StructuredScheduleDecision final {
  StructuredEntityRef loop;
  StructuredScheduleDecisionKind kind;
  std::uint64_t factor = 0;

  friend bool operator==(const StructuredScheduleDecision &lhs,
                         const StructuredScheduleDecision &rhs) {
    return lhs.loop == rhs.loop && lhs.kind == rhs.kind &&
           lhs.factor == rhs.factor;
  }
};

struct MaterializedStructuredScheduleCandidate final {
  StructuredProgramCandidate structuredProgram;
  std::optional<StructuredEntityRef> trackedSpatialRegion;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

struct StructuredScheduleDecisionDomain final {
  std::vector<StructuredScheduleDecision> decisions;
  std::uint64_t inspectedLoopScopes = 0;
};

llvm::ArrayRef<std::uint8_t> structuredScheduleDecisionSchemaBytes();
llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredScheduleDecision(const StructuredScheduleDecision &decision);
llvm::Expected<StructuredScheduleDecision>
adoptStructuredScheduleDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes);

/// Enumerates the finite legal schedule domain in canonical loop order. The
/// Fabric is consumed only for proved aggregate-capacity pruning; a surviving
/// decision is not a Mapping feasibility claim.
llvm::Expected<StructuredScheduleDecisionDomain>
enumerateStructuredScheduleDecisions(const StructuredProgramCandidate &parent,
                                     const fabric::FinalizedFabricRoot &fabric,
                                     std::uint64_t scopeExpansionLimit);

/// Applies exactly one typed decision to a private clone and finalizes the
/// complete immutable child. Failure publishes no partial candidate.
llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleDecision(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision,
    std::optional<StructuredEntityRef> trackedSpatialRegion = std::nullopt);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
