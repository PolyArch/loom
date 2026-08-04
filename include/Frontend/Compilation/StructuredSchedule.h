#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
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
/// positive factor.
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
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

/// Enumerates the finite legal schedule domain in canonical loop order. The
/// Fabric is consumed only for proved aggregate-capacity pruning; a surviving
/// decision is not a Mapping feasibility claim.
llvm::Expected<std::vector<StructuredScheduleDecision>>
enumerateStructuredScheduleDecisions(const StructuredProgramCandidate &parent,
                                     const fabric::FinalizedFabricRoot &fabric,
                                     std::uint64_t scopeExpansionLimit);

/// Applies exactly one typed decision to a private clone and finalizes the
/// complete immutable child. Failure publishes no partial candidate.
llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleDecision(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
