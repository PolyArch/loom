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
};

/// One atomic schedule decision over an exact parent-local loop. A zero factor
/// is canonical only for Interchange; Tile and Unroll carry a positive factor.
struct StructuredScheduleDecision final {
  StructuredEntityRef loop;
  StructuredScheduleDecisionKind kind;
  std::uint64_t factor = 0;
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
llvm::Expected<StructuredProgramCandidate>
materializeStructuredScheduleDecision(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCHEDULE_H
