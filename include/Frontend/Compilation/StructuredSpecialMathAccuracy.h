#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSPECIALMATHACCURACY_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSPECIALMATHACCURACY_H

#include "Common/SpecialMathAccuracy.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::frontend {

struct StructuredSpecialMathAccuracyDecision final {
  StructuredEntityRef operation;
  SpecialMathAccuracyTier accuracy;

  friend bool operator==(const StructuredSpecialMathAccuracyDecision &lhs,
                         const StructuredSpecialMathAccuracyDecision &rhs) {
    return lhs.operation == rhs.operation && lhs.accuracy == rhs.accuracy;
  }
};

struct MaterializedStructuredSpecialMathCandidate final {
  StructuredProgramCandidate structuredProgram;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

/// Whether a registered special-math operation still requires its typed
/// accuracy decision. Other operations and already selected tiers return
/// false; malformed selected state remains the strict schema owner's error.
bool hasUnresolvedStructuredSpecialMathAccuracy(mlir::Operation *operation);

llvm::ArrayRef<std::uint8_t> structuredSpecialMathAccuracyDecisionSchemaBytes();
llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredSpecialMathAccuracyDecision(
    const StructuredSpecialMathAccuracyDecision &decision);
llvm::Expected<StructuredSpecialMathAccuracyDecision>
adoptStructuredSpecialMathAccuracyDecision(
    llvm::ArrayRef<std::uint8_t> canonicalBytes);

/// Returns the canonical tier domain for the first unresolved selected-Spatial
/// special-math operation. An empty result means that every such operation is
/// closed. Repeated immutable lineage edges compose decisions for multiple
/// operations without storing a per-program decision table.
llvm::Expected<std::vector<StructuredSpecialMathAccuracyDecision>>
enumerateStructuredSpecialMathAccuracyDecisions(
    const StructuredProgramCandidate &parent);

/// Projects every currently legal typed actor form for one registered
/// special-math operation without selecting a tier. Earlier soft capacity
/// analysis consumes this domain; only the decision materializer closes it.
llvm::Expected<std::vector<dataflow::CanonicalActorSchemaProjection>>
projectStructuredSpecialMathAccuracyDomain(mlir::Operation *operation);

llvm::Expected<MaterializedStructuredSpecialMathCandidate>
materializeStructuredSpecialMathAccuracyDecision(
    const StructuredProgramCandidate &parent,
    const StructuredSpecialMathAccuracyDecision &decision);

llvm::Expected<MaterializedStructuredOwnershipCandidate>
materializeStructuredSpecialMathAccuracyDecision(
    MaterializedStructuredOwnershipCandidate parent,
    const StructuredSpecialMathAccuracyDecision &decision);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSPECIALMATHACCURACY_H
