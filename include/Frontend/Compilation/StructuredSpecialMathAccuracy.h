#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSPECIALMATHACCURACY_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSPECIALMATHACCURACY_H

#include "Common/SpecialMathAccuracy.h"
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
