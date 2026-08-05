#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDEXECUTIONSHAPE_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDEXECUTIONSHAPE_H

#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Raising/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::frontend {

/// One candidate-wide disposition for every exactly representable
/// llvm.intr.fmuladd owned by the selected Spatial region. The intrinsic is a
/// target choice, so the complete region receives one explicit shape and no
/// operation receives an implicit default.
struct StructuredExecutionShapeDecision final {
  raising::FMulAddExecutionShape fmuladdShape;

  friend bool operator==(const StructuredExecutionShapeDecision &lhs,
                         const StructuredExecutionShapeDecision &rhs) {
    return lhs.fmuladdShape == rhs.fmuladdShape;
  }
};

struct MaterializedStructuredExecutionShapeCandidate final {
  StructuredProgramCandidate structuredProgram;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

llvm::ArrayRef<std::uint8_t> structuredExecutionShapeDecisionSchemaBytes();
llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredExecutionShapeDecision(
    const StructuredExecutionShapeDecision &decision);
llvm::Expected<StructuredExecutionShapeDecision>
adoptStructuredExecutionShapeDecision(
    llvm::ArrayRef<std::uint8_t> canonicalBytes);

/// Returns the canonical Fused/Split domain exactly when a selected Spatial
/// region contains an unresolved, exactly representable llvm.intr.fmuladd.
/// A candidate with no unresolved choice has an empty domain. A selected
/// region containing an unrepresentable choice is rejected rather than
/// partially materialized.
llvm::Expected<std::vector<StructuredExecutionShapeDecision>>
enumerateStructuredExecutionShapeDecisions(
    const StructuredProgramCandidate &parent);

/// Applies one complete decision to a private clone. Only operations owned by
/// selected Spatial regions are changed; residual InstructionCore code keeps
/// its original LLVM representation.
llvm::Expected<MaterializedStructuredExecutionShapeCandidate>
materializeStructuredExecutionShapeDecision(
    const StructuredProgramCandidate &parent,
    const StructuredExecutionShapeDecision &decision);

/// Applies the same decision while preserving the ownership candidate's
/// removable block-activity and source-provenance projections.
llvm::Expected<MaterializedStructuredOwnershipCandidate>
materializeStructuredExecutionShapeDecision(
    MaterializedStructuredOwnershipCandidate parent,
    const StructuredExecutionShapeDecision &decision);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDEXECUTIONSHAPE_H
