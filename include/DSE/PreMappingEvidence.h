#ifndef LOOM_DSE_PREMAPPINGEVIDENCE_H
#define LOOM_DSE_PREMAPPINGEVIDENCE_H

#include "DSE/PreMappingExploration.h"

#include "llvm/Support/JSON.h"

namespace loom::dse {

/// Canonical diagnostic serialization for the bounded pre-Mapping planner.
/// These objects are invocation evidence, not Artifact content.
llvm::json::Object
serializePreMappingWorkCounter(const PreMappingWorkCounter &counter);
llvm::json::Object
serializePreMappingWorkAccounting(const PreMappingWorkAccounting &accounting);
llvm::json::Object serializePreMappingEvaluationTiming(
    const StructuredOwnershipEvaluationTiming &timing);
llvm::json::Object serializePreMappingFunnelSummary(
    llvm::ArrayRef<PreMappingCandidatePlanningRecord> inventory,
    llvm::ArrayRef<SelectedPreMappingCompilation> selected,
    const PreMappingWorkAccounting &accounting,
    const StructuredOwnershipEvaluationTiming &evaluationTiming);
llvm::json::Object serializePreMappingCandidateProjection(
    const PreMappingCandidateProjection &projection);
llvm::json::Object serializePreMappingMaterializedProjection(
    const PreMappingMaterializedProjection &projection);
llvm::json::Object serializePreMappingCandidatePlanningRecord(
    const PreMappingCandidatePlanningRecord &record);
llvm::json::Object serializePreMappingSelectionEvidence(
    const CompletedPreMappingSelection &selection);
llvm::json::Object serializePreMappingIncompleteEvidence(
    const IncompletePreMappingExploration &incomplete);

} // namespace loom::dse

#endif // LOOM_DSE_PREMAPPINGEVIDENCE_H
