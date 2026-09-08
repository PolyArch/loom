#include "DSE/StructuredFunctionalReplayBudget.h"

#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "DSE/StructuredOwnershipInvocation.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Frontend/IR/LoomOps.h"
#include "Simulator/NativeSimulationOracle.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace loom::dse {
namespace {
constexpr std::uint64_t defaultWavefrontStepsPerActivation = 100000;
constexpr std::uint64_t defaultEventsPerActivation = 1000000;
/// One activation's replay work grows with the executable leaves it runs:
/// a whole-loop region replays every iteration inside one activation, and
/// the CGRA model spends tens of event frames per actor firing, so the
/// aggregate grant adds a per-leaf allowance to the per-activation base.
constexpr std::uint64_t defaultWavefrontStepsPerLeafExecution = 32;
constexpr std::uint64_t defaultEventsPerLeafExecution = 256;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_replay_budget_invalid: " +
                                     message);
}

ArtifactRootReference reference(const ArtifactSchemaDescriptor &schema,
                                const ArtifactIdentity &identity) {
  return {schema.identity.str(), schema.version, identity};
}
} // namespace

llvm::Expected<sim::SourceBackedDfgValidationLimits>
planStructuredFunctionalReplayBudget(
    const StructuredFunctionalReplayBudget &budget,
    const frontend::MaterializedOwnershipCandidate &candidate,
    const frontend::StructuredProgramCandidate &source,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    const StructuredOwnershipSharedEvaluation *sharedEvaluation,
    ExecutionControlView executionControl) {
  if ((budget.maxWavefrontSteps && *budget.maxWavefrontSteps == 0) ||
      (budget.maxEventCount && *budget.maxEventCount == 0) ||
      budget.maxRetainedCaptureBytes == 0 ||
      budget.maxSimulationWallTime <
          std::chrono::steady_clock::duration::zero())
    return invalid("explicit execution limits must be positive");
  sim::SourceBackedDfgValidationLimits limits{
      budget.maxWavefrontSteps.value_or(defaultWavefrontStepsPerActivation),
      budget.maxEventCount.value_or(defaultEventsPerActivation),
      budget.maxRetainedCaptureBytes, budget.maxSimulationWallTime};
  if (executionControl.stopRequested()) {
    limits.maxSimulationWallTime = std::chrono::steady_clock::duration::zero();
    return limits;
  }
  std::uint64_t activations = 0;
  std::uint64_t leafExecutions = 0;
  if (!budget.maxWavefrontSteps || !budget.maxEventCount) {
    std::shared_ptr<const sim::NativeStructuredProgramObservations> shared;
    std::optional<sim::NativeStructuredProgramObservations> owned;
    const sim::NativeStructuredProgramObservations *observations;
    if (sharedEvaluation) {
      auto profiled = sharedEvaluation->profiledObservations(
          reference(frontend::structuredProgramArtifactSchema,
                    candidate.structuredProgram.identity()),
          reference(frontend::structuredProgramArtifactSchema,
                    source.identity()),
          reference(sim::simulationWorkloadSchema, workload.identity()),
          reference(sim::simulationRuntimeInputSchema, runtimeInput.identity()),
          candidate.structuredProgram, source, workload, runtimeInput);
      if (!profiled)
        return profiled.takeError();
      shared = std::move(*profiled);
      observations = shared.get();
    } else {
      auto profiled = sim::executeProfiledSelectedStructuredProgram(
          candidate.structuredProgram, source, workload, runtimeInput);
      if (!profiled)
        return profiled.takeError();
      owned.emplace(std::move(*profiled));
      observations = &*owned;
    }
    auto view = candidate.structuredProgram.view();
    if (!view)
      return view.takeError();
    std::vector<frontend::StructuredEntityRef> regions;
    for (const auto &entity :
         view->entities(frontend::StructuredEntityKind::Operation))
      if (llvm::isa_and_nonnull<loom::SpatialRegionOp>(entity.operation))
        regions.push_back(entity.reference);
    auto activity = evaluation::models::projectStructuredScopeActivity(
        candidate.structuredProgram, *observations, regions,
        evaluation::models::StructuredScopeActivityDomain::OwnedRegions);
    if (!activity)
      return activity.takeError();
    for (const auto &region : *activity) {
      auto sum =
          llvm::checkedAddUnsigned(activations, region.dynamicActivations);
      auto leaves = llvm::checkedAddUnsigned(leafExecutions,
                                             region.dynamicLeafExecutions);
      if (!sum || !leaves)
        return invalid("selected-region activation count overflows");
      activations = *sum;
      leafExecutions = *leaves;
    }
    // An empty observed domain still reaches the existing Inapplicable check.
    const auto planningActivations = std::max<std::uint64_t>(activations, 1);
    const auto grant = [&](std::uint64_t perActivation,
                           std::uint64_t perLeaf) -> std::optional<std::uint64_t> {
      auto base = llvm::checkedMulUnsigned(planningActivations, perActivation);
      auto work = llvm::checkedMulUnsigned(leafExecutions, perLeaf);
      if (!base || !work)
        return std::nullopt;
      return llvm::checkedAddUnsigned(*base, *work);
    };
    auto waves = grant(defaultWavefrontStepsPerActivation,
                       defaultWavefrontStepsPerLeafExecution);
    auto events =
        grant(defaultEventsPerActivation, defaultEventsPerLeafExecution);
    if ((!budget.maxWavefrontSteps && !waves) ||
        (!budget.maxEventCount && !events))
      return invalid("aggregate default work grant overflows");
    if (!budget.maxWavefrontSteps)
      limits.maxWavefrontSteps = *waves;
    if (!budget.maxEventCount)
      limits.maxEventCount = *events;
  }
  if (auto remaining = executionControl.remainingTime())
    limits.maxSimulationWallTime = std::min(
        limits.maxSimulationWallTime,
        std::max(*remaining, std::chrono::steady_clock::duration::zero()));
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "structured_functional_replay_budget";
        fields["source_program"] = formatArtifactIdentityHex(source.identity());
        fields["structured_program"] =
            formatArtifactIdentityHex(candidate.structuredProgram.identity());
        fields["workload"] = formatArtifactIdentityHex(workload.identity());
        fields["runtime_input"] =
            formatArtifactIdentityHex(runtimeInput.identity());
        if (!budget.maxWavefrontSteps || !budget.maxEventCount) {
          fields["observed_region_activations"] = activations;
          fields["observed_region_leaf_executions"] = leafExecutions;
        }
        fields["aggregate_wavefront_limit"] = limits.maxWavefrontSteps;
        fields["aggregate_event_limit"] = limits.maxEventCount;
        fields["wavefront_limit_explicit"] =
            budget.maxWavefrontSteps.has_value();
        fields["event_limit_explicit"] = budget.maxEventCount.has_value();
      });
  return limits;
}
} // namespace loom::dse
