#include "SystemActivityInternal.h"
#include "SimulationExecutionInternal.h"
#include "Simulator/SystemActivity.h"
#include "Deployment/Deployment.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5BuiltinModels.h"

namespace loom::sim {
namespace detail {
/// One accelerated phase stays inside the computation interval that selected
/// it, measures a nonnegative service integral, and cannot consume more
/// service than its own span.
static llvm::Error
validateAcceleratedPhase(const SystemAcceleratedPhase &phase,
                         const SystemComputationInterval &interval) {
  if (phase.beginTick < interval.beginTick || phase.endTick > interval.endTick ||
      phase.beginTick > phase.endTick ||
      phase.beginMemoryOccupiedTicks < interval.beginMemoryOccupiedTicks ||
      phase.endMemoryOccupiedTicks > interval.endMemoryOccupiedTicks ||
      phase.beginMemoryOccupiedTicks > phase.endMemoryOccupiedTicks ||
      phase.occupiedTicks() > phase.elapsedTicks())
    return invalid("System accelerated phase is outside its computation "
                   "interval time or service domain");
  return llvm::Error::success();
}

llvm::Error validateSystemMemoryActivity(const SystemSimulationExecution &execution,
                                        const SystemExecutionContext &context) {
  if (!execution.memoryActivity && execution.computationInterval.has_value())
    return invalid(
        "System computation intervals require native memory activity");
  if (execution.acceleratedPhases && !execution.computationInterval)
    return invalid("System accelerated phases require a computation interval");
  if (!execution.memoryActivity)
    return llvm::Error::success();
  const auto &progress = execution.progressObservations;
  if (!std::holds_alternative<RetiredExecution>(execution.terminal) ||
      !progress.programExitVisible || progress.programEntryAccepted.delta != 0 ||
      progress.programExitVisible->delta != 0 ||
      progress.programEntryAccepted.gem5Tick >= progress.programExitVisible->gem5Tick)
    return invalid("System memory activity requires a positive retired gem5 program window");
  const auto elapsed = progress.programExitVisible->gem5Tick -
                       progress.programEntryAccepted.gem5Tick;
  if (execution.memoryActivity->occupiedTicks > elapsed)
    return invalid("System memory service occupancy exceeds its full program window");
  for (const SystemRootLifecycleObservation &observation : progress.rootLifecycle)
    if (observation.memoryOccupiedTicks > execution.memoryActivity->occupiedTicks)
      return invalid("root lifecycle memory service sample exceeds the full "
                     "program service occupancy");
  if (execution.computationInterval) {
    const auto &interval = *execution.computationInterval;
    if (interval.beginTick < progress.programEntryAccepted.gem5Tick ||
        interval.beginTick >= interval.endTick ||
        interval.endTick > progress.programExitVisible->gem5Tick ||
        interval.beginMemoryOccupiedTicks > interval.endMemoryOccupiedTicks ||
        interval.endMemoryOccupiedTicks >
            execution.memoryActivity->occupiedTicks ||
        interval.endMemoryOccupiedTicks - interval.beginMemoryOccupiedTicks >
            interval.endTick - interval.beginTick)
      return invalid("System computation interval is outside its native time "
                     "or service domain");
    if (execution.acceleratedPhases) {
      const auto &phases = *execution.acceleratedPhases;
      if (llvm::Error error =
              validateAcceleratedPhase(phases.configurationResidency, interval))
        return error;
      if (llvm::Error error =
              validateAcceleratedPhase(phases.invocation, interval))
        return error;
      if (phases.invocation.beginTick < phases.configurationResidency.beginTick ||
          phases.invocation.endTick < phases.configurationResidency.endTick)
        return invalid("System invocation phase precedes its configuration "
                       "residency phase");
    }
  }
  const auto kind = context.request->modelBinding().descriptorRef().modelKind();
  using evaluation::BuiltinEvaluationModel;
  using evaluation::builtinEvaluationModelKind;
  if (kind != builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemDfg) &&
      kind != builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemCgra) &&
      kind != builtinEvaluationModelKind(BuiltinEvaluationModel::Gem5SystemRtl))
    return invalid("System memory activity requires a gem5 System provider");
  const ArtifactRootReference *subject = nullptr;
  for (const auto &role : context.request->subjectBindings().roleBindings())
    for (const auto &candidate : role.subjects)
      if (candidate.schemaIdentity == runtime::gem5SimulationBindingSchema.identity) {
        if (subject && *subject != candidate)
          return invalid("System memory activity has multiple gem5 bindings");
        subject = &candidate;
      }
  if (!subject)
    return invalid("System memory activity has no gem5 binding");
  auto binding = runtime::importGem5SimulationBinding(*subject, *context.artifactStore);
  if (!binding)
    return binding.takeError();
  auto fabric = deployment::deploymentFabric(context.inputs->deployment.deployment(),
                                              *context.artifactStore);
  if (!fabric)
    return fabric.takeError();
  if (*fabric != binding->binding().fabric())
    return invalid("System memory activity binding names another Fabric");
  auto memory = runtime::projectGem5SharedMemory(binding->binding());
  if (!memory)
    return memory.takeError();
  if (!*memory)
    return invalid("System memory activity has no single shared SimpleMemory capacity domain");
  return llvm::Error::success();
}
}

llvm::Expected<std::optional<evaluation::ExactRatio>>
projectSystemMemoryUtilization(const CanonicalSimulationExecution &execution,
                              const evaluation::CaseArtifactResolution &resolution,
                              const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto *system = execution.system();
  if (!system)
    return detail::invalid("System memory utilization requires a System execution");
  if (!system->memoryActivity)
    return std::optional<evaluation::ExactRatio>{};
  auto context = detail::resolveSystemExecutionContext(system->request, resolution,
                                                       artifacts, blobs);
  if (!context)
    return context.takeError();
  if (auto error = detail::validateSystemMemoryActivity(*system, *context))
    return std::move(error);
  const auto &progress = system->progressObservations;
  const auto elapsed = progress.programExitVisible->gem5Tick -
                       progress.programEntryAccepted.gem5Tick;
  auto ratio = evaluation::ExactRatio::get(system->memoryActivity->occupiedTicks, elapsed);
  if (!ratio)
    return ratio.takeError();
  return std::optional<evaluation::ExactRatio>{*ratio};
}
}
