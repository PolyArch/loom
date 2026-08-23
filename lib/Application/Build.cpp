#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "ExecutionGlue.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/Objective.h"
#include "DSE/ProductionOwners.h"
#include "DSE/Promotion.h"
#include "DSE/ResolvedConfigView.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/DeploymentPipeline.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Executable/ExecutableElf.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationDiagnostics.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Hardware/Implementation/FabricModel.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/PnrDerivedContext.h"
#include "Runtime/FabricModelPlatform.h"
#include "Runtime/Gem5DispatchABI.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialInvocation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>

namespace loom::application {
namespace {

using MonotonicClock = std::chrono::steady_clock;

constexpr std::uint64_t kPortableRiscVHostImageBase = 0x80000000;
constexpr std::uint64_t kExecutablePageBytes = 4096;
constexpr std::uint64_t kApplicationReplayExecutionLimit = 1000000;

std::uint64_t elapsedNanoseconds(MonotonicClock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             MonotonicClock::now() - begin)
      .count();
}

void emitElapsed(ApplicationBuildOperation operation,
                 MonotonicClock::time_point begin,
                 std::uint64_t deterministicWork = 1) {
  emitApplicationBuildOperationStatistics(
      {operation, elapsedNanoseconds(begin), deterministicWork});
}

class ApplicationBuildOperationTimer final {
public:
  explicit ApplicationBuildOperationTimer(ApplicationBuildOperation operation)
      : operation_(operation), begin_(MonotonicClock::now()) {}

  ~ApplicationBuildOperationTimer() { emitElapsed(operation_, begin_); }

  ApplicationBuildOperationTimer(const ApplicationBuildOperationTimer &) =
      delete;
  ApplicationBuildOperationTimer &
  operator=(const ApplicationBuildOperationTimer &) = delete;

private:
  ApplicationBuildOperation operation_;
  MonotonicClock::time_point begin_;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_build_invalid: " + message);
}

llvm::Expected<std::uint64_t> nextExecutableImageBase(std::uint64_t end) {
  if (end >
      std::numeric_limits<std::uint64_t>::max() - (kExecutablePageBytes - 1))
    return invalid("executable image range cannot be page-aligned");
  return (end + kExecutablePageBytes - 1) & ~(kExecutablePageBytes - 1);
}

struct SourceSimulationInputs final {
  sim::CanonicalSimulationWorkload workload;
  sim::CanonicalSimulationRuntimeInput runtimeInput;
};

struct ImportedApplicationMapping final {
  mapping::FinalizedSystemMapping mapping;
  dataflow::CanonicalDataflowArtifact dataflow;
  dataflow::CanonicalDataflowProgramView dataflowView;
  fabric::FinalizedFabricRoot system;
};

llvm::Expected<ArtifactRootReference>
requireUniqueSystemMapping(const dse::JointDesignExecution &execution) {
  std::vector<ArtifactRootReference> mappings;
  for (const dse::JointMappedPair &pair : execution.mappedPairs)
    mappings.insert(mappings.end(), pair.systemMappings.begin(),
                    pair.systemMappings.end());
  llvm::sort(mappings, artifactRootReferenceLess);
  mappings.erase(std::unique(mappings.begin(), mappings.end()), mappings.end());
  if (!execution.summary.selectedMapping)
    return invalid("Deployment requires one explicitly selected SystemMapping");
  if (!llvm::is_contained(mappings, *execution.summary.selectedMapping))
    return invalid("selected SystemMapping is outside the verified result");
  return *execution.summary.selectedMapping;
}

llvm::Expected<ImportedApplicationMapping>
importApplicationMapping(const dse::JointDesignExecution &execution,
                         const ArtifactStore &artifacts) {
  auto reference = requireUniqueSystemMapping(execution);
  if (!reference)
    return reference.takeError();
  auto mapping = mapping::importSystemMapping(*reference, artifacts);
  if (!mapping)
    return mapping.takeError();
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      mapping->view().dataflowIdentity()};
  auto dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  const ArtifactRootReference systemReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version, mapping->view().fabricIdentity()};
  auto system = fabric::importEntireFabricRoot(systemReference, artifacts);
  if (!system)
    return system.takeError();
  auto systemView = fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();
  return ImportedApplicationMapping{std::move(*mapping), std::move(*dataflow),
                                    std::move(*dataflowView),
                                    std::move(*system)};
}

llvm::Expected<const PreparedApplicationSoftware *>
findPreparedSoftware(const PreparedApplicationBuild &prepared,
                     const ArtifactIdentity &dataflowIdentity) {
  const PreparedApplicationSoftware *selected = nullptr;
  for (const PreparedApplicationSoftware &software : prepared.software) {
    if (software.compilation.canonicalDataflow.artifact != dataflowIdentity)
      continue;
    if (selected)
      return invalid("prepared build repeats one Canonical Dataflow owner");
    selected = &software;
  }
  if (!selected)
    return invalid("SystemMapping names a foreign prepared software owner");
  return selected;
}

struct ApplicationRuntimeValidation final {
  ApplicationMappingRuntimeDisposition disposition =
      ApplicationMappingRuntimeDisposition::ProofNotEstablished;
  std::vector<ArtifactRootReference> evidence;
  std::optional<std::uint64_t> dfgCycles;
  std::optional<std::uint64_t> cgraCycles;
  std::optional<dse::SpatialFifoRuntimeFeedback> spatialFifoFeedback;
  std::optional<dse::SpatialOperandQueueRuntimeFeedback>
      spatialOperandQueueFeedback;
};

ApplicationMappingRuntimeDisposition
runtimeDisposition(evaluation::EvidenceOutcomeKind outcome) {
  using Evidence = evaluation::EvidenceOutcomeKind;
  switch (outcome) {
  case Evidence::Completed:
    return ApplicationMappingRuntimeDisposition::Completed;
  case Evidence::Unsupported:
    return ApplicationMappingRuntimeDisposition::Unsupported;
  case Evidence::ExecutionFailed:
    return ApplicationMappingRuntimeDisposition::ExecutionFailed;
  case Evidence::CancelledOrTimeout:
    return ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
  }
  llvm_unreachable("unknown Evaluation Evidence outcome");
}

void emitRuntimeEvidenceFailure(
    llvm::StringRef model, const evaluation::EvaluationEvidence &evidence) {
  std::optional<evaluation::OutcomeReason> reason;
  std::visit(
      [&](const auto &outcome) {
        using Outcome = std::decay_t<decltype(outcome)>;
        if constexpr (!std::is_same_v<Outcome, evaluation::CompletedEvidence>)
          reason = outcome.reason;
      },
      evidence.outcome());
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "application_runtime_validation";
        fields["model"] = model;
        fields["outcome"] = evaluation::toString(evidence.outcomeKind());
        fields["reason"] = reason ? evaluation::toString(*reason) : "none";
      });
}

llvm::Expected<std::vector<pnr::SystemBindingPartitionIntent>>
deriveSystemBindingPartitionIntent(const dse::ResourceTimeScheduleHint &hint) {
  std::map<std::uint64_t, pnr::SystemBindingPartitionIntent> byRoot;
  for (const dse::ResourceTimeHintState &state : hint.states)
    for (const dse::ResourceTimeHintAllocation &allocation : state.active) {
      if (allocation.resourceUnits.size() != 1 ||
          allocation.resourceUnits.front() == 0)
        return invalid("resource-time allocation has no scalar System "
                       "partition count");
      auto [position, inserted] = byRoot.try_emplace(
          allocation.region.entity.value(),
          pnr::SystemBindingPartitionIntent{allocation.region,
                                            allocation.resourceUnits.front()});
      if (!inserted) {
        if (position->second.root != allocation.region)
          return invalid("resource-time partition intent crosses Dataflow "
                         "owners");
        position->second.partitionCount = std::max(
            position->second.partitionCount, allocation.resourceUnits.front());
      }
    }
  if (byRoot.empty())
    return invalid("resource-time schedule has no System partition intent");
  std::vector<pnr::SystemBindingPartitionIntent> result;
  result.reserve(byRoot.size());
  for (auto &[ordinal, partition] : byRoot) {
    (void)ordinal;
    result.push_back(std::move(partition));
  }
  return result;
}

llvm::Expected<std::optional<MonotonicClock::time_point>>
applicationReplayDeadline(const dse::PlanExecutionPolicy &policy) {
  if (!policy.dispatchNotAfterUnixNanoseconds())
    return std::nullopt;
  if (*policy.dispatchNotAfterUnixNanoseconds() >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid("Mapping deadline exceeds the clock representation");
  const auto deadline = std::chrono::system_clock::time_point{
      std::chrono::nanoseconds{static_cast<std::int64_t>(
          *policy.dispatchNotAfterUnixNanoseconds())}};
  const auto remaining = deadline - std::chrono::system_clock::now();
  if (remaining <= std::chrono::system_clock::duration::zero())
    return MonotonicClock::now();
  return MonotonicClock::now() +
         std::chrono::duration_cast<MonotonicClock::duration>(remaining);
}

llvm::Expected<ArtifactRootReference>
requireExecutionOutput(const evaluation::EvaluationEvidence &evidence) {
  std::vector<ArtifactRootReference> executions;
  for (const evaluation::ModelOutputBinding &binding :
       evidence.outputBindings())
    for (const ArtifactRootReference &reference : binding.artifacts)
      if (reference.schemaIdentity == sim::simulationExecutionSchema.identity &&
          reference.schemaVersion == sim::simulationExecutionSchema.version)
        executions.push_back(reference);
  llvm::sort(executions, artifactRootReferenceLess);
  executions.erase(std::unique(executions.begin(), executions.end()),
                   executions.end());
  if (executions.size() != 1)
    return invalid("completed simulation did not publish one execution");
  return executions.front();
}

llvm::Expected<std::uint64_t>
requireCompletedCycleMetric(const evaluation::EvaluationEvidence &evidence) {
  const auto *completed =
      std::get_if<evaluation::CompletedEvidence>(&evidence.outcome());
  if (!completed || completed->metricResults.size() != 1)
    return invalid("completed simulation did not publish one cycle metric");
  const auto *point = std::get_if<evaluation::PointObservation>(
      &completed->metricResults.front().observation);
  if (!point)
    return invalid("completed simulation cycle metric is not a point");
  const auto *integer = std::get_if<evaluation::IntegerValue>(&point->value);
  if (!integer || integer->value() < 0)
    return invalid("completed simulation cycle metric is not nonnegative");
  return static_cast<std::uint64_t>(integer->value());
}

llvm::Error accumulateCycle(std::optional<std::uint64_t> &total,
                            std::uint64_t value, llvm::StringRef subject) {
  const std::uint64_t current = total.value_or(0);
  if (value > std::numeric_limits<std::uint64_t>::max() - current)
    return invalid(subject + " cycle count overflows uint64");
  total = current + value;
  return llvm::Error::success();
}

llvm::Expected<ApplicationRuntimeValidation> validateApplicationMappingRuntime(
    const PreparedApplicationBuild &prepared,
    const PreparedApplicationMappingAlternative &alternative,
    const dse::JointDesignExecution &execution,
    const dse::PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto imported = importApplicationMapping(execution, artifacts);
  if (!imported)
    return imported.takeError();
  if (imported->mapping.view().dataflowIdentity() !=
      alternative.dataflow.artifact)
    return invalid("runtime validation selected a foreign software owner");
  auto software = findPreparedSoftware(
      prepared, imported->mapping.view().dataflowIdentity());
  if (!software)
    return software.takeError();
  if ((*software)->replayCases.empty())
    return ApplicationRuntimeValidation{
        ApplicationMappingRuntimeDisposition::ProofNotEstablished,
        {},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};

  auto contexts = mapping::projectSystemExecutionContexts(
      imported->dataflowView, imported->mapping.view().executionBindings());
  if (!contexts)
    return contexts.takeError();
  auto deadline = applicationReplayDeadline(executionPolicy);
  if (!deadline)
    return deadline.takeError();

  ApplicationRuntimeValidation validation;
  validation.disposition = ApplicationMappingRuntimeDisposition::Completed;
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       (*software)->replayCases) {
    if (*deadline && MonotonicClock::now() >= **deadline) {
      validation.disposition =
          ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
      return validation;
    }
    auto inputs = sim::importSpatialSimulationInputs(
        replay.workload, replay.runtimeInput, artifacts);
    if (!inputs)
      return inputs.takeError();
    if (inputs->dataflow.identity() != alternative.dataflow.artifact)
      return invalid("source-backed replay names a foreign final Dataflow");
    const sim::SpatialSimulationWorkload *workload = inputs->workload.spatial();
    if (!workload)
      return invalid("source-backed replay is not a Spatial workload");
    auto selectedContext = mapping::selectSystemSpatialExecutionContext(
        *contexts, workload->launchRef, workload->denseCoordinates);
    if (!selectedContext)
      return selectedContext.takeError();
    auto spatialMapping = mapping::importSpatialMapping(
        selectedContext->spatialMapping, artifacts);
    if (!spatialMapping)
      return spatialMapping.takeError();
    const ArtifactRootReference module{
        fabric::fabricArtifactSchema.identity.str(),
        fabric::fabricArtifactSchema.version,
        spatialMapping->view().fabricIdentity()};

    auto preparedDfg = evaluation::models::prepareDfgSimulationEvaluation(
        alternative.dataflow, replay.workload, replay.runtimeInput,
        alternative.plan.resolvedConfig, artifacts, blobs);
    if (!preparedDfg)
      return preparedDfg.takeError();
    auto dfgEvidence = evaluation::models::evaluateDfgSimulation(
        *preparedDfg, {kApplicationReplayExecutionLimit, *deadline}, artifacts,
        blobs);
    if (!dfgEvidence)
      return dfgEvidence.takeError();
    auto dfgEvidenceReference =
        evaluation::publishEvaluationEvidence(*dfgEvidence, artifacts);
    if (!dfgEvidenceReference)
      return dfgEvidenceReference.takeError();
    validation.evidence.push_back(*dfgEvidenceReference);
    if (dfgEvidence->outcomeKind() !=
        evaluation::EvidenceOutcomeKind::Completed) {
      emitRuntimeEvidenceFailure("dfg_simulation", *dfgEvidence);
      validation.disposition = runtimeDisposition(dfgEvidence->outcomeKind());
      return validation;
    }
    auto dfgExecution = requireExecutionOutput(*dfgEvidence);
    if (!dfgExecution)
      return dfgExecution.takeError();
    auto dfgCycles = requireCompletedCycleMetric(*dfgEvidence);
    if (!dfgCycles)
      return dfgCycles.takeError();
    if (llvm::Error error =
            accumulateCycle(validation.dfgCycles, *dfgCycles, "DFG"))
      return std::move(error);

    auto preparedCgra = evaluation::models::prepareCgraSimulationEvaluation(
        alternative.dataflow, module, selectedContext->spatialMapping,
        replay.workload, replay.runtimeInput, alternative.plan.resolvedConfig,
        artifacts, blobs);
    if (!preparedCgra)
      return preparedCgra.takeError();
    auto cgraEvaluation =
        evaluation::models::evaluateCgraSimulationWithDiagnostics(
            *preparedCgra, {kApplicationReplayExecutionLimit, *deadline},
            artifacts, blobs);
    if (!cgraEvaluation)
      return cgraEvaluation.takeError();
    evaluation::EvaluationEvidence &cgraEvidence = cgraEvaluation->evidence;
    if (cgraEvaluation->closedWait) {
      auto operandFeedback = dse::deriveSpatialOperandQueueRuntimeFeedback(
          imported->mapping.reference(), *cgraEvaluation->closedWait,
          artifacts);
      if (!operandFeedback)
        return operandFeedback.takeError();
      dse::emitSpatialOperandQueueRuntimeFeedback(*operandFeedback);
      const auto operandPriority =
          [](dse::SpatialOperandQueueRuntimeFeedbackDisposition value) {
            switch (value) {
            case dse::SpatialOperandQueueRuntimeFeedbackDisposition::Exact:
              return 2;
            case dse::SpatialOperandQueueRuntimeFeedbackDisposition::
                ProofNotEstablished:
              return 1;
            case dse::SpatialOperandQueueRuntimeFeedbackDisposition::
                Unsupported:
              return 0;
            }
            llvm_unreachable(
                "unknown Spatial operand-queue feedback disposition");
          };
      if (!validation.spatialOperandQueueFeedback ||
          operandPriority(operandFeedback->disposition) >
              operandPriority(
                  validation.spatialOperandQueueFeedback->disposition))
        validation.spatialOperandQueueFeedback = std::move(*operandFeedback);
      auto feedback = dse::deriveSpatialFifoRuntimeFeedback(
          imported->mapping.reference(), selectedContext->spatialMapping,
          *cgraEvaluation->closedWait, artifacts);
      if (!feedback)
        return feedback.takeError();
      dse::emitSpatialFifoRuntimeFeedback(*feedback);
      const auto priority = [](dse::SpatialFifoRuntimeFeedbackDisposition
                                   value) {
        switch (value) {
        case dse::SpatialFifoRuntimeFeedbackDisposition::Exact:
          return 2;
        case dse::SpatialFifoRuntimeFeedbackDisposition::ProofNotEstablished:
          return 1;
        case dse::SpatialFifoRuntimeFeedbackDisposition::Unsupported:
          return 0;
        }
        llvm_unreachable("unknown Spatial FIFO feedback disposition");
      };
      if (!validation.spatialFifoFeedback ||
          priority(feedback->disposition) >
              priority(validation.spatialFifoFeedback->disposition))
        validation.spatialFifoFeedback = std::move(*feedback);
    }
    auto cgraEvidenceReference =
        evaluation::publishEvaluationEvidence(cgraEvidence, artifacts);
    if (!cgraEvidenceReference)
      return cgraEvidenceReference.takeError();
    validation.evidence.push_back(*cgraEvidenceReference);
    if (cgraEvidence.outcomeKind() !=
        evaluation::EvidenceOutcomeKind::Completed) {
      emitRuntimeEvidenceFailure("cgra_simulation", cgraEvidence);
      validation.disposition = runtimeDisposition(cgraEvidence.outcomeKind());
      return validation;
    }
    auto cgraExecution = requireExecutionOutput(cgraEvidence);
    if (!cgraExecution)
      return cgraExecution.takeError();
    auto cgraCycles = requireCompletedCycleMetric(cgraEvidence);
    if (!cgraCycles)
      return cgraCycles.takeError();
    if (llvm::Error error =
            accumulateCycle(validation.cgraCycles, *cgraCycles, "CGRA"))
      return std::move(error);

    auto comparison = evaluation::models::prepareSimulationComparisonEvaluation(
        *dfgExecution, preparedDfg->resolution, *cgraExecution,
        preparedCgra->resolution, alternative.plan.resolvedConfig, artifacts,
        blobs);
    if (!comparison)
      return comparison.takeError();
    auto comparisonEvidence = evaluation::models::evaluateSimulationComparison(
        *comparison, artifacts, blobs);
    if (!comparisonEvidence)
      return comparisonEvidence.takeError();
    auto comparisonEvidenceReference =
        evaluation::publishEvaluationEvidence(*comparisonEvidence, artifacts);
    if (!comparisonEvidenceReference)
      return comparisonEvidenceReference.takeError();
    validation.evidence.push_back(*comparisonEvidenceReference);
    if (comparisonEvidence->outcomeKind() !=
        evaluation::EvidenceOutcomeKind::Completed) {
      emitRuntimeEvidenceFailure("simulation_comparison", *comparisonEvidence);
      validation.disposition =
          runtimeDisposition(comparisonEvidence->outcomeKind());
      return validation;
    }
    const auto *completed = std::get_if<evaluation::CompletedEvidence>(
        &comparisonEvidence->outcome());
    if (!completed || completed->findingResults.size() != 1)
      return invalid("simulation comparison has no unique result");
    const evaluation::FindingResultValue &comparisonResult =
        completed->findingResults.front().result;
    if (std::holds_alternative<evaluation::AbsentFinding>(comparisonResult))
      continue;
    validation.disposition =
        std::holds_alternative<evaluation::NotApplicableFinding>(
            comparisonResult)
            ? ApplicationMappingRuntimeDisposition::ProofNotEstablished
            : ApplicationMappingRuntimeDisposition::ExecutionFailed;
    return validation;
  }
  llvm::sort(validation.evidence, artifactRootReferenceLess);
  validation.evidence.erase(
      std::unique(validation.evidence.begin(), validation.evidence.end()),
      validation.evidence.end());
  return validation;
}

llvm::Expected<deployment::CanonicalTypeBytes>
canonicalTypeBytes(mlir::Type type) {
  auto encoded = dataflow::encodeCanonicalType(type);
  if (!encoded)
    return encoded.takeError();
  return deployment::CanonicalTypeBytes(encoded->bytes().begin(),
                                        encoded->bytes().end());
}

llvm::Expected<deployment::HostProgramEntry>
deriveHostProgramEntry(const PreparedApplicationSoftware &software,
                       llvm::StringRef entrySymbol,
                       const ArtifactStore &artifacts) {
  auto structured = frontend::importStructuredProgram(
      software.compilation.structuredProgram, artifacts);
  if (!structured)
    return structured.takeError();
  auto references =
      frontend::resolveDefinedLlvmCallables(*structured, {entrySymbol});
  if (!references)
    return references.takeError();
  auto view = structured->view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(references->front());
  if (!entity)
    return entity.takeError();
  auto function =
      llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity->operation);
  if (!function)
    return invalid("application entry is not an LLVM function");
  const mlir::LLVM::LLVMFunctionType type = function.getFunctionType();
  if (type.isVarArg())
    return invalid("variadic application entries are unsupported");

  deployment::HostProgramEntry entry{0, entrySymbol.str(), {}, {}, {}};
  for (mlir::Type parameter : type.getParams()) {
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(parameter))
      return invalid(
          "pointer application entry requires System memory ingress");
    auto encoded = canonicalTypeBytes(parameter);
    if (!encoded)
      return encoded.takeError();
    entry.valueArgumentTypes.push_back(std::move(*encoded));
  }
  if (!mlir::isa<mlir::LLVM::LLVMVoidType>(type.getReturnType())) {
    auto encoded = canonicalTypeBytes(type.getReturnType());
    if (!encoded)
      return encoded.takeError();
    entry.valueResultTypes.push_back(std::move(*encoded));
  }
  return entry;
}

bool targetGroupContains(const InstructionCompilerTargetGroup &group,
                         const ArtifactIdentity &fabricIdentity,
                         fabric::AccCoreOccurrenceRef accCore) {
  return llvm::any_of(group.processors(), [&](const auto &processor) {
    return processor.artifact == fabricIdentity &&
           processor.entity.core == accCore;
  });
}

llvm::Expected<std::vector<std::vector<dataflow::RootThreadLaunchRef>>>
projectTargetGroupRoots(
    const mapping::SystemExecutionContextProjection &contexts,
    const SystemCompilerTargetBindings &targets,
    const ArtifactIdentity &fabricIdentity) {
  std::vector<std::vector<dataflow::RootThreadLaunchRef>> roots(
      targets.instructionGroups().size());
  for (const mapping::SystemInstructionContextDomain &domain :
       contexts.instructionDomains) {
    std::optional<std::size_t> selected;
    for (const auto indexed : llvm::enumerate(targets.instructionGroups())) {
      if (!targetGroupContains(indexed.value(), fabricIdentity,
                               domain.context.accCore))
        continue;
      if (selected)
        return invalid("InstructionCore belongs to multiple target groups");
      selected = indexed.index();
    }
    if (!selected)
      return invalid("SystemMapping selects an unresolved InstructionCore");
    roots[*selected].push_back(domain.root);
  }
  for (auto &groupRoots : roots) {
    llvm::sort(groupRoots, [](const auto &lhs, const auto &rhs) {
      return lhs.entity.value() < rhs.entity.value();
    });
    groupRoots.erase(std::unique(groupRoots.begin(), groupRoots.end()),
                     groupRoots.end());
  }
  return roots;
}

llvm::Expected<FinalizedInstructionCoreBinary> buildInstructionBinary(
    const llvm::Module &finalLinkedModule,
    const ArtifactRootReference &dataflowReference,
    const FinalizedCompilerTargetBinding &target,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<dataflow::RootedGraphLaunchRef> spatialInvocations,
    std::uint64_t imageBase, const CompilerTargetLinkWorkspace &workspace,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (roots.empty())
    return invalid("cannot build an InstructionCoreBinary without roots");
  auto module = detail::materializeInstructionDispatchModule(finalLinkedModule,
                                                             roots.size());
  if (!module)
    return module.takeError();
  if (llvm::Error error =
          validateModuleCompilerTarget(**module, target.binding()))
    return std::move(error);

  std::vector<ThreadEntryBinding> table;
  table.reserve(roots.size());
  for (const auto indexed : llvm::enumerate(roots)) {
    std::optional<ThreadEntrySpatialInvocationBinding> invocation;
    for (dataflow::RootedGraphLaunchRef graph : spatialInvocations) {
      if (graph.rootThreadLaunch != indexed.value())
        continue;
      if (invocation)
        return invalid("InstructionCore root has multiple invocation graphs");
      invocation = ThreadEntrySpatialInvocationBinding{graph};
    }
    table.push_back({indexed.value(), indexed.index(), std::move(invocation)});
  }
  for (dataflow::RootedGraphLaunchRef graph : spatialInvocations)
    if (llvm::none_of(table, [&](const ThreadEntryBinding &entry) {
          return entry.rootThreadLaunch == graph.rootThreadLaunch &&
                 entry.spatialInvocation.has_value();
        }))
      return invalid("InstructionCore invocation graph has no selected root");
  auto object = emitCompilerTargetObject(std::move(*module), target.binding());
  if (!object)
    return object.takeError();
  auto executable = linkCompilerTargetExecutable(
      *object, target.binding(), "__loom_thread_entry_0", imageBase, workspace);
  if (!executable)
    return executable.takeError();
  return finalizeInstructionCoreBinary({dataflowReference,
                                        target.reference(),
                                        std::move(*executable),
                                        std::move(table),
                                        {}},
                                       artifacts, blobs);
}

mlir::DialectRegistry applicationDialectRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return registry;
}

llvm::Expected<SourceSimulationInputs>
makeSourceSimulationInputs(const frontend::StructuredProgramCandidate &program,
                           ApplicationSourceInvocation invocation) {
  if (invocation.entrySymbol.empty())
    return invalid("source invocation requires an ABI entry symbol");
  auto entries = frontend::resolveDefinedLlvmCallables(
      program, {llvm::StringRef(invocation.entrySymbol)});
  if (!entries)
    return entries.takeError();
  if (entries->size() != 1)
    return invalid("source invocation entry does not resolve uniquely");

  sim::StructuredProgramSimulationWorkload workloadDraft{entries->front()};
  workloadDraft.argumentPlan = std::move(invocation.argumentPlan);
  workloadDraft.observableContract.returnValue = invocation.observeReturnValue;
  for (const ApplicationPointerMemoryObservable &observable :
       invocation.memoryObservables) {
    workloadDraft.observableContract.memories.push_back(
        {sim::EntryPointerArgumentTarget{observable.argumentOrdinal},
         observable.form});
  }

  auto view = program.view();
  if (!view)
    return view.takeError();
  auto workload = sim::finalizeSimulationWorkload(workloadDraft, *view);
  if (!workload)
    return workload.takeError();

  sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload->identity()};
  runtimeDraft.runtimeValues = std::move(invocation.runtimeValues);
  runtimeDraft.memoryObjects = std::move(invocation.memoryObjects);
  runtimeDraft.pointerBindings = std::move(invocation.pointerBindings);
  auto runtimeInput =
      sim::finalizeSimulationRuntimeInput(runtimeDraft, *workload, *view);
  if (!runtimeInput)
    return runtimeInput.takeError();
  return SourceSimulationInputs{std::move(*workload), std::move(*runtimeInput)};
}

llvm::Expected<std::variant<std::vector<ArtifactRootReference>,
                            UnsupportedApplicationBuild>>
publishApplicationWorkloads(
    const frontend::PublishedPreMappingCompilation &published,
    const dataflow::CanonicalDataflowArtifact &canonical,
    llvm::StringRef entrySymbol, const ArtifactStore &artifacts) {
  auto view = canonical.view();
  if (!view)
    return view.takeError();
  auto roots =
      view->projectRootThreadLaunchesReachableFromAbiEntry(entrySymbol);
  if (!roots)
    return roots.takeError();

  std::vector<ArtifactRootReference> workloads;
  for (dataflow::RootThreadLaunchRef root : *roots) {
    auto invocationPaths =
        view->projectRootThreadInvocationPathsFromAbiEntry(entrySymbol, root);
    if (!invocationPaths)
      return invocationPaths.takeError();
    if (llvm::any_of(*invocationPaths,
                     [](const auto &path) { return path.calls.empty(); }))
      return std::variant<std::vector<ArtifactRootReference>,
                          UnsupportedApplicationBuild>{
          UnsupportedApplicationBuild{
              ApplicationBuildUnsupportedKind::DirectInvocationBoundary,
              published.canonicalDataflow, root}};
    llvm::Error workloadError = llvm::Error::success();
    bool unsupportedCoordinates = false;
    view->forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
      if (workloadError || unsupportedCoordinates ||
          launch.rootThreadLaunch != root)
        return;
      auto coordinates = view->enumerateStaticDenseCoordinates(
          launch, runtime::gem5MaximumDynamicSpatialInvocations, entrySymbol);
      if (!coordinates) {
        workloadError = coordinates.takeError();
        return;
      }
      if (!*coordinates) {
        unsupportedCoordinates = true;
        return;
      }
      auto shapes = sim::projectSpatialSimulationBoundaryShapes(*view, launch);
      if (!shapes) {
        workloadError = shapes.takeError();
        return;
      }
      auto writableRoots =
          sim::projectSpatialInvocationWritableMemoryRoots(*view, launch);
      if (!writableRoots) {
        workloadError = writableRoots.takeError();
        return;
      }
      for (const std::vector<std::uint64_t> &point : **coordinates) {
        sim::SpatialSimulationWorkload workloadDraft{launch};
        workloadDraft.denseCoordinates = point;
        workloadDraft.valueInputPlan.assign(shapes->valueInputs.size(),
                                            sim::RuntimeValueInput{});
        workloadDraft.observableContract.valueResults.resize(
            shapes->valueResults.size());
        std::iota(workloadDraft.observableContract.valueResults.begin(),
                  workloadDraft.observableContract.valueResults.end(), 0);
        for (dataflow::LogicalMemoryRootRef memory : *writableRoots)
          workloadDraft.observableContract.memories.push_back(
              {dataflow::LogicalMemoryRootOrViewRef{memory},
               sim::MemoryObservationForm::DiffFromRuntimeInput});
        auto workload = sim::finalizeSimulationWorkload(workloadDraft, *view);
        if (!workload) {
          workloadError = workload.takeError();
          return;
        }
        auto reference = sim::publishSimulationWorkload(*workload, artifacts);
        if (!reference) {
          workloadError = reference.takeError();
          return;
        }
        workloads.push_back(std::move(*reference));
      }
    });
    if (workloadError)
      return std::move(workloadError);
    if (unsupportedCoordinates)
      return std::variant<std::vector<ArtifactRootReference>,
                          UnsupportedApplicationBuild>{
          UnsupportedApplicationBuild{
              ApplicationBuildUnsupportedKind::RootCoordinates,
              published.canonicalDataflow, root}};
  }
  llvm::sort(workloads, artifactRootReferenceLess);
  workloads.erase(std::unique(workloads.begin(), workloads.end()),
                  workloads.end());
  if (workloads.empty())
    return invalid("source entry reaches no Spatial workload");
  return std::variant<std::vector<ArtifactRootReference>,
                      UnsupportedApplicationBuild>{std::move(workloads)};
}

} // namespace

llvm::Expected<ApplicationBuildPreparationOutcome> prepareApplicationBuild(
    const llvm::Module &finalLinkedModule, ApplicationBuildRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::ApplicationPreparation);
  if (llvm::Error error = dse::registerProductionDseOwners())
    return std::move(error);
  auto system = fabric::importEntireFabricRoot(request.system, artifacts);
  if (!system)
    return system.takeError();
  auto systemView = fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();

  auto source = frontend::raiseLlvmModuleToStructured(
      llvm::CloneModule(finalLinkedModule), *system,
      request.compilationOptions.raising);
  if (!source)
    return source.takeError();
  if (!request.operatorProtocolSymbols.empty()) {
    if (!request.preMappingOptions.ownership.protocolCallableRoots.empty())
      return invalid("operator protocol has two competing declarations");
    llvm::SmallVector<llvm::StringRef> symbols;
    symbols.reserve(request.operatorProtocolSymbols.size());
    for (const std::string &symbol : request.operatorProtocolSymbols)
      symbols.push_back(symbol);
    auto roots = frontend::resolveDefinedLlvmCallables(
        source->structuredProgram, symbols);
    if (!roots)
      return roots.takeError();
    request.preMappingOptions.ownership.protocolCallableRoots =
        std::move(*roots);
  }
  auto sourceInputs = makeSourceSimulationInputs(source->structuredProgram,
                                                 request.sourceInvocation);
  if (!sourceInputs)
    return sourceInputs.takeError();

  auto preMapping = dse::exploreStructuredCompilationToPreMapping(
      std::move(*source), sourceInputs->workload, sourceInputs->runtimeInput,
      *system, request.resolvedConfig, request.preMappingOptions, artifacts,
      blobs);
  if (!preMapping)
    return preMapping.takeError();
  if (auto *incomplete =
          std::get_if<dse::IncompletePreMappingExploration>(&*preMapping)) {
    emitApplicationPreMappingIncompleteDiagnostics(*incomplete);
    return ApplicationBuildPreparationOutcome{std::move(*incomplete)};
  }
  if (auto *noFeasible =
          std::get_if<dse::CompletedPreMappingNoFeasibleCandidate>(
              &*preMapping))
    return ApplicationBuildPreparationOutcome{std::move(*noFeasible)};

  auto completed =
      std::get<dse::CompletedPreMappingSelection>(std::move(*preMapping));
  if (completed.selected.empty())
    return invalid("completed pre-Mapping selection is empty");
  for (std::size_t index = 0; index != completed.selected.size(); ++index)
    if (completed.selected[index].preferenceRank != index)
      return invalid("pre-Mapping software preference ranks are not dense");
  if (completed.selected.size() > request.jointPolicy.maximumSoftwareFrontier())
    return invalid("pre-Mapping software frontier exceeds its joint bound");
  if (completed.selected.size() > request.jointPolicy.maximumPairEvaluations())
    return invalid("pre-Mapping alternatives exceed the pair-evaluation "
                   "bound");

  struct PendingResourceTimeCandidate final {
    dse::SelectedPreMappingCompilation compilation;
    std::size_t planningRecordOrdinal = 0;
    ComponentViewDigest candidateIdentity;
    std::shared_ptr<const dse::ResourceTimeDataflowProjection> projection;
    std::uint64_t inputPreferenceRank = 0;
  };

  // The projection-only pass is deliberately limited to the already
  // materialized Canonical Dataflow view and the provider-owned resource-time
  // projection. Workload publication and joint-plan construction belong after
  // the bounded resource-time funnel so rejected estimates cannot trigger
  // Mapping work.
  std::vector<PendingResourceTimeCandidate> pendingCandidates;
  std::vector<dse::ResourceTimeMappingCandidateInput> resourceTimeInputs;
  pendingCandidates.reserve(completed.selected.size());
  resourceTimeInputs.reserve(completed.selected.size());
  std::uint64_t resourceTimeProjectionRequests = 0;
  std::uint64_t resourceTimeProjectionElapsedNanoseconds = 0;
  std::uint64_t resourceTimeProjectionCacheHits = 0;
  std::uint64_t resourceTimeProjectionCacheMisses = 0;
  std::uint64_t resourceTimeProjectionCacheCapacityBypasses = 0;
  std::uint64_t resourceTimeProjectionCacheRetainedBytes = 0;
  std::map<std::string,
           std::pair<std::shared_ptr<const dse::ResourceTimeDataflowProjection>,
                     std::uint64_t>>
      resourceTimeProjectionCache;
  auto resourceTimeModelSnapshot =
      dse::resourceTimeAnalyticModelSnapshotDigest();
  if (!resourceTimeModelSnapshot)
    return resourceTimeModelSnapshot.takeError();
  auto resourceTimeConfig =
      dse::projectResolvedDseConfigView(request.resolvedConfig);
  if (!resourceTimeConfig)
    return resourceTimeConfig.takeError();
  auto alternativePolicy = dse::JointDesignPolicy::get(
      1, 1, 1, request.jointPolicy.maximumTechMappingsPerModule(),
      request.jointPolicy.maximumSpatialMappingsPerPair());
  if (!alternativePolicy)
    return alternativePolicy.takeError();
  for (dse::SelectedPreMappingCompilation &selected : completed.selected) {
    if (!selected.planningRecordOrdinal ||
        *selected.planningRecordOrdinal >= completed.candidateInventory.size())
      return invalid("selected software has no exact planning record");
    const std::size_t planningRecordOrdinal = *selected.planningRecordOrdinal;
    const dse::PreMappingCandidatePlanningRecord &planningRecord =
        completed.candidateInventory[planningRecordOrdinal];
    if (!planningRecord.structuredProgram)
      return invalid("selected software has no Structured lineage root");
    auto candidateIdentity = dse::computePreMappingCandidateIdentity(
        planningRecord, completed.sourceProgram, completed.fabric,
        completed.workload, completed.runtimeInput,
        completed.frontierPolicyDigest);
    if (!candidateIdentity)
      return candidateIdentity.takeError();
    if (!planningRecord.candidateIdentity ||
        *planningRecord.candidateIdentity != *candidateIdentity)
      return invalid("pre-Mapping candidate identity failed its application "
                     "join validation");
    if (!planningRecord.canonicalDataflow ||
        planningRecord.canonicalDataflow->artifact !=
            selected.compilation.canonicalDataflow.identity())
      return invalid("selected software and planning Dataflow disagree");
    auto dataflowView = selected.compilation.canonicalDataflow.view();
    if (!dataflowView)
      return dataflowView.takeError();
    const ArtifactRootReference dataflow{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        selected.compilation.canonicalDataflow.identity()};
    dse::ResourceTimeInvocationKey invocation{
        *planningRecord.structuredProgram,
        dataflow,
        request.system,
        completed.workload,
        completed.runtimeInput,
        resourceTimeConfig->digest(),
        *resourceTimeModelSnapshot,
        request.sourceInvocation.entrySymbol,
        planningRecord.estimatedRuntimePicoseconds};
    auto projectionKey = dse::deriveResourceTimeProjectionCacheKey(invocation);
    if (!projectionKey)
      return projectionKey.takeError();
    const std::string projectionKeySpelling =
        formatComponentViewDigestHex(*projectionKey);
    ++resourceTimeProjectionRequests;
    std::shared_ptr<const dse::ResourceTimeDataflowProjection>
        resourceTimeProjection;
    auto cachedProjection =
        resourceTimeProjectionCache.find(projectionKeySpelling);
    if (cachedProjection != resourceTimeProjectionCache.end()) {
      resourceTimeProjection = cachedProjection->second.first;
      ++resourceTimeProjectionCacheHits;
    } else {
      ++resourceTimeProjectionCacheMisses;
      const MonotonicClock::time_point projectionBegin = MonotonicClock::now();
      auto computedProjection = dse::projectResourceTimeDataflow(
          *dataflowView, *systemView, request.sourceInvocation.entrySymbol,
          planningRecord.estimatedRuntimePicoseconds);
      const std::uint64_t projectionElapsed =
          elapsedNanoseconds(projectionBegin);
      resourceTimeProjectionElapsedNanoseconds =
          projectionElapsed > std::numeric_limits<std::uint64_t>::max() -
                                  resourceTimeProjectionElapsedNanoseconds
              ? std::numeric_limits<std::uint64_t>::max()
              : resourceTimeProjectionElapsedNanoseconds + projectionElapsed;
      if (!computedProjection)
        return computedProjection.takeError();
      resourceTimeProjection =
          std::make_shared<const dse::ResourceTimeDataflowProjection>(
              std::move(*computedProjection));
      const std::uint64_t retainedBytes =
          dse::resourceTimeProjectionRetainedBytes(*resourceTimeProjection);
      const bool fitsEntryLimit =
          resourceTimeProjectionCache.size() <
          request.resourceTimePolicy.maximumInvocationMemoEntries;
      const std::uint64_t availableBytes =
          request.resourceTimePolicy.maximumInvocationMemoBytes >=
                  resourceTimeProjectionCacheRetainedBytes
              ? request.resourceTimePolicy.maximumInvocationMemoBytes -
                    resourceTimeProjectionCacheRetainedBytes
              : 0;
      if (fitsEntryLimit && retainedBytes <= availableBytes) {
        resourceTimeProjectionCache.emplace(
            projectionKeySpelling,
            std::make_pair(resourceTimeProjection, retainedBytes));
        resourceTimeProjectionCacheRetainedBytes += retainedBytes;
      } else {
        ++resourceTimeProjectionCacheCapacityBypasses;
      }
    }
    if (request.resourceTimePolicy.availableResourceUnits.empty())
      request.resourceTimePolicy.availableResourceUnits =
          resourceTimeProjection->availableResourceUnits;
    else if (request.resourceTimePolicy.availableResourceUnits !=
             resourceTimeProjection->availableResourceUnits)
      return invalid("resource-time policy capacity disagrees with the exact "
                     "System projection");
    const auto maximumResourceBound =
        llvm::max_element(resourceTimeProjection->regionBounds,
                          [](const auto &lhs, const auto &rhs) {
                            return lhs.maximumUsefulResourceUnits <
                                   rhs.maximumUsefulResourceUnits;
                          });
    if (maximumResourceBound == resourceTimeProjection->regionBounds.end())
      return invalid("resource-time projection has no region bound");
    const std::uint64_t maximumUsefulResourceUnits =
        maximumResourceBound->maximumUsefulResourceUnits;
    const std::uint64_t inputPreferenceRank = selected.preferenceRank;
    pendingCandidates.push_back(
        {std::move(selected), planningRecordOrdinal, *candidateIdentity,
         std::move(resourceTimeProjection), inputPreferenceRank});
    const PendingResourceTimeCandidate &pending = pendingCandidates.back();
    resourceTimeInputs.push_back(
        {*candidateIdentity, pending.inputPreferenceRank,
         planningRecord.ownedProtocolRoots.size(),
         pending.projection->acceleratedGraphCount,
         pending.projection->acceleratedActorCount, maximumUsefulResourceUnits,
         std::move(invocation), pending.projection->resourceClasses,
         pending.projection->regions});
  }
  auto resourceTimeFunnel = dse::selectResourceTimeMappingFinalists(
      resourceTimeInputs, request.resourceTimePolicy,
      request.preMappingOptions.executionControl);
  if (!resourceTimeFunnel)
    return resourceTimeFunnel.takeError();
  resourceTimeFunnel->accounting.dataflowProjectionRequests =
      resourceTimeProjectionRequests;
  resourceTimeFunnel->accounting.dataflowProjectionCacheHits =
      resourceTimeProjectionCacheHits;
  resourceTimeFunnel->accounting.dataflowProjectionCacheMisses =
      resourceTimeProjectionCacheMisses;
  resourceTimeFunnel->accounting.dataflowProjectionCacheCapacityBypasses =
      resourceTimeProjectionCacheCapacityBypasses;
  resourceTimeFunnel->accounting.dataflowProjectionCacheEntries =
      resourceTimeProjectionCache.size();
  resourceTimeFunnel->accounting.dataflowProjectionCacheRetainedBytes =
      resourceTimeProjectionCacheRetainedBytes;
  resourceTimeFunnel->accounting.dataflowProjectionElapsedNanoseconds =
      resourceTimeProjectionElapsedNanoseconds;
  if (llvm::Error error = dse::validateResourceTimeMappingFunnelAccounting(
          resourceTimeFunnel->accounting))
    return std::move(error);
  const auto emitResourceTimeFunnelTerminal = [&](llvm::StringRef status) {
    const auto &accounting = resourceTimeFunnel->accounting;
    const auto counterObject = [](const dse::ResourceTimeWorkCounter &counter) {
      return llvm::json::Object{
          {"limit", counter.limit},
          {"planned", counter.planned},
          {"reserved", counter.reserved},
          {"consumed", counter.consumed},
          {"rejected", counter.rejected},
          {"cancelled", counter.cancelled},
          {"elapsed_nanoseconds", counter.elapsedNanoseconds}};
    };
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "resource_time_application_funnel";
          fields["status"] = status;
          llvm::json::Object frontierWork{
              {"source_projections",
               counterObject(accounting.frontierAccounting.sourceProjections)},
              {"actions", counterObject(accounting.frontierAccounting.actions)},
              {"states", counterObject(accounting.frontierAccounting.states)},
              {"estimates",
               counterObject(accounting.frontierAccounting.estimates)},
              {"finalists",
               counterObject(accounting.frontierAccounting.finalists)},
              {"state_memo_hits", accounting.frontierAccounting.stateMemoHits},
              {"state_memo_misses",
               accounting.frontierAccounting.stateMemoMisses},
              {"state_memo_envelope_updates",
               accounting.frontierAccounting.stateMemoEnvelopeUpdates},
              {"state_memo_dominated_states",
               accounting.frontierAccounting.stateMemoDominatedStates},
              {"states_pruned_by_beam",
               accounting.frontierAccounting.statesPrunedByBeam},
              {"incremental_lower_bound_updates",
               accounting.frontierAccounting.incrementalLowerBoundUpdates},
              {"maximum_retained_bytes",
               accounting.frontierAccounting.maximumRetainedBytes}};
          llvm::json::Object funnel{
              {"generated_candidates", accounting.generatedCandidates},
              {"screened_candidates", accounting.screenedCandidates},
              {"detailed_frontier_candidates",
               accounting.detailedFrontierCandidates},
              {"successive_halving_deferred_candidates",
               accounting.successiveHalvingDeferredCandidates},
              {"sound_gate_rejected_candidates",
               accounting.soundGateRejectedCandidates},
              {"estimated_candidates", accounting.estimatedCandidates},
              {"incomplete_candidates", accounting.incompleteCandidates},
              {"mapping_finalists", accounting.mappingFinalists},
              {"dataflow_projection_requests",
               accounting.dataflowProjectionRequests},
              {"dataflow_projection_cache_hits",
               accounting.dataflowProjectionCacheHits},
              {"dataflow_projection_cache_misses",
               accounting.dataflowProjectionCacheMisses},
              {"dataflow_projection_cache_capacity_bypasses",
               accounting.dataflowProjectionCacheCapacityBypasses},
              {"dataflow_projection_cache_entries",
               accounting.dataflowProjectionCacheEntries},
              {"dataflow_projection_cache_retained_bytes",
               accounting.dataflowProjectionCacheRetainedBytes},
              {"dataflow_projection_elapsed_nanoseconds",
               accounting.dataflowProjectionElapsedNanoseconds},
              {"dataflow_materialized_candidates",
               accounting.dataflowMaterializedCandidates},
              {"mapping_plan_candidates", accounting.mappingPlanCandidates},
              {"unsupported_before_mapping_candidates",
               accounting.unsupportedBeforeMappingCandidates},
              {"mapping_calls_deferred_by_model",
               accounting.mappingCallsDeferredByModel},
              {"mapping_calls_avoided_by_sound_gate",
               accounting.mappingCallsAvoidedBySoundGate},
              {"mapping_calls_withheld_by_incomplete",
               accounting.mappingCallsWithheldByIncomplete},
              {"exact_invocation_memo_hits",
               accounting.exactInvocationMemoHits},
              {"exact_invocation_memo_misses",
               accounting.exactInvocationMemoMisses},
              {"exact_invocation_memo_single_flight_waits",
               accounting.exactInvocationMemoSingleFlightWaits},
              {"exact_invocation_memo_coalesced_uncached_results",
               accounting.exactInvocationMemoCoalescedUncachedResults},
              {"exact_invocation_memo_cancelled_waits",
               accounting.exactInvocationMemoCancelledWaits},
              {"exact_invocation_memo_capacity_bypasses",
               accounting.exactInvocationMemoCapacityBypasses},
              {"exact_invocation_memo_entries",
               accounting.exactInvocationMemoEntries},
              {"exact_invocation_memo_retained_bytes",
               accounting.exactInvocationMemoRetainedBytes},
              {"frontier_work", std::move(frontierWork)},
              {"elapsed_nanoseconds", accounting.elapsedNanoseconds},
              {"truncated", resourceTimeFunnel->truncated}};
          if (resourceTimeFunnel->incompleteReason)
            funnel["incomplete_reason"] =
                dse::resourceTimeFrontierIncompleteReasonSpelling(
                    *resourceTimeFunnel->incompleteReason);
          fields["resource_time_funnel"] = std::move(funnel);
        });
  };
  if (resourceTimeFunnel->incompleteReason ==
      dse::ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)
    emitResourceTimeFunnelTerminal("cancelled_or_timeout");
  if (resourceTimeFunnel->incompleteReason ==
      dse::ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)
    return ApplicationBuildPreparationOutcome{
        IncompleteApplicationResourceTimePlanning{
            *resourceTimeFunnel->incompleteReason,
            std::move(*resourceTimeFunnel),
            std::move(completed.candidateInventory), completed.sourceProgram,
            completed.fabric, completed.workload, completed.runtimeInput,
            completed.frontierPolicyDigest}};
  if (resourceTimeFunnel->preferenceOrder.empty())
    emitResourceTimeFunnelTerminal(resourceTimeFunnel->incompleteReason
                                       ? "incomplete"
                                       : "no_mapping_finalist");
  if (resourceTimeFunnel->preferenceOrder.empty() &&
      resourceTimeFunnel->incompleteReason)
    return ApplicationBuildPreparationOutcome{
        IncompleteApplicationResourceTimePlanning{
            *resourceTimeFunnel->incompleteReason,
            std::move(*resourceTimeFunnel),
            std::move(completed.candidateInventory), completed.sourceProgram,
            completed.fabric, completed.workload, completed.runtimeInput,
            completed.frontierPolicyDigest}};
  if (resourceTimeFunnel->preferenceOrder.empty())
    return ApplicationBuildPreparationOutcome{
        dse::CompletedPreMappingNoFeasibleCandidate{
            std::move(completed.satisfiedEvidence),
            std::move(completed.planGenerateInvocations)}};

  std::vector<PreparedApplicationSoftware> preparedSoftware;
  std::vector<PreparedApplicationMappingAlternative> mappingAlternatives;
  std::optional<UnsupportedApplicationBuild> firstUnsupported;
  preparedSoftware.reserve(resourceTimeFunnel->preferenceOrder.size());
  mappingAlternatives.reserve(resourceTimeFunnel->preferenceOrder.size());
  std::vector<ComponentViewDigest> promotedIdentities;
  promotedIdentities.reserve(resourceTimeFunnel->preferenceOrder.size());
  for (const ComponentViewDigest &identity :
       resourceTimeFunnel->preferenceOrder) {
    auto pending = llvm::find_if(
        pendingCandidates, [&](const PendingResourceTimeCandidate &candidate) {
          return candidate.candidateIdentity == identity;
        });
    if (pending == pendingCandidates.end())
      return invalid("resource-time finalist has no application candidate");
    const auto evaluation = llvm::find_if(
        resourceTimeFunnel->evaluations, [&](const auto &candidate) {
          return candidate.candidateIdentity == identity;
        });
    if (evaluation == resourceTimeFunnel->evaluations.end())
      return invalid("resource-time finalist has no funnel evaluation");
    auto published = frontend::publishPreMappingCompilation(
        pending->compilation.compilation, artifacts);
    if (!published)
      return published.takeError();
    auto workloads = publishApplicationWorkloads(
        *published, pending->compilation.compilation.canonicalDataflow,
        request.sourceInvocation.entrySymbol, artifacts);
    if (!workloads)
      return workloads.takeError();
    if (auto *unsupported =
            std::get_if<UnsupportedApplicationBuild>(&*workloads)) {
      if (!firstUnsupported)
        firstUnsupported = std::move(*unsupported);
      auto &record =
          completed.candidateInventory[pending->planningRecordOrdinal];
      record.disposition =
          dse::PreMappingCandidatePlanningDisposition::Unsupported;
      ++resourceTimeFunnel->accounting.unsupportedBeforeMappingCandidates;
      continue;
    }
    auto roots =
        std::get<std::vector<ArtifactRootReference>>(std::move(*workloads));
    ++resourceTimeFunnel->accounting.dataflowMaterializedCandidates;
    if (pending->compilation.functionalReplay)
      ++resourceTimeFunnel->accounting.functionalReplayCandidates;
    // Deployment reconstructs this exact invocation plan again. Validate it
    // before any Tech/Spatial/System provider is dispatched so a candidate
    // with an inexact dynamic capture becomes a typed unsupported finalist,
    // rather than a late deployment failure after expensive Mapping work.
    auto invocationDataflow =
        pending->compilation.compilation.canonicalDataflow.view();
    if (!invocationDataflow)
      return invocationDataflow.takeError();
    auto invocationPreflight = detail::deriveApplicationSpatialInvocationPlan(
        *invocationDataflow, request.sourceInvocation.entrySymbol);
    if (!invocationPreflight) {
      const std::string diagnostic =
          llvm::toString(invocationPreflight.takeError());
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "application_resource_time_preflight";
            fields["operation"] = "resource_time_application_preflight";
            fields["disposition"] = "unsupported";
            fields["diagnostic"] = diagnostic;
            fields["candidate_identity"] =
                formatComponentViewDigestHex(identity);
          });
      if (!firstUnsupported)
        firstUnsupported = UnsupportedApplicationBuild{
            ApplicationBuildUnsupportedKind::DynamicInvocationBoundary,
            published->canonicalDataflow,
            pending->projection->regions.front().region};
      auto &record =
          completed.candidateInventory[pending->planningRecordOrdinal];
      record.disposition =
          dse::PreMappingCandidatePlanningDisposition::Unsupported;
      ++resourceTimeFunnel->accounting.unsupportedBeforeMappingCandidates;
      continue;
    }
    std::optional<std::vector<pnr::SystemBindingPartitionIntent>> partitions;
    if (evaluation->bestHint) {
      auto derivedPartitions =
          deriveSystemBindingPartitionIntent(*evaluation->bestHint);
      if (!derivedPartitions)
        return derivedPartitions.takeError();
      partitions = std::move(*derivedPartitions);
    }
    auto mappingPlan = dse::buildJointDesignExplorationPlan(
        {{roots}, {request.system}}, request.physicalTimingProfiles,
        *alternativePolicy, request.resolvedConfig, artifacts, nullptr,
        partitions
            ? llvm::ArrayRef<pnr::SystemBindingPartitionIntent>(*partitions)
            : llvm::ArrayRef<pnr::SystemBindingPartitionIntent>());
    if (!mappingPlan)
      return mappingPlan.takeError();
    ++resourceTimeFunnel->accounting.mappingPlanCandidates;
    const std::uint64_t rank = mappingAlternatives.size();
    auto &record = completed.candidateInventory[pending->planningRecordOrdinal];
    record.disposition = dse::PreMappingCandidatePlanningDisposition::Retained;
    record.preferenceRank = rank;
    promotedIdentities.push_back(identity);
    std::vector<sim::SourceBackedDfgReplayCaseReference> replayCases;
    if (pending->compilation.functionalReplay)
      replayCases = pending->compilation.functionalReplay->replayCases;
    preparedSoftware.push_back({rank, pending->planningRecordOrdinal, identity,
                                std::move(*published), std::move(roots),
                                std::move(replayCases)});
    const ArtifactRootReference dataflow =
        preparedSoftware.back().compilation.canonicalDataflow;
    mappingAlternatives.push_back(
        {rank, pending->planningRecordOrdinal, identity, dataflow,
         pending->projection->regions, pending->projection->regionBounds,
         std::move(*mappingPlan)});
  }
  if (mappingAlternatives.empty()) {
    emitResourceTimeFunnelTerminal("all_finalists_rejected_before_mapping");
    if (firstUnsupported)
      return ApplicationBuildPreparationOutcome{std::move(*firstUnsupported)};
    return ApplicationBuildPreparationOutcome{
        dse::CompletedPreMappingNoFeasibleCandidate{
            std::move(completed.satisfiedEvidence),
            std::move(completed.planGenerateInvocations)}};
  }
  if (mappingAlternatives.size() != preparedSoftware.size())
    return invalid("resource-time promotion lost its application join");
  for (dse::PreMappingCandidatePlanningRecord &record :
       completed.candidateInventory) {
    if (!record.candidateIdentity ||
        llvm::is_contained(promotedIdentities, *record.candidateIdentity))
      continue;
    if (record.disposition ==
        dse::PreMappingCandidatePlanningDisposition::Retained) {
      record.disposition =
          dse::PreMappingCandidatePlanningDisposition::HeuristicPruned;
      record.preferenceRank.reset();
    }
  }
  PreparedApplicationBuild prepared{
      std::move(request.sourceInvocation),
      request.jointPolicy,
      std::move(preparedSoftware),
      std::move(completed.satisfiedEvidence),
      std::move(completed.planGenerateInvocations),
      std::move(completed.protocolDependencyProjection),
      std::move(completed.candidateInventory),
      completed.frontierPolicy,
      completed.eligibleCoordinateCount,
      completed.coordinateFrontierTruncated,
      std::move(completed.frontierAccounting),
      completed.evaluationTiming,
      completed.sharedEvaluationStatistics,
      completed.evaluationCacheStatistics,
      std::move(completed.retainedPlanIncompleteness),
      std::move(mappingAlternatives),
      request.resourceTimePolicy,
      std::move(*resourceTimeFunnel),
      completed.requestedPlannerMode,
      completed.resolvedPlannerMode,
      completed.completeness,
      std::move(completed.shadowRecall),
      completed.sourceProgram,
      completed.fabric,
      completed.workload,
      completed.runtimeInput,
      completed.frontierPolicyDigest};
  emitApplicationPlanningDiagnostics(prepared);
  return ApplicationBuildPreparationOutcome{std::move(prepared)};
}

llvm::Expected<ApplicationMappingExecution>
executeApplicationMapping(const PreparedApplicationBuild &prepared,
                          ApplicationMappingExecutionRequest request,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::MappingExecution);
  if (prepared.mappingAlternatives.empty())
    return invalid("Mapping execution has no software alternative");
  pnr::PnrDerivedContextSession pnrDerivedContextSession;
  llvm::scope_exit emitPnrDerivedContextSession([&] {
    const pnr::PnrDerivedContextSessionStatistics statistics =
        pnrDerivedContextSession.statistics();
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "application_pnr_derived_context_session";
          fields["requests"] = statistics.requests;
          fields["cache_hits"] = statistics.cacheHits;
          fields["cache_misses"] = statistics.cacheMisses;
          fields["coalesced_waits"] = statistics.coalescedWaits;
          fields["revalidation_count"] = statistics.revalidationCount;
          fields["unique_constructions"] = statistics.uniqueConstructions;
          fields["uncached_constructions"] = statistics.uncachedConstructions;
          fields["construction_time_ns"] = statistics.constructionNanoseconds;
          fields["construction_time_saved_ns"] =
              statistics.constructionNanosecondsSaved;
          fields["deterministic_work"] = statistics.deterministicWork;
          fields["retained_bytes"] = statistics.retainedBytes;
          fields["retained_bytes_reused"] = statistics.retainedBytesReused;
          fields["entry_count"] = statistics.entryCount;
          fields["entry_limit"] = statistics.entryLimit;
        });
  });
  std::vector<ArtifactRootReference> evidence = prepared.satisfiedEvidence;
  evidence.insert(evidence.end(), request.preexistingEvidence.begin(),
                  request.preexistingEvidence.end());
  std::vector<const dse::JointDesignExplorationPlan *> plans;
  plans.reserve(prepared.mappingAlternatives.size());
  for (const PreparedApplicationMappingAlternative &alternative :
       prepared.mappingAlternatives)
    plans.push_back(&alternative.plan);
  std::vector<ApplicationMappingCandidateOutcome> outcomes;
  std::vector<dse::JointDesignAttemptRecord> attempts;
  std::uint64_t attemptedSoftwarePlans = 0;
  std::uint64_t hardwareReopenSearches = 0;
  std::uint64_t hardwareParentPromotions = 0;
  std::uint64_t hardwareReopensDeferredByQuality = 0;
  std::uint64_t hardwareReopensWithheldWithoutExactFeedback = 0;
  std::uint64_t hardwareRepairProbeLimit = 0;
  std::uint64_t hardwareRepairProbesPlanned = 0;
  std::uint64_t hardwareRepairProbesReserved = 0;
  std::uint64_t hardwareRepairProbesConsumed = 0;
  std::uint64_t hardwareRepairProbesRejected = 0;
  std::uint64_t hardwareRepairProbesCancelled = 0;
  std::uint64_t parentTechDecisions = 0;
  std::uint64_t parentSpatialDecisions = 0;
  std::uint64_t preservedTechDecisions = 0;
  std::uint64_t preservedSpatialDecisions = 0;
  std::uint64_t reopenedTechDecisions = 0;
  std::uint64_t reopenedSpatialDecisions = 0;
  std::uint64_t repairedTechDecisions = 0;
  std::uint64_t repairedSpatialDecisions = 0;
  std::uint64_t invalidationRootCount = 0;
  std::uint64_t invalidationConeDecisionCount = 0;
  std::uint64_t parentRouteNodeCount = 0;
  std::uint64_t preservedRouteNodeCount = 0;
  std::uint64_t reopenedRouteNodeCount = 0;
  std::uint64_t repairedRouteNodeCount = 0;
  std::uint64_t parentServiceLegCount = 0;
  std::uint64_t preservedServiceLegCount = 0;
  std::uint64_t reopenedServiceLegCount = 0;
  std::uint64_t verifiedAlternatives = 0;
  std::uint64_t techMappingDispatches = 0;
  std::uint64_t spatialPnrDispatches = 0;
  std::uint64_t systemPnrDispatches = 0;
  const std::size_t mappingImportEntryLimit =
      prepared.mappingAlternatives.size() >
              std::numeric_limits<std::size_t>::max() / 16
          ? std::numeric_limits<std::size_t>::max()
          : std::max<std::size_t>(1, prepared.mappingAlternatives.size() * 16);
  // Keep one invocation-local immutable import session across all finalist
  // joins. Nested Spectrum verification reuses it but still runs its own
  // active-set and endpoint verifier.
  mapping::SystemMappingImportSession mappingImportSession(
      artifacts, mappingImportEntryLimit);

  const auto appendOutcomes = [&](const dse::JointDesignExecution &execution,
                                  std::size_t planOrdinalBase) -> llvm::Error {
    for (const dse::JointDesignAttemptRecord &attempt :
         execution.summary.attempts) {
      if (attempt.planOrdinal >
          std::numeric_limits<std::uint64_t>::max() - planOrdinalBase)
        return invalid("joint Mapping plan ordinal overflowed");
      const std::uint64_t planOrdinal = attempt.planOrdinal + planOrdinalBase;
      if (planOrdinal >= prepared.mappingAlternatives.size())
        return invalid("joint Mapping outcome has a foreign plan ordinal");
      const PreparedApplicationMappingAlternative &alternative =
          prepared.mappingAlternatives[planOrdinal];
      for (const ArtifactRootReference &mappingReference :
           attempt.systemMappings) {
        auto mapping =
            mapping::importSystemMapping(mappingReference, artifacts);
        if (!mapping)
          return mapping.takeError();
        if (mapping->view().dataflowIdentity() !=
                alternative.dataflow.artifact ||
            mapping->view().fabricIdentity() != attempt.system.artifact)
          return invalid("joint Mapping outcome disagrees with its exact "
                         "software/System owners");
      }
      if (alternative.preMappingCandidateRecordOrdinal >=
          prepared.candidateInventory.size())
        return invalid("Mapping outcome has a foreign planning-record ordinal");
      std::optional<dse::ResourceTimeSpectrumFunnelResult> resourceTimeSpectrum;
      if (!attempt.systemMappings.empty()) {
        const auto evaluation =
            llvm::find_if(prepared.resourceTimeFunnel.evaluations,
                          [&](const auto &candidate) {
                            return candidate.candidateIdentity ==
                                   alternative.candidateIdentity;
                          });
        if (evaluation == prepared.resourceTimeFunnel.evaluations.end())
          return invalid("Mapping outcome has no resource-time evaluation");
        if (!evaluation->retainedHints.empty()) {
          auto verified = dse::verifyResourceTimeMappingFinalists(
              evaluation->retainedHints, alternative.resourceTimeRegions,
              alternative.resourceTimeRegionBounds, attempt.systemMappings,
              artifacts, {}, evaluation->concurrencyBounds);
          if (!verified)
            return verified.takeError();
          resourceTimeSpectrum.emplace(std::move(*verified));
        }
      }
      outcomes.push_back(ApplicationMappingCandidateOutcome{
          alternative.preMappingCandidateRecordOrdinal,
          planOrdinal,
          alternative.dataflow,
          attempt.system,
          attempt.disposition,
          attempt.incompleteNodeOrdinal,
          attempt.incompleteReason,
          attempt.systemMappings,
          prepared
              .candidateInventory[alternative.preMappingCandidateRecordOrdinal],
          alternative.plan.systemBindingPartitions,
          ApplicationMappingRuntimeDisposition::NotRequested,
          {},
          {},
          std::move(resourceTimeSpectrum)});
      dse::JointDesignAttemptRecord adjusted = attempt;
      adjusted.planOrdinal = planOrdinal;
      attempts.push_back(std::move(adjusted));
    }
    return llvm::Error::success();
  };

  const auto executeTail =
      [&](std::size_t firstPlan) -> llvm::Expected<dse::JointDesignExecution> {
    llvm::ArrayRef<const dse::JointDesignExplorationPlan *> tail(plans);
    tail = tail.drop_front(firstPlan);
    std::string journalRoot = request.journalRoot;
    if (firstPlan != 0) {
      llvm::SmallString<256> childJournal(journalRoot);
      llvm::sys::path::append(childJournal,
                              "runtime-qualified-" + std::to_string(firstPlan));
      journalRoot = childJournal.str().str();
    }
    std::uint64_t maximumUsefulAccCoreCount = 0;
    for (std::size_t ordinal = firstPlan;
         ordinal != prepared.mappingAlternatives.size(); ++ordinal)
      for (const auto &bound :
           prepared.mappingAlternatives[ordinal].resourceTimeRegionBounds)
        maximumUsefulAccCoreCount = std::max(maximumUsefulAccCoreCount,
                                             bound.maximumUsefulResourceUnits);
    return dse::executeJointDesignWithHardwareReopen(
        tail, prepared.jointPolicy,
        {request.producer, std::move(journalRoot), evidence,
         prepared.preMappingFrontierPolicy.stoppingPolicy,
         request.boundedQuality,
         maximumUsefulAccCoreCount == 0
             ? std::nullopt
             : std::optional<std::uint64_t>(maximumUsefulAccCoreCount),
         request.siteCapacity, request.executionPolicy},
        artifacts, blobs);
  };

  std::optional<dse::JointDesignExecution> selectedExecution;
  std::size_t firstPlan = 0;
  while (firstPlan < plans.size()) {
    auto execution = executeTail(firstPlan);
    if (!execution)
      return execution.takeError();
    if (llvm::Error error = appendOutcomes(*execution, firstPlan))
      return std::move(error);
    attemptedSoftwarePlans += execution->summary.attemptedSoftwarePlans;
    hardwareReopenSearches += execution->summary.hardwareReopenSearches;
    hardwareParentPromotions += execution->summary.hardwareParentPromotions;
    hardwareReopensDeferredByQuality +=
        execution->summary.hardwareReopensDeferredByQuality;
    hardwareReopensWithheldWithoutExactFeedback +=
        execution->summary.hardwareReopensWithheldWithoutExactFeedback;
    hardwareRepairProbeLimit += execution->summary.hardwareRepairProbeLimit;
    hardwareRepairProbesPlanned +=
        execution->summary.hardwareRepairProbesPlanned;
    hardwareRepairProbesReserved +=
        execution->summary.hardwareRepairProbesReserved;
    hardwareRepairProbesConsumed +=
        execution->summary.hardwareRepairProbesConsumed;
    hardwareRepairProbesRejected +=
        execution->summary.hardwareRepairProbesRejected;
    hardwareRepairProbesCancelled +=
        execution->summary.hardwareRepairProbesCancelled;
    parentTechDecisions += execution->summary.parentTechDecisions;
    parentSpatialDecisions += execution->summary.parentSpatialDecisions;
    preservedTechDecisions += execution->summary.preservedTechDecisions;
    preservedSpatialDecisions += execution->summary.preservedSpatialDecisions;
    reopenedTechDecisions += execution->summary.reopenedTechDecisions;
    reopenedSpatialDecisions += execution->summary.reopenedSpatialDecisions;
    repairedTechDecisions += execution->summary.repairedTechDecisions;
    repairedSpatialDecisions += execution->summary.repairedSpatialDecisions;
    invalidationRootCount += execution->summary.invalidationRootCount;
    invalidationConeDecisionCount +=
        execution->summary.invalidationConeDecisionCount;
    parentRouteNodeCount += execution->summary.parentRouteNodeCount;
    preservedRouteNodeCount += execution->summary.preservedRouteNodeCount;
    reopenedRouteNodeCount += execution->summary.reopenedRouteNodeCount;
    repairedRouteNodeCount += execution->summary.repairedRouteNodeCount;
    parentServiceLegCount += execution->summary.parentServiceLegCount;
    preservedServiceLegCount += execution->summary.preservedServiceLegCount;
    reopenedServiceLegCount += execution->summary.reopenedServiceLegCount;
    verifiedAlternatives += execution->summary.verifiedAlternatives;
    techMappingDispatches += execution->summary.techMappingDispatchCount;
    spatialPnrDispatches += execution->summary.spatialPnrDispatchCount;
    systemPnrDispatches += execution->summary.systemPnrDispatchCount;

    if (!execution->summary.selectedPlanOrdinal ||
        !execution->summary.selectedMapping) {
      selectedExecution.emplace(std::move(*execution));
      break;
    }
    if (*execution->summary.selectedPlanOrdinal >
        std::numeric_limits<std::uint64_t>::max() - firstPlan)
      return invalid("selected Mapping plan ordinal overflowed");
    const std::uint64_t selectedPlanOrdinal =
        *execution->summary.selectedPlanOrdinal + firstPlan;
    if (selectedPlanOrdinal >= prepared.mappingAlternatives.size())
      return invalid("selected Mapping has a foreign plan ordinal");
    auto runtime = validateApplicationMappingRuntime(
        prepared, prepared.mappingAlternatives[selectedPlanOrdinal], *execution,
        request.executionPolicy, artifacts, blobs);
    if (!runtime)
      return runtime.takeError();
    bool joined = false;
    for (ApplicationMappingCandidateOutcome &outcome : outcomes) {
      if (outcome.planOrdinal != selectedPlanOrdinal ||
          !llvm::is_contained(outcome.systemMappings,
                              *execution->summary.selectedMapping))
        continue;
      outcome.runtimeDisposition = runtime->disposition;
      outcome.runtimeEvidence = runtime->evidence;
      joined = true;
    }
    if (!joined)
      return invalid("runtime validation has no exact Mapping attempt join");
    auto consumeRepairedExecutions =
        [&](auto &repaired) -> llvm::Expected<bool> {
      for (std::size_t childOrdinal = 0;
           childOrdinal != repaired->executions.size(); ++childOrdinal) {
        dse::JointDesignExecution &childExecution =
            repaired->executions[childOrdinal];
        if (childOrdinal >= repaired->childSystems.size())
          return invalid("hardware repair lost its child System");
        techMappingDispatches +=
            childExecution.summary.techMappingDispatchCount;
        spatialPnrDispatches += childExecution.summary.spatialPnrDispatchCount;
        systemPnrDispatches += childExecution.summary.systemPnrDispatchCount;
        std::vector<ArtifactRootReference> childMappings;
        for (const dse::JointMappedPair &pair : childExecution.mappedPairs)
          childMappings.insert(childMappings.end(), pair.systemMappings.begin(),
                               pair.systemMappings.end());
        llvm::sort(childMappings, artifactRootReferenceLess);
        childMappings.erase(
            std::unique(childMappings.begin(), childMappings.end()),
            childMappings.end());
        attempts.push_back({selectedPlanOrdinal,
                            repaired->childSystems[childOrdinal],
                            childMappings.empty()
                                ? dse::JointDesignAttemptDisposition::Incomplete
                                : dse::JointDesignAttemptDisposition::Verified,
                            std::nullopt,
                            childMappings.empty()
                                ? std::optional<dse::DsePlanIncompleteReason>(
                                      dse::CandidateGeneratorIncompleteReason::
                                          ProofNotEstablished)
                                : std::nullopt,
                            childMappings});
        if (childMappings.empty() || !childExecution.summary.selectedMapping)
          continue;
        auto childRuntime = validateApplicationMappingRuntime(
            prepared, prepared.mappingAlternatives[selectedPlanOrdinal],
            childExecution, request.executionPolicy, artifacts, blobs);
        if (!childRuntime)
          return childRuntime.takeError();
        std::optional<dse::ResourceTimeSpectrumFunnelResult> childSpectrum;
        const auto evaluation = llvm::find_if(
            prepared.resourceTimeFunnel.evaluations,
            [&](const auto &candidate) {
              return candidate.candidateIdentity ==
                     prepared.mappingAlternatives[selectedPlanOrdinal]
                         .candidateIdentity;
            });
        if (evaluation != prepared.resourceTimeFunnel.evaluations.end() &&
            !evaluation->retainedHints.empty()) {
          auto verified = dse::verifyResourceTimeMappingFinalists(
              evaluation->retainedHints,
              prepared.mappingAlternatives[selectedPlanOrdinal]
                  .resourceTimeRegions,
              prepared.mappingAlternatives[selectedPlanOrdinal]
                  .resourceTimeRegionBounds,
              childMappings, artifacts, {}, evaluation->concurrencyBounds);
          if (!verified)
            return verified.takeError();
          childSpectrum.emplace(std::move(*verified));
        }
        outcomes.push_back(ApplicationMappingCandidateOutcome{
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .preMappingCandidateRecordOrdinal,
            selectedPlanOrdinal,
            prepared.mappingAlternatives[selectedPlanOrdinal].dataflow,
            repaired->childSystems[childOrdinal],
            dse::JointDesignAttemptDisposition::Verified,
            std::nullopt,
            std::nullopt,
            childMappings,
            prepared.candidateInventory
                [prepared.mappingAlternatives[selectedPlanOrdinal]
                     .preMappingCandidateRecordOrdinal],
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .plan.systemBindingPartitions,
            childRuntime->disposition,
            childRuntime->evidence,
            {},
            std::move(childSpectrum)});
        if (childRuntime->disposition ==
            ApplicationMappingRuntimeDisposition::Completed) {
          childExecution.summary.selectedPlanOrdinal = selectedPlanOrdinal;
          selectedExecution.emplace(std::move(childExecution));
          return true;
        }
      }
      return false;
    };
    if (runtime->disposition !=
            ApplicationMappingRuntimeDisposition::Completed &&
        runtime->spatialFifoFeedback &&
        runtime->spatialFifoFeedback->disposition ==
            dse::SpatialFifoRuntimeFeedbackDisposition::Exact) {
      llvm::SmallString<256> feedbackJournal(request.journalRoot);
      llvm::sys::path::append(feedbackJournal,
                              "fifo-runtime-feedback-" +
                                  std::to_string(selectedPlanOrdinal));
      auto repaired = dse::executeSpatialFifoHardwareFeedbackReopen(
          prepared.mappingAlternatives[selectedPlanOrdinal].plan, *execution,
          prepared.jointPolicy, *runtime->spatialFifoFeedback,
          {request.producer, feedbackJournal.str().str(), evidence,
           dse::JointDesignStoppingPolicy::FirstVerified, std::nullopt,
           std::nullopt, request.siteCapacity, request.executionPolicy},
          artifacts, blobs);
      if (!repaired)
        return repaired.takeError();
      for (std::size_t childOrdinal = 0;
           childOrdinal != repaired->executions.size(); ++childOrdinal) {
        dse::JointDesignExecution &childExecution =
            repaired->executions[childOrdinal];
        if (childOrdinal >= repaired->childSystems.size())
          return invalid("FIFO hardware repair lost its child System");
        techMappingDispatches +=
            childExecution.summary.techMappingDispatchCount;
        spatialPnrDispatches += childExecution.summary.spatialPnrDispatchCount;
        systemPnrDispatches += childExecution.summary.systemPnrDispatchCount;
        std::vector<ArtifactRootReference> childMappings;
        for (const dse::JointMappedPair &pair : childExecution.mappedPairs)
          childMappings.insert(childMappings.end(), pair.systemMappings.begin(),
                               pair.systemMappings.end());
        llvm::sort(childMappings, artifactRootReferenceLess);
        childMappings.erase(
            std::unique(childMappings.begin(), childMappings.end()),
            childMappings.end());
        attempts.push_back({selectedPlanOrdinal,
                            repaired->childSystems[childOrdinal],
                            childMappings.empty()
                                ? dse::JointDesignAttemptDisposition::Incomplete
                                : dse::JointDesignAttemptDisposition::Verified,
                            std::nullopt,
                            childMappings.empty()
                                ? std::optional<dse::DsePlanIncompleteReason>(
                                      dse::CandidateGeneratorIncompleteReason::
                                          ProofNotEstablished)
                                : std::nullopt,
                            childMappings});
        if (childMappings.empty() || !childExecution.summary.selectedMapping)
          continue;
        auto childRuntime = validateApplicationMappingRuntime(
            prepared, prepared.mappingAlternatives[selectedPlanOrdinal],
            childExecution, request.executionPolicy, artifacts, blobs);
        if (!childRuntime)
          return childRuntime.takeError();
        std::optional<dse::ResourceTimeSpectrumFunnelResult> childSpectrum;
        const auto evaluation = llvm::find_if(
            prepared.resourceTimeFunnel.evaluations,
            [&](const auto &candidate) {
              return candidate.candidateIdentity ==
                     prepared.mappingAlternatives[selectedPlanOrdinal]
                         .candidateIdentity;
            });
        if (evaluation != prepared.resourceTimeFunnel.evaluations.end() &&
            !evaluation->retainedHints.empty()) {
          auto verified = dse::verifyResourceTimeMappingFinalists(
              evaluation->retainedHints,
              prepared.mappingAlternatives[selectedPlanOrdinal]
                  .resourceTimeRegions,
              prepared.mappingAlternatives[selectedPlanOrdinal]
                  .resourceTimeRegionBounds,
              childMappings, artifacts, {}, evaluation->concurrencyBounds);
          if (!verified)
            return verified.takeError();
          childSpectrum.emplace(std::move(*verified));
        }
        outcomes.push_back(ApplicationMappingCandidateOutcome{
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .preMappingCandidateRecordOrdinal,
            selectedPlanOrdinal,
            prepared.mappingAlternatives[selectedPlanOrdinal].dataflow,
            repaired->childSystems[childOrdinal],
            dse::JointDesignAttemptDisposition::Verified,
            std::nullopt,
            std::nullopt,
            childMappings,
            prepared.candidateInventory
                [prepared.mappingAlternatives[selectedPlanOrdinal]
                     .preMappingCandidateRecordOrdinal],
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .plan.systemBindingPartitions,
            childRuntime->disposition,
            childRuntime->evidence,
            {},
            std::move(childSpectrum)});
        if (childRuntime->disposition !=
            ApplicationMappingRuntimeDisposition::Completed)
          continue;
        childExecution.summary.selectedPlanOrdinal = selectedPlanOrdinal;
        selectedExecution.emplace(std::move(childExecution));
        break;
      }
      if (selectedExecution && selectedExecution->summary.selectedPlanOrdinal)
        break;
    }
    if (runtime->disposition !=
            ApplicationMappingRuntimeDisposition::Completed &&
        runtime->spatialOperandQueueFeedback &&
        runtime->spatialOperandQueueFeedback->disposition ==
            dse::SpatialOperandQueueRuntimeFeedbackDisposition::Exact) {
      llvm::SmallString<256> feedbackJournal(request.journalRoot);
      llvm::sys::path::append(feedbackJournal,
                              "operand-buffer-runtime-feedback-" +
                                  std::to_string(selectedPlanOrdinal));
      auto repaired = dse::executeSpatialOperandBufferHardwareFeedbackReopen(
          prepared.mappingAlternatives[selectedPlanOrdinal].plan, *execution,
          prepared.jointPolicy, *runtime->spatialOperandQueueFeedback,
          {request.producer, feedbackJournal.str().str(), evidence,
           dse::JointDesignStoppingPolicy::FirstVerified, std::nullopt,
           std::nullopt, request.siteCapacity, request.executionPolicy},
          artifacts, blobs);
      if (!repaired)
        return repaired.takeError();
      hardwareRepairProbeLimit += repaired->candidateLimit;
      hardwareRepairProbesPlanned += repaired->candidatesPlanned;
      hardwareRepairProbesReserved += repaired->candidatesReserved;
      hardwareRepairProbesConsumed += repaired->candidatesConsumed;
      hardwareRepairProbesRejected += repaired->candidatesRejected;
      hardwareRepairProbesCancelled += repaired->candidatesCancelled;
      auto selected = consumeRepairedExecutions(repaired);
      if (!selected)
        return selected.takeError();
      if (*selected)
        break;
    }
    if (runtime->disposition ==
        ApplicationMappingRuntimeDisposition::Completed) {
      execution->summary.selectedPlanOrdinal = selectedPlanOrdinal;
      selectedExecution.emplace(std::move(*execution));
      break;
    }

    execution->summary.selectedPlanOrdinal.reset();
    execution->summary.selectedMapping.reset();
    selectedExecution.emplace(std::move(*execution));
    if (runtime->disposition ==
        ApplicationMappingRuntimeDisposition::CancelledOrTimeout)
      break;
    firstPlan = static_cast<std::size_t>(selectedPlanOrdinal) + 1;
    // A verified Mapping is not an application result.  Runtime validation
    // may reject the selected QoR winner (for example, a functional replay
    // mismatch or an execution timeout).  Continue through the remaining
    // bounded software frontier for every stopping policy; otherwise
    // BoundedQuality would silently turn one failed application-level check
    // into a terminal Mapping failure.
  }
  if (!selectedExecution)
    return invalid("joint Mapping execution produced no bounded outcome");
  const auto qualityRuntimeDisposition =
      [](const std::optional<dse::JointDesignQualityIncompleteReason> &reason) {
        if (!reason)
          return ApplicationMappingRuntimeDisposition::Completed;
        switch (*reason) {
        case dse::JointDesignQualityIncompleteReason::Unsupported:
          return ApplicationMappingRuntimeDisposition::Unsupported;
        case dse::JointDesignQualityIncompleteReason::ProofNotEstablished:
          return ApplicationMappingRuntimeDisposition::ProofNotEstablished;
        case dse::JointDesignQualityIncompleteReason::ExecutionFailed:
          return ApplicationMappingRuntimeDisposition::ExecutionFailed;
        case dse::JointDesignQualityIncompleteReason::CancelledOrTimeout:
          return ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
        }
        llvm_unreachable("unknown application quality disposition");
      };
  for (ApplicationMappingCandidateOutcome &outcome : outcomes)
    for (const dse::JointDesignQualityObservation &observation :
         selectedExecution->summary.qualityObservations)
      if (llvm::is_contained(outcome.systemMappings, observation.candidate)) {
        outcome.qualityObjectiveCodes = observation.objectiveCodes;
        if (outcome.runtimeDisposition ==
            ApplicationMappingRuntimeDisposition::NotRequested)
          outcome.runtimeDisposition =
              qualityRuntimeDisposition(observation.incompleteReason);
      }
  selectedExecution->summary.attemptedSoftwarePlans = attemptedSoftwarePlans;
  selectedExecution->summary.hardwareReopenSearches = hardwareReopenSearches;
  selectedExecution->summary.hardwareParentPromotions =
      hardwareParentPromotions;
  selectedExecution->summary.hardwareReopensDeferredByQuality =
      hardwareReopensDeferredByQuality;
  selectedExecution->summary.hardwareReopensWithheldWithoutExactFeedback =
      hardwareReopensWithheldWithoutExactFeedback;
  selectedExecution->summary.hardwareRepairProbeLimit =
      hardwareRepairProbeLimit;
  selectedExecution->summary.hardwareRepairProbesPlanned =
      hardwareRepairProbesPlanned;
  selectedExecution->summary.hardwareRepairProbesReserved =
      hardwareRepairProbesReserved;
  selectedExecution->summary.hardwareRepairProbesConsumed =
      hardwareRepairProbesConsumed;
  selectedExecution->summary.hardwareRepairProbesRejected =
      hardwareRepairProbesRejected;
  selectedExecution->summary.hardwareRepairProbesCancelled =
      hardwareRepairProbesCancelled;
  selectedExecution->summary.parentTechDecisions = parentTechDecisions;
  selectedExecution->summary.parentSpatialDecisions = parentSpatialDecisions;
  selectedExecution->summary.preservedTechDecisions = preservedTechDecisions;
  selectedExecution->summary.preservedSpatialDecisions =
      preservedSpatialDecisions;
  selectedExecution->summary.reopenedTechDecisions = reopenedTechDecisions;
  selectedExecution->summary.reopenedSpatialDecisions =
      reopenedSpatialDecisions;
  selectedExecution->summary.repairedTechDecisions = repairedTechDecisions;
  selectedExecution->summary.repairedSpatialDecisions =
      repairedSpatialDecisions;
  selectedExecution->summary.invalidationRootCount = invalidationRootCount;
  selectedExecution->summary.invalidationConeDecisionCount =
      invalidationConeDecisionCount;
  selectedExecution->summary.parentRouteNodeCount = parentRouteNodeCount;
  selectedExecution->summary.preservedRouteNodeCount = preservedRouteNodeCount;
  selectedExecution->summary.reopenedRouteNodeCount = reopenedRouteNodeCount;
  selectedExecution->summary.repairedRouteNodeCount = repairedRouteNodeCount;
  selectedExecution->summary.parentServiceLegCount = parentServiceLegCount;
  selectedExecution->summary.preservedServiceLegCount =
      preservedServiceLegCount;
  selectedExecution->summary.reopenedServiceLegCount = reopenedServiceLegCount;
  selectedExecution->summary.verifiedAlternatives = verifiedAlternatives;
  selectedExecution->summary.techMappingDispatchCount = techMappingDispatches;
  selectedExecution->summary.spatialPnrDispatchCount = spatialPnrDispatches;
  selectedExecution->summary.systemPnrDispatchCount = systemPnrDispatches;
  selectedExecution->summary.attempts = std::move(attempts);
  if (!selectedExecution->summary.selectedMapping)
    selectedExecution->summary.declaredWorkExhausted =
        firstPlan >= plans.size();

  ApplicationMappingProvenance provenance;
  provenance.sourceProgram = prepared.preMappingSourceProgram;
  provenance.fabric = prepared.preMappingFabric;
  provenance.workload = prepared.preMappingWorkload;
  provenance.runtimeInput = prepared.preMappingRuntimeInput;
  provenance.frontierPolicyDigest = prepared.preMappingFrontierPolicyDigest;
  provenance.resourceTimeFunnelAccounting =
      prepared.resourceTimeFunnel.accounting;
  provenance.resourceTimeFunnelTruncated =
      prepared.resourceTimeFunnel.truncated;
  provenance.resourceTimeFunnelIncompleteReason =
      prepared.resourceTimeFunnel.incompleteReason;
  provenance.preMappingCompleteness = prepared.preMappingCompleteness;
  provenance.requestedPlannerMode = prepared.preMappingRequestedPlannerMode;
  provenance.resolvedPlannerMode = prepared.preMappingResolvedPlannerMode;
  ApplicationMappingExecution result{std::move(*selectedExecution),
                                     std::move(outcomes),
                                     std::move(provenance)};
  emitApplicationMappingDiagnostics(result);
  return result;
}

llvm::Expected<dse::JointBoundedQualityPolicy>
makeApplicationBoundedQualityPolicy(
    const PreparedApplicationBuild &prepared,
    const dse::PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  dse::CandidateMeasureObjectiveCatalogs catalogs;
  const auto dimension = [](std::uint32_t ordinal) {
    return dse::CandidateMeasureObjectiveDimension{
        ordinal, ResolvedObjectiveDirection::Minimize, 0,
        std::numeric_limits<std::uint64_t>::max()};
  };
  // The application QoR contract is deliberately small and exact: software
  // work, accelerator work, then imported AccCore cost.  Area, power, and
  // physical timing are not silently approximated here; a future provider
  // must add those dimensions at this owner with their own completed evidence.
  catalogs.dimensions = {dimension(0), dimension(1), dimension(2)};
  catalogs.weightedLevels = {{{{0, 1}}}, {{{1, 1}}}, {{{2, 1}}}};
  catalogs.totalOrderings = {{{0, 1, 2}}};
  auto program = dse::ObjectiveProgram::getCandidateMeasures(catalogs);
  if (!program)
    return program.takeError();
  auto sharedProgram =
      std::make_shared<const dse::ObjectiveProgram>(std::move(*program));

  dse::JointBoundedQualityPolicy result;
  result.objectiveProgram = sharedProgram;
  result.objectiveDimensionLabels = {"dfg_cycles", "cgra_cycles",
                                     "acc_core_count"};
  result.paretoDimensions = {0, 1, 2};
  result.finalTotalOrdering = 0;
  result.acquire = [&prepared, executionPolicy, &artifacts, &blobs,
                    sharedProgram](const dse::JointDesignExecution &execution,
                                   std::uint64_t planOrdinal)
      -> llvm::Expected<dse::JointDesignQualityAcquisition> {
    if (planOrdinal >= prepared.mappingAlternatives.size())
      return invalid("bounded-quality selected a foreign software plan");
    auto imported = importApplicationMapping(execution, artifacts);
    if (!imported)
      return imported.takeError();
    auto runtime = validateApplicationMappingRuntime(
        prepared, prepared.mappingAlternatives[planOrdinal], execution,
        executionPolicy, artifacts, blobs);
    if (!runtime)
      return runtime.takeError();
    switch (runtime->disposition) {
    case ApplicationMappingRuntimeDisposition::Completed:
      break;
    case ApplicationMappingRuntimeDisposition::Unsupported:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::Unsupported,
              execution.summary.selectedMapping}};
    case ApplicationMappingRuntimeDisposition::CancelledOrTimeout:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::CancelledOrTimeout,
              execution.summary.selectedMapping}};
    case ApplicationMappingRuntimeDisposition::ExecutionFailed:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ExecutionFailed,
              execution.summary.selectedMapping}};
    case ApplicationMappingRuntimeDisposition::ProofNotEstablished:
    case ApplicationMappingRuntimeDisposition::NotRequested:
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ProofNotEstablished,
              execution.summary.selectedMapping}};
    }
    if (!runtime->dfgCycles || !runtime->cgraCycles)
      return dse::JointDesignQualityAcquisition{
          dse::IncompleteJointDesignQuality{
              dse::JointDesignQualityIncompleteReason::ProofNotEstablished,
              execution.summary.selectedMapping}};
    const std::array<std::uint64_t, 3> measures = {
        *runtime->dfgCycles, *runtime->cgraCycles,
        static_cast<std::uint64_t>(
            imported->system.view().accCoreOccurrences().size())};
    dse::ObjectiveVector objective = sharedProgram->makeVector();
    if (llvm::Error error =
            sharedProgram->evaluateCandidateMeasures(measures, objective))
      return std::move(error);
    if (!execution.summary.selectedMapping)
      return invalid("bounded-quality acquisition has no selected mapping");
    return dse::JointDesignQualityAcquisition{
        std::vector<dse::CandidateObjectiveVector>{
            {*execution.summary.selectedMapping, std::move(objective)}}};
  };
  return result;
}

llvm::Expected<ApplicationDeploymentArtifacts> buildApplicationDeployment(
    const PreparedApplicationBuild &prepared,
    const ApplicationMappingExecution &mappingExecution,
    const llvm::Module &finalLinkedModule, ApplicationDeploymentRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::DeploymentConstruction);
  auto operationBegin = MonotonicClock::now();
  auto imported =
      importApplicationMapping(mappingExecution.execution, artifacts);
  emitElapsed(ApplicationBuildOperation::MappingImport, operationBegin);
  if (!imported)
    return imported.takeError();
  auto software = findPreparedSoftware(
      prepared, imported->mapping.view().dataflowIdentity());
  if (!software)
    return software.takeError();

  mlir::DialectRegistry registry = applicationDialectRegistry();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  operationBegin = MonotonicClock::now();
  hardware::PackedConfigurationABIDerivationStatistics derivationStatistics;
  auto abiDraft = hardware::derivePackedConfigurationABIDraft(
      imported->system, context, {}, &derivationStatistics);
  if (!abiDraft)
    return abiDraft.takeError();
  hardware::emitPackedConfigurationABIDerivationStatistics(
      derivationStatistics);
  auto abi =
      hardware::finalizeConfigurationABI(std::move(*abiDraft), artifacts);
  if (!abi)
    return abi.takeError();
  hardware::emitConfigurationABIConstructionStatistics(
      abi->constructionStatistics());
  emitElapsed(ApplicationBuildOperation::ConfigurationAbiDerivation,
              operationBegin);

  operationBegin = MonotonicClock::now();
  auto subjects = mapping::projectSystemExecutionSpatialCoreSubjects(
      imported->dataflowView, imported->mapping.view().executionBindings());
  if (!subjects)
    return subjects.takeError();
  std::vector<deployment::DeploymentHardwareBinding> hardwareBindings;
  hardwareBindings.reserve(subjects->size());
  for (fabric::SpatialCoreOccurrenceRef subject : *subjects) {
    auto implementation = hardware::finalizeFabricModelHardwareImplementation(
        *abi, subject, artifacts, blobs);
    if (!implementation)
      return implementation.takeError();
    auto runtimeBinding = runtime::finalizeFabricModelRuntimePlatformBinding(
        *implementation, artifacts, blobs);
    if (!runtimeBinding)
      return runtimeBinding.takeError();
    hardwareBindings.push_back(
        {implementation->reference(), runtimeBinding->reference()});
  }
  emitElapsed(ApplicationBuildOperation::HardwareBindingDerivation,
              operationBegin, hardwareBindings.size());

  operationBegin = MonotonicClock::now();
  auto targets = resolveSystemCompilerTargetBindings(
      imported->system, request.compilerTargetPolicy, artifacts);
  emitElapsed(ApplicationBuildOperation::CompilerTargetResolution,
              operationBegin);
  if (!targets)
    return targets.takeError();
  if (llvm::Error error = validateModuleCompilerTarget(
          finalLinkedModule, targets->host().binding()))
    return std::move(error);
  auto contexts = mapping::projectSystemExecutionContexts(
      imported->dataflowView, imported->mapping.view().executionBindings());
  if (!contexts)
    return contexts.takeError();
  auto roots = projectTargetGroupRoots(*contexts, *targets,
                                       imported->system.reference().artifact);
  if (!roots)
    return roots.takeError();
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      imported->mapping.view().dataflowIdentity()};
  auto invocationPlan = detail::deriveApplicationSpatialInvocationPlan(
      imported->dataflowView, prepared.sourceInvocation.entrySymbol);
  if (!invocationPlan)
    return invocationPlan.takeError();
  std::vector<dataflow::RootThreadLaunchRef> mappedRoots;
  for (llvm::ArrayRef<dataflow::RootThreadLaunchRef> groupRoots : *roots)
    mappedRoots.insert(mappedRoots.end(), groupRoots.begin(), groupRoots.end());
  llvm::sort(mappedRoots, [](const auto &lhs, const auto &rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  mappedRoots.erase(std::unique(mappedRoots.begin(), mappedRoots.end()),
                    mappedRoots.end());
  std::vector<dataflow::RootThreadLaunchRef> invocationRoots;
  invocationRoots.reserve(invocationPlan->launches.size());
  for (const detail::ApplicationSpatialInvocationPlan::Launch &launch :
       invocationPlan->launches)
    invocationRoots.push_back(launch.root);
  llvm::sort(invocationRoots, [](const auto &lhs, const auto &rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  if (mappedRoots.empty())
    return invalid("SystemMapping selects no InstructionCore binary target");
  if (std::adjacent_find(invocationRoots.begin(), invocationRoots.end()) !=
          invocationRoots.end() ||
      mappedRoots != invocationRoots)
    return invalid(
        "SystemMapping roots differ from the dynamic invocation roots");

  operationBegin = MonotonicClock::now();
  auto hostEntry = deriveHostProgramEntry(
      **software, prepared.sourceInvocation.entrySymbol, artifacts);
  if (!hostEntry)
    return hostEntry.takeError();
  hostEntry->abiSymbol = detail::applicationHostEntrySymbol.str();
  auto hostModule = detail::materializeHostDispatchModule(
      finalLinkedModule, imported->dataflow,
      prepared.sourceInvocation.entrySymbol, *invocationPlan);
  if (!hostModule)
    return hostModule.takeError();
  if (llvm::Error error =
          validateModuleCompilerTarget(**hostModule, targets->host().binding()))
    return std::move(error);
  auto hostObject = emitCompilerTargetObject(std::move(*hostModule),
                                             targets->host().binding());
  if (!hostObject)
    return hostObject.takeError();
  auto hostExecutable = linkCompilerTargetExecutable(
      *hostObject, targets->host().binding(),
      detail::applicationHostEntrySymbol, kPortableRiscVHostImageBase,
      request.linkerWorkspace);
  if (!hostExecutable)
    return hostExecutable.takeError();
  auto hostLoadRange = projectCompilerTargetExecutableLoadRange(
      *hostExecutable, targets->host().binding());
  if (!hostLoadRange)
    return hostLoadRange.takeError();
  auto firstInstructionImageBase = nextExecutableImageBase(hostLoadRange->end);
  if (!firstInstructionImageBase)
    return firstInstructionImageBase.takeError();
  std::uint64_t instructionImageBase = *firstInstructionImageBase;
  auto hostProgram =
      deployment::finalizeHostProgramLeaf({targets->host().reference(),
                                           std::move(*hostExecutable),
                                           {std::move(*hostEntry)},
                                           {},
                                           {}},
                                          artifacts, blobs);
  if (!hostProgram)
    return hostProgram.takeError();
  emitElapsed(ApplicationBuildOperation::HostProgramFinalization,
              operationBegin);

  operationBegin = MonotonicClock::now();
  std::vector<ArtifactRootReference> binaries;
  for (const auto indexed : llvm::enumerate(targets->instructionGroups())) {
    if ((*roots)[indexed.index()].empty())
      continue;
    std::vector<dataflow::RootedGraphLaunchRef> invocationGraphs;
    invocationGraphs.reserve((*roots)[indexed.index()].size());
    for (const detail::ApplicationSpatialInvocationPlan::Launch &launch :
         invocationPlan->launches)
      if (llvm::is_contained((*roots)[indexed.index()], launch.root))
        invocationGraphs.push_back(launch.graph);
    if (invocationGraphs.size() != (*roots)[indexed.index()].size())
      return invalid("InstructionCore target omits a dynamic invocation graph");
    auto binary = buildInstructionBinary(
        finalLinkedModule, dataflowReference, indexed.value().binding(),
        (*roots)[indexed.index()], invocationGraphs, instructionImageBase,
        request.linkerWorkspace, artifacts, blobs);
    if (!binary)
      return binary.takeError();
    std::uint64_t imageEnd = 0;
    for (const InstructionLoadSegment &segment :
         binary->binary().loadSegments())
      imageEnd =
          std::max(imageEnd, segment.virtualAddress + segment.memorySize);
    auto nextImageBase = nextExecutableImageBase(imageEnd);
    if (!nextImageBase)
      return nextImageBase.takeError();
    instructionImageBase = *nextImageBase;
    binaries.push_back(binary->reference());
  }
  emitElapsed(ApplicationBuildOperation::InstructionBinaryFinalization,
              operationBegin, binaries.size());

  operationBegin = MonotonicClock::now();
  auto deployment = deployment::buildDeploymentFromLinkedProgram(
      {imported->mapping.reference(), std::move(*hostProgram), binaries,
       hardwareBindings},
      finalLinkedModule, artifacts, blobs);
  if (!deployment)
    return deployment.takeError();
  emitElapsed(ApplicationBuildOperation::DeclarativeDeploymentFinalization,
              operationBegin);
  return ApplicationDeploymentArtifacts{
      abi->reference(), abi->constructionStatistics(),
      std::move(hardwareBindings), std::move(binaries), std::move(*deployment)};
}

} // namespace loom::application
