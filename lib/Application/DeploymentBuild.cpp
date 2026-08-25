#include "Application/ActivationDecision.h"
#include "Application/BuildDiagnostics.h"
#include "ApplicationRuntimeValidationInternal.h"
#include "BuildInternal.h"
#include "ExecutionGlue.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/DeploymentPipeline.h"
#include "Fabric/IR/FabricDialect.h"
#include "Frontend/Executable/ExecutableElf.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationDiagnostics.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Hardware/Implementation/FabricModel.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Runtime/FabricModelPlatform.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application {

using build_detail::ApplicationBuildOperationTimer;
using build_detail::emitElapsed;
using build_detail::invalid;
using build_detail::MonotonicClock;
using build_detail::verifyResourceTimeAlternative;

namespace {

constexpr std::uint64_t kPortableRiscVHostImageBase = 0x80000000;
constexpr std::uint64_t kExecutablePageBytes = 4096;
constexpr std::uint64_t kApplicationHostProgramEntryOrdinal = 0;

struct FinalizedApplicationActivationInputs final {
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
};

llvm::Expected<FinalizedApplicationActivationInputs>
finalizeApplicationActivationInputs(
    const PreparedApplicationBuild &prepared,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts) {
  auto source = sim::importStructuredProgramSimulationInputs(
      prepared.preMappingWorkload, prepared.preMappingRuntimeInput, artifacts);
  if (!source)
    return source.takeError();
  if (source->structuredProgram.identity() !=
      prepared.preMappingSourceProgram.artifact)
    return invalid(
        "application source inputs name a foreign StructuredProgram");
  const sim::StructuredProgramSimulationWorkload *sourceWorkload =
      source->workload.structuredProgram();
  const sim::StructuredProgramSimulationRuntimeInput *sourceRuntime =
      source->runtimeInput.structuredProgram();
  if (!sourceWorkload || !sourceRuntime)
    return invalid(
        "application source inputs are not StructuredProgram inputs");
  if (!sourceWorkload->observableContract.memories.empty() ||
      !sourceRuntime->memoryObjects.empty() ||
      !sourceRuntime->pointerBindings.empty())
    return invalid("Deployment activation memory ingress is unsupported");

  sim::SystemSimulationWorkload activation{
      {deployment.reference().artifact, kApplicationHostProgramEntryOrdinal}};
  activation.valueInputPlan.reserve(sourceWorkload->argumentPlan.size());
  for (const sim::StructuredProgramArgumentSource &argument :
       sourceWorkload->argumentPlan) {
    if (const auto *fixed =
            std::get_if<sim::CanonicalValueSequence>(&argument)) {
      activation.valueInputPlan.push_back(*fixed);
      continue;
    }
    if (std::holds_alternative<sim::StructuredRuntimeValueInput>(argument)) {
      activation.valueInputPlan.push_back(sim::RuntimeValueInput{});
      continue;
    }
    return invalid("Deployment activation pointer ingress is unsupported");
  }
  if (sourceWorkload->observableContract.returnValue)
    activation.observableContract.valueResults.push_back(0);
  auto workload =
      sim::finalizeSimulationWorkload(activation, deployment, artifacts);
  if (!workload)
    return workload.takeError();

  sim::SystemSimulationRuntimeInputDraft runtime{workload->identity()};
  runtime.runtimeEntryValues.reserve(sourceRuntime->runtimeValues.size());
  for (const sim::StructuredRuntimeValueEntry &value :
       sourceRuntime->runtimeValues)
    runtime.runtimeEntryValues.push_back({value.argumentOrdinal, value.value});
  auto runtimeInput = sim::finalizeSimulationRuntimeInput(
      runtime, *workload, deployment, artifacts);
  if (!runtimeInput)
    return runtimeInput.takeError();
  auto workloadReference = sim::publishSimulationWorkload(*workload, artifacts);
  if (!workloadReference)
    return workloadReference.takeError();
  auto runtimeReference =
      sim::publishSimulationRuntimeInput(*runtimeInput, artifacts);
  if (!runtimeReference)
    return runtimeReference.takeError();
  return FinalizedApplicationActivationInputs{std::move(*workloadReference),
                                              std::move(*runtimeReference)};
}

llvm::Expected<std::uint64_t> nextExecutableImageBase(std::uint64_t end) {
  if (end >
      std::numeric_limits<std::uint64_t>::max() - (kExecutablePageBytes - 1))
    return invalid("executable image range cannot be page-aligned");
  return (end + kExecutablePageBytes - 1) & ~(kExecutablePageBytes - 1);
}

llvm::Expected<deployment::CanonicalTypeBytes>
canonicalTypeBytes(mlir::Type type) {
  auto encoded = dataflow::encodeCanonicalType(type);
  if (!encoded)
    return encoded.takeError();
  return deployment::CanonicalTypeBytes(encoded->bytes().begin(),
                                        encoded->bytes().end());
}

llvm::Expected<FinalizedApplicationRuntimeManifest> finalizeRuntimeManifest(
    const PreparedApplicationBuild &prepared,
    const PreparedApplicationSoftware &software,
    const ApplicationMappingExecution &mappingExecution,
    std::uint64_t selectedPlan, const ArtifactRootReference &selectedMapping,
    const deployment::FinalizedDeployment &deployment,
    const FinalizedApplicationActivationInputs &activationInputs,
    const std::optional<pnr::ResourceTimeTransitionGraph> &transitionGraph,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (!mappingExecution.execution.summary.selectedPlanOrdinal ||
      *mappingExecution.execution.summary.selectedPlanOrdinal != selectedPlan ||
      !mappingExecution.execution.summary.selectedMapping ||
      *mappingExecution.execution.summary.selectedMapping != selectedMapping)
    return invalid("Deployment selection differs from its Mapping summary");
  const auto invocationRunKey = mappingExecution.execution.invocationRunKey();
  if (!invocationRunKey)
    return invalid("Deployment selection has no Mapping invocation run key");

  if (!mappingExecution.provenance.pairDecision)
    return invalid("Deployment selection has no application pair decision");
  const ApplicationPairDecisionRecord &pair =
      *mappingExecution.provenance.pairDecision;
  if (pair.manifestJoinStatus !=
          ApplicationPairManifestJoinStatus::OwnerScopedPlanningClosure ||
      !pair.invocationRunKey || !pair.pairIdentity || !pair.sourceProgram ||
      !pair.fabric || !pair.workload || !pair.runtimeInput ||
      !pair.selectedCandidateIdentity || !pair.selectedSystem)
    return invalid("Deployment selection lacks an exact pair manifest join");
  if (*pair.invocationRunKey != *invocationRunKey)
    return invalid("Deployment pair run key differs from Mapping execution");
  if (mappingExecution.provenance.sourceProgram != pair.sourceProgram ||
      mappingExecution.provenance.fabric != pair.fabric ||
      mappingExecution.provenance.workload != pair.workload ||
      mappingExecution.provenance.runtimeInput != pair.runtimeInput)
    return invalid("Deployment pair roots differ from Mapping provenance");
  if (*pair.sourceProgram != prepared.preMappingSourceProgram ||
      *pair.fabric != prepared.preMappingFabric ||
      *pair.workload != prepared.preMappingWorkload ||
      *pair.runtimeInput != prepared.preMappingRuntimeInput)
    return invalid("Deployment pair roots differ from application preparation");

  auto expectedPair = deriveApplicationPairIdentity(
      *pair.sourceProgram, *pair.fabric, *pair.workload, *pair.runtimeInput);
  if (!expectedPair)
    return expectedPair.takeError();
  if (*expectedPair != *pair.pairIdentity)
    return invalid("Deployment pair identity differs from its exact roots");

  const ApplicationMappingCandidateOutcome *selectedOutcome = nullptr;
  std::vector<ComponentViewDigest> selectedScheduleHints;
  for (const ApplicationMappingCandidateOutcome &outcome :
       mappingExecution.candidateOutcomes) {
    if (outcome.planOrdinal != selectedPlan ||
        !llvm::is_contained(outcome.systemMappings, selectedMapping))
      continue;
    if (!selectedOutcome) {
      selectedOutcome = &outcome;
    } else if (outcome.system != selectedOutcome->system ||
               outcome.disposition != selectedOutcome->disposition ||
               outcome.runtimeDisposition !=
                   selectedOutcome->runtimeDisposition ||
               outcome.systemMappings != selectedOutcome->systemMappings ||
               outcome.runtimeEvidence != selectedOutcome->runtimeEvidence ||
               outcome.oracleEvidence != selectedOutcome->oracleEvidence) {
      return invalid(
          "equivalent selected Mapping outcomes have different evidence");
    }
    selectedScheduleHints.push_back(outcome.resourceTimeScheduleHintDigest);
  }
  if (!selectedOutcome ||
      selectedOutcome->runtimeDisposition !=
          ApplicationMappingRuntimeDisposition::Completed ||
      selectedOutcome->runtimeEvidence.empty() ||
      selectedOutcome->oracleEvidence.empty())
    return invalid("Deployment selection lacks completed runtime evidence");
  const PreparedApplicationMappingAlternative &selectedAlternative =
      prepared.mappingAlternatives[selectedPlan];
  if (selectedOutcome->preMappingCandidateRecordOrdinal !=
          selectedAlternative.preMappingCandidateRecordOrdinal ||
      selectedOutcome->dataflow != selectedAlternative.dataflow ||
      !selectedOutcome->planningRecord ||
      !selectedOutcome->planningRecord->candidateIdentity ||
      *selectedOutcome->planningRecord->candidateIdentity !=
          selectedAlternative.candidateIdentity)
    return invalid("Deployment selection differs from its planning record");
  if (selectedOutcome->system != *pair.selectedSystem)
    return invalid("Deployment selection differs from the pair System");

  const ApplicationPairCandidateRecord *selectedCandidate = nullptr;
  for (const ApplicationPairCandidateRecord &candidate : pair.candidates) {
    if (!candidate.selected)
      continue;
    if (selectedCandidate)
      return invalid(
          "Application pair decision repeats its selected candidate");
    selectedCandidate = &candidate;
  }
  if (!selectedCandidate || !selectedCandidate->candidateIdentity ||
      *selectedCandidate->candidateIdentity !=
          *pair.selectedCandidateIdentity ||
      *selectedCandidate->candidateIdentity !=
          selectedAlternative.candidateIdentity ||
      selectedCandidate->planningRecordOrdinal !=
          selectedAlternative.preMappingCandidateRecordOrdinal ||
      !selectedCandidate->planOrdinal ||
      *selectedCandidate->planOrdinal != selectedPlan)
    return invalid("Deployment selection differs from the selected candidate");

  std::vector<ComponentViewDigest> observedScheduleHints;
  for (const ApplicationPairMappingObservation &observation :
       selectedCandidate->mappingObservations) {
    if (observation.planOrdinal != selectedPlan ||
        !llvm::is_contained(observation.systemMappings, selectedMapping))
      continue;
    if (observation.system != selectedOutcome->system ||
        observation.mappingDisposition != selectedOutcome->disposition ||
        observation.runtimeDisposition != selectedOutcome->runtimeDisposition ||
        observation.runtimeEvidence != selectedOutcome->runtimeEvidence ||
        observation.oracleEvidence != selectedOutcome->oracleEvidence)
      return invalid("Deployment selection differs from its pair observation");
    observedScheduleHints.push_back(observation.scheduleHintDigest);
  }
  const auto canonicalizeHints = [](auto &digests) {
    llvm::sort(digests, [](const auto &lhs, const auto &rhs) {
      return lhs.bytes() < rhs.bytes();
    });
  };
  canonicalizeHints(selectedScheduleHints);
  canonicalizeHints(observedScheduleHints);
  std::vector<ComponentViewDigest> preparedScheduleHints =
      selectedAlternative.equivalentScheduleHintDigests;
  canonicalizeHints(preparedScheduleHints);
  if (selectedScheduleHints.empty() ||
      selectedScheduleHints != observedScheduleHints ||
      selectedScheduleHints != preparedScheduleHints ||
      std::adjacent_find(selectedScheduleHints.begin(),
                         selectedScheduleHints.end()) !=
          selectedScheduleHints.end())
    return invalid(
        "Deployment selection differs from its equivalent schedule hints");

  const dse::PreMappingCandidatePlanningRecord &planning =
      *selectedOutcome->planningRecord;
  if (!planning.structuredProgram || !planning.canonicalDataflow ||
      !planning.projection || planning.ownedProtocolRoots.empty() ||
      *planning.canonicalDataflow != selectedAlternative.dataflow)
    return invalid("Deployment selection has no exact planning preimage");
  const auto evaluation = llvm::find_if(
      prepared.resourceTimeFunnel.evaluations, [&](const auto &candidate) {
        return candidate.candidateIdentity ==
               selectedAlternative.candidateIdentity;
      });
  if (evaluation == prepared.resourceTimeFunnel.evaluations.end())
    return invalid("Deployment selection has no resource-time owner");
  std::vector<dse::ResourceTimeScheduleHint> exactScheduleHints;
  exactScheduleHints.reserve(selectedScheduleHints.size());
  for (const ComponentViewDigest &digest : selectedScheduleHints) {
    auto hint = build_detail::findResourceTimeScheduleHint(*evaluation, digest);
    if (!hint)
      return hint.takeError();
    exactScheduleHints.push_back(**hint);
  }
  if (!mappingExecution.execution.invocationManifest())
    return invalid("Deployment selection has no exact DSE occurrence");
  std::vector<dse::JointDesignInvocationManifestReference>
      supportingInvocations(
          mappingExecution.execution.supportingInvocationManifests().begin(),
          mappingExecution.execution.supportingInvocationManifests().end());
  auto activationDecision = ApplicationActivationDecision::get(
      {*pair.sourceProgram,
       *pair.fabric,
       *pair.workload,
       *pair.runtimeInput,
       software.replayCases,
       *mappingExecution.execution.invocationManifest(),
       std::move(supportingInvocations),
       {*planning.structuredProgram, *planning.canonicalDataflow,
        planning.ownedProtocolRoots, planning.projection->identity,
        prepared.preMappingFrontierPolicyDigest},
       selectedAlternative.candidateIdentity,
       selectedPlan,
       std::move(exactScheduleHints),
       *pair.selectedSystem,
       selectedMapping,
       pair.disposition,
       selectedOutcome->runtimeEvidence,
       selectedOutcome->oracleEvidence},
      artifacts, blobs);
  if (!activationDecision)
    return activationDecision.takeError();
  auto finalizedActivationDecision = publishApplicationActivationDecision(
      std::move(*activationDecision), artifacts);
  if (!finalizedActivationDecision)
    return finalizedActivationDecision.takeError();

  auto manifest =
      ApplicationRuntimeManifest::get({*pair.sourceProgram,
                                       *pair.fabric,
                                       *pair.workload,
                                       *pair.runtimeInput,
                                       software.replayCases,
                                       finalizedActivationDecision->reference(),
                                       *pair.pairIdentity,
                                       *pair.invocationRunKey,
                                       pair.disposition,
                                       *pair.selectedCandidateIdentity,
                                       selectedPlan,
                                       selectedScheduleHints,
                                       *pair.selectedSystem,
                                       selectedMapping,
                                       deployment.reference(),
                                       activationInputs.workload,
                                       activationInputs.runtimeInput,
                                       {},
                                       selectedOutcome->runtimeEvidence,
                                       selectedOutcome->oracleEvidence,
                                       transitionGraph},
                                      artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  return publishApplicationRuntimeManifest(std::move(*manifest), artifacts);
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

} // namespace

llvm::Expected<ApplicationDeploymentArtifacts> buildApplicationDeployment(
    const PreparedApplicationBuild &prepared,
    const ApplicationMappingExecution &mappingExecution,
    const llvm::Module &finalLinkedModule, ApplicationDeploymentRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::DeploymentConstruction);
  auto operationBegin = MonotonicClock::now();
  auto imported =
      detail::importApplicationMapping(mappingExecution.execution, artifacts);
  emitElapsed(ApplicationBuildOperation::MappingImport, operationBegin);
  if (!imported)
    return imported.takeError();
  auto software = detail::findPreparedSoftware(
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
  const auto deriveHardwareBindings =
      [&](const mapping::FinalizedSystemMapping &systemMapping)
      -> llvm::Expected<std::vector<deployment::DeploymentHardwareBinding>> {
    auto subjects = mapping::projectSystemExecutionSpatialCoreSubjects(
        imported->dataflowView, systemMapping.view().executionBindings());
    if (!subjects)
      return subjects.takeError();
    std::vector<deployment::DeploymentHardwareBinding> bindings;
    bindings.reserve(subjects->size());
    for (fabric::SpatialCoreOccurrenceRef subject : *subjects) {
      auto implementation = hardware::finalizeFabricModelHardwareImplementation(
          *abi, subject, artifacts, blobs);
      if (!implementation)
        return implementation.takeError();
      auto runtimeBinding = runtime::finalizeFabricModelRuntimePlatformBinding(
          *implementation, artifacts, blobs);
      if (!runtimeBinding)
        return runtimeBinding.takeError();
      bindings.push_back(
          {implementation->reference(), runtimeBinding->reference()});
    }
    return bindings;
  };
  auto selectedHardwareBindings = deriveHardwareBindings(imported->mapping);
  if (!selectedHardwareBindings)
    return selectedHardwareBindings.takeError();
  emitElapsed(ApplicationBuildOperation::HardwareBindingDerivation,
              operationBegin, selectedHardwareBindings->size());

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
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      imported->mapping.view().dataflowIdentity()};
  auto invocationPlan = detail::deriveApplicationSpatialInvocationPlan(
      imported->dataflowView, prepared.sourceInvocation.entrySymbol);
  if (!invocationPlan)
    return invocationPlan.takeError();
  std::vector<dataflow::RootThreadLaunchRef> invocationRoots;
  invocationRoots.reserve(invocationPlan->launches.size());
  for (const detail::ApplicationSpatialInvocationPlan::Launch &launch :
       invocationPlan->launches)
    invocationRoots.push_back(launch.root);
  llvm::sort(invocationRoots, [](const auto &lhs, const auto &rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  if (std::adjacent_find(invocationRoots.begin(), invocationRoots.end()) !=
      invocationRoots.end())
    return invalid("dynamic invocation repeats one root");

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
  const auto buildInstructionBinaries =
      [&](const mapping::FinalizedSystemMapping &systemMapping)
      -> llvm::Expected<std::vector<ArtifactRootReference>> {
    auto contexts = mapping::projectSystemExecutionContexts(
        imported->dataflowView, systemMapping.view().executionBindings());
    if (!contexts)
      return contexts.takeError();
    auto roots = projectTargetGroupRoots(*contexts, *targets,
                                         imported->system.reference().artifact);
    if (!roots)
      return roots.takeError();
    std::vector<dataflow::RootThreadLaunchRef> mappedRoots;
    for (llvm::ArrayRef<dataflow::RootThreadLaunchRef> groupRoots : *roots)
      mappedRoots.insert(mappedRoots.end(), groupRoots.begin(),
                         groupRoots.end());
    llvm::sort(mappedRoots, [](const auto &lhs, const auto &rhs) {
      return lhs.entity.value() < rhs.entity.value();
    });
    mappedRoots.erase(std::unique(mappedRoots.begin(), mappedRoots.end()),
                      mappedRoots.end());
    if (mappedRoots.empty())
      return invalid("SystemMapping selects no InstructionCore binary target");
    if (mappedRoots != invocationRoots)
      return invalid(
          "SystemMapping roots differ from the dynamic invocation roots");

    std::vector<ArtifactRootReference> result;
    std::uint64_t imageBase = instructionImageBase;
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
        return invalid(
            "InstructionCore target omits a dynamic invocation graph");
      auto binary = buildInstructionBinary(
          finalLinkedModule, dataflowReference, indexed.value().binding(),
          (*roots)[indexed.index()], invocationGraphs, imageBase,
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
      imageBase = *nextImageBase;
      result.push_back(binary->reference());
    }
    return result;
  };
  auto selectedBinaries = buildInstructionBinaries(imported->mapping);
  if (!selectedBinaries)
    return selectedBinaries.takeError();
  emitElapsed(ApplicationBuildOperation::InstructionBinaryFinalization,
              operationBegin, selectedBinaries->size());

  operationBegin = MonotonicClock::now();
  auto deployment = deployment::buildDeploymentFromLinkedProgram(
      {imported->mapping.reference(), *hostProgram, *selectedBinaries,
       *selectedHardwareBindings},
      finalLinkedModule, artifacts, blobs);
  if (!deployment)
    return deployment.takeError();
  auto activationInputs =
      finalizeApplicationActivationInputs(prepared, *deployment, artifacts);
  if (!activationInputs)
    return activationInputs.takeError();

  const std::optional<std::uint64_t> selectedPlan =
      mappingExecution.execution.summary.selectedPlanOrdinal;
  if (selectedPlan && *selectedPlan >= prepared.mappingAlternatives.size())
    return invalid("selected resource-time plan ordinal is out of range");

  std::vector<const ApplicationIncrementalMappingObservation *>
      pathObservations;
  std::vector<ArtifactRootReference> endpointMappings = {
      imported->mapping.reference()};
  if (mappingExecution.provenance.resourceTimeMappingPath) {
    if (!selectedPlan)
      return invalid("resource-time Mapping path has no selected entry plan");
    const ApplicationResourceTimeMappingPath &path =
        *mappingExecution.provenance.resourceTimeMappingPath;
    if (path.scheduleOwnerPlanOrdinal != *selectedPlan ||
        path.scheduleOwnerPlanOrdinal >= prepared.mappingAlternatives.size())
      return invalid("resource-time Mapping path has a foreign schedule owner");
    const PreparedApplicationMappingAlternative &scheduleOwner =
        prepared.mappingAlternatives[path.scheduleOwnerPlanOrdinal];
    if (path.scheduleHintDigest !=
        scheduleOwner.resourceTimeScheduleHintDigest)
      return invalid("resource-time Mapping path lost its schedule digest");
    std::uint64_t parentPlanOrdinal = path.scheduleOwnerPlanOrdinal;
    ArtifactRootReference parentMapping = imported->mapping.reference();
    for (const std::uint64_t observationOrdinal : path.observationOrdinals) {
      if (observationOrdinal >=
          mappingExecution.provenance.incrementalMappingObservations.size())
        return invalid("resource-time Mapping path has a foreign observation");
      const ApplicationIncrementalMappingObservation &observation =
          mappingExecution.provenance
              .incrementalMappingObservations[observationOrdinal];
      if (!observation.verified || !observation.childMapping ||
          observation.parentMapping != parentMapping ||
          observation.parentPlanOrdinal != parentPlanOrdinal ||
          observation.parentPlanOrdinal >=
              prepared.mappingAlternatives.size() ||
          observation.childPlanOrdinal >=
              prepared.mappingAlternatives.size() ||
          observation.parentScheduleHintDigest !=
              prepared.mappingAlternatives[observation.parentPlanOrdinal]
                  .resourceTimeScheduleHintDigest ||
          observation.childScheduleHintDigest !=
              prepared.mappingAlternatives[observation.childPlanOrdinal]
                  .resourceTimeScheduleHintDigest ||
          llvm::is_contained(endpointMappings, *observation.childMapping))
        return invalid("resource-time Mapping path is not one exact verified "
                       "repair chain");
      const PreparedApplicationMappingAlternative &childAlternative =
          prepared.mappingAlternatives[observation.childPlanOrdinal];
      if (childAlternative.candidateIdentity !=
              scheduleOwner.candidateIdentity ||
          childAlternative.dataflow != scheduleOwner.dataflow ||
          childAlternative.plan.pairOutputs.size() != 1 ||
          scheduleOwner.plan.pairOutputs.size() != 1 ||
          childAlternative.plan.pairOutputs.front().pair.system !=
              scheduleOwner.plan.pairOutputs.front().pair.system)
        return invalid("resource-time Mapping path changes its schedule "
                       "candidate or immutable System");
      pathObservations.push_back(&observation);
      endpointMappings.push_back(*observation.childMapping);
      parentMapping = *observation.childMapping;
      parentPlanOrdinal = observation.childPlanOrdinal;
    }
  }

  std::vector<dse::ResourceTimeMappingDeploymentEndpoint> endpoints;
  endpoints.reserve(endpointMappings.size());
  endpoints.push_back({imported->mapping.reference(), deployment->reference()});
  const auto reportEndpointIncomplete =
      [&](const ArtifactRootReference &mappingReference, llvm::Error error) {
        const std::string diagnostic = llvm::toString(std::move(error));
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
            mapping_debug::Event::MappingFailure,
            [&](llvm::json::Object &fields) {
              fields["operation"] =
                  "application_resource_time_deployment_endpoint";
              fields["mapping"] =
                  formatArtifactRootReferenceJson(mappingReference);
              fields["disposition"] = "proof_not_established";
              fields["diagnostic"] = diagnostic;
            });
      };
  for (const ArtifactRootReference &mappingReference : endpointMappings) {
    if (mappingReference == imported->mapping.reference())
      continue;
    auto mapping = mapping::importSystemMapping(mappingReference, artifacts);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() !=
            imported->mapping.view().dataflowIdentity() ||
        mapping->view().fabricIdentity() !=
            imported->mapping.view().fabricIdentity())
      return invalid("resource-time endpoint Mapping changes its application "
                     "or immutable System");
    auto bindings = deriveHardwareBindings(*mapping);
    if (!bindings) {
      reportEndpointIncomplete(mappingReference, bindings.takeError());
      break;
    }
    auto endpointBinaries = buildInstructionBinaries(*mapping);
    if (!endpointBinaries) {
      reportEndpointIncomplete(mappingReference, endpointBinaries.takeError());
      break;
    }
    auto endpointDeployment = deployment::buildDeploymentFromLinkedProgram(
        {mappingReference, *hostProgram, *endpointBinaries, *bindings},
        finalLinkedModule, artifacts, blobs);
    if (!endpointDeployment) {
      reportEndpointIncomplete(mappingReference,
                               endpointDeployment.takeError());
      break;
    }
    endpoints.push_back({mappingReference, endpointDeployment->reference()});
  }
  std::optional<dse::ResourceTimeSpectrumFunnelResult> resourceTimeSpectrum;
  std::vector<ApplicationResourceTimeTransitionEvidence>
      resourceTimeTransitions;
  std::optional<pnr::ResourceTimeTransitionGraph> resourceTimeTransitionGraph;
  if (selectedPlan && endpoints.size() > 1) {
    const ApplicationResourceTimeMappingPath &path =
        *mappingExecution.provenance.resourceTimeMappingPath;
    const PreparedApplicationMappingAlternative &scheduleOwner =
        prepared.mappingAlternatives[path.scheduleOwnerPlanOrdinal];
    std::optional<dse::ResourceTimeSpectrumFunnelResult> longestIncomplete;
    for (std::size_t prefixLength = endpoints.size(); prefixLength > 1;
         --prefixLength) {
      auto verified = verifyResourceTimeAlternative(
          prepared.resourceTimeFunnel, scheduleOwner,
          llvm::ArrayRef(endpointMappings).take_front(prefixLength), artifacts,
          blobs, path.scheduleHintDigest,
          llvm::ArrayRef(endpoints).take_front(prefixLength),
          request.executionControl);
      if (!verified)
        return verified.takeError();
      if (!*verified)
        continue;
      if (!longestIncomplete)
        longestIncomplete = **verified;
      const auto *spectrum =
          std::get_if<dse::VerifiedResourceTimeSpectrum>(
              &(*verified)->verification);
      if (!spectrum)
        continue;
      const dse::VerifiedResourceTimeSpectrumScenario *pathScenario = nullptr;
      for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
           spectrum->scenarios) {
        if (!scenario.transitionGraph ||
            scenario.transitionGraph->entry.mapping !=
                endpoints.front().mapping ||
            scenario.transitionGraph->entry.deployment !=
                endpoints.front().deployment ||
            scenario.transitionGraph->transitions.size() != prefixLength - 1)
          continue;
        bool exactPath =
            scenario.transitionGraph->endpoints.size() == prefixLength;
        for (std::size_t index = 0; exactPath && index + 1 < prefixLength;
             ++index) {
          const pnr::ResourceTimeTransition &edge =
              scenario.transitionGraph->transitions[index];
          exactPath = edge.parent.mapping == endpoints[index].mapping &&
                      edge.parent.deployment == endpoints[index].deployment &&
                      edge.child.mapping == endpoints[index + 1].mapping &&
                      edge.child.deployment == endpoints[index + 1].deployment;
        }
        if (!exactPath)
          continue;
        if (pathScenario)
          return invalid("resource-time schedule produced more than one exact "
                         "Mapping path");
        pathScenario = &scenario;
      }
      if (!pathScenario)
        return invalid("verified resource-time spectrum lost its exact ordered "
                       "Mapping path");

      resourceTimeTransitionGraph = *pathScenario->transitionGraph;
      resourceTimeSpectrum = **verified;
      for (std::size_t edgeOrdinal = 0;
           edgeOrdinal != pathScenario->transitions.transitions.size();
           ++edgeOrdinal) {
        const ApplicationIncrementalMappingObservation &candidate =
            *pathObservations[edgeOrdinal];
        const pnr::ResourceTimeTransition &edge =
            pathScenario->transitions.transitions[edgeOrdinal];
        if (edge.parent.mapping != candidate.parentMapping ||
            !candidate.childMapping ||
            edge.child.mapping != *candidate.childMapping ||
            edge.status != pnr::ResourceTimeTransitionStatus::Verified)
          return invalid("resource-time Mapping path edge lost its exact repair "
                         "observation");
        const PreparedApplicationMappingAlternative &childAlternative =
            prepared.mappingAlternatives[candidate.childPlanOrdinal];
        auto childVerified = verifyResourceTimeAlternative(
            prepared.resourceTimeFunnel, childAlternative,
            {*candidate.childMapping}, artifacts, blobs,
            candidate.childScheduleHintDigest, {}, request.executionControl);
        if (!childVerified)
          return childVerified.takeError();
        if (!*childVerified ||
            !std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
                (*childVerified)->verification))
          return invalid("resource-time path child lost its independently "
                         "verified Mapping spectrum");
        resourceTimeTransitions.push_back(
            {edge,
             **verified,
             std::move(**childVerified),
             {candidate.reopenedRoots, candidate.reuseDisposition,
              candidate.preservedTechMappings,
              candidate.preservedSpatialMappings,
              candidate.repairedTechMappings,
              candidate.repairedSpatialMappings,
              candidate.preservedSystemBindings,
              candidate.reopenedSystemBindings,
              candidate.coldWallTimeNanoseconds,
              candidate.incrementalWallTimeNanoseconds,
              candidate.coldVerifierRetainedBytes,
              candidate.incrementalVerifierRetainedBytes,
              candidate.coldVerifierWork, candidate.incrementalVerifierWork,
              candidate.coldProviderWork,
              candidate.incrementalProviderWork,
              candidate.coldDfgCycles, candidate.coldCgraCycles,
              candidate.incrementalDfgCycles,
              candidate.incrementalCgraCycles}});
      }
      break;
    }
    if (!resourceTimeSpectrum && longestIncomplete)
      resourceTimeSpectrum = std::move(*longestIncomplete);
  }

  if (selectedPlan && !resourceTimeTransitionGraph) {
    const PreparedApplicationMappingAlternative &alternative =
        prepared.mappingAlternatives[*selectedPlan];
    for (const ApplicationMappingCandidateOutcome &outcome :
         mappingExecution.candidateOutcomes) {
      if (outcome.planOrdinal != *selectedPlan ||
          outcome.systemMappings.empty() ||
          !llvm::is_contained(outcome.systemMappings,
                              imported->mapping.reference()))
        continue;
      auto verified = verifyResourceTimeAlternative(
          prepared.resourceTimeFunnel, alternative,
          {imported->mapping.reference()}, artifacts, blobs,
          outcome.resourceTimeScheduleHintDigest, {}, request.executionControl);
      if (!verified)
        return verified.takeError();
      if (!*verified)
        continue;
      const bool completed =
          std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
              (*verified)->verification);
      if (!resourceTimeSpectrum || completed)
        resourceTimeSpectrum = std::move(**verified);
      if (completed)
        break;
    }
  }

  if (selectedPlan && !resourceTimeTransitionGraph) {
    resourceTimeTransitionGraph.emplace(pnr::ResourceTimeTransitionGraph{
        {imported->mapping.reference(), deployment->reference()}, {}, {}});
    resourceTimeTransitionGraph->endpoints.push_back(
        resourceTimeTransitionGraph->entry);
  }
  if (resourceTimeTransitionGraph)
    if (llvm::Error error = pnr::verifyResourceTimeTransitionGraph(
            *resourceTimeTransitionGraph, artifacts, blobs))
      return std::move(error);
  if (!selectedPlan)
    return invalid("Deployment selection has no selected plan ordinal");
  auto runtimeManifest = finalizeRuntimeManifest(
      prepared, **software, mappingExecution, *selectedPlan,
      imported->mapping.reference(), *deployment, *activationInputs,
      resourceTimeTransitionGraph, artifacts, blobs);
  if (!runtimeManifest)
    return runtimeManifest.takeError();
  emitElapsed(ApplicationBuildOperation::DeclarativeDeploymentFinalization,
              operationBegin);
  return ApplicationDeploymentArtifacts{abi->reference(),
                                        abi->constructionStatistics(),
                                        std::move(*selectedHardwareBindings),
                                        std::move(*selectedBinaries),
                                        std::move(resourceTimeTransitions),
                                        std::move(resourceTimeSpectrum),
                                        std::move(*runtimeManifest),
                                        std::move(*deployment)};
}

} // namespace loom::application
