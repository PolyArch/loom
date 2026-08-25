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

  const std::optional<std::uint64_t> selectedPlan =
      mappingExecution.execution.summary.selectedPlanOrdinal;
  if (selectedPlan && *selectedPlan >= prepared.mappingAlternatives.size())
    return invalid("selected resource-time plan ordinal is out of range");

  std::vector<const ApplicationIncrementalMappingObservation *>
      transitionCandidates;
  if (selectedPlan) {
    for (const ApplicationIncrementalMappingObservation &observation :
         mappingExecution.provenance.incrementalMappingObservations) {
      if (!observation.verified || !observation.childMapping ||
          observation.parentPlanOrdinal != *selectedPlan ||
          observation.parentMapping != imported->mapping.reference() ||
          *observation.childMapping == observation.parentMapping)
        continue;
      if (observation.childPlanOrdinal >= prepared.mappingAlternatives.size())
        return invalid("resource-time adjacency has a foreign child plan");
      transitionCandidates.push_back(&observation);
    }
  }
  llvm::sort(transitionCandidates, [](const auto *lhs, const auto *rhs) {
    if (*lhs->childMapping != *rhs->childMapping)
      return artifactRootReferenceLess(*lhs->childMapping, *rhs->childMapping);
    if (lhs->childPlanOrdinal != rhs->childPlanOrdinal)
      return lhs->childPlanOrdinal < rhs->childPlanOrdinal;
    if (lhs->childScheduleHintDigest != rhs->childScheduleHintDigest)
      return lhs->childScheduleHintDigest.bytes() <
             rhs->childScheduleHintDigest.bytes();
    return lhs->parentScheduleHintDigest.bytes() <
           rhs->parentScheduleHintDigest.bytes();
  });
  transitionCandidates.erase(
      std::unique(transitionCandidates.begin(), transitionCandidates.end(),
                  [](const auto *lhs, const auto *rhs) {
                    return lhs->childMapping == rhs->childMapping &&
                           lhs->parentScheduleHintDigest ==
                               rhs->parentScheduleHintDigest &&
                           lhs->childScheduleHintDigest ==
                               rhs->childScheduleHintDigest;
                  }),
      transitionCandidates.end());

  std::vector<ArtifactRootReference> endpointMappings = {
      imported->mapping.reference()};
  for (const ApplicationIncrementalMappingObservation *candidate :
       transitionCandidates)
    endpointMappings.push_back(*candidate->childMapping);
  llvm::sort(endpointMappings, artifactRootReferenceLess);
  endpointMappings.erase(
      std::unique(endpointMappings.begin(), endpointMappings.end()),
      endpointMappings.end());

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
      continue;
    }
    auto endpointBinaries = buildInstructionBinaries(*mapping);
    if (!endpointBinaries) {
      reportEndpointIncomplete(mappingReference, endpointBinaries.takeError());
      continue;
    }
    auto endpointDeployment = deployment::buildDeploymentFromLinkedProgram(
        {mappingReference, *hostProgram, *endpointBinaries, *bindings},
        finalLinkedModule, artifacts, blobs);
    if (!endpointDeployment) {
      reportEndpointIncomplete(mappingReference,
                               endpointDeployment.takeError());
      continue;
    }
    endpoints.push_back({mappingReference, endpointDeployment->reference()});
  }
  llvm::sort(endpoints, [](const auto &lhs, const auto &rhs) {
    return artifactRootReferenceLess(lhs.mapping, rhs.mapping);
  });

  std::optional<dse::ResourceTimeSpectrumFunnelResult> resourceTimeSpectrum;
  std::vector<ApplicationResourceTimeTransitionEvidence>
      resourceTimeTransitions;
  if (selectedPlan) {
    const PreparedApplicationMappingAlternative &alternative =
        prepared.mappingAlternatives[*selectedPlan];
    for (const ApplicationIncrementalMappingObservation *candidate :
         transitionCandidates) {
      const PreparedApplicationMappingAlternative &childAlternative =
          prepared.mappingAlternatives[candidate->childPlanOrdinal];
      const ArtifactRootReference childMapping = *candidate->childMapping;
      auto childVerified = verifyResourceTimeAlternative(
          prepared.resourceTimeFunnel, childAlternative, {childMapping},
          artifacts, blobs, candidate->childScheduleHintDigest);
      if (!childVerified)
        return childVerified.takeError();
      if (!*childVerified)
        continue;
      if (!std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
              (*childVerified)->verification)) {
        if (!resourceTimeSpectrum)
          resourceTimeSpectrum = std::move(**childVerified);
        continue;
      }

      std::array<ArtifactRootReference, 2> transitionMappings = {
          imported->mapping.reference(), childMapping};
      std::vector<dse::ResourceTimeMappingDeploymentEndpoint>
          transitionEndpoints;
      for (const ArtifactRootReference &mappingReference : transitionMappings) {
        const auto endpoint = llvm::find_if(endpoints, [&](const auto &row) {
          return row.mapping == mappingReference;
        });
        if (endpoint != endpoints.end())
          transitionEndpoints.push_back(*endpoint);
      }
      auto verified = verifyResourceTimeAlternative(
          prepared.resourceTimeFunnel, alternative, transitionMappings,
          artifacts, blobs, candidate->parentScheduleHintDigest,
          transitionEndpoints);
      if (!verified)
        return verified.takeError();
      if (!*verified)
        continue;
      const bool completed =
          std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
              (*verified)->verification);
      if (completed) {
        const auto &spectrum = std::get<dse::VerifiedResourceTimeSpectrum>(
            (*verified)->verification);
        const pnr::ResourceTimeTransition *verifiedEdge = nullptr;
        for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
             spectrum.scenarios)
          for (const pnr::ResourceTimeTransition &candidateEdge :
               scenario.transitions.transitions) {
            if (candidateEdge.parent.mapping != transitionMappings[0] ||
                candidateEdge.child.mapping != transitionMappings[1])
              continue;
            if (verifiedEdge)
              return invalid("resource-time application evidence repeats one "
                             "verified edge");
            verifiedEdge = &candidateEdge;
          }
        if (!verifiedEdge ||
            verifiedEdge->status != pnr::ResourceTimeTransitionStatus::Verified)
          return invalid("resource-time application evidence lost its exact "
                         "verified edge");
        resourceTimeTransitions.push_back(
            {*verifiedEdge, std::move(**verified), std::move(**childVerified)});
        continue;
      }
      if (!resourceTimeSpectrum)
        resourceTimeSpectrum = std::move(**verified);
    }

    if (!resourceTimeTransitions.empty())
      resourceTimeSpectrum = resourceTimeTransitions.front().parentSpectrum;

    if (resourceTimeTransitions.empty() && !resourceTimeSpectrum) {
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
            outcome.resourceTimeScheduleHintDigest);
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
  }
  std::optional<pnr::ResourceTimeTransitionGraph> resourceTimeTransitionGraph;
  if (selectedPlan) {
    resourceTimeTransitionGraph.emplace(pnr::ResourceTimeTransitionGraph{
        {imported->mapping.reference(), deployment->reference()}, {}, {}});
    resourceTimeTransitionGraph->endpoints.push_back(
        resourceTimeTransitionGraph->entry);
    const auto appendEndpoint = [&](const auto &endpoint) {
      if (!llvm::is_contained(resourceTimeTransitionGraph->endpoints, endpoint))
        resourceTimeTransitionGraph->endpoints.push_back(endpoint);
    };
    for (const ApplicationResourceTimeTransitionEvidence &evidence :
         resourceTimeTransitions) {
      appendEndpoint(evidence.transition.parent);
      appendEndpoint(evidence.transition.child);
      resourceTimeTransitionGraph->transitions.push_back(evidence.transition);
    }
    if (llvm::Error error = pnr::verifyResourceTimeTransitionGraph(
            *resourceTimeTransitionGraph, artifacts, blobs))
      return std::move(error);
  }
  emitElapsed(ApplicationBuildOperation::DeclarativeDeploymentFinalization,
              operationBegin);
  return ApplicationDeploymentArtifacts{abi->reference(),
                                        abi->constructionStatistics(),
                                        std::move(*selectedHardwareBindings),
                                        std::move(*selectedBinaries),
                                        std::move(resourceTimeTransitionGraph),
                                        std::move(resourceTimeTransitions),
                                        std::move(resourceTimeSpectrum),
                                        std::move(*deployment)};
}

} // namespace loom::application
