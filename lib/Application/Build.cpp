#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "ExecutionGlue.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/ProductionOwners.h"
#include "DSE/ResolvedConfigView.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/DeploymentPipeline.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Executable/ExecutableElf.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationDiagnostics.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Hardware/Implementation/FabricModel.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Runtime/FabricModelPlatform.h"
#include "Simulator/SpatialInvocation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>

namespace loom::application {
namespace {

using MonotonicClock = std::chrono::steady_clock;

constexpr std::uint64_t kPortableRiscVHostImageBase = 0x80000000;
constexpr std::uint64_t kExecutablePageBytes = 4096;

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
  if (mappings.size() != 1)
    return invalid("Deployment requires exactly one selected SystemMapping");
  return mappings.front();
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
    std::optional<dataflow::RootedGraphLaunchRef> spatialInvocation,
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
    if (spatialInvocation &&
        spatialInvocation->rootThreadLaunch == indexed.value())
      invocation = ThreadEntrySpatialInvocationBinding{*spatialInvocation};
    table.push_back({indexed.value(), indexed.index(), std::move(invocation)});
  }
  if (spatialInvocation &&
      llvm::none_of(table, [](const ThreadEntryBinding &entry) {
        return entry.spatialInvocation.has_value();
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
  llvm::Error workloadError = llvm::Error::success();
  for (dataflow::RootThreadLaunchRef root : *roots) {
    auto domain = view->projectRootThreadLogicalDomain(root);
    if (!domain)
      return domain.takeError();
    if (domain->coordinateRank != 0)
      return std::variant<std::vector<ArtifactRootReference>,
                          UnsupportedApplicationBuild>{
          UnsupportedApplicationBuild{
              ApplicationBuildUnsupportedKind::RootCoordinates,
              published.canonicalDataflow, root}};
    view->forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
      if (workloadError || launch.rootThreadLaunch != root)
        return;
      auto shapes = sim::projectSpatialSimulationBoundaryShapes(*view, launch);
      if (!shapes) {
        workloadError = shapes.takeError();
        return;
      }
      sim::SpatialSimulationWorkload workloadDraft{launch};
      workloadDraft.valueInputPlan.assign(shapes->valueInputs.size(),
                                          sim::RuntimeValueInput{});
      workloadDraft.observableContract.valueResults.resize(
          shapes->valueResults.size());
      std::iota(workloadDraft.observableContract.valueResults.begin(),
                workloadDraft.observableContract.valueResults.end(), 0);
      auto writableRoots =
          sim::projectSpatialInvocationWritableMemoryRoots(*view, launch);
      if (!writableRoots) {
        workloadError = writableRoots.takeError();
        return;
      }
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
    });
    if (workloadError)
      return std::move(workloadError);
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
          std::get_if<dse::IncompletePreMappingExploration>(&*preMapping))
    return ApplicationBuildPreparationOutcome{std::move(*incomplete)};
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

  std::vector<PreparedApplicationSoftware> preparedSoftware;
  std::vector<PreparedApplicationMappingAlternative> mappingAlternatives;
  preparedSoftware.reserve(completed.selected.size());
  mappingAlternatives.reserve(completed.selected.size());
  auto alternativePolicy = dse::JointDesignPolicy::get(
      1, 1, 1, request.jointPolicy.maximumSpatialMappingsPerPair());
  if (!alternativePolicy)
    return alternativePolicy.takeError();
  for (dse::SelectedPreMappingCompilation &selected : completed.selected) {
    auto published =
        frontend::publishPreMappingCompilation(selected.compilation, artifacts);
    if (!published)
      return published.takeError();
    auto workloads = publishApplicationWorkloads(
        *published, selected.compilation.canonicalDataflow,
        request.sourceInvocation.entrySymbol, artifacts);
    if (!workloads)
      return workloads.takeError();
    if (auto *unsupported =
            std::get_if<UnsupportedApplicationBuild>(&*workloads))
      return ApplicationBuildPreparationOutcome{std::move(*unsupported)};
    auto roots =
        std::get<std::vector<ArtifactRootReference>>(std::move(*workloads));
    auto mappingPlan = dse::buildJointDesignExplorationPlan(
        {{roots}, {request.system}}, request.physicalTimingProfiles,
        *alternativePolicy, request.resolvedConfig, artifacts);
    if (!mappingPlan)
      return mappingPlan.takeError();
    const ArtifactRootReference dataflow = published->canonicalDataflow;
    mappingAlternatives.push_back(
        {selected.preferenceRank, dataflow, std::move(*mappingPlan)});
    preparedSoftware.push_back(
        {selected.preferenceRank, std::move(*published), std::move(roots)});
  }
  return ApplicationBuildPreparationOutcome{PreparedApplicationBuild{
      std::move(request.sourceInvocation), std::move(preparedSoftware),
      std::move(completed.satisfiedEvidence),
      std::move(completed.planGenerateInvocations),
      std::move(mappingAlternatives)}};
}

llvm::Expected<dse::JointDesignExecution>
executeApplicationMapping(const PreparedApplicationBuild &prepared,
                          ApplicationMappingExecutionRequest request,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::MappingExecution);
  if (llvm::Error error = dse::registerProductionDseOwners())
    return std::move(error);
  if (request.journalRoot.empty())
    return invalid("Mapping execution requires a journal root");
  if (prepared.mappingAlternatives.empty())
    return invalid("Mapping execution has no software alternative");
  for (std::size_t index = 0; index != prepared.mappingAlternatives.size();
       ++index)
    if (prepared.mappingAlternatives[index].preferenceRank != index)
      return invalid("Mapping alternative preference ranks are not dense");

  std::vector<ArtifactRootReference> evidence = prepared.satisfiedEvidence;
  evidence.insert(evidence.end(), request.preexistingEvidence.begin(),
                  request.preexistingEvidence.end());
  auto scheduler = dse::SiteScheduler::create(std::move(request.siteCapacity));
  if (!scheduler)
    return scheduler.takeError();

  std::optional<dse::JointDesignExecution> lastCompletedNoFeasible;
  for (const PreparedApplicationMappingAlternative &alternative :
       prepared.mappingAlternatives) {
    const ResolvedConfig &config = alternative.plan.resolvedConfig;
    auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                         canonicalResolvedConfigBytes(config));
    if (!publishedConfig)
      return publishedConfig.takeError();
    if (*publishedConfig != resolvedConfigIdentity(config))
      return invalid("ResolvedConfig publication changed its identity");

    std::vector<ArtifactRootReference> semanticInputs =
        dse::projectJointDesignSemanticInputs(alternative.plan);
    auto closure = dse::DseRunClosure::get(request.producer, semanticInputs,
                                           config, evidence, artifacts);
    if (!closure)
      return closure.takeError();
    auto configView = dse::projectResolvedDseConfigView(config);
    if (!configView)
      return configView.takeError();

    llvm::SmallString<256> alternativeJournal(request.journalRoot);
    llvm::sys::path::append(
        alternativeJournal,
        llvm::toHex(closure->runKey().bytes(), /*LowerCase=*/true));
    if (std::error_code error =
            llvm::sys::fs::create_directories(alternativeJournal))
      return invalid("cannot create Mapping alternative journal: " +
                     error.message());
    auto journal =
        dse::openExecutionJournal(alternativeJournal, *closure, *configView);
    if (!journal)
      return journal.takeError();
    auto execution = dse::executeJointDesignExploration(
        alternative.plan, *closure, *journal, *scheduler,
        request.executionPolicy, artifacts, blobs);
    if (!execution)
      return execution.takeError();

    std::size_t mappingCount = 0;
    for (const dse::JointMappedPair &pair : execution->mappedPairs)
      mappingCount += pair.systemMappings.size();
    if (mappingCount != 0)
      return std::move(*execution);
    if (std::holds_alternative<dse::IncompleteDsePlanExecution>(
            execution->planExecution))
      return std::move(*execution);
    lastCompletedNoFeasible = std::move(*execution);
  }
  return std::move(*lastCompletedNoFeasible);
}

llvm::Expected<ApplicationDeploymentArtifacts> buildApplicationDeployment(
    const PreparedApplicationBuild &prepared,
    const dse::JointDesignExecution &mappingExecution,
    const llvm::Module &finalLinkedModule, ApplicationDeploymentRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::DeploymentConstruction);
  auto operationBegin = MonotonicClock::now();
  auto imported = importApplicationMapping(mappingExecution, artifacts);
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
  std::optional<std::size_t> activeTargetGroup;
  for (const auto indexed : llvm::enumerate(*roots)) {
    if (indexed.value().empty())
      continue;
    if (activeTargetGroup || indexed.value().size() != 1 ||
        indexed.value().front() != invocationPlan->root)
      return invalid(
          "initial dynamic dispatch requires one active InstructionCore "
          "target containing the reachable root");
    activeTargetGroup = indexed.index();
  }
  if (!activeTargetGroup)
    return invalid("SystemMapping selects no InstructionCore binary target");

  operationBegin = MonotonicClock::now();
  auto hostEntry = deriveHostProgramEntry(
      **software, prepared.sourceInvocation.entrySymbol, artifacts);
  if (!hostEntry)
    return hostEntry.takeError();
  hostEntry->abiSymbol = detail::applicationHostEntrySymbol.str();
  auto hostModule = detail::materializeHostDispatchModule(
      finalLinkedModule, prepared.sourceInvocation.entrySymbol,
      *invocationPlan);
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
    auto binary = buildInstructionBinary(
        finalLinkedModule, dataflowReference, indexed.value().binding(),
        (*roots)[indexed.index()], invocationPlan->graph, instructionImageBase,
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
