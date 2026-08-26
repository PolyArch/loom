#include "DSE/FuReverseSynthesisWorkflow.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/FuReverseSynthesis.h"
#include "DSE/InvocationManifest.h"
#include "DSE/PlanExecutor.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/SiteScheduler.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Deployment/Deployment.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "DeploymentTestSupport.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr llvm::StringLiteral testName = "fu-reverse-synthesis-workflow";

[[noreturn]] void fail(const std::string &message) {
  std::cerr << testName.str() << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message.str());
}

void requireError(llvm::Error error, llvm::StringRef expected) {
  if (!error)
    fail("expected an error containing '" + expected.str() + "'");
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(expected))
    fail("unexpected error: " + message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact parseDataflow(mlir::MLIRContext &context,
                                                  llvm::StringRef source) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact loadDataflow(mlir::MLIRContext &context,
                                                 llvm::StringRef path) {
  auto source = llvm::MemoryBuffer::getFile(path);
  if (!source)
    fail("cannot read Dataflow graph set: " + source.getError().message());
  return parseDataflow(context, (*source)->getBuffer());
}

dataflow::CanonicalDataflowArtifact
rootlessAddProgram(mlir::MLIRContext &context) {
  return parseDataflow(context, R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @add(%start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = arith.addi %lhs, %rhs : i32
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact
partiallyRootedAddSubProgram(mlir::MLIRContext &context) {
  return parseDataflow(context, R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @add(%start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = arith.addi %lhs, %rhs : i32
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.graph private @sub(%start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = arith.subi %lhs, %rhs : i32
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %lhs: i32, %rhs: i32) ctrl (%ctrl: none) {
    %sum, %done = dataflow.graph.launch @add deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %lhs = arith.constant 19 : i32
    %rhs = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%lhs, %rhs)
        : (i32, i32) -> !dataflow.thread_token
    return
  }
}
)mlir");
}

loom::ResolvedConfig workflowConfig() {
  loom::ResolvedConfig config =
      take(loom::resolveConfigProfile("quick_explore"));
  config.dse.techMapping.candidatePublicationLimit = 2;
  config.dse.spatialPnr.search.initializer.seedAttemptCount = 1;
  config.dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::ExhaustConfiguredWork;
  config.dse.systemPnr.search.initializer.seedAttemptCount = 1;
  config.dse.systemPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::ExhaustConfiguredWork;
  return config;
}

void requireTypedReachabilityRejection(llvm::Error error) {
  if (!error)
    fail("unsupported graph reachability passed verification");
  std::optional<loom::dse::FuReverseSynthesisFailure> failure;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error), [&](const loom::dse::FuReverseSynthesisError &error) {
        failure = error.failure();
      });
  if (remaining)
    fail("unsupported graph reachability returned an untyped error: " +
         llvm::toString(std::move(remaining)));
  require(
      failure ==
          loom::dse::FuReverseSynthesisFailure::UnsupportedGraphReachability,
      "graph reachability returned the wrong typed failure");
}

void requireTypedReachabilityRejection(
    llvm::Expected<loom::dse::FuReverseSynthesisCandidateWorkflow> workflow) {
  if (workflow)
    fail("unsupported graph reachability entered System PnR");
  requireTypedReachabilityRejection(workflow.takeError());
}

class EmptySystemMappingExecutor final
    : public loom::dse::detail::DsePlanWorkExecutor {
public:
  explicit EmptySystemMappingExecutor(
      const loom::dse::CompletedDsePlanExecution &execution)
      : invocations_(execution.generateInvocations().begin(),
                     execution.generateInvocations().end()),
        work_(execution.generateWorkSummaries().begin(),
              execution.generateWorkSummaries().end()) {}

  bool shouldStopBeforeDispatch() const override { return false; }

  llvm::Expected<std::vector<loom::dse::PromotionEvidenceExecutionResult>>
  execute(llvm::ArrayRef<loom::dse::PromotionEvidenceExecutionTask>,
          const loom::ArtifactStore &, const loom::BlobStore &) override {
    return std::vector<loom::dse::PromotionEvidenceExecutionResult>{};
  }

  llvm::Expected<loom::dse::CandidateGeneratorProviderResult>
  executeGenerate(std::uint64_t planNodeOrdinal,
                  llvm::ArrayRef<loom::dse::CandidateGeneratorInputBinding>,
                  llvm::ArrayRef<loom::dse::CandidateGeneratorOutputDemand>,
                  const loom::dse::ResolvedCandidateGeneratorBinding &,
                  const loom::ArtifactStore &,
                  const loom::BlobStore &) override {
    return resultFor(planNodeOrdinal);
  }

  llvm::Expected<std::vector<loom::dse::CandidateGeneratorProviderResult>>
  executeGenerateBatch(
      llvm::ArrayRef<loom::dse::detail::DseGenerateExecutionTask> tasks,
      const loom::ArtifactStore &, const loom::BlobStore &) override {
    std::vector<loom::dse::CandidateGeneratorProviderResult> results;
    results.reserve(tasks.size());
    for (const auto &task : tasks) {
      auto result = resultFor(task.planNodeOrdinal);
      if (!result)
        return result.takeError();
      results.push_back(std::move(*result));
    }
    return results;
  }

  llvm::Error beginPromotion(
      std::uint64_t, llvm::ArrayRef<loom::ArtifactRootReference>,
      llvm::ArrayRef<loom::dse::EvidenceObligationTemplateRef>) override {
    return llvm::Error::success();
  }

private:
  llvm::Expected<loom::dse::CandidateGeneratorProviderResult>
  resultFor(std::uint64_t planNodeOrdinal) const {
    const auto invocation =
        llvm::find_if(invocations_, [&](const auto &candidate) {
          return candidate.planNodeOrdinal == planNodeOrdinal;
        });
    const auto work = llvm::find_if(work_, [&](const auto &candidate) {
      return candidate.planNodeOrdinal == planNodeOrdinal;
    });
    if (invocation == invocations_.end() || work == work_.end())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "missing production invocation record");
    auto outputs = invocation->outputBindings;
    auto lineage = invocation->lineageEdges;
    if (planNodeOrdinal == 3) {
      for (auto &output : outputs)
        output.artifacts.clear();
      lineage.clear();
    }
    return loom::dse::CandidateGeneratorProviderResult{
        loom::dse::CompletedCandidateGeneratorResult{std::move(outputs),
                                                     std::move(lineage)},
        work->units, std::nullopt, false};
  }

  std::vector<loom::dse::GenerateInvocationRecord> invocations_;
  std::vector<loom::dse::GenerateInvocationWorkSummary> work_;
};

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one canonical Dataflow graph-set path");
  loom::deployment::test::TemporaryTree tree(testName);
  loom::ArtifactStore store(tree.path("artifacts"));
  loom::BlobStore blobs(tree.path("blobs"));
  mlir::MLIRContext context = makeContext();

  auto rootless = rootlessAddProgram(context);
  const auto rootlessReference =
      take(dataflow::publishCanonicalDataflow(rootless, store));
  requireTypedReachabilityRejection(
      loom::dse::buildFuReverseSynthesisCandidateWorkflow(
          rootlessReference, workflowConfig(), store));
  auto partiallyRooted = partiallyRootedAddSubProgram(context);
  const auto partiallyRootedReference =
      take(dataflow::publishCanonicalDataflow(partiallyRooted, store));
  requireTypedReachabilityRejection(
      loom::dse::buildFuReverseSynthesisCandidateWorkflow(
          partiallyRootedReference, workflowConfig(), store));

  auto program = loadDataflow(context, argv[1]);
  const auto programReference =
      take(dataflow::publishCanonicalDataflow(program, store));
  auto workflow = take(loom::dse::buildFuReverseSynthesisCandidateWorkflow(
      programReference, workflowConfig(), store));
  auto configView =
      take(loom::dse::projectResolvedDseConfigView(workflow.resolvedConfig()));
  const auto storedConfig = take(
      store.put(loom::ResolvedConfig::artifactSchema,
                loom::canonicalResolvedConfigBytes(workflow.resolvedConfig())));
  require(storedConfig ==
              loom::resolvedConfigIdentity(workflow.resolvedConfig()),
          "workflow ResolvedConfig publication changed identity");
  auto producer = take(loom::dse::DseProducerSemanticBuildIdentity::get(
      "loom.test.fu_reverse_synthesis_workflow.v1"));
  auto closure = take(
      loom::dse::DseRunClosure::get(std::move(producer), {programReference},
                                    workflow.resolvedConfig(), {}, store));
  const std::string journalPath = tree.path("journal");
  std::filesystem::create_directories(journalPath);
  auto journal =
      take(loom::dse::openExecutionJournal(journalPath, closure, configView));
  auto scheduler = take(loom::dse::SiteScheduler::create(
      take(loom::dse::SiteCapacity::get(2, 0, 0))));
  const auto policy = take(loom::dse::PlanExecutionPolicy::get(
      1, take(loom::dse::SiteResourceClaim::get(1, 0, 0))));
  auto outcome = take(loom::dse::executeDsePlan(
      configView, closure, journal, scheduler, policy, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&outcome);
  if (!completed) {
    const auto &incomplete =
        std::get<loom::dse::IncompleteDsePlanExecution>(outcome);
    fail("bounded workflow did not complete at node " +
         std::to_string(incomplete.nodeOrdinal()) + ": " +
         loom::dse::toString(incomplete.reason()).str());
  }
  require(completed->generateInvocations().size() == 5,
          "bounded workflow did not execute every production owner");
  require(
      take(loom::dse::classifyFuReverseSynthesisWorkflow(workflow,
                                                         *completed)) ==
          loom::dse::FuReverseSynthesisWorkflowDisposition::CompleteCandidate,
      "complete workflow was classified as no-feasible");
  EmptySystemMappingExecutor emptySystemMapping(*completed);
  auto noSystemOutcome = take(loom::dse::detail::executeDsePlanWithWorkExecutor(
      configView, store, blobs, &emptySystemMapping));
  const auto *noSystem =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&noSystemOutcome);
  require(noSystem && noSystem->resolve(workflow.systemMappings()).empty() &&
              !noSystem->resolve(workflow.portableRtlImplementations()).empty(),
          "terminal-output fixture did not isolate empty System PnR");
  require(
      take(
          loom::dse::classifyFuReverseSynthesisWorkflow(workflow, *noSystem)) ==
          loom::dse::FuReverseSynthesisWorkflowDisposition::NoFeasibleCandidate,
      "empty System PnR was classified as a complete candidate");
  auto genericOutcome = take(
      loom::dse::projectDsePlanInvocationOutcome(configView, noSystemOutcome));
  require(std::holds_alternative<loom::dse::InvocationCompletedSelection>(
              genericOutcome),
          "generic plan no longer exposes independent terminal outputs");
  auto artifacts = take(loom::dse::projectFuReverseSynthesisWorkflowArtifacts(
      workflow, *completed, store, blobs));
  require(artifacts.techMappings.size() == 2 &&
              artifacts.spatialMappings.size() == 2 &&
              artifacts.jointSpatialMappings.size() == 1,
          "workflow lost per-graph evidence or the deployable joint mapping");
  auto unreachableSubstitution = artifacts;
  unreachableSubstitution.dataflow = partiallyRootedReference;
  requireTypedReachabilityRejection(
      loom::dse::verifyFuReverseSynthesisWorkflowArtifacts(
          unreachableSubstitution, store, blobs));

  auto module =
      take(loom::fabric::importEntireFabricRoot(artifacts.module, store));
  auto system =
      take(loom::fabric::importEntireFabricRoot(artifacts.system, store));
  auto normalizedTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          module.view()));
  auto alternateTiming = take(loom::fabric::createFabricPhysicalTimingProfile(
      module.view(),
      loom::fabric::FabricPhysicalTimingProfileKind::NormalizedHeuristic,
      "loom.test.alternate_timing", "normalized", "alternate",
      normalizedTiming.requiredCombinationalDelayQuanta(),
      normalizedTiming.traversals()));
  const auto alternateTimingReference = take(
      loom::fabric::publishFabricPhysicalTimingProfile(alternateTiming, store));
  auto timingSubstitution = artifacts;
  timingSubstitution.physicalTimingProfiles = {alternateTimingReference};
  requireError(loom::dse::verifyFuReverseSynthesisWorkflowArtifacts(
                   timingSubstitution, store, blobs),
               "exact normalized Module projection");

  auto alternateAbiDraft =
      take(loom::hardware::derivePackedConfigurationABIDraft(system, context));
  require(!alternateAbiDraft.programmingUnits.empty(),
          "bounded System has no packed programming unit");
  ++alternateAbiDraft.programmingUnits.front().payloadBitCount;
  auto alternateAbi = take(loom::hardware::finalizeConfigurationABI(
      std::move(alternateAbiDraft), store));
  auto abiSubstitution = artifacts;
  abiSubstitution.configurationAbi = alternateAbi.reference();
  requireError(loom::dse::verifyFuReverseSynthesisWorkflowArtifacts(
                   abiSubstitution, store, blobs),
               "exact packed System projection");

  auto reopenedJournal =
      take(loom::dse::openExecutionJournal(journalPath, closure, configView));
  auto replay = take(loom::dse::resumeDsePlan(
      configView, closure, reopenedJournal, scheduler, policy, store, blobs,
      loom::dse::InvocationManifestRetention::Release));
  const auto *replayed =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&replay);
  require(replayed && replayed->generateInvocations().size() == 5,
          "workflow journal did not replay a complete invocation");
  for (std::size_t ordinal = 0; ordinal != 5; ++ordinal)
    require(!replayed->generateInvocationWasDispatched(ordinal),
            "workflow replay redispatched finalized provider work");
  require(replayed->resolve(workflow.spatialMappings()) ==
                  completed->resolve(workflow.spatialMappings()) &&
              replayed->resolve(workflow.jointSpatialMappings()) ==
                  completed->resolve(workflow.jointSpatialMappings()) &&
              replayed->resolve(workflow.systemMappings()) ==
                  completed->resolve(workflow.systemMappings()) &&
              replayed->resolve(workflow.portableRtlImplementations()) ==
                  completed->resolve(workflow.portableRtlImplementations()),
          "workflow replay changed Mapping or RTL identities");

  loom::ArtifactStore replayStore(tree.path("artifacts"));
  loom::BlobStore replayBlobs(tree.path("blobs"));
  if (llvm::Error error = loom::dse::verifyFuReverseSynthesisWorkflowArtifacts(
          artifacts, replayStore, replayBlobs))
    fail("independent workflow import failed: " +
         llvm::toString(std::move(error)));

  require(artifacts.systemMappings.size() == 1,
          "bounded workflow did not select one SystemMapping");
  auto systemMapping = take(loom::mapping::importSystemMapping(
      artifacts.systemMappings.front(), store));
  std::vector<loom::hardware::FinalizedHardwareImplementation> implementations;
  for (const auto &reference : artifacts.portableRtlImplementations)
    implementations.push_back(take(
        loom::hardware::importHardwareImplementation(reference, store, blobs)));
  auto deployment = loom::deployment::test::buildMappedSystemDeployment(
      testName, program, system, systemMapping, implementations, {}, store,
      blobs, tree);
  auto importedDeployment = take(loom::deployment::importDeployment(
      deployment.reference(), replayStore, replayBlobs));
  require(importedDeployment.deployment().systemMapping() ==
              systemMapping.reference(),
          "Deployment selected another SystemMapping");
  require(!importedDeployment.deployment().hardwareBindings().empty(),
          "Deployment omitted the mapped SpatialCore implementation");
  for (const auto &binding : importedDeployment.deployment().hardwareBindings())
    require(llvm::is_contained(artifacts.portableRtlImplementations,
                               binding.hardwareImplementation),
            "Deployment selected an implementation outside portable RTL");

  auto configurationAbi = take(loom::hardware::importConfigurationABI(
      artifacts.configurationAbi, replayStore));
  std::uint64_t directImageCount = 0;
  std::uint64_t localImageCount = 0;
  for (const loom::ArtifactRootReference &reference :
       importedDeployment.deployment().configurationImages()) {
    const auto image = take(loom::deployment::importHardwareConfigurationImage(
        reference, replayStore));
    const loom::hardware::ProgrammingUnit *unit =
        configurationAbi.abi().findProgrammingUnit(
            image.image().programmingUnitId());
    require(unit != nullptr, "Deployment image lost its programming unit");
    const loom::hardware::ProgrammingUnitOccurrenceScope scope =
        loom::hardware::deriveProgrammingUnitOccurrenceScope(*unit);
    if (scope.includesDirectSystemResources && scope.spatialCores.empty())
      ++directImageCount;
    else if (!scope.includesDirectSystemResources &&
             scope.spatialCores.size() == 1)
      ++localImageCount;
    else
      fail("Deployment image has a mixed programming-unit scope");
  }
  require(directImageCount != 0 && localImageCount != 0,
          "Deployment omitted direct or local configuration images");

  const auto &spatialLaunch =
      importedDeployment.deployment().spatialLaunchImage();
  require(spatialLaunch.has_value(),
          "mapped reverse-FU Deployment has no SpatialLaunchImage");
  const auto &spatialBytes = spatialLaunch->canonicalBytes().bytes();
  auto spatialJson = take(llvm::json::parse(
      llvm::StringRef(reinterpret_cast<const char *>(spatialBytes.data()),
                      spatialBytes.size())));
  auto *spatialRoot = spatialJson.getAsObject();
  auto *payload = spatialRoot ? spatialRoot->getObject("payload") : nullptr;
  auto *rows = payload ? payload->getArray("rows") : nullptr;
  require(rows && !rows->empty(),
          "SpatialLaunchImage has no mapped graph rows");
  std::uint64_t targetCaseCount = 0;
  for (llvm::json::Value &rowValue : *rows) {
    auto *row = rowValue.getAsObject();
    auto *targetCases = row ? row->getArray("target_cases") : nullptr;
    require(targetCases && !targetCases->empty(),
            "SpatialLaunchImage row has no target case");
    for (llvm::json::Value &caseValue : *targetCases) {
      auto *targetCase = caseValue.getAsObject();
      auto *references =
          targetCase ? targetCase->getArray("required_configuration_image_refs")
                     : nullptr;
      require(references != nullptr,
              "SpatialLaunchImage target has no configuration closure");
      std::vector<loom::ArtifactRootReference> required;
      for (llvm::json::Value &referenceValue : *references) {
        auto *object = referenceValue.getAsObject();
        require(object != nullptr,
                "SpatialLaunchImage contains a malformed image reference");
        required.push_back(take(loom::parseArtifactRootReferenceJson(*object)));
      }
      require(llvm::ArrayRef<loom::ArtifactRootReference>(required) ==
                  importedDeployment.deployment().configurationImages(),
              "single-core target omitted a global or local image");
      ++targetCaseCount;
    }
  }
  require(targetCaseCount != 0,
          "SpatialLaunchImage did not expose a target case");
  return EXIT_SUCCESS;
}
