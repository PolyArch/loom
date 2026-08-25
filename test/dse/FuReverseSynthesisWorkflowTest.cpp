#include "DSE/FuReverseSynthesisWorkflow.h"
#include "Common/ArtifactStore.h"
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

dataflow::CanonicalDataflowArtifact
rootedAddSubProgram(mlir::MLIRContext &context) {
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
    %sum, %add_done = dataflow.graph.launch @add deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    %difference, %sub_done = dataflow.graph.launch @sub deps(%add_done)
        values(%sum, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %sub_done : none
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

} // namespace

int main() {
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

  auto program = rootedAddSubProgram(context);
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
      configView, closure, reopenedJournal, scheduler, policy, store, blobs));
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
  return EXIT_SUCCESS;
}
