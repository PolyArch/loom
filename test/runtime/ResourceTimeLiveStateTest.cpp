#include "Application/ResourceTimeExecution.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "DeploymentTestSupport.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "PnR/System/SystemMappingMigration.h"
#include "Runtime/InProcessPlatform.h"

#include "MappedRtlSimulationTestSupport.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Mapping/IR/MappingDialect.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;

namespace {

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    deployment::test::fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    deployment::test::fail(test, llvm::toString(std::move(error)));
}

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mapping::MappingDialect,
                  fabric::FabricDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact
buildLiveStateDataflow(llvm::StringRef test, mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @state_update(
      %start: none, %increment: i32, %memory: memref<16xi32>) -> i32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %address = arith.constant 0 : index
    %prior, %read = dataflow.load %memory[%address] %start : memref<16xi32>
    %updated = arith.addi %prior, %increment : i32
    %stored = dataflow.store %memory[%address] %updated %read : memref<16xi32>
    dataflow.graph.return values(%updated : i32) streams() memories()
        complete(%stored : none)
  }
  dataflow.thread private @sentinel domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %memory: memref<16xi32>, %increment: i32) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @state_update deps(%ctrl)
        values(%increment) stream_inputs() memories(%memory) stream_outputs()
        : (none, i32, memref<16xi32>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%memory: memref<16xi32>) {
    %producer_increment = arith.constant 7 : i32
    %consumer_increment = arith.constant 0 : i32
    %sentinel = dataflow.thread.launch @sentinel()
        : () -> !dataflow.thread_token
    %producer = dataflow.thread.launch @worker(
        %memory, %producer_increment) wait(%sentinel)
        : (memref<16xi32>, i32) -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @worker(
        %memory, %consumer_increment) wait(%producer)
        : (memref<16xi32>, i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  deployment::test::require(test, static_cast<bool>(module),
                            "cannot parse live-state Dataflow fixture");
  return take(test, dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildOrderedChannelDataflow(llvm::StringRef test, mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @sentinel domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @ordered_worker
      domain(#dataflow.thread_domain<dense>)(
          %ordered: !dataflow.channel<i32>, %value: i32) ctrl (%ctrl: none) {
    dataflow.channel.send %ordered, %value : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @ordered_consumer
      domain(#dataflow.thread_domain<dense>)(
          %ordered: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %value = dataflow.channel.receive %ordered : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  func.func private @host() {
    %ordered = dataflow.channel.create : !dataflow.channel<i32>
    %payload = arith.constant 23 : i32
    %sentinel = dataflow.thread.launch @sentinel()
        : () -> !dataflow.thread_token
    %producer = dataflow.thread.launch @ordered_worker(%ordered, %payload)
        wait(%sentinel)
        : (!dataflow.channel<i32>, i32) -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @ordered_consumer(%ordered)
        wait(%producer)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  deployment::test::require(test, static_cast<bool>(module),
                            "cannot parse ordered-channel Dataflow fixture");
  return take(test, dataflow::finalizeCanonicalDataflow(*module));
}

struct OrderedRoots final {
  dataflow::RootThreadLaunchRef sentinel;
  dataflow::RootThreadLaunchRef producer;
  dataflow::RootThreadLaunchRef consumer;
};

OrderedRoots
deriveOrderedRoots(llvm::StringRef test,
                   const dataflow::CanonicalDataflowProgramView &dataflow) {
  deployment::test::require(test, dataflow.rootThreadLaunches().size() == 3,
                            "ordered replay requires three root launches");
  std::vector<dataflow::RootThreadLaunchRef> roots;
  std::vector<dataflow::EventFamilyKey> events;
  for (const auto &root : dataflow.rootThreadLaunches()) {
    roots.push_back(root.ref);
    events.push_back(dataflow::rootThreadStartEventFamily(root.ref));
    events.push_back(dataflow::rootThreadCompletionEventFamily(root.ref));
  }
  const auto progress =
      take(test, mapping::freezeMappingProgressModel(dataflow, events));
  const auto precedes = [&](dataflow::RootThreadLaunchRef lhs,
                            dataflow::RootThreadLaunchRef rhs) {
    return take(test,
                mapping::mappingEventPrecedes(
                    progress, dataflow::rootThreadCompletionEventFamily(lhs),
                    dataflow::rootThreadStartEventFamily(rhs)));
  };
  std::vector<std::pair<unsigned, dataflow::RootThreadLaunchRef>> ranked;
  ranked.reserve(roots.size());
  for (const auto root : roots)
    ranked.emplace_back(
        static_cast<unsigned>(llvm::count_if(roots,
                                             [&](const auto candidate) {
                                               return root != candidate &&
                                                      precedes(root, candidate);
                                             })),
        root);
  llvm::sort(ranked, [](const auto &lhs, const auto &rhs) {
    return lhs.first > rhs.first;
  });
  for (std::size_t index = 0; index != roots.size(); ++index)
    roots[index] = ranked[index].second;
  deployment::test::require(
      test,
      ranked[0].first == 2 && ranked[1].first == 1 && ranked[2].first == 0 &&
          precedes(roots[0], roots[1]) && precedes(roots[1], roots[2]),
      "Dataflow fixture did not preserve the ordered root chain");
  return {roots[0], roots[1], roots[2]};
}

std::vector<fabric::AccCoreOccurrenceRef>
rootTargets(llvm::StringRef test,
            const dataflow::CanonicalDataflowProgramView &dataflow,
            const OrderedRoots &roots, fabric::AccCoreOccurrenceRef sentinel,
            fabric::AccCoreOccurrenceRef producer,
            fabric::AccCoreOccurrenceRef consumer) {
  std::vector<fabric::AccCoreOccurrenceRef> result;
  result.reserve(dataflow.rootThreadLaunches().size());
  for (const auto &root : dataflow.rootThreadLaunches()) {
    if (root.ref == roots.sentinel)
      result.push_back(sentinel);
    else if (root.ref == roots.producer)
      result.push_back(producer);
    else {
      deployment::test::require(test, root.ref == roots.consumer,
                                "fixture contains an unexpected root launch");
      result.push_back(consumer);
    }
  }
  return result;
}

fabric::FabricPhysicalOccurrenceOwnerRef
physicalCore(llvm::StringRef test, fabric::AccCoreOccurrenceRef core) {
  return take(test, fabric::FabricPhysicalOccurrenceOwnerRef::create(
                        fabric::FabricInventoryOwnerRef::of(core)));
}

std::vector<ArtifactIdentity> implementationIdentities(
    llvm::StringRef test, const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::vector<std::optional<ArtifactIdentity>> indexed(
      deployment.deployment().hardwareBindings().size());
  for (const auto &binding : deployment.deployment().hardwareBindings()) {
    auto runtimeBinding =
        take(test, runtime::importRuntimePlatformBinding(
                       binding.runtimePlatformBinding, artifacts, blobs));
    const auto *reported = std::get_if<runtime::HardwareReportedIdentity>(
        &runtimeBinding.binding().identityVerification());
    deployment::test::require(test, reported != nullptr,
                              "replay requires reported provider identity");
    bool matched = false;
    for (std::size_t ordinal = 0; ordinal != indexed.size(); ++ordinal) {
      if (!(reported->implementationIdentityEndpoint ==
            runtime::inProcessRuntimeEndpoint(
                runtime::RuntimeEndpointClass::Identity, ordinal)))
        continue;
      deployment::test::require(test, !indexed[ordinal],
                                "provider identity endpoint is duplicated");
      indexed[ordinal] = binding.hardwareImplementation.artifact;
      matched = true;
      break;
    }
    deployment::test::require(test, matched,
                              "provider identity endpoint is out of range");
  }
  std::vector<ArtifactIdentity> result;
  result.reserve(indexed.size());
  for (const auto &identity : indexed) {
    deployment::test::require(test, identity.has_value(),
                              "provider identity coverage is incomplete");
    result.push_back(*identity);
  }
  return result;
}

pnr::ResourceTimeTransition
makeTransitionDraft(const ArtifactRootReference &dataflow,
                    const OrderedRoots &roots,
                    const mapping::FinalizedSystemMapping &parentMapping,
                    const deployment::FinalizedDeployment &parentDeployment,
                    const mapping::FinalizedSystemMapping &childMapping,
                    const deployment::FinalizedDeployment &childDeployment,
                    fabric::AccCoreOccurrenceRef producerCore,
                    fabric::AccCoreOccurrenceRef consumerCore) {
  pnr::ResourceTimeTransition transition;
  transition.trigger =
      dataflow::rootThreadCompletionEventFamily(roots.producer);
  transition.safePoint = pnr::ResourceTimeSafePointReference{
      dataflow, pnr::ResourceTimeSafePointKind::Completion};
  transition.parent = {parentMapping.reference(), parentDeployment.reference()};
  transition.child = {childMapping.reference(), childDeployment.reference()};
  transition.beforeActive = {
      {roots.producer, {physicalCore("makeTransitionDraft", producerCore)}}};
  transition.afterActive = {
      {roots.consumer, {physicalCore("makeTransitionDraft", consumerCore)}}};
  transition.completedBefore = {roots.sentinel};
  return transition;
}

template <typename T>
void expectSelectionError(llvm::StringRef test, llvm::Expected<T> value,
                          runtime::ResourceTimeSelectionErrorReason reason) {
  if (value)
    deployment::test::fail(test, "accepted an invalid lifecycle event");
  std::optional<runtime::ResourceTimeSelectionErrorReason> observed;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const runtime::ResourceTimeSelectionError &error) {
        observed = error.reason();
      },
      [&](const llvm::ErrorInfoBase &error) {
        deployment::test::fail(test, error.message());
      });
  deployment::test::require(test, observed == reason,
                            "lifecycle event returned another typed error");
}

void expectTransitionRefusal(llvm::StringRef test,
                             llvm::Expected<pnr::ResourceTimeTransition> value,
                             pnr::ResourceTimeTransitionRefusalReason reason) {
  if (value)
    deployment::test::fail(test, "accepted non-migratable live state");
  std::optional<pnr::ResourceTimeTransitionRefusalReason> observed;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const pnr::ResourceTimeTransitionRefusal &error) {
        observed = error.reason();
      },
      [&](const llvm::ErrorInfoBase &error) {
        deployment::test::fail(test, error.message());
      });
  deployment::test::require(test, observed == reason,
                            "finalizer returned another typed refusal");
}

void copiedLiveStateExecutesAtTheApplicationSafePoint() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  mlir::MLIRContext context = makeContext();
  const auto dataflowArtifact = buildLiveStateDataflow(test, context);
  const auto dataflow = take(test, dataflowArtifact.view());
  const auto roots = deriveOrderedRoots(test, dataflow);
  deployment::test::require(test, dataflow.logicalMemoryRoots().size() == 1,
                            "replay requires one logical-memory owner");
  const auto byteCount =
      take(test,
           dataflow.staticMemoryByteExtent(dataflow::LogicalMemoryRootOrViewRef{
               dataflow.logicalMemoryRoots().front().ref}));
  deployment::test::require(test, byteCount == std::optional<std::uint64_t>(64),
                            "replay logical memory is not exactly 64 bytes");

  const auto hardware = eda::test::buildMappedSpatialHardwareFixture(
      test, dataflowArtifact, context, artifacts, blobs,
      deployment::test::MappedSpatialSystemSpec{2});
  const auto system =
      take(test, fabric::requireSystemRoot(hardware.system.view()));
  const auto cores = system.artifact().accCoreOccurrences();
  deployment::test::require(test, cores.size() == 2,
                            "replay System does not have two AccCores");
  const std::vector<ArtifactRootReference> spatialMappings = {
      hardware.spatialMapping.reference()};
  const auto parentTargets =
      rootTargets(test, dataflow, roots, cores[1], cores[0], cores[0]);
  const auto childTargets =
      rootTargets(test, dataflow, roots, cores[0], cores[0], cores[1]);
  const auto parentMapping = deployment::test::buildMappedSystemMapping(
      test, dataflowArtifact, hardware.system, spatialMappings, artifacts,
      parentTargets);
  const auto childMapping = deployment::test::buildMappedSystemMapping(
      test, dataflowArtifact, hardware.system, spatialMappings, artifacts,
      childTargets);
  deployment::test::require(
      test, parentMapping.reference() != childMapping.reference(),
      "replay endpoints have one SystemMapping");
  const auto parent = deployment::test::buildMappedSystemDeployment(
      test, dataflowArtifact, hardware.system, parentMapping,
      hardware.implementations, {}, artifacts, blobs, tree);
  const auto child = deployment::test::buildMappedSystemDeployment(
      test, dataflowArtifact, hardware.system, childMapping,
      hardware.implementations, {}, artifacts, blobs, tree);
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, dataflow.identity()};
  const auto transition =
      take(test, pnr::finalizeResourceTimeTransition(
                     makeTransitionDraft(dataflowReference, roots,
                                         parentMapping, parent, childMapping,
                                         child, cores[0], cores[1]),
                     artifacts, blobs));
  deployment::test::require(
      test,
      transition.logicalMemories.size() == 1 &&
          transition.logicalMemories.front().migration ==
              pnr::ResourceTimeLiveStateMigration::Copied &&
          transition.logicalMemories.front().parentBinding !=
              transition.logicalMemories.front().childBinding &&
          transition.logicalMemories.front().migrationTimePicoseconds != 0 &&
          transition.reprogrammingTimePicoseconds.value_or(0) != 0 &&
          transition.migrationTimePicoseconds.value_or(0) != 0,
      "finalizer did not derive one nonzero copied correspondence");
  const auto execution =
      take(test, pnr::deriveResourceTimeTransitionExecutionPlan(
                     transition, artifacts, blobs));
  std::uint64_t changedWords = 0;
  for (const auto &image : execution.configurationImages)
    changedWords += image.changedWordOrdinals.size();
  deployment::test::require(
      test,
      execution.logicalMemoryCopies.size() == 1 &&
          execution.logicalMemoryCopies.front().byteCount == 64 &&
          !execution.configurationImages.empty() && changedWords != 0,
      "execution projection lost the copy or configuration-word delta");
  const auto sourceTarget =
      take(test, pnr::canonicalResourceTimeMemoryTargetBytes(
                     execution.logicalMemoryCopies.front().source));
  const auto destinationTarget =
      take(test, pnr::canonicalResourceTimeMemoryTargetBytes(
                     execution.logicalMemoryCopies.front().destination));
  deployment::test::require(test, sourceTarget != destinationTarget,
                            "copy endpoints name one physical target");
  const auto cost =
      runtime::inProcessRuntimeProviderDescriptor().resourceTimeCostModel;
  deployment::test::require(test, cost.has_value(),
                            "in-process provider has no transition cost model");
  deployment::test::require(
      test,
      execution.migrationTimePicoseconds ==
              cost->memoryCopySetupPicoseconds +
                  64 * cost->memoryCopyBytePicoseconds &&
          execution.reprogrammingTimePicoseconds ==
              changedWords * cost->configurationWordPicoseconds +
                  execution.configurationImages.size() *
                      cost->configurationCommitPicoseconds,
      "transition cost was not derived from the provider model");

  const auto parentIdentities =
      implementationIdentities(test, parent, artifacts, blobs);
  deployment::test::require(
      test,
      parentIdentities ==
          implementationIdentities(test, child, artifacts, blobs),
      "transition endpoints disagree on provider identity ordering");
  auto provider = take(test, runtime::createInProcessRuntimeProvider(
                                 {{parentIdentities, std::nullopt, {}}}));
  auto loaded = take(
      test, runtime::loadDeployment(parent, {provider, 0}, artifacts, blobs));
  std::array<std::uint8_t, 64> producerState;
  std::iota(producerState.begin(), producerState.end(), std::uint8_t{0});
  const pnr::ResourceTimeTransitionGraph graph{
      transition.parent, {transition.parent, transition.child}, {transition}};
  auto replay =
      take(test,
           application::ApplicationResourceTimeExecutionSession::createPrepared(
               graph, loaded, artifacts, blobs));
  const auto apply = [&](dataflow::EventFamilyKey event,
                         std::uint64_t occurrence, std::uint64_t tick) {
    return replay.apply({event, occurrence, {tick, 0}}, loaded);
  };
  const auto sentinelStart = take(
      test, apply(dataflow::rootThreadStartEventFamily(roots.sentinel), 1, 1));
  const auto sentinelCompletion = take(
      test,
      apply(dataflow::rootThreadCompletionEventFamily(roots.sentinel), 1, 2));
  const auto producerStart = take(
      test, apply(dataflow::rootThreadStartEventFamily(roots.producer), 2, 3));
  // Publication after preparation proves that the provider reads live state
  // only when the compiler-known completion edge commits.
  requireSuccess(test, provider->setLiveMemoryTarget(
                           0, execution.logicalMemoryCopies.front().source,
                           producerState));
  const auto producerCompletion = take(
      test,
      apply(dataflow::rootThreadCompletionEventFamily(roots.producer), 2, 4));
  deployment::test::require(
      test,
      sentinelStart.outcome ==
              application::ApplicationResourceTimeEventOutcome::RootStarted &&
          sentinelCompletion.outcome ==
              application::ApplicationResourceTimeEventOutcome::
                  NoLegalTransition &&
          producerStart.outcome ==
              application::ApplicationResourceTimeEventOutcome::RootStarted &&
          producerCompletion.outcome ==
              application::ApplicationResourceTimeEventOutcome::SelectedChild &&
          loaded.deployment().reference() == child.reference(),
      "application replay did not select the prepared child at the safe point");

  expectSelectionError(
      test, apply(dataflow::rootThreadStartEventFamily(roots.producer), 4, 5),
      runtime::ResourceTimeSelectionErrorReason::ActiveSetMismatch);
  (void)take(test,
             apply(dataflow::rootThreadStartEventFamily(roots.consumer), 3, 6));
  const auto copied =
      take(test, provider->readLiveMemoryTarget(
                     0, execution.logicalMemoryCopies.front().destination));
  const std::uint64_t consumerChecksum =
      std::accumulate(copied.begin(), copied.end(), std::uint64_t{0});
  deployment::test::require(
      test,
      copied == std::vector<std::uint8_t>(producerState.begin(),
                                          producerState.end()) &&
          consumerChecksum == 2016,
      "consumer did not observe the complete producer live state");
  const auto consumerCompletion = take(
      test,
      apply(dataflow::rootThreadCompletionEventFamily(roots.consumer), 3, 7));
  requireSuccess(test, replay.joinMappedRoots());
  const auto statistics = provider->statistics();
  deployment::test::require(
      test,
      consumerCompletion.outcome ==
              application::ApplicationResourceTimeEventOutcome::
                  NoLegalTransition &&
          replay.joined() && statistics.activationPreparationCount == 1 &&
          statistics.activationReplacementCount == 1 &&
          statistics.preparedConfigurationWordCount == changedWords &&
          statistics.preparedLogicalMemoryCopyCount == 1 &&
          statistics.copiedLogicalMemoryByteCount == 64,
      "provider or application replay lost its exact transition evidence");

  const auto orderedDataflowArtifact =
      buildOrderedChannelDataflow(test, context);
  const auto orderedDataflow = take(test, orderedDataflowArtifact.view());
  const auto orderedRoots = deriveOrderedRoots(test, orderedDataflow);
  const auto orderedParentTargets = rootTargets(
      test, orderedDataflow, orderedRoots, cores[1], cores[0], cores[0]);
  const auto orderedChildTargets = rootTargets(
      test, orderedDataflow, orderedRoots, cores[0], cores[0], cores[1]);
  const auto orderedParentMapping = deployment::test::buildMappedSystemMapping(
      test, orderedDataflowArtifact, hardware.system, {}, artifacts,
      orderedParentTargets);
  const auto orderedChildMapping = deployment::test::buildMappedSystemMapping(
      test, orderedDataflowArtifact, hardware.system, {}, artifacts,
      orderedChildTargets);
  const auto orderedParent = deployment::test::buildMappedSystemDeployment(
      test, orderedDataflowArtifact, hardware.system, orderedParentMapping,
      hardware.implementations, {}, artifacts, blobs, tree);
  const auto orderedChild = deployment::test::buildMappedSystemDeployment(
      test, orderedDataflowArtifact, hardware.system, orderedChildMapping,
      hardware.implementations, {}, artifacts, blobs, tree);
  const ArtifactRootReference orderedDataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, orderedDataflow.identity()};
  expectTransitionRefusal(
      test,
      pnr::finalizeResourceTimeTransition(
          makeTransitionDraft(orderedDataflowReference, orderedRoots,
                              orderedParentMapping, orderedParent,
                              orderedChildMapping, orderedChild, cores[0],
                              cores[1]),
          artifacts, blobs),
      pnr::ResourceTimeTransitionRefusalReason::OrderedChannelState);
}

} // namespace

int main() {
  copiedLiveStateExecutesAtTheApplicationSafePoint();
  return 0;
}
