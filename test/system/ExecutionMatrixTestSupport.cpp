#include "ExecutionMatrixTestSupport.h"
#include "ExecutionMatrixFixtureSupport.h"

#include "ExecutionMatrixGuestPrograms.h"
#include "ExecutionMatrixLifecycle.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Common/MappingDebugLog.h"
#include "Common/TimeoutBudgets.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/Deployment.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "EDA/Adapters/OpenSource/MappedRtlSimulation.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/MappedRtlSimulation.h"
#include "Evaluation/Models/SystemRuntimeParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/LocalConfig.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/ShellProbe.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Mapping/IR/MappingDialect.h"
#include "Runtime/Gem5BridgeABI.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Runtime/Gem5SystemExecution.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialObservationComparison.h"

#include "DeploymentTestSupport.h"
#include "MappedRtlSimulationTestSupport.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::system_test {
namespace {

#if !defined(LOOM_TEST_BUILD_JOBS) || !defined(LOOM_TEST_RTL_BUILD_WORKER_LIMIT)
#error "Loom test build limits must be defined"
#endif
using deployment::test::fail;
using deployment::test::require;

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

bool isSpatialCell(ExecutionMatrixCell cell) {
  return cell == ExecutionMatrixCell::SpatialDfg ||
         cell == ExecutionMatrixCell::SpatialCgra ||
         cell == ExecutionMatrixCell::SpatialRtl ||
         cell == ExecutionMatrixCell::PairedSpatialCgra;
}

bool isPairedCell(ExecutionMatrixCell cell) {
  return cell == ExecutionMatrixCell::PairedSpatialCgra ||
         cell == ExecutionMatrixCell::PairedSystemCgra;
}

bool usesCgraEngine(ExecutionMatrixCell cell) {
  return cell == ExecutionMatrixCell::SpatialCgra ||
         cell == ExecutionMatrixCell::SystemCgra || isPairedCell(cell);
}

constexpr std::uint64_t kHostLoadAddress = 0x80000000;
constexpr std::uint64_t kInstructionLoadAddress = 0x80100000;

deployment::CanonicalTypeBytes memoryInterfaceType(llvm::StringRef test,
                                                   mlir::MLIRContext &context) {
  auto encoded = take(test, dataflow::encodeCanonicalType(mlir::MemRefType::get(
                                {16}, mlir::IntegerType::get(&context, 8))));
  return {encoded.bytes().begin(), encoded.bytes().end()};
}

struct PublishedSystemInputs final {
  sim::CanonicalSimulationWorkload workload;
  sim::CanonicalSimulationRuntimeInput runtimeInput;
  ArtifactRootReference workloadReference;
  ArtifactRootReference runtimeInputReference;
};

PublishedSystemInputs
publishSystemInputs(llvm::StringRef test,
                    const deployment::FinalizedDeployment &deployment,
                    ArtifactStore &artifacts) {
  const deployment::DeploymentExternalInterfaceRef memoryInterface{
      deployment.reference().artifact, 0};
  sim::SystemSimulationWorkload workloadDraft{
      {deployment.reference().artifact, 0}};
  workloadDraft.observableContract.memories = {
      {memoryInterface, sim::MemoryObservationForm::FullState}};
  auto workload = take(test, sim::finalizeSimulationWorkload(
                                 workloadDraft, deployment, artifacts));
  sim::SystemSimulationRuntimeInputDraft runtimeDraft{workload.identity()};
  runtimeDraft.memoryObjects = {
      sim::RuntimeMemoryObject(std::vector<sim::SemanticMemoryByte>(
          16, {sim::SemanticState::Defined, 0}))};
  runtimeDraft.memoryInterfaceBindings = {{memoryInterface, 0, 0}};
  auto runtime = take(test, sim::finalizeSimulationRuntimeInput(
                                runtimeDraft, workload, deployment, artifacts));
  ArtifactRootReference workloadReference =
      take(test, sim::publishSimulationWorkload(workload, artifacts));
  ArtifactRootReference runtimeInputReference =
      take(test, sim::publishSimulationRuntimeInput(runtime, artifacts));
  return {std::move(workload), std::move(runtime), std::move(workloadReference),
          std::move(runtimeInputReference)};
}

std::optional<fabric::SpatialCoreOccurrenceRef>
spatialCoreOf(const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  if (const auto *transport = endpoint.transport()) {
    if (transport->owner.kind() !=
        fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
      return std::nullopt;
    return std::get<fabric::SpatialCoreOccurrenceRef>(transport->owner.payload);
  }
  const auto *memory = endpoint.memory();
  if (!memory ||
      memory->owner.kind() !=
          fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return std::nullopt;
  return std::get<fabric::SpatialCoreOccurrenceRef>(memory->owner.payload);
}

runtime::Gem5SimObjectRef
gem5Object(const runtime::Gem5ModelContractDescriptor &descriptor,
           std::vector<std::uint8_t> payload) {
  return {runtime::gem5ModelContractDescriptorRef(descriptor),
          std::move(payload)};
}

runtime::Gem5SimPortRef gem5Port(runtime::Gem5SimObjectRef object) {
  return {std::move(object), 0, {}};
}

const runtime::Gem5ModelContractDescriptor &
processorModel(llvm::StringRef test,
               const fabric::InstructionCoreMicroarchitecturalRealization
                   *microarchitecture) {
  require(test, microarchitecture != nullptr,
          "System processor has no microarchitectural realization");
  if (microarchitecture->inOrder())
    return runtime::gem5RiscvTimingCpuModel();
  require(test, microarchitecture->outOfOrder() != nullptr,
          "System processor has an unsupported realization");
  return runtime::gem5RiscvO3CpuModel();
}

runtime::FinalizedGem5SimulationBinding buildGem5Binding(
    llvm::StringRef test, const fabric::FinalizedFabricRoot &system,
    const ArtifactRootReference &interconnect,
    runtime::Gem5BuildIdentity buildIdentity, const ArtifactStore &artifacts) {
  requireSuccess(test, runtime::registerBuiltinGem5ModelContracts());
  const auto view = take(test, fabric::requireSystemRoot(system.view()));
  runtime::Gem5SimulationBindingDraft draft{system.reference(),
                                            interconnect,
                                            std::move(buildIdentity),
                                            runtime::gem5BridgeAbiIdentity,
                                            {}};
  std::uint64_t cpuId = 0;
  for (const fabric::HostCoreOccurrenceRef core :
       view.artifact().hostCoreOccurrences()) {
    draft.correspondences.push_back(runtime::Gem5ProcessorCorrespondence{
        runtime::Gem5ProcessorFabricRef(core),
        gem5Object(
            processorModel(test, view.instructionCoreMicroarchitecture(core)),
            runtime::encodeGem5RiscvCpuParameters({cpuId++, 1000}))});
  }
  for (const fabric::AccCoreOccurrenceRef core :
       view.artifact().accCoreOccurrences()) {
    draft.correspondences.push_back(runtime::Gem5ProcessorCorrespondence{
        runtime::Gem5ProcessorFabricRef(
            fabric::InstructionCoreContextRef{core}),
        gem5Object(
            processorModel(test, view.instructionCoreMicroarchitecture(
                                     fabric::InstructionCoreContextRef{core})),
            runtime::encodeGem5RiscvCpuParameters({cpuId++, 1000}))});
  }

  std::map<std::vector<std::uint8_t>, runtime::Gem5SimObjectRef> bridgeObjects;
  for (const fabric::FabricSpatialAttachmentRecordView &attachment :
       view.spatialAttachments()) {
    const auto spatialCore = spatialCoreOf(attachment.spatialEndpoint);
    require(test, spatialCore.has_value(),
            "System attachment has no SpatialCore owner");
    const std::vector<std::uint8_t> coreKey =
        fabric::canonicalFabricBytes(*spatialCore);
    auto [object, inserted] = bridgeObjects.try_emplace(coreKey);
    if (inserted) {
      const std::uint64_t bridgeOrdinal = bridgeObjects.size() - 1;
      object->second = gem5Object(
          runtime::gem5SpatialBridgeModel(),
          runtime::encodeGem5SpatialBridgeParameters(
              {0x10000000 + bridgeOrdinal * 0x10000, 0x1000, 10000, 1 << 20}));
    }
    draft.correspondences.push_back(runtime::Gem5SpatialBridgeCorrespondence{
        *spatialCore, attachment.spatialEndpoint, gem5Port(object->second)});
  }

  const auto memoryObject =
      gem5Object(runtime::gem5SimpleMemoryModel(),
                 runtime::encodeGem5SimpleMemoryParameters(
                     {kHostLoadAddress, 0x20000000, 20000}));
  for (const fabric::SystemMemoryServiceRef service :
       view.artifact().systemMemoryServices())
    draft.correspondences.push_back(runtime::Gem5MemoryOrServiceCorrespondence{
        runtime::Gem5MemoryOrServiceFabricRef(service), memoryObject,
        gem5Port(memoryObject)});
  for (const fabric::SystemServiceEndpointRef endpoint :
       view.artifact().systemServiceEndpoints())
    draft.correspondences.push_back(runtime::Gem5MemoryOrServiceCorrespondence{
        runtime::Gem5MemoryOrServiceFabricRef(endpoint), memoryObject,
        gem5Port(memoryObject)});

  const auto transportObject = gem5Object(runtime::gem5SystemXBarModel(), {});
  for (const fabric::SystemTransportResourceRef resource :
       view.transportResources())
    draft.correspondences.push_back(runtime::Gem5TransportCorrespondence{
        runtime::Gem5TransportFabricRef(resource), transportObject,
        gem5Port(transportObject)});
  for (const fabric::FabricTransportEndpointRef &endpoint :
       view.artifact().transportEndpoints()) {
    const auto kind = endpoint.owner.kind();
    if (kind !=
            fabric::FabricTransportEndpointOwnerKind::SystemServiceEndpoint &&
        kind !=
            fabric::FabricTransportEndpointOwnerKind::SystemTransportResource)
      continue;
    draft.correspondences.push_back(runtime::Gem5TransportCorrespondence{
        runtime::Gem5TransportFabricRef(endpoint), transportObject,
        gem5Port(transportObject)});
  }
  const auto externalObject =
      gem5Object(runtime::gem5ExternalEndpointModel(), {});
  for (const fabric::ExternalBoundaryRef boundary :
       view.artifact().externalBoundaries())
    draft.correspondences.push_back(runtime::Gem5ExternalEndpointCorrespondence{
        boundary, externalObject, gem5Port(externalObject)});
  return take(test, runtime::finalizeGem5SimulationBinding(std::move(draft),
                                                           artifacts));
}

runtime::Gem5BuildIdentity structuralGem5Identity() {
  return {"https://gem5.googlesource.com/public/gem5",
          "0123456789abcdef0123456789abcdef01234567",
          "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
          "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"};
}

struct Gem5Readiness final {
  runtime::Gem5BuildIdentity buildIdentity;
  std::string binary;
  std::string path;
};

Gem5Readiness readGem5Readiness(llvm::StringRef test,
                                llvm::StringRef readinessPath) {
  require(test, !readinessPath.empty(), "gem5 readiness path is required");
  std::error_code error;
  const std::filesystem::path path =
      std::filesystem::weakly_canonical(readinessPath.str(), error);
  require(test, !error && path.is_absolute(),
          "gem5 readiness path is not canonical");
  auto buffer = llvm::MemoryBuffer::getFile(path.string(), false, false);
  if (!buffer)
    fail(test, buffer.getError().message());
  auto value = llvm::json::parse((*buffer)->getBuffer());
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  const llvm::json::Object *object = value->getAsObject();
  require(test, object != nullptr, "gem5 readiness is not an object");
  const auto repository = object->getString("gem5_repository_identity");
  const auto commit = object->getString("gem5_full_commit_identity");
  const auto configuration = object->getString("build_configuration_digest");
  const auto binaryFingerprint = object->getString("binary_sha256");
  const auto binary = object->getString("binary");
  require(test,
          repository && commit && configuration && binaryFingerprint && binary,
          "gem5 readiness omits an identity field");
  return {{repository->str(), commit->str(), configuration->str(),
           binaryFingerprint->str()},
          binary->str(),
          path.string()};
}

struct SharedFixture final {
  std::unique_ptr<mlir::MLIRContext> context;
  dataflow::CanonicalDataflowArtifact dataflow;
  ArtifactRootReference dataflowReference;
  eda::test::MappedSpatialHardwareFixture hardware;
  ArtifactRootReference interconnect;
  hardware::FinalizedHardwareImplementation implementation;
  mapping::FinalizedSystemMapping systemMapping;
  deployment::FinalizedDeployment deployment;
  ArtifactRootReference spatialWorkload;
  ArtifactRootReference spatialRuntimeInput;
  PublishedSystemInputs systemInputs;
};

struct PairedFingerprints final {
  std::string work;
  std::string config;
};

void appendFramedBytes(std::vector<std::uint8_t> &destination,
                       llvm::ArrayRef<std::uint8_t> value) {
  const std::uint64_t size = value.size();
  for (int shift = 56; shift >= 0; shift -= 8)
    destination.push_back(static_cast<std::uint8_t>(size >> shift));
  destination.insert(destination.end(), value.begin(), value.end());
}

void appendFramedText(std::vector<std::uint8_t> &destination,
                      llvm::StringRef value) {
  appendFramedBytes(
      destination,
      {reinterpret_cast<const std::uint8_t *>(value.data()), value.size()});
}

std::string fingerprint(llvm::StringRef test, llvm::StringRef descriptor,
                        llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  auto digest = computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
       descriptor.size()},
      canonicalBytes);
  return formatComponentViewDigestHex(take(test, std::move(digest)));
}

PairedFingerprints pairedFingerprints(llvm::StringRef test,
                                      const SharedFixture &fixture) {
  std::vector<std::uint8_t> workBytes;
  for (const ArtifactRootReference &reference :
       {fixture.dataflowReference, fixture.hardware.module.reference(),
        fixture.hardware.techMapping,
        fixture.hardware.spatialMapping.reference(), fixture.spatialWorkload,
        fixture.spatialRuntimeInput})
    appendFramedBytes(workBytes, encodeArtifactRootReference(reference));
  appendFramedText(workBytes, "cgra");
  appendFramedText(workBytes, "none");
  const auto appendU64 = [&](std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (unsigned index = 0; index != bytes.size(); ++index)
      bytes[index] = static_cast<std::uint8_t>(value >> (56 - index * 8));
    appendFramedBytes(workBytes, bytes);
  };
  appendU64(runtime::gem5MaximumSpatialWork);
  appendU64(pairedMeasurementInvocationCount);

  const CanonicalSemanticBytes configBytes =
      canonicalResolvedConfigBytes(defaultResolvedConfig());
  return {fingerprint(test, "loom.simulation.paired_cgra_work.v3", workBytes),
          fingerprint(test, "loom.simulation.paired_cgra_config.v1",
                      configBytes.bytes())};
}

void traceSpatialFixture(llvm::StringRef test, const SharedFixture &fixture,
                         const ArtifactStore &artifacts) {
  if (!mapping_debug::enabled(mapping_debug::Level::Detail))
    return;

  const auto dataflow = take(test, fixture.dataflow.view());
  const auto tech =
      take(test,
           mapping::importTechMapping(fixture.hardware.techMapping, artifacts));
  const auto &spatial = fixture.hardware.spatialMapping.view();
  for (const dataflow::CanonicalActorView &actor : dataflow.actors())
    llvm::errs() << "[loom][system-fixture][actor] actor="
                 << actor.ref.entity.value()
                 << " graph=" << actor.graph.entity.value()
                 << " op=" << actor.op->getName().getStringRef() << '\n';
  for (const mapping::TechComputeRealizationView &realization :
       tech.view().computeRealizations()) {
    llvm::errs() << "[loom][system-fixture][realization] id="
                 << realization.entityId << " actors=";
    for (const auto [ordinal, actor] : llvm::enumerate(realization.actors)) {
      if (ordinal)
        llvm::errs() << ',';
      llvm::errs() << actor.actor.entity.value();
    }
    llvm::errs() << '\n';
  }
  for (const mapping::SpatialComputeBindingView &binding :
       spatial.computeBindings())
    llvm::errs() << "[loom][system-fixture][binding] realization="
                 << binding.realization
                 << " occurrence=" << fabric::printFabricRef(binding.occurrence)
                 << '\n';

  const auto printProducer =
      [](llvm::raw_ostream &os,
         const dataflow::CanonicalGraphProducerEndpointRef &producer) {
        std::visit(
            [&](const auto &endpoint) {
              using Endpoint = std::decay_t<decltype(endpoint)>;
              if constexpr (std::is_same_v<Endpoint,
                                           dataflow::ActorTokenResultRef>)
                os << "actor_result:" << endpoint.actor.entity.value() << ':'
                   << endpoint.ordinal;
              else
                os << "graph_ingress:"
                   << std::visit(
                          [](const auto &ingress) {
                            return ingress.graph.entity.value();
                          },
                          endpoint);
            },
            producer);
      };
  const auto printConsumer =
      [](llvm::raw_ostream &os,
         const dataflow::CanonicalGraphConsumerEndpointRef &consumer) {
        std::visit(
            [&](const auto &endpoint) {
              using Endpoint = std::decay_t<decltype(endpoint)>;
              if constexpr (std::is_same_v<Endpoint,
                                           dataflow::ActorTokenOperandRef>)
                os << "actor_operand:" << endpoint.actor.entity.value() << ':'
                   << endpoint.ordinal;
              else
                os << "graph_egress:"
                   << std::visit(
                          [](const auto &egress) {
                            return egress.graph.entity.value();
                          },
                          endpoint);
            },
            consumer);
      };
  for (const auto [ordinal, route] : llvm::enumerate(spatial.routeTrees())) {
    llvm::errs() << "[loom][system-fixture][route] ordinal=" << ordinal
                 << " producer=";
    printProducer(llvm::errs(), route.logicalNet);
    llvm::errs() << " root=" << fabric::printFabricRef(route.rootEndpoint)
                 << " sinks=";
    for (const auto [sinkOrdinal, sink] : llvm::enumerate(route.sinks)) {
      if (sinkOrdinal)
        llvm::errs() << ',';
      printConsumer(llvm::errs(), sink.sink);
    }
    llvm::errs() << '\n';
    for (const mapping::SpatialRouteNodeView &node : route.nodes) {
      llvm::errs() << "[loom][system-fixture][route-node] route=" << ordinal
                   << " node=" << node.ordinal << " parent=";
      if (node.parentOrdinal)
        llvm::errs() << *node.parentOrdinal;
      else
        llvm::errs() << "root";
      llvm::errs() << " endpoint=" << fabric::printFabricRef(node.endpoint)
                   << '\n';
    }
    for (const auto [sinkOrdinal, sink] : llvm::enumerate(route.sinks))
      llvm::errs() << "[loom][system-fixture][route-sink] route=" << ordinal
                   << " sink=" << sinkOrdinal << " node=" << sink.nodeOrdinal
                   << '\n';
  }
}
SharedFixture
buildSharedFixture(llvm::StringRef test, ExecutionMatrixCell cell,
                   ArtifactStore &artifacts, BlobStore &blobs,
                   const deployment::test::TemporaryTree &tree,
                   ExecutionMatrixLifecycleRecorder *lifecycle = nullptr) {
  std::optional<ExecutionMatrixLifecycleTimer> setupChildTimer;
  if (lifecycle)
    setupChildTimer.emplace(
        *lifecycle,
        ExecutionMatrixLifecycleOperation::DataflowConstructionAndPublication);
  auto context = makeContext();
  const bool paired = isPairedCell(cell);
  auto dataflow = buildCanonicalApplication(test, *context, paired);
  const ArtifactRootReference dataflowReference =
      take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  const auto observeHardwareConstruction = [&](const auto operation,
                                               const auto boundary) {
    using Boundary = eda::test::MappedSpatialHardwareFixtureBoundary;
    using HardwareOperation = eda::test::MappedSpatialHardwareFixtureOperation;
    if (!lifecycle)
      return;
    if (operation == HardwareOperation::DataflowPublication) {
      require(test, setupChildTimer.has_value(),
              "Dataflow publication escaped its setup interval");
      if (boundary == Boundary::End)
        setupChildTimer.reset();
      return;
    }
    setupChildTimer.reset();
    if (boundary == Boundary::Begin)
      setupChildTimer.emplace(*lifecycle, operation);
  };
  auto hardware = eda::test::buildMappedSpatialHardwareFixture(
      test, dataflow, *context, artifacts, blobs,
      deployment::test::MappedSpatialSystemSpec{paired ? 1U : 4U, true, true},
      eda::test::MappedRtlFixtureTopology::HeterogeneousPortable,
      eda::test::MappedRtlRouteCoverage::AnyLegal,
      eda::test::MappedSystemInterconnect::Gem5EventTransport,
      observeHardwareConstruction);
  require(test, hardware.interconnect.has_value(),
          "System fixture omitted its typed interconnect implementation");
  const ArtifactRootReference interconnect = *hardware.interconnect;
  require(test, !hardware.implementations.empty(),
          "System fixture omitted SpatialCore implementations");
  const auto systemView =
      take(test, fabric::requireSystemRoot(hardware.system.view()));
  const auto cores = systemView.artifact().accCoreOccurrences();
  const auto dataflowView = take(test, dataflow.view());
  require(test, cores.size() == dataflowView.rootThreadLaunches().size(),
          "anchor core count does not match its root launch count");
  if (lifecycle)
    setupChildTimer.emplace(
        *lifecycle, ExecutionMatrixLifecycleOperation::SystemMappingAndPnr);
  auto systemMapping = deployment::test::buildMappedSystemMapping(
      test, dataflow, hardware.system, {hardware.spatialMapping.reference()},
      artifacts, cores);
  setupChildTimer.reset();
  deployment::test::MappedSystemExecutablePrograms programs;
  if (lifecycle)
    setupChildTimer.emplace(
        *lifecycle, ExecutionMatrixLifecycleOperation::GuestCompileAndLink);
  const bool orderedSequence = cell == ExecutionMatrixCell::SystemDfg ||
                               cell == ExecutionMatrixCell::SystemCgra;
  const std::string hostSource =
      paired            ? pairedInvocationHostProgramSource()
      : orderedSequence ? orderedChannelHostProgramSource().str()
                        : singleInvocationHostProgramSource().str();
  programs.hostProgramBytes =
      compileGuestProgram(test, tree, "system-host", hostSource,
                          kHostLoadAddress, "loom_host_entry", true);
  programs.instructionProgramBytes = compileGuestProgram(
      test, tree, "spatial-dispatch", spatialInstructionProgramSource(),
      kInstructionLoadAddress, "__loom_thread_entry_0", false);
  setupChildTimer.reset();
  if (lifecycle)
    setupChildTimer.emplace(*lifecycle,
                            ExecutionMatrixLifecycleOperation::
                                RuntimeBindingAndDeploymentFinalization);
  programs.hostEntries = {{0, "loom_host_entry", {}, {}, {0}}};
  programs.hostInterfaces = {{0, deployment::HostExternalInterfaceKind::Memory,
                              deployment::HostExternalInterfaceDirection::InOut,
                              memoryInterfaceType(test, *context)}};
  auto deployment = deployment::test::buildMappedSystemDeployment(
      test, dataflow, hardware.system, systemMapping, hardware.implementations,
      std::move(programs), artifacts, blobs, tree);
  setupChildTimer.reset();
  if (lifecycle)
    setupChildTimer.emplace(
        *lifecycle,
        ExecutionMatrixLifecycleOperation::WorkloadAndRuntimeInputPublication);
  const auto [spatialWorkload, spatialRuntimeInput] =
      publishSpatialInputs(test, dataflow, artifacts, paired);
  auto spatialInputs =
      take(test, sim::importSpatialSimulationInputs(
                     spatialWorkload, spatialRuntimeInput, artifacts));
  const sim::SpatialSimulationWorkload *spatial =
      spatialInputs.workload.spatial();
  require(test, spatial != nullptr,
          "System fixture Spatial workload has no launch selection");
  auto launchSelection =
      take(test, deployment::resolveDeploymentSpatialLaunchSelection(
                     deployment, spatial->launchRef, spatial->denseCoordinates,
                     artifacts, blobs));
  const auto selectedImplementation =
      llvm::find_if(hardware.implementations, [&](const auto &candidate) {
        return candidate.reference() == launchSelection.hardwareImplementation;
      });
  require(test, selectedImplementation != hardware.implementations.end(),
          "Deployment selected an implementation outside the fixture");
  auto implementation = *selectedImplementation;
  auto systemInputs = publishSystemInputs(test, deployment, artifacts);
  setupChildTimer.reset();
  return {std::move(context),
          std::move(dataflow),
          dataflowReference,
          std::move(hardware),
          interconnect,
          std::move(implementation),
          std::move(systemMapping),
          std::move(deployment),
          spatialWorkload,
          spatialRuntimeInput,
          std::move(systemInputs)};
}
evaluation::CaseArtifactResolution
buildResolution(llvm::StringRef test, const SharedFixture &fixture,
                const std::optional<ArtifactRootReference> &gem5Binding,
                const ArtifactRootReference &workload,
                const ArtifactRootReference &runtimeInput,
                bool systemWorkload) {
  std::vector<evaluation::CaseArtifactResolution::Entry> entries{
      {fixture.dataflowReference, {}},
      {fixture.hardware.module.reference(), {}},
      {fixture.hardware.system.reference(),
       {fixture.hardware.module.reference()}},
      {fixture.hardware.techMapping,
       {fixture.dataflowReference, fixture.hardware.module.reference()}},
      {fixture.hardware.spatialMapping.reference(),
       {fixture.dataflowReference, fixture.hardware.module.reference(),
        fixture.hardware.techMapping}},
      {fixture.systemMapping.reference(),
       {fixture.dataflowReference, fixture.hardware.system.reference(),
        fixture.hardware.spatialMapping.reference()}},
      {fixture.interconnect, {fixture.hardware.system.reference()}},
  };
  std::vector<ArtifactRootReference> deploymentParents{
      fixture.dataflowReference, fixture.systemMapping.reference(),
      fixture.hardware.spatialMapping.reference()};
  for (const hardware::FinalizedHardwareImplementation &implementation :
       fixture.hardware.implementations) {
    entries.push_back(
        {implementation.reference(), {fixture.hardware.system.reference()}});
    deploymentParents.push_back(implementation.reference());
  }
  entries.push_back(
      {fixture.deployment.reference(), std::move(deploymentParents)});
  if (gem5Binding)
    entries.push_back(
        {*gem5Binding,
         {fixture.hardware.system.reference(), fixture.interconnect}});
  entries.push_back({workload,
                     {systemWorkload ? fixture.deployment.reference()
                                     : fixture.dataflowReference}});
  entries.push_back({runtimeInput,
                     {systemWorkload ? fixture.deployment.reference()
                                     : fixture.dataflowReference,
                      workload}});
  return take(test, evaluation::CaseArtifactResolution::get(entries));
}

evaluation::EvaluationRequest
buildMappedRtlRequest(llvm::StringRef test, const SharedFixture &fixture,
                      llvm::StringRef verilatorIdentity,
                      const evaluation::CaseArtifactResolution &resolution,
                      const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto subjects = take(
      test,
      evaluation::EvaluationSubjectBindings::get(
          {{evaluation::models::mappedRtlHardwareImplementationSubjectRole(),
            {fixture.implementation.reference()}},
           {evaluation::models::mappedRtlDeploymentSubjectRole(),
            {fixture.deployment.reference()}}}));
  auto evaluationCase = take(
      test, evaluation::EvaluationCase::get(
                evaluation::mappedRtlSimulationCaseSignatureRef(),
                std::move(subjects), fixture.spatialWorkload,
                fixture.spatialRuntimeInput, {}, resolution, artifacts, blobs));
  auto cycleCount = take(
      test, evaluation::MetricRequest::get(
                {evaluation::MetricKind::CycleCount,
                 evaluation::EvaluationScope{evaluation::ScopeFormRef(0), {}}},
                {}, evaluationCase, resolution, artifacts));
  ResolvedConfig config = defaultResolvedConfig();
  config.evaluation.mappedRtlSimulator =
      evaluation::models::MappedRtlSimulatorBinding{verilatorIdentity.str()};
  auto model =
      take(test, evaluation::ResolvedModelBinding::project(
                     evaluation::models::mappedRtlSimulatorModelDescriptorRef(),
                     {}, config));
  auto request =
      take(test, evaluation::EvaluationRequest::get(
                     evaluationCase, {cycleCount}, {}, std::move(model), 0,
                     resolution, artifacts, blobs));
  (void)take(test, evaluation::publishEvaluationRequest(request, artifacts));
  return request;
}

evaluation::EvaluationRequest
buildSystemRequest(llvm::StringRef test, ExecutionMatrixCell cell,
                   const SharedFixture &fixture,
                   const runtime::FinalizedGem5SimulationBinding &gem5Binding,
                   llvm::StringRef verilatorIdentity,
                   const evaluation::CaseArtifactResolution &resolution,
                   const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto subjects = take(test, evaluation::EvaluationSubjectBindings::get(
                                 {{evaluation::CaseSubjectRoleRef(0),
                                   {fixture.deployment.reference()}},
                                  {evaluation::CaseSubjectRoleRef(1),
                                   {gem5Binding.reference()}}}));
  auto evaluationCase = take(
      test, evaluation::EvaluationCase::get(
                evaluation::systemSimulationCaseSignatureRef(),
                std::move(subjects), fixture.systemInputs.workloadReference,
                fixture.systemInputs.runtimeInputReference, {}, resolution,
                artifacts, blobs));
  const evaluation::BuiltinEvaluationModel modelKind =
      cell == ExecutionMatrixCell::SystemDfg
          ? evaluation::BuiltinEvaluationModel::Gem5SystemDfg
      : usesCgraEngine(cell)
          ? evaluation::BuiltinEvaluationModel::Gem5SystemCgra
          : evaluation::BuiltinEvaluationModel::Gem5SystemRtl;
  std::vector<evaluation::MetricRequest> metrics;
  if (cell != ExecutionMatrixCell::SystemDfg)
    metrics.push_back(
        take(test,
             evaluation::MetricRequest::get(
                 {evaluation::MetricKind::Runtime,
                  evaluation::EvaluationScope{evaluation::ScopeFormRef(0), {}}},
                 {}, evaluationCase, resolution, artifacts)));
  ResolvedConfig config = defaultResolvedConfig();
  if (cell == ExecutionMatrixCell::SystemRtl)
    config.evaluation.mappedRtlSimulator =
        evaluation::models::MappedRtlSimulatorBinding{verilatorIdentity.str()};
  auto descriptor =
      take(test, evaluation::builtinEvaluationModelDescriptorRef(modelKind));
  auto model = take(
      test, evaluation::ResolvedModelBinding::project(descriptor, {}, config));
  auto request =
      take(test, evaluation::EvaluationRequest::get(
                     evaluationCase, std::move(metrics), {}, std::move(model),
                     0, resolution, artifacts, blobs));
  (void)take(test, evaluation::publishEvaluationRequest(request, artifacts));
  return request;
}

struct ToolBinding final {
  external_tool::LocalToolConfig local;
  external_tool::ResolvedToolBinding resolved;
};

ToolBinding
resolveHostTool(llvm::StringRef test,
                const external_tool::ExternalToolProviderDescriptor &provider,
                const deployment::test::TemporaryTree &tree) {
  external_tool::LocalToolConfig local;
  local.runtimePolicy = external_tool::RuntimePolicy::Host;
  const std::filesystem::path probePath =
      tree.path(provider.binding.key + "-probe");
  std::filesystem::create_directories(probePath);
  external_tool::ShellToolBindingProbe probe(probePath.string(),
                                             provider.versionProbe);
  auto resolved =
      take(test,
           external_tool::resolveToolBinding(
               provider.binding, local,
               external_tool::captureToolEnvironment(provider.binding), probe));
  local.tools[provider.binding.key].binding.executable = resolved.executable;
  return {std::move(local), std::move(resolved)};
}

struct CompletedRun final {
  struct DiagnosticSummary final {
    std::vector<runtime::Gem5SpatialInvocationProjection> spatialInvocations;
    runtime::Gem5SystemAttemptProfile attemptProfile;
    std::uint64_t gem5Ticks = 0;
  };

  evaluation::EvaluationEvidence evidence;
  ArtifactRootReference evidenceReference;
  sim::CanonicalSimulationExecution execution;
  std::optional<DiagnosticSummary> gem5Diagnostics;
  std::vector<external_tool::ExternalToolCommandExecutionObservation>
      externalCommands;
};

CompletedRun importCompleted(
    llvm::StringRef test, evaluation::EvaluationEvidence evidence,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    ExecutionMatrixLifecycleRecorder *lifecycle = nullptr,
    ExecutionMatrixLifecycleOperation operation =
        ExecutionMatrixLifecycleOperation::OrdinaryExecutionImport,
    std::vector<external_tool::ExternalToolCommandExecutionObservation>
        externalCommands = {}) {
  std::optional<ExecutionMatrixLifecycleTimer> timer;
  if (lifecycle)
    timer.emplace(*lifecycle, operation);
  if (evidence.outcomeKind() != evaluation::EvidenceOutcomeKind::Completed) {
    std::string diagnostic = evaluation::toString(evidence.outcomeKind()).str();
    std::visit(
        [&](const auto &outcome) {
          using Outcome = std::decay_t<decltype(outcome)>;
          if constexpr (!std::is_same_v<Outcome, evaluation::CompletedEvidence>)
            diagnostic += ": " + evaluation::toString(outcome.reason).str();
        },
        evidence.outcome());
    fail(test, "execution published " + diagnostic);
  }
  require(test,
          evidence.outputBindings().size() == 1 &&
              evidence.outputBindings().front().artifacts.size() == 1,
          "execution did not publish one SimulationExecution");
  auto execution =
      take(test, sim::importSimulationExecution(
                     evidence.outputBindings().front().artifacts.front(),
                     resolution, artifacts, blobs));
  require(test, execution.request() == evidence.requestRef(),
          "SimulationExecution is not coupled to its exact Evidence Request");
  require(test,
          std::holds_alternative<sim::RetiredExecution>(execution.terminal()),
          "execution did not retire normally");
  ArtifactRootReference evidenceReference =
      evaluation::evaluationEvidenceReference(evidence);
  return {std::move(evidence), std::move(evidenceReference),
          std::move(execution), std::nullopt, std::move(externalCommands)};
}

void requireSpatialOracle(llvm::StringRef test,
                          const sim::CanonicalSimulationExecution &execution,
                          bool paired = false) {
  const auto *spatial = execution.spatial();
  require(test, spatial != nullptr,
          "Spatial matrix cell published a System execution");
  if (paired) {
    require(test,
            spatial->functionalObservations.valueResults.empty() &&
                spatial->functionalObservations.streamOutputs.empty() &&
                spatial->functionalObservations.memories.empty(),
            "paired Spatial execution observed an unselected boundary");
    require(test,
            spatial->progressObservations.graphRetirementVisible.has_value(),
            "paired Spatial execution has no graph-retirement observation");
    return;
  }
  require(test, spatial->functionalObservations.streamOutputs.size() == 1,
          "project stream observation is not total");
  const sim::CanonicalStreamSequence &stream =
      spatial->functionalObservations.streamOutputs.front();
  require(test,
          stream.values.tokenCount == 1 && stream.values.lanes.size() == 1 &&
              stream.values.lanes.front().state ==
                  sim::SemanticState::Defined &&
              stream.values.lanes.front().bits.getZExtValue() == 7 &&
              stream.termination == sim::StreamTermination::ClosedAfterLast,
          "Spatial execution differs from the independent stream oracle");
  require(test,
          spatial->progressObservations.graphRetirementVisible.has_value(),
          "Spatial execution has no graph-retirement observation");
}

void requireSystemOracle(llvm::StringRef test,
                         const sim::CanonicalSimulationExecution &execution) {
  const auto *system = execution.system();
  require(test, system != nullptr,
          "System matrix cell published a Spatial execution");
  require(test, system->functionalObservations.memories.size() == 1,
          "System memory observation is not total");
  const auto *memory = std::get_if<sim::FullMemoryObservation>(
      &system->functionalObservations.memories.front());
  require(test,
          memory && memory->bytes.size() == 16 &&
              memory->bytes.front().state == sim::SemanticState::Defined &&
              memory->bytes.front().value == 7,
          "System execution differs from the independent memory oracle");
  require(test,
          system->progressObservations.programExitVisible.has_value() &&
              system->progressObservations.programExitVisible->gem5Tick >=
                  system->progressObservations.programEntryAccepted.gem5Tick,
          "System execution has no valid gem5 progress interval");
}

void requireFreshAttempt(
    llvm::StringRef test,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation
        &execution) {
  require(
      test,
      execution.manifestDigest == prepared.manifestDigest &&
          execution.reusePolicy ==
              external_tool::ExternalToolResultReusePolicy::RequireFresh &&
          execution.cacheAvailability ==
              external_tool::ExternalToolResultCacheAvailability::Disabled &&
          execution.cacheLookup ==
              external_tool::ExternalToolResultCacheLookup::NotAttempted &&
          execution.cacheDiscard ==
              external_tool::ExternalToolResultCacheDiscard::NotAttempted &&
          execution.cachePublication ==
              external_tool::ExternalToolResultCachePublication::NotAttempted &&
          !execution.waitedForCacheKeyLock && execution.invokedExternalTool,
      "conformance execution did not use the exact fresh external attempt");
}

void requireSuccessfulAttempt(
    llvm::StringRef test,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation
        &execution) {
  if (execution.exitCode == 0)
    return;
  const std::filesystem::path stderrPath =
      std::filesystem::path(prepared.bundleRoot) / "outputs/stderr.log";
  auto buffer = llvm::MemoryBuffer::getFile(stderrPath.string(), false, false);
  const std::string diagnostic =
      buffer ? (*buffer)->getBuffer().str() : std::string();
  fail(test, "external matrix cell exited with " +
                 std::to_string(execution.exitCode) + ": " + diagnostic);
}

CompletedRun
runExternal(llvm::StringRef test, const evaluation::EvaluationRequest &request,
            const evaluation::CaseArtifactResolution &resolution,
            external_tool::LocalToolConfig local, llvm::StringRef bundlePath,
            const ArtifactStore &artifacts, const BlobStore &blobs,
            ExecutionMatrixLifecycleRecorder *lifecycle = nullptr) {
  auto preparation = [&] {
    std::optional<ExecutionMatrixLifecycleTimer> timer;
    if (lifecycle)
      timer.emplace(*lifecycle,
                    ExecutionMatrixLifecycleOperation::OrdinaryPrepare);
    return take(test, evaluation::prepareEvaluationModelInvocation(
                          request, resolution, artifacts, blobs,
                          external_tool::ExternalToolPreparationContext{
                              std::move(local), bundlePath.str()}));
  }();
  auto *prepared =
      std::get_if<evaluation::EvaluationModelPreparedInvocation>(&preparation);
  require(test, prepared != nullptr,
          "available external provider returned terminal Evidence at prepare");
  const external_tool::PreparedExternalToolInvocation &external =
      prepared->externalInvocation();
  const external_tool::ExternalToolInvocationExecutionObservation execution =
      [&] {
        std::optional<ExecutionMatrixLifecycleTimer> timer;
        if (lifecycle)
          timer.emplace(
              *lifecycle,
              ExecutionMatrixLifecycleOperation::OrdinaryExternalExecution);
        return take(
            test,
            external_tool::executeExternalToolInvocationBundleObserved(
                external, {},
                external_tool::ExternalToolResultReusePolicy::RequireFresh));
      }();
  requireFreshAttempt(test, external, execution);
  requireSuccessfulAttempt(test, external, execution);
  auto evidence = [&] {
    std::optional<ExecutionMatrixLifecycleTimer> timer;
    if (lifecycle)
      timer.emplace(
          *lifecycle,
          ExecutionMatrixLifecycleOperation::OrdinaryImportAndEvidenceAssembly);
    return take(test, evaluation::importEvaluationModelInvocation(
                          request, resolution, *prepared, execution, artifacts,
                          blobs));
  }();
  {
    std::optional<ExecutionMatrixLifecycleTimer> timer;
    if (lifecycle)
      timer.emplace(
          *lifecycle,
          ExecutionMatrixLifecycleOperation::OrdinaryEvidencePublication);
    (void)take(test,
               evaluation::publishEvaluationEvidence(evidence, artifacts));
  }
  return importCompleted(
      test, std::move(evidence), resolution, artifacts, blobs, lifecycle,
      ExecutionMatrixLifecycleOperation::OrdinaryExecutionImport,
      execution.commandExecutions);
}

CompletedRun runGem5Diagnostic(
    llvm::StringRef test, const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    external_tool::LocalToolConfig local, llvm::StringRef bundlePath,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    ExecutionMatrixLifecycleRecorder &lifecycle) {
  auto preparation = [&] {
    ExecutionMatrixLifecycleTimer timer(
        lifecycle, ExecutionMatrixLifecycleOperation::DiagnosticPrepare);
    return take(test, runtime::prepareGem5SystemDiagnosticInvocation(
                          request, resolution, artifacts, blobs,
                          external_tool::ExternalToolPreparationContext{
                              std::move(local), bundlePath.str()}));
  }();
  auto *prepared =
      std::get_if<external_tool::PreparedExternalToolInvocation>(&preparation);
  require(test, prepared != nullptr,
          "diagnostic gem5 provider rejected an available request");
  const external_tool::ExternalToolInvocationExecutionObservation execution =
      [&] {
        ExecutionMatrixLifecycleTimer timer(
            lifecycle,
            ExecutionMatrixLifecycleOperation::DiagnosticExternalExecution);
        return take(
            test,
            external_tool::executeExternalToolInvocationBundleObserved(
                *prepared, {},
                external_tool::ExternalToolResultReusePolicy::RequireFresh));
      }();
  requireFreshAttempt(test, *prepared, execution);
  requireSuccessfulAttempt(test, *prepared, execution);
  runtime::Gem5SystemDiagnosticEvaluation diagnostics = [&] {
    ExecutionMatrixLifecycleTimer timer(
        lifecycle,
        ExecutionMatrixLifecycleOperation::DiagnosticImportAndEvidenceAssembly);
    return take(
        test, runtime::importGem5SystemDiagnosticInvocation(
                  request, resolution, *prepared, execution, artifacts, blobs));
  }();
  {
    ExecutionMatrixLifecycleTimer timer(
        lifecycle,
        ExecutionMatrixLifecycleOperation::DiagnosticEvidencePublication);
    (void)take(test, evaluation::publishEvaluationEvidence(diagnostics.evidence,
                                                           artifacts));
  }
  std::vector<runtime::Gem5SpatialInvocationProjection> spatialInvocations =
      std::move(diagnostics.spatialInvocations);
  runtime::Gem5SystemAttemptProfile attemptProfile =
      std::move(diagnostics.attemptProfile);
  CompletedRun completed = importCompleted(
      test, std::move(diagnostics.evidence), resolution, artifacts, blobs,
      &lifecycle, ExecutionMatrixLifecycleOperation::DiagnosticExecutionImport,
      execution.commandExecutions);
  const sim::SystemSimulationExecution *system = completed.execution.system();
  require(test, system != nullptr,
          "diagnostic attempt did not publish a System execution");
  require(test,
          system->progressObservations.terminalObserved.gem5Tick >=
              system->progressObservations.programEntryAccepted.gem5Tick,
          "diagnostic gem5 progress interval is reversed");
  completed.gem5Diagnostics = CompletedRun::DiagnosticSummary{
      std::move(spatialInvocations), std::move(attemptProfile),
      system->progressObservations.terminalObserved.gem5Tick -
          system->progressObservations.programEntryAccepted.gem5Tick};
  return completed;
}

CompletedRun runSpatialCell(llvm::StringRef test, ExecutionMatrixCell cell,
                            const SharedFixture &fixture,
                            ArtifactStore &artifacts, BlobStore &blobs,
                            const deployment::test::TemporaryTree &tree) {
  if (cell == ExecutionMatrixCell::SpatialDfg) {
    auto prepared =
        take(test, evaluation::models::prepareDfgSimulationEvaluation(
                       fixture.dataflowReference, fixture.spatialWorkload,
                       fixture.spatialRuntimeInput, defaultResolvedConfig(),
                       artifacts, blobs));
    auto evidence =
        take(test, evaluation::models::evaluateDfgSimulation(
                       prepared, {100000, std::nullopt}, artifacts, blobs));
    (void)take(test,
               evaluation::publishEvaluationEvidence(evidence, artifacts));
    return importCompleted(test, std::move(evidence), prepared.resolution,
                           artifacts, blobs);
  }
  if (usesCgraEngine(cell)) {
    auto prepared =
        take(test,
             evaluation::models::prepareCgraSimulationEvaluation(
                 fixture.dataflowReference, fixture.hardware.module.reference(),
                 fixture.hardware.spatialMapping.reference(),
                 fixture.spatialWorkload, fixture.spatialRuntimeInput,
                 defaultResolvedConfig(), artifacts, blobs));
    auto evidence =
        take(test, evaluation::models::evaluateCgraSimulation(
                       prepared, {100000, std::nullopt}, artifacts, blobs));
    (void)take(test,
               evaluation::publishEvaluationEvidence(evidence, artifacts));
    return importCompleted(test, std::move(evidence), prepared.resolution,
                           artifacts, blobs);
  }

  ToolBinding verilator =
      resolveHostTool(test, external_tool::verilatorProvider(), tree);
  auto resolution =
      buildResolution(test, fixture, std::nullopt, fixture.spatialWorkload,
                      fixture.spatialRuntimeInput, false);
  auto request = buildMappedRtlRequest(
      test, fixture, verilator.resolved.version, resolution, artifacts, blobs);
  verilator.local.tools[external_tool::verilatorProvider().binding.key]
      .providerOptions["max_cycles"] = 128;
  verilator.local.tools[external_tool::verilatorProvider().binding.key]
      .providerOptions["build_jobs"] = LOOM_TEST_BUILD_JOBS;
  return runExternal(test, request, resolution, std::move(verilator.local),
                     tree.path("spatial-rtl-bundle"), artifacts, blobs);
}

CompletedRun
runSystemCell(llvm::StringRef test, ExecutionMatrixCell cell,
              ExecutionMatrixAttemptKind attempt, const SharedFixture &fixture,
              const Gem5Readiness &readiness, ArtifactStore &artifacts,
              BlobStore &blobs, const deployment::test::TemporaryTree &tree,
              ExecutionMatrixLifecycleRecorder &lifecycle,
              const runtime::FinalizedGem5SimulationBinding *sharedGem5Binding =
                  nullptr) {
  std::optional<runtime::FinalizedGem5SimulationBinding> ownedGem5Binding;
  if (!sharedGem5Binding) {
    ExecutionMatrixLifecycleTimer timer(
        lifecycle, ExecutionMatrixLifecycleOperation::Gem5Binding);
    ownedGem5Binding.emplace(
        buildGem5Binding(test, fixture.hardware.system, fixture.interconnect,
                         readiness.buildIdentity, artifacts));
    sharedGem5Binding = &*ownedGem5Binding;
  }
  const runtime::FinalizedGem5SimulationBinding &gem5Binding =
      *sharedGem5Binding;
  struct InvocationInputs final {
    evaluation::CaseArtifactResolution resolution;
    evaluation::EvaluationRequest request;
    external_tool::LocalToolConfig local;
  };
  InvocationInputs inputs = [&] {
    ExecutionMatrixLifecycleTimer timer(
        lifecycle, ExecutionMatrixLifecycleOperation::RequestConstruction);
    std::string verilatorIdentity;
    std::optional<ToolBinding> verilator;
    if (cell == ExecutionMatrixCell::SystemRtl) {
      verilator.emplace(
          resolveHostTool(test, external_tool::verilatorProvider(), tree));
      verilatorIdentity = verilator->resolved.version;
    }
    auto resolution =
        buildResolution(test, fixture, gem5Binding.reference(),
                        fixture.systemInputs.workloadReference,
                        fixture.systemInputs.runtimeInputReference, true);
    auto request =
        buildSystemRequest(test, cell, fixture, gem5Binding, verilatorIdentity,
                           resolution, artifacts, blobs);
    external_tool::LocalToolConfig local;
    local.runtimePolicy = external_tool::RuntimePolicy::Host;
    auto &gem5 = local.tools[external_tool::gem5Provider().binding.key];
    gem5.binding.executable = readiness.binary;
    gem5.providerOptions["readiness"] = readiness.path;
    if (verilator) {
      auto &verilatorConfig =
          local.tools[external_tool::verilatorProvider().binding.key];
      verilatorConfig =
          std::move(verilator->local
                        .tools[external_tool::verilatorProvider().binding.key]);
      verilatorConfig.providerOptions["max_cycles"] = 128;
      verilatorConfig.providerOptions["build_jobs"] = LOOM_TEST_BUILD_JOBS;
      verilatorConfig.providerOptions["build_workers"] =
          LOOM_TEST_RTL_BUILD_WORKER_LIMIT;
    }
    return InvocationInputs{std::move(resolution), std::move(request),
                            std::move(local)};
  }();
  CompletedRun completed =
      attempt == ExecutionMatrixAttemptKind::Ordinary
          ? runExternal(test, inputs.request, inputs.resolution,
                        std::move(inputs.local), tree.path("system-bundle"),
                        artifacts, blobs, &lifecycle)
          : runGem5Diagnostic(test, inputs.request, inputs.resolution,
                              std::move(inputs.local),
                              tree.path("system-diagnostic-bundle"), artifacts,
                              blobs, lifecycle);
  if (completed.gem5Diagnostics) {
    const CompletedRun::DiagnosticSummary &diagnostics =
        *completed.gem5Diagnostics;
    const std::uint64_t expectedBridgeCount = isPairedCell(cell) ? 1 : 4;
    require(test,
            diagnostics.attemptProfile.bridgeCount == expectedBridgeCount &&
                diagnostics.attemptProfile.acceleratorInvocationCount ==
                    diagnostics.spatialInvocations.size() &&
                !diagnostics.spatialInvocations.empty(),
            "gem5 diagnostics differ from the exact bridge execution");
    if (usesCgraEngine(cell))
      require(test,
              diagnostics.attemptProfile.cgraEngine.has_value() &&
                  diagnostics.attemptProfile.cgraEngine->invocationCount ==
                      diagnostics.spatialInvocations.size(),
              "CGRA engine diagnostics differ from bridge retirement");
    else
      require(test, !diagnostics.attemptProfile.cgraEngine.has_value(),
              "non-CGRA execution published a CGRA engine profile");
    require(
        test,
        diagnostics.attemptProfile.engineProcessCpuNanoseconds.has_value() ==
            (usesCgraEngine(cell) || cell == ExecutionMatrixCell::SystemDfg),
        "gem5 engine CPU observation has the wrong ownership");
  }
  if (cell == ExecutionMatrixCell::PairedSystemCgra)
    require(test,
            completed.gem5Diagnostics.has_value() &&
                completed.gem5Diagnostics->spatialInvocations.size() ==
                    pairedMeasurementInvocationCount,
            "paired System execution has the wrong launch count");
  if (cell == ExecutionMatrixCell::SystemCgra &&
      attempt == ExecutionMatrixAttemptKind::Ordinary) {
    auto sample = take(
        test, evaluation::models::importSystemRuntimeTrainingEvidenceSample(
                  completed.evidenceReference, artifacts, blobs));
    require(test,
            !sample.groundTruthTargetKey.empty() &&
                !sample.sampleGroupKey.empty() &&
                sample.features.fabric.acceleratorCoreOccurrenceCount == 4 &&
                sample.features.mapping.spatialContextDomainCount == 4,
            "System Runtime training projection lost the heterogeneous "
            "execution context");
  }
  return completed;
}

std::uint64_t
deterministicWork(const sim::CanonicalSimulationExecution &execution) {
  if (const auto *spatial = execution.spatial())
    return spatial->progressObservations.terminalObserved.referenceCycle
        .numerator();
  return execution.system()->progressObservations.terminalObserved.gem5Tick;
}

std::uint64_t durationNanoseconds(std::chrono::steady_clock::duration value) {
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
  return nanoseconds > 0 ? static_cast<std::uint64_t>(nanoseconds) : 0;
}

std::uint64_t processCpuNanoseconds(llvm::StringRef test) {
  timespec current{};
  require(test, ::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current) == 0,
          "cannot sample the process CPU clock");
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  require(test,
          current.tv_sec >= 0 && current.tv_nsec >= 0 &&
              static_cast<std::uint64_t>(current.tv_nsec) <
                  nanosecondsPerSecond &&
              static_cast<std::uint64_t>(current.tv_sec) <=
                  (std::numeric_limits<std::uint64_t>::max() -
                   static_cast<std::uint64_t>(current.tv_nsec)) /
                      nanosecondsPerSecond,
          "process CPU clock is outside the measurement domain");
  return static_cast<std::uint64_t>(current.tv_sec) * nanosecondsPerSecond +
         static_cast<std::uint64_t>(current.tv_nsec);
}

std::uint64_t
spatialReferenceCycles(llvm::StringRef test,
                       const sim::SpatialProgressObservations &progress) {
  const std::optional<std::uint64_t> cycles =
      runtime::integralSpatialReferenceCycleDistance(progress);
  require(test, cycles.has_value(),
          "Spatial measurement has a nonintegral or reversed cycle interval");
  return *cycles;
}

std::uint64_t peakResidentBytes(llvm::StringRef test, const rusage &usage) {
  require(test, usage.ru_maxrss >= 0,
          "resident-memory observation is outside its domain");
  const std::uint64_t kibibytes = usage.ru_maxrss;
  require(test, kibibytes <= std::numeric_limits<std::uint64_t>::max() / 1024,
          "resident-memory observation overflows bytes");
  return kibibytes * 1024;
}

struct PairedMeasurement final {
  std::uint64_t acceleratorReferenceCycles = 0;
  std::uint64_t eventFrames = 0;
  std::uint64_t activeWallNanoseconds = 0;
  std::uint64_t activeCpuNanoseconds = 0;
  std::optional<std::uint64_t> gem5Ticks;
  std::uint64_t setupWallNanoseconds = 0;
  std::uint64_t peakResidentBytes = 0;
  llvm::StringRef source;
  llvm::StringRef rssScope;
};

void recordPairedMeasurement(ExecutionMatrixCell cell,
                             const PairedFingerprints &fingerprints,
                             const PairedMeasurement &measurement) {
  llvm::StringRef test = executionMatrixCellName(cell);
  require(test,
          isPairedCell(cell) && measurement.acceleratorReferenceCycles != 0 &&
              measurement.eventFrames != 0 &&
              measurement.activeWallNanoseconds != 0,
          "paired measurement contains no completed active work");
  const ExecutionMatrixInvocation invocation{
      cell, cell == ExecutionMatrixCell::PairedSystemCgra
                ? ExecutionMatrixAttemptKind::Diagnostic
                : ExecutionMatrixAttemptKind::Ordinary};
  llvm::outs() << "paired-simulation"
               << " schema=loom.paired_simulation_measurement.2"
               << " cell=" << executionMatrixCellName(cell)
               << " attempt=" << executionMatrixAttemptName(invocation.attempt)
               << " invocation=" << executionMatrixInvocationName(invocation)
               << " work_fingerprint=" << fingerprints.work
               << " config_fingerprint=" << fingerprints.config
               << " accelerator_reference_cycles="
               << measurement.acceleratorReferenceCycles
               << " cgra_event_frames=" << measurement.eventFrames
               << " active_wall_ns=" << measurement.activeWallNanoseconds
               << " active_cpu_ns=" << measurement.activeCpuNanoseconds
               << " gem5_ticks=";
  if (measurement.gem5Ticks)
    llvm::outs() << *measurement.gem5Ticks;
  else
    llvm::outs() << "not_applicable";
  llvm::outs() << " setup_wall_ns=" << measurement.setupWallNanoseconds
               << " process_peak_rss_bytes=" << measurement.peakResidentBytes
               << " measurement_source=" << measurement.source
               << " rss_scope=" << measurement.rssScope << '\n';
}

void recordRunStatistics(ExecutionMatrixInvocation matrixInvocation,
                         const CompletedRun &completed, const rusage &after) {
  emitExecutionMatrixExternalCommands(matrixInvocation,
                                      completed.externalCommands);
  const CompletedRun::DiagnosticSummary *diagnostics =
      completed.gem5Diagnostics ? &*completed.gem5Diagnostics : nullptr;
  requireSuccess(
      executionMatrixCellName(matrixInvocation.cell),
      emitExecutionMatrixRunSummary(
          matrixInvocation, deterministicWork(completed.execution),
          after.ru_maxrss, diagnostics ? &diagnostics->attemptProfile : nullptr,
          diagnostics
              ? llvm::ArrayRef(diagnostics->spatialInvocations)
              : llvm::ArrayRef<runtime::Gem5SpatialInvocationProjection>{}));
}

struct ReplaySignature final {
  std::vector<ArtifactRootReference> roots;
  std::uint64_t work = 0;
  std::optional<sim::SystemFunctionalObservations> systemFunctional;
};

ReplaySignature runExecutionMatrixCellOnce(ExecutionMatrixInvocation invocation,
                                           llvm::StringRef readinessPath,
                                           llvm::StringRef treeName,
                                           bool emitStatistics) {
  const ExecutionMatrixCell cell = invocation.cell;
  const llvm::StringRef test = executionMatrixCellName(cell);
  ExecutionMatrixLifecycleRecorder lifecycle;
  std::optional<ExecutionMatrixLifecycleTimer> cleanupTimer;
  ReplaySignature signature = [&]() -> ReplaySignature {
    deployment::test::TemporaryTree tree(treeName);
    ArtifactStore artifacts(tree.path("artifacts"));
    BlobStore blobs(tree.path("blobs"));

    std::optional<ExecutionMatrixLifecycleTimer> setupTimer;
    setupTimer.emplace(lifecycle, ExecutionMatrixLifecycleOperation::Setup);
    SharedFixture fixture =
        buildSharedFixture(test, cell, artifacts, blobs, tree, &lifecycle);
    const std::optional<PairedFingerprints> pairedFingerprintsValue =
        isPairedCell(cell) ? std::optional<PairedFingerprints>(
                                 pairedFingerprints(test, fixture))
                           : std::nullopt;
    traceSpatialFixture(test, fixture, artifacts);
    const std::uint64_t setupWallNanoseconds = setupTimer->finish();
    setupTimer.reset();

    ExecutionMatrixImportSessions importSessions(artifacts, blobs);

    rusage after{};
    CompletedRun completed = [&] {
      ExecutionMatrixLifecycleTimer hostTimer(
          lifecycle, ExecutionMatrixLifecycleOperation::HostLifecycle);
      if (isSpatialCell(cell))
        return runSpatialCell(test, cell, fixture, artifacts, blobs, tree);
      const Gem5Readiness readiness = [&] {
        ExecutionMatrixLifecycleTimer timer(
            lifecycle, ExecutionMatrixLifecycleOperation::Gem5Readiness);
        return readGem5Readiness(test, readinessPath);
      }();
      return runSystemCell(test, cell, invocation.attempt, fixture, readiness,
                           artifacts, blobs, tree, lifecycle);
    }();
    if (emitStatistics)
      require(test, getrusage(RUSAGE_CHILDREN, &after) == 0,
              "cannot sample child resource use");
    {
      ExecutionMatrixLifecycleTimer oracleTimer(
          lifecycle, ExecutionMatrixLifecycleOperation::OracleVerification);
      if (completed.execution.spatial())
        requireSpatialOracle(test, completed.execution, isPairedCell(cell));
      else
        requireSystemOracle(test, completed.execution);
    }
    if (emitStatistics && cell == ExecutionMatrixCell::PairedSystemCgra) {
      require(test,
              pairedFingerprintsValue.has_value() &&
                  completed.gem5Diagnostics.has_value() &&
                  invocation.attempt == ExecutionMatrixAttemptKind::Diagnostic,
              "paired System measurement lost its attempt-local diagnostics");
      const CompletedRun::DiagnosticSummary &diagnostics =
          *completed.gem5Diagnostics;
      require(test,
              diagnostics.attemptProfile.cgraEngine.has_value() &&
                  diagnostics.spatialInvocations.size() ==
                      pairedMeasurementInvocationCount,
              "paired System measurement lacks exact CGRA work");
      const runtime::Gem5CgraEngineAttemptProfile &engine =
          *diagnostics.attemptProfile.cgraEngine;
      std::uint64_t totalCycles = 0;
      for (const runtime::Gem5SpatialInvocationProjection &invocation :
           diagnostics.spatialInvocations) {
        require(test,
                invocation.acceleratorReferenceCycles.has_value() &&
                    *invocation.acceleratorReferenceCycles <=
                        std::numeric_limits<std::uint64_t>::max() - totalCycles,
                "paired System measurement cycle total is invalid");
        totalCycles += *invocation.acceleratorReferenceCycles;
      }
      recordPairedMeasurement(
          cell, *pairedFingerprintsValue,
          {totalCycles, engine.eventFrameCount,
           diagnostics.attemptProfile.simulationWallNanoseconds,
           diagnostics.attemptProfile.gem5SimulationProcessCpuNanoseconds,
           diagnostics.gem5Ticks, setupWallNanoseconds,
           peakResidentBytes(test, after), "fresh_system_diagnostic",
           "child_process_lifetime"});
    } else if (emitStatistics)
      recordRunStatistics(invocation, completed, after);

    std::vector<ArtifactRootReference> roots{
        fixture.dataflowReference,
        fixture.hardware.module.reference(),
        fixture.hardware.system.reference(),
        fixture.hardware.techMapping,
        fixture.hardware.spatialMapping.reference(),
        fixture.systemMapping.reference(),
        fixture.interconnect,
        fixture.deployment.reference(),
    };
    for (const hardware::FinalizedHardwareImplementation &implementation :
         fixture.hardware.implementations)
      roots.push_back(implementation.reference());
    if (isSpatialCell(cell)) {
      roots.push_back(fixture.spatialWorkload);
      roots.push_back(fixture.spatialRuntimeInput);
    } else {
      roots.push_back(fixture.systemInputs.workloadReference);
      roots.push_back(fixture.systemInputs.runtimeInputReference);
    }
    roots.push_back(completed.evidence.requestRef());
    roots.push_back(completed.evidenceReference);
    roots.push_back(
        completed.evidence.outputBindings().front().artifacts.front());
    // Gem5 System progress is anchored to the EventQueue, while the external
    // bridge is serviced through host poll events. Keep those timing-dependent
    // result roots out of the structural replay signature; functional
    // observations remain compared exactly below.
    roots.erase(std::remove_if(
                    roots.begin(), roots.end(),
                    [](const ArtifactRootReference &root) {
                      return root.schemaIdentity ==
                                 evaluation::EvaluationEvidence::artifactSchema
                                     .identity ||
                             root.schemaIdentity ==
                                 sim::simulationExecutionSchema.identity;
                    }),
                roots.end());
    std::optional<sim::SystemFunctionalObservations> systemFunctional;
    if (const auto *system = completed.execution.system())
      systemFunctional = system->functionalObservations;
    if (!isSpatialCell(cell))
      require(test, importSessions.reusedOneExactGem5FactsClosure(),
              "System execution did not reuse one exact Gem5 facts closure");
    if (emitStatistics)
      importSessions.emitStatistics(invocation);
    cleanupTimer.emplace(lifecycle, ExecutionMatrixLifecycleOperation::Cleanup);
    return {std::move(roots), deterministicWork(completed.execution),
            std::move(systemFunctional)};
  }();
  cleanupTimer.reset();
  if (emitStatistics)
    lifecycle.emit(invocation);
  return signature;
}

} // namespace

void runExecutionMatrixCell(ExecutionMatrixInvocation invocation,
                            llvm::StringRef gem5ReadinessPath) {
  const ExecutionMatrixCell cell = invocation.cell;
  if (cell == ExecutionMatrixCell::PairedSpatialCgra) {
    runPairedSpatialCgraBatch(1, 1);
    return;
  }
  requireSuccess(executionMatrixCellName(cell),
                 evaluation::registerProductionEvaluationRegistry());
  requireSuccess(executionMatrixCellName(cell),
                 eda::open_source::registerMappedRtlSimulationProvider());
  (void)runExecutionMatrixCellOnce(
      invocation, gem5ReadinessPath,
      "execution-matrix-" + executionMatrixInvocationName(invocation), true);
}

void runSystemExecutionAttemptPair(ExecutionMatrixCell cell,
                                   llvm::StringRef gem5ReadinessPath) {
  const llvm::StringRef test = executionMatrixCellName(cell);
  require(test,
          cell == ExecutionMatrixCell::SystemCgra ||
              cell == ExecutionMatrixCell::SystemRtl,
          "attempt pair requires a CGRA or RTL System cell");
  requireSuccess(test, evaluation::registerProductionEvaluationRegistry());
  requireSuccess(test, eda::open_source::registerMappedRtlSimulationProvider());

  const ExecutionMatrixInvocation ordinaryInvocation{
      cell, ExecutionMatrixAttemptKind::Ordinary};
  const ExecutionMatrixInvocation diagnosticInvocation{
      cell, ExecutionMatrixAttemptKind::Diagnostic};
  ExecutionMatrixLifecycleRecorder pairLifecycle;
  ExecutionMatrixLifecycleRecorder ordinaryLifecycle;
  ExecutionMatrixLifecycleRecorder diagnosticLifecycle;
  std::optional<ExecutionMatrixLifecycleTimer> cleanupTimer;
  [&] {
    deployment::test::TemporaryTree tree(
        "execution-matrix-" + std::string(executionMatrixCellName(cell)) +
        "-attempt-pair");
    ArtifactStore artifacts(tree.path("artifacts"));
    BlobStore blobs(tree.path("blobs"));

    std::optional<ExecutionMatrixLifecycleTimer> setupTimer;
    setupTimer.emplace(pairLifecycle, ExecutionMatrixLifecycleOperation::Setup);
    SharedFixture fixture =
        buildSharedFixture(test, cell, artifacts, blobs, tree, &pairLifecycle);
    traceSpatialFixture(test, fixture, artifacts);
    const std::uint64_t setupWallNanoseconds = setupTimer->finish();
    setupTimer.reset();
    const Gem5Readiness readiness = [&] {
      ExecutionMatrixLifecycleTimer timer(
          pairLifecycle, ExecutionMatrixLifecycleOperation::Gem5Readiness);
      return readGem5Readiness(test, gem5ReadinessPath);
    }();
    ExecutionMatrixImportSessions importSessions(artifacts, blobs);

    std::optional<runtime::FinalizedGem5SimulationBinding> gem5Binding;
    {
      ExecutionMatrixLifecycleTimer bindingTimer(
          pairLifecycle, ExecutionMatrixLifecycleOperation::Gem5Binding);
      gem5Binding.emplace(buildGem5Binding(test, fixture.hardware.system,
                                           fixture.interconnect,
                                           readiness.buildIdentity, artifacts));
    }
    CompletedRun ordinary = [&] {
      ExecutionMatrixLifecycleTimer hostTimer(
          ordinaryLifecycle, ExecutionMatrixLifecycleOperation::HostLifecycle);
      return runSystemCell(test, cell, ExecutionMatrixAttemptKind::Ordinary,
                           fixture, readiness, artifacts, blobs, tree,
                           ordinaryLifecycle, &*gem5Binding);
    }();
    rusage ordinaryUsage{};
    require(test, getrusage(RUSAGE_CHILDREN, &ordinaryUsage) == 0,
            "cannot sample ordinary child resource use");
    {
      ExecutionMatrixLifecycleTimer oracleTimer(
          ordinaryLifecycle,
          ExecutionMatrixLifecycleOperation::OracleVerification);
      requireSystemOracle(test, ordinary.execution);
    }
    recordRunStatistics(ordinaryInvocation, ordinary, ordinaryUsage);
    require(test, importSessions.reusedOneExactGem5FactsClosure(),
            "ordinary attempt did not construct and reuse one Gem5 facts "
            "closure");
    importSessions.emitStatistics(ordinaryInvocation);

    CompletedRun diagnostic = [&] {
      ExecutionMatrixLifecycleTimer hostTimer(
          diagnosticLifecycle,
          ExecutionMatrixLifecycleOperation::HostLifecycle);
      return runSystemCell(test, cell, ExecutionMatrixAttemptKind::Diagnostic,
                           fixture, readiness, artifacts, blobs, tree,
                           diagnosticLifecycle, &*gem5Binding);
    }();
    rusage diagnosticUsage{};
    require(test, getrusage(RUSAGE_CHILDREN, &diagnosticUsage) == 0,
            "cannot sample diagnostic child resource use");
    {
      ExecutionMatrixLifecycleTimer oracleTimer(
          diagnosticLifecycle,
          ExecutionMatrixLifecycleOperation::OracleVerification);
      requireSystemOracle(test, diagnostic.execution);
    }
    recordRunStatistics(diagnosticInvocation, diagnostic, diagnosticUsage);
    require(test,
            importSessions.reusedOneExactGem5FactsClosureAcrossAttemptPair(),
            "diagnostic attempt reconstructed the shared Gem5 facts closure");
    importSessions.emitStatistics(diagnosticInvocation);

    const sim::SystemSimulationExecution *ordinarySystem =
        ordinary.execution.system();
    const sim::SystemSimulationExecution *diagnosticSystem =
        diagnostic.execution.system();
    require(test,
            ordinarySystem && diagnosticSystem &&
                sim::haveExactlyEqualSystemFunctionalObservations(
                    ordinarySystem->functionalObservations,
                    diagnosticSystem->functionalObservations),
            "ordinary and diagnostic attempts changed canonical observations");
    const std::uint64_t setupConstructions =
        pairLifecycle.operationCount(ExecutionMatrixLifecycleOperation::Setup);
    const std::uint64_t gem5BindingConstructions = pairLifecycle.operationCount(
        ExecutionMatrixLifecycleOperation::Gem5Binding);
    require(test,
            setupConstructions == 1 && gem5BindingConstructions == 1 &&
                ordinaryLifecycle.operationCount(
                    ExecutionMatrixLifecycleOperation::Setup) == 0 &&
                diagnosticLifecycle.operationCount(
                    ExecutionMatrixLifecycleOperation::Setup) == 0 &&
                ordinaryLifecycle.operationCount(
                    ExecutionMatrixLifecycleOperation::Gem5Binding) == 0 &&
                diagnosticLifecycle.operationCount(
                    ExecutionMatrixLifecycleOperation::Gem5Binding) == 0,
            "attempt-local lifecycle reconstructed shared setup state");
    const ExecutionMatrixImportSummary imports = importSessions.summary();
    llvm::outs() << "execution-matrix-attempt-pair"
                 << " schema=loom.execution_matrix_attempt_pair.1"
                 << " cell=" << executionMatrixCellName(cell)
                 << " setup_constructions=" << setupConstructions
                 << " gem5_binding_constructions=" << gem5BindingConstructions
                 << " setup_wall_ns=" << setupWallNanoseconds
                 << " gem5_facts_requests=" << imports.gem5FactsRequests
                 << " gem5_facts_hits=" << imports.gem5FactsHits
                 << " gem5_facts_misses=" << imports.gem5FactsMisses
                 << " gem5_facts_construction_attempts="
                 << imports.gem5FactsConstructionAttempts
                 << " gem5_facts_unique_constructions="
                 << imports.gem5FactsUniqueConstructions
                 << " gem5_facts_revalidations="
                 << imports.gem5FactsRevalidationCount
                 << " gem5_facts_entries=" << imports.gem5FactsEntryCount
                 << '\n';
    cleanupTimer.emplace(pairLifecycle,
                         ExecutionMatrixLifecycleOperation::Cleanup);
  }();
  cleanupTimer.reset();
  pairLifecycle.emitAttemptPair(cell);
  ordinaryLifecycle.emit(ordinaryInvocation);
  diagnosticLifecycle.emit(diagnosticInvocation);
}

void runPairedSpatialCgraBatch(std::uint64_t warmupRuns,
                               std::uint64_t measurementRuns) {
  constexpr ExecutionMatrixCell cell = ExecutionMatrixCell::PairedSpatialCgra;
  const llvm::StringRef test = executionMatrixCellName(cell);
  require(test,
          warmupRuns != 0 && measurementRuns != 0 &&
              warmupRuns <=
                  std::numeric_limits<std::uint64_t>::max() - measurementRuns,
          "paired Spatial batch requires bounded warmup and measurement runs");
  requireSuccess(test, evaluation::registerProductionEvaluationRegistry());
  requireSuccess(test, eda::open_source::registerMappedRtlSimulationProvider());

  deployment::test::TemporaryTree tree("execution-matrix-paired-spatial-batch");
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const auto setupStart = std::chrono::steady_clock::now();
  SharedFixture fixture =
      buildSharedFixture(test, cell, artifacts, blobs, tree);
  const PairedFingerprints fingerprints = pairedFingerprints(test, fixture);
  traceSpatialFixture(test, fixture, artifacts);
  auto prepared = take(
      test, evaluation::models::prepareCgraSimulationEvaluation(
                fixture.dataflowReference, fixture.hardware.module.reference(),
                fixture.hardware.spatialMapping.reference(),
                fixture.spatialWorkload, fixture.spatialRuntimeInput,
                defaultResolvedConfig(), artifacts, blobs));
  auto ordinaryEvidence =
      take(test, evaluation::models::evaluateCgraSimulation(
                     prepared, {runtime::gem5MaximumSpatialWork, std::nullopt},
                     artifacts, blobs));
  (void)take(
      test, evaluation::publishEvaluationEvidence(ordinaryEvidence, artifacts));
  CompletedRun ordinary = importCompleted(
      test, std::move(ordinaryEvidence), prepared.resolution, artifacts, blobs);
  requireSpatialOracle(test, ordinary.execution, true);
  const sim::SpatialSimulationExecution *ordinarySpatial =
      ordinary.execution.spatial();
  require(test, ordinarySpatial != nullptr,
          "paired Spatial oracle published a System execution");
  const std::uint64_t expectedCycles =
      spatialReferenceCycles(test, ordinarySpatial->progressObservations);
  require(test,
          expectedCycles <= std::numeric_limits<std::uint64_t>::max() /
                                pairedMeasurementInvocationCount,
          "paired Spatial cycle total overflows");
  const std::uint64_t totalCycles =
      expectedCycles * pairedMeasurementInvocationCount;
  const auto setupEnd = std::chrono::steady_clock::now();

  std::optional<std::uint64_t> expectedEventFrames;
  const std::uint64_t runCount = warmupRuns + measurementRuns;
  for (std::uint64_t ordinal = 0; ordinal != runCount; ++ordinal) {
    const std::uint64_t cpuStarted = processCpuNanoseconds(test);
    const auto wallStarted = std::chrono::steady_clock::now();
    for (std::uint64_t invocation = 0;
         invocation != pairedMeasurementInvocationCount; ++invocation) {
      auto outcome =
          take(test, sim::simulateCgraWorkload(
                         prepared.workloadExecution, prepared.workload,
                         prepared.runtimeInput, runtime::gem5MaximumSpatialWork,
                         std::nullopt));
      require(test,
              outcome.state == sim::SpatialExecutionSessionState::Retired &&
                  outcome.retired.has_value(),
              "paired Spatial active attempt did not retire");
      const sim::RetiredCgraSimulation &retired = *outcome.retired;
      require(test,
              sim::haveExactlyEqualSpatialFunctionalObservations(
                  ordinarySpatial->functionalObservations,
                  retired.observations),
              "paired Spatial active attempt differs from ordinary Evidence");
      const std::uint64_t cycles =
          spatialReferenceCycles(test, retired.progress);
      require(test, cycles == expectedCycles,
              "paired Spatial active attempt changed reference-cycle work");
      if (!expectedEventFrames)
        expectedEventFrames = retired.counters.eventFrameCount;
      else
        require(test, retired.counters.eventFrameCount == *expectedEventFrames,
                "paired Spatial active attempt changed event-frame work");
    }
    const auto wallFinished = std::chrono::steady_clock::now();
    const std::uint64_t cpuFinished = processCpuNanoseconds(test);
    require(test,
            *expectedEventFrames <=
                std::numeric_limits<std::uint64_t>::max() /
                    pairedMeasurementInvocationCount,
            "paired Spatial event-frame total overflows");
    if (ordinal < warmupRuns)
      continue;
    rusage usage{};
    require(test, ::getrusage(RUSAGE_SELF, &usage) == 0,
            "cannot sample paired Spatial resident memory");
    recordPairedMeasurement(cell, fingerprints,
                            {totalCycles,
                             *expectedEventFrames *
                                 pairedMeasurementInvocationCount,
                             durationNanoseconds(wallFinished - wallStarted),
                             cpuFinished - cpuStarted, std::nullopt,
                             durationNanoseconds(setupEnd - setupStart),
                             peakResidentBytes(test, usage),
                             "direct_spatial_attempt",
                             "self_process_lifetime"});
  }
}

void verifyDeterministicSystemReplay(llvm::StringRef gem5ReadinessPath) {
  constexpr std::size_t replayCount = 3;
  const llvm::StringRef test = "deterministic-system-replay";
  requireSuccess(test, evaluation::registerProductionEvaluationRegistry());
  requireSuccess(test, eda::open_source::registerMappedRtlSimulationProvider());
  std::array<std::future<ReplaySignature>, replayCount> pending;
  for (std::size_t replay = 0; replay < replayCount; ++replay) {
    pending[replay] = std::async(std::launch::async, [=] {
      const ExecutionMatrixAttemptKind attempt =
          replay + 1 == replayCount ? ExecutionMatrixAttemptKind::Diagnostic
                                    : ExecutionMatrixAttemptKind::Ordinary;
      return runExecutionMatrixCellOnce(
          {ExecutionMatrixCell::SystemCgra, attempt}, gem5ReadinessPath,
          test.str() + "-" + std::to_string(replay), false);
    });
  }
  std::optional<ReplaySignature> expected;
  for (std::size_t replay = 0; replay < replayCount; ++replay) {
    ReplaySignature observed = pending[replay].get();
    if (!expected) {
      expected = std::move(observed);
      continue;
    }
    require(test, observed.roots == expected->roots,
            "stable artifact roots changed across clean replay");
    require(test,
            observed.systemFunctional.has_value() &&
                expected->systemFunctional.has_value() &&
                sim::haveExactlyEqualSystemFunctionalObservations(
                    *observed.systemFunctional, *expected->systemFunctional),
            "System functional observations changed across clean replay");
  }
  llvm::outs() << "execution-matrix ordinary_replays=" << replayCount - 1
               << " diagnostic_replays=1"
               << " roots=" << expected->roots.size()
               << " deterministic_work=" << expected->work << '\n';
}

void verifyHeterogeneousSystemAnchor() {
  const llvm::StringRef test = "heterogeneous-system-anchor";
  requireSuccess(test, evaluation::registerProductionEvaluationRegistry());
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  SharedFixture fixture = buildSharedFixture(
      test, ExecutionMatrixCell::SpatialDfg, artifacts, blobs, tree);

  const auto dataflow = take(test, fixture.dataflow.view());
  require(test,
          dataflow.rootThreadLaunches().size() == 4 &&
              dataflow.staticGraphLaunches().size() == 4 &&
              dataflow.actors().size() >= 7,
          "canonical anchor does not contain four nonempty graph launches");

  const fabric::FabricArtifactView &module = fixture.hardware.module.view();
  require(test,
          module.peOccurrences().size() >= 4 &&
              module.switchOccurrences().size() > 1 &&
              !module.memoryOccurrences().empty() &&
              !module.fifoOccurrences().empty() &&
              !module.boundaryOccurrences().empty(),
          "SpatialCore anchor omits a required distributed hierarchy class");

  const auto system =
      take(test, fabric::requireSystemRoot(fixture.hardware.system.view()));
  require(test, system.artifact().accCoreOccurrences().size() == 4,
          "System anchor does not contain four AccCore occurrences");
  std::size_t inOrder = 0;
  std::size_t outOfOrder = 0;
  for (const fabric::AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    const auto *realization = system.instructionCoreMicroarchitecture(
        fabric::InstructionCoreContextRef{core});
    require(test, realization != nullptr,
            "AccCore has no InstructionCore realization");
    if (realization->kind() == fabric::InstructionCoreRealizationKind::InOrder)
      ++inOrder;
    else
      ++outOfOrder;
  }
  require(test, inOrder == 2 && outOfOrder == 2,
          "System anchor does not retain heterogeneous and repeated cores");
  require(test,
          !system.artifact().systemMemoryServices().empty() &&
              system.transportResources().size() >= 5 &&
              system.artifact().pointConnections().size() >= 10,
          "System anchor omits external memory or finite ring transport");

  auto contexts = take(
      test, mapping::projectSystemExecutionContexts(
                dataflow, fixture.systemMapping.view().executionBindings()));
  std::set<std::vector<std::uint8_t>> usedCores;
  for (const mapping::SystemSpatialContextDomain &domain :
       contexts.spatialDomains)
    usedCores.insert(fabric::canonicalFabricBytes(
        fabric::SpatialCoreOccurrenceRef{domain.context.accCore}));
  require(test,
          usedCores.size() == system.artifact().accCoreOccurrences().size(),
          "at least one released AccCore is unused by the SystemMapping");

  auto obligations = take(test, mapping::projectSystemServiceObligations(
                                    dataflow, fixture.systemMapping.view()
                                                  .executionBindings()
                                                  .rootThreadLaunches()));
  bool hasMulticast = false;
  for (const mapping::SystemServiceObligationProjection &obligation :
       obligations)
    hasMulticast |= obligation.sinks.size() >= 2;
  require(test, hasMulticast,
          "SystemMapping does not preserve the shared channel multicast");
  require(test,
          fixture.systemInputs.workload.system()
                      ->observableContract.memories.size() == 1 &&
              fixture.systemInputs.runtimeInput.system()
                      ->memoryInterfaceBindings.size() == 1,
          "System anchor has no terminal memory observable");

  const auto tech =
      take(test,
           mapping::importTechMapping(fixture.hardware.techMapping, artifacts));
    auto progress = take(
      test, mapping::deriveSpatialMappingProgressClosure(
                dataflow, tech.view(), module,
                fixture.hardware.spatialMapping.view().computeBindings(),
                fixture.hardware.spatialMapping.view().registerFifoTransfers(),
                fixture.hardware.spatialMapping.view().routeTrees(),
                fixture.hardware.spatialMapping.view().resourceUses(),
                fixture.hardware.spatialMapping.view().physicalTagSegments()));
  require(test,
          progress.kind ==
              mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet,
          "selected routes do not close the canonical progress proof");
  std::vector<mapping::SpatialRouteTreeView> unbufferedRoutes(
      fixture.hardware.spatialMapping.view().routeTrees().begin(),
      fixture.hardware.spatialMapping.view().routeTrees().end());
  bool removedProgressBoundary = false;
  auto bypassBuffered = [&](auto &traversal) {
    if (!traversal)
      return;
    const auto *fifo =
        std::get_if<fabric::FabricFifoTraversalPayload>(&traversal->payload);
    if (!fifo || fifo->mode != fabric::FabricFifoTraversalMode::Buffered)
      return;
    traversal = fabric::FabricPhysicalTraversalRef::fifoTraversal(
        fifo->owner, fabric::FabricFifoTraversalMode::Bypass);
    removedProgressBoundary = true;
  };
  for (mapping::SpatialRouteTreeView &route : unbufferedRoutes) {
    bypassBuffered(route.localTraversal);
    for (mapping::SpatialRouteNodeView &node : route.nodes)
      bypassBuffered(node.incomingTraversal);
    for (mapping::SpatialRouteSinkView &sink : route.sinks)
      bypassBuffered(sink.localTraversal);
  }
  require(test, removedProgressBoundary,
          "canonical anchor has no Buffered FIFO progress boundary");
  auto closedWait = take(
      test, mapping::deriveSpatialMappingProgressClosure(
                dataflow, tech.view(), module,
                fixture.hardware.spatialMapping.view().computeBindings(),
                fixture.hardware.spatialMapping.view().registerFifoTransfers(),
                unbufferedRoutes,
                fixture.hardware.spatialMapping.view().resourceUses(),
                fixture.hardware.spatialMapping.view().physicalTagSegments()));
  require(test,
          closedWait.kind ==
              mapping::MappingProgressClosureKind::ProvenClosedWaitSet,
          "route verifier accepted the unbuffered atomic multicast");

  const auto binding =
      buildGem5Binding(test, fixture.hardware.system, fixture.interconnect,
                       structuralGem5Identity(), artifacts);
  require(test, !binding.binding().correspondences().empty(),
          "gem5 binding does not cover the heterogeneous System");
  llvm::outs() << "heterogeneous-system-anchor roots=4 acc_cores=4"
               << " spatial_pes=" << module.peOccurrences().size()
               << " switches=" << module.switchOccurrences().size()
               << " transport_resources=" << system.transportResources().size()
               << " service_realizations="
               << fixture.systemMapping.view().serviceRealizations().size()
               << '\n';
}

} // namespace loom::system_test
