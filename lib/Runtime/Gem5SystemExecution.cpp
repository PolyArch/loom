#include "Runtime/Gem5SystemExecution.h"

#include "Gem5SystemExecutionInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Deployment/Deployment.h"
#include "Deployment/DeploymentReference.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "Deployment/Package.h"
#include "EDA/Adapters/OpenSource/MappedRtlExecution.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Runtime/Gem5SpatialChannel.h"
#include "Runtime/Gem5SpatialChannelPlan.h"
#include "Runtime/SpatialInvocationWire.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialInvocation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Matchers.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#ifndef LOOM_GEM5_CONFIG_SCRIPT_PATH
#error "LOOM_GEM5_CONFIG_SCRIPT_PATH is required"
#endif
#ifndef LOOM_TIMEOUT_BUDGETS_PATH
#error "LOOM_TIMEOUT_BUDGETS_PATH is required"
#endif
#ifndef LOOM_GEM5_DFG_ENGINE_PATH
#error "LOOM_GEM5_DFG_ENGINE_PATH is required"
#endif
#ifndef LOOM_GEM5_CGRA_ENGINE_PATH
#error "LOOM_GEM5_CGRA_ENGINE_PATH is required"
#endif
#ifndef LOOM_GEM5_RTL_ENGINE_SOURCE_PATH
#error "LOOM_GEM5_RTL_ENGINE_SOURCE_PATH is required"
#endif
#ifndef LOOM_GEM5_BRIDGE_HEADER_PATH
#error "LOOM_GEM5_BRIDGE_HEADER_PATH is required"
#endif
#ifndef LOOM_GEM5_INVOCATION_WIRE_HEADER_PATH
#error "LOOM_GEM5_INVOCATION_WIRE_HEADER_PATH is required"
#endif
#ifndef LOOM_GEM5_CHANNEL_PLAN_HEADER_PATH
#error "LOOM_GEM5_CHANNEL_PLAN_HEADER_PATH is required"
#endif

namespace loom::runtime {
using namespace gem5_system;
namespace {

using namespace evaluation;
using namespace external_tool;

llvm::Expected<std::string> readFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return invalid("cannot read '" + path +
                   "': " + buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

llvm::Expected<std::filesystem::path>
readinessPath(const LocalToolConfig &config,
              const ExternalToolProviderDescriptor &provider,
              const ResolvedToolBinding &tool) {
  const auto configured = config.tools.find(provider.binding.key);
  if (configured != config.tools.end()) {
    if (const llvm::json::Value *value =
            configured->second.providerOptions.get("readiness")) {
      const auto path = value->getAsString();
      if (!path)
        return invalid("gem5.provider_options.readiness must be a string");
      std::filesystem::path configuredPath(path->str());
      if (!configuredPath.is_absolute() ||
          configuredPath.lexically_normal() != configuredPath)
        return invalid("gem5 readiness path must be absolute and canonical");
      return configuredPath;
    }
  }
  const std::filesystem::path executable(tool.executable);
  if (!executable.is_absolute())
    return invalid("resolved gem5 executable path is not absolute");
  const std::filesystem::path root =
      executable.parent_path().parent_path().parent_path();
  return root / "loom-gem5-readiness.json";
}

llvm::Expected<ReadinessIdentity>
verifyReadiness(const Gem5SystemFacts &facts,
                const FinalizedGem5SimulationBinding &binding,
                const LocalToolConfig &config,
                const ExternalToolProviderDescriptor &provider,
                const ResolvedToolBinding &tool) {
  auto path = readinessPath(config, provider, tool);
  if (!path)
    return path.takeError();
  auto contents = readFile(path->string());
  if (!contents)
    return contents.takeError();
  auto value = llvm::json::parse(*contents);
  if (!value)
    return invalid("gem5 readiness stamp is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return invalid("gem5 readiness stamp is not an object");
  const auto schema = object->getString("schema");
  const auto bridgeAbi = object->getString("bridge_abi_identity");
  const auto repository = object->getString("gem5_repository_identity");
  const auto commit = object->getString("gem5_full_commit_identity");
  const auto configuration = object->getString("build_configuration_digest");
  const auto binary = object->getString("binary");
  const auto binarySha = object->getString("binary_sha256");
  const auto versionProbe = object->getString("version_probe");
  if (!schema || !bridgeAbi || !repository || !commit || !configuration ||
      !binary || !binarySha || !versionProbe)
    return invalid("gem5 readiness stamp omits an identity field");
  const Gem5BuildIdentity &expected = binding.binding().gem5BuildIdentity();
  if (*schema != "loom.gem5_build_readiness.1" ||
      *bridgeAbi != binding.binding().bridgeAbiIdentity() ||
      *repository != expected.repositoryIdentity ||
      *commit != expected.fullCommitIdentity ||
      *configuration != expected.buildConfigurationDigest ||
      *binarySha != expected.binaryFingerprint)
    return invalid("gem5 readiness identity differs from the exact binding");
  std::error_code error;
  const std::filesystem::path resolvedTool =
      std::filesystem::weakly_canonical(tool.executable, error);
  if (error)
    return invalid("cannot canonicalize the resolved gem5 executable");
  const std::filesystem::path recordedBinary(binary->str());
  if (recordedBinary != resolvedTool || !versionProbe->contains(tool.version))
    return invalid("gem5 readiness does not describe the resolved executable");
  auto fingerprint = parseExternalFileFingerprint(*binarySha);
  if (!fingerprint)
    return fingerprint.takeError();
  auto observed = fingerprintExternalFile(tool.executable);
  if (!observed)
    return observed.takeError();
  if (*observed != *fingerprint)
    return invalid("resolved gem5 executable differs from its readiness stamp");
  for (const Gem5SpatialLaunchProjection &launch : facts.spatialLaunches) {
    if (launch.bridgeSessionOrdinal >= facts.spatialBridgeSessions.size())
      return invalid("Spatial launch names an absent bridge session");
    if (launch.launchPayload.size() >
        facts.spatialBridgeSessions[launch.bridgeSessionOrdinal]
            .bridge.maximumMessageBytes)
      return invalid("Deployment launch image exceeds a bridge message limit");
  }
  return ReadinessIdentity{binarySha->str(), std::move(*fingerprint)};
}

std::vector<std::string>
inheritedEnvironment(const LocalToolConfig &config,
                     const ExternalToolProviderDescriptor &provider) {
  const auto configured = config.tools.find(provider.binding.key);
  if (configured == config.tools.end())
    return {};
  return configured->second.inheritEnvironment;
}

llvm::Expected<std::string> renderProjection(const Gem5SystemFacts &facts,
                                             const ReadinessIdentity &readiness,
                                             bool diagnostics) {
  std::vector<std::vector<llvm::ArrayRef<llvm::StringLiteral>>>
      processorOperationClasses;
  processorOperationClasses.reserve(facts.processors.size());
  for (const Gem5ProcessorProjection &processor : facts.processors) {
    std::vector<llvm::ArrayRef<llvm::StringLiteral>> operationClasses;
    operationClasses.reserve(processor.executionUnits.size());
    for (const fabric::ExecutionUnitRecord &unit : processor.executionUnits) {
      auto projected = projectGem5O3OperationClasses(unit.operationClass);
      if (!projected)
        return projected.takeError();
      operationClasses.push_back(*projected);
    }
    processorOperationClasses.push_back(std::move(operationClasses));
  }
  std::vector<std::string> accCoreReferences;
  std::vector<std::string> executionContextKeys;
  accCoreReferences.reserve(facts.spatialBridgeSessions.size());
  executionContextKeys.reserve(facts.spatialLaunches.size());
  for (const Gem5SpatialBridgeSession &session : facts.spatialBridgeSessions)
    accCoreReferences.push_back(formatArtifactLocalPayloadHex(
        fabric::canonicalFabricBytes(session.accCore)));
  for (const Gem5SpatialLaunchProjection &launch : facts.spatialLaunches) {
    auto contextBytes = mapping::encodeExecutionContextKey(
        mapping::ExecutionContextKey(launch.context));
    if (!contextBytes)
      return contextBytes.takeError();
    executionContextKeys.push_back(
        formatArtifactLocalPayloadHex(*contextBytes));
  }
  std::string output;
  llvm::raw_string_ostream stream(output);
  llvm::json::OStream json(stream, 0);
  const std::uint64_t ticksPerCycle =
      facts.processors.front().parameters.clockPeriodTicks;
  const std::string engine = facts.engine == Gem5SystemEngine::Dfg
                                 ? kDfgEnginePath.str()
                                 : kCgraEnginePath.str();
  json.object([&] {
    json.attribute("schema", "loom.gem5_system_projection.11");
    json.attribute("gem5_binary_sha256", readiness.binarySha256);
    json.attribute("clock", std::to_string(ticksPerCycle) + "ps");
    json.attributeObject("memory", [&] {
      json.attribute("base", facts.memory.baseAddress);
      json.attribute("size", facts.memory.sizeBytes);
      json.attribute("latency",
                     std::to_string(facts.memory.latencyTicks) + "ps");
    });
    json.attributeObject("host", [&] {
      json.attribute("elf", kHostElfPath);
      json.attribute("cpu_id", facts.hostCpuId);
      json.attribute("entry_symbol", facts.hostEntrySymbol);
      json.attribute("result_address",
                     facts.programResult ? facts.programResult->address : 0);
      json.attribute("result_size",
                     facts.programResult ? facts.programResult->size : 0);
      json.attribute("return_address", facts.hostReturnAddress);
    });
    json.attributeArray("instruction_images", [&] {
      for (const Gem5InstructionImage &image : facts.instructionImages)
        json.value(image.path);
    });
    json.attributeArray("runtime_images", [&] {
      for (const Gem5RuntimeImage &image : facts.runtimeImages)
        json.object([&] {
          json.attribute("path", image.path);
          json.attribute("address", image.address);
        });
    });
    json.attributeObject("system_memory", [&] {
      json.attribute("interface_table_address",
                     facts.memoryInterfaceTableAddress);
      json.attribute("interface_table_entries",
                     facts.memoryInterfaceTableEntries);
      json.attribute("observation_path", kMemoryResultPath);
      json.attributeArray("observations", [&] {
        if (facts.programResult)
          json.object([&] {
            json.attribute("address", facts.programResult->address);
            json.attribute("size", facts.programResult->size);
          });
        for (const Gem5MemoryObservationProjection &observation :
             facts.memoryObservations)
          json.object([&] {
            json.attribute("address", observation.address);
            json.attribute("size", observation.size);
          });
      });
    });
    json.attributeObject("dispatch", [&] {
      json.attribute("pio_address", facts.dispatchAddress);
      json.attribute("pio_latency", std::to_string(ticksPerCycle) + "ps");
      json.attribute("stack_base", facts.stackBase);
      json.attribute("stack_stride", facts.stackStride);
      json.attribute("root_event_trace_path", kRootLifecycleResultPath);
      json.attributeArray("targets", [&] {
        for (const Gem5SpatialLaunchProjection &launch :
             facts.spatialLaunches) {
          const Gem5DispatchTarget &target = launch.dispatchTarget;
          json.object([&] {
            json.attribute("cpu_id", target.cpuId);
            json.attribute("image_ordinal", target.imageOrdinal);
            json.attribute("entry_symbol", target.entrySymbol);
            json.attribute("bridge_address", target.bridgeAddress);
            json.attribute("launch_address", target.launchAddress);
            json.attribute("launch_size", target.launchSize);
          });
        }
      });
    });
    json.attributeArray("processors", [&] {
      for (std::size_t processorOrdinal = 0;
           processorOrdinal != facts.processors.size(); ++processorOrdinal) {
        const Gem5ProcessorProjection &processor =
            facts.processors[processorOrdinal];
        json.object([&] {
          json.attribute("cpu_id", processor.parameters.cpuId);
          json.attribute("model",
                         processor.model == Gem5ProcessorModelKind::TimingSimple
                             ? "timing_simple"
                             : "o3");
          json.attribute("num_threads", processor.hardwareThreadCount);
          json.attributeArray("execution_units", [&] {
            for (std::size_t unitOrdinal = 0;
                 unitOrdinal != processor.executionUnits.size();
                 ++unitOrdinal) {
              const fabric::ExecutionUnitRecord &unit =
                  processor.executionUnits[unitOrdinal];
              json.object([&] {
                json.attributeArray("operation_classes", [&] {
                  for (llvm::StringLiteral operationClass :
                       processorOperationClasses[processorOrdinal][unitOrdinal])
                    json.value(operationClass);
                });
                json.attribute("count", unit.count);
                json.attribute("latency_cycles", unit.latencyCycles);
                json.attribute("initiation_interval", unit.initiationInterval);
              });
            }
          });
          json.attributeObject("pipeline", [&] {
            if (!processor.outOfOrder)
              return;
            const auto &pipeline = *processor.outOfOrder;
            json.attribute("fetch_width", pipeline.fetchWidth);
            json.attribute("decode_width", pipeline.decodeWidth);
            json.attribute("rename_width", pipeline.renameWidth);
            json.attribute("dispatch_width", pipeline.dispatchWidth);
            json.attribute("issue_width", pipeline.issueWidth);
            json.attribute("writeback_width", pipeline.writebackWidth);
            json.attribute("commit_width", pipeline.commitWidth);
            json.attribute("reorder_buffer_entries",
                           pipeline.reorderBufferEntries);
            json.attribute("issue_queue_entries", pipeline.issueQueueEntries);
            json.attribute("load_queue_entries", pipeline.loadQueueEntries);
            json.attribute("store_queue_entries", pipeline.storeQueueEntries);
            json.attribute("physical_integer_registers",
                           pipeline.physicalIntegerRegisters);
            json.attribute("physical_float_registers",
                           pipeline.physicalFloatRegisters);
            json.attribute("physical_vector_registers",
                           pipeline.physicalVectorRegisters);
          });
        });
      }
    });
    json.attributeArray("bridges", [&] {
      for (const auto indexed : llvm::enumerate(facts.spatialBridgeSessions)) {
        const Gem5SpatialBridgeSession &session = indexed.value();
        const bool sharedEngine = facts.engine != Gem5SystemEngine::Rtl;
        const std::string socket =
            spatialBridgeSocketPath(sharedEngine ? 0 : indexed.index());
        json.object([&] {
          json.attributeArray("dispatch_target_ordinals", [&] {
            for (std::size_t launchOrdinal : session.launchOrdinals)
              json.value(launchOrdinal);
          });
          json.attribute("acc_core_ref", accCoreReferences[indexed.index()]);
          json.attributeArray("execution_context_keys", [&] {
            for (std::size_t launchOrdinal : session.launchOrdinals)
              json.value(executionContextKeys[launchOrdinal]);
          });
          json.attributeArray("spatial_workloads", [&] {
            for (std::size_t launchOrdinal : session.launchOrdinals)
              json.value(
                  formatArtifactIdentityHex(facts.spatialLaunches[launchOrdinal]
                                                .spatialWorkload.artifact));
          });
          json.attribute("pio_address", session.bridge.pioAddress);
          json.attribute("pio_size", session.bridge.pioSize);
          json.attribute("session_ordinal", indexed.index());
          json.attribute("pio_latency",
                         std::to_string(session.bridge.pioLatencyTicks) + "ps");
          json.attribute("engine_socket", socket);
          json.attributeArray("engine_command", [&] {
            if (facts.engine == Gem5SystemEngine::Rtl)
              return;
            if (indexed.index() != 0)
              return;
            json.value(engine);
            json.value("--artifact-store");
            json.value(kPackageObjectPath);
            json.value("--socket");
            json.value(socket);
            for (std::size_t launchOrdinal = 0;
                 launchOrdinal != facts.spatialLaunches.size();
                 ++launchOrdinal) {
              const Gem5SpatialLaunchProjection &launch =
                  facts.spatialLaunches[launchOrdinal];
              json.value("--expected-launch");
              json.value(spatialLaunchPath(launchOrdinal));
              json.value("--workload");
              json.value(
                  formatArtifactIdentityHex(launch.spatialWorkload.artifact));
              json.value("--runtime-input");
              json.value(launch.spatialRuntimeInput
                             ? formatArtifactIdentityHex(
                                   launch.spatialRuntimeInput->artifact)
                             : "none");
              json.value("--channel-projection");
              json.value(launch.channelProjectionPath);
              json.value("--bridge-ordinal");
              json.value(std::to_string(launch.bridgeSessionOrdinal));
              if (facts.engine == Gem5SystemEngine::Cgra) {
                json.value("--fabric");
                json.value(formatArtifactIdentityHex(launch.fabric.artifact));
                json.value("--spatial-mapping");
                json.value(
                    formatArtifactIdentityHex(launch.spatialMapping.artifact));
              }
            }
            json.value("--dataflow");
            json.value(formatArtifactIdentityHex(facts.dataflow.artifact));
            json.value("--maximum-work");
            json.value(std::to_string(gem5MaximumSpatialWork));
            json.value("--ticks-per-cycle");
            json.value(std::to_string(ticksPerCycle));
            json.value("--maximum-invocations");
            json.value(std::to_string(gem5MaximumDynamicSpatialInvocations));
            json.value("--bridge-count");
            json.value(std::to_string(facts.spatialBridgeSessions.size()));
            if (diagnostics && facts.engine == Gem5SystemEngine::Cgra) {
              json.value("--performance-profile");
              json.value(kCgraEnginePerformanceProfilePath);
            }
          });
          json.attribute("result_path",
                         spatialBridgeResultPath(indexed.index()));
          json.attribute("maximum_message_bytes",
                         session.bridge.maximumMessageBytes);
          json.attribute("maximum_invocations",
                         gem5MaximumDynamicSpatialInvocations);
        });
      }
    });
    json.attribute("maximum_ticks", kMaximumGem5Ticks);
  });
  stream << '\n';
  stream.flush();
  return output;
}

std::vector<std::string> gem5AttemptOutputPaths(const Gem5SystemFacts &facts,
                                                bool diagnostics) {
  std::vector<std::string> outputs{kSystemResultPath.str(),
                                   kMemoryResultPath.str(),
                                   kRootLifecycleResultPath.str()};
  for (std::size_t ordinal = 0; ordinal != facts.spatialBridgeSessions.size();
       ++ordinal)
    outputs.push_back(spatialBridgeResultPath(ordinal));
  if (facts.engine == Gem5SystemEngine::Rtl)
    for (std::size_t ordinal = 0; ordinal != facts.spatialLaunches.size();
         ++ordinal)
      outputs.push_back(mappedRtlLaunchResultPath(ordinal));
  if (diagnostics) {
    outputs.push_back(kGem5PerformanceProfilePath.str());
    if (facts.engine == Gem5SystemEngine::Cgra)
      outputs.push_back(kCgraEnginePerformanceProfilePath.str());
  }
  llvm::sort(outputs);
  return outputs;
}

ExternalToolInvocationImportExpectation makeExpectation(
    const ExternalToolSemanticContract &contract, const Gem5SystemFacts &facts,
    bool diagnostics,
    llvm::ArrayRef<ExternalToolInvocationSemanticInput>
        additionalSemanticInputs = {},
    std::optional<ExternalFileFingerprint> gem5Binary = std::nullopt) {
  ExternalToolInvocationImportExpectation expectation;
  expectation.semanticContract = contract;
  for (const MaterializedBundleFile &file : facts.semanticInputs) {
    if (!file.sourceArtifact)
      continue;
    expectation.semanticInputs.push_back(
        {file.relativePath, *file.sourceArtifact,
         computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
             reinterpret_cast<const std::uint8_t *>(file.contents.data()),
             file.contents.size()))});
  }
  expectation.semanticInputs.insert(expectation.semanticInputs.end(),
                                    additionalSemanticInputs.begin(),
                                    additionalSemanticInputs.end());
  llvm::sort(expectation.semanticInputs, [](const auto &lhs, const auto &rhs) {
    return lhs.relativePath < rhs.relativePath;
  });
  if (gem5Binary)
    expectation.externalInputs.push_back(
        {"gem5_binary", std::move(*gem5Binary)});
  expectation.declaredOutputs = gem5AttemptOutputPaths(facts, diagnostics);
  return expectation;
}

llvm::Expected<ExternalFileFingerprint>
gem5BinaryFingerprint(const FinalizedGem5SimulationBinding &binding) {
  return parseExternalFileFingerprint(
      binding.binding().gem5BuildIdentity().binaryFingerprint);
}

llvm::Expected<eda::open_source::MappedRtlExecutionClosure>
mappedRtlClosure(const EvaluationRequest &request, const Gem5SystemFacts &facts,
                 const Gem5SpatialLaunchProjection &launch,
                 const ExternalToolSemanticContract &contract) {
  if (!launch.spatialRuntimeInput)
    return invalid("gem5 RTL does not admit dynamic Spatial invocation");
  const auto *binding = request.modelBinding()
                            .resolvedModelConfig()
                            .getIf<models::MappedRtlSimulatorBinding>();
  if (!binding)
    return invalid("gem5 RTL Request has no HDL simulator binding");
  if (llvm::Error error = models::validateMappedRtlSimulatorBinding(*binding))
    return std::move(error);
  return eda::open_source::MappedRtlExecutionClosure{
      *binding,
      contract,
      launch.hardwareImplementation,
      facts.deployment,
      launch.spatialWorkload,
      *launch.spatialRuntimeInput};
}

EvaluationModelResult terminalResult(EvaluationEvidenceOutcome outcome) {
  return EvaluationModelResult{{{kExecutionOutput, {}}}, std::move(outcome)};
}

llvm::Expected<EvaluationModelResult>
classifyFailedAttempt(const FailedExternalToolInvocationAttempt &failed) {
  switch (failed.status) {
  case InvocationCompletionStatus::Success:
    return invalid("failed gem5 attempt carries success status");
  case InvocationCompletionStatus::MissingEnvironment:
  case InvocationCompletionStatus::ModuleActivationFailed:
  case InvocationCompletionStatus::VersionMismatch:
    return terminalResult(
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable});
  case InvocationCompletionStatus::BundleContentMismatch:
    return invalid("gem5 invocation bundle changed before execution");
  case InvocationCompletionStatus::ToolExit:
  case InvocationCompletionStatus::MissingOutput:
    return terminalResult(ExecutionFailedEvidence{OutcomeReason::ToolFailure});
  }
  llvm_unreachable("closed invocation status");
}

struct Gem5AttemptResult final {
  std::uint64_t entryTick = 0;
  std::uint64_t exitTick = 0;
  std::string cause;
};

std::uint32_t readBigEndianU32(llvm::StringRef bytes, std::size_t offset) {
  std::uint32_t value = 0;
  for (std::size_t index = 0; index != 4; ++index)
    value = (value << 8) | static_cast<unsigned char>(bytes[offset + index]);
  return value;
}

std::uint64_t readBigEndianU64(llvm::StringRef bytes, std::size_t offset) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index != 8; ++index)
    value = (value << 8) | static_cast<unsigned char>(bytes[offset + index]);
  return value;
}

llvm::Expected<std::vector<sim::SystemRootLifecycleObservation>>
parseRootLifecycleResult(llvm::StringRef bytes, const Gem5SystemFacts &facts) {
  constexpr std::size_t headerBytes = 4;
  constexpr std::size_t recordBytes = 36;
  if (bytes.size() < headerBytes ||
      readBigEndianU32(bytes, 0) != gem5RootLifecycleTraceMagic)
    return invalid("gem5 root lifecycle result has the wrong header");
  if ((bytes.size() - headerBytes) % recordBytes != 0)
    return invalid("gem5 root lifecycle result has a partial record");
  const std::size_t recordCount = (bytes.size() - headerBytes) / recordBytes;
  const std::size_t sessionCount =
      std::max<std::size_t>(facts.spatialBridgeSessions.size(), 1);
  if (recordCount >
      2 * static_cast<std::size_t>(gem5MaximumDynamicSpatialInvocations) *
          sessionCount)
    return invalid("gem5 root lifecycle result exceeds its invocation bound");

  std::vector<sim::SystemRootLifecycleObservation> observations;
  observations.reserve(recordCount);
  for (std::size_t offset = headerBytes; offset != bytes.size();
       offset += recordBytes) {
    const std::uint64_t entity = readBigEndianU64(bytes, offset);
    const std::uint64_t occurrence = readBigEndianU64(bytes, offset + 8);
    const std::uint32_t action = readBigEndianU32(bytes, offset + 16);
    const std::uint64_t tick = readBigEndianU64(bytes, offset + 20);
    const std::uint64_t delta = readBigEndianU64(bytes, offset + 28);
    if (action >
        static_cast<std::uint32_t>(Gem5RootLifecycleAction::Completion))
      return invalid("gem5 root lifecycle result has an unknown action");
    const dataflow::RootThreadLaunchRef root{
        facts.dataflow.artifact, dataflow::RootThreadLaunchId(entity)};
    const dataflow::EventFamilyKey event =
        action == static_cast<std::uint32_t>(Gem5RootLifecycleAction::Start)
            ? dataflow::rootThreadStartEventFamily(root)
            : dataflow::rootThreadCompletionEventFamily(root);
    observations.push_back({event, occurrence, {tick, delta}});
  }
  return observations;
}

llvm::Expected<Gem5AttemptResult> parseAttemptResult(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return invalid("gem5 result is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object || object->size() != 4)
    return invalid("gem5 result does not have the exact result shape");
  const auto schema = object->getString("schema");
  const auto entry = object->getInteger("entry_tick");
  const auto exit = object->getInteger("exit_tick");
  const auto cause = object->getString("cause");
  if (!schema || *schema != "loom.gem5_system_attempt.1" || !entry || !exit ||
      !cause || *entry < 0 || *exit < 0 || *entry > *exit)
    return invalid("gem5 result fields are invalid");
  return Gem5AttemptResult{static_cast<std::uint64_t>(*entry),
                           static_cast<std::uint64_t>(*exit), cause->str()};
}

llvm::Expected<std::uint64_t>
requiredProfileInteger(const llvm::json::Object &object,
                       llvm::StringRef field) {
  const auto value = object.getInteger(field);
  if (!value || *value < 0)
    return invalid("gem5 performance profile field '" + field +
                   "' is not a nonnegative integer");
  return static_cast<std::uint64_t>(*value);
}

llvm::Error assignProfileInteger(const llvm::json::Object &object,
                                 llvm::StringRef field,
                                 std::uint64_t &destination) {
  auto value = requiredProfileInteger(object, field);
  if (!value)
    return value.takeError();
  destination = *value;
  return llvm::Error::success();
}

llvm::Expected<Gem5SystemAttemptProfile>
parseGem5SystemAttemptProfile(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return invalid("gem5 performance profile is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object || object->size() != 14)
    return invalid("gem5 performance profile has the wrong shape");
  const auto schema = object->getString("schema");
  if (!schema || *schema != "loom.gem5_system_performance_profile.4")
    return invalid("gem5 performance profile has the wrong schema");
  Gem5SystemAttemptProfile profile;
  const auto assign = [&](llvm::StringRef field,
                          std::uint64_t &destination) -> llvm::Error {
    return assignProfileInteger(*object, field, destination);
  };
  if (llvm::Error error = assign("configuration_wall_nanoseconds",
                                 profile.configurationWallNanoseconds))
    return std::move(error);
  if (llvm::Error error = assign("engine_startup_wall_nanoseconds",
                                 profile.engineStartupWallNanoseconds))
    return std::move(error);
  if (llvm::Error error = assign("simulation_wall_nanoseconds",
                                 profile.simulationWallNanoseconds))
    return std::move(error);
  if (llvm::Error error = assign("gem5_simulation_cpu_nanoseconds",
                                 profile.gem5SimulationProcessCpuNanoseconds))
    return std::move(error);
  if (llvm::Error error = assign("observation_wall_nanoseconds",
                                 profile.observationWallNanoseconds))
    return std::move(error);
  if (llvm::Error error = assign("observation_cpu_nanoseconds",
                                 profile.observationProcessCpuNanoseconds))
    return std::move(error);
  const llvm::json::Value *engineCpu =
      object->get("engine_process_cpu_nanoseconds");
  if (!engineCpu)
    return invalid("gem5 performance profile field "
                   "'engine_process_cpu_nanoseconds' is missing");
  if (!engineCpu->getAsNull()) {
    auto parsed =
        requiredProfileInteger(*object, "engine_process_cpu_nanoseconds");
    if (!parsed)
      return parsed.takeError();
    profile.engineProcessCpuNanoseconds = *parsed;
  }
  if (llvm::Error error = assign("bridge_callback_cpu_nanoseconds",
                                 profile.bridgeCallbackCpuNanoseconds))
    return std::move(error);
  if (llvm::Error error = assign("bridge_engine_wait_nanoseconds",
                                 profile.bridgeEngineWaitNanoseconds))
    return std::move(error);
  if (llvm::Error error =
          assign("bridge_message_count", profile.bridgeMessageCount))
    return std::move(error);
  if (llvm::Error error = assign("accelerator_invocation_count",
                                 profile.acceleratorInvocationCount))
    return std::move(error);
  if (llvm::Error error =
          assign("bridge_clock_failure_count", profile.bridgeClockFailureCount))
    return std::move(error);
  if (llvm::Error error = assign("bridge_count", profile.bridgeCount))
    return std::move(error);
  return profile;
}

llvm::Expected<Gem5CgraEngineAttemptProfile>
parseCgraEngineAttemptProfile(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return invalid("CGRA engine performance profile is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object || object->size() != 6)
    return invalid("CGRA engine performance profile has the wrong shape");
  const auto schema = object->getString("schema");
  const auto engine = object->getString("engine");
  if (!schema || *schema != "loom.gem5_spatial_engine_performance.1" ||
      !engine || *engine != "cgra")
    return invalid("CGRA engine performance profile identity is invalid");
  Gem5CgraEngineAttemptProfile profile;
  if (llvm::Error error = assignProfileInteger(*object, "invocation_count",
                                               profile.invocationCount))
    return std::move(error);
  if (llvm::Error error = assignProfileInteger(
          *object, "active_wall_nanoseconds", profile.activeWallNanoseconds))
    return std::move(error);
  if (llvm::Error error =
          assignProfileInteger(*object, "active_cpu_nanoseconds",
                               profile.activeProcessCpuNanoseconds))
    return std::move(error);
  if (llvm::Error error = assignProfileInteger(*object, "event_frame_count",
                                               profile.eventFrameCount))
    return std::move(error);
  return profile;
}

llvm::Error validateFreshDiagnosticExecution(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &execution) {
  if (llvm::Error error = validateExternalToolInvocationExecutionObservation(
          prepared, execution))
    return error;
  if (execution.reusePolicy != ExternalToolResultReusePolicy::RequireFresh ||
      execution.cacheAvailability !=
          ExternalToolResultCacheAvailability::Disabled ||
      execution.cacheLookup != ExternalToolResultCacheLookup::NotAttempted ||
      execution.cacheDiscard != ExternalToolResultCacheDiscard::NotAttempted ||
      execution.cachePublication !=
          ExternalToolResultCachePublication::NotAttempted ||
      execution.waitedForCacheKeyLock ||
      (execution.exitCode == 0 && !execution.invokedExternalTool))
    return invalid("gem5 diagnostics require an uncached fresh external "
                   "attempt");
  return llvm::Error::success();
}

struct Gem5SystemDiagnosticSidecar final {
  std::vector<Gem5SpatialInvocationProjection> spatialInvocations;
  Gem5SystemAttemptProfile attemptProfile;
};

llvm::Expected<std::uint64_t> readResultU64(llvm::StringRef bytes,
                                            std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("System memory result is truncated");
  std::uint64_t value = 0;
  for (unsigned index = 0; index != 8; ++index)
    value = (value << 8) | static_cast<std::uint8_t>(bytes[offset + index]);
  offset += 8;
  return value;
}

llvm::Expected<std::vector<std::vector<std::uint8_t>>>
parseMemoryResult(llvm::StringRef bytes, const Gem5SystemFacts &facts) {
  if (bytes.size() < 12 || bytes.take_front(4) != "LGM1")
    return invalid("System memory result has the wrong header");
  std::size_t offset = 4;
  auto count = readResultU64(bytes, offset);
  if (!count)
    return count.takeError();
  const std::size_t expectedCount =
      facts.memoryObservations.size() + (facts.programResult ? 1 : 0);
  if (*count != expectedCount)
    return invalid("System memory result has the wrong observation count");
  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(expectedCount);
  const auto readSnapshot = [&](std::uint64_t expectedAddress,
                                std::uint64_t expectedSize) -> llvm::Error {
    auto address = readResultU64(bytes, offset);
    auto size = readResultU64(bytes, offset);
    if (!address || !size)
      return llvm::joinErrors(address ? llvm::Error::success()
                                      : address.takeError(),
                              size ? llvm::Error::success() : size.takeError());
    if (*address != expectedAddress || *size != expectedSize)
      return invalid("System memory result differs from its exact projection");
    if (*size > bytes.size() - offset)
      return invalid("System memory result payload is truncated");
    result.emplace_back(bytes.bytes_begin() + offset,
                        bytes.bytes_begin() + offset + *size);
    offset += static_cast<std::size_t>(*size);
    return llvm::Error::success();
  };
  if (facts.programResult)
    if (llvm::Error error = readSnapshot(facts.programResult->address,
                                         facts.programResult->size))
      return std::move(error);
  for (const Gem5MemoryObservationProjection &expected :
       facts.memoryObservations) {
    if (llvm::Error error = readSnapshot(expected.address, expected.size))
      return std::move(error);
  }
  if (offset != bytes.size())
    return invalid("System memory result has trailing bytes");
  return result;
}

llvm::Expected<sim::SystemFunctionalObservations>
projectSystemObservations(const Gem5SystemFacts &facts,
                          const sim::ImportedSystemSimulationInputs &inputs,
                          llvm::ArrayRef<std::vector<std::uint8_t>> snapshots) {
  const std::size_t memoryOffset = facts.programResult ? 1 : 0;
  if (snapshots.size() != facts.memoryObservations.size() + memoryOffset)
    return invalid("System memory snapshot count is not total");
  const sim::SystemSimulationRuntimeInput &runtime =
      *inputs.runtimeInput.system();
  sim::SystemFunctionalObservations observations;
  if (facts.programResult) {
    const Gem5ProgramResultProjection &projection = *facts.programResult;
    if (snapshots.front().size() != projection.size ||
        projection.shape.lanesPerToken == 0 ||
        projection.shape.laneBitWidth == 0 ||
        projection.shape.lanesPerToken > std::numeric_limits<unsigned>::max() /
                                             projection.shape.laneBitWidth)
      return invalid("System program result snapshot has the wrong shape");
    const unsigned bitCount = static_cast<unsigned>(
        projection.shape.lanesPerToken * projection.shape.laneBitWidth);
    llvm::APInt bits(bitCount, 0);
    for (std::size_t byte = 0; byte != snapshots.front().size(); ++byte) {
      const std::size_t orderedByte =
          projection.littleEndian ? byte : snapshots.front().size() - 1 - byte;
      for (unsigned bit = 0; bit != 8; ++bit)
        if ((snapshots.front()[byte] & (1U << bit)) != 0)
          bits.setBit(static_cast<unsigned>(orderedByte * 8 + bit));
    }
    auto lanes = sim::unpackDefinedSpatialSimulationToken(
        bits, {projection.shape.lanesPerToken, projection.shape.laneBitWidth});
    if (!lanes)
      return lanes.takeError();
    observations.valueResults.push_back(sim::PublishedValueResult{
        sim::CanonicalValueSequence{1, std::move(*lanes)}});
  }
  observations.memories.reserve(facts.memoryObservations.size());
  for (std::size_t ordinal = 0; ordinal != facts.memoryObservations.size();
       ++ordinal) {
    const Gem5MemoryObservationProjection &projection =
        facts.memoryObservations[ordinal];
    if (projection.objectOrdinal >= runtime.memoryObjects.size())
      return invalid("System memory projection names an absent object");
    const sim::RuntimeMemoryObject &object =
        runtime.memoryObjects[projection.objectOrdinal];
    if (projection.objectByteOffset > object.initialBytes.size() ||
        projection.size !=
            object.initialBytes.size() - projection.objectByteOffset ||
        snapshots[memoryOffset + ordinal].size() != projection.size)
      return invalid("System memory projection no longer matches its baseline");
    std::vector<sim::SemanticMemoryByte> finalBytes;
    finalBytes.reserve(snapshots[memoryOffset + ordinal].size());
    for (std::uint8_t byte : snapshots[memoryOffset + ordinal])
      finalBytes.push_back({sim::SemanticState::Defined, byte});
    if (projection.form == sim::MemoryObservationForm::FullState) {
      observations.memories.push_back(
          sim::FullMemoryObservation{std::move(finalBytes)});
      continue;
    }
    sim::DiffMemoryObservation diff;
    diff.byteCount = projection.size;
    llvm::ArrayRef<sim::SemanticMemoryByte> baseline(object.initialBytes);
    baseline = baseline.drop_front(projection.objectByteOffset);
    std::size_t byte = 0;
    while (byte != finalBytes.size()) {
      if (baseline[byte].state == sim::SemanticState::Defined &&
          baseline[byte].value == finalBytes[byte].value) {
        ++byte;
        continue;
      }
      const std::size_t begin = byte;
      while (byte != finalBytes.size() &&
             (baseline[byte].state != sim::SemanticState::Defined ||
              baseline[byte].value != finalBytes[byte].value))
        ++byte;
      diff.runs.push_back(
          {begin, std::vector<sim::SemanticMemoryByte>(
                      finalBytes.begin() + begin, finalBytes.begin() + byte)});
    }
    observations.memories.push_back(std::move(diff));
  }
  return observations;
}

} // namespace

std::optional<std::uint64_t> integralSpatialReferenceCycleDistance(
    const sim::SpatialProgressObservations &progress) {
  if (!progress.graphRetirementVisible)
    return std::nullopt;
  const sim::SpatialEventCoordinate &launch = progress.launchAccepted;
  const sim::SpatialEventCoordinate &retirement =
      *progress.graphRetirementVisible;
  if (sim::compareSpatialEventCoordinates(retirement, launch) < 0)
    return std::nullopt;

  using u128 = unsigned __int128;
  const u128 retirementScaled =
      static_cast<u128>(retirement.referenceCycle.numerator()) *
      launch.referenceCycle.denominator();
  const u128 launchScaled =
      static_cast<u128>(launch.referenceCycle.numerator()) *
      retirement.referenceCycle.denominator();
  const u128 commonDenominator =
      static_cast<u128>(launch.referenceCycle.denominator()) *
      retirement.referenceCycle.denominator();
  const u128 difference = retirementScaled - launchScaled;
  if (difference % commonDenominator != 0)
    return std::nullopt;
  const u128 quotient = difference / commonDenominator;
  if (quotient > std::numeric_limits<std::uint64_t>::max())
    return std::nullopt;
  return static_cast<std::uint64_t>(quotient);
}

static llvm::Expected<EvaluationModelProviderPreparation>
prepareGem5SystemInvocationImpl(const EvaluationRequest &request,
                                const CaseArtifactResolution &resolution,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs,
                                const ExternalToolPreparationContext &context,
                                bool diagnostics) {
  auto factsOrUnsupported = deriveFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<UnsupportedEvidence>(factsOrUnsupported->get()))
    return EvaluationModelProviderPreparation{*unsupported};
  const Gem5SystemFacts &facts =
      std::get<Gem5SystemFacts>(**factsOrUnsupported);
  if (facts.engine == Gem5SystemEngine::Rtl &&
      llvm::any_of(facts.spatialBridgeSessions, [](const auto &session) {
        return session.launchOrdinals.size() != 1;
      }))
    return EvaluationModelProviderPreparation{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  const ExternalToolProviderDescriptor &gem5ToolProvider = gem5Provider();
  const std::filesystem::path destination(context.bundleDestination);
  const std::filesystem::path probeRoot = destination.parent_path();
  ShellToolBindingProbe gem5Probe(probeRoot.string(),
                                  gem5ToolProvider.versionProbe);
  auto gem5Tool = resolveToolBinding(
      gem5ToolProvider.binding, context.localConfig,
      captureToolEnvironment(gem5ToolProvider.binding), gem5Probe);
  if (!gem5Tool)
    return gem5Tool.takeError();
  auto readiness = verifyReadiness(facts, facts.binding, context.localConfig,
                                   gem5ToolProvider, *gem5Tool);
  if (!readiness)
    return readiness.takeError();
  const std::string gem5Executable = gem5Tool->executable;
  const ResolvedExternalFile gem5ExternalFile{"gem5_binary", "gem5_readiness",
                                              gem5Executable,
                                              readiness->binaryFingerprint};

  const ExternalToolProviderDescriptor &container = polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeRoot.string(),
                                       container.versionProbe);
  auto contract = deriveExternalToolSemanticContract(request);
  if (!contract)
    return contract.takeError();
  auto configuration = readFile(LOOM_GEM5_CONFIG_SCRIPT_PATH);
  if (!configuration)
    return configuration.takeError();
  auto timeoutBudgets = readFile(LOOM_TIMEOUT_BUDGETS_PATH);
  if (!timeoutBudgets)
    return timeoutBudgets.takeError();
  std::vector<MaterializedBundleFile> files = facts.semanticInputs;
  files.push_back({kConfigurationScriptPath.str(), std::move(*configuration),
                   std::nullopt, false});
  files.push_back({kTimeoutBudgetsPath.str(), std::move(*timeoutBudgets),
                   std::nullopt, false});
  auto projection = renderProjection(facts, *readiness, diagnostics);
  if (!projection)
    return projection.takeError();
  files.push_back(
      {kProjectionPath.str(), std::move(*projection), std::nullopt, false});

  if (facts.engine == Gem5SystemEngine::Rtl) {
    auto options = eda::open_source::resolveMappedRtlExecutionAttemptOptions(
        context.localConfig);
    if (!options)
      return options.takeError();
    const ExternalToolProviderDescriptor &verilatorToolProvider =
        verilatorProvider();
    ShellToolBindingProbe verilatorProbe(probeRoot.string(),
                                         verilatorToolProvider.versionProbe);
    auto verilatorTool = resolveToolBinding(
        verilatorToolProvider.binding, context.localConfig,
        captureToolEnvironment(verilatorToolProvider.binding), verilatorProbe);
    if (!verilatorTool)
      return verilatorTool.takeError();
    const std::string verilatorExecutable = verilatorTool->executable;
    auto runtime = resolveInvocationRuntime(
        *verilatorTool, context.localConfig, container.binding,
        captureToolEnvironment(container.binding), containerProbe,
        verilatorToolProvider.runtimeCompatibility,
        [&](const ResolvedToolBinding &resolvedTool,
            const ResolvedToolBinding &resolvedContainer,
            llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
          return probeContainerToolComposition(
              probeRoot.string(), resolvedTool,
              verilatorToolProvider.versionProbe, resolvedContainer, os,
              options->inheritedEnvironment);
        });
    if (!runtime)
      return runtime.takeError();
    auto engineSource = readFile(LOOM_GEM5_RTL_ENGINE_SOURCE_PATH);
    auto bridgeHeader = readFile(LOOM_GEM5_BRIDGE_HEADER_PATH);
    auto channelPlanHeader = readFile(LOOM_GEM5_CHANNEL_PLAN_HEADER_PATH);
    auto invocationWireHeader = readFile(LOOM_GEM5_INVOCATION_WIRE_HEADER_PATH);
    if (!engineSource)
      return engineSource.takeError();
    if (!bridgeHeader)
      return bridgeHeader.takeError();
    if (!channelPlanHeader)
      return channelPlanHeader.takeError();
    if (!invocationWireHeader)
      return invocationWireHeader.takeError();

    std::vector<std::vector<std::string>> commands;
    std::vector<std::string> executables;
    std::vector<std::string> resultPaths;
    std::vector<std::vector<std::string>> engineCommands;
    commands.reserve(facts.spatialLaunches.size() + 1);
    executables.reserve(facts.spatialLaunches.size());
    resultPaths.reserve(facts.spatialLaunches.size());
    engineCommands.reserve(facts.spatialLaunches.size());
    for (const auto indexed : llvm::enumerate(facts.spatialLaunches)) {
      const Gem5SpatialLaunchProjection &launch = indexed.value();
      auto closure = mappedRtlClosure(request, facts, launch, *contract);
      if (!closure)
        return closure.takeError();
      if (verilatorTool->version !=
          closure->simulatorBinding.stableHdlSimulatorBuildIdentity)
        return invalid(
            "resolved Verilator build differs from the gem5 RTL binding");
      const std::string prefix = mappedRtlLaunchPrefix(indexed.index());
      auto projection =
          eda::open_source::deriveMappedRtlExecutionBundleProjection(
              *closure, options->cycleLimit, options->buildJobs, artifacts,
              blobs, prefix);
      if (!projection)
        return projection.takeError();
      if (const auto *unsupported =
              std::get_if<UnsupportedEvidence>(&*projection))
        return EvaluationModelProviderPreparation{*unsupported};
      auto rtl = std::get<eda::open_source::MappedRtlExecutionBundleProjection>(
          std::move(*projection));
      files.push_back(
          {rtl.testbenchPath, std::move(rtl.testbench), std::nullopt, false});
      files.push_back({rtl.bridgedVerilatorDriverPath,
                       std::move(rtl.bridgedVerilatorDriver), std::nullopt,
                       false});
      files.push_back(
          {rtl.bridgeEngineSourcePath, *engineSource, std::nullopt, false});
      const std::filesystem::path engineDirectory =
          std::filesystem::path(rtl.bridgeEngineSourcePath).parent_path();
      files.push_back(
          {(engineDirectory /
            std::filesystem::path(kBridgeHeaderPath.str()).filename())
               .generic_string(),
           *bridgeHeader, std::nullopt, false});
      files.push_back(
          {(engineDirectory /
            std::filesystem::path(kChannelPlanHeaderPath.str()).filename())
               .generic_string(),
           *channelPlanHeader, std::nullopt, false});
      files.push_back(
          {(engineDirectory /
            std::filesystem::path(kInvocationWireHeaderPath.str()).filename())
               .generic_string(),
           *invocationWireHeader, std::nullopt, false});
      files.insert(files.end(),
                   std::make_move_iterator(rtl.semanticInputs.begin()),
                   std::make_move_iterator(rtl.semanticInputs.end()));
      commands.push_back(
          {verilatorExecutable, "-f", rtl.bridgedVerilatorDriverPath});
      executables.push_back(rtl.simulatorExecutablePath);
      resultPaths.push_back(rtl.resultPath);
      std::vector<std::string> engineCommand{
          rtl.simulatorExecutablePath,
          "--socket",
          spatialBridgeSocketPath(launch.bridgeSessionOrdinal),
          "--expected-launch",
          spatialLaunchPath(indexed.index()),
          "--mapped-result",
          rtl.resultPath,
          "--channel-plan",
          launch.channelEnginePlanPath,
          "--ticks-per-cycle",
          std::to_string(facts.processors.front().parameters.clockPeriodTicks)};
      if (indexed.index() != 0)
        engineCommand.push_back("--peer");
      engineCommands.push_back(std::move(engineCommand));
    }

    std::string peerManifest = "loom.gem5_rtl_peers 1.0\n";
    for (std::size_t ordinal = 1; ordinal != engineCommands.size(); ++ordinal) {
      for (std::size_t argument = 0; argument != engineCommands[ordinal].size();
           ++argument) {
        if (argument != 0)
          peerManifest.push_back('\t');
        peerManifest += engineCommands[ordinal][argument];
      }
      peerManifest.push_back('\n');
    }
    peerManifest += "end\n";
    files.push_back({kRtlPeerManifestPath.str(), std::move(peerManifest),
                     std::nullopt, false});
    std::vector<std::string> primaryCommand = std::move(engineCommands.front());
    primaryCommand.insert(primaryCommand.end(),
                          {"--peer-manifest", kRtlPeerManifestPath.str()});
    for (std::size_t ordinal = 1; ordinal != executables.size(); ++ordinal)
      primaryCommand.insert(primaryCommand.end(),
                            {"--peer-executable", executables[ordinal]});
    primaryCommand.insert(
        primaryCommand.end(),
        {"--gem5", gem5Executable, "--gem5-output", "outputs/gem5",
         "--gem5-config", kConfigurationScriptPath.str(), "--projection",
         kProjectionPath.str(), "--system-result", kSystemResultPath.str()});
    if (diagnostics)
      primaryCommand.insert(
          primaryCommand.end(),
          {"--gem5-performance-profile", kGem5PerformanceProfilePath.str()});
    commands.push_back(std::move(primaryCommand));
    std::vector<std::string> declaredOutputs =
        gem5AttemptOutputPaths(facts, diagnostics);
    ExternalToolInvocationBundleSpec specification{
        std::move(*contract),
        std::move(*verilatorTool),
        verilatorToolProvider.versionProbe,
        std::move(*runtime),
        container.versionProbe,
        std::move(commands),
        std::move(options->inheritedEnvironment),
        std::move(declaredOutputs),
        std::move(files),
        {gem5ExternalFile},
        {},
        std::move(executables)};
    specification.diagnosticCommandOrdinals = {specification.commands.size() -
                                               1};
    llvm::sort(specification.files, [](const auto &lhs, const auto &rhs) {
      return lhs.relativePath < rhs.relativePath;
    });
    auto prepared = finalizeExternalToolInvocationBundle(
        context.bundleDestination, specification);
    if (!prepared)
      return prepared.takeError();
    return EvaluationModelProviderPreparation{std::move(*prepared)};
  }

  const std::vector<std::string> inherited =
      inheritedEnvironment(context.localConfig, gem5ToolProvider);
  auto runtime = resolveInvocationRuntime(
      *gem5Tool, context.localConfig, container.binding,
      captureToolEnvironment(container.binding), containerProbe,
      gem5ToolProvider.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &resolvedContainer,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        return probeContainerToolComposition(probeRoot.string(), resolvedTool,
                                             gem5ToolProvider.versionProbe,
                                             resolvedContainer, os, inherited);
      });
  if (!runtime)
    return runtime.takeError();
  const llvm::StringRef engineSource = facts.engine == Gem5SystemEngine::Dfg
                                           ? LOOM_GEM5_DFG_ENGINE_PATH
                                           : LOOM_GEM5_CGRA_ENGINE_PATH;
  auto engine = readFile(engineSource);
  if (!engine)
    return engine.takeError();
  files.push_back({facts.engine == Gem5SystemEngine::Dfg
                       ? kDfgEnginePath.str()
                       : kCgraEnginePath.str(),
                   std::move(*engine), std::nullopt, true});

  ExternalToolInvocationBundleSpec specification{
      std::move(*contract),
      std::move(*gem5Tool),
      gem5ToolProvider.versionProbe,
      std::move(*runtime),
      container.versionProbe,
      {},
      inherited,
      gem5AttemptOutputPaths(facts, diagnostics),
      std::move(files),
      {gem5ExternalFile},
      {},
      {}};
  specification.commands = {{specification.tool.executable, "-d",
                             "outputs/gem5", kConfigurationScriptPath.str(),
                             "--projection", kProjectionPath.str(), "--result",
                             kSystemResultPath.str()}};
  if (diagnostics)
    specification.commands.front().insert(
        specification.commands.front().end(),
        {"--performance-profile", kGem5PerformanceProfilePath.str()});
  auto prepared = finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
  if (!prepared)
    return prepared.takeError();
  return EvaluationModelProviderPreparation{std::move(*prepared)};
}

llvm::Expected<EvaluationModelProviderPreparation> prepareGem5SystemInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  return prepareGem5SystemInvocationImpl(request, resolution, artifacts, blobs,
                                         context, false);
}

llvm::Expected<EvaluationModelProviderPreparation>
prepareGem5SystemDiagnosticInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  ArtifactImportCacheScope cacheScope(artifacts, &blobs);
  RequestVerifier verifier(resolution, artifacts, blobs);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  return prepareGem5SystemInvocationImpl(request, resolution, artifacts, blobs,
                                         context, true);
}

static llvm::Expected<EvaluationModelResult> importGem5SystemInvocationImpl(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    Gem5SystemDiagnosticSidecar *diagnostics) {
  std::vector<Gem5SpatialInvocationProjection> spatialInvocations;
  auto factsOrUnsupported = deriveFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (std::holds_alternative<UnsupportedEvidence>(**factsOrUnsupported))
    return invalid("prepared gem5 invocation is outside provider capability");
  const Gem5SystemFacts &facts =
      std::get<Gem5SystemFacts>(**factsOrUnsupported);
  auto contract = deriveExternalToolSemanticContract(request);
  if (!contract)
    return contract.takeError();
  auto fingerprint = gem5BinaryFingerprint(facts.binding);
  if (!fingerprint)
    return fingerprint.takeError();
  std::vector<ExternalToolInvocationSemanticInput> mappedRtlInputs;
  std::vector<eda::open_source::MappedRtlExecutionClosure> rtlClosures;
  if (facts.engine == Gem5SystemEngine::Rtl) {
    rtlClosures.reserve(facts.spatialLaunches.size());
    for (const auto indexed : llvm::enumerate(facts.spatialLaunches)) {
      auto closure =
          mappedRtlClosure(request, facts, indexed.value(), *contract);
      if (!closure)
        return closure.takeError();
      auto expectation =
          eda::open_source::deriveMappedRtlExecutionImportExpectation(
              *closure, artifacts, blobs,
              mappedRtlLaunchPrefix(indexed.index()));
      if (!expectation)
        return expectation.takeError();
      mappedRtlInputs.insert(
          mappedRtlInputs.end(),
          std::make_move_iterator(expectation->semanticInputs.begin()),
          std::make_move_iterator(expectation->semanticInputs.end()));
      rtlClosures.push_back(std::move(*closure));
    }
  }
  auto attempt = importExternalToolInvocationAttempt(
      prepared, makeExpectation(*contract, facts, diagnostics != nullptr,
                                mappedRtlInputs, *fingerprint));
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (const auto *failed =
          std::get_if<FailedExternalToolInvocationAttempt>(&*attempt))
    return classifyFailedAttempt(*failed);
  ImportedExternalToolInvocationBundle imported =
      std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
  auto systemText =
      readExternalToolInvocationDeclaredOutput(imported, kSystemResultPath);
  if (!systemText)
    return systemText.takeError();
  auto systemResult = parseAttemptResult(*systemText);
  if (!systemResult)
    return systemResult.takeError();
  std::optional<Gem5SystemAttemptProfile> attemptProfile;
  if (diagnostics) {
    auto profileText = readExternalToolInvocationDeclaredOutput(
        imported, kGem5PerformanceProfilePath);
    if (!profileText)
      return profileText.takeError();
    auto parsed = parseGem5SystemAttemptProfile(*profileText);
    if (!parsed)
      return parsed.takeError();
    attemptProfile = std::move(*parsed);
    if (facts.engine == Gem5SystemEngine::Cgra) {
      auto engineText = readExternalToolInvocationDeclaredOutput(
          imported, kCgraEnginePerformanceProfilePath);
      if (!engineText)
        return engineText.takeError();
      auto engine = parseCgraEngineAttemptProfile(*engineText);
      if (!engine)
        return engine.takeError();
      attemptProfile->cgraEngine = std::move(*engine);
    }
    diagnostics->attemptProfile = *attemptProfile;
  }
  if (!llvm::StringRef(systemResult->cause).contains("m5_exit"))
    return terminalResult(
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
  if (attemptProfile) {
    if (attemptProfile->bridgeCount != facts.spatialBridgeSessions.size())
      return invalid("gem5 performance profile bridge count differs from "
                     "the exact System projection");
    if (attemptProfile->bridgeCount == 0 ||
        attemptProfile->acceleratorInvocationCount == 0 ||
        attemptProfile->bridgeMessageCount == 0)
      return invalid("completed gem5 performance profile has no bridge "
                     "activity");
    if (attemptProfile->simulationWallNanoseconds == 0 ||
        attemptProfile->bridgeClockFailureCount != 0)
      return invalid("completed gem5 performance profile has inconsistent "
                     "active timing");
  }
  for (const auto indexed : llvm::enumerate(facts.spatialBridgeSessions)) {
    const Gem5SpatialBridgeSession &session = indexed.value();
    auto bridgeText = readExternalToolInvocationDeclaredOutput(
        imported, spatialBridgeResultPath(indexed.index()));
    if (!bridgeText)
      return bridgeText.takeError();
    std::vector<std::uint8_t> bridgeBytes(bridgeText->begin(),
                                          bridgeText->end());
    Gem5BridgeResultCollection bridgeResults;
    std::string bridgeDiagnostic;
    if (!decodeGem5BridgeResultCollection(bridgeBytes, bridgeResults,
                                          bridgeDiagnostic))
      return invalid("bridge result is invalid: " + bridgeDiagnostic);
    if (session.launchOrdinals.empty() || bridgeResults.results.empty() ||
        bridgeResults.results.size() > gem5MaximumDynamicSpatialInvocations)
      return invalid("bridge result count is outside its session limits");
    std::vector<bool> observedSessionEntries(session.launchOrdinals.size());
    std::uint64_t previousCompletionTick = systemResult->entryTick;
    for (const auto resultIndexed : llvm::enumerate(bridgeResults.results)) {
      const Gem5BridgeResult &bridgeResult = resultIndexed.value();
      if (bridgeResult.status > 1 ||
          bridgeResult.sequence != resultIndexed.index() ||
          bridgeResult.completionTick < previousCompletionTick ||
          bridgeResult.completionTick > systemResult->exitTick)
        return invalid("bridge completion is inconsistent with gem5 time");
      previousCompletionTick = bridgeResult.completionTick;
      SpatialInvocationResultWire invocationResult;
      std::string invocationDiagnostic;
      if (!decodeSpatialInvocationResultWire(
              bridgeResult.result, invocationResult, invocationDiagnostic))
        return invalid("bridge invocation result is invalid: " +
                       invocationDiagnostic);
      if (invocationResult.sessionEntryOrdinal >= session.launchOrdinals.size())
        return invalid("bridge result names an absent session entry");
      const std::size_t sessionEntryOrdinal =
          static_cast<std::size_t>(invocationResult.sessionEntryOrdinal);
      observedSessionEntries[sessionEntryOrdinal] = true;
      const std::size_t launchOrdinal =
          session.launchOrdinals[sessionEntryOrdinal];
      if (launchOrdinal >= facts.spatialLaunches.size())
        return invalid("bridge session names an absent Spatial launch");
      const Gem5SpatialLaunchProjection &launch =
          facts.spatialLaunches[launchOrdinal];
      if (launch.bridgeSessionOrdinal != indexed.index())
        return invalid("Spatial launch and bridge session disagree");
      std::optional<sim::ImportedSpatialSimulationWorkload> spatialWorkload;
      if (facts.engine != Gem5SystemEngine::Rtl) {
        auto loaded = sim::importSpatialSimulationWorkload(
            launch.spatialWorkload, artifacts);
        if (!loaded)
          return loaded.takeError();
        spatialWorkload.emplace(std::move(*loaded));
      } else {
        if (session.launchOrdinals.size() != 1 ||
            launchOrdinal >= rtlClosures.size())
          return invalid("gem5 RTL import lost its exact mapped RTL closure");
      }
      std::optional<sim::CanonicalSimulationRuntimeInput> staticRuntime;
      if (spatialWorkload && launch.spatialRuntimeInput) {
        auto loaded = sim::importSpatialSimulationRuntimeInput(
            *launch.spatialRuntimeInput, *spatialWorkload, artifacts);
        if (!loaded)
          return loaded.takeError();
        staticRuntime.emplace(std::move(*loaded));
      }
      std::optional<sim::SpatialEngineBoundaryResult> spatialResult;
      if (facts.engine == Gem5SystemEngine::Rtl) {
        auto mappedText = readExternalToolInvocationDeclaredOutput(
            imported, mappedRtlLaunchResultPath(launchOrdinal));
        if (!mappedText)
          return mappedText.takeError();
        const llvm::ArrayRef<std::uint8_t> mappedBytes(
            reinterpret_cast<const std::uint8_t *>(mappedText->data()),
            mappedText->size());
        if (!invocationResult.invocation.empty() ||
            invocationResult.runtimeInput ||
            mappedBytes != llvm::ArrayRef<std::uint8_t>(
                               invocationResult.spatialBoundaryResult))
          return invalid("bridge payload differs from the mapped RTL result");
        auto mappedResult =
            eda::open_source::parseMappedRtlSimulationResult(*mappedText);
        if (!mappedResult)
          return mappedResult.takeError();
        if (mappedResult->terminal ==
            eda::open_source::MappedRtlTerminalStatus::StoppedByLimit) {
          if (bridgeResult.status != 1)
            return invalid("bridge status disagrees with the RTL terminal");
          if (diagnostics)
            diagnostics->spatialInvocations = spatialInvocations;
          return terminalResult(
              CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
        }
        if (bridgeResult.status != 0)
          return invalid("bridge status disagrees with the RTL terminal");
        auto boundary =
            eda::open_source::projectMappedRtlSpatialEngineBoundaryResult(
                rtlClosures[launchOrdinal], *mappedResult, artifacts, blobs);
        if (!boundary)
          return boundary.takeError();
        spatialResult = std::move(*boundary);
      } else {
        llvm::Expected<sim::SpatialEngineBoundaryResult> boundary =
            [&]() -> llvm::Expected<sim::SpatialEngineBoundaryResult> {
          if (invocationResult.invocation.empty()) {
            if (!staticRuntime || invocationResult.runtimeInput)
              return invalid(
                  "static bridge result has inconsistent runtime input");
            return sim::decodeSpatialEngineBoundaryResult(
                invocationResult.spatialBoundaryResult, *spatialWorkload,
                *staticRuntime);
          }
          if (staticRuntime)
            return invalid(
                "dynamic bridge result has a competing static runtime input");
          if (!invocationResult.runtimeInput)
            return invalid(
                "dynamic bridge result omits its effective runtime input");
          SpatialInvocationWire wire;
          std::string diagnostic;
          if (!decodeSpatialInvocationWire(invocationResult.invocation, wire,
                                           diagnostic))
            return invalid(diagnostic);
          auto runtimeIdentity = ArtifactIdentity::fromBytes(
              invocationResult.runtimeInput->identity);
          if (!runtimeIdentity)
            return runtimeIdentity.takeError();
          auto view = spatialWorkload->dataflow.view();
          if (!view)
            return view.takeError();
          auto runtime = sim::importSimulationRuntimeInput(
              invocationResult.runtimeInput->canonicalBytes,
              spatialWorkload->workload, *view, *runtimeIdentity);
          if (!runtime)
            return runtime.takeError();
          if (llvm::Error error =
                  sim::validateEffectiveSpatialInvocationRuntimeInput(
                      *spatialWorkload, wire, *runtime))
            return std::move(error);
          auto decoded = sim::decodeSpatialEngineBoundaryResult(
              invocationResult.spatialBoundaryResult, *spatialWorkload,
              *runtime);
          if (!decoded)
            return decoded.takeError();
          auto writes = sim::projectSpatialInvocationResultWrites(
              wire, *spatialWorkload, decoded->functionalObservations);
          if (!writes)
            return writes.takeError();
          return decoded;
        }();
        if (!boundary)
          return boundary.takeError();
        const std::uint32_t expectedStatus =
            std::holds_alternative<sim::RetiredExecution>(boundary->terminal)
                ? 0U
                : 1U;
        if (bridgeResult.status != expectedStatus)
          return invalid("bridge status disagrees with the Spatial terminal");
        spatialResult = std::move(*boundary);
      }
      if (!spatialResult || !std::holds_alternative<sim::RetiredExecution>(
                                spatialResult->terminal)) {
        if (diagnostics)
          diagnostics->spatialInvocations = spatialInvocations;
        return terminalResult(
            CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
      }
      if (diagnostics) {
        if (!spatialResult->progressObservations.graphRetirementVisible)
          return invalid("retired bridge invocation has no graph-retirement "
                         "coordinate");
        const std::optional<std::uint64_t> acceleratorReferenceCycles =
            integralSpatialReferenceCycleDistance(
                spatialResult->progressObservations);
        spatialInvocations.push_back(
            {indexed.index(), bridgeResult.sequence, sessionEntryOrdinal,
             launchOrdinal, bridgeResult.completionTick,
             spatialResult->progressObservations, acceleratorReferenceCycles});
      }
    }
    if (llvm::is_contained(observedSessionEntries, false))
      return invalid("bridge results omit a declared session entry");
  }

  if (diagnostics) {
    if (!attemptProfile)
      return invalid("gem5 diagnostic import lost its attempt profile");
    const std::uint64_t invocationCount = spatialInvocations.size();
    if (attemptProfile->acceleratorInvocationCount != invocationCount)
      return invalid("gem5 performance profile invocation count differs from "
                     "the strict bridge result import");
    if (attemptProfile->cgraEngine &&
        attemptProfile->cgraEngine->invocationCount != invocationCount)
      return invalid("CGRA engine profile invocation count differs from the "
                     "strict bridge result import");
    diagnostics->spatialInvocations = spatialInvocations;
    diagnostics->attemptProfile = std::move(*attemptProfile);
  }

  auto memoryText =
      readExternalToolInvocationDeclaredOutput(imported, kMemoryResultPath);
  if (!memoryText)
    return memoryText.takeError();
  auto snapshots = parseMemoryResult(*memoryText, facts);
  if (!snapshots)
    return snapshots.takeError();
  if (!request.workload() || !request.runtimeInput())
    return invalid("System Request lost its workload/runtime pair");
  auto systemInputs = importCachedSystemInputs(
      *request.workload(), *request.runtimeInput(), artifacts, blobs);
  if (!systemInputs)
    return systemInputs.takeError();
  auto functional =
      projectSystemObservations(facts, **systemInputs, *snapshots);
  if (!functional)
    return functional.takeError();
  auto lifecycleText = readExternalToolInvocationDeclaredOutput(
      imported, kRootLifecycleResultPath);
  if (!lifecycleText)
    return lifecycleText.takeError();
  auto rootLifecycle = parseRootLifecycleResult(*lifecycleText, facts);
  if (!rootLifecycle)
    return rootLifecycle.takeError();

  sim::SystemSimulationExecution execution{
      evaluationRequestReference(request),
      sim::RetiredExecution{},
      std::move(*functional),
      {{systemResult->entryTick, 0},
       sim::SystemEventCoordinate{systemResult->exitTick, 0},
       {systemResult->exitTick, 0},
       std::move(*rootLifecycle)},
      {}};
  auto finalized =
      sim::finalizeSimulationExecution(execution, resolution, artifacts, blobs);
  if (!finalized)
    return finalized.takeError();
  auto executionReference =
      sim::publishSimulationExecution(*finalized, artifacts);
  if (!executionReference)
    return executionReference.takeError();

  const std::uint64_t duration =
      systemResult->exitTick - systemResult->entryTick;
  if (duration >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return terminalResult(
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure});
  std::vector<MetricResult> metrics;
  metrics.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::Runtime)
      return invalid("gem5 System Request contains an unsupported metric");
    auto runtime = DecimalValue::get(static_cast<std::int64_t>(duration), -12);
    if (!runtime)
      return runtime.takeError();
    metrics.push_back({UncertaintyKind::ExactWithinModel,
                       PointObservation{MetricValue(std::move(*runtime))},
                       {}});
  }
  return EvaluationModelResult{
      {{kExecutionOutput, {std::move(*executionReference)}}},
      CompletedEvidence{std::move(metrics), {}}};
}

llvm::Expected<EvaluationModelResult> importGem5SystemInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return importGem5SystemInvocationImpl(request, resolution, prepared,
                                        artifacts, blobs, nullptr);
}

llvm::Expected<Gem5SystemDiagnosticEvaluation>
importGem5SystemDiagnosticInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ArtifactImportCacheScope cacheScope(artifacts, &blobs);
  RequestVerifier verifier(resolution, artifacts, blobs);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  if (llvm::Error error = validateFreshDiagnosticExecution(prepared, execution))
    return std::move(error);
  const auto finalize = [&](EvaluationModelResult result)
      -> llvm::Expected<Gem5SystemDiagnosticEvaluation> {
    auto evidence = EvaluationEvidence::get(
        request, std::move(result.outputBindings), std::move(result.outcome),
        resolution, artifacts, blobs);
    if (!evidence)
      return evidence.takeError();
    return Gem5SystemDiagnosticEvaluation{std::move(*evidence), {}, {}};
  };
  if (execution.exitCode == externalToolExecutionStoppedExitCode)
    return finalize(terminalResult(
        CancelledOrTimeoutEvidence{OutcomeReason::ExternalCancellation}));
  if (execution.exitCode != 0)
    return finalize(
        terminalResult(ExecutionFailedEvidence{OutcomeReason::ToolFailure}));
  Gem5SystemDiagnosticSidecar diagnostics;
  auto result = importGem5SystemInvocationImpl(request, resolution, prepared,
                                               artifacts, blobs, &diagnostics);
  if (!result)
    return result.takeError();
  auto evidence = EvaluationEvidence::get(
      request, std::move(result->outputBindings), std::move(result->outcome),
      resolution, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  return Gem5SystemDiagnosticEvaluation{
      std::move(*evidence), std::move(diagnostics.spatialInvocations),
      std::move(diagnostics.attemptProfile)};
}

} // namespace loom::runtime
