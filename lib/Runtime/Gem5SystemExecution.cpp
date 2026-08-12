#include "Runtime/Gem5SystemExecution.h"

#include "Gem5SystemExecutionInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
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
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Runtime/Gem5SimulationBinding.h"
#include "Runtime/Gem5SpatialChannel.h"
#include "Runtime/Gem5SpatialChannelPlan.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

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
  for (const Gem5SpatialLaunchProjection &launch : facts.spatialLaunches)
    if (launch.launchPayload.size() > launch.bridge.maximumMessageBytes)
      return invalid("Deployment launch image exceeds a bridge message limit");
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

std::string renderProjection(const Gem5SystemFacts &facts,
                             const ReadinessIdentity &readiness) {
  std::string output;
  llvm::raw_string_ostream stream(output);
  llvm::json::OStream json(stream, 0);
  const std::uint64_t ticksPerCycle =
      facts.processors.front().parameters.clockPeriodTicks;
  const std::string engine = facts.engine == Gem5SystemEngine::Dfg
                                 ? kDfgEnginePath.str()
                                 : kCgraEnginePath.str();
  json.object([&] {
    json.attribute("schema", "loom.gem5_system_projection.3");
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
      for (const Gem5ProcessorProjection &processor : facts.processors) {
        json.object([&] {
          json.attribute("cpu_id", processor.parameters.cpuId);
          json.attribute("model",
                         processor.model == Gem5ProcessorModelKind::TimingSimple
                             ? "timing_simple"
                             : "o3");
          json.attribute("num_threads", processor.hardwareThreadCount);
          json.attributeArray("execution_units", [&] {
            for (const fabric::ExecutionUnitRecord &unit :
                 processor.executionUnits)
              json.object([&] {
                json.attribute("operation_class",
                               static_cast<std::uint32_t>(unit.operationClass));
                json.attribute("count", unit.count);
                json.attribute("latency_cycles", unit.latencyCycles);
                json.attribute("initiation_interval", unit.initiationInterval);
              });
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
      for (const auto indexed : llvm::enumerate(facts.spatialLaunches)) {
        const Gem5SpatialLaunchProjection &launch = indexed.value();
        const std::string socket = spatialBridgeSocketPath(indexed.index());
        json.object([&] {
          json.attribute("pio_address", launch.bridge.pioAddress);
          json.attribute("pio_size", launch.bridge.pioSize);
          json.attribute("pio_latency",
                         std::to_string(launch.bridge.pioLatencyTicks) + "ps");
          json.attribute("engine_socket", socket);
          json.attributeArray("engine_command", [&] {
            if (facts.engine == Gem5SystemEngine::Rtl)
              return;
            json.value(engine);
            json.value("--artifact-store");
            json.value(kPackageObjectPath);
            json.value("--socket");
            json.value(socket);
            json.value("--expected-launch");
            json.value(spatialLaunchPath(indexed.index()));
            json.value("--workload");
            json.value(
                formatArtifactIdentityHex(launch.spatialWorkload.artifact));
            json.value("--runtime-input");
            json.value(
                formatArtifactIdentityHex(launch.spatialRuntimeInput.artifact));
            json.value("--channel-projection");
            json.value(launch.channelProjectionPath);
            json.value("--dataflow");
            json.value(formatArtifactIdentityHex(facts.dataflow.artifact));
            json.value("--maximum-work");
            json.value(std::to_string(kMaximumSpatialWork));
            json.value("--ticks-per-cycle");
            json.value(std::to_string(ticksPerCycle));
            if (facts.engine == Gem5SystemEngine::Cgra) {
              json.value("--fabric");
              json.value(formatArtifactIdentityHex(launch.fabric.artifact));
              json.value("--spatial-mapping");
              json.value(
                  formatArtifactIdentityHex(launch.spatialMapping.artifact));
            }
          });
          json.attribute("result_path",
                         spatialBridgeResultPath(indexed.index()));
          json.attribute("maximum_message_bytes",
                         launch.bridge.maximumMessageBytes);
        });
      }
    });
    json.attribute("maximum_ticks", kMaximumGem5Ticks);
  });
  stream << '\n';
  stream.flush();
  return output;
}

ExternalToolInvocationImportExpectation makeExpectation(
    const ExternalToolSemanticContract &contract, const Gem5SystemFacts &facts,
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
  expectation.declaredOutputs = {kSystemResultPath.str(),
                                 kMemoryResultPath.str()};
  for (std::size_t ordinal = 0; ordinal != facts.spatialLaunches.size();
       ++ordinal)
    expectation.declaredOutputs.push_back(spatialBridgeResultPath(ordinal));
  if (facts.engine == Gem5SystemEngine::Rtl)
    for (std::size_t ordinal = 0; ordinal != facts.spatialLaunches.size();
         ++ordinal)
      expectation.declaredOutputs.push_back(mappedRtlLaunchResultPath(ordinal));
  llvm::sort(expectation.declaredOutputs);
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
      launch.spatialRuntimeInput};
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
  if (*count != facts.memoryObservations.size())
    return invalid("System memory result has the wrong observation count");
  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(facts.memoryObservations.size());
  for (const Gem5MemoryObservationProjection &expected :
       facts.memoryObservations) {
    auto address = readResultU64(bytes, offset);
    auto size = readResultU64(bytes, offset);
    if (!address || !size)
      return llvm::joinErrors(address ? llvm::Error::success()
                                      : address.takeError(),
                              size ? llvm::Error::success() : size.takeError());
    if (*address != expected.address || *size != expected.size)
      return invalid("System memory result differs from its exact projection");
    if (*size > bytes.size() - offset)
      return invalid("System memory result payload is truncated");
    result.emplace_back(bytes.bytes_begin() + offset,
                        bytes.bytes_begin() + offset + *size);
    offset += static_cast<std::size_t>(*size);
  }
  if (offset != bytes.size())
    return invalid("System memory result has trailing bytes");
  return result;
}

llvm::Expected<sim::SystemFunctionalObservations>
projectSystemObservations(const Gem5SystemFacts &facts,
                          const sim::ImportedSystemSimulationInputs &inputs,
                          llvm::ArrayRef<std::vector<std::uint8_t>> snapshots) {
  if (snapshots.size() != facts.memoryObservations.size())
    return invalid("System memory snapshot count is not total");
  const sim::SystemSimulationRuntimeInput &runtime =
      *inputs.runtimeInput.system();
  sim::SystemFunctionalObservations observations;
  observations.memories.reserve(snapshots.size());
  for (std::size_t ordinal = 0; ordinal != snapshots.size(); ++ordinal) {
    const Gem5MemoryObservationProjection &projection =
        facts.memoryObservations[ordinal];
    if (projection.objectOrdinal >= runtime.memoryObjects.size())
      return invalid("System memory projection names an absent object");
    const sim::RuntimeMemoryObject &object =
        runtime.memoryObjects[projection.objectOrdinal];
    if (projection.objectByteOffset > object.initialBytes.size() ||
        projection.size !=
            object.initialBytes.size() - projection.objectByteOffset ||
        snapshots[ordinal].size() != projection.size)
      return invalid("System memory projection no longer matches its baseline");
    std::vector<sim::SemanticMemoryByte> finalBytes;
    finalBytes.reserve(snapshots[ordinal].size());
    for (std::uint8_t byte : snapshots[ordinal])
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

llvm::Expected<EvaluationModelProviderPreparation> prepareGem5SystemInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ExternalToolPreparationContext &context) {
  auto factsOrUnsupported = deriveFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<UnsupportedEvidence>(&*factsOrUnsupported))
    return EvaluationModelProviderPreparation{*unsupported};
  Gem5SystemFacts facts =
      std::get<Gem5SystemFacts>(std::move(*factsOrUnsupported));

  auto subjects = systemSubjects(request);
  if (!subjects)
    return subjects.takeError();
  auto binding = importGem5SimulationBinding(subjects->second, artifacts);
  if (!binding)
    return binding.takeError();
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
  auto readiness = verifyReadiness(facts, *binding, context.localConfig,
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
  std::vector<MaterializedBundleFile> files = std::move(facts.semanticInputs);
  files.push_back({kConfigurationScriptPath.str(), std::move(*configuration),
                   std::nullopt, false});
  files.push_back({kProjectionPath.str(), renderProjection(facts, *readiness),
                   std::nullopt, false});

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
    if (!engineSource)
      return engineSource.takeError();
    if (!bridgeHeader)
      return bridgeHeader.takeError();
    if (!channelPlanHeader)
      return channelPlanHeader.takeError();

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
          spatialBridgeSocketPath(indexed.index()),
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
    commands.push_back(std::move(primaryCommand));
    std::vector<std::string> declaredOutputs{kSystemResultPath.str(),
                                             kMemoryResultPath.str()};
    for (std::size_t ordinal = 0; ordinal != facts.spatialLaunches.size();
         ++ordinal) {
      declaredOutputs.push_back(spatialBridgeResultPath(ordinal));
      declaredOutputs.push_back(resultPaths[ordinal]);
    }
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
    llvm::sort(specification.declaredOutputs);
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
      {kSystemResultPath.str(), kMemoryResultPath.str()},
      std::move(files),
      {gem5ExternalFile},
      {},
      {}};
  for (std::size_t ordinal = 0; ordinal != facts.spatialLaunches.size();
       ++ordinal)
    specification.declaredOutputs.push_back(spatialBridgeResultPath(ordinal));
  llvm::sort(specification.declaredOutputs);
  specification.commands = {{specification.tool.executable, "-d",
                             "outputs/gem5", kConfigurationScriptPath.str(),
                             "--projection", kProjectionPath.str(), "--result",
                             kSystemResultPath.str()}};
  auto prepared = finalizeExternalToolInvocationBundle(
      context.bundleDestination, specification);
  if (!prepared)
    return prepared.takeError();
  return EvaluationModelProviderPreparation{std::move(*prepared)};
}

llvm::Expected<EvaluationModelResult> importGem5SystemInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto factsOrUnsupported = deriveFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (std::holds_alternative<UnsupportedEvidence>(*factsOrUnsupported))
    return invalid("prepared gem5 invocation is outside provider capability");
  Gem5SystemFacts facts =
      std::get<Gem5SystemFacts>(std::move(*factsOrUnsupported));
  auto contract = deriveExternalToolSemanticContract(request);
  if (!contract)
    return contract.takeError();
  auto subjects = systemSubjects(request);
  if (!subjects)
    return subjects.takeError();
  auto binding = importGem5SimulationBinding(subjects->second, artifacts);
  if (!binding)
    return binding.takeError();
  auto fingerprint = gem5BinaryFingerprint(*binding);
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
      prepared,
      makeExpectation(*contract, facts, mappedRtlInputs, *fingerprint));
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
  if (!llvm::StringRef(systemResult->cause).contains("m5_exit"))
    return terminalResult(
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
  for (const auto indexed : llvm::enumerate(facts.spatialLaunches)) {
    const Gem5SpatialLaunchProjection &launch = indexed.value();
    auto bridgeText = readExternalToolInvocationDeclaredOutput(
        imported, spatialBridgeResultPath(indexed.index()));
    if (!bridgeText)
      return bridgeText.takeError();
    std::vector<std::uint8_t> bridgeBytes(bridgeText->begin(),
                                          bridgeText->end());
    Gem5BridgeResult bridgeResult;
    std::string bridgeDiagnostic;
    if (!decodeGem5BridgeResult(bridgeBytes, bridgeResult, bridgeDiagnostic))
      return invalid("bridge result is invalid: " + bridgeDiagnostic);
    if (bridgeResult.status > 1 || bridgeResult.sequence != 0 ||
        bridgeResult.completionTick < systemResult->entryTick ||
        bridgeResult.completionTick > systemResult->exitTick)
      return invalid("bridge completion is inconsistent with gem5 time");

    std::optional<sim::SpatialEngineBoundaryResult> spatialResult;
    if (facts.engine == Gem5SystemEngine::Rtl) {
      if (indexed.index() >= rtlClosures.size())
        return invalid("gem5 RTL import lost its exact mapped RTL closure");
      auto mappedText = readExternalToolInvocationDeclaredOutput(
          imported, mappedRtlLaunchResultPath(indexed.index()));
      if (!mappedText)
        return mappedText.takeError();
      const llvm::ArrayRef<std::uint8_t> mappedBytes(
          reinterpret_cast<const std::uint8_t *>(mappedText->data()),
          mappedText->size());
      if (mappedBytes != llvm::ArrayRef<std::uint8_t>(bridgeResult.result))
        return invalid("bridge payload differs from the mapped RTL result");
      auto mappedResult =
          eda::open_source::parseMappedRtlSimulationResult(*mappedText);
      if (!mappedResult)
        return mappedResult.takeError();
      if (mappedResult->terminal ==
          eda::open_source::MappedRtlTerminalStatus::StoppedByLimit) {
        if (bridgeResult.status != 1)
          return invalid("bridge status disagrees with the RTL terminal");
        return terminalResult(
            CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
      }
      if (bridgeResult.status != 0)
        return invalid("bridge status disagrees with the RTL terminal");
      auto boundary =
          eda::open_source::projectMappedRtlSpatialEngineBoundaryResult(
              rtlClosures[indexed.index()], *mappedResult, artifacts, blobs);
      if (!boundary)
        return boundary.takeError();
      spatialResult = std::move(*boundary);
    } else {
      auto boundary = sim::decodeSpatialEngineBoundaryResult(
          bridgeResult.result, launch.spatialWorkload,
          launch.spatialRuntimeInput, artifacts);
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
    if (!spatialResult ||
        !std::holds_alternative<sim::RetiredExecution>(spatialResult->terminal))
      return terminalResult(
          CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached});
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

  sim::SystemSimulationExecution execution{
      evaluationRequestReference(request),
      sim::RetiredExecution{},
      std::move(*functional),
      {{systemResult->entryTick, 0},
       sim::SystemEventCoordinate{systemResult->exitTick, 0},
       {systemResult->exitTick, 0}},
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

} // namespace loom::runtime
