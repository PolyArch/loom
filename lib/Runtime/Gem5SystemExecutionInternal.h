#ifndef LOOM_LIB_RUNTIME_GEM5SYSTEMEXECUTIONINTERNAL_H
#define LOOM_LIB_RUNTIME_GEM5SYSTEMEXECUTIONINTERNAL_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Request.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/InvocationBundle.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::runtime::gem5_system {

inline constexpr evaluation::CaseSubjectRoleRef kDeploymentRole(0);
inline constexpr evaluation::CaseSubjectRoleRef kBindingRole(1);
inline constexpr evaluation::ModelOutputSlotRef kExecutionOutput(0);
inline constexpr llvm::StringLiteral kSystemResultPath =
    "outputs/system-result.json";
inline constexpr llvm::StringLiteral kMemoryResultPath =
    "outputs/system-memory.result";
inline constexpr llvm::StringLiteral kProjectionPath =
    "drivers/gem5-system-projection.json";
inline constexpr llvm::StringLiteral kConfigurationScriptPath =
    "drivers/configure_loom_system.py";
inline constexpr llvm::StringLiteral kDfgEnginePath =
    "drivers/loom-gem5-dfg-engine";
inline constexpr llvm::StringLiteral kCgraEnginePath =
    "drivers/loom-gem5-cgra-engine";
inline constexpr llvm::StringLiteral kBridgeHeaderPath =
    "drivers/Gem5BridgeWire.h";
inline constexpr llvm::StringLiteral kChannelPlanHeaderPath =
    "drivers/Gem5SpatialChannelPlan.h";
inline constexpr llvm::StringLiteral kRtlPeerManifestPath =
    "drivers/gem5-rtl-peers.txt";
inline constexpr llvm::StringLiteral kPackageObjectPath =
    "inputs/package/objects";
inline constexpr llvm::StringLiteral kHostElfPath = "inputs/host.elf";
inline constexpr llvm::StringLiteral kThreadDispatchPath =
    "inputs/thread-dispatch.bin";
inline constexpr llvm::StringLiteral kAdmissionPath = "inputs/admission.bin";
inline constexpr llvm::StringLiteral kMemoryTablePath =
    "inputs/system-memory-table.bin";
inline constexpr std::uint64_t kMaximumGem5Ticks = 20'000'000;
inline constexpr std::uint64_t kMaximumSpatialWork = 1'000'000;
inline constexpr std::uint64_t kGem5PageBytes = 4096;
inline constexpr std::uint64_t kGem5StackBytes = 64 * 1024;
inline constexpr std::uint64_t kThreadDispatchApertureBytes = 4096;
inline constexpr std::uint64_t kSpatialChannelBufferBytes = 1024 * 1024;
inline constexpr std::uint64_t kMaximumDenseSpatialLaunches = 4096;

enum class Gem5SystemEngine { Dfg, Cgra, Rtl };
enum class Gem5ProcessorModelKind { TimingSimple, O3 };

struct Gem5ProcessorProjection final {
  Gem5ProcessorFabricRef processor;
  Gem5ProcessorModelKind model = Gem5ProcessorModelKind::TimingSimple;
  Gem5RiscvCpuParameters parameters;
  std::uint32_t hardwareThreadCount = 0;
  std::vector<fabric::ExecutionUnitRecord> executionUnits;
  std::optional<fabric::OutOfOrderMicroarchitectureDeclaration> outOfOrder;
};

struct Gem5InstructionImage final {
  ArtifactRootReference reference;
  std::string path;
};

struct Gem5RuntimeImage final {
  std::string path;
  std::uint64_t address = 0;
};

struct Gem5DispatchTarget final {
  std::uint64_t cpuId = 0;
  std::uint64_t imageOrdinal = 0;
  std::string entrySymbol;
  std::uint64_t bridgeAddress = 0;
  std::uint64_t launchAddress = 0;
  std::uint64_t launchSize = 0;
};

struct Gem5MemoryObservationProjection final {
  std::uint64_t objectOrdinal = 0;
  std::uint64_t objectByteOffset = 0;
  std::uint64_t address = 0;
  std::uint64_t size = 0;
  sim::MemoryObservationForm form = sim::MemoryObservationForm::FullState;
};

struct ReadinessIdentity final {
  std::string binarySha256;
  external_tool::ExternalFileFingerprint binaryFingerprint;
};

struct Gem5SpatialLaunchProjection final {
  ArtifactRootReference fabric;
  ArtifactRootReference spatialMapping;
  ArtifactRootReference hardwareImplementation;
  ArtifactRootReference spatialWorkload;
  ArtifactRootReference spatialRuntimeInput;
  std::string channelProjectionPath;
  std::string channelEnginePlanPath;
  std::vector<std::uint8_t> launchPayload;
  Gem5DispatchTarget dispatchTarget;
  Gem5SpatialBridgeParameters bridge;
};

struct Gem5SystemFacts final {
  Gem5SystemEngine engine;
  ArtifactRootReference deployment;
  ArtifactRootReference binding;
  ArtifactRootReference dataflow;
  std::vector<Gem5SpatialLaunchProjection> spatialLaunches;
  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
  std::vector<Gem5ProcessorProjection> processors;
  std::string hostEntrySymbol;
  std::uint64_t hostCpuId = 0;
  std::vector<Gem5InstructionImage> instructionImages;
  std::vector<Gem5RuntimeImage> runtimeImages;
  std::uint64_t memoryInterfaceTableAddress = 0;
  std::uint64_t memoryInterfaceTableEntries = 0;
  std::vector<Gem5MemoryObservationProjection> memoryObservations;
  std::uint64_t dispatchAddress = 0;
  std::uint64_t stackBase = 0;
  std::uint64_t stackStride = 0;
  Gem5SimpleMemoryParameters memory;
};

using Gem5SystemFactsOrUnsupported =
    std::variant<Gem5SystemFacts, evaluation::UnsupportedEvidence>;

llvm::Error invalid(const llvm::Twine &message);

llvm::Expected<std::pair<ArtifactRootReference, ArtifactRootReference>>
systemSubjects(const evaluation::EvaluationRequest &request);

std::string instructionImagePath(std::size_t ordinal);
std::string spatialLaunchPath(std::size_t ordinal);
std::string spatialChannelProjectionPath(std::size_t ordinal);
std::string spatialChannelEnginePlanPath(std::size_t ordinal);
std::string spatialChannelBufferPath(std::size_t ordinal);
std::string spatialBridgeSocketPath(std::size_t ordinal);
std::string spatialBridgeResultPath(std::size_t ordinal);
std::string mappedRtlLaunchPrefix(std::size_t ordinal);
std::string mappedRtlLaunchResultPath(std::size_t ordinal);

llvm::Expected<std::shared_ptr<const sim::ImportedSystemSimulationInputs>>
importCachedSystemInputs(const ArtifactRootReference &workload,
                         const ArtifactRootReference &runtimeInput,
                         const ArtifactStore &artifacts,
                         const BlobStore &blobs);

llvm::Expected<Gem5SystemFactsOrUnsupported>
deriveFacts(const evaluation::EvaluationRequest &request,
            const evaluation::CaseArtifactResolution &resolution,
            const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::runtime::gem5_system

#endif // LOOM_LIB_RUNTIME_GEM5SYSTEMEXECUTIONINTERNAL_H
