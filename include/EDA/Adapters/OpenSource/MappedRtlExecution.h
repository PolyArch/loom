#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLEXECUTION_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLEXECUTION_H

#include "EDA/Adapters/OpenSource/MappedRtlSimulation.h"

#include "Evaluation/Evidence.h"
#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "ExternalTool/InvocationBundle.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::eda::open_source {

/// Complete typed inputs for one mapped-RTL engine. Evaluation descriptors
/// derive this closure from their own exact Request; the engine does not infer
/// subjects, configuration, or semantic identity.
struct MappedRtlExecutionClosure final {
  evaluation::models::MappedRtlSimulatorBinding simulatorBinding;
  external_tool::ExternalToolSemanticContract semanticContract;
  ArtifactRootReference hardwareImplementation;
  ArtifactRootReference deployment;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
};

struct MappedRtlExecutionBundleProjection final {
  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
  std::string testbenchPath;
  std::string standaloneVerilatorDriverPath;
  std::string bridgedVerilatorDriverPath;
  std::string bridgeEngineSourcePath;
  std::string simulatorExecutablePath;
  std::string resultPath;
  std::string testbench;
  std::string standaloneVerilatorDriver;
  std::string bridgedVerilatorDriver;
};

struct MappedRtlExecutionAttemptOptions final {
  std::uint64_t cycleLimit = 0;
  std::uint64_t buildJobs = 0;
  std::uint64_t debugVerbosity = 0;
  std::vector<std::string> inheritedEnvironment;
};

using MappedRtlExecutionProjectionOrUnsupported =
    std::variant<MappedRtlExecutionBundleProjection,
                 evaluation::UnsupportedEvidence>;

llvm::Expected<MappedRtlExecutionAttemptOptions>
resolveMappedRtlExecutionAttemptOptions(
    const external_tool::LocalToolConfig &localConfig);

/// Derives the exact RTL materialization and both environment adapters from
/// one closure. The generated harness semantics are shared; only the C++ main
/// selected by the Verilator driver differs between standalone and bridge use.
llvm::Expected<MappedRtlExecutionProjectionOrUnsupported>
deriveMappedRtlExecutionBundleProjection(
    const MappedRtlExecutionClosure &closure, std::uint64_t cycleLimit,
    std::uint64_t buildJobs, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef pathPrefix = {});

llvm::Expected<external_tool::ExternalToolInvocationImportExpectation>
deriveMappedRtlExecutionImportExpectation(
    const MappedRtlExecutionClosure &closure, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef pathPrefix = {});

/// Projects a strict retired RTL result into the shared Spatial engine
/// boundary. Stopped-by-limit classification remains with the descriptor.
llvm::Expected<sim::SpatialEngineBoundaryResult>
projectMappedRtlSpatialEngineBoundaryResult(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlSimulationResult &result, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLEXECUTION_H
