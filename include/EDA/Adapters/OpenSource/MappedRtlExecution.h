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

struct MappedRtlHierarchyBlock final {
  std::string module;
  std::uint64_t bodyLines = 0;
  std::uint64_t transitiveBodyLines = 0;
  std::uint64_t rootInstanceMultiplicity = 0;
  std::vector<std::string> sourceClosureModules;
};

/// How Verilator compiles one mapped-RTL bundle. A plan that selects at least
/// one block is Verilated hierarchically: Verilator elaborates each block
/// through a child argument file and emits the `V<top>_hier.mk` makefile whose
/// `hier_build` target owns the C++ build. A plan without a block has no child
/// argument file and no block metacomment, so Verilator emits the ordinary
/// `V<top>.mk` whose target is the simulator executable.
enum class MappedRtlVerilationStyle : std::uint8_t { Flat, Hierarchical };

struct MappedRtlHierarchyPlan final {
  std::string selectionPolicy;
  std::uint64_t baselineBlockCount = 0;
  std::uint64_t baselineRootSourceClosureModuleCount = 0;
  std::uint64_t baselineRootSourceClosureBodyLines = 0;
  std::uint64_t baselineRootSourceClosureBytes = 0;
  std::uint64_t rootSourceClosureBodyLines = 0;
  std::uint64_t rootSourceClosureBytes = 0;
  std::string sourcePath;
  std::string sourceSha256;
  std::uint64_t sourceByteCount = 0;
  std::uint64_t framingByteCount = 0;
  std::uint64_t preambleByteCount = 0;
  std::string hardwareRootModule;
  std::uint64_t hardwareRootBodyLines = 0;
  std::uint64_t hardwareRootTransitiveBodyLines = 0;
  std::vector<std::string> hardwareRootSourceClosureModules;
  std::vector<MappedRtlHierarchyBlock> blocks;
  std::vector<std::string> derivedRtlPaths;
  std::string rtlLibraryDirectoryPath;
  std::string manifestPath;
  std::string workDirectoryPath;
  MappedRtlVerilationStyle style = MappedRtlVerilationStyle::Flat;
  std::string verilationMakefileName;
};

/// The frozen auxiliary tools of one mapped-RTL build: GNU make, the C++
/// compiler, linker, and archiver named in the generated build, and the
/// hierarchy Verilator launcher that the generated hierarchy makefile runs in
/// place of Verilator.
struct MappedRtlBuildTools final {
  std::string make;
  std::string cxx;
  std::string linker;
  // The linker executable plus any driver-mode argument required to link C++.
  std::string linkerInvocation;
  std::string archiver;
  std::string hierarchyLauncher;
  std::vector<external_tool::ResolvedAuxiliaryToolExecutable> provenance;
};

struct MappedRtlExecutionBundleProjection final {
  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
  std::vector<external_tool::MaterializedBundleFile> toolLocalInputs;
  MappedRtlHierarchyPlan hierarchy;
  std::vector<std::string> configurationProgramPaths;
  std::string testbenchPath;
  std::string standaloneVerilatorDriverPath;
  std::string bridgedVerilatorDriverPath;
  std::string bridgeEngineSourcePath;
  std::string simulatorExecutablePath;
  std::string resultPath;
  std::string testbench;
  std::string standaloneVerilatorDriver;
  std::string bridgedVerilatorDriver;
  /// The frozen build command: make, its work directory and generated
  /// makefile, the job count, the target of the Verilation style, and the
  /// exact tool variables of the generated build.
  std::vector<std::string> buildCommand;
};

/// The frozen mapped-RTL parallelism of an attempt that does not choose its
/// own. Both counts are the measured operating point of the product Matmul
/// bundle on a sixteen-core host: eight Verilation and build jobs, and eight
/// model threads, which simulate three times faster than one thread. They are
/// the single owner of these defaults; command-line front ends present them.
inline constexpr std::uint64_t mappedRtlDefaultBuildJobs = 8;
inline constexpr std::uint64_t mappedRtlDefaultBuildWorkers = 1;
inline constexpr std::uint64_t mappedRtlDefaultModelThreads = 8;

/// The Verilator parallelism and thread contract of one mapped-RTL attempt.
/// `buildJobs` is Verilator's `-j`, the Verilation job count and the make job
/// count of the generated build; `modelThreads` is the simulation thread
/// count emitted as both `--threads` and `--hierarchical-threads` so the
/// generated main, the root model, and the hierarchical schedule agree. Both
/// use the closed domain {1, 2, 4, 8}.
struct MappedRtlExecutionAttemptOptions final {
  std::uint64_t cycleLimit = 0;
  std::uint64_t buildJobs = 0;
  std::uint64_t buildWorkers = 0;
  std::uint64_t modelThreads = 0;
  std::vector<std::string> inheritedEnvironment;
};

/// The build inputs of one bundle projection: the attempt limits, the job
/// share of this bundle, the frozen Verilator executable, and the frozen
/// auxiliary build tools.
struct MappedRtlVerilationPlan final {
  std::uint64_t cycleLimit = 0;
  std::uint64_t buildJobs = 0;
  std::uint64_t modelThreads = 0;
  std::string verilatorExecutable;
  MappedRtlBuildTools buildTools;
};

using MappedRtlExecutionProjectionOrUnsupported =
    std::variant<MappedRtlExecutionBundleProjection,
                 evaluation::UnsupportedEvidence>;

llvm::Expected<MappedRtlExecutionAttemptOptions>
resolveMappedRtlExecutionAttemptOptions(
    const external_tool::LocalToolConfig &localConfig);

llvm::Expected<MappedRtlBuildTools>
resolveMappedRtlBuildTools(
    const external_tool::LocalToolConfig &localConfig);

/// Derives the exact RTL materialization and both environment adapters from
/// one closure. The generated harness semantics are shared; only the C++ main
/// selected by the Verilator driver differs between standalone and bridge use.
llvm::Expected<MappedRtlExecutionProjectionOrUnsupported>
deriveMappedRtlExecutionBundleProjection(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlVerilationPlan &plan, const ArtifactStore &artifacts,
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
