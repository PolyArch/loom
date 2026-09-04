#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLEXECUTION_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLEXECUTION_H

#include "EDA/Adapters/OpenSource/MappedRtlSimulation.h"

#include "Evaluation/Evidence.h"
#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "ExternalTool/InvocationBundle.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
namespace external_tool {
struct ExternalToolProviderDescriptor;
}
} // namespace loom

namespace loom::eda::open_source {

/// The closed HDL simulator set of the mapped-RTL provider. Every member is a
/// catalog-owned backend tool that compiles the same generated harness and
/// writes the same result protocol; the Request's stable HDL simulator build
/// identity names exactly one member through that tool's version-probe marker.
enum class MappedRtlHdlSimulator : std::uint8_t { Verilator, Vcs, Xcelium };

/// Every member in declaration order, for the closed-set iterations of the
/// spelling, classification, and option owners.
inline constexpr MappedRtlHdlSimulator mappedRtlHdlSimulators[]{
    MappedRtlHdlSimulator::Verilator, MappedRtlHdlSimulator::Vcs,
    MappedRtlHdlSimulator::Xcelium};

/// The canonical command-line spelling of one simulator and its inverse.
llvm::StringRef mappedRtlHdlSimulatorSpelling(MappedRtlHdlSimulator simulator);
std::optional<MappedRtlHdlSimulator>
parseMappedRtlHdlSimulator(llvm::StringRef spelling);

/// The catalog provider that compiles and runs the harness for one member.
const external_tool::ExternalToolProviderDescriptor &
mappedRtlHdlSimulatorProvider(MappedRtlHdlSimulator simulator);

/// Recovers the member named by one stable HDL simulator build identity: the
/// unique member whose catalog version-probe marker the identity contains.
/// The identity's exact release is qualified separately through the catalog's
/// validated-release relation.
std::optional<MappedRtlHdlSimulator>
classifyMappedRtlHdlSimulator(llvm::StringRef stableHdlSimulatorBuildIdentity);

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
  std::string preamble;
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
  std::string configurationTransportReceiptPath;
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

/// The closed parallelism domain shared by Verilation jobs, make jobs, and
/// simulation model threads.
constexpr bool isMappedRtlParallelismCount(std::uint64_t value) {
  return value == 1 || value == 2 || value == 4 || value == 8;
}

/// The parallelism and thread contract of one mapped-RTL attempt, read from
/// the selected simulator's provider options. `buildJobs` is the compiler's
/// `-j`: for Verilator the Verilation job count and the make job count of the
/// generated build, for VCS its parallel compilation count. `modelThreads` is
/// Verilator's simulation thread count emitted as both `--threads` and
/// `--hierarchical-threads` so the generated main, the root model, and the
/// hierarchical schedule agree; VCS simulates single-threaded and admits no
/// thread option; Xcelium elaborates and simulates single-threaded and admits
/// the cycle limit alone. Both counts satisfy `isMappedRtlParallelismCount`.
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

/// The compile inputs of one VCS bundle: the attempt limit, the parallel
/// compilation count, and the frozen VCS executable. VCS compiles and links
/// the simulator itself, so the bundle freezes no auxiliary build tool.
struct MappedRtlVcsCompilationPlan final {
  std::uint64_t cycleLimit = 0;
  std::uint64_t buildJobs = 0;
  std::string vcsExecutable;
};

/// The elaboration inputs of one Xcelium bundle: the attempt limit and the
/// frozen xrun launcher. Xcelium elaborates the harness into a snapshot inside
/// its library directory and simulates that snapshot through the same
/// launcher. The bundle has no separate C++ build command or tool-produced
/// executable; the suite's compiler, elaborator, and simulator executables
/// are frozen as typed auxiliary tools by provider preparation.
struct MappedRtlXceliumElaborationPlan final {
  std::uint64_t cycleLimit = 0;
  std::string xrunExecutable;
};

/// The bundle of an event-driven member (VCS or Xcelium): the semantic
/// inputs, the shared harness, the member's argument file, the frozen compile
/// command, the tool-produced executables that command creates (the VCS
/// simulator; none for Xcelium, whose snapshot is a library), and the
/// simulation command. The harness compiles the semantic RTL source directly;
/// no hierarchy plan or derived library exists for these members.
struct MappedRtlEventDrivenBundleProjection final {
  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
  std::string testbenchPath;
  std::string driverPath;
  std::vector<std::string> toolProducedExecutables;
  std::string resultPath;
  std::string configurationTransportReceiptPath;
  std::string testbench;
  std::string driver;
  std::vector<std::string> compileCommand;
  std::vector<std::string> simulationCommand;
};

using MappedRtlExecutionProjectionOrUnsupported =
    std::variant<MappedRtlExecutionBundleProjection,
                 evaluation::UnsupportedEvidence>;

using MappedRtlEventDrivenProjectionOrUnsupported =
    std::variant<MappedRtlEventDrivenBundleProjection,
                 evaluation::UnsupportedEvidence>;

llvm::Expected<MappedRtlExecutionAttemptOptions>
resolveMappedRtlExecutionAttemptOptions(
    const external_tool::LocalToolConfig &localConfig,
    MappedRtlHdlSimulator simulator);

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

/// Derives the VCS materialization of one closure. The generated harness is
/// the same source the Verilator projection compiles.
llvm::Expected<MappedRtlEventDrivenProjectionOrUnsupported>
deriveMappedRtlVcsBundleProjection(const MappedRtlExecutionClosure &closure,
                                   const MappedRtlVcsCompilationPlan &plan,
                                   const ArtifactStore &artifacts,
                                   const BlobStore &blobs);

/// Derives the Xcelium materialization of one closure: the same harness, one
/// xrun argument file, the elaboration command, and the snapshot simulation.
llvm::Expected<MappedRtlEventDrivenProjectionOrUnsupported>
deriveMappedRtlXceliumBundleProjection(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlXceliumElaborationPlan &plan, const ArtifactStore &artifacts,
    const BlobStore &blobs);

llvm::Expected<external_tool::ExternalToolInvocationImportExpectation>
deriveMappedRtlExecutionImportExpectation(
    const MappedRtlExecutionClosure &closure, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef pathPrefix = {});

/// Validates one strictly parsed completion receipt against the exact
/// ConfigurationABI transport layouts selected by this execution closure.
/// Each program must account for every payload write and active-word
/// comparison and exactly one atomic commit and passing status read.
llvm::Error validateMappedRtlConfigurationTransportReceipt(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlConfigurationTransportReceipt &receipt,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Projects a strict retired RTL result into the shared Spatial engine
/// boundary. Stopped-by-limit classification remains with the descriptor.
llvm::Expected<sim::SpatialEngineBoundaryResult>
projectMappedRtlSpatialEngineBoundaryResult(
    const MappedRtlExecutionClosure &closure,
    const MappedRtlSimulationResult &result, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLEXECUTION_H
