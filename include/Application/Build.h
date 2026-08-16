#ifndef LOOM_APPLICATION_BUILD_H
#define LOOM_APPLICATION_BUILD_H

#include "Config/ResolvedConfig.h"
#include "DSE/InvocationManifest.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/PreMappingExploration.h"
#include "Deployment/Deployment.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/CompilerTargetLinker.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace llvm {
class Module;
}

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

inline constexpr llvm::StringLiteral applicationBuildProducerIdentity{
    "loom.application.build.v1"};

struct ApplicationPointerMemoryObservable final {
  std::uint64_t argumentOrdinal = 0;
  sim::MemoryObservationForm form = sim::MemoryObservationForm::FullState;
};

/// One exact source invocation expressed in the existing Structured Program
/// workload and runtime-input domains. Symbol resolution is invocation-local;
/// all persistent inputs remain owned by SimulationWorkload and
/// SimulationRuntimeInput.
struct ApplicationSourceInvocation final {
  std::string entrySymbol;
  std::vector<sim::StructuredProgramArgumentSource> argumentPlan;
  bool observeReturnValue = false;
  std::vector<ApplicationPointerMemoryObservable> memoryObservables;
  std::vector<sim::StructuredRuntimeValueEntry> runtimeValues;
  std::vector<sim::RuntimeMemoryObject> memoryObjects;
  std::vector<sim::StructuredPointerBindingDraft> pointerBindings;
};

struct ApplicationBuildRequest final {
  ApplicationSourceInvocation sourceInvocation;
  ArtifactRootReference system;
  std::vector<ArtifactRootReference> physicalTimingProfiles;
  ResolvedConfig resolvedConfig;
  dse::JointDesignPolicy jointPolicy;
  frontend::PreMappingCompilationOptions compilationOptions;
  dse::PreMappingExplorationOptions preMappingOptions;
};

struct PreparedApplicationSoftware final {
  frontend::PublishedPreMappingCompilation compilation;
  std::vector<ArtifactRootReference> workloads;
};

struct PreparedApplicationBuild final {
  ApplicationSourceInvocation sourceInvocation;
  std::vector<PreparedApplicationSoftware> software;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<dse::DsePlanGenerateInvocationRecords>
      preMappingGenerateInvocations;
  dse::JointDesignExplorationPlan mappingPlan;
};

struct ApplicationDeploymentRequest final {
  CompilerTargetPolicy compilerTargetPolicy;
  CompilerTargetLinkWorkspace linkerWorkspace;
};

struct ApplicationDeploymentArtifacts final {
  ArtifactRootReference configurationAbi;
  hardware::ConfigurationABIConstructionStatistics configurationAbiConstruction;
  std::vector<deployment::DeploymentHardwareBinding> hardwareBindings;
  std::vector<ArtifactRootReference> instructionCoreBinaries;
  deployment::FinalizedDeployment deployment;
};

struct ApplicationMappingExecutionRequest final {
  dse::DseProducerSemanticBuildIdentity producer;
  std::string journalRoot;
  std::vector<ArtifactRootReference> preexistingEvidence;
  dse::SiteCapacity siteCapacity;
  dse::PlanExecutionPolicy executionPolicy;
};

enum class ApplicationBuildUnsupportedKind : std::uint8_t {
  RootCoordinates,
};

struct UnsupportedApplicationBuild final {
  ApplicationBuildUnsupportedKind kind;
  ArtifactRootReference canonicalDataflow;
  dataflow::RootThreadLaunchRef root;
};

using ApplicationBuildPreparationOutcome = std::variant<
    PreparedApplicationBuild, dse::CompletedPreMappingNoFeasibleCandidate,
    dse::IncompletePreMappingExploration, UnsupportedApplicationBuild>;

/// Composes final-link, compiler, Simulation input, and joint-DSE owners
/// without creating another persistent application or candidate identity.
llvm::Expected<ApplicationBuildPreparationOutcome>
prepareApplicationBuild(const llvm::Module &finalLinkedModule,
                        ApplicationBuildRequest request,
                        const ArtifactStore &artifacts, const BlobStore &blobs);

/// Executes or resumes a prepared Mapping plan through the shared bounded
/// journal, scheduler, exact repair, and independent Mapping verifiers.
llvm::Expected<dse::JointDesignExecution>
executeApplicationMapping(const PreparedApplicationBuild &prepared,
                          ApplicationMappingExecutionRequest request,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

/// Requires one uniquely selected, independently imported SystemMapping and
/// derives the complete declarative Deployment closure. The host executable
/// and InstructionCore executables are generated from the exact final-linked
/// module and target bindings; no RTL generation or compilation occurs here.
llvm::Expected<ApplicationDeploymentArtifacts> buildApplicationDeployment(
    const PreparedApplicationBuild &prepared,
    const dse::JointDesignExecution &mappingExecution,
    const llvm::Module &finalLinkedModule, ApplicationDeploymentRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_BUILD_H
