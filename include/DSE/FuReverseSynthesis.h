#ifndef LOOM_DSE_FUREVERSESYNTHESIS_H
#define LOOM_DSE_FUREVERSESYNTHESIS_H

#include "Common/ArtifactStore.h"
#include "DSE/CandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    fuReverseSynthesisCandidateGeneratorKind(23);

enum class FuReverseSynthesisOutput : std::uint32_t {
  Module,
  TechMapping,
  JointTechMapping,
  System,
  PhysicalTimingProfile,
  ConfigurationAbi,
};

constexpr CandidateGeneratorOutputSlotRef
fuReverseSynthesisOutputSlot(FuReverseSynthesisOutput output) {
  return CandidateGeneratorOutputSlotRef(static_cast<std::uint32_t>(output));
}

enum class FuReverseSynthesisFailure : std::uint8_t {
  EmptyGraphSet,
  InvalidGraphReference,
  DuplicateGraph,
  UnsupportedGraphInterface,
  UnsupportedActorInventory,
  UnsupportedActorSchema,
  UnsupportedActorProjection,
  UnsupportedGraphTopology,
  UnsupportedGraphReachability,
  CapabilityDerivationRejected,
  FabricFinalizationFailed,
  MappingInfeasible,
  MappingIncomplete,
  CancelledOrTimeout,
  MappingInvalid,
  MappingInternal,
  CoverageNotEstablished,
};

class FuReverseSynthesisError final
    : public llvm::ErrorInfo<FuReverseSynthesisError> {
public:
  static char ID;

  FuReverseSynthesisError(FuReverseSynthesisFailure failure,
                          std::string message)
      : failure_(failure), message_(std::move(message)) {}

  FuReverseSynthesisFailure failure() const { return failure_; }
  llvm::StringRef diagnostic() const { return message_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  FuReverseSynthesisFailure failure_;
  std::string message_;
};

/// One transient proof that a canonical input graph has a complete legal
/// binding to the synthesized FU. The witness is constructed before Mapping
/// and is never persisted as a Mapping artifact.
struct FuSynthesisCoverageWitness final {
  ::dataflow::GraphRef graph;
  ArtifactIdentity fabric;
  ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate;
  std::vector<::loom::mapping::TechComputeActorView> actors;
  std::vector<::loom::mapping::TechComputeBoundaryView> boundaries;
};

class ScalarIntegerAddSubFuSynthesisResult;

/// Deterministic System-level children of one synthesized bounded FU Module.
/// The System is a one-AccCore execution shell; timing and configuration
/// remain owned by their existing artifact schemas.
class ScalarIntegerAddSubFuSystemArtifacts final {
public:
  const ::loom::fabric::FinalizedFabricRoot &system() const { return system_; }
  const ArtifactRootReference &physicalTimingProfile() const {
    return physicalTimingProfile_;
  }
  const ::loom::hardware::FinalizedConfigurationABI &configurationAbi() const {
    return configurationAbi_;
  }

private:
  ScalarIntegerAddSubFuSystemArtifacts(
      ::loom::fabric::FinalizedFabricRoot system,
      ArtifactRootReference physicalTimingProfile,
      ::loom::hardware::FinalizedConfigurationABI configurationAbi)
      : system_(std::move(system)),
        physicalTimingProfile_(std::move(physicalTimingProfile)),
        configurationAbi_(std::move(configurationAbi)) {}

  ::loom::fabric::FinalizedFabricRoot system_;
  ArtifactRootReference physicalTimingProfile_;
  ::loom::hardware::FinalizedConfigurationABI configurationAbi_;

  friend llvm::Expected<ScalarIntegerAddSubFuSystemArtifacts>
  materializeScalarIntegerAddSubFuSystemArtifacts(
      const ::loom::fabric::FinalizedFabricRoot &, const ArtifactStore &);
};

/// One materialized synthesis attempt after Fabric finalization. An
/// incomplete attempt retains its exact Fabric, every completed per-graph
/// TechMapping, any completed whole-domain TechMapping, and the pre-Mapping
/// coverage witnesses so an orchestration owner can publish a truthful
/// retained prefix.
class ScalarIntegerAddSubFuSynthesisAttempt final {
public:
  const ::loom::fabric::FinalizedFabricRoot &fabric() const { return fabric_; }
  llvm::ArrayRef<::loom::mapping::FinalizedTechMapping>
  perGraphMappings() const {
    return perGraphMappings_;
  }
  const std::optional<::loom::mapping::FinalizedTechMapping> &
  jointMapping() const {
    return jointMapping_;
  }
  llvm::ArrayRef<FuSynthesisCoverageWitness> coverage() const {
    return coverage_;
  }
  std::optional<FuReverseSynthesisFailure> termination() const {
    return termination_;
  }
  llvm::StringRef terminationMessage() const { return terminationMessage_; }
  bool complete() const { return !termination_; }
  std::uint64_t plannedMappingInvocations() const {
    return coverage_.size() + 1;
  }
  std::uint64_t consumedMappingInvocations() const {
    return consumedMappingInvocations_;
  }

private:
  ScalarIntegerAddSubFuSynthesisAttempt(
      ::loom::fabric::FinalizedFabricRoot fabric,
      std::vector<::loom::mapping::FinalizedTechMapping> perGraphMappings,
      std::optional<::loom::mapping::FinalizedTechMapping> jointMapping,
      std::vector<FuSynthesisCoverageWitness> coverage,
      std::optional<FuReverseSynthesisFailure> termination,
      std::string terminationMessage, std::uint64_t consumedMappingInvocations)
      : fabric_(std::move(fabric)),
        perGraphMappings_(std::move(perGraphMappings)),
        jointMapping_(std::move(jointMapping)), coverage_(std::move(coverage)),
        termination_(termination),
        terminationMessage_(std::move(terminationMessage)),
        consumedMappingInvocations_(consumedMappingInvocations) {}

  ::loom::fabric::FinalizedFabricRoot fabric_;
  std::vector<::loom::mapping::FinalizedTechMapping> perGraphMappings_;
  std::optional<::loom::mapping::FinalizedTechMapping> jointMapping_;
  std::vector<FuSynthesisCoverageWitness> coverage_;
  std::optional<FuReverseSynthesisFailure> termination_;
  std::string terminationMessage_;
  std::uint64_t consumedMappingInvocations_ = 0;

  friend llvm::Expected<ScalarIntegerAddSubFuSynthesisAttempt>
  attemptScalarIntegerAddSubFuSynthesis(
      const ::dataflow::CanonicalDataflowProgramView &,
      llvm::ArrayRef<::dataflow::GraphRef>,
      const ::loom::mapping::ResolvedTechMappingConfigView &,
      const ArtifactStore &, ExecutionControlView);
  friend llvm::Expected<ScalarIntegerAddSubFuSynthesisResult>
  synthesizeScalarIntegerAddSubFu(
      const ::dataflow::CanonicalDataflowProgramView &,
      llvm::ArrayRef<::dataflow::GraphRef>,
      const ::loom::mapping::ResolvedTechMappingConfigView &,
      const ArtifactStore &, ExecutionControlView);
};

/// A fully materialized bounded synthesis result. Each input graph has one
/// independently finalized coverage Mapping. One whole-domain Mapping assigns
/// them to distinct resident contexts of the shared FU for deployment.
class ScalarIntegerAddSubFuSynthesisResult final {
public:
  const ::loom::fabric::FinalizedFabricRoot &fabric() const { return fabric_; }
  llvm::ArrayRef<::loom::mapping::FinalizedTechMapping>
  perGraphMappings() const {
    return perGraphMappings_;
  }
  const ::loom::mapping::FinalizedTechMapping &jointMapping() const {
    return jointMapping_;
  }
  llvm::ArrayRef<FuSynthesisCoverageWitness> coverage() const {
    return coverage_;
  }

private:
  ScalarIntegerAddSubFuSynthesisResult(
      ::loom::fabric::FinalizedFabricRoot fabric,
      std::vector<::loom::mapping::FinalizedTechMapping> perGraphMappings,
      ::loom::mapping::FinalizedTechMapping jointMapping,
      std::vector<FuSynthesisCoverageWitness> coverage)
      : fabric_(std::move(fabric)),
        perGraphMappings_(std::move(perGraphMappings)),
        jointMapping_(std::move(jointMapping)), coverage_(std::move(coverage)) {
  }

  ::loom::fabric::FinalizedFabricRoot fabric_;
  std::vector<::loom::mapping::FinalizedTechMapping> perGraphMappings_;
  ::loom::mapping::FinalizedTechMapping jointMapping_;
  std::vector<FuSynthesisCoverageWitness> coverage_;

  friend llvm::Expected<ScalarIntegerAddSubFuSynthesisResult>
  synthesizeScalarIntegerAddSubFu(
      const ::dataflow::CanonicalDataflowProgramView &,
      llvm::ArrayRef<::dataflow::GraphRef>,
      const ::loom::mapping::ResolvedTechMappingConfigView &,
      const ArtifactStore &, ExecutionControlView);
};

/// Resolves an exact published transient witness and rejects copied or mutated
/// bindings, including graph-to-actor and Fabric-owner mismatches.
llvm::Expected<const FuSynthesisCoverageWitness *>
resolveFuSynthesisCoverage(const ScalarIntegerAddSubFuSynthesisResult &result,
                           const FuSynthesisCoverageWitness &witness);

llvm::Expected<ScalarIntegerAddSubFuSynthesisAttempt>
attemptScalarIntegerAddSubFuSynthesis(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const ::loom::mapping::ResolvedTechMappingConfigView &mappingConfig,
    const ArtifactStore &store, ExecutionControlView executionControl = {});

/// Synthesizes a single explicit FU for a non-empty set of canonical graphs.
/// Each graph must be exactly `(i32, i32) -> i32` with one overflow-free
/// integer add/sub actor followed by one terminal token sync. The resulting FU
/// contains one shared add/sub resource, one sync resource, and one complete
/// capability template with one resident instruction context per graph.
/// TechMapping materialization proves every graph's actor, port, boundary, and
/// internal-edge correspondence independently and as one deployable union.
llvm::Expected<ScalarIntegerAddSubFuSynthesisResult>
synthesizeScalarIntegerAddSubFu(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const ::loom::mapping::ResolvedTechMappingConfigView &mappingConfig,
    const ArtifactStore &store, ExecutionControlView executionControl = {});

/// Recomputes the explicit canonical graph set and requires the exact Fabric
/// identity produced by this bounded synthesis owner.
llvm::Error verifyScalarIntegerAddSubFuFabricLineage(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const ArtifactStore &store);

/// Requires a one-graph TechMapping to equal the coverage witness derived for
/// that graph from the exact complete-domain synthesized Fabric.
llvm::Error verifyScalarIntegerAddSubFuMappingLineage(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const ::loom::mapping::FinalizedTechMapping &mapping,
    const ArtifactStore &store);

/// Requires one whole-domain TechMapping to equal the union of every
/// per-graph coverage witness derived from the exact synthesized Fabric.
llvm::Error verifyScalarIntegerAddSubFuJointMappingLineage(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::GraphRef> graphs,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const ::loom::mapping::FinalizedTechMapping &mapping,
    const ArtifactStore &store);

/// Builds the unique one-AccCore System shell used to carry this bounded
/// Module through System Mapping and portable operation-leaf RTL. The shell
/// has no memory fabric; its minimal message carrier supports only the none
/// and i32 transfers required by the admitted value-only graph domain.
llvm::Expected<ScalarIntegerAddSubFuSystemArtifacts>
materializeScalarIntegerAddSubFuSystemArtifacts(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ArtifactStore &store);

/// Independently reconstructs the System identity and strictly imports its
/// normalized timing and packed ConfigurationABI children.
llvm::Error verifyScalarIntegerAddSubFuSystemLineage(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ScalarIntegerAddSubFuSystemArtifacts &artifacts,
    const ArtifactStore &store);

llvm::Error verifyScalarIntegerAddSubFuSystemIdentity(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ::loom::fabric::FinalizedFabricRoot &system,
    const ArtifactStore &store);

llvm::Error verifyScalarIntegerAddSubFuPhysicalTimingLineage(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ArtifactRootReference &physicalTimingProfile,
    const ArtifactStore &store);

llvm::Error verifyScalarIntegerAddSubFuConfigurationAbiLineage(
    const ::loom::fabric::FinalizedFabricRoot &system,
    const ArtifactRootReference &configurationAbi, const ArtifactStore &store);

const CandidateGeneratorDescriptor &
fuReverseSynthesisCandidateGeneratorDescriptor();
llvm::Error registerFuReverseSynthesisCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFuReverseSynthesisCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFuReverseSynthesisCandidateGeneratorBinding(
    const ::loom::mapping::ResolvedTechMappingConfigView &mappingConfig);

} // namespace loom::dse

#endif // LOOM_DSE_FUREVERSESYNTHESIS_H
