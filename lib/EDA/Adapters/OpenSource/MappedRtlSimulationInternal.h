#ifndef LOOM_LIB_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLSIMULATIONINTERNAL_H
#define LOOM_LIB_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLSIMULATIONINTERNAL_H

#include "EDA/Adapters/OpenSource/MappedRtlExecution.h"

#include "Deployment/Deployment.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Evaluation/Evidence.h"
#include "ExternalTool/InvocationBundle.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "Hardware/RTL/ConfigurationTransport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::eda::open_source::detail {

struct RtlPort final {
  std::string name;
  hardware::RepresentationSignalDirection direction =
      hardware::RepresentationSignalDirection::Input;
  std::uint64_t bitWidth = 0;
};

struct ClockPort final {
  std::string name;
  std::uint64_t periodFs = 0;
  std::uint64_t phaseFs = 0;
};

struct ResetPort final {
  std::string name;
  bool assertedValue = true;
};

struct TransportPort final {
  std::string prefix;
  std::uint32_t payloadBitWidth = 0;
  std::optional<llvm::APInt> physicalTag;
};

struct InputTokenStream final {
  TransportPort port;
  std::uint64_t tokenCount = 0;
  std::vector<llvm::APInt> tokens;
};

struct OutputTokenStream final {
  TransportPort port;
  std::uint32_t tokenBitWidth = 0;
};

struct ConfigurationProgram final {
  hardware::rtl::ConfigurationTransportUnitLayout layout;
  std::string portPrefix;
  std::vector<std::uint8_t> image;
};

struct RuntimeMemoryImage final {
  std::uint64_t canonicalBaseAddress = 0;
  std::vector<sim::SemanticMemoryByte> initialBytes;
};

struct MemoryBoundaryPort final {
  std::string prefix;
  std::uint32_t addressBitWidth = 0;
  std::uint32_t dataBitWidth = 0;
  std::uint32_t maskBitWidth = 0;
  std::uint64_t rootObjectOrdinal = 0;
  std::uint64_t rootByteOffset = 0;
};

struct MemoryObservationPlan final {
  std::uint64_t objectOrdinal = 0;
  std::uint64_t byteOffset = 0;
  sim::MemoryObservationForm form = sim::MemoryObservationForm::FullState;
};

struct MappedRtlObservationFacts final {
  std::shared_ptr<const sim::ImportedSpatialSimulationInputs> inputs;
  std::vector<RuntimeMemoryImage> memoryImages;
  std::vector<MemoryObservationPlan> memoryObservations;
};

struct MappedRtlInvocationFacts final {
  evaluation::models::MappedRtlSimulatorBinding simulatorBinding;
  external_tool::ExternalToolSemanticContract semanticContract;
  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
  std::vector<std::string> rtlPaths;
  std::string top;
  std::vector<RtlPort> rootPorts;
  std::vector<ClockPort> clockPorts;
  std::vector<ResetPort> resetPorts;
  std::string selectedClock;
  std::uint64_t selectedClockPeriodFs = 0;
  std::vector<ConfigurationProgram> configurationPrograms;
  std::optional<InputTokenStream> startInput;
  std::vector<InputTokenStream> valueInputs;
  std::vector<InputTokenStream> streamInputs;
  std::vector<OutputTokenStream> valueResults;
  std::vector<OutputTokenStream> streamOutputs;
  std::vector<TransportPort> completionOutputs;
  std::vector<RuntimeMemoryImage> memoryImages;
  std::vector<MemoryBoundaryPort> memoryBoundaryPorts;
  std::vector<MemoryObservationPlan> memoryObservations;
  std::uint64_t cycleLimit = 0;
};

using MappedRtlFactsOrUnsupported =
    std::variant<MappedRtlInvocationFacts, evaluation::UnsupportedEvidence>;

llvm::Expected<MappedRtlFactsOrUnsupported> deriveMappedRtlInvocationFacts(
    const MappedRtlExecutionClosure &closure,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<external_tool::ExternalToolInvocationImportExpectation>
deriveMappedRtlImportExpectation(
    const MappedRtlExecutionClosure &closure,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<MappedRtlObservationFacts> deriveMappedRtlObservationFacts(
    const MappedRtlExecutionClosure &closure,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<std::string>
renderMappedRtlTestbench(const MappedRtlInvocationFacts &facts);

llvm::Expected<std::string>
renderMappedRtlVerilatorDriver(const MappedRtlInvocationFacts &facts,
                               std::uint64_t buildJobs);

llvm::Expected<std::string>
renderMappedRtlBridgedVerilatorDriver(const MappedRtlInvocationFacts &facts,
                                      std::uint64_t buildJobs);

llvm::Expected<sim::SpatialFunctionalObservations>
projectMappedRtlFunctionalObservations(const MappedRtlObservationFacts &facts,
                                       const MappedRtlSimulationResult &result);

} // namespace loom::eda::open_source::detail

#endif // LOOM_LIB_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLSIMULATIONINTERNAL_H
