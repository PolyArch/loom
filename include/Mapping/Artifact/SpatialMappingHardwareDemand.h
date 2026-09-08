#ifndef LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGHARDWAREDEMAND_H
#define LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGHARDWAREDEMAND_H

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::mapping {

/// A directional boundary-capacity proposal after bounded Mapping failure.
/// Distinct logical nets may share transport, so this is not an infeasibility
/// proof. Each added gateway contributes one endpoint in each direction.
class SpatialGraphBoundaryCapacitySuggestion final {
public:
  static llvm::Expected<SpatialGraphBoundaryCapacitySuggestion>
  get(ArtifactRootReference module, ArtifactRootReference techMapping,
      std::uint64_t inputDemandCount, std::uint64_t inputEndpointCount,
      std::uint64_t outputDemandCount, std::uint64_t outputEndpointCount);

  const ArtifactRootReference &module() const { return module_; }
  const ArtifactRootReference &techMapping() const { return techMapping_; }
  std::uint64_t demandCount() const {
    return inputDemandCount_ + outputDemandCount_;
  }
  std::uint64_t endpointCount() const {
    return inputEndpointCount_ + outputEndpointCount_;
  }
  std::uint64_t inputDemandCount() const { return inputDemandCount_; }
  std::uint64_t inputEndpointCount() const { return inputEndpointCount_; }
  std::uint64_t outputDemandCount() const { return outputDemandCount_; }
  std::uint64_t outputEndpointCount() const { return outputEndpointCount_; }
  std::uint64_t proposedAdditionalBoundaryPairs() const;

private:
  SpatialGraphBoundaryCapacitySuggestion(ArtifactRootReference module,
                                          ArtifactRootReference techMapping,
                                          std::uint64_t inputDemandCount,
                                          std::uint64_t inputEndpointCount,
                                          std::uint64_t outputDemandCount,
                                          std::uint64_t outputEndpointCount)
      : module_(std::move(module)), techMapping_(std::move(techMapping)),
        inputDemandCount_(inputDemandCount),
        inputEndpointCount_(inputEndpointCount),
        outputDemandCount_(outputDemandCount),
        outputEndpointCount_(outputEndpointCount) {}

  ArtifactRootReference module_;
  ArtifactRootReference techMapping_;
  std::uint64_t inputDemandCount_;
  std::uint64_t inputEndpointCount_;
  std::uint64_t outputDemandCount_;
  std::uint64_t outputEndpointCount_;
};

/// A hardware search proposal from one retained routing candidate. The
/// capacity counts guaranteed resident channels, not FIFO storage entries.
/// It is not an infeasibility proof for the current Module.
class SpatialFifoChannelCapacitySuggestion final {
public:
  static llvm::Expected<SpatialFifoChannelCapacitySuggestion>
  get(ArtifactRootReference module, ArtifactRootReference techMapping,
      fabric::FabricFifoOccurrenceRef owner, std::uint64_t selectedChannels,
      std::uint64_t proposedChannels,
      std::vector<dataflow::CanonicalGraphProducerEndpointRef> logicalNets,
      std::vector<fabric::FabricPhysicalTraversalRef> routeAnchors);

  const ArtifactRootReference &module() const { return module_; }
  const ArtifactRootReference &techMapping() const { return techMapping_; }
  fabric::FabricFifoOccurrenceRef owner() const { return owner_; }
  std::uint64_t selectedChannels() const { return selectedChannels_; }
  std::uint64_t proposedChannels() const { return proposedChannels_; }
  llvm::ArrayRef<dataflow::CanonicalGraphProducerEndpointRef>
  logicalNets() const {
    return logicalNets_;
  }
  llvm::ArrayRef<fabric::FabricPhysicalTraversalRef> routeAnchors() const {
    return routeAnchors_;
  }

private:
  SpatialFifoChannelCapacitySuggestion(
      ArtifactRootReference module, ArtifactRootReference techMapping,
      fabric::FabricFifoOccurrenceRef owner, std::uint64_t selectedChannels,
      std::uint64_t proposedChannels,
      std::vector<dataflow::CanonicalGraphProducerEndpointRef> logicalNets,
      std::vector<fabric::FabricPhysicalTraversalRef> routeAnchors)
      : module_(std::move(module)), techMapping_(std::move(techMapping)),
        owner_(owner), selectedChannels_(selectedChannels),
        proposedChannels_(proposedChannels),
        logicalNets_(std::move(logicalNets)),
        routeAnchors_(std::move(routeAnchors)) {}

  ArtifactRootReference module_;
  ArtifactRootReference techMapping_;
  fabric::FabricFifoOccurrenceRef owner_;
  std::uint64_t selectedChannels_;
  std::uint64_t proposedChannels_;
  std::vector<dataflow::CanonicalGraphProducerEndpointRef> logicalNets_;
  std::vector<fabric::FabricPhysicalTraversalRef> routeAnchors_;
};

class TechMappingView;

llvm::Expected<std::optional<SpatialGraphBoundaryCapacitySuggestion>>
deriveSpatialGraphBoundaryCapacitySuggestion(
    const ArtifactRootReference &module,
    const ArtifactRootReference &techMapping, const TechMappingView &tech,
    const fabric::FabricArtifactView &fabric);

using SpatialMappingHardwareFeedback =
    std::variant<SpatialGraphBoundaryCapacitySuggestion,
                 SpatialFifoChannelCapacitySuggestion>;

llvm::ArrayRef<std::uint8_t> spatialMappingHardwareFeedbackSchemaBytes();

std::vector<std::uint8_t> encodeSpatialMappingHardwareFeedback(
    const SpatialMappingHardwareFeedback &feedback);

llvm::Expected<SpatialMappingHardwareFeedback>
adoptSpatialMappingHardwareFeedback(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &module,
    llvm::ArrayRef<ArtifactRootReference> techMappings,
    const ArtifactStore &store);

/// Prefer a reservation proposal from admitted routes to a boundary-capacity
/// proposal. Within a family retain the largest requested capacity, then the
/// larger witness and canonical bytes.
void retainSpatialMappingHardwareFeedback(
    std::optional<SpatialMappingHardwareFeedback> &retained,
    SpatialMappingHardwareFeedback candidate);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGHARDWAREDEMAND_H
