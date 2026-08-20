#ifndef LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGHARDWAREDEMAND_H
#define LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGHARDWAREDEMAND_H

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::mapping {

/// One exact graph-boundary Hall deficit observed for a TechMapping on a
/// Module. The directional split is retained because a bidirectional gateway
/// contributes one endpoint to each independent direction.
class SpatialGraphBoundaryEndpointHallDeficit final {
public:
  static llvm::Expected<SpatialGraphBoundaryEndpointHallDeficit>
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
  std::uint64_t requiredBoundaryPairs() const;

private:
  SpatialGraphBoundaryEndpointHallDeficit(ArtifactRootReference module,
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

llvm::ArrayRef<std::uint8_t>
spatialGraphBoundaryEndpointHallFeedbackSchemaBytes();

std::vector<std::uint8_t> encodeSpatialGraphBoundaryEndpointHallFeedback(
    const SpatialGraphBoundaryEndpointHallDeficit &feedback);

llvm::Expected<SpatialGraphBoundaryEndpointHallDeficit>
adoptSpatialGraphBoundaryEndpointHallFeedback(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &module,
    llvm::ArrayRef<ArtifactRootReference> techMappings,
    const ArtifactStore &store);

/// Retains the largest exact boundary-pair gap, followed by the larger Hall
/// demand set and canonical bytes.
void retainSpatialGraphBoundaryEndpointHallFeedback(
    std::optional<SpatialGraphBoundaryEndpointHallDeficit> &retained,
    SpatialGraphBoundaryEndpointHallDeficit candidate);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGHARDWAREDEMAND_H
