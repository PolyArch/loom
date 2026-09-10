#ifndef LOOM_LIB_PNR_SPATIALCOMPUTEPROGRESSINDEX_H
#define LOOM_LIB_PNR_SPATIALCOMPUTEPROGRESSINDEX_H

#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "PnR/SpatialComputeProgressState.h"

namespace loom::pnr {

class FrozenSpatialCapacityIndex;
class FrozenSpatialResourceIndex;
class FrozenSpatialTransferIndex;
class SpatialCandidateState;

namespace detail {

/// Frozen compute activations share the Mapping causality and Fabric capacity
/// owners. Only context choice, route occupancy and durable result cuts vary.
class SpatialComputeProgressIndex final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialComputeProgressIndex>>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const ::loom::mapping::TechMappingView &techMapping,
        const ::loom::fabric::FabricArtifactView &fabric,
        const FrozenSpatialCapacityIndex &capacity,
        const FrozenSpatialResourceIndex &resources,
        const FrozenSpatialTransferIndex &transfers);

  llvm::Expected<SpatialComputeProgressStateHandle>
  project(const SpatialCandidateState &candidate,
          const SpatialComputeProgressStateHandle &previous = {}) const;

  std::size_t retainedStorageBytes() const;
  std::size_t activationCount() const { return activations_.size(); }

private:
  struct Release final {
    ::loom::mapping::MappingProgressCausalReleaseProjection clause;
    PnrIndex sink = getInvalidPnrIndex();
  };
  struct Activation final {
    ::loom::mapping::MappingProgressActivationProjection projection;
    std::vector<Release> release;
  };

  SpatialComputeProgressIndex(::loom::mapping::FrozenMappingProgressModel model,
                              std::vector<PnrIndex> capacityDimensions,
                              std::vector<Activation> activations,
                              std::vector<PnrIndex> useActivationOffsets)
      : model_(std::move(model)),
        capacityDimensions_(std::move(capacityDimensions)),
        activations_(std::move(activations)),
        useActivationOffsets_(std::move(useActivationOffsets)) {}

  ::loom::mapping::FrozenMappingProgressModel model_;
  std::vector<PnrIndex> capacityDimensions_;
  std::vector<Activation> activations_;
  std::vector<PnrIndex> useActivationOffsets_;
};

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALCOMPUTEPROGRESSINDEX_H
