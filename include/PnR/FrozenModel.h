#ifndef LOOM_PNR_FROZENMODEL_H
#define LOOM_PNR_FROZENMODEL_H

#include "Mapping/Artifact.h"
#include "PnR/FrozenRealizationGraph.h"
#include "PnR/FrozenRoutingGraph.h"
#include "PnR/PnrConfig.h"
#include "PnR/PnrProblemInputs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace loom::pnr {

namespace detail {
class FrozenModelBuilder;
} // namespace detail

class FrozenModelCacheKey final {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const FrozenModelCacheKey &lhs,
                         const FrozenModelCacheKey &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const FrozenModelCacheKey &lhs,
                         const FrozenModelCacheKey &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit FrozenModelCacheKey(Storage bytes) : bytes_(bytes) {}

  Storage bytes_;

  friend class detail::FrozenModelBuilder;
};

class FrozenModel final {
public:
  FrozenModel(const FrozenModel &) = delete;
  FrozenModel(FrozenModel &&) = delete;
  FrozenModel &operator=(const FrozenModel &) = delete;
  FrozenModel &operator=(FrozenModel &&) = delete;

  const mapping::ArtifactIdentity &dataflowIdentity() const {
    return dataflowIdentity_;
  }
  const mapping::ArtifactIdentity &techMappingIdentity() const {
    return techMappingIdentity_;
  }
  const mapping::ArtifactIdentity &fabricIdentity() const {
    return fabricIdentity_;
  }
  const mapping::ArtifactIdentity &constraintSetIdentity() const {
    return constraintSetIdentity_;
  }
  const ResolvedPnrConfigView &config() const { return config_; }
  llvm::ArrayRef<DeterministicWorkBudgetEntry> workBudget() const {
    return workBudget_;
  }
  const FrozenRealizationGraph &realizations() const { return realizations_; }
  const FrozenRoutingGraph &routing() const { return routing_; }
  const FrozenModelCacheKey &cacheKey() const { return cacheKey_; }

private:
  FrozenModel(mapping::ArtifactIdentity dataflowIdentity,
              mapping::ArtifactIdentity techMappingIdentity,
              mapping::ArtifactIdentity fabricIdentity,
              mapping::ArtifactIdentity constraintSetIdentity,
              ResolvedPnrConfigView config,
              std::vector<DeterministicWorkBudgetEntry> workBudget,
              FrozenRealizationGraph realizations, FrozenRoutingGraph routing,
              FrozenModelCacheKey cacheKey)
      : dataflowIdentity_(std::move(dataflowIdentity)),
        techMappingIdentity_(std::move(techMappingIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        constraintSetIdentity_(std::move(constraintSetIdentity)),
        config_(std::move(config)), workBudget_(std::move(workBudget)),
        realizations_(std::move(realizations)), routing_(std::move(routing)),
        cacheKey_(cacheKey) {}

  mapping::ArtifactIdentity dataflowIdentity_;
  mapping::ArtifactIdentity techMappingIdentity_;
  mapping::ArtifactIdentity fabricIdentity_;
  mapping::ArtifactIdentity constraintSetIdentity_;
  ResolvedPnrConfigView config_;
  std::vector<DeterministicWorkBudgetEntry> workBudget_;
  FrozenRealizationGraph realizations_;
  FrozenRoutingGraph routing_;
  FrozenModelCacheKey cacheKey_;

  friend class detail::FrozenModelBuilder;
};

using FrozenModelHandle = std::shared_ptr<const FrozenModel>;

llvm::Expected<FrozenModelCacheKey>
deriveFrozenModelCacheKey(const PnrProblemInputs &inputs);

llvm::Expected<FrozenModelHandle>
freezeSpatialPnrModel(const PnrProblemInputs &inputs);

llvm::Error revalidateFrozenModelCacheHit(const FrozenModel &model,
                                          const PnrProblemInputs &inputs);

std::string formatFrozenModelCacheKeyHex(const FrozenModelCacheKey &key);

} // namespace loom::pnr

#endif // LOOM_PNR_FROZENMODEL_H
