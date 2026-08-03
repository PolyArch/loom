#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <tuple>
#include <variant>
#include <vector>

using namespace loom::pnr;

namespace {

llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

bool rangeContains(PnrIndex offset, PnrIndex count, PnrIndex value) {
  return value >= offset && value - offset < count;
}

bool exposureTargetAgrees(
    const FrozenSpatialMemoryBindingTargetOption &bindingTarget,
    const FrozenSpatialMemoryExposureOption &option) {
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &bindingTarget.target)) {
    const auto *local =
        std::get_if<::loom::fabric::LocalMemoryServiceRef>(&option.target);
    return local && local->underlying() == region->service;
  }
  return std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
      option.target);
}

} // namespace

llvm::Expected<const FrozenSpatialMemoryDispatchDomain *>
SpatialCandidateState::memoryDispatchDomain(PnrIndex useOrdinal) const {
  const auto &problem = *problem_;
  const auto uses = problem.memory().rootedUses();
  if (useOrdinal >= uses.size())
    return candidateError("rooted memory use is out of range");
  const auto &use = uses[useOrdinal];
  const auto &realizations = problem.realizations();
  if (use.actor >= realizations.memoryActors().size())
    return candidateError("rooted memory use has a foreign actor");
  const PnrIndex owner = realizations.memoryActorRealizations()[use.actor];
  if (owner >= realizations.memoryRealizations().size())
    return candidateError("rooted memory use has a foreign realization");
  const auto &realization = realizations.memoryRealizations()[owner];
  if (use.actor < realization.actorOffset ||
      use.actor - realization.actorOffset >= realization.actorCount)
    return candidateError("rooted memory use is outside its actor slice");
  const PnrIndex placement = memoryBindings_[owner].placement;
  const auto placementOffsets = problem.memory().memoryPlacementDomainOffsets();
  if (placement >= realizations.memoryPlacements().size() ||
      placement + 1 >= placementOffsets.size())
    return candidateError("rooted memory use has no selected placement");
  const PnrIndex localActor = use.actor - realization.actorOffset;
  const PnrIndex domainOrdinal = placementOffsets[placement] + localActor;
  if (domainOrdinal >= placementOffsets[placement + 1] ||
      domainOrdinal >= problem.memory().dispatchDomains().size())
    return candidateError("rooted memory use has no dispatch domain");
  const auto &domain = problem.memory().dispatchDomains()[domainOrdinal];
  if (domain.placement != placement || domain.actor != use.actor)
    return candidateError("rooted memory dispatch domain is inconsistent");
  return &domain;
}

namespace {

bool admitsRegion(const FrozenSpatialMemoryIndex &memory,
                  const FrozenSpatialMemoryDispatchOption &option,
                  std::uint64_t region) {
  const auto regions = memory.dispatchServiceRegionOrdinals().slice(
      option.serviceRegionOffset, option.serviceRegionCount);
  return std::binary_search(regions.begin(), regions.end(), region);
}

} // namespace

llvm::Error
SpatialCandidateState::validateLogicalMemoryBinding(PnrIndex binding) const {
  const auto &memory = problem_->memory();
  if (binding >= memory.logicalBindings().size())
    return candidateError("logical memory binding is out of range");
  const auto &selection = logicalMemoryBindings_[binding];
  if (selection.target >= memory.bindingTargets().size())
    return candidateError("logical memory binding target is out of range");
  const auto &target = memory.bindingTargets()[selection.target];
  if (std::holds_alternative<FrozenSpatialMemoryBoundaryProxy>(target.target)) {
    if (selection.physicalOffsetBytes != 0)
      return candidateError("BoundaryProxy carries a physical byte offset");
    return llvm::Error::success();
  }
  const auto extent = memory.logicalBindings()[binding].staticExtentBytes;
  if (!extent)
    return candidateError(
        "unbounded logical memory selects a local service region");
  if (selection.physicalOffsetBytes > target.sizeBytes ||
      *extent > target.sizeBytes - selection.physicalOffsetBytes)
    return candidateError(
        "logical memory interval exceeds its selected local region");
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::validateLogicalMemoryBindingOverlap(
    PnrIndex binding) const {
  const auto &memory = problem_->memory();
  if (binding >= logicalMemoryBindings_.size())
    return candidateError("logical memory binding is out of range");
  const auto &selection = logicalMemoryBindings_[binding];
  if (selection.target >= memory.bindingTargets().size())
    return candidateError("logical memory binding target is out of range");
  if (!std::holds_alternative<::loom::fabric::FabricMemoryServiceRegionRef>(
          memory.bindingTargets()[selection.target].target))
    return llvm::Error::success();
  const auto extent = memory.logicalBindings()[binding].staticExtentBytes;
  if (!extent)
    return candidateError(
        "unbounded logical memory selects a local service region");
  const std::uint64_t begin = selection.physicalOffsetBytes;
  const std::uint64_t end = begin + *extent;
  for (PnrIndex other = 0; other < logicalMemoryBindings_.size(); ++other) {
    if (other == binding)
      continue;
    const auto &candidate = logicalMemoryBindings_[other];
    if (candidate.target != selection.target)
      continue;
    const auto otherExtent = memory.logicalBindings()[other].staticExtentBytes;
    if (!otherExtent)
      return candidateError(
          "unbounded logical memory selects a local service region");
    const std::uint64_t otherBegin = candidate.physicalOffsetBytes;
    const std::uint64_t otherEnd = otherBegin + *otherExtent;
    if (begin < otherEnd && otherBegin < end)
      return candidateError(
          "logical memories overlap in one local service region");
  }
  return llvm::Error::success();
}

llvm::Error
SpatialCandidateState::validateMemoryUseDispatch(PnrIndex useOrdinal) const {
  const auto &memory = problem_->memory();
  auto domain = memoryDispatchDomain(useOrdinal);
  if (!domain)
    return domain.takeError();
  const PnrIndex selected = memoryUseDispatches_[useOrdinal];
  if (!rangeContains((*domain)->optionOffset, (*domain)->optionCount, selected))
    return candidateError(
        "rooted memory use selects outside its dispatch domain");
  const auto &option = memory.dispatchOptions()[selected];
  const auto &use = memory.rootedUses()[useOrdinal];

  if (!use.logicalBinding) {
    if (std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(
            option.target))
      return candidateError("fence use selects an addressed local service");
    return llvm::Error::success();
  }
  if (*use.logicalBinding >= logicalMemoryBindings_.size())
    return candidateError("addressed use has a foreign logical binding");
  const auto &binding = logicalMemoryBindings_[*use.logicalBinding];
  const auto &target = memory.bindingTargets()[binding.target];
  if (const auto *region =
          std::get_if<::loom::fabric::FabricMemoryServiceRegionRef>(
              &target.target)) {
    const auto *dispatch =
        std::get_if<::loom::fabric::LocalMemoryServiceRef>(&option.target);
    if (!dispatch || dispatch->underlying() != region->service ||
        !admitsRegion(memory, option, region->ordinal))
      return candidateError(
          "local addressed dispatch disagrees with its MemoryBinding");
    return llvm::Error::success();
  }
  if (!std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
          option.target))
    return candidateError(
        "BoundaryProxy addressed use does not select a manager endpoint");
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::validateMemoryExposureSelection(
    PnrIndex exposureOrdinal) const {
  const auto &memory = problem_->memory();
  if (exposureOrdinal >= memory.exposures().size())
    return candidateError("memory exposure is out of range");
  const auto &exposure = memory.exposures()[exposureOrdinal];
  if (exposure.logicalBinding >= logicalMemoryBindings_.size())
    return candidateError("memory exposure has a foreign logical binding");
  const PnrIndex selected = memoryExposureSelections_[exposureOrdinal];
  if (selected >= memory.exposureOptions().size())
    return candidateError("memory exposure option is out of range");
  const auto &option = memory.exposureOptions()[selected];
  if (option.provider >= memory.exposureProviders().size())
    return candidateError("memory exposure option has a foreign provider");
  const auto &binding = logicalMemoryBindings_[exposure.logicalBinding];
  if (binding.target >= memory.bindingTargets().size())
    return candidateError("memory exposure binding target is out of range");
  if (!exposureTargetAgrees(memory.bindingTargets()[binding.target], option))
    return candidateError(
        "memory exposure dispatch disagrees with its MemoryBinding");
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::rebuildMemoryExposureUsage() {
  const auto &memory = problem_->memory();
  if (memoryExposureSelections_.size() != memory.exposures().size())
    return candidateError(
        "memory exposure selection dimensions are incomplete");
  memoryExposureProviderRefcounts_.clear();
  memoryExposureProviderRefcounts_.reserve(memory.exposures().size() * 2);
  memoryExposureProviderBindingCounts_.assign(memory.exposureProviders().size(),
                                              0);
  for (PnrIndex exposure = 0; exposure < memory.exposures().size();
       ++exposure) {
    if (llvm::Error error = validateMemoryExposureSelection(exposure))
      return error;
    const PnrIndex provider =
        memory.exposureOptions()[memoryExposureSelections_[exposure]].provider;
    const auto key =
        std::make_pair(memory.exposures()[exposure].logicalBinding, provider);
    PnrIndex &refcount = memoryExposureProviderRefcounts_[key];
    if (refcount == std::numeric_limits<PnrIndex>::max())
      return candidateError("memory exposure provider refcount overflows");
    if (refcount++ == 0)
      ++memoryExposureProviderBindingCounts_[provider];
  }
  return llvm::Error::success();
}

void SpatialCandidateState::changeMemoryExposureUsage(PnrIndex exposure,
                                                      PnrIndex oldOption,
                                                      PnrIndex newOption) {
  const auto &memory = problem_->memory();
  assert(exposure < memory.exposures().size());
  assert(oldOption < memory.exposureOptions().size());
  assert(newOption < memory.exposureOptions().size());
  const PnrIndex oldProvider = memory.exposureOptions()[oldOption].provider;
  const PnrIndex newProvider = memory.exposureOptions()[newOption].provider;
  if (oldProvider == newProvider)
    return;

  const PnrIndex binding = memory.exposures()[exposure].logicalBinding;
  const auto updateProvider = [&](PnrIndex provider, bool add) {
    assert(provider < memory.exposureProviders().size());
    const auto key = std::make_pair(binding, provider);
    auto [entry, inserted] =
        memoryExposureProviderRefcounts_.try_emplace(key, 0);
    (void)inserted;
    PnrIndex &refcount = entry->second;
    PnrIndex &bindingCount = memoryExposureProviderBindingCounts_[provider];
    const std::uint64_t capacity =
        memory.exposureProviders()[provider].maxExposedBindings;
    const std::uint64_t oldOveruse =
        bindingCount > capacity ? bindingCount - capacity : 0;
    if (add) {
      assert(refcount != std::numeric_limits<PnrIndex>::max());
      if (refcount++ == 0)
        ++bindingCount;
    } else {
      assert(refcount != 0);
      if (--refcount == 0) {
        assert(bindingCount != 0);
        --bindingCount;
      }
    }
    const std::uint64_t newOveruse =
        bindingCount > capacity ? bindingCount - capacity : 0;
    assert(capacityOveruse_ >= oldOveruse);
    capacityOveruse_ = capacityOveruse_ - oldOveruse + newOveruse;
  };
  updateProvider(oldProvider, false);
  updateProvider(newProvider, true);
}

llvm::Error SpatialCandidateState::verifyMemorySelections() const {
  const auto &memory = problem_->memory();
  if (logicalMemoryBindings_.size() != memory.logicalBindings().size() ||
      memoryUseDispatches_.size() != memory.rootedUses().size() ||
      memoryExposureSelections_.size() != memory.exposures().size() ||
      memoryExposureProviderBindingCounts_.size() !=
          memory.exposureProviders().size())
    return candidateError("memory selection dimensions are incomplete");
  for (PnrIndex binding = 0; binding < logicalMemoryBindings_.size(); ++binding)
    if (llvm::Error error = validateLogicalMemoryBinding(binding))
      return error;
  for (PnrIndex use = 0; use < memoryUseDispatches_.size(); ++use)
    if (llvm::Error error = validateMemoryUseDispatch(use))
      return error;
  llvm::DenseMap<std::pair<PnrIndex, PnrIndex>, PnrIndex> expectedRefcounts;
  expectedRefcounts.reserve(memory.exposures().size() * 2);
  std::vector<PnrIndex> expectedBindingCounts(memory.exposureProviders().size(),
                                              0);
  for (PnrIndex exposure = 0; exposure < memoryExposureSelections_.size();
       ++exposure) {
    if (llvm::Error error = validateMemoryExposureSelection(exposure))
      return error;
    const PnrIndex provider =
        memory.exposureOptions()[memoryExposureSelections_[exposure]].provider;
    const auto key =
        std::make_pair(memory.exposures()[exposure].logicalBinding, provider);
    PnrIndex &refcount = expectedRefcounts[key];
    if (refcount++ == 0)
      ++expectedBindingCounts[provider];
  }
  if (expectedBindingCounts != memoryExposureProviderBindingCounts_)
    return candidateError(
        "memory exposure provider counts diverge from selected options");
  for (const auto &entry : expectedRefcounts) {
    const auto actual = memoryExposureProviderRefcounts_.find(entry.first);
    if (actual == memoryExposureProviderRefcounts_.end() ||
        actual->second != entry.second)
      return candidateError(
          "memory exposure provider refcounts are incomplete");
  }
  for (const auto &entry : memoryExposureProviderRefcounts_) {
    if (entry.second == 0)
      continue;
    const auto expected = expectedRefcounts.find(entry.first);
    if (expected == expectedRefcounts.end() || expected->second != entry.second)
      return candidateError(
          "memory exposure provider refcounts contain a stale selection");
  }

  struct Allocation final {
    PnrIndex target = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  std::vector<Allocation> allocations;
  allocations.reserve(logicalMemoryBindings_.size());
  for (PnrIndex binding = 0; binding < logicalMemoryBindings_.size();
       ++binding) {
    const auto &selection = logicalMemoryBindings_[binding];
    const auto &target = memory.bindingTargets()[selection.target];
    if (!std::holds_alternative<::loom::fabric::FabricMemoryServiceRegionRef>(
            target.target))
      continue;
    const std::uint64_t extent =
        *memory.logicalBindings()[binding].staticExtentBytes;
    allocations.push_back({selection.target, selection.physicalOffsetBytes,
                           selection.physicalOffsetBytes + extent});
  }
  llvm::sort(allocations, [](const Allocation &left, const Allocation &right) {
    return std::tie(left.target, left.begin, left.end) <
           std::tie(right.target, right.begin, right.end);
  });
  for (std::size_t index = 1; index < allocations.size(); ++index)
    if (allocations[index - 1].target == allocations[index].target &&
        allocations[index].begin < allocations[index - 1].end)
      return candidateError(
          "logical memories overlap in one local service region");
  return llvm::Error::success();
}
