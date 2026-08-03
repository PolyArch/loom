#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
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

llvm::Error SpatialCandidateState::verifyMemorySelections() const {
  const auto &memory = problem_->memory();
  if (logicalMemoryBindings_.size() != memory.logicalBindings().size() ||
      memoryUseDispatches_.size() != memory.rootedUses().size())
    return candidateError("memory selection dimensions are incomplete");
  for (PnrIndex binding = 0; binding < logicalMemoryBindings_.size(); ++binding)
    if (llvm::Error error = validateLogicalMemoryBinding(binding))
      return error;
  for (PnrIndex use = 0; use < memoryUseDispatches_.size(); ++use)
    if (llvm::Error error = validateMemoryUseDispatch(use))
      return error;

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
