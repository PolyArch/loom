#include "SpatialMemoryConstraintModel.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

using Projection = ::mapping::SpatialConstraintProjection;
using Key = std::vector<std::uint8_t>;

llvm::Error freezeInvalid(Projection projection, const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid,
      ("invalid Spatial memory constraint projection: " + message).str(),
      projection);
}

llvm::Error runtimeInvalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_memory_constraint_invalid: %s", message.str().c_str());
}

template <typename Ref> Key fabricKey(const Ref &reference) {
  return canonicalFabricBytes(reference);
}

::dataflow::LogicalMemoryRootRef
rootOf(const ::dataflow::LogicalMemoryRootOrViewRef &memory) {
  return std::visit(
      [](const auto &reference) -> ::dataflow::LogicalMemoryRootRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, ::dataflow::LogicalMemoryRootRef>)
          return reference;
        else
          return reference.root;
      },
      memory);
}

llvm::Expected<std::uint64_t> addressBound(Projection projection,
                                           const llvm::APInt &value) {
  if (value.getActiveBits() > 64)
    return freezeInvalid(projection,
                         "address interval exceeds the u64 address domain");
  return value.getZExtValue();
}

void normalizeAddresses(std::vector<SpatialMemoryAddressInterval> &values) {
  llvm::sort(values, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.service, lhs.lower, lhs.upper) <
           std::tie(rhs.service, rhs.lower, rhs.upper);
  });
  std::size_t write = 0;
  for (const auto &value : values) {
    if (write != 0 && values[write - 1].service == value.service &&
        value.lower <= values[write - 1].upper) {
      values[write - 1].upper = std::max(values[write - 1].upper, value.upper);
      continue;
    }
    values[write++] = value;
  }
  values.resize(write);
}

bool addressSubset(llvm::ArrayRef<SpatialMemoryAddressInterval> selected,
                   llvm::ArrayRef<SpatialMemoryAddressInterval> domain) {
  std::size_t allowed = 0;
  for (const auto &interval : selected) {
    while (allowed < domain.size() &&
           (domain[allowed].service < interval.service ||
            (domain[allowed].service == interval.service &&
             domain[allowed].upper <= interval.lower)))
      ++allowed;
    if (allowed == domain.size() ||
        domain[allowed].service != interval.service ||
        domain[allowed].lower > interval.lower ||
        domain[allowed].upper < interval.upper)
      return false;
  }
  return true;
}

bool addressesIntersect(llvm::ArrayRef<SpatialMemoryAddressInterval> lhs,
                        llvm::ArrayRef<SpatialMemoryAddressInterval> rhs) {
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < lhs.size() && right < rhs.size()) {
    if (lhs[left].service < rhs[right].service ||
        (lhs[left].service == rhs[right].service &&
         lhs[left].upper <= rhs[right].lower)) {
      ++left;
      continue;
    }
    if (rhs[right].service < lhs[left].service ||
        (lhs[left].service == rhs[right].service &&
         rhs[right].upper <= lhs[left].lower)) {
      ++right;
      continue;
    }
    return true;
  }
  return false;
}

bool servicesIntersect(llvm::ArrayRef<PnrIndex> lhs,
                       llvm::ArrayRef<PnrIndex> rhs) {
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < lhs.size() && right < rhs.size()) {
    if (lhs[left] < rhs[right])
      ++left;
    else if (rhs[right] < lhs[left])
      ++right;
    else
      return true;
  }
  return false;
}

llvm::Expected<PnrIndex> checkedIndex(std::size_t value, llvm::StringRef table,
                                      PnrCapacityMeasure measure) {
  return checkedPnrIndex(
      {"SpatialMemoryConstraintModel", table, table, measure}, value);
}

} // namespace

llvm::Expected<std::shared_ptr<const SpatialMemoryConstraintModel>>
SpatialMemoryConstraintModel::create(const FrozenSpatialMemoryIndex &memory,
                                     const FrozenConstraintIndex &constraints) {
  auto result = std::make_shared<SpatialMemoryConstraintModel>();

  const FrozenConstraintShard &serviceShard =
      constraints.shard(Projection::MemoryBoundServices);
  const FrozenConstraintShard &addressShard =
      constraints.shard(Projection::MemoryAddressRegion);

  std::vector<std::uint64_t> rootEntities;
  std::map<std::uint64_t, ::dataflow::LogicalMemoryRootRef> rootReferences;
  rootEntities.reserve(memory.logicalBindings().size() +
                       serviceShard.subjects().size() +
                       addressShard.subjects().size());
  for (const auto &binding : memory.logicalBindings()) {
    const auto root = rootOf(binding.logicalMemory);
    rootEntities.push_back(root.entity.value());
    rootReferences.try_emplace(root.entity.value(), root);
  }
  const auto appendRootSubjects = [&](const FrozenConstraintShard &shard,
                                      Projection projection) -> llvm::Error {
    for (const auto &subject : shard.subjects()) {
      const auto *root =
          std::get_if<::dataflow::LogicalMemoryRootRef>(&subject);
      if (!root)
        return freezeInvalid(projection,
                             "memory projection has a non-root subject");
      rootEntities.push_back(root->entity.value());
      rootReferences.try_emplace(root->entity.value(), *root);
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          appendRootSubjects(serviceShard, Projection::MemoryBoundServices))
    return std::move(error);
  if (llvm::Error error =
          appendRootSubjects(addressShard, Projection::MemoryAddressRegion))
    return std::move(error);
  llvm::sort(rootEntities);
  rootEntities.erase(std::unique(rootEntities.begin(), rootEntities.end()),
                     rootEntities.end());
  std::map<std::uint64_t, PnrIndex> rootOrdinals;
  for (auto [ordinal, entity] : llvm::enumerate(rootEntities)) {
    auto checked = checkedIndex(ordinal, "roots", PnrCapacityMeasure::Index);
    if (!checked)
      return checked.takeError();
    rootOrdinals.emplace(entity, *checked);
  }

  result->bindingRoots_.reserve(memory.logicalBindings().size());
  result->bindingExtents_.reserve(memory.logicalBindings().size());
  result->rootBindingOffsets_.assign(rootEntities.size() + 1, 0);
  for (const auto &binding : memory.logicalBindings()) {
    const auto found =
        rootOrdinals.find(rootOf(binding.logicalMemory).entity.value());
    if (found == rootOrdinals.end())
      return freezeInvalid(Projection::MemoryBoundServices,
                           "logical binding has no root projection");
    result->bindingRoots_.push_back(found->second);
    result->bindingExtents_.push_back(binding.staticExtentBytes);
    ++result->rootBindingOffsets_[found->second + 1];
  }
  for (std::size_t index = 1; index < result->rootBindingOffsets_.size();
       ++index)
    result->rootBindingOffsets_[index] +=
        result->rootBindingOffsets_[index - 1];
  result->rootBindings_.resize(memory.logicalBindings().size());
  std::vector<PnrIndex> bindingCursors = result->rootBindingOffsets_;
  for (auto [binding, root] : llvm::enumerate(result->bindingRoots_))
    result->rootBindings_[bindingCursors[root]++] =
        static_cast<PnrIndex>(binding);

  std::map<Key, PnrIndex> serviceOrdinals;
  const auto rememberService = [&](const FabricMemoryServiceRef &service) {
    serviceOrdinals.try_emplace(fabricKey(service), 0);
  };
  for (const auto &target : memory.bindingTargets())
    if (const auto *region =
            std::get_if<FabricMemoryServiceRegionRef>(&target.target))
      rememberService(region->service);
  for (const auto &value : serviceShard.domainValues())
    if (const auto *service = std::get_if<FabricMemoryServiceRef>(&value))
      rememberService(*service);
  for (const auto &value : addressShard.domainValues())
    if (const auto *region =
            std::get_if<SpatialConstraintAddressRegion>(&value))
      rememberService(region->service);
  for (auto entry : llvm::enumerate(serviceOrdinals)) {
    auto &[key, ordinal] = entry.value();
    (void)key;
    auto checked =
        checkedIndex(entry.index(), "services", PnrCapacityMeasure::Index);
    if (!checked)
      return checked.takeError();
    ordinal = *checked;
  }

  result->targets_.reserve(memory.bindingTargets().size());
  result->targetSizes_.reserve(memory.bindingTargets().size());
  for (const auto &target : memory.bindingTargets()) {
    TargetProjection projection;
    projection.addressBaseBytes = target.addressBaseBytes;
    if (const auto *region =
            std::get_if<FabricMemoryServiceRegionRef>(&target.target)) {
      const auto found = serviceOrdinals.find(fabricKey(region->service));
      if (found == serviceOrdinals.end())
        return freezeInvalid(Projection::MemoryBoundServices,
                             "local target service has no dense ordinal");
      projection.service = found->second;
    }
    result->targets_.push_back(projection);
    result->targetSizes_.push_back(target.sizeBytes);
  }

  result->rootDomains_.resize(rootEntities.size());
  for (auto entry : llvm::enumerate(rootEntities)) {
    auto checkedRoot =
        checkedIndex(entry.index(), "roots", PnrCapacityMeasure::Index);
    if (!checkedRoot)
      return checkedRoot.takeError();
    const PnrIndex root = *checkedRoot;
    RootDomain &record = result->rootDomains_[root];
    const SpatialConstraintSubject subject{rootReferences.at(entry.value())};

    const auto serviceDomain = serviceShard.restrictedDomain(subject);
    record.servicesRestricted = serviceDomain.has_value();
    auto serviceOffset =
        checkedIndex(result->serviceDomainValues_.size(),
                     "service_domain_values", PnrCapacityMeasure::Offset);
    if (!serviceOffset)
      return serviceOffset.takeError();
    record.serviceOffset = *serviceOffset;
    if (serviceDomain) {
      for (const auto &value : *serviceDomain) {
        const auto *service = std::get_if<FabricMemoryServiceRef>(&value);
        if (!service)
          return freezeInvalid(Projection::MemoryBoundServices,
                               "service domain has a non-service value");
        const auto found = serviceOrdinals.find(fabricKey(*service));
        if (found == serviceOrdinals.end())
          return freezeInvalid(Projection::MemoryBoundServices,
                               "service domain has no dense ordinal");
        result->serviceDomainValues_.push_back(found->second);
      }
      auto values = llvm::MutableArrayRef(result->serviceDomainValues_)
                        .drop_front(record.serviceOffset);
      llvm::sort(values);
      auto uniqueEnd = std::unique(values.begin(), values.end());
      result->serviceDomainValues_.resize(
          record.serviceOffset +
          static_cast<std::size_t>(uniqueEnd - values.begin()));
    }
    auto serviceCount =
        checkedIndex(result->serviceDomainValues_.size() - record.serviceOffset,
                     "service_domain_values", PnrCapacityMeasure::Count);
    if (!serviceCount)
      return serviceCount.takeError();
    record.serviceCount = *serviceCount;

    const auto addressDomain = addressShard.restrictedDomain(subject);
    record.addressesRestricted = addressDomain.has_value();
    result->hasAddressDomainRestrictions_ |= record.addressesRestricted;
    auto addressOffset =
        checkedIndex(result->addressDomainValues_.size(),
                     "address_domain_values", PnrCapacityMeasure::Offset);
    if (!addressOffset)
      return addressOffset.takeError();
    record.addressOffset = *addressOffset;
    if (addressDomain) {
      for (const auto &value : *addressDomain) {
        const auto *region =
            std::get_if<SpatialConstraintAddressRegion>(&value);
        if (!region)
          return freezeInvalid(Projection::MemoryAddressRegion,
                               "address domain has a non-region value");
        const auto service = serviceOrdinals.find(fabricKey(region->service));
        if (service == serviceOrdinals.end())
          return freezeInvalid(Projection::MemoryAddressRegion,
                               "address service has no dense ordinal");
        for (const auto &interval : region->intervals) {
          auto lower =
              addressBound(Projection::MemoryAddressRegion, interval.lower);
          if (!lower)
            return lower.takeError();
          auto upper =
              addressBound(Projection::MemoryAddressRegion, interval.upper);
          if (!upper)
            return upper.takeError();
          result->addressDomainValues_.push_back(
              {service->second, *lower, *upper});
        }
      }
      auto values = std::vector<SpatialMemoryAddressInterval>(
          result->addressDomainValues_.begin() + record.addressOffset,
          result->addressDomainValues_.end());
      result->addressDomainValues_.resize(record.addressOffset);
      normalizeAddresses(values);
      result->addressDomainValues_.insert(result->addressDomainValues_.end(),
                                          values.begin(), values.end());
    }
    auto addressCount =
        checkedIndex(result->addressDomainValues_.size() - record.addressOffset,
                     "address_domain_values", PnrCapacityMeasure::Count);
    if (!addressCount)
      return addressCount.takeError();
    record.addressCount = *addressCount;
  }

  const auto appendRelations =
      [&](const FrozenConstraintShard &shard,
          SpatialMemoryConstraintProjection projection,
          llvm::ArrayRef<FrozenConstraintRelation> relations,
          SpatialMemoryConstraintRelationKind kind) -> llvm::Error {
    for (const FrozenConstraintRelation &relation : relations) {
      auto offset =
          checkedIndex(result->relationMembers_.size(), "relation_members",
                       PnrCapacityMeasure::Offset);
      if (!offset)
        return offset.takeError();
      for (PnrIndex subjectOrdinal : shard.relationMembers().slice(
               relation.memberOffset, relation.memberCount)) {
        if (subjectOrdinal >= shard.subjects().size())
          return freezeInvalid(shard.projection(),
                               "relation member is out of range");
        const auto *root = std::get_if<::dataflow::LogicalMemoryRootRef>(
            &shard.subjects()[subjectOrdinal]);
        if (!root)
          return freezeInvalid(shard.projection(),
                               "relation member is not a logical root");
        const auto found = rootOrdinals.find(root->entity.value());
        if (found == rootOrdinals.end())
          return freezeInvalid(shard.projection(),
                               "relation root has no dense ordinal");
        result->relationMembers_.push_back(found->second);
      }
      result->relations_.push_back(
          {projection, kind, *offset, relation.memberCount});
    }
    return llvm::Error::success();
  };
  for (const auto &[shard, projection] :
       {std::pair<const FrozenConstraintShard *,
                  SpatialMemoryConstraintProjection>{
            &serviceShard, SpatialMemoryConstraintProjection::BoundServices},
        {&addressShard, SpatialMemoryConstraintProjection::AddressRegion}}) {
    if (llvm::Error error =
            appendRelations(*shard, projection, shard->equalityClasses(),
                            SpatialMemoryConstraintRelationKind::Equal))
      return std::move(error);
    if (llvm::Error error =
            appendRelations(*shard, projection, shard->disjointGroups(),
                            SpatialMemoryConstraintRelationKind::Disjoint))
      return std::move(error);
  }

  result->hasConstraints_ = llvm::any_of(result->rootDomains_,
                                         [](const RootDomain &domain) {
                                           return domain.servicesRestricted ||
                                                  domain.addressesRestricted;
                                         }) ||
                            !result->relations_.empty();
  return std::shared_ptr<const SpatialMemoryConstraintModel>(std::move(result));
}

llvm::Expected<PnrIndex>
SpatialMemoryConstraintModel::logicalBindingChoiceCapacity(
    PnrIndex binding) const {
  if (binding >= bindingRoots_.size())
    return runtimeInvalid("logical binding choice capacity is out of range");
  if (!hasAddressDomainRestrictions_)
    return checkedPnrIndex({"SpatialMemoryConstraintModel", "binding_choices",
                            "Action", PnrCapacityMeasure::Count},
                           targets_.size());
  const RootDomain &root = rootDomains_[bindingRoots_[binding]];
  const std::uint64_t segmentCount =
      root.addressesRestricted ? std::max<PnrIndex>(root.addressCount, 1) : 1;
  const std::uint64_t boundaries =
      2 + 2 * static_cast<std::uint64_t>(bindingRoots_.size());
  const std::uint64_t localTargets = targets_.empty() ? 0 : targets_.size() - 1;
  const auto segmentCapacity =
      llvm::checkedMulUnsigned(segmentCount, boundaries);
  if (!segmentCapacity)
    return runtimeInvalid("logical binding choice capacity overflows u64");
  const auto localCapacity =
      llvm::checkedMulUnsigned(localTargets, *segmentCapacity);
  if (!localCapacity)
    return runtimeInvalid("logical binding choice capacity overflows u64");
  const auto total = llvm::checkedAddUnsigned(*localCapacity, std::uint64_t{1});
  if (!total)
    return runtimeInvalid("logical binding choice capacity overflows u64");
  return checkedPnrIndex({"SpatialMemoryConstraintModel", "binding_choices",
                          "Action", PnrCapacityMeasure::Count},
                         *total);
}

llvm::Expected<PnrIndex>
SpatialMemoryConstraintModel::collectLogicalBindingChoices(
    PnrIndex binding,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> current,
    llvm::MutableArrayRef<SpatialLogicalMemoryBindingSelection> output) const {
  if (binding >= bindingRoots_.size() || current.size() != bindingRoots_.size())
    return runtimeInvalid("logical binding choice input is malformed");
  const auto extent = bindingExtents_[binding];
  const PnrIndex rootOrdinal = bindingRoots_[binding];
  const RootDomain &root = rootDomains_[rootOrdinal];
  const auto serviceDomain = llvm::ArrayRef(serviceDomainValues_)
                                 .slice(root.serviceOffset, root.serviceCount);
  const auto addressDomain = llvm::ArrayRef(addressDomainValues_)
                                 .slice(root.addressOffset, root.addressCount);
  std::size_t count = 0;
  const auto append = [&](PnrIndex target, std::uint64_t offset) {
    if (count == output.size())
      return false;
    output[count++] = {target, offset};
    return true;
  };

  for (PnrIndex targetOrdinal = 0; targetOrdinal < targets_.size();
       ++targetOrdinal) {
    const TargetProjection &target = targets_[targetOrdinal];
    if (target.service == getInvalidPnrIndex()) {
      if (!append(targetOrdinal, 0))
        return runtimeInvalid("logical binding choice storage is too small");
      continue;
    }
    if (root.servicesRestricted &&
        !std::binary_search(serviceDomain.begin(), serviceDomain.end(),
                            target.service))
      continue;
    if (!extent)
      continue;
    if (*extent == 0) {
      if (!append(targetOrdinal, 0))
        return runtimeInvalid("logical binding choice storage is too small");
      continue;
    }
    if (!hasAddressDomainRestrictions_) {
      std::uint64_t nextOffset = 0;
      for (PnrIndex other = 0; other < current.size(); ++other) {
        if (other == binding || current[other].target == getInvalidPnrIndex() ||
            current[other].target != targetOrdinal)
          continue;
        const auto otherExtent = bindingExtents_[other];
        if (!otherExtent)
          return runtimeInvalid(
              "local memory choice has no finite prior extent");
        const auto otherEnd = llvm::checkedAddUnsigned(
            current[other].physicalOffsetBytes, *otherExtent);
        if (!otherEnd)
          return runtimeInvalid("prior memory choice interval overflows u64");
        nextOffset = std::max(nextOffset, *otherEnd);
      }
      if (nextOffset <= targetSizes_[targetOrdinal] &&
          *extent <= targetSizes_[targetOrdinal] - nextOffset)
        if (!append(targetOrdinal, nextOffset))
          return runtimeInvalid("logical binding choice storage is too small");
      continue;
    }
    const auto targetEnd = llvm::checkedAddUnsigned(
        target.addressBaseBytes, targetSizes_[targetOrdinal]);
    if (!targetEnd)
      return runtimeInvalid("memory target address interval overflows u64");

    std::vector<std::pair<std::uint64_t, std::uint64_t>> segments;
    if (root.addressesRestricted) {
      for (const auto &allowed : addressDomain) {
        if (allowed.service != target.service)
          continue;
        const std::uint64_t lower =
            std::max(allowed.lower, target.addressBaseBytes);
        const std::uint64_t upper = std::min(allowed.upper, *targetEnd);
        if (lower < upper && *extent <= upper - lower)
          segments.emplace_back(lower, upper);
      }
    } else if (*extent <= *targetEnd - target.addressBaseBytes) {
      segments.emplace_back(target.addressBaseBytes, *targetEnd);
    }

    const auto overlaps = [&](std::uint64_t offset) {
      const std::uint64_t end = offset + *extent;
      for (PnrIndex other = 0; other < current.size(); ++other) {
        if (other == binding || current[other].target == getInvalidPnrIndex() ||
            current[other].target != targetOrdinal)
          continue;
        const auto otherExtent = bindingExtents_[other];
        if (!otherExtent)
          return true;
        const std::uint64_t otherBegin = current[other].physicalOffsetBytes;
        const std::uint64_t otherEnd = otherBegin + *otherExtent;
        if (offset < otherEnd && otherBegin < end)
          return true;
      }
      return false;
    };
    const auto appendPhysical = [&](std::uint64_t physical, std::uint64_t lower,
                                    std::uint64_t upper) -> llvm::Error {
      if (physical < lower || physical > upper - *extent)
        return llvm::Error::success();
      const std::uint64_t offset = physical - target.addressBaseBytes;
      if (overlaps(offset))
        return llvm::Error::success();
      if (!append(targetOrdinal, offset))
        return runtimeInvalid("logical binding choice storage is too small");
      return llvm::Error::success();
    };
    for (const auto &[lower, upper] : segments) {
      if (llvm::Error error = appendPhysical(lower, lower, upper))
        return std::move(error);
      if (llvm::Error error = appendPhysical(upper - *extent, lower, upper))
        return std::move(error);
      for (PnrIndex other = 0; other < current.size(); ++other) {
        if (other == binding || current[other].target == getInvalidPnrIndex() ||
            current[other].target != targetOrdinal)
          continue;
        const auto otherExtent = bindingExtents_[other];
        if (!otherExtent)
          continue;
        const std::uint64_t otherBegin = current[other].physicalOffsetBytes;
        const std::uint64_t otherEnd = otherBegin + *otherExtent;
        if (otherEnd <=
            std::numeric_limits<std::uint64_t>::max() - target.addressBaseBytes)
          if (llvm::Error error = appendPhysical(
                  target.addressBaseBytes + otherEnd, lower, upper))
            return std::move(error);
        if (otherBegin >= *extent)
          if (llvm::Error error = appendPhysical(
                  target.addressBaseBytes + otherBegin - *extent, lower, upper))
            return std::move(error);
      }
    }
  }
  auto selected = output.take_front(count);
  llvm::sort(selected, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.target, lhs.physicalOffsetBytes) <
           std::tie(rhs.target, rhs.physicalOffsetBytes);
  });
  const auto end = std::unique(
      selected.begin(), selected.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.target == rhs.target &&
               lhs.physicalOffsetBytes == rhs.physicalOffsetBytes;
      });
  return static_cast<PnrIndex>(end - selected.begin());
}

llvm::Error SpatialMemoryConstraintModel::projectRoot(
    PnrIndex root,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    std::vector<PnrIndex> &services,
    std::vector<SpatialMemoryAddressInterval> &addresses) const {
  services.clear();
  addresses.clear();
  if (root + 1 >= rootBindingOffsets_.size())
    return runtimeInvalid("memory projection root is out of range");
  for (PnrIndex binding :
       llvm::ArrayRef(rootBindings_)
           .slice(rootBindingOffsets_[root],
                  rootBindingOffsets_[root + 1] - rootBindingOffsets_[root])) {
    if (binding >= selections.size())
      return runtimeInvalid("memory projection binding is absent");
    const auto &selection = selections[binding];
    if (selection.target >= targets_.size())
      return runtimeInvalid("memory projection target is out of range");
    const TargetProjection &target = targets_[selection.target];
    if (target.service == getInvalidPnrIndex())
      continue;
    services.push_back(target.service);
    const auto extent = bindingExtents_[binding];
    if (!extent)
      return runtimeInvalid("local memory projection has no finite extent");
    if (*extent == 0)
      continue;
    const auto lower = llvm::checkedAddUnsigned(target.addressBaseBytes,
                                                selection.physicalOffsetBytes);
    if (!lower)
      return runtimeInvalid("memory projection address base overflows u64");
    const auto upper = llvm::checkedAddUnsigned(*lower, *extent);
    if (!upper)
      return runtimeInvalid("memory projection address interval overflows u64");
    addresses.push_back({target.service, *lower, *upper});
  }
  llvm::sort(services);
  services.erase(std::unique(services.begin(), services.end()), services.end());
  normalizeAddresses(addresses);
  return llvm::Error::success();
}

llvm::Error SpatialMemoryConstraintModel::verifyRootDomain(
    PnrIndex root,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections) const {
  std::vector<PnrIndex> services;
  std::vector<SpatialMemoryAddressInterval> addresses;
  if (llvm::Error error = projectRoot(root, selections, services, addresses))
    return error;
  const RootDomain &domain = rootDomains_[root];
  if (domain.servicesRestricted) {
    const auto allowed = llvm::ArrayRef(serviceDomainValues_)
                             .slice(domain.serviceOffset, domain.serviceCount);
    if (!std::includes(allowed.begin(), allowed.end(), services.begin(),
                       services.end()))
      return runtimeInvalid(
          "memory-bound service projection leaves its admissible domain");
  }
  if (domain.addressesRestricted) {
    const auto allowed = llvm::ArrayRef(addressDomainValues_)
                             .slice(domain.addressOffset, domain.addressCount);
    if (!addressSubset(addresses, allowed))
      return runtimeInvalid(
          "memory address projection leaves its admissible domain");
  }
  return llvm::Error::success();
}

llvm::Error SpatialMemoryConstraintModel::verifyRelation(
    const SpatialMemoryConstraintRelation &relation,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections) const {
  const auto members = llvm::ArrayRef(relationMembers_)
                           .slice(relation.memberOffset, relation.memberCount);
  std::vector<PnrIndex> referenceServices;
  std::vector<PnrIndex> currentServices;
  std::vector<SpatialMemoryAddressInterval> referenceAddresses;
  std::vector<SpatialMemoryAddressInterval> currentAddresses;
  if (members.empty())
    return runtimeInvalid("memory relation has no members");
  if (llvm::Error error = projectRoot(members.front(), selections,
                                      referenceServices, referenceAddresses))
    return error;
  for (PnrIndex root : members.drop_front()) {
    if (llvm::Error error =
            projectRoot(root, selections, currentServices, currentAddresses))
      return error;
    const bool equal =
        relation.projection == SpatialMemoryConstraintProjection::BoundServices
            ? referenceServices == currentServices
            : referenceAddresses == currentAddresses;
    const bool disjoint =
        relation.projection == SpatialMemoryConstraintProjection::BoundServices
            ? !servicesIntersect(referenceServices, currentServices)
            : !addressesIntersect(referenceAddresses, currentAddresses);
    if (relation.kind == SpatialMemoryConstraintRelationKind::Equal && !equal)
      return runtimeInvalid("memory projection equality is violated");
    if (relation.kind == SpatialMemoryConstraintRelationKind::Disjoint &&
        !disjoint)
      return runtimeInvalid("memory projection disjointness is violated");
    if (relation.kind == SpatialMemoryConstraintRelationKind::Disjoint) {
      referenceServices.insert(referenceServices.end(), currentServices.begin(),
                               currentServices.end());
      llvm::sort(referenceServices);
      referenceServices.erase(
          std::unique(referenceServices.begin(), referenceServices.end()),
          referenceServices.end());
      referenceAddresses.insert(referenceAddresses.end(),
                                currentAddresses.begin(),
                                currentAddresses.end());
      normalizeAddresses(referenceAddresses);
    }
  }
  return llvm::Error::success();
}

llvm::Error SpatialMemoryConstraintModel::verify(
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections) const {
  if (selections.size() != bindingRoots_.size())
    return runtimeInvalid("memory constraint selection shape is incomplete");
  if (!hasConstraints_)
    return llvm::Error::success();
  for (PnrIndex root = 0; root < rootDomains_.size(); ++root)
    if (llvm::Error error = verifyRootDomain(root, selections))
      return error;
  for (const auto &relation : relations_)
    if (llvm::Error error = verifyRelation(relation, selections))
      return error;
  return llvm::Error::success();
}
