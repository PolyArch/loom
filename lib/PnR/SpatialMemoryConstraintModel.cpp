#include "SpatialMemoryConstraintModel.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

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

char SpatialMemoryConstraintSolveFailure::ID;

void SpatialMemoryConstraintSolveFailure::log(llvm::raw_ostream &stream) const {
  stream << "Spatial memory relation closure exhausted its assignment work "
            "limit";
}

std::error_code
SpatialMemoryConstraintSolveFailure::convertToErrorCode() const {
  return std::make_error_code(std::errc::resource_unavailable_try_again);
}

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

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

struct SpatialMemoryConstraintScratch::Storage final {
  std::vector<SpatialLogicalMemoryBindingSelection> workingSelections;
  std::vector<SpatialLogicalMemoryBindingSelection> fixedSelections;
  std::vector<SpatialLogicalMemoryBindingSelection> choices;
  std::vector<std::uint8_t> fixedMarks;
  std::vector<std::uint8_t> closureRootMarks;
  std::vector<PnrIndex> closureRoots;
  std::vector<PnrIndex> closureBindings;
  std::vector<PnrIndex> servicesA;
  std::vector<PnrIndex> servicesB;
  std::vector<SpatialMemoryAddressInterval> addressesA;
  std::vector<SpatialMemoryAddressInterval> addressesB;
  std::uint64_t assignmentLimit = 0;
  std::uint64_t assignmentAttempts = 0;
  const SpatialMemoryConstraintModel *preparedModel = nullptr;
};

SpatialMemoryConstraintScratch::SpatialMemoryConstraintScratch()
    : storage_(std::make_unique<Storage>()) {}

SpatialMemoryConstraintScratch::~SpatialMemoryConstraintScratch() = default;

llvm::ArrayRef<SpatialLogicalMemoryBindingSelection>
SpatialMemoryConstraintScratch::solution() const {
  return storage_->workingSelections;
}

std::size_t SpatialMemoryConstraintScratch::retainedStorageBytes() const {
  return retainedBytes(storage_->workingSelections) +
         retainedBytes(storage_->fixedSelections) +
         retainedBytes(storage_->choices) +
         retainedBytes(storage_->fixedMarks) +
         retainedBytes(storage_->closureRootMarks) +
         retainedBytes(storage_->closureRoots) +
         retainedBytes(storage_->closureBindings) +
         retainedBytes(storage_->servicesA) +
         retainedBytes(storage_->servicesB) +
         retainedBytes(storage_->addressesA) +
         retainedBytes(storage_->addressesB);
}

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
      if (result->relationMembers_.size() > getPnrIndexMax() ||
          relation.memberCount >
              getPnrIndexMax() - result->relationMembers_.size())
        return freezeInvalid(shard.projection(),
                             "relation members overflow PnrIndex");
      if (result->relations_.size() == getPnrIndexMax())
        return freezeInvalid(shard.projection(),
                             "relation count overflows PnrIndex");
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

  std::vector<std::vector<PnrIndex>> rootRelations(rootEntities.size());
  for (PnrIndex relation = 0; relation < result->relations_.size();
       ++relation) {
    const SpatialMemoryConstraintRelation &record =
        result->relations_[relation];
    for (PnrIndex root : llvm::ArrayRef(result->relationMembers_)
                             .slice(record.memberOffset, record.memberCount))
      rootRelations[root].push_back(relation);
  }
  result->rootRelations_.reserve(result->relationMembers_.size());
  result->rootRelationOffsets_.reserve(rootEntities.size() + 1);
  result->rootRelationOffsets_.push_back(0);
  for (std::vector<PnrIndex> &incidence : rootRelations) {
    llvm::sort(incidence);
    incidence.erase(std::unique(incidence.begin(), incidence.end()),
                    incidence.end());
    if (result->rootRelations_.size() > getPnrIndexMax() ||
        incidence.size() > getPnrIndexMax() - result->rootRelations_.size())
      return freezeInvalid(Projection::MemoryBoundServices,
                           "root relation incidence overflows PnrIndex");
    result->rootRelations_.insert(result->rootRelations_.end(),
                                  incidence.begin(), incidence.end());
    result->rootRelationOffsets_.push_back(
        static_cast<PnrIndex>(result->rootRelations_.size()));
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
    const auto visitSegment = [&](std::uint64_t lower,
                                  std::uint64_t upper) -> llvm::Error {
      if (llvm::Error error = appendPhysical(lower, lower, upper))
        return error;
      if (llvm::Error error = appendPhysical(upper - *extent, lower, upper))
        return error;
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
            return error;
        if (otherBegin >= *extent)
          if (llvm::Error error = appendPhysical(
                  target.addressBaseBytes + otherBegin - *extent, lower, upper))
            return error;
      }
      return llvm::Error::success();
    };
    if (root.addressesRestricted) {
      for (const auto &allowed : addressDomain) {
        if (allowed.service != target.service)
          continue;
        const std::uint64_t lower =
            std::max(allowed.lower, target.addressBaseBytes);
        const std::uint64_t upper = std::min(allowed.upper, *targetEnd);
        if (lower < upper && *extent <= upper - lower)
          if (llvm::Error error = visitSegment(lower, upper))
            return error;
      }
    } else if (*extent <= *targetEnd - target.addressBaseBytes) {
      if (llvm::Error error = visitSegment(target.addressBaseBytes, *targetEnd))
        return error;
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

llvm::Error SpatialMemoryConstraintModel::prepareScratch(
    SpatialMemoryConstraintScratch &scratch) const {
  auto &storage = *scratch.storage_;
  PnrIndex choiceCapacity = 0;
  for (PnrIndex binding = 0; binding < bindingRoots_.size(); ++binding) {
    auto capacity = logicalBindingChoiceCapacity(binding);
    if (!capacity)
      return capacity.takeError();
    choiceCapacity = std::max(choiceCapacity, *capacity);
  }
  storage.workingSelections.resize(bindingRoots_.size());
  storage.fixedSelections.resize(bindingRoots_.size());
  storage.choices.resize(choiceCapacity);
  storage.fixedMarks.resize(bindingRoots_.size());
  storage.closureRootMarks.resize(rootDomains_.size());
  storage.closureRoots.clear();
  storage.closureRoots.reserve(rootDomains_.size());
  storage.closureBindings.clear();
  storage.closureBindings.reserve(bindingRoots_.size());
  storage.servicesA.clear();
  storage.servicesA.reserve(bindingRoots_.size());
  storage.servicesB.clear();
  storage.servicesB.reserve(bindingRoots_.size());
  storage.addressesA.clear();
  storage.addressesA.reserve(bindingRoots_.size());
  storage.addressesB.clear();
  storage.addressesB.reserve(bindingRoots_.size());
  storage.preparedModel = this;
  return llvm::Error::success();
}

llvm::Expected<bool> SpatialMemoryConstraintModel::solveCanonicalClosure(
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> current,
    llvm::ArrayRef<PnrIndex> fixedBindings,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> fixedSelections,
    std::uint64_t assignmentLimit,
    llvm::function_ref<llvm::Expected<bool>(PnrIndex, PnrIndex)>
        targetSupported,
    SpatialMemoryConstraintScratch &scratch) const {
  auto &storage = *scratch.storage_;
  if (storage.preparedModel != this ||
      storage.workingSelections.size() != bindingRoots_.size())
    return runtimeInvalid("memory constraint scratch is not prepared");
  if (current.size() != bindingRoots_.size() ||
      fixedBindings.size() != fixedSelections.size() || fixedBindings.empty() ||
      assignmentLimit == 0)
    return runtimeInvalid("memory closure input is malformed");

  storage.assignmentLimit = assignmentLimit;
  storage.assignmentAttempts = 0;
  std::copy(current.begin(), current.end(), storage.workingSelections.begin());
  std::fill(storage.fixedMarks.begin(), storage.fixedMarks.end(), 0);
  std::fill(storage.closureRootMarks.begin(), storage.closureRootMarks.end(),
            0);
  storage.closureRoots.clear();
  storage.closureBindings.clear();
  for (auto [ordinal, binding] : llvm::enumerate(fixedBindings)) {
    if (binding >= bindingRoots_.size() ||
        fixedSelections[ordinal].target >= targets_.size())
      return runtimeInvalid("memory closure fixed binding is out of range");
    if (storage.fixedMarks[binding] &&
        (storage.fixedSelections[binding].target !=
             fixedSelections[ordinal].target ||
         storage.fixedSelections[binding].physicalOffsetBytes !=
             fixedSelections[ordinal].physicalOffsetBytes))
      return runtimeInvalid("memory closure fixes one binding twice");
    storage.fixedMarks[binding] = 1;
    storage.fixedSelections[binding] = fixedSelections[ordinal];
    const PnrIndex root = bindingRoots_[binding];
    if (!storage.closureRootMarks[root]) {
      storage.closureRootMarks[root] = 1;
      storage.closureRoots.push_back(root);
    }
  }
  for (std::size_t cursor = 0; cursor < storage.closureRoots.size(); ++cursor) {
    const PnrIndex root = storage.closureRoots[cursor];
    for (PnrIndex relation : llvm::ArrayRef(rootRelations_)
                                 .slice(rootRelationOffsets_[root],
                                        rootRelationOffsets_[root + 1] -
                                            rootRelationOffsets_[root])) {
      const SpatialMemoryConstraintRelation &record = relations_[relation];
      for (PnrIndex member :
           llvm::ArrayRef(relationMembers_)
               .slice(record.memberOffset, record.memberCount))
        if (!storage.closureRootMarks[member]) {
          storage.closureRootMarks[member] = 1;
          storage.closureRoots.push_back(member);
        }
    }
  }
  for (PnrIndex binding = 0; binding < bindingRoots_.size(); ++binding)
    if (storage.closureRootMarks[bindingRoots_[binding]])
      storage.workingSelections[binding] = {getInvalidPnrIndex(), 0};
  for (PnrIndex binding = 0; binding < bindingRoots_.size(); ++binding)
    if (storage.closureRootMarks[bindingRoots_[binding]] &&
        storage.fixedMarks[binding])
      storage.closureBindings.push_back(binding);
  for (PnrIndex binding = 0; binding < bindingRoots_.size(); ++binding)
    if (storage.closureRootMarks[bindingRoots_[binding]] &&
        !storage.fixedMarks[binding]) {
      storage.closureBindings.push_back(binding);
    }
  return solveClosureAt(0, current, targetSupported, scratch);
}

llvm::Expected<bool> SpatialMemoryConstraintModel::solveClosureAt(
    std::size_t cursor,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> current,
    llvm::function_ref<llvm::Expected<bool>(PnrIndex, PnrIndex)>
        targetSupported,
    SpatialMemoryConstraintScratch &scratch) const {
  auto &storage = *scratch.storage_;
  if (cursor == storage.closureBindings.size()) {
    return constraintsSatisfied(storage.workingSelections, scratch);
  }
  const PnrIndex binding = storage.closureBindings[cursor];
  const auto collect = [&]() -> llvm::Expected<PnrIndex> {
    return collectLogicalBindingChoices(binding, storage.workingSelections,
                                        storage.choices);
  };
  const auto attempt = [&](SpatialLogicalMemoryBindingSelection selection)
      -> llvm::Expected<bool> {
    if (storage.assignmentAttempts == storage.assignmentLimit)
      return llvm::make_error<SpatialMemoryConstraintSolveFailure>();
    ++storage.assignmentAttempts;
    auto supported = targetSupported(binding, selection.target);
    if (!supported)
      return supported.takeError();
    if (!*supported)
      return false;
    storage.workingSelections[binding] = selection;
    auto consistent = partialConstraintsSatisfied(
        binding, storage.workingSelections, scratch);
    if (!consistent) {
      storage.workingSelections[binding] = {getInvalidPnrIndex(), 0};
      return consistent.takeError();
    }
    if (!*consistent) {
      storage.workingSelections[binding] = {getInvalidPnrIndex(), 0};
      return false;
    }
    auto solved = solveClosureAt(cursor + 1, current, targetSupported, scratch);
    if (!solved || !*solved)
      storage.workingSelections[binding] = {getInvalidPnrIndex(), 0};
    return solved;
  };

  if (storage.fixedMarks[binding]) {
    auto count = collect();
    if (!count)
      return count.takeError();
    const SpatialLogicalMemoryBindingSelection fixed =
        storage.fixedSelections[binding];
    const bool present = llvm::any_of(
        llvm::ArrayRef(storage.choices).take_front(*count),
        [&](const SpatialLogicalMemoryBindingSelection &choice) {
          return choice.target == fixed.target &&
                 choice.physicalOffsetBytes == fixed.physicalOffsetBytes;
        });
    if (!present)
      return false;
    return attempt(fixed);
  }

  auto count = collect();
  if (!count)
    return count.takeError();
  const SpatialLogicalMemoryBindingSelection preferred = current[binding];
  const bool hasPreferred = llvm::any_of(
      llvm::ArrayRef(storage.choices).take_front(*count),
      [&](const SpatialLogicalMemoryBindingSelection &choice) {
        return choice.target == preferred.target &&
               choice.physicalOffsetBytes == preferred.physicalOffsetBytes;
      });
  if (hasPreferred) {
    auto solved = attempt(preferred);
    if (!solved || *solved)
      return solved;
  }
  for (PnrIndex ordinal = 0;; ++ordinal) {
    count = collect();
    if (!count)
      return count.takeError();
    if (ordinal >= *count)
      break;
    const SpatialLogicalMemoryBindingSelection choice =
        storage.choices[ordinal];
    if (hasPreferred && choice.target == preferred.target &&
        choice.physicalOffsetBytes == preferred.physicalOffsetBytes)
      continue;
    auto solved = attempt(choice);
    if (!solved || *solved)
      return solved;
  }
  return false;
}

llvm::Error SpatialMemoryConstraintModel::projectRoot(
    PnrIndex root,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    std::vector<PnrIndex> &services,
    std::vector<SpatialMemoryAddressInterval> &addresses) const {
  auto complete = projectAssignedRoot(root, selections, services, addresses);
  if (!complete)
    return complete.takeError();
  if (!*complete)
    return runtimeInvalid("memory projection binding is absent");
  return llvm::Error::success();
}

llvm::Expected<bool> SpatialMemoryConstraintModel::projectAssignedRoot(
    PnrIndex root,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    std::vector<PnrIndex> &services,
    std::vector<SpatialMemoryAddressInterval> &addresses) const {
  services.clear();
  addresses.clear();
  if (root + 1 >= rootBindingOffsets_.size())
    return runtimeInvalid("memory projection root is out of range");
  bool complete = true;
  for (PnrIndex binding :
       llvm::ArrayRef(rootBindings_)
           .slice(rootBindingOffsets_[root],
                  rootBindingOffsets_[root + 1] - rootBindingOffsets_[root])) {
    if (binding >= selections.size())
      return runtimeInvalid("memory projection binding is absent");
    const auto &selection = selections[binding];
    if (selection.target == getInvalidPnrIndex()) {
      complete = false;
      continue;
    }
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
  return complete;
}

llvm::Expected<bool> SpatialMemoryConstraintModel::rootDomainSatisfied(
    PnrIndex root,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    SpatialMemoryConstraintScratch &scratch) const {
  auto &storage = *scratch.storage_;
  if (llvm::Error error =
          projectRoot(root, selections, storage.servicesA, storage.addressesA))
    return std::move(error);
  const RootDomain &domain = rootDomains_[root];
  if (domain.servicesRestricted) {
    const auto allowed = llvm::ArrayRef(serviceDomainValues_)
                             .slice(domain.serviceOffset, domain.serviceCount);
    if (!std::includes(allowed.begin(), allowed.end(),
                       storage.servicesA.begin(), storage.servicesA.end()))
      return false;
  }
  if (domain.addressesRestricted) {
    const auto allowed = llvm::ArrayRef(addressDomainValues_)
                             .slice(domain.addressOffset, domain.addressCount);
    if (!addressSubset(storage.addressesA, allowed))
      return false;
  }
  return true;
}

llvm::Expected<bool> SpatialMemoryConstraintModel::rootDomainPartiallySatisfied(
    PnrIndex root,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    SpatialMemoryConstraintScratch &scratch) const {
  auto &storage = *scratch.storage_;
  auto complete = projectAssignedRoot(root, selections, storage.servicesA,
                                      storage.addressesA);
  if (!complete)
    return complete.takeError();
  const RootDomain &domain = rootDomains_[root];
  if (domain.servicesRestricted) {
    const auto allowed = llvm::ArrayRef(serviceDomainValues_)
                             .slice(domain.serviceOffset, domain.serviceCount);
    if (!std::includes(allowed.begin(), allowed.end(),
                       storage.servicesA.begin(), storage.servicesA.end()))
      return false;
  }
  if (domain.addressesRestricted) {
    const auto allowed = llvm::ArrayRef(addressDomainValues_)
                             .slice(domain.addressOffset, domain.addressCount);
    if (!addressSubset(storage.addressesA, allowed))
      return false;
  }
  return true;
}

llvm::Expected<bool> SpatialMemoryConstraintModel::relationSatisfied(
    const SpatialMemoryConstraintRelation &relation,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    SpatialMemoryConstraintScratch &scratch) const {
  auto &storage = *scratch.storage_;
  const auto members = llvm::ArrayRef(relationMembers_)
                           .slice(relation.memberOffset, relation.memberCount);
  if (members.empty())
    return runtimeInvalid("memory relation has no members");
  if (llvm::Error error = projectRoot(members.front(), selections,
                                      storage.servicesA, storage.addressesA))
    return error;
  for (PnrIndex root : members.drop_front()) {
    if (llvm::Error error = projectRoot(root, selections, storage.servicesB,
                                        storage.addressesB))
      return error;
    const bool equal =
        relation.projection == SpatialMemoryConstraintProjection::BoundServices
            ? storage.servicesA == storage.servicesB
            : storage.addressesA == storage.addressesB;
    const bool disjoint =
        relation.projection == SpatialMemoryConstraintProjection::BoundServices
            ? !servicesIntersect(storage.servicesA, storage.servicesB)
            : !addressesIntersect(storage.addressesA, storage.addressesB);
    if (relation.kind == SpatialMemoryConstraintRelationKind::Equal && !equal)
      return false;
    if (relation.kind == SpatialMemoryConstraintRelationKind::Disjoint &&
        !disjoint)
      return false;
    if (relation.kind == SpatialMemoryConstraintRelationKind::Disjoint) {
      storage.servicesA.insert(storage.servicesA.end(),
                               storage.servicesB.begin(),
                               storage.servicesB.end());
      llvm::sort(storage.servicesA);
      storage.servicesA.erase(
          std::unique(storage.servicesA.begin(), storage.servicesA.end()),
          storage.servicesA.end());
      storage.addressesA.insert(storage.addressesA.end(),
                                storage.addressesB.begin(),
                                storage.addressesB.end());
      normalizeAddresses(storage.addressesA);
    }
  }
  return true;
}

llvm::Expected<bool> SpatialMemoryConstraintModel::relationPartiallySatisfied(
    const SpatialMemoryConstraintRelation &relation,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    SpatialMemoryConstraintScratch &scratch) const {
  auto &storage = *scratch.storage_;
  const auto members = llvm::ArrayRef(relationMembers_)
                           .slice(relation.memberOffset, relation.memberCount);
  if (members.empty())
    return runtimeInvalid("memory relation has no members");

  if (relation.kind == SpatialMemoryConstraintRelationKind::Disjoint) {
    storage.servicesA.clear();
    storage.addressesA.clear();
    for (PnrIndex root : members) {
      auto complete = projectAssignedRoot(root, selections, storage.servicesB,
                                          storage.addressesB);
      if (!complete)
        return complete.takeError();
      const bool intersects =
          relation.projection ==
                  SpatialMemoryConstraintProjection::BoundServices
              ? servicesIntersect(storage.servicesA, storage.servicesB)
              : addressesIntersect(storage.addressesA, storage.addressesB);
      if (intersects)
        return false;
      storage.servicesA.insert(storage.servicesA.end(),
                               storage.servicesB.begin(),
                               storage.servicesB.end());
      llvm::sort(storage.servicesA);
      storage.servicesA.erase(
          std::unique(storage.servicesA.begin(), storage.servicesA.end()),
          storage.servicesA.end());
      storage.addressesA.insert(storage.addressesA.end(),
                                storage.addressesB.begin(),
                                storage.addressesB.end());
      normalizeAddresses(storage.addressesA);
    }
    return true;
  }

  std::optional<PnrIndex> completeReference;
  for (PnrIndex root : members) {
    auto complete = projectAssignedRoot(root, selections, storage.servicesB,
                                        storage.addressesB);
    if (!complete)
      return complete.takeError();
    if (*complete) {
      completeReference = root;
      storage.servicesA = storage.servicesB;
      storage.addressesA = storage.addressesB;
      break;
    }
  }
  if (!completeReference)
    return true;
  for (PnrIndex root : members) {
    auto complete = projectAssignedRoot(root, selections, storage.servicesB,
                                        storage.addressesB);
    if (!complete)
      return complete.takeError();
    const bool supported =
        relation.projection == SpatialMemoryConstraintProjection::BoundServices
            ? (*complete ? storage.servicesA == storage.servicesB
                         : std::includes(storage.servicesA.begin(),
                                         storage.servicesA.end(),
                                         storage.servicesB.begin(),
                                         storage.servicesB.end()))
            : (*complete
                   ? storage.addressesA == storage.addressesB
                   : addressSubset(storage.addressesB, storage.addressesA));
    if (!supported)
      return false;
  }
  return true;
}

llvm::Expected<bool> SpatialMemoryConstraintModel::partialConstraintsSatisfied(
    PnrIndex binding,
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    SpatialMemoryConstraintScratch &scratch) const {
  if (binding >= bindingRoots_.size())
    return runtimeInvalid("memory constraint binding is out of range");
  const PnrIndex root = bindingRoots_[binding];
  auto domain = rootDomainPartiallySatisfied(root, selections, scratch);
  if (!domain)
    return domain.takeError();
  if (!*domain)
    return false;
  for (PnrIndex relation :
       llvm::ArrayRef(rootRelations_)
           .slice(rootRelationOffsets_[root], rootRelationOffsets_[root + 1] -
                                                  rootRelationOffsets_[root])) {
    auto satisfied =
        relationPartiallySatisfied(relations_[relation], selections, scratch);
    if (!satisfied)
      return satisfied.takeError();
    if (!*satisfied)
      return false;
  }
  return true;
}

llvm::Expected<bool> SpatialMemoryConstraintModel::constraintsSatisfied(
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    SpatialMemoryConstraintScratch &scratch) const {
  for (PnrIndex root = 0; root < rootDomains_.size(); ++root) {
    auto satisfied = rootDomainSatisfied(root, selections, scratch);
    if (!satisfied)
      return satisfied.takeError();
    if (!*satisfied)
      return false;
  }
  for (const SpatialMemoryConstraintRelation &relation : relations_) {
    auto satisfied = relationSatisfied(relation, selections, scratch);
    if (!satisfied)
      return satisfied.takeError();
    if (!*satisfied)
      return false;
  }
  return true;
}

llvm::Error SpatialMemoryConstraintModel::verify(
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections) const {
  SpatialMemoryConstraintScratch scratch;
  if (llvm::Error error = prepareScratch(scratch))
    return error;
  return verify(selections, scratch);
}

llvm::Error SpatialMemoryConstraintModel::verify(
    llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
    SpatialMemoryConstraintScratch &scratch) const {
  if (selections.size() != bindingRoots_.size())
    return runtimeInvalid("memory constraint selection shape is incomplete");
  if (scratch.storage_->preparedModel != this)
    return runtimeInvalid("memory constraint scratch is not prepared");
  if (!hasConstraints_)
    return llvm::Error::success();
  auto satisfied = constraintsSatisfied(selections, scratch);
  if (!satisfied)
    return satisfied.takeError();
  if (!*satisfied)
    return runtimeInvalid("memory projection constraints are violated");
  return llvm::Error::success();
}
