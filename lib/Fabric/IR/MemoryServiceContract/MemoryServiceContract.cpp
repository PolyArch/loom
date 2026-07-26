#include "Fabric/IR/MemoryServiceContract.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryCapabilityRelation.h"
#include "MemoryServiceContractInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <system_error>

using dataflow::semantics::ServiceKind;

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

llvm::Error validateOwner(MemoryServiceOwnerKind owner) {
  switch (owner) {
  case MemoryServiceOwnerKind::Local:
  case MemoryServiceOwnerKind::System:
    return llvm::Error::success();
  }
  return invalid("memory service has an unknown owner kind");
}

llvm::Error validateReleaseVisibility(ReleaseVisibilityPoint point) {
  switch (point) {
  case ReleaseVisibilityPoint::AtLinearization:
  case ReleaseVisibilityPoint::ByRetirement:
    return llvm::Error::success();
  }
  return invalid("memory service has an unknown release visibility point");
}

llvm::Expected<std::vector<std::uint8_t>>
regionBehaviorKey(const MemoryServiceRegionDeclaration &region) {
  std::vector<std::uint8_t> bytes;
  auto appendU32 = [&](std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  auto appendU64 = [&](std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  switch (region.behavior) {
  case MemoryServiceRegionBehavior::Storage:
    appendU32(0);
    if (region.mmioAcceptedAccessDomain)
      return invalid("Storage memory region carries an MMIO access domain");
    break;
  case MemoryServiceRegionBehavior::Mmio: {
    appendU32(1);
    if (!region.mmioAcceptedAccessDomain)
      return invalid("MMIO memory region has no accepted access domain");
    auto access =
        encodeParameterizedMemoryAccessDomain(*region.mmioAcceptedAccessDomain);
    if (!access)
      return access.takeError();
    appendU64(access->size());
    bytes.insert(bytes.end(), access->begin(), access->end());
    break;
  }
  }
  return bytes;
}

struct IndexedRegion {
  MemoryServiceRegionDeclaration declaration;
  std::uint64_t originalOrdinal = 0;
  std::vector<std::uint8_t> behaviorKey;
};

struct NormalizedRegions {
  std::vector<MemoryServiceRegionDeclaration> regions;
  std::vector<std::uint64_t> oldToNew;
};

llvm::Expected<NormalizedRegions>
normalizeRegions(llvm::ArrayRef<MemoryServiceRegionDeclaration> regions) {
  if (regions.empty())
    return invalid("memory service has no region");
  std::vector<IndexedRegion> ordered;
  ordered.reserve(regions.size());
  for (std::uint64_t ordinal = 0; ordinal < regions.size(); ++ordinal) {
    const MemoryServiceRegionDeclaration &region = regions[ordinal];
    if (region.sizeBytes == 0)
      return invalid("memory service region has zero size");
    if (!llvm::checkedAddUnsigned(region.addressBaseBytes, region.sizeBytes))
      return invalid("memory service region address interval overflows u64");
    auto behavior = regionBehaviorKey(region);
    if (!behavior)
      return behavior.takeError();
    ordered.push_back({region, ordinal, std::move(*behavior)});
  }
  llvm::sort(ordered, [](const IndexedRegion &left,
                         const IndexedRegion &right) {
    if (left.declaration.addressBaseBytes != right.declaration.addressBaseBytes)
      return left.declaration.addressBaseBytes <
             right.declaration.addressBaseBytes;
    return left.behaviorKey < right.behaviorKey;
  });

  NormalizedRegions result;
  result.oldToNew.resize(regions.size());
  std::vector<std::vector<std::uint8_t>> behaviorKeys;
  for (IndexedRegion &entry : ordered) {
    const std::uint64_t begin = entry.declaration.addressBaseBytes;
    if (!result.regions.empty()) {
      MemoryServiceRegionDeclaration &previous = result.regions.back();
      const std::uint64_t previousEnd = *llvm::checkedAddUnsigned(
          previous.addressBaseBytes, previous.sizeBytes);
      if (begin < previousEnd)
        return invalid("memory service regions overlap");
      if (begin == previousEnd && behaviorKeys.back() == entry.behaviorKey) {
        previous.sizeBytes = *llvm::checkedAddUnsigned(
            previous.sizeBytes, entry.declaration.sizeBytes);
        result.oldToNew[entry.originalOrdinal] = result.regions.size() - 1;
        continue;
      }
    }
    result.oldToNew[entry.originalOrdinal] = result.regions.size();
    result.regions.push_back(std::move(entry.declaration));
    behaviorKeys.push_back(std::move(entry.behaviorKey));
  }
  return result;
}

bool clausesArePlain(const MemoryActorContractDomain &actors) {
  return llvm::all_of(actors.clauses(), [](const auto &clause) {
    return std::holds_alternative<LoadStorePlainContractClause>(clause);
  });
}

llvm::Error
validateConsistency(MemoryServiceOwnerKind owner,
                    const MemoryServiceCapabilityDeclaration &capability) {
  const bool plain = clausesArePlain(capability.actorContractDomain);
  if (std::holds_alternative<NoMemoryServiceConsistency>(
          capability.consistencyBinding)) {
    if (!plain)
      return invalid("non-plain memory capability has no consistency binding");
    return llvm::Error::success();
  }

  if (const auto *local = std::get_if<LocalProviderConsistency>(
          &capability.consistencyBinding)) {
    if (owner != MemoryServiceOwnerKind::Local)
      return invalid("System memory service uses LocalProvider consistency");
    if (llvm::Error error =
            validateReleaseVisibility(local->releaseVisibilityPoint))
      return error;
    if (const auto *bounded =
            std::get_if<LocalBoundedCompletionCycles>(&local->progress))
      if (bounded->maxIssueToRetireCycles == 0)
        return invalid("local memory service completion bound is zero");
    return llvm::Error::success();
  }

  if (owner != MemoryServiceOwnerKind::System)
    return invalid("local memory service uses a System consistency domain");
  return llvm::Error::success();
}

llvm::Error
validateCapability(MemoryServiceOwnerKind owner,
                   const ResourceContract &resourceContract,
                   llvm::ArrayRef<MemoryServiceRegionDeclaration> regions,
                   MemoryServiceCapabilityDeclaration &capability) {
  auto serviceKind = dataflow::semantics::getMemoryServiceKind(
      capability.actorContractDomain.actorSchema());
  if (!serviceKind)
    return serviceKind.takeError();
  const bool fence = *serviceKind == ServiceKind::MemoryFence;
  if (fence != !capability.accessDomain)
    return invalid("memory service access domain is absent exactly for fence");

  llvm::sort(capability.serviceRegionOrdinals);
  if (std::adjacent_find(capability.serviceRegionOrdinals.begin(),
                         capability.serviceRegionOrdinals.end()) !=
      capability.serviceRegionOrdinals.end())
    return invalid("memory service capability repeats a region ordinal");
  llvm::sort(capability.admissibleUsePatterns,
             [](UsePatternKey left, UsePatternKey right) {
               return left.ordinal() < right.ordinal();
             });
  if (capability.admissibleUsePatterns.empty())
    return invalid("memory service capability has no admissible use pattern");
  if (std::adjacent_find(capability.admissibleUsePatterns.begin(),
                         capability.admissibleUsePatterns.end()) !=
      capability.admissibleUsePatterns.end())
    return invalid("memory service capability repeats a use pattern");
  for (UsePatternKey pattern : capability.admissibleUsePatterns)
    if (pattern.ordinal() >= resourceContract.usePatternCount())
      return invalid(
          "memory service capability references an unknown use pattern");

  if (fence) {
    if (!capability.serviceRegionOrdinals.empty() ||
        capability.serviceBeatWidthBits != 0)
      return invalid("fence memory service capability has addressed facts");
  } else {
    if (capability.serviceRegionOrdinals.empty())
      return invalid("addressed memory service capability has no region");
    if (capability.serviceBeatWidthBits == 0)
      return invalid("addressed memory service capability has zero beat width");
    for (std::uint64_t ordinal : capability.serviceRegionOrdinals) {
      if (ordinal >= regions.size())
        return invalid(
            "memory service capability references an unknown region");
      const MemoryServiceRegionDeclaration &region = regions[ordinal];
      if (region.behavior != MemoryServiceRegionBehavior::Mmio)
        continue;
      auto covered = detail::memoryAccessDomainCovers(
          *region.mmioAcceptedAccessDomain, *capability.accessDomain);
      if (!covered)
        return covered.takeError();
      if (!*covered)
        return invalid("MMIO region does not accept the complete capability "
                       "access domain");
    }
  }
  return validateConsistency(owner, capability);
}

llvm::Expected<MemoryServiceContractDeclaration>
normalizeDeclaration(mlir::MLIRContext *context, MemoryServiceOwnerKind owner,
                     MemoryServiceContractDeclaration declaration) {
  if (!context)
    return invalid("memory service contract requires an MLIR context");
  if (llvm::Error error = validateOwner(owner))
    return std::move(error);
  auto normalizedRegions = normalizeRegions(declaration.regions);
  if (!normalizedRegions)
    return normalizedRegions.takeError();

  for (MemoryServiceCapabilityDeclaration &capability :
       declaration.capabilities) {
    llvm::sort(capability.serviceRegionOrdinals);
    if (std::adjacent_find(capability.serviceRegionOrdinals.begin(),
                           capability.serviceRegionOrdinals.end()) !=
        capability.serviceRegionOrdinals.end())
      return invalid("memory service capability repeats a region ordinal");
    for (std::uint64_t &ordinal : capability.serviceRegionOrdinals) {
      if (ordinal >= normalizedRegions->oldToNew.size())
        return invalid(
            "memory service capability references an unknown region");
      ordinal = normalizedRegions->oldToNew[ordinal];
    }
    llvm::sort(capability.serviceRegionOrdinals);
    capability.serviceRegionOrdinals.erase(
        std::unique(capability.serviceRegionOrdinals.begin(),
                    capability.serviceRegionOrdinals.end()),
        capability.serviceRegionOrdinals.end());
  }
  declaration.regions = std::move(normalizedRegions->regions);
  if (declaration.capabilities.empty())
    return invalid("memory service has no capability");
  for (MemoryServiceCapabilityDeclaration &capability :
       declaration.capabilities)
    if (llvm::Error error =
            validateCapability(owner, declaration.resourceContract,
                               declaration.regions, capability))
      return std::move(error);

  std::vector<bool> regionUsed(declaration.regions.size(), false);
  std::vector<bool> patternUsed(declaration.resourceContract.usePatternCount(),
                                false);
  for (const MemoryServiceCapabilityDeclaration &capability :
       declaration.capabilities) {
    for (std::uint64_t region : capability.serviceRegionOrdinals)
      regionUsed[region] = true;
    for (UsePatternKey pattern : capability.admissibleUsePatterns)
      patternUsed[pattern.ordinal()] = true;
  }
  if (llvm::is_contained(regionUsed, false))
    return invalid("memory service contract has an unreachable region");
  if (llvm::is_contained(patternUsed, false))
    return invalid("memory service contract has an unreachable use pattern");

  std::vector<detail::MemoryCapabilityRelationEntry> relation;
  relation.reserve(declaration.capabilities.size());
  std::vector<std::vector<std::uint8_t>> physicalFacts;
  physicalFacts.reserve(declaration.capabilities.size());
  for (const MemoryServiceCapabilityDeclaration &capability :
       declaration.capabilities) {
    auto facts = detail::encodeMemoryServiceCapabilityPhysicalFacts(
        {capability.serviceRegionOrdinals, capability.serviceBeatWidthBits,
         capability.consistencyBinding});
    if (!facts)
      return facts.takeError();
    physicalFacts.push_back(*facts);
    relation.push_back({capability.actorContractDomain, capability.accessDomain,
                        std::move(*facts), capability.admissibleUsePatterns});
  }

  for (std::size_t left = 0; left < relation.size(); ++left) {
    for (std::size_t right = left + 1; right < relation.size(); ++right) {
      if (physicalFacts[left] == physicalFacts[right])
        continue;
      auto overlap = detail::memoryCapabilityDomainsOverlap(
          relation[left].actorContractDomain, relation[left].accessDomain,
          relation[right].actorContractDomain, relation[right].accessDomain);
      if (!overlap)
        return overlap.takeError();
      if (*overlap)
        return invalid("overlapping memory service capabilities assign "
                       "different physical facts");
    }
  }

  auto normalized =
      detail::normalizeMemoryCapabilityRelation(context, relation);
  if (!normalized)
    return normalized.takeError();
  declaration.capabilities.clear();
  declaration.capabilities.reserve(normalized->size());
  for (detail::MemoryCapabilityRelationEntry &entry : *normalized) {
    auto facts =
        detail::decodeMemoryServiceCapabilityPhysicalFacts(entry.physicalFacts);
    if (!facts)
      return facts.takeError();
    MemoryServiceCapabilityDeclaration capability{
        std::move(entry.actorContractDomain),
        std::move(entry.accessDomain),
        std::move(facts->serviceRegionOrdinals),
        facts->serviceBeatWidthBits,
        std::move(entry.admissibleUsePatterns),
        std::move(facts->consistencyBinding)};
    declaration.capabilities.push_back(std::move(capability));
  }
  return declaration;
}

} // namespace

llvm::Expected<MemoryServiceContractRecord> MemoryServiceContractRecord::create(
    mlir::MLIRContext *context, MemoryServiceOwnerKind owner,
    MemoryServiceContractDeclaration declaration) {
  auto normalized =
      normalizeDeclaration(context, owner, std::move(declaration));
  if (!normalized)
    return normalized.takeError();
  return MemoryServiceContractRecord(std::move(*normalized));
}

llvm::Expected<MemoryServiceContractRecord>
MemoryServiceContractRecord::fromCanonical(
    mlir::MLIRContext *context, MemoryServiceOwnerKind owner,
    MemoryServiceContractDeclaration declaration) {
  MemoryServiceContractDeclaration original = declaration;
  auto normalized =
      normalizeDeclaration(context, owner, std::move(declaration));
  if (!normalized)
    return normalized.takeError();
  MemoryServiceContractRecord originalRecord(std::move(original));
  MemoryServiceContractRecord normalizedRecord(std::move(*normalized));
  auto originalBytes = encodeMemoryServiceContractRecord(originalRecord);
  auto normalizedBytes = encodeMemoryServiceContractRecord(normalizedRecord);
  if (!originalBytes)
    return originalBytes.takeError();
  if (!normalizedBytes)
    return normalizedBytes.takeError();
  if (*originalBytes != *normalizedBytes)
    return invalid("memory service contract record is not canonical");
  return normalizedRecord;
}

llvm::Error
validateLocalMemoryServiceCapacity(const MemoryServiceContractRecord &record,
                                   std::uint64_t capacityBytes) {
  if (capacityBytes == 0)
    return invalid("local memory service capacity is zero");
  for (const MemoryServiceRegionDeclaration &region : record.regions()) {
    auto end =
        llvm::checkedAddUnsigned(region.addressBaseBytes, region.sizeBytes);
    if (!end || *end > capacityBytes)
      return invalid("local memory service region exceeds capacity");
  }
  return llvm::Error::success();
}

} // namespace fabric
