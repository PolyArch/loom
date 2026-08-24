#include "Fabric/Artifact/FabricSystemReferenceRemapper.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include <set>
#include <type_traits>
#include <variant>

namespace loom::fabric {

llvm::Error FabricSystemReferenceRemapper::missing(llvm::StringRef kind) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      llvm::Twine("fabric_system_reference_remap_invalid: source ") + kind +
          " has no exact child correspondence");
}

llvm::Expected<FabricSystemReferenceRemapper>
FabricSystemReferenceRemapper::get(
    llvm::ArrayRef<FabricSystemEntityCorrespondence> entities,
    llvm::ArrayRef<FabricSystemTransferPatternCorrespondence>
        transferPatterns) {
  std::map<EntityKey, FabricEntityId> entityMap;
  std::set<EntityKey> targetEntities;
  for (const auto &entry : entities) {
    if (entry.source.kind != entry.target.kind)
      return missing("entity changes kind");
    if (!entityMap.emplace(EntityKey{entry.source.kind, entry.source.id},
                           entry.target.id)
             .second)
      return missing("entity is duplicated");
    if (!targetEntities.insert({entry.target.kind, entry.target.id}).second)
      return missing("entity target is duplicated");
  }
  std::map<std::vector<std::uint8_t>, FabricTransferPatternRef> patternMap;
  std::set<std::vector<std::uint8_t>> targetPatterns;
  for (const auto &entry : transferPatterns) {
    if (!patternMap.emplace(canonicalFabricBytes(entry.source), entry.target)
             .second)
      return missing("transfer pattern is duplicated");
    if (!targetPatterns.insert(canonicalFabricBytes(entry.target)).second)
      return missing("transfer-pattern target is duplicated");
  }
  return FabricSystemReferenceRemapper(std::move(entityMap),
                                       std::move(patternMap));
}

llvm::Expected<FabricTransportEndpointOwnerRef>
FabricSystemReferenceRemapper::remap(
    const FabricTransportEndpointOwnerRef &reference) const {
  return std::visit(
      [&](const auto &payload)
          -> llvm::Expected<FabricTransportEndpointOwnerRef> {
        auto mapped = remap(payload);
        if (!mapped)
          return mapped.takeError();
        return FabricTransportEndpointOwnerRef::of(std::move(*mapped));
      },
      reference.payload);
}

llvm::Expected<FabricMemoryEndpointOwnerRef>
FabricSystemReferenceRemapper::remap(
    const FabricMemoryEndpointOwnerRef &reference) const {
  return std::visit(
      [&](const auto &payload) -> llvm::Expected<FabricMemoryEndpointOwnerRef> {
        auto mapped = remap(payload);
        if (!mapped)
          return mapped.takeError();
        return FabricMemoryEndpointOwnerRef::of(std::move(*mapped));
      },
      reference.payload);
}

llvm::Expected<FabricInventoryOwnerRef>
FabricSystemReferenceRemapper::remap(
    const FabricInventoryOwnerRef &reference) const {
  return std::visit(
      [&](const auto &payload) -> llvm::Expected<FabricInventoryOwnerRef> {
        auto mapped = remap(payload);
        if (!mapped)
          return mapped.takeError();
        return FabricInventoryOwnerRef::of(std::move(*mapped));
      },
      reference.payload);
}

llvm::Expected<FabricHardwareDomainMemberRef>
FabricSystemReferenceRemapper::remap(
    const FabricHardwareDomainMemberRef &reference) const {
  return std::visit(
      [&](const auto &payload)
          -> llvm::Expected<FabricHardwareDomainMemberRef> {
        auto mapped = remap(payload);
        if (!mapped)
          return mapped.takeError();
        return FabricHardwareDomainMemberRef::create(*mapped);
      },
      reference.payload());
}

llvm::Expected<FabricMemoryServiceRef>
FabricSystemReferenceRemapper::remap(
    const FabricMemoryServiceRef &reference) const {
  return std::visit(
      [&](const auto &payload) -> llvm::Expected<FabricMemoryServiceRef> {
        auto mapped = remap(payload);
        if (!mapped)
          return mapped.takeError();
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<Payload, FabricMemoryOccurrenceRef>)
          return FabricMemoryServiceRef::local(std::move(*mapped));
        else
          return FabricMemoryServiceRef::system(std::move(*mapped));
      },
      reference.payload);
}

llvm::Expected<FabricTransferPatternRef>
FabricSystemReferenceRemapper::remap(
    const FabricTransferPatternRef &reference) const {
  auto found = transferPatterns_.find(canonicalFabricBytes(reference));
  if (found == transferPatterns_.end())
    return missing("transfer pattern");
  return found->second;
}

llvm::Expected<FabricPhysicalTraversalRef>
FabricSystemReferenceRemapper::remap(
    const FabricPhysicalTraversalRef &reference) const {
  return std::visit(
      [&](const auto &payload) -> llvm::Expected<FabricPhysicalTraversalRef> {
        auto mapped = remap(payload);
        if (!mapped)
          return mapped.takeError();
        return FabricPhysicalTraversalRef::of(std::move(*mapped));
      },
      reference.payload);
}

} // namespace loom::fabric
