#include "Mapping/Tech/TechMappingHardwareDemand.h"

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace {

constexpr llvm::StringLiteral feedbackSchema =
    "loom.mapping.tech_compute_context_hall_feedback.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tech_mapping_hardware_demand_invalid: " +
                                     message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("payload is truncated");
  std::uint64_t value = 0;
  for (unsigned index = 0; index != 8; ++index)
    value = (value << 8) | bytes[offset++];
  return value;
}

std::vector<std::uint8_t>
contextKey(const loom::fabric::InstructionContextRef &context) {
  return loom::fabric::canonicalFabricBytes(context);
}

bool contextLess(const loom::fabric::InstructionContextRef &lhs,
                 const loom::fabric::InstructionContextRef &rhs) {
  return contextKey(lhs) < contextKey(rhs);
}

llvm::Expected<std::vector<loom::fabric::InstructionContextRef>>
deriveCompatibleContexts(loom::fabric::FabricFuCapabilityTemplateRef capability,
                         const loom::fabric::FabricArtifactView &fabric) {
  if (llvm::Error error = loom::fabric::validateFabricRef(fabric, capability))
    return std::move(error);
  auto placements =
      deriveSpatialComputeContextPlacementDomain(capability, fabric);
  if (!placements)
    return placements.takeError();
  std::vector<loom::fabric::InstructionContextRef> contexts;
  for (const auto &placement : *placements)
    contexts.insert(contexts.end(), placement.contexts.begin(),
                    placement.contexts.end());
  llvm::sort(contexts, contextLess);
  contexts.erase(std::unique(contexts.begin(), contexts.end()), contexts.end());
  return contexts;
}

} // namespace

llvm::Expected<TechMappingComputeContextHallDeficit>
TechMappingComputeContextHallDeficit::get(
    std::uint64_t coverDemandCount, std::uint64_t coverMaximumMatching,
    llvm::ArrayRef<TechMappingComputeContextHallDemandGroup> inputGroups) {
  if (coverDemandCount == 0 || coverMaximumMatching >= coverDemandCount)
    return invalid("cover matching is not deficient");
  if (inputGroups.empty())
    return invalid("Hall demand group set is empty");

  std::vector<TechMappingComputeContextHallDemandGroup> groups(
      inputGroups.begin(), inputGroups.end());
  llvm::sort(groups, [](const auto &lhs, const auto &rhs) {
    return loom::fabric::canonicalFabricBytes(lhs.capability) <
           loom::fabric::canonicalFabricBytes(rhs.capability);
  });
  std::uint64_t hallDemandCount = 0;
  std::map<std::vector<std::uint8_t>, loom::fabric::InstructionContextRef>
      contextUnion;
  std::optional<std::vector<std::uint8_t>> previousCapability;
  for (TechMappingComputeContextHallDemandGroup &group : groups) {
    const std::vector<std::uint8_t> capability =
        loom::fabric::canonicalFabricBytes(group.capability);
    if (previousCapability && *previousCapability == capability)
      return invalid("Hall demand groups contain a duplicate capability");
    previousCapability = capability;
    if (group.demandCount == 0)
      return invalid("Hall demand multiplicity is zero");
    if (group.demandCount >
        std::numeric_limits<std::uint64_t>::max() - hallDemandCount)
      return invalid("Hall demand count overflows u64");
    hallDemandCount += group.demandCount;
    llvm::sort(group.compatibleContexts, contextLess);
    if (std::adjacent_find(group.compatibleContexts.begin(),
                           group.compatibleContexts.end()) !=
        group.compatibleContexts.end())
      return invalid("compatible context set contains a duplicate");
    for (const auto &context : group.compatibleContexts)
      contextUnion.emplace(contextKey(context), context);
  }
  if (hallDemandCount > coverDemandCount)
    return invalid("Hall demand set exceeds its cover");
  const std::uint64_t hallContextValueCount = contextUnion.size();
  if (hallContextValueCount >= hallDemandCount)
    return invalid("Hall demand set has no positive context deficit");
  return TechMappingComputeContextHallDeficit(
      coverDemandCount, coverMaximumMatching, hallDemandCount,
      hallContextValueCount, std::move(groups));
}

llvm::ArrayRef<std::uint8_t>
techMappingComputeContextHallFeedbackSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(feedbackSchema.data()),
          feedbackSchema.size()};
}

std::vector<std::uint8_t> encodeTechMappingComputeContextHallFeedback(
    const TechMappingComputeContextHallDeficit &feedback) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, feedback.coverDemandCount());
  appendU64(bytes, feedback.coverMaximumMatching());
  appendU64(bytes, feedback.groups().size());
  for (const auto &group : feedback.groups()) {
    const std::vector<std::uint8_t> capability =
        loom::fabric::canonicalFabricBytes(group.capability);
    appendU64(bytes, capability.size());
    bytes.insert(bytes.end(), capability.begin(), capability.end());
    appendU64(bytes, group.demandCount);
  }
  return bytes;
}

llvm::Expected<TechMappingComputeContextHallDeficit>
adoptTechMappingComputeContextHallFeedback(
    llvm::ArrayRef<std::uint8_t> bytes,
    const loom::fabric::FabricArtifactView &fabric) {
  std::size_t offset = 0;
  auto coverDemandCount = readU64(bytes, offset);
  if (!coverDemandCount)
    return coverDemandCount.takeError();
  auto coverMaximumMatching = readU64(bytes, offset);
  if (!coverMaximumMatching)
    return coverMaximumMatching.takeError();
  auto groupCount = readU64(bytes, offset);
  if (!groupCount)
    return groupCount.takeError();
  if (*groupCount > bytes.size() / 16)
    return invalid("Hall demand group count exceeds its payload");

  std::vector<TechMappingComputeContextHallDemandGroup> groups;
  groups.reserve(*groupCount);
  for (std::uint64_t index = 0; index != *groupCount; ++index) {
    auto capabilitySize = readU64(bytes, offset);
    if (!capabilitySize)
      return capabilitySize.takeError();
    if (*capabilitySize > bytes.size() - offset)
      return invalid("capability reference is truncated");
    auto capability = loom::fabric::decodeFabricRef<
        loom::fabric::FabricFuCapabilityTemplateRef>(
        bytes.slice(offset, *capabilitySize));
    if (!capability)
      return capability.takeError();
    offset += *capabilitySize;
    auto demandCount = readU64(bytes, offset);
    if (!demandCount)
      return demandCount.takeError();
    auto contexts = deriveCompatibleContexts(*capability, fabric);
    if (!contexts)
      return contexts.takeError();
    groups.push_back({*capability, *demandCount, std::move(*contexts)});
  }
  if (offset != bytes.size())
    return invalid("payload has trailing bytes");
  auto result = TechMappingComputeContextHallDeficit::get(
      *coverDemandCount, *coverMaximumMatching, groups);
  if (!result)
    return result.takeError();
  const std::vector<std::uint8_t> canonical =
      encodeTechMappingComputeContextHallFeedback(*result);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("payload is not canonical");
  return result;
}

void retainTechMappingComputeContextHallFeedback(
    std::optional<TechMappingComputeContextHallDeficit> &retained,
    TechMappingComputeContextHallDeficit candidate) {
  if (!retained || candidate.deficit() > retained->deficit() ||
      (candidate.deficit() == retained->deficit() &&
       candidate.hallDemandCount() > retained->hallDemandCount()) ||
      (candidate.deficit() == retained->deficit() &&
       candidate.hallDemandCount() == retained->hallDemandCount() &&
       encodeTechMappingComputeContextHallFeedback(candidate) <
           encodeTechMappingComputeContextHallFeedback(*retained)))
    retained = std::move(candidate);
}

} // namespace loom::mapping
