#include "Fabric/IR/PhysicalTagResourceContract.h"

#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <limits>
#include <vector>

using namespace fabric;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid Physical Tag resource contract: " +
                                     message);
}

} // namespace

llvm::Expected<ResourceContract> fabric::appendPhysicalTagAssignmentPatterns(
    const ResourceContract *base, llvm::ArrayRef<std::uint32_t> tagWidthBits) {
  if (tagWidthBits.empty()) {
    if (!base)
      return invalid("an owner without a base has no assignment pattern");
    return ResourceContract::create(base->declaration());
  }

  ResourceContractDeclaration declaration =
      base ? base->declaration() : ResourceContractDeclaration{};
  if (tagWidthBits.size() > std::numeric_limits<std::uint32_t>::max() -
                                declaration.usePatterns.size())
    return invalid("use-pattern inventory exceeds u32");

  if (declaration.requesters.empty())
    declaration.requesters.emplace_back(0);
  if (declaration.eligibilityCount == 0)
    declaration.eligibilityCount = 1;
  if (declaration.eventCount == 0) {
    declaration.eventCount = 1;
    for (TimingContractDeclaration &timing : declaration.timingContracts)
      timing.eventRank.push_back(0);
  }
  if (declaration.timingContracts.empty())
    declaration.timingContracts.push_back(TimingContractDeclaration{
        TimingContractKey(0),
        std::vector<std::uint32_t>(declaration.eventCount, 0)});

  declaration.usePatterns.reserve(declaration.usePatterns.size() +
                                  tagWidthBits.size());
  for (std::uint32_t width : tagWidthBits) {
    declaration.usePatterns.push_back(
        UsePatternDeclaration{UsePatternKey(static_cast<std::uint32_t>(
                                  declaration.usePatterns.size())),
                              RequesterKey(0),
                              EligibilityKey(0),
                              EventKey(0),
                              EventKey(0),
                              std::nullopt,
                              TimingContractKey(0),
                              {},
                              {},
                              {},
                              {UsePatternValueSchema::physicalTag(width)}});
  }
  return ResourceContract::create(declaration);
}

std::optional<std::uint32_t>
fabric::physicalTagAssignmentPatternWidth(const UsePattern &pattern) {
  if (!pattern.claims.empty() || pattern.commit ||
      pattern.internalTransactionCount != 0 || !pattern.parameters.empty() ||
      pattern.sharingAssignments.size() != 1)
    return std::nullopt;
  const UsePatternValueSchema schema = pattern.sharingAssignments.front();
  if (schema.kind != UsePatternValueKind::PhysicalTag || schema.bitWidth == 0)
    return std::nullopt;
  return schema.bitWidth;
}
