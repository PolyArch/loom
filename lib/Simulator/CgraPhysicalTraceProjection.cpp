#include "CgraPhysicalTraceProjection.h"

#include "CGRAExecutionPlan.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include <map>
#include <system_error>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

llvm::Error unsupported(llvm::Twine message) {
  return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                 message);
}

llvm::Expected<llvm::ArrayRef<::loom::fabric::FabricUsePatternRef>>
patterns(const CgraFrozenExecutionPlan &plan, std::uint64_t actionOrdinal) {
  if (actionOrdinal >= plan.physicalUses.size())
    return invalid("CGRA trace names an unknown physical action");
  const CgraPhysicalUsePlan &use = plan.physicalUses[actionOrdinal];
  if (use.patternCount == 0 ||
      use.patternOffset > plan.physicalUsePatterns.size() ||
      use.patternCount > plan.physicalUsePatterns.size() - use.patternOffset)
    return invalid("CGRA trace physical-use slice is malformed");
  return llvm::ArrayRef(plan.physicalUsePatterns)
      .slice(use.patternOffset, use.patternCount);
}

} // namespace

llvm::Expected<PhysicalActionTarget>
projectPhysicalUseTarget(const CgraFrozenExecutionPlan &plan,
                         std::uint64_t actionOrdinal) {
  auto selected = patterns(plan, actionOrdinal);
  if (!selected)
    return selected.takeError();
  if (selected->size() != 1)
    return unsupported(
        "CGRA trace cannot encode a grouped non-transfer physical action");
  return PhysicalActionTarget{PhysicalUseTarget{selected->front()}};
}

llvm::Expected<PhysicalActionTarget> projectPhysicalTransferTarget(
    const CgraFrozenExecutionPlan &plan, std::uint64_t actionOrdinal,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef> traversals) {
  if (traversals.empty())
    return invalid("CGRA transfer trace has no selected traversal");
  std::map<std::vector<std::uint8_t>,
           ::loom::fabric::FabricPhysicalTraversalRef>
      canonical;
  for (const auto &traversal : traversals)
    canonical.try_emplace(::loom::fabric::canonicalFabricBytes(traversal),
                          traversal);
  if (canonical.size() != traversals.size())
    return invalid("CGRA transfer trace repeats a selected traversal");

  auto selected = patterns(plan, actionOrdinal);
  if (!selected)
    return selected.takeError();
  PhysicalTransferTarget target;
  target.traversals.reserve(canonical.size());
  for (const auto &[key, traversal] : canonical) {
    (void)key;
    target.traversals.push_back(traversal);
  }
  target.usePatterns.assign(selected->begin(), selected->end());
  return PhysicalActionTarget{std::move(target)};
}

} // namespace loom::sim::detail
