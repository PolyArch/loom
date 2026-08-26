#include "PnR/SpatialExactRepair.h"

#include "SpatialBindingRelationModel.h"
#include "SpatialMemoryCompatibility.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

using namespace loom::pnr;

std::optional<std::string> loom::pnr::unsupportedSpatialExactRepairDomain(
    const FrozenSpatialPnrProblem &problem) {
  const ResolvedPnrExactRepairPolicy &policy =
      problem.config().policy().search.exactRepair;
  if (policy.kind == ResolvedPnrExactRepairKind::Disabled)
    return std::nullopt;
  if (policy.kind != ResolvedPnrExactRepairKind::CpSat)
    return "the selected exact-repair provider is not implemented";

  const detail::SpatialBindingRelationModel &bindings =
      problem.bindingRelations();

  const FrozenSpatialCapacityIndex &capacity = problem.capacity();
  if (llvm::any_of(capacity.memoryOperationPlanOveruse(),
                   [](std::uint64_t value) { return value != 0; }))
    return "CpSat_3_0 does not encode mutable memory operation-plan atomic "
           "capacity";
  if (llvm::any_of(capacity.memoryDispatchOptionOveruse(),
                   [](std::uint64_t value) { return value != 0; }))
    return "CpSat_3_0 does not encode mutable memory dispatch atomic "
           "capacity";

  const FrozenSpatialMemoryIndex &memory = problem.memory();
  for (PnrIndex provider = 0; provider < memory.exposureProviders().size();
       ++provider) {
    std::set<PnrIndex> possibleBindings;
    for (const FrozenSpatialMemoryExposure &exposure : memory.exposures()) {
      const bool canSelectProvider =
          llvm::any_of(memory.exposureOptions(), [&](const auto &option) {
            if (option.provider != provider)
              return false;
            return llvm::any_of(
                memory.bindingTargets(), [&](const auto &target) {
                  return detail::memoryExposureMatchesTarget(target, option);
                });
          });
      if (canSelectProvider)
        possibleBindings.insert(exposure.logicalBinding);
    }
    if (possibleBindings.size() >
        memory.exposureProviders()[provider].maxExposedBindings)
      return "CpSat_3_0 does not encode mutable memory exposure-provider "
             "capacity";
  }

  const detail::InitializerRelationModel &relations = bindings.relations();
  const PnrIndex computeCount = bindings.computeDecisionCount();
  const auto contextOveruse = capacity.computeInstructionContextOveruse();
  std::vector<std::uint8_t> visited(bindings.decisionCount(), 0);
  std::vector<PnrIndex> worklist;
  for (PnrIndex seed = 0; seed < computeCount; ++seed) {
    const bool canOveruse =
        llvm::any_of(bindings.computeChoices(seed), [&](const auto &choice) {
          return choice.instructionContext >= contextOveruse.size() ||
                 contextOveruse[choice.instructionContext] != 0;
        });
    if (!canOveruse)
      continue;
    std::fill(visited.begin(), visited.end(), 0);
    worklist.clear();
    visited[seed] = 1;
    worklist.push_back(seed);
    for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const PnrIndex decision = worklist[cursor];
      for (PnrIndex relation : bindings.decisionRelations(decision)) {
        if (!bindings.relationIsConstraint(relation))
          continue;
        for (const detail::InitializerRelationMember &member :
             relations.members(relations.relations()[relation])) {
          if (member.decision >= computeCount)
            return "CpSat_3_0 atomic-capacity relation closure contains a "
                   "non-compute decision";
          if (!visited[member.decision]) {
            visited[member.decision] = 1;
            worklist.push_back(member.decision);
          }
        }
      }
    }
  }
  return std::nullopt;
}
