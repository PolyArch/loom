#include "PnR/SpatialActionExecutor.h"

#include "SpatialActionExecutorInternal.h"
#include "SpatialCandidateStateInternal.h"
#include "SpatialMemoryCompatibility.h"
#include "SpatialMemoryConstraintModel.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>
#include <variant>

using namespace loom;
using namespace loom::pnr;
using detail::executorError;
using detail::intrinsicTransitionFailure;
using detail::rangeContains;

llvm::Error SpatialActionExecutorScratch::recordExplicitLogicalMemoryBinding(
    const SpatialCandidateState &candidate,
    SpatialLogicalMemoryBindingAction action) {
  if (action.binding >= explicitLogicalMemorySelections_.size() ||
      action.target >= candidate.problem().memory().bindingTargets().size())
    return executorError("logical-memory Action is out of range");
  const SpatialLogicalMemoryBindingSelection selection{
      action.target, action.physicalOffsetBytes};
  if (explicitLogicalMemoryMarks_[action.binding] == dependencyEpoch_) {
    const auto &prior = explicitLogicalMemorySelections_[action.binding];
    if (prior.target != selection.target ||
        prior.physicalOffsetBytes != selection.physicalOffsetBytes)
      return intrinsicTransitionFailure(
          "one ActionBatch selects conflicting logical-memory bindings");
    return llvm::Error::success();
  }
  explicitLogicalMemoryMarks_[action.binding] = dependencyEpoch_;
  explicitLogicalMemorySelections_[action.binding] = selection;
  explicitLogicalMemoryBindings_.push_back(action.binding);
  explicitLogicalMemoryChoices_.push_back(selection);
  return llvm::Error::success();
}

llvm::Expected<bool>
SpatialActionExecutorScratch::explicitLogicalMemoryTargetSupported(
    const SpatialCandidateState &candidate, PnrIndex binding,
    PnrIndex targetOrdinal) const {
  auto supported =
      candidate.logicalMemoryBindingTargetSupported(binding, targetOrdinal);
  if (!supported)
    return supported.takeError();
  if (!*supported)
    return false;

  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  if (binding >= memory.logicalBindings().size() ||
      targetOrdinal >= memory.bindingTargets().size())
    return executorError("logical-memory target support is out of range");
  const FrozenSpatialMemoryBindingTargetOption &target =
      memory.bindingTargets()[targetOrdinal];
  const auto patterns =
      candidate.problem().capacity().memoryDispatchOptionPatterns();
  const auto optionSupported = [&](PnrIndex use,
                                   PnrIndex option) -> llvm::Expected<bool> {
    auto domain = candidate.memoryDispatchDomain(use);
    if (!domain)
      return domain.takeError();
    if (!rangeContains((*domain)->optionOffset, (*domain)->optionCount, option))
      return false;
    return detail::memoryDispatchMatchesTarget(
        memory, memory.dispatchOptions()[option], target);
  };

  const auto uses =
      memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                 memory.bindingUseOffsets()[binding + 1] -
                                     memory.bindingUseOffsets()[binding]);
  for (PnrIndex use : uses) {
    if (use >= explicitMemoryDispatchUseMarks_.size())
      return executorError("logical-memory use is out of range");
    if (explicitMemoryDispatchUseMarks_[use] == dependencyEpoch_) {
      auto exact = optionSupported(use, explicitMemoryDispatchSelections_[use]);
      if (!exact)
        return exact.takeError();
      if (!*exact)
        return false;
    }
    const PnrIndex group = memory.rootedUseServiceGroups()[use];
    if (group == getInvalidPnrIndex())
      continue;
    if (group >= memory.serviceUseGroups().size() ||
        group >= explicitMemoryDispatchGroupMarks_.size())
      return executorError("logical-memory use group is out of range");
    if (explicitMemoryDispatchGroupMarks_[group] != dependencyEpoch_)
      continue;
    const PnrIndex requiredPattern = explicitMemoryDispatchPatterns_[group];
    const FrozenSpatialMemoryServiceUseGroup &record =
        memory.serviceUseGroups()[group];
    if (record.logicalBinding != binding)
      return executorError("logical-memory use group has a foreign binding");
    for (PnrIndex member :
         memory.serviceGroupUses().slice(record.useOffset, record.useCount)) {
      if (member >= explicitMemoryDispatchUseMarks_.size())
        return executorError("logical-memory use group has a foreign member");
      if (explicitMemoryDispatchUseMarks_[member] == dependencyEpoch_) {
        const PnrIndex option = explicitMemoryDispatchSelections_[member];
        auto exact = optionSupported(member, option);
        if (!exact)
          return exact.takeError();
        if (!*exact || patterns[option] != requiredPattern)
          return false;
        continue;
      }
      auto domain = candidate.memoryDispatchDomain(member);
      if (!domain)
        return domain.takeError();
      bool matching = false;
      for (PnrIndex option = (*domain)->optionOffset;
           option < (*domain)->optionOffset + (*domain)->optionCount; ++option)
        matching |= patterns[option] == requiredPattern &&
                    detail::memoryDispatchMatchesTarget(
                        memory, memory.dispatchOptions()[option], target);
      if (!matching)
        return false;
    }
  }

  const auto exposures = memory.bindingExposures().slice(
      memory.bindingExposureOffsets()[binding],
      memory.bindingExposureOffsets()[binding + 1] -
          memory.bindingExposureOffsets()[binding]);
  for (PnrIndex exposure : exposures) {
    if (exposure >= explicitMemoryExposureMarks_.size())
      return executorError("logical-memory exposure is out of range");
    if (explicitMemoryExposureMarks_[exposure] != dependencyEpoch_)
      continue;
    const PnrIndex option = explicitMemoryExposureSelections_[exposure];
    if (option >= memory.exposureOptions().size() ||
        !detail::memoryExposureMatchesTarget(target,
                                             memory.exposureOptions()[option]))
      return false;
  }
  return true;
}

llvm::Error
SpatialActionExecutorScratch::reconcileExplicitLogicalMemoryBindings(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  if (!explicitLogicalMemoryBindings_.empty()) {
    auto solved = candidate.problem().memoryConstraints().solveCanonicalClosure(
        candidate.logicalMemoryBindings_, explicitLogicalMemoryBindings_,
        explicitLogicalMemoryChoices_,
        candidate.problem()
            .config()
            .policy()
            .search.initializer.assignmentAttemptLimitPerSeed,
        [&](PnrIndex binding, PnrIndex target) -> llvm::Expected<bool> {
          return explicitLogicalMemoryTargetSupported(candidate, binding,
                                                      target);
        },
        *memoryConstraintScratch_);
    if (!solved)
      return llvm::handleErrors(
          solved.takeError(),
          [&](const detail::SpatialMemoryConstraintSolveFailure &)
              -> llvm::Error {
            return llvm::make_error<SpatialActionTransitionFailure>(
                SpatialActionTransitionFailureKind::WorkLimit,
                "Spatial memory relation closure exhausted its assignment "
                "work limit");
          });
    if (!*solved)
      return intrinsicTransitionFailure(
          "logical-memory Action has no relation-closed assignment");

    const auto solution = memoryConstraintScratch_->solution();
    std::optional<PnrIndex> boundaryTarget;
    for (auto [ordinal, target] :
         llvm::enumerate(candidate.problem().memory().bindingTargets()))
      if (std::holds_alternative<FrozenSpatialMemoryBoundaryProxy>(
              target.target)) {
        boundaryTarget = static_cast<PnrIndex>(ordinal);
        break;
      }
    for (PnrIndex binding = 0; binding < solution.size(); ++binding) {
      const auto &current = candidate.logicalMemoryBinding(binding);
      const auto &replacement = solution[binding];
      if (current.target == replacement.target &&
          current.physicalOffsetBytes == replacement.physicalOffsetBytes)
        continue;
      markChangedLogicalMemoryBinding(binding);
      if (!boundaryTarget)
        return executorError("logical-memory closure has no BoundaryProxy");
      if (!std::holds_alternative<FrozenSpatialMemoryBoundaryProxy>(
              candidate.problem()
                  .memory()
                  .bindingTargets()[current.target]
                  .target))
        if (llvm::Error error =
                move.setLogicalMemoryBinding(binding, *boundaryTarget, 0))
          return error;
    }
    for (PnrIndex binding = 0; binding < solution.size(); ++binding) {
      const auto &replacement = solution[binding];
      const auto &current = candidate.logicalMemoryBinding(binding);
      if (current.target == replacement.target &&
          current.physicalOffsetBytes == replacement.physicalOffsetBytes)
        continue;
      if (llvm::Error error = move.setLogicalMemoryBinding(
              binding, replacement.target, replacement.physicalOffsetBytes))
        return error;
    }
    for (PnrIndex binding = 0; binding < solution.size(); ++binding) {
      const auto &replacement = solution[binding];
      const auto &current = candidate.logicalMemoryBinding(binding);
      if (current.target != replacement.target ||
          current.physicalOffsetBytes != replacement.physicalOffsetBytes)
        return executorError(
            "logical-memory closure lost its selected binding");
    }
    for (PnrIndex binding : explicitLogicalMemoryBindings_)
      if (candidate.logicalMemoryBinding(binding).target !=
              explicitLogicalMemorySelections_[binding].target ||
          candidate.logicalMemoryBinding(binding).physicalOffsetBytes !=
              explicitLogicalMemorySelections_[binding].physicalOffsetBytes)
        return executorError(
            "logical-memory closure replaced an explicit choice");
  }
  for (PnrIndex binding : changedLogicalMemoryBindings_)
    if (llvm::Error error =
            reconcileLogicalMemoryBinding(move, candidate, binding))
      return error;
  return llvm::Error::success();
}

void SpatialActionExecutorScratch::markChangedLogicalMemoryBinding(
    PnrIndex binding) {
  if (changedLogicalMemoryMarks_[binding] == dependencyEpoch_)
    return;
  changedLogicalMemoryMarks_[binding] = dependencyEpoch_;
  changedLogicalMemoryBindings_.push_back(binding);
}

llvm::Error SpatialActionExecutorScratch::reconcileLogicalMemoryBinding(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate,
    PnrIndex binding) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  if (binding >= memory.logicalBindings().size())
    return executorError("logical-memory Action anchor is out of range");
  const PnrIndex targetOrdinal = candidate.logicalMemoryBinding(binding).target;
  if (targetOrdinal >= memory.bindingTargets().size())
    return executorError("logical-memory Action selected a foreign target");
  const FrozenSpatialMemoryBindingTargetOption &target =
      memory.bindingTargets()[targetOrdinal];
  const auto patterns =
      candidate.problem().capacity().memoryDispatchOptionPatterns();

  const auto matchingOption = [&](PnrIndex use,
                                  std::optional<PnrIndex> requiredPattern)
      -> llvm::Expected<std::optional<PnrIndex>> {
    auto domain = candidate.memoryDispatchDomain(use);
    if (!domain)
      return domain.takeError();
    for (PnrIndex option = (*domain)->optionOffset;
         option < (*domain)->optionOffset + (*domain)->optionCount; ++option) {
      if (!detail::memoryDispatchMatchesTarget(
              memory, memory.dispatchOptions()[option], target))
        continue;
      if (!requiredPattern || patterns[option] == *requiredPattern)
        return std::optional<PnrIndex>{option};
    }
    return std::optional<PnrIndex>{};
  };
  const auto selectionMatches = [&](PnrIndex use, PnrIndex option) {
    return detail::memoryDispatchMatchesTarget(
        memory, memory.dispatchOptions()[option], target);
  };

  const auto bindingUses =
      memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                 memory.bindingUseOffsets()[binding + 1] -
                                     memory.bindingUseOffsets()[binding]);
  for (PnrIndex use : bindingUses) {
    const PnrIndex group = memory.rootedUseServiceGroups()[use];
    if (group == getInvalidPnrIndex()) {
      const PnrIndex current = candidate.memoryUseDispatch(use);
      if (selectionMatches(use, current))
        continue;
      auto replacement = matchingOption(use, std::nullopt);
      if (!replacement)
        return replacement.takeError();
      if (!*replacement)
        return intrinsicTransitionFailure(
            "logical-memory target has no compatible dispatch");
      if (llvm::Error error = move.setMemoryUseDispatch(use, **replacement))
        return error;
      continue;
    }
    if (group >= memory.serviceUseGroups().size())
      return executorError("memory use selects a foreign service-use group");
    const FrozenSpatialMemoryServiceUseGroup &record =
        memory.serviceUseGroups()[group];
    const auto groupUses =
        memory.serviceGroupUses().slice(record.useOffset, record.useCount);
    if (groupUses.empty() || groupUses.front() != use)
      continue;

    const PnrIndex currentPattern =
        patterns[candidate.memoryUseDispatch(groupUses.front())];
    const bool currentCompatible =
        llvm::all_of(groupUses, [&](PnrIndex member) {
          const PnrIndex option = candidate.memoryUseDispatch(member);
          return selectionMatches(member, option) &&
                 patterns[option] == currentPattern;
        });
    if (currentCompatible)
      continue;

    auto firstDomain = candidate.memoryDispatchDomain(groupUses.front());
    if (!firstDomain)
      return firstDomain.takeError();
    std::optional<PnrIndex> selectedPattern;
    for (PnrIndex option = (*firstDomain)->optionOffset;
         option < (*firstDomain)->optionOffset + (*firstDomain)->optionCount;
         ++option) {
      if (!selectionMatches(groupUses.front(), option))
        continue;
      const PnrIndex pattern = patterns[option];
      bool common = true;
      for (PnrIndex member : groupUses) {
        auto compatible = matchingOption(member, pattern);
        if (!compatible)
          return compatible.takeError();
        if (!*compatible) {
          common = false;
          break;
        }
      }
      if (common) {
        selectedPattern = pattern;
        break;
      }
    }
    if (!selectedPattern)
      return intrinsicTransitionFailure(
          "logical-memory target has no common service UsePattern");
    for (PnrIndex member : groupUses) {
      auto replacement = matchingOption(member, selectedPattern);
      if (!replacement)
        return replacement.takeError();
      if (!*replacement)
        return executorError("common memory dispatch disappeared");
      if (llvm::Error error = move.setMemoryUseDispatch(member, **replacement))
        return error;
    }
  }

  const auto exposures = memory.bindingExposures().slice(
      memory.bindingExposureOffsets()[binding],
      memory.bindingExposureOffsets()[binding + 1] -
          memory.bindingExposureOffsets()[binding]);
  for (PnrIndex exposure : exposures) {
    const PnrIndex current = candidate.memoryExposureSelection(exposure);
    if (detail::memoryExposureMatchesTarget(target,
                                            memory.exposureOptions()[current]))
      continue;
    std::optional<PnrIndex> replacement;
    for (PnrIndex option = 0; option < memory.exposureOptions().size();
         ++option)
      if (detail::memoryExposureMatchesTarget(
              target, memory.exposureOptions()[option])) {
        replacement = option;
        break;
      }
    if (!replacement)
      return intrinsicTransitionFailure(
          "logical-memory target has no compatible exposure");
    if (llvm::Error error =
            move.setMemoryExposureSelection(exposure, *replacement))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::recordExplicitMemoryDispatch(
    const SpatialCandidateState &candidate, PnrIndex use, PnrIndex option) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  if (use >= memory.rootedUseServiceGroups().size() ||
      option >=
          candidate.problem().capacity().memoryDispatchOptionPatterns().size())
    return executorError("memory-dispatch Action is out of range");
  if (use >= explicitMemoryDispatchSelections_.size())
    return executorError("memory-dispatch Action has a foreign use");
  if (explicitMemoryDispatchUseMarks_[use] == dependencyEpoch_ &&
      explicitMemoryDispatchSelections_[use] != option)
    return intrinsicTransitionFailure(
        "one ActionBatch selects conflicting options for one memory use");
  explicitMemoryDispatchUseMarks_[use] = dependencyEpoch_;
  explicitMemoryDispatchSelections_[use] = option;
  const PnrIndex group = memory.rootedUseServiceGroups()[use];
  if (group == getInvalidPnrIndex())
    return llvm::Error::success();
  if (group >= explicitMemoryDispatchPatterns_.size())
    return executorError("memory-dispatch Action has a foreign group");
  const PnrIndex pattern =
      candidate.problem().capacity().memoryDispatchOptionPatterns()[option];
  if (explicitMemoryDispatchGroupMarks_[group] == dependencyEpoch_) {
    if (explicitMemoryDispatchPatterns_[group] != pattern)
      return intrinsicTransitionFailure(
          "one ActionBatch selects conflicting memory UsePatterns");
    return llvm::Error::success();
  }
  explicitMemoryDispatchGroupMarks_[group] = dependencyEpoch_;
  explicitMemoryDispatchPatterns_[group] = pattern;
  explicitMemoryDispatchGroups_.push_back(group);
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::reconcileExplicitMemoryDispatches(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  const auto patterns =
      candidate.problem().capacity().memoryDispatchOptionPatterns();
  for (PnrIndex group : explicitMemoryDispatchGroups_) {
    if (group >= memory.serviceUseGroups().size())
      return executorError("explicit memory dispatch has a foreign group");
    const FrozenSpatialMemoryServiceUseGroup &record =
        memory.serviceUseGroups()[group];
    if (record.logicalBinding >= memory.logicalBindings().size())
      return executorError("memory dispatch group has a foreign binding");
    const PnrIndex targetOrdinal =
        candidate.logicalMemoryBinding(record.logicalBinding).target;
    if (targetOrdinal >= memory.bindingTargets().size())
      return executorError("memory dispatch group has a foreign target");
    const FrozenSpatialMemoryBindingTargetOption &target =
        memory.bindingTargets()[targetOrdinal];
    const PnrIndex requiredPattern = explicitMemoryDispatchPatterns_[group];
    const auto members =
        memory.serviceGroupUses().slice(record.useOffset, record.useCount);
    for (PnrIndex member : members) {
      if (member >= explicitMemoryDispatchUseMarks_.size())
        return executorError("memory dispatch group has a foreign member");
      PnrIndex selected =
          explicitMemoryDispatchUseMarks_[member] == dependencyEpoch_
              ? explicitMemoryDispatchSelections_[member]
              : candidate.memoryUseDispatch(member);
      if (explicitMemoryDispatchUseMarks_[member] == dependencyEpoch_ &&
          candidate.memoryUseDispatch(member) != selected) {
        if (llvm::Error error = move.setMemoryUseDispatch(member, selected))
          return error;
      }
      if (patterns[selected] == requiredPattern &&
          detail::memoryDispatchMatchesTarget(
              memory, memory.dispatchOptions()[selected], target))
        continue;
      if (explicitMemoryDispatchUseMarks_[member] == dependencyEpoch_)
        return intrinsicTransitionFailure(
            "explicit memory-dispatch Action is incompatible with its group");
      auto domain = candidate.memoryDispatchDomain(member);
      if (!domain)
        return domain.takeError();
      selected = getInvalidPnrIndex();
      for (PnrIndex option = (*domain)->optionOffset;
           option < (*domain)->optionOffset + (*domain)->optionCount; ++option)
        if (patterns[option] == requiredPattern &&
            detail::memoryDispatchMatchesTarget(
                memory, memory.dispatchOptions()[option], target)) {
          selected = option;
          break;
        }
      if (selected == getInvalidPnrIndex())
        return intrinsicTransitionFailure(
            "memory-dispatch Action has no group-compatible selection");
      if (llvm::Error error = move.setMemoryUseDispatch(member, selected))
        return error;
    }
  }
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::recordExplicitMemoryExposure(
    const SpatialCandidateState &candidate, PnrIndex exposure,
    PnrIndex option) {
  if (exposure >= explicitMemoryExposureSelections_.size() ||
      option >= candidate.problem().memory().exposureOptions().size())
    return executorError("memory-exposure Action is out of range");
  if (explicitMemoryExposureMarks_[exposure] == dependencyEpoch_ &&
      explicitMemoryExposureSelections_[exposure] != option)
    return intrinsicTransitionFailure(
        "one ActionBatch selects conflicting memory exposures");
  explicitMemoryExposureMarks_[exposure] = dependencyEpoch_;
  explicitMemoryExposureSelections_[exposure] = option;
  return llvm::Error::success();
}

llvm::Error SpatialActionExecutorScratch::reconcileExplicitMemoryExposures(
    SpatialMoveTransaction &move, SpatialCandidateState &candidate) {
  const FrozenSpatialMemoryIndex &memory = candidate.problem().memory();
  for (PnrIndex exposure = 0; exposure < explicitMemoryExposureMarks_.size();
       ++exposure) {
    if (explicitMemoryExposureMarks_[exposure] != dependencyEpoch_)
      continue;
    const PnrIndex option = explicitMemoryExposureSelections_[exposure];
    const PnrIndex binding = memory.exposures()[exposure].logicalBinding;
    const PnrIndex target = candidate.logicalMemoryBinding(binding).target;
    if (target >= memory.bindingTargets().size() ||
        option >= memory.exposureOptions().size())
      return executorError("explicit memory exposure is out of range");
    if (!detail::memoryExposureMatchesTarget(memory.bindingTargets()[target],
                                             memory.exposureOptions()[option]))
      return intrinsicTransitionFailure(
          "explicit memory-exposure Action is incompatible with its binding");
    if (candidate.memoryExposureSelection(exposure) != option)
      if (llvm::Error error = move.setMemoryExposureSelection(exposure, option))
        return error;
  }
  return llvm::Error::success();
}

