#include "../TestAllocationProbe.h"
#include "TechMappingCandidateTestSupport.h"

#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialActionExecutor.h"
#include "PnR/SpatialCandidateInitializer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace pnr = loom::pnr;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial memory Action test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

const pnr::FrozenSpatialMemoryDispatchDomain *
dispatchDomain(const pnr::FrozenSpatialPnrProblem &problem,
               const pnr::SpatialCandidateState &candidate, pnr::PnrIndex use) {
  const auto &memory = problem.memory();
  if (use >= memory.rootedUses().size())
    return nullptr;
  const auto &rootedUse = memory.rootedUses()[use];
  const auto &realizations = problem.realizations();
  if (rootedUse.actor >= realizations.memoryActorRealizations().size())
    return nullptr;
  const pnr::PnrIndex realization =
      realizations.memoryActorRealizations()[rootedUse.actor];
  const pnr::PnrIndex placement =
      candidate.memoryBinding(realization).placement;
  const auto found =
      llvm::find_if(memory.dispatchDomains(), [&](const auto &domain) {
        return domain.actor == rootedUse.actor && domain.placement == placement;
      });
  return found == memory.dispatchDomains().end() ? nullptr : &*found;
}

bool dispatchMatchesTarget(
    const pnr::FrozenSpatialMemoryIndex &memory,
    const pnr::FrozenSpatialMemoryDispatchOption &option,
    const pnr::FrozenSpatialMemoryBindingTargetOption &target) {
  if (const auto *region =
          std::get_if<loom::fabric::FabricMemoryServiceRegionRef>(
              &target.target)) {
    const auto *local =
        std::get_if<loom::fabric::LocalMemoryServiceRef>(&option.target);
    if (!local || local->underlying() != region->service)
      return false;
    const auto regions = memory.dispatchServiceRegionOrdinals().slice(
        option.serviceRegionOffset, option.serviceRegionCount);
    return std::binary_search(regions.begin(), regions.end(), region->ordinal);
  }
  return std::holds_alternative<loom::fabric::ManagerEndpointRef>(
      option.target);
}

std::vector<pnr::SpatialMemoryUseDispatchAction>
commonDispatchActions(const pnr::FrozenSpatialPnrProblem &problem,
                      const pnr::SpatialCandidateState &candidate,
                      llvm::ArrayRef<pnr::PnrIndex> members,
                      pnr::PnrIndex targetOrdinal) {
  const auto &memory = problem.memory();
  const auto patterns = problem.capacity().memoryDispatchOptionPatterns();
  if (members.empty() || targetOrdinal >= memory.bindingTargets().size())
    return {};
  const auto *firstDomain = dispatchDomain(problem, candidate, members.front());
  if (!firstDomain)
    return {};
  const auto &target = memory.bindingTargets()[targetOrdinal];
  for (pnr::PnrIndex first = firstDomain->optionOffset;
       first < firstDomain->optionOffset + firstDomain->optionCount; ++first) {
    if (!dispatchMatchesTarget(memory, memory.dispatchOptions()[first], target))
      continue;
    const pnr::PnrIndex pattern = patterns[first];
    std::vector<pnr::SpatialMemoryUseDispatchAction> actions;
    actions.push_back({members.front(), first});
    bool common = true;
    for (pnr::PnrIndex member : members.drop_front()) {
      const auto *domain = dispatchDomain(problem, candidate, member);
      std::optional<pnr::PnrIndex> selected;
      if (domain)
        for (pnr::PnrIndex option = domain->optionOffset;
             option < domain->optionOffset + domain->optionCount; ++option)
          if (patterns[option] == pattern &&
              dispatchMatchesTarget(memory, memory.dispatchOptions()[option],
                                    target)) {
            selected = option;
            break;
          }
      if (!selected) {
        common = false;
        break;
      }
      actions.push_back({member, *selected});
    }
    if (common)
      return actions;
  }
  return {};
}

} // namespace

void loom::test::exerciseSpatialActionSequence(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialCandidateState &candidate, std::uint64_t proposalCount) {
  pnr::SpatialActionDomainScratch domain;
  pnr::SpatialActionExecutorScratch executor;
  requireSuccess(domain.prepare(*problem));
  requireSuccess(domain.rebuild(candidate));
  requireSuccess(executor.prepare(candidate));
  pnr::DeterministicPnrRandomStream proposals =
      pnr::DeterministicPnrRandomStream::create(
          UINT64_C(0x89abcdef01234567), 0,
          pnr::PnrRandomStreamPurpose::ActionProposal);
  pnr::DeterministicPnrRandomStream resolutions =
      pnr::DeterministicPnrRandomStream::create(
          UINT64_C(0xfedcba9876543210), 0,
          pnr::PnrRandomStreamPurpose::Acceptance);
  std::uint64_t committed = 0;
  std::uint64_t discarded = 0;
  std::uint64_t rejected = 0;

  for (std::uint64_t proposal = 0; proposal < proposalCount; ++proposal) {
    requireSuccess(domain.rebuild(candidate));
    if ((proposal % 32) == 0) {
      const std::array<std::size_t, 6> cardinalities{
          domain.view().realizationAnchors.size(),
          domain.view().realizationChoices.size(),
          domain.view().transportAnchors.size(),
          domain.view().transportChoices.size(),
          domain.view().resourceAnchors.size(),
          domain.view().resourceChoices.size()};
      const std::uint64_t movable = domain.movableDecisionCount();
      const auto objective =
          take(problem->objectiveProgram().evaluate(candidate));
      requireSuccess(domain.rebuild(candidate));
      const std::array<std::size_t, 6> rebuiltCardinalities{
          domain.view().realizationAnchors.size(),
          domain.view().realizationChoices.size(),
          domain.view().transportAnchors.size(),
          domain.view().transportChoices.size(),
          domain.view().resourceAnchors.size(),
          domain.view().resourceChoices.size()};
      const auto rebuiltObjective =
          take(problem->objectiveProgram().evaluate(candidate));
      if (cardinalities != rebuiltCardinalities ||
          movable != domain.movableDecisionCount() ||
          objective.codes() != rebuiltObjective.codes())
        fail("warm Spatial cache rebuild changed a candidate answer");
      requireSuccess(executor.prepare(candidate));
      if (executor.currentObjective().codes() != objective.codes())
        fail("Spatial Action executor cache diverged from the full objective");
    }

    auto action = take(pnr::proposeSpatialAction(
        loom::ResolvedPnrActionProposalPolicy{1, 1, 1}, domain.view(),
        proposals));
    if (!action)
      continue;
    auto probe = executor.probe(candidate, *action);
    if (!probe) {
      bool typed = false;
      llvm::Error unhandled = llvm::handleErrors(
          probe.takeError(),
          [&](const pnr::SpatialActionTransitionFailure &) -> llvm::Error {
            typed = true;
            return llvm::Error::success();
          });
      requireSuccess(std::move(unhandled));
      if (!typed)
        fail("Spatial Action failure lost its transition classification");
      ++rejected;
      requireSuccess(candidate.verify());
    } else if ((resolutions.nextU64() & 3U) == 0) {
      requireSuccess(probe->discard());
      ++discarded;
      requireSuccess(candidate.verify());
    } else {
      requireSuccess(probe->commit());
      ++committed;
      requireSuccess(candidate.verify());
    }
  }
  if (committed == 0 || discarded == 0 || committed + discarded + rejected == 0)
    fail("Spatial Action sequence did not exercise commit and discard");
}

void loom::test::exerciseSpatialMemoryActionDomain(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialCandidateState &candidate) {
  if (!allocationProbeIsCalibrated())
    fail("heap allocation probe did not observe its calibration calls");
  for (std::uint32_t attempt = 0;
       attempt < problem->config().policy().search.initializer.seedAttemptCount;
       ++attempt) {
    std::uint64_t assignmentAttempts = 0;
    auto initialized = take(pnr::createSpatialCandidateInitializerAttempt(
        problem, attempt, assignmentAttempts));
    requireSuccess(initialized.candidate->verify());
    const auto patterns = problem->capacity().memoryDispatchOptionPatterns();
    for (const auto &group : problem->memory().serviceUseGroups()) {
      std::optional<pnr::PnrIndex> selectedPattern;
      for (pnr::PnrIndex use : problem->memory().serviceGroupUses().slice(
               group.useOffset, group.useCount)) {
        const pnr::PnrIndex pattern =
            patterns[initialized.candidate->memoryUseDispatch(use)];
        if (selectedPattern && *selectedPattern != pattern)
          fail("diversified initializer split one service UsePattern");
        selectedPattern = pattern;
      }
    }
  }
  pnr::SpatialActionDomainScratch domain;
  requireSuccess(domain.prepare(*problem));
  requireSuccess(domain.rebuild(candidate));
  const auto &memory = problem->memory();
  std::optional<pnr::SpatialLogicalMemoryBindingAction> logicalBinding;
  for (const pnr::SpatialResourceAllocationAction &action :
       domain.view().resourceChoices)
    if (const auto *choice =
            std::get_if<pnr::SpatialLogicalMemoryBindingAction>(&action);
        choice &&
        memory.bindingUseOffsets()[choice->binding + 1] >
            memory.bindingUseOffsets()[choice->binding] &&
        choice->target !=
            candidate.logicalMemoryBinding(choice->binding).target) {
      logicalBinding = *choice;
      break;
    }
  if (!logicalBinding)
    fail("logical-memory target choices are absent from the Action domain");

  const auto originalBinding =
      candidate.logicalMemoryBinding(logicalBinding->binding);
  const auto uses = memory.bindingUses().slice(
      memory.bindingUseOffsets()[logicalBinding->binding],
      memory.bindingUseOffsets()[logicalBinding->binding + 1] -
          memory.bindingUseOffsets()[logicalBinding->binding]);
  const auto exposures = memory.bindingExposures().slice(
      memory.bindingExposureOffsets()[logicalBinding->binding],
      memory.bindingExposureOffsets()[logicalBinding->binding + 1] -
          memory.bindingExposureOffsets()[logicalBinding->binding]);
  std::vector<pnr::PnrIndex> originalDispatches;
  std::vector<pnr::PnrIndex> originalExposures;
  for (pnr::PnrIndex use : uses)
    originalDispatches.push_back(candidate.memoryUseDispatch(use));
  for (pnr::PnrIndex exposure : exposures)
    originalExposures.push_back(candidate.memoryExposureSelection(exposure));
  const std::vector<std::uint64_t> originalEnvelopeBits(
      candidate.activeResourceTimeEnvelopeBits().begin(),
      candidate.activeResourceTimeEnvelopeBits().end());
  const std::uint64_t originalCapacityOveruse =
      candidate.atomicCapacityOveruse();
  const auto originalObjective =
      take(problem->objectiveProgram().evaluate(candidate));

  const pnr::PnrIndex serviceGroup =
      uses.empty() ? pnr::getInvalidPnrIndex()
                   : memory.rootedUseServiceGroups()[uses.front()];
  if (serviceGroup >= memory.serviceUseGroups().size())
    fail("logical-memory fixture has no service-use group");
  const auto &group = memory.serviceUseGroups()[serviceGroup];
  const auto groupUses =
      memory.serviceGroupUses().slice(group.useOffset, group.useCount);
  const auto explicitDispatches = commonDispatchActions(
      *problem, candidate, groupUses, logicalBinding->target);
  if (explicitDispatches.size() != groupUses.size())
    fail("logical-memory target has no explicit common-pattern dispatch batch");
  pnr::SpatialActionExecutorScratch executor;
  requireSuccess(executor.prepare(candidate));
  const pnr::SpatialMappingAction action =
      pnr::SpatialResourceAllocationAction{*logicalBinding};
  std::vector<pnr::SpatialMappingAction> batch;
  batch.push_back(action);
  for (const auto &dispatch : explicitDispatches)
    batch.push_back(pnr::SpatialResourceAllocationAction{dispatch});
  auto probe = take(executor.probeBatch(candidate, batch));
  if (candidate.logicalMemoryBinding(logicalBinding->binding).target !=
      logicalBinding->target)
    fail("logical-memory Action did not select its target");
  const auto patterns = problem->capacity().memoryDispatchOptionPatterns();
  const pnr::PnrIndex selectedPattern =
      patterns[explicitDispatches.front().dispatchOption];
  for (const auto &dispatch : explicitDispatches) {
    if (candidate.memoryUseDispatch(dispatch.use) != dispatch.dispatchOption)
      fail("memory dependency closure replaced an explicit dispatch");
    if (patterns[candidate.memoryUseDispatch(dispatch.use)] != selectedPattern)
      fail("memory dependency closure split one service UsePattern");
  }
  const bool dispatchChanged =
      llvm::any_of(llvm::enumerate(uses), [&](auto indexedUse) {
        return candidate.memoryUseDispatch(indexedUse.value()) !=
               originalDispatches[indexedUse.index()];
      });
  const bool exposureChanged =
      llvm::any_of(llvm::enumerate(exposures), [&](auto indexedExposure) {
        return candidate.memoryExposureSelection(indexedExposure.value()) !=
               originalExposures[indexedExposure.index()];
      });
  if (!dispatchChanged && !exposureChanged)
    fail("logical-memory Action did not close a dependent selection");
  requireSuccess(probe.commit());
  requireSuccess(candidate.verify());

  const pnr::SpatialMappingAction restoreAction =
      pnr::SpatialResourceAllocationAction{
          pnr::SpatialLogicalMemoryBindingAction{
              logicalBinding->binding, originalBinding.target,
              originalBinding.physicalOffsetBytes}};
  auto restore = take(executor.probe(candidate, restoreAction));
  requireSuccess(restore.commit());

  const auto requireRestored = [&]() {
    const auto &binding =
        candidate.logicalMemoryBinding(logicalBinding->binding);
    if (binding.target != originalBinding.target ||
        binding.physicalOffsetBytes != originalBinding.physicalOffsetBytes)
      fail("logical-memory rollback changed its binding");
    for (auto indexedUse : llvm::enumerate(uses))
      if (candidate.memoryUseDispatch(indexedUse.value()) !=
          originalDispatches[indexedUse.index()])
        fail("logical-memory rollback changed a dispatch");
    for (auto indexedExposure : llvm::enumerate(exposures))
      if (candidate.memoryExposureSelection(indexedExposure.value()) !=
          originalExposures[indexedExposure.index()])
        fail("logical-memory rollback changed an exposure");
    if (!llvm::equal(candidate.activeResourceTimeEnvelopeBits(),
                     originalEnvelopeBits) ||
        candidate.atomicCapacityOveruse() != originalCapacityOveruse)
      fail("logical-memory rollback changed derived resource state");
    const auto objective =
        take(problem->objectiveProgram().evaluate(candidate));
    if (objective.codes() != originalObjective.codes())
      fail("logical-memory rollback changed the objective");
    requireSuccess(candidate.verify());
  };
  requireRestored();

  if (!memory.exposures().empty()) {
    const pnr::SpatialMemoryExposureAction explicitExposure{
        0, candidate.memoryExposureSelection(0)};
    const pnr::SpatialMappingAction exposureAction =
        pnr::SpatialResourceAllocationAction{explicitExposure};
    auto exposureProbe = take(executor.probe(candidate, exposureAction));
    if (candidate.memoryExposureSelection(explicitExposure.exposure) !=
        explicitExposure.exposureOption)
      fail("memory closure replaced an exact explicit exposure");
    requireSuccess(exposureProbe.discard());
    if (candidate.memoryExposureSelection(explicitExposure.exposure) !=
        explicitExposure.exposureOption)
      fail("explicit exposure batch did not roll back exactly");
    requireSuccess(candidate.verify());
  }

  std::optional<pnr::SpatialMemoryUseDispatchAction> firstConflict;
  std::optional<pnr::SpatialMemoryUseDispatchAction> secondConflict;
  if (groupUses.size() >= 2) {
    const auto *firstDomain = dispatchDomain(*problem, candidate, groupUses[0]);
    const auto *secondDomain =
        dispatchDomain(*problem, candidate, groupUses[1]);
    if (firstDomain && secondDomain)
      for (pnr::PnrIndex first = firstDomain->optionOffset;
           first < firstDomain->optionOffset + firstDomain->optionCount;
           ++first)
        for (pnr::PnrIndex second = secondDomain->optionOffset;
             second < secondDomain->optionOffset + secondDomain->optionCount;
             ++second)
          if (patterns[first] != patterns[second]) {
            firstConflict = {groupUses[0], first};
            secondConflict = {groupUses[1], second};
            break;
          }
  }
  if (!firstConflict || !secondConflict)
    fail("memory fixture has no conflicting service UsePatterns");
  const std::array<pnr::SpatialMappingAction, 2> conflictingBatch{
      pnr::SpatialResourceAllocationAction{*firstConflict},
      pnr::SpatialResourceAllocationAction{*secondConflict}};
  auto conflict = executor.probeBatch(candidate, conflictingBatch);
  if (conflict)
    fail("conflicting memory UsePatterns produced an Action probe");
  bool typedConflict = false;
  llvm::Error unhandled = llvm::handleErrors(
      conflict.takeError(),
      [&](const pnr::SpatialActionTransitionFailure &) -> llvm::Error {
        typedConflict = true;
        return llvm::Error::success();
      });
  requireSuccess(std::move(unhandled));
  if (!typedConflict)
    fail("conflicting memory UsePatterns lost typed rollback");
  requireRestored();

  auto discarded = take(executor.probe(candidate, action));
  requireSuccess(discarded.discard());
  requireRestored();

  for (std::uint64_t warm = 0; warm < 8; ++warm) {
    requireSuccess(domain.rebuild(candidate));
    auto warmProbe = take(executor.probe(candidate, action));
    requireSuccess(warmProbe.discard());
  }
  const std::size_t warmDomainBytes = domain.retainedStorageBytes();
  const std::size_t warmExecutorBytes = executor.retainedStorageBytes();
  startAllocationProbe();
  for (std::uint64_t replay = 0; replay < 32; ++replay) {
    requireSuccess(domain.rebuild(candidate));
    auto replayProbe = take(executor.probe(candidate, action));
    requireSuccess(replayProbe.discard());
  }
  if (stopAllocationProbe() != 0)
    fail("warm local Spatial Action performed a heap allocation");
  if (domain.retainedStorageBytes() != warmDomainBytes ||
      executor.retainedStorageBytes() != warmExecutorBytes)
    fail("warm local Spatial Action changed retained scratch storage");
  requireRestored();
  auto sequenceCandidate = take(pnr::createCanonicalSpatialCandidate(problem));
  exerciseSpatialActionSequence(problem, *sequenceCandidate, 512);
}
