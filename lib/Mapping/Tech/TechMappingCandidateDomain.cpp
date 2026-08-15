#include "TechMappingCandidateDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping::detail {
llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tech_mapping_generation_invalid: " + message);
}

llvm::Expected<bool>
TechMatchRowCollector::beginSeed(std::vector<std::uint8_t> key) {
  if (activeSeedKey_)
    return invalid("previous match-row seed has no typed outcome");
  if (previousSeedKey_ && key < *previousSeedKey_)
    return invalid("match-row seeds are not in canonical key order");
  previousSeedKey_ = key;
  activeSeedKey_ = std::move(key);
  if (executionControl_.stopRequested()) {
    activeSeedKey_.reset();
    interrupted_ = true;
    return false;
  }
  if (atLimit()) {
    activeSeedKey_.reset();
    truncated_ = true;
    return false;
  }
  ++accounting_.matchRowAttempts;
  ++accounting_.matchRowFirstVisits;
  return true;
}

llvm::Error TechMatchRowCollector::reject(TechMatchSeedRejectionReason reason) {
  if (!activeSeedKey_)
    return invalid("match-row rejection has no active prospective seed");
  if (reason == TechMatchSeedRejectionReason::Count)
    return invalid("match-row rejection reason is not a concrete variant");
  ++rejectionCounts_[static_cast<std::size_t>(reason)];
  activeSeedKey_.reset();
  return llvm::Error::success();
}

llvm::Error TechMatchRowCollector::rejectCanonicalSeedRange(
    std::vector<std::uint8_t> firstKey, std::vector<std::uint8_t> lastKey,
    std::uint64_t count, bool countOverflow,
    TechMatchSeedRejectionReason reason) {
  if (activeSeedKey_)
    return invalid("cannot reject a seed range with an active seed");
  if (count == 0)
    return invalid("canonical seed range is empty");
  if (reason == TechMatchSeedRejectionReason::Count)
    return invalid("match-row rejection reason is not a concrete variant");
  if (lastKey < firstKey || (previousSeedKey_ && firstKey < *previousSeedKey_))
    return invalid("match-row seed range is not in canonical key order");

  const std::uint64_t available = limit_ - accounting_.matchRowAttempts;
  const std::uint64_t charged = std::min(count, available);
  accounting_.matchRowAttempts += charged;
  accounting_.matchRowFirstVisits += charged;
  rejectionCounts_[static_cast<std::size_t>(reason)] += charged;
  if (charged != count || countOverflow) {
    truncated_ = true;
    return llvm::Error::success();
  }
  previousSeedKey_ = std::move(lastKey);
  return llvm::Error::success();
}

llvm::Expected<std::size_t>
TechMatchRowCollector::actorSlot(::dataflow::ActorRef actor) const {
  auto found = llvm::lower_bound(
      actors_, actor,
      [](const ::dataflow::ActorRef &lhs, const ::dataflow::ActorRef &rhs) {
        return lhs.entity.value() < rhs.entity.value();
      });
  if (found == actors_.end() || *found != actor)
    return invalid("match row names an actor outside the invocation cover");
  return static_cast<std::size_t>(found - actors_.begin());
}

llvm::Error TechMatchRowCollector::admit(
    TechMatchRealization realization,
    llvm::ArrayRef<::dataflow::ActorRef> coveredActors) {
  if (!activeSeedKey_)
    return invalid("match-row admission has no active prospective seed");
  if (coveredActors.empty())
    return invalid("match row covers no actor");
  std::vector<std::size_t> slots;
  slots.reserve(coveredActors.size());
  for (const ::dataflow::ActorRef &actor : coveredActors) {
    auto slot = actorSlot(actor);
    if (!slot)
      return slot.takeError();
    slots.push_back(*slot);
  }
  llvm::sort(slots);
  if (std::adjacent_find(slots.begin(), slots.end()) != slots.end())
    return invalid("match row covers one actor more than once");
  rows_.push_back(TechMatchRow{std::move(*activeSeedKey_), std::move(slots),
                               std::move(realization)});
  activeSeedKey_.reset();
  return llvm::Error::success();
}

llvm::Expected<std::vector<TechMatchRow>> TechMatchRowCollector::takeRows() {
  if (activeSeedKey_)
    return invalid("match-row seed has no typed outcome");
  return std::move(rows_);
}

void appendU32(std::vector<std::uint8_t> &key, std::uint32_t value) {
  key.push_back(static_cast<std::uint8_t>(value >> 24));
  key.push_back(static_cast<std::uint8_t>(value >> 16));
  key.push_back(static_cast<std::uint8_t>(value >> 8));
  key.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &key, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    key.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendBytes(std::vector<std::uint8_t> &key,
                 llvm::ArrayRef<std::uint8_t> bytes) {
  appendU64(key, bytes.size());
  key.insert(key.end(), bytes.begin(), bytes.end());
}

llvm::Expected<TechMatchDomain>
deriveTechMatchDomain(const TechMappingGenerationInputs &inputs,
                      llvm::ArrayRef<::dataflow::CanonicalActorView> selected,
                      TechMappingGenerationAccounting &accounting) {
  using MatchRowFamily =
      std::variant<::loom::fabric::FabricFuCapabilityTemplateRef,
                   ::loom::fabric::FabricMemoryEngineTemplateRef>;
  std::vector<::dataflow::ActorRef> actors;
  actors.reserve(selected.size());
  for (const auto &actor : selected)
    actors.push_back(actor.ref);

  const auto computeFamilies = deriveComputeRowFamilies(inputs, selected);
  const auto memoryFamilies = deriveMemoryRowFamilies(inputs, selected);
  std::vector<MatchRowFamily> families;
  families.reserve(computeFamilies.size() + memoryFamilies.size());
  const std::size_t familyDepth =
      std::max(computeFamilies.size(), memoryFamilies.size());
  for (std::size_t ordinal = 0; ordinal < familyDepth; ++ordinal) {
    if (ordinal < computeFamilies.size())
      families.emplace_back(computeFamilies[ordinal]);
    if (ordinal < memoryFamilies.size())
      families.emplace_back(memoryFamilies[ordinal]);
  }

  std::vector<TechMatchRow> rows;
  std::vector<std::unique_ptr<TechMatchRowFamilyCursor>> cursors;
  cursors.reserve(families.size());
  for (const MatchRowFamily &family : families) {
    auto cursor = std::visit(
        [&](const auto &selectedFamily)
            -> llvm::Expected<std::unique_ptr<TechMatchRowFamilyCursor>> {
          using Family = std::decay_t<decltype(selectedFamily)>;
          if constexpr (std::is_same_v<
                            Family,
                            ::loom::fabric::FabricFuCapabilityTemplateRef>)
            return createComputeRowFamilyCursor(inputs, selected,
                                                selectedFamily);
          else
            return createMemoryRowFamilyCursor(inputs, selected,
                                               selectedFamily);
        },
        family);
    if (!cursor)
      return cursor.takeError();
    cursors.push_back(std::move(*cursor));
  }
  std::vector<bool> familyExhausted(families.size(), false);
  std::vector<std::uint64_t> familyVisits(families.size(), 0);
  bool interruptionObserved = false;
  const auto visitFamily = [&](std::size_t familyOrdinal,
                               std::uint64_t quota) -> llvm::Error {
    if (quota == 0)
      return llvm::Error::success();
    const std::uint64_t attemptsBefore = accounting.matchRowAttempts;
    TechMatchRowCollector collector(actors, attemptsBefore + quota, accounting,
                                    inputs.executionControl);
    if (familyVisits[familyOrdinal]++ != 0)
      ++accounting.matchRowCursorResumptions;
    if (llvm::Error error = cursors[familyOrdinal]->advance(collector))
      return error;
    if (collector.interrupted()) {
      interruptionObserved = true;
      return llvm::Error::success();
    }
    familyExhausted[familyOrdinal] = cursors[familyOrdinal]->exhausted();
    auto familyRows = collector.takeRows();
    if (!familyRows)
      return familyRows.takeError();
    rows.insert(rows.end(), std::make_move_iterator(familyRows->begin()),
                std::make_move_iterator(familyRows->end()));
    return llvm::Error::success();
  };

  const std::uint64_t familyCount = families.size();
  const std::uint64_t baseQuota =
      familyCount == 0 ? 0 : inputs.config.matchRowAttemptLimit() / familyCount;
  const std::uint64_t quotaRemainder =
      familyCount == 0 ? 0 : inputs.config.matchRowAttemptLimit() % familyCount;
  for (std::size_t ordinal = 0; ordinal < families.size(); ++ordinal) {
    if (interruptionObserved || inputs.executionControl.stopRequested()) {
      interruptionObserved = true;
      break;
    }
    const std::uint64_t quota =
        baseQuota + (static_cast<std::uint64_t>(ordinal) < quotaRemainder);
    if (llvm::Error error = visitFamily(ordinal, quota))
      return std::move(error);
  }

  while (accounting.matchRowAttempts < inputs.config.matchRowAttemptLimit()) {
    if (interruptionObserved || inputs.executionControl.stopRequested()) {
      interruptionObserved = true;
      break;
    }
    std::vector<std::size_t> openFamilies;
    for (auto [ordinal, exhausted] : llvm::enumerate(familyExhausted))
      if (!exhausted)
        openFamilies.push_back(ordinal);
    if (openFamilies.empty())
      break;

    const std::uint64_t remaining =
        inputs.config.matchRowAttemptLimit() - accounting.matchRowAttempts;
    const std::uint64_t quota = remaining / openFamilies.size();
    const std::uint64_t remainder = remaining % openFamilies.size();
    bool madeProgress = false;
    for (auto [rank, ordinal] : llvm::enumerate(openFamilies)) {
      if (interruptionObserved || inputs.executionControl.stopRequested()) {
        interruptionObserved = true;
        break;
      }
      const std::uint64_t familyQuota =
          quota + (static_cast<std::uint64_t>(rank) < remainder);
      const std::uint64_t attemptsBefore = accounting.matchRowAttempts;
      const bool exhaustedBefore = familyExhausted[ordinal];
      if (llvm::Error error = visitFamily(ordinal, familyQuota))
        return std::move(error);
      madeProgress |= accounting.matchRowAttempts != attemptsBefore ||
                      familyExhausted[ordinal] != exhaustedBefore;
    }
    if (!madeProgress)
      break;
  }

  const bool exhausted =
      llvm::all_of(familyExhausted, [](bool family) { return family; });
  const bool interrupted =
      interruptionObserved || inputs.executionControl.stopRequested();
  llvm::sort(rows, [](const TechMatchRow &lhs, const TechMatchRow &rhs) {
    return lhs.key < rhs.key;
  });
  rows.erase(std::unique(rows.begin(), rows.end(),
                         [](const TechMatchRow &lhs, const TechMatchRow &rhs) {
                           return lhs.key == rhs.key;
                         }),
             rows.end());
  return TechMatchDomain{std::move(actors), std::move(rows),
                         exhausted && !interrupted, interrupted};
}

} // namespace loom::mapping::detail
