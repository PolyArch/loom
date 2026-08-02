#include "TechMappingCandidateDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
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
  if (atLimit()) {
    truncated_ = true;
    return false;
  }
  if (previousSeedKey_ && key < *previousSeedKey_)
    return invalid("match-row seeds are not in canonical key order");
  ++accounting_.matchRowAttempts;
  previousSeedKey_ = key;
  activeSeedKey_ = std::move(key);
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
  std::vector<::dataflow::ActorRef> actors;
  actors.reserve(selected.size());
  for (const auto &actor : selected)
    actors.push_back(actor.ref);

  TechMatchRowCollector collector(actors, inputs.config.matchRowAttemptLimit(),
                                  accounting);
  if (llvm::Error error = deriveComputeRows(inputs, selected, collector))
    return std::move(error);
  if (!collector.truncated())
    if (llvm::Error error = deriveMemoryRows(inputs, selected, collector))
      return std::move(error);

  auto collectedRows = collector.takeRows();
  if (!collectedRows)
    return collectedRows.takeError();
  std::vector<TechMatchRow> rows = std::move(*collectedRows);
  llvm::sort(rows, [](const TechMatchRow &lhs, const TechMatchRow &rhs) {
    return lhs.key < rhs.key;
  });
  rows.erase(std::unique(rows.begin(), rows.end(),
                         [](const TechMatchRow &lhs, const TechMatchRow &rhs) {
                           return lhs.key == rhs.key;
                         }),
             rows.end());
  return TechMatchDomain{std::move(actors), std::move(rows),
                         !collector.truncated()};
}

} // namespace loom::mapping::detail
