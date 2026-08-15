#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <vector>

using namespace loom::pnr;

namespace {

constexpr std::size_t bitsPerWord = std::numeric_limits<std::uint64_t>::digits;

llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

bool contains(PnrIndex offset, PnrIndex count, PnrIndex value) {
  return value >= offset && value - offset < count;
}

llvm::Error checkedAdd(std::uint64_t &value, std::uint64_t amount,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    return candidateError(subject + " overflows u64");
  value += amount;
  return llvm::Error::success();
}

} // namespace

PnrIndex
SpatialCandidateState::resourceTimeEnvelopeRefcount(PnrIndex envelope) const {
  assert(envelope < resourceTimeEnvelopeRefcounts_.size());
  return resourceTimeEnvelopeRefcounts_[envelope];
}

bool SpatialCandidateState::resourceTimeEnvelopeActive(
    PnrIndex envelope) const {
  assert(envelope < resourceTimeEnvelopeRefcounts_.size());
  const std::size_t word = envelope / bitsPerWord;
  const std::size_t bit = envelope % bitsPerWord;
  return (activeResourceTimeEnvelopeBits_[word] & (UINT64_C(1) << bit)) != 0;
}

llvm::Expected<PnrIndex>
SpatialCandidateState::memoryServiceResourceTimeEnvelope(
    PnrIndex group, PnrIndex pattern) const {
  const auto offsets = problem_->capacity().memoryServiceGroupEnvelopeOffsets();
  const auto records = problem_->capacity().memoryServicePatternEnvelopes();
  if (group >= problem_->memory().serviceUseGroups().size() ||
      offsets.size() != problem_->memory().serviceUseGroups().size() + 1 ||
      offsets[group] > offsets[group + 1] ||
      offsets[group + 1] > records.size())
    return candidateError("memory service envelope index is incomplete");
  const auto choices =
      records.slice(offsets[group], offsets[group + 1] - offsets[group]);
  const auto found = std::lower_bound(
      choices.begin(), choices.end(), pattern,
      [](const FrozenSpatialMemoryServicePatternEnvelope &record,
         PnrIndex selectedPattern) {
        return record.pattern < selectedPattern;
      });
  if (found == choices.end() || found->pattern != pattern)
    return candidateError(
        "selected memory service pattern has no resource-time envelope");
  if (found->envelope >= problem_->capacity().resourceTimeEnvelopes().size())
    return candidateError("memory service envelope is out of range");
  return found->envelope;
}

llvm::Expected<std::vector<PnrIndex>>
SpatialCandidateState::deriveResourceTimeEnvelopeRefcounts() const {
  const auto &capacity = problem_->capacity();
  std::vector<PnrIndex> result(capacity.resourceTimeEnvelopes().size(), 0);
  const auto add = [&](PnrIndex envelope) -> llvm::Error {
    if (envelope >= result.size())
      return candidateError("selected resource-time envelope is out of range");
    if (result[envelope] == std::numeric_limits<PnrIndex>::max())
      return candidateError("resource-time envelope refcount overflows");
    ++result[envelope];
    return llvm::Error::success();
  };

  const auto contextOffsets =
      capacity.computeInstructionContextEnvelopeOffsets();
  for (const SpatialComputeBindingSelection &binding : computeBindings_) {
    if (contextOffsets.empty() ||
        binding.instructionContext >= contextOffsets.size() - 1 ||
        contextOffsets[binding.instructionContext] >
            contextOffsets[binding.instructionContext + 1] ||
        contextOffsets[binding.instructionContext + 1] > result.size())
      return candidateError(
          "compute binding has no resource-time envelope slice");
    for (PnrIndex envelope = contextOffsets[binding.instructionContext];
         envelope != contextOffsets[binding.instructionContext + 1]; ++envelope)
      if (llvm::Error error = add(envelope))
        return std::move(error);
  }

  const auto planEnvelopes = capacity.memoryOperationPlanEnvelopes();
  for (PnrIndex plan : memoryOperationPlans_) {
    if (plan >= planEnvelopes.size())
      return candidateError(
          "memory operation plan has no resource-time envelope");
    if (llvm::Error error = add(planEnvelopes[plan]))
      return std::move(error);
  }

  const auto optionPatterns = capacity.memoryDispatchOptionPatterns();
  const auto useGroups = problem_->memory().rootedUseServiceGroups();
  if (memoryUseDispatches_.size() != useGroups.size())
    return candidateError("memory service-use envelope domain is incomplete");
  for (PnrIndex use = 0; use < memoryUseDispatches_.size(); ++use) {
    const PnrIndex option = memoryUseDispatches_[use];
    if (option >= optionPatterns.size())
      return candidateError(
          "memory dispatch has no resource-time pattern projection");
    const PnrIndex pattern = optionPatterns[option];
    if (pattern == getInvalidPnrIndex())
      continue;
    const PnrIndex group = useGroups[use];
    if (group == getInvalidPnrIndex())
      return candidateError(
          "memory service pattern has no owner-derived group");
    auto envelope = memoryServiceResourceTimeEnvelope(group, pattern);
    if (!envelope)
      return envelope.takeError();
    result[*envelope] = 1;
  }
  return result;
}

llvm::Error SpatialCandidateState::rebuildResourceTimeEnvelopeSelections() {
  auto refcounts = deriveResourceTimeEnvelopeRefcounts();
  if (!refcounts)
    return refcounts.takeError();
  resourceTimeEnvelopeRefcounts_ = std::move(*refcounts);
  const std::size_t envelopeCount = resourceTimeEnvelopeRefcounts_.size();
  activeResourceTimeEnvelopeBits_.assign(
      envelopeCount / bitsPerWord + (envelopeCount % bitsPerWord != 0), 0);
  activeResourceTimeEnvelopeCount_ = 0;
  resourceReleaseLatencyCycles_ = 0;
  resourceMinimumInitiationIntervalCycles_ = 1;
  for (PnrIndex envelope = 0; envelope < resourceTimeEnvelopeRefcounts_.size();
       ++envelope) {
    if (resourceTimeEnvelopeRefcounts_[envelope] == 0)
      continue;
    const std::size_t word = envelope / bitsPerWord;
    const std::size_t bit = envelope % bitsPerWord;
    activeResourceTimeEnvelopeBits_[word] |= UINT64_C(1) << bit;
    if (activeResourceTimeEnvelopeCount_ ==
        std::numeric_limits<PnrIndex>::max())
      return candidateError("active resource-time envelope count overflows");
    ++activeResourceTimeEnvelopeCount_;
    const FrozenSpatialResourceTimeEnvelope &record =
        problem_->capacity().resourceTimeEnvelopes()[envelope];
    if (llvm::Error error = checkedAdd(
            resourceReleaseLatencyCycles_, record.releaseLatencyCycles,
            "resource release latency"))
      return error;
    resourceMinimumInitiationIntervalCycles_ =
        std::max(resourceMinimumInitiationIntervalCycles_,
                 record.minimumInitiationIntervalCycles);
  }
  return llvm::Error::success();
}

llvm::Error
SpatialCandidateState::verifyResourceTimeEnvelopeSelections() const {
  auto expected = deriveResourceTimeEnvelopeRefcounts();
  if (!expected)
    return expected.takeError();
  if (*expected != resourceTimeEnvelopeRefcounts_)
    return candidateError(
        "selected resource-time envelopes diverge from candidate decisions");
  const std::size_t expectedWordCount =
      expected->size() / bitsPerWord + (expected->size() % bitsPerWord != 0);
  if (activeResourceTimeEnvelopeBits_.size() != expectedWordCount)
    return candidateError("active resource-time bitset has the wrong shape");
  std::vector<std::uint64_t> expectedBits(expectedWordCount, 0);
  PnrIndex expectedCount = 0;
  std::uint64_t expectedRelease = 0;
  std::uint64_t expectedInterval = 1;
  for (PnrIndex envelope = 0; envelope < expected->size(); ++envelope) {
    if ((*expected)[envelope] == 0)
      continue;
    expectedBits[envelope / bitsPerWord] |= UINT64_C(1)
                                            << (envelope % bitsPerWord);
    ++expectedCount;
    const FrozenSpatialResourceTimeEnvelope &record =
        problem_->capacity().resourceTimeEnvelopes()[envelope];
    if (llvm::Error error = checkedAdd(expectedRelease,
                                       record.releaseLatencyCycles,
                                       "resource release latency"))
      return error;
    expectedInterval =
        std::max(expectedInterval, record.minimumInitiationIntervalCycles);
  }
  if (expectedBits != activeResourceTimeEnvelopeBits_ ||
      expectedCount != activeResourceTimeEnvelopeCount_ ||
      expectedRelease != resourceReleaseLatencyCycles_ ||
      expectedInterval != resourceMinimumInitiationIntervalCycles_)
    return candidateError(
        "active resource-time bitset diverges from envelope refcounts");
  return llvm::Error::success();
}

llvm::Error
SpatialCandidateState::applyResourceTimeEnvelopeDelta(PnrIndex envelope,
                                                      bool add) {
  if (envelope >= resourceTimeEnvelopeRefcounts_.size())
    return candidateError("resource-time envelope is out of range");
  PnrIndex &refcount = resourceTimeEnvelopeRefcounts_[envelope];
  const std::size_t word = envelope / bitsPerWord;
  const std::uint64_t mask = UINT64_C(1) << (envelope % bitsPerWord);
  const FrozenSpatialResourceTimeEnvelope &record =
      problem_->capacity().resourceTimeEnvelopes()[envelope];
  if (add) {
    if (refcount == std::numeric_limits<PnrIndex>::max())
      return candidateError("resource-time envelope refcount overflows");
    if (refcount == 0) {
      std::uint64_t release = resourceReleaseLatencyCycles_;
      if (llvm::Error error = checkedAdd(release, record.releaseLatencyCycles,
                                         "resource release latency"))
        return error;
      if (activeResourceTimeEnvelopeCount_ ==
          std::numeric_limits<PnrIndex>::max())
        return candidateError("active resource-time envelope count overflows");
      resourceReleaseLatencyCycles_ = release;
      resourceMinimumInitiationIntervalCycles_ =
          std::max(resourceMinimumInitiationIntervalCycles_,
                   record.minimumInitiationIntervalCycles);
    }
    if (refcount++ == 0) {
      activeResourceTimeEnvelopeBits_[word] |= mask;
      ++activeResourceTimeEnvelopeCount_;
    }
    return llvm::Error::success();
  }
  if (refcount == 0)
    return candidateError("resource-time envelope refcount underflows");
  if (refcount == 1) {
    if (resourceReleaseLatencyCycles_ < record.releaseLatencyCycles ||
        activeResourceTimeEnvelopeCount_ == 0)
      return candidateError("resource-time envelope totals underflow");
  }
  if (--refcount == 0) {
    activeResourceTimeEnvelopeBits_[word] &= ~mask;
    --activeResourceTimeEnvelopeCount_;
    resourceReleaseLatencyCycles_ -= record.releaseLatencyCycles;
    if (resourceMinimumInitiationIntervalCycles_ != 1 &&
        record.minimumInitiationIntervalCycles ==
            resourceMinimumInitiationIntervalCycles_) {
      resourceMinimumInitiationIntervalCycles_ = 1;
      for (PnrIndex active = 0;
           active < resourceTimeEnvelopeRefcounts_.size(); ++active) {
        if (resourceTimeEnvelopeRefcounts_[active] == 0)
          continue;
        resourceMinimumInitiationIntervalCycles_ = std::max(
            resourceMinimumInitiationIntervalCycles_,
            problem_->capacity()
                .resourceTimeEnvelopes()[active]
                .minimumInitiationIntervalCycles);
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::replaceResourceTimeEnvelopeSlice(
    PnrIndex oldOffset, PnrIndex oldCount, PnrIndex newOffset,
    PnrIndex newCount) {
  const PnrIndex size =
      static_cast<PnrIndex>(resourceTimeEnvelopeRefcounts_.size());
  if (oldOffset > size || oldCount > size - oldOffset || newOffset > size ||
      newCount > size - newOffset)
    return candidateError("resource-time envelope slice is out of range");
  if (oldOffset == newOffset && oldCount == newCount)
    return llvm::Error::success();
  for (PnrIndex envelope = oldOffset; envelope != oldOffset + oldCount;
       ++envelope)
    if (!contains(newOffset, newCount, envelope) &&
        resourceTimeEnvelopeRefcounts_[envelope] == 0)
      return candidateError("resource-time envelope refcount is incomplete");
  for (PnrIndex envelope = newOffset; envelope != newOffset + newCount;
       ++envelope)
    if (!contains(oldOffset, oldCount, envelope) &&
        resourceTimeEnvelopeRefcounts_[envelope] ==
            std::numeric_limits<PnrIndex>::max())
      return candidateError("resource-time envelope refcount overflows");
  for (PnrIndex envelope = oldOffset; envelope != oldOffset + oldCount;
       ++envelope)
    if (!contains(newOffset, newCount, envelope))
      if (llvm::Error error =
              applyResourceTimeEnvelopeDelta(envelope, false))
        return error;
  for (PnrIndex envelope = newOffset; envelope != newOffset + newCount;
       ++envelope)
    if (!contains(oldOffset, oldCount, envelope))
      if (llvm::Error error = applyResourceTimeEnvelopeDelta(envelope, true))
        return error;
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::replaceResourceTimeEnvelope(
    std::optional<PnrIndex> oldEnvelope, std::optional<PnrIndex> newEnvelope) {
  if (oldEnvelope == newEnvelope)
    return llvm::Error::success();
  if ((oldEnvelope && *oldEnvelope >= resourceTimeEnvelopeRefcounts_.size()) ||
      (newEnvelope && *newEnvelope >= resourceTimeEnvelopeRefcounts_.size()))
    return candidateError("resource-time envelope is out of range");
  if (oldEnvelope && resourceTimeEnvelopeRefcounts_[*oldEnvelope] == 0)
    return candidateError("resource-time envelope refcount is incomplete");
  if (newEnvelope && resourceTimeEnvelopeRefcounts_[*newEnvelope] ==
                         std::numeric_limits<PnrIndex>::max())
    return candidateError("resource-time envelope refcount overflows");
  if (oldEnvelope)
    if (llvm::Error error =
            applyResourceTimeEnvelopeDelta(*oldEnvelope, false))
      return error;
  if (newEnvelope)
    if (llvm::Error error = applyResourceTimeEnvelopeDelta(*newEnvelope, true))
      return error;
  return llvm::Error::success();
}
