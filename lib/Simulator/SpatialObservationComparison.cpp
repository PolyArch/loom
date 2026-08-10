#include "Simulator/SpatialObservationComparison.h"

#include "llvm/ADT/STLExtras.h"

namespace loom::sim {
namespace {

bool sameMemoryByte(const SemanticMemoryByte &lhs,
                    const SemanticMemoryByte &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != SemanticState::Defined || lhs.value == rhs.value);
}

bool sameValue(const CanonicalValueSequence &lhs,
               const CanonicalValueSequence &rhs) {
  return lhs.tokenCount == rhs.tokenCount && lhs.lanes == rhs.lanes;
}

bool sameValueResult(const ValueResultObservation &lhs,
                     const ValueResultObservation &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *published = std::get_if<PublishedValueResult>(&lhs))
    return sameValue(published->value,
                     std::get<PublishedValueResult>(rhs).value);
  return true;
}

bool sameStream(const CanonicalStreamSequence &lhs,
                const CanonicalStreamSequence &rhs) {
  return lhs.termination == rhs.termination &&
         sameValue(lhs.values, rhs.values);
}

bool sameMemory(const MemoryObservationPayload &lhs,
                const MemoryObservationPayload &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *full = std::get_if<FullMemoryObservation>(&lhs)) {
    const auto &other = std::get<FullMemoryObservation>(rhs);
    return full->bytes.size() == other.bytes.size() &&
           llvm::equal(full->bytes, other.bytes, sameMemoryByte);
  }
  const auto &diff = std::get<DiffMemoryObservation>(lhs);
  const auto &other = std::get<DiffMemoryObservation>(rhs);
  if (diff.byteCount != other.byteCount ||
      diff.runs.size() != other.runs.size())
    return false;
  return llvm::equal(diff.runs, other.runs,
                     [](const MemoryDiffRun &left, const MemoryDiffRun &right) {
                       return left.byteOffset == right.byteOffset &&
                              left.changedBytes.size() ==
                                  right.changedBytes.size() &&
                              llvm::equal(left.changedBytes, right.changedBytes,
                                          sameMemoryByte);
                     });
}

} // namespace

bool haveExactlyEqualSpatialFunctionalObservations(
    const SpatialFunctionalObservations &reference,
    const SpatialFunctionalObservations &candidate) {
  return reference.valueResults.size() == candidate.valueResults.size() &&
         reference.streamOutputs.size() == candidate.streamOutputs.size() &&
         reference.memories.size() == candidate.memories.size() &&
         llvm::equal(reference.valueResults, candidate.valueResults,
                     sameValueResult) &&
         llvm::equal(reference.streamOutputs, candidate.streamOutputs,
                     sameStream) &&
         llvm::equal(reference.memories, candidate.memories, sameMemory);
}

bool haveExactlyEqualSystemFunctionalObservations(
    const SystemFunctionalObservations &reference,
    const SystemFunctionalObservations &candidate) {
  return reference.valueResults.size() == candidate.valueResults.size() &&
         reference.externalValueOutputs.size() ==
             candidate.externalValueOutputs.size() &&
         reference.externalStreamOutputs.size() ==
             candidate.externalStreamOutputs.size() &&
         reference.memories.size() == candidate.memories.size() &&
         llvm::equal(reference.valueResults, candidate.valueResults,
                     sameValueResult) &&
         llvm::equal(reference.externalValueOutputs,
                     candidate.externalValueOutputs, sameValueResult) &&
         llvm::equal(reference.externalStreamOutputs,
                     candidate.externalStreamOutputs, sameStream) &&
         llvm::equal(reference.memories, candidate.memories, sameMemory);
}

} // namespace loom::sim
