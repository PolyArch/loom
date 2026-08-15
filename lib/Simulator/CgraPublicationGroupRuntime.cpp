#include "CgraTransportRuntime.h"

#include <limits>
#include <system_error>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

llvm::Expected<std::uint64_t> CgraTransportRuntime::allocatePublicationGroup(
    llvm::ArrayRef<std::uint64_t> slots) {
  if (slots.size() < 2)
    return invalid("CGRA atomic publication group has fewer than two tokens");
  std::uint64_t groupOrdinal = 0;
  if (freePublicationGroups_.empty()) {
    if (publicationGroups_.size() ==
        std::numeric_limits<std::uint64_t>::max())
      return llvm::createStringError(
          std::errc::value_too_large,
          "CGRA publication group ordinal overflows u64");
    groupOrdinal = publicationGroups_.size();
    publicationGroups_.emplace_back();
  } else {
    groupOrdinal = freePublicationGroups_.back();
    freePublicationGroups_.pop_back();
  }
  PublicationGroup &group = publicationGroups_[groupOrdinal];
  if (group.active)
    return invalid("CGRA publication group slot is already active");
  group.active = true;
  group.transferSlots.assign(slots.begin(), slots.end());
  for (std::uint64_t slot : slots) {
    if (slot >= inFlight_.size() || !inFlight_[slot].active ||
        inFlight_[slot].publicationGroup != invalidCgraTransportOrdinal)
      return invalid("CGRA publication group names an invalid token");
    inFlight_[slot].publicationGroup = groupOrdinal;
  }
  return groupOrdinal;
}

llvm::Expected<bool> CgraTransportRuntime::tryPublishPublicationGroup(
    std::uint64_t groupOrdinal, CgraTransportFrame &frame) {
  if (groupOrdinal >= publicationGroups_.size() ||
      !publicationGroups_[groupOrdinal].active)
    return invalid("CGRA publication event names an inactive atomic group");
  PublicationGroup &group = publicationGroups_[groupOrdinal];
  for (std::uint64_t slot : group.transferSlots) {
    if (slot >= inFlight_.size() || !inFlight_[slot].active ||
        inFlight_[slot].publicationGroup != groupOrdinal ||
        inFlight_[slot].published)
      return invalid("CGRA atomic publication group has a stale token");
    if (!inFlight_[slot].publicationReady)
      return false;
  }

  bool ready = true;
  for (std::uint64_t slot : group.transferSlots) {
    const InFlight &inFlight = inFlight_[slot];
    ready &= canPublish(bindings_[inFlight.bindingOrdinal],
                        inFlight.operandCapacityReserved);
  }
  if (!ready) {
    for (std::uint64_t slot : group.transferSlots) {
      InFlight &inFlight = inFlight_[slot];
      inFlight.publicationReady = false;
      if (!blocked_.test(inFlight.bindingOrdinal))
        frame.blockedTransfers.push_back(inFlight.bindingOrdinal);
      blocked_.set(inFlight.bindingOrdinal);
    }
    return false;
  }

  for (std::uint64_t slot : group.transferSlots)
    if (llvm::Error error = commitOperandQueueEnqueue(slot))
      return std::move(error);
  for (std::uint64_t slot : group.transferSlots)
    publish(slot, frame);
  group.active = false;
  group.transferSlots.clear();
  freePublicationGroups_.push_back(groupOrdinal);
  return true;
}

} // namespace loom::sim::detail
