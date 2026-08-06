#include "Fabric/IR/ModuleDomain.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"
#include <limits>

#include <cstddef>
#include <system_error>
#include <variant>

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_module_domain_invalid: " +
                                     message.str());
}

loom::fabric::FabricOrdinal slotCount(ModuleDomainSlotCounts counts,
                                      loom::fabric::FabricClockResetKind kind) {
  switch (kind) {
  case loom::fabric::FabricClockResetKind::Clock:
    return counts.clocks;
  case loom::fabric::FabricClockResetKind::Reset:
    return counts.resets;
  }
  return 0;
}

template <typename Ref>
bool isCanonicalStrictlyIncreasing(llvm::ArrayRef<Ref> values) {
  if (values.empty())
    return true;
  std::vector<std::uint8_t> previous =
      loom::fabric::canonicalFabricBytes(values.front());
  for (const Ref &value : values.drop_front()) {
    std::vector<std::uint8_t> current =
        loom::fabric::canonicalFabricBytes(value);
    if (!(previous < current))
      return false;
    previous = std::move(current);
  }
  return true;
}

} // namespace

llvm::Expected<ModuleDomainSlotCounts> validateModuleDomainRelation(
    loom::fabric::FabricModuleTemplateRef module,
    llvm::ArrayRef<loom::fabric::FabricModuleDomainSlotRef> slots,
    llvm::ArrayRef<loom::fabric::FabricModuleDomainMemberRef> members,
    llvm::ArrayRef<loom::fabric::ModuleDomainAssignment> assignments) {
  for (const loom::fabric::FabricModuleDomainSlotRef &slot : slots)
    if (static_cast<std::uint32_t>(slot.kind) >=
        loom::fabric::fabricClosedBound(loom::fabric::FabricClockResetKind{}))
      return invalid("slot inventory contains an unknown kind");
  if (!isCanonicalStrictlyIncreasing(slots))
    return invalid("slot inventory is not canonical sorted-unique");
  if (!isCanonicalStrictlyIncreasing(members))
    return invalid("derived member inventory is not canonical sorted-unique");
  if (!isCanonicalStrictlyIncreasing(assignments))
    return invalid("assignment relation is not canonical sorted-unique");

  ModuleDomainSlotCounts counts;
  for (loom::fabric::FabricClockResetKind kind :
       {loom::fabric::FabricClockResetKind::Clock,
        loom::fabric::FabricClockResetKind::Reset}) {
    loom::fabric::FabricOrdinal ordinal = 0;
    for (const loom::fabric::FabricModuleDomainSlotRef &slot : slots) {
      if (slot.kind != kind)
        continue;
      if (slot.module != module)
        return invalid("slot inventory names a foreign Module");
      if (slot.ordinal != ordinal)
        return invalid("slot inventory is not dense within its kind");
      ++ordinal;
    }
    if (kind == loom::fabric::FabricClockResetKind::Clock)
      counts.clocks = ordinal;
    else
      counts.resets = ordinal;
  }

  for (const loom::fabric::FabricModuleDomainMemberRef &member : members) {
    if (member.kind() != loom::fabric::FabricModuleDomainMemberKind::Boundary)
      continue;
    const auto &boundary =
        std::get<loom::fabric::FabricModuleBoundaryEndpointRef>(member.payload);
    if (boundary.module != module)
      return invalid("derived boundary member names a foreign Module");
  }

  if (members.size() > std::numeric_limits<std::size_t>::max() / 2 ||
      assignments.size() != members.size() * 2)
    return invalid("assignment count is not total over Module members");

  for (auto [index, member] : llvm::enumerate(members)) {
    const loom::fabric::ModuleDomainAssignment &clock = assignments[index * 2];
    const loom::fabric::ModuleDomainAssignment &reset =
        assignments[index * 2 + 1];
    if (clock.member != member || reset.member != member)
      return invalid("assignment relation does not match the member inventory");
    if (clock.slot.module != module || reset.slot.module != module)
      return invalid("assignment selects a foreign Module slot");
    if (clock.slot.kind != loom::fabric::FabricClockResetKind::Clock ||
        reset.slot.kind != loom::fabric::FabricClockResetKind::Reset)
      return invalid("each member must have one Clock and one Reset row");
    if (clock.slot.ordinal >= counts.clocks ||
        reset.slot.ordinal >= counts.resets)
      return invalid("assignment selects an out-of-range Module slot");
  }
  return counts;
}

llvm::Error validateModuleInstanceDomainSlotBindings(
    ModuleDomainSlotCounts child, ModuleDomainSlotCounts parent,
    llvm::ArrayRef<ModuleInstanceDomainSlotBinding> bindings) {
  if (child.clocks >
      std::numeric_limits<loom::fabric::FabricOrdinal>::max() - child.resets)
    return invalid("child slot count overflows the ordinal domain");
  const loom::fabric::FabricOrdinal expectedCount = child.clocks + child.resets;
  if (expectedCount > static_cast<loom::fabric::FabricOrdinal>(bindings.size()))
    return invalid("binding count does not equal the child slot count");
  if (bindings.size() != static_cast<std::size_t>(expectedCount))
    return invalid("binding count does not equal the child slot count");

  std::size_t index = 0;
  for (loom::fabric::FabricClockResetKind kind :
       {loom::fabric::FabricClockResetKind::Clock,
        loom::fabric::FabricClockResetKind::Reset}) {
    for (loom::fabric::FabricOrdinal childOrdinal = 0;
         childOrdinal < slotCount(child, kind); ++childOrdinal, ++index) {
      const ModuleInstanceDomainSlotBinding &binding = bindings[index];
      if (binding.kind != kind || binding.childSlotOrdinal != childOrdinal)
        return invalid(
            "bindings are not the canonical total child-slot relation");
      if (binding.parentSlotOrdinal >= slotCount(parent, kind))
        return invalid("binding selects an out-of-range parent slot");
    }
  }
  return llvm::Error::success();
}

mlir::DenseI64ArrayAttr encodeModuleInstanceDomainSlotBindings(
    mlir::MLIRContext *context,
    llvm::ArrayRef<ModuleInstanceDomainSlotBinding> bindings) {
  llvm::SmallVector<std::int64_t, 12> fields;
  fields.reserve(bindings.size() * 3);
  for (const ModuleInstanceDomainSlotBinding &binding : bindings) {
    fields.push_back(llvm::bit_cast<std::int64_t>(
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(binding.kind))));
    fields.push_back(llvm::bit_cast<std::int64_t>(binding.childSlotOrdinal));
    fields.push_back(llvm::bit_cast<std::int64_t>(binding.parentSlotOrdinal));
  }
  return mlir::DenseI64ArrayAttr::get(context, fields);
}

llvm::Expected<std::vector<ModuleInstanceDomainSlotBinding>>
decodeModuleInstanceDomainSlotBindings(mlir::DenseI64ArrayAttr encoded) {
  llvm::ArrayRef<std::int64_t> fields = encoded.asArrayRef();
  if (fields.size() % 3 != 0)
    return invalid("binding property does not contain complete triples");

  std::vector<ModuleInstanceDomainSlotBinding> bindings;
  bindings.reserve(fields.size() / 3);
  for (std::size_t index = 0; index < fields.size(); index += 3) {
    // Ordinal fields are bit-preserved through the signed i64 container, so
    // every FabricOrdinal bit pattern round-trips. Only the kind field is a
    // closed enum and is validated as one.
    const std::int64_t kindField = fields[index];
    if (kindField < 0 || llvm::bit_cast<std::uint64_t>(kindField) >=
                             loom::fabric::fabricClosedBound(
                                 loom::fabric::FabricClockResetKind{}))
      return invalid("binding property contains an unknown slot kind");
    bindings.push_back({
        static_cast<loom::fabric::FabricClockResetKind>(kindField),
        llvm::bit_cast<loom::fabric::FabricOrdinal>(fields[index + 1]),
        llvm::bit_cast<loom::fabric::FabricOrdinal>(fields[index + 2]),
    });
  }
  return bindings;
}

llvm::Expected<loom::fabric::FabricOrdinal>
ModuleDomainAuthoringRelation::declareSlot(
    loom::fabric::FabricClockResetKind kind) {
  switch (kind) {
  case loom::fabric::FabricClockResetKind::Clock:
    if (clockSlots_ ==
        std::numeric_limits<loom::fabric::FabricOrdinal>::max())
      return invalid("Clock slot inventory overflows the ordinal domain");
    return clockSlots_++;
  case loom::fabric::FabricClockResetKind::Reset:
    if (resetSlots_ ==
        std::numeric_limits<loom::fabric::FabricOrdinal>::max())
      return invalid("Reset slot inventory overflows the ordinal domain");
    return resetSlots_++;
  }
  return invalid("domain slot kind is outside the catalog");
}

loom::fabric::FabricOrdinal ModuleDomainAuthoringRelation::declaredSlotCount(
    loom::fabric::FabricClockResetKind kind) const {
  switch (kind) {
  case loom::fabric::FabricClockResetKind::Clock:
    return clockSlots_;
  case loom::fabric::FabricClockResetKind::Reset:
    return resetSlots_;
  }
  return 0;
}

llvm::Error ModuleDomainAuthoringRelation::noteInstanceBindings(
    mlir::Operation *instance,
    std::vector<ModuleInstanceDomainSlotBinding> rows) {
  if (!instance)
    return invalid("instance domain slot binding has no draft operation");
  // An explicit empty binding range is the slotless-Module form and records
  // no transient state.
  if (rows.empty())
    return llvm::Error::success();
  for (const InstanceBindingRecord &record : instanceBindings_)
    if (record.instance == instance)
      return invalid("instance domain slot binding is already recorded");
  instanceBindings_.push_back({instance, std::move(rows)});
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::noteInternalMember(
    mlir::Operation *owner, InternalMemberRole role,
    loom::fabric::FabricOrdinal subOrdinal) {
  if (!owner)
    return invalid("internal domain member has no draft operation");
  // Exhaustive catalog check: no value outside the five roles is admitted.
  if (role != InternalMemberRole::Occurrence &&
      role != InternalMemberRole::InstructionContext &&
      role != InternalMemberRole::FuNode &&
      role != InternalMemberRole::MemoryOperationPort &&
      role != InternalMemberRole::LocalMemoryService)
    return invalid("internal domain member role is outside the catalog");
  MemberKey key;
  key.internal = true;
  key.owner = owner;
  key.role = role;
  key.ordinal = subOrdinal;
  for (const MemberKey &existing : internalMembers_)
    if (existing == key)
      return invalid("internal domain member is already registered");
  internalMembers_.push_back(key);
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::assignOne(
    MemberKey member, loom::fabric::FabricClockResetKind slotKind,
    loom::fabric::FabricOrdinal slotOrdinal) {
  if (slotKind != loom::fabric::FabricClockResetKind::Clock &&
      slotKind != loom::fabric::FabricClockResetKind::Reset)
    return invalid("domain slot kind is outside the catalog");
  const loom::fabric::FabricOrdinal limit =
      slotCount({clockSlots_, resetSlots_}, slotKind);
  if (slotOrdinal >= limit)
    return invalid("assignment selects an out-of-range Module slot");
  for (const AssignmentRow &row : assignments_)
    if (row.member == member && row.slotKind == slotKind)
      return invalid("domain member already has an assignment for this slot "
                     "kind");
  assignments_.push_back({member, slotKind, slotOrdinal});
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::assignBoundary(
    loom::fabric::FabricPortDirection direction,
    loom::fabric::FabricOrdinal endpointOrdinal,
    loom::fabric::FabricClockResetKind slotKind,
    loom::fabric::FabricOrdinal slotOrdinal) {
  if (direction != loom::fabric::FabricPortDirection::Input &&
      direction != loom::fabric::FabricPortDirection::Output)
    return invalid(
        "boundary domain member direction is outside the catalog");
  MemberKey key;
  key.direction = direction;
  key.ordinal = endpointOrdinal;
  return assignOne(key, slotKind, slotOrdinal);
}

llvm::Error ModuleDomainAuthoringRelation::assignInternal(
    mlir::Operation *owner, InternalMemberRole role,
    loom::fabric::FabricOrdinal subOrdinal,
    loom::fabric::FabricClockResetKind slotKind,
    loom::fabric::FabricOrdinal slotOrdinal) {
  MemberKey key;
  key.internal = true;
  key.owner = owner;
  key.role = role;
  key.ordinal = subOrdinal;
  for (const MemberKey &existing : internalMembers_)
    if (existing == key)
      return assignOne(key, slotKind, slotOrdinal);
  return invalid("assignment names an unregistered internal domain member");
}

bool ModuleDomainAuthoringRelation::empty() const {
  return clockSlots_ == 0 && resetSlots_ == 0 && internalMembers_.empty() &&
         assignments_.empty() && instanceBindings_.empty();
}

llvm::Error ModuleDomainAuthoringRelation::validateTotality(
    loom::fabric::FabricOrdinal inputCount,
    loom::fabric::FabricOrdinal outputCount) const {
  const loom::fabric::FabricOrdinal max =
      std::numeric_limits<loom::fabric::FabricOrdinal>::max();
  if (internalMembers_.size() > max)
    return invalid("internal member inventory exceeds the ordinal domain");
  if (inputCount > max - outputCount)
    return invalid("boundary member inventory overflows the ordinal domain");
  const loom::fabric::FabricOrdinal boundaryCount = inputCount + outputCount;
  if (boundaryCount >
      max - static_cast<loom::fabric::FabricOrdinal>(internalMembers_.size()))
    return invalid("member inventory overflows the ordinal domain");
  const loom::fabric::FabricOrdinal memberCount =
      boundaryCount +
      static_cast<loom::fabric::FabricOrdinal>(internalMembers_.size());
  if (memberCount > max / 2)
    return invalid("member inventory overflows the assignment domain");
  if (memberCount > std::numeric_limits<std::size_t>::max() / 2)
    return invalid("member inventory exceeds the host container domain");
  // Per-member exactness needs no second accounting authority: rows enter
  // only through assignOne, which rejects a second same-kind row for one
  // member; the sweep below rejects any row naming a member outside the
  // signature or the registered internal inventory; and this cardinality
  // check requires exactly two rows per member. Together these imply exactly
  // one Clock row and one Reset row per valid member.
  if (assignments_.size() != static_cast<std::size_t>(memberCount) * 2)
    return invalid("domain assignments are not total over the Module member "
                   "inventory");
  for (const AssignmentRow &row : assignments_) {
    if (!row.member.internal) {
      loom::fabric::FabricOrdinal bound = 0;
      switch (row.member.direction) {
      case loom::fabric::FabricPortDirection::Input:
        bound = inputCount;
        break;
      case loom::fabric::FabricPortDirection::Output:
        bound = outputCount;
        break;
      }
      if (row.member.ordinal >= bound)
        return invalid("assignment names a boundary member outside the "
                       "signature");
      continue;
    }
    bool registered = false;
    for (const MemberKey &existing : internalMembers_)
      if (existing == row.member) {
        registered = true;
        break;
      }
    if (!registered)
      return invalid("assignment names an unregistered internal domain "
                     "member");
  }
  return llvm::Error::success();
}

} // namespace fabric
