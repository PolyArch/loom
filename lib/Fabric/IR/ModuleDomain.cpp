#include "Fabric/IR/ModuleDomain.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/bit.h"
#include <algorithm>
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

llvm::Expected<std::vector<ModuleInstanceDomainSlotBinding>>
decodeInstanceBindings(mlir::Operation *operation) {
  auto instance = mlir::dyn_cast_or_null<fabric::InstantiateOp>(operation);
  if (!instance)
    return invalid("instance domain slot binding is not owned by a "
                   "fabric.instantiate operation");
  mlir::DenseI64ArrayAttr encoded = instance.getDomainSlotBindingsAttr();
  if (!encoded)
    return invalid("Module instance domain slot binding property is absent");
  auto rows = decodeModuleInstanceDomainSlotBindings(encoded);
  if (!rows)
    return rows.takeError();
  if (rows->empty())
    return invalid("Module instance domain slot binding is empty");
  return rows;
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

template <typename Ref>
mlir::ArrayAttr encodeRefRange(mlir::MLIRContext *context,
                               llvm::ArrayRef<Ref> values) {
  llvm::SmallVector<mlir::Attribute, 8> encoded;
  encoded.reserve(values.size());
  for (const Ref &value : values) {
    std::vector<std::uint8_t> bytes = loom::fabric::canonicalFabricBytes(value);
    llvm::SmallVector<std::int8_t, 32> signedBytes;
    signedBytes.reserve(bytes.size());
    for (std::uint8_t byte : bytes)
      signedBytes.push_back(static_cast<std::int8_t>(byte));
    encoded.push_back(mlir::DenseI8ArrayAttr::get(context, signedBytes));
  }
  return mlir::ArrayAttr::get(context, encoded);
}

template <typename Ref>
llvm::Expected<std::vector<Ref>> decodeRefRange(mlir::ArrayAttr encoded,
                                                const llvm::Twine &name) {
  if (!encoded)
    return invalid(name + " carrier is absent");
  std::vector<Ref> values;
  values.reserve(encoded.size());
  for (auto [ordinal, attribute] : llvm::enumerate(encoded)) {
    auto bytes = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attribute);
    if (!bytes)
      return invalid(name + " row #" + llvm::Twine(ordinal) +
                     " is not a canonical byte-array record");
    std::vector<std::uint8_t> unsignedBytes;
    unsignedBytes.reserve(bytes.size());
    for (std::int8_t byte : bytes.asArrayRef())
      unsignedBytes.push_back(static_cast<std::uint8_t>(byte));
    auto value = loom::fabric::decodeFabricRef<Ref>(unsignedBytes);
    if (!value)
      return value.takeError();
    if (loom::fabric::canonicalFabricBytes(*value) != unsignedBytes)
      return invalid(name + " row is not canonically encoded");
    values.push_back(std::move(*value));
  }
  return values;
}

} // namespace

mlir::ArrayAttr encodeModuleDomainSlots(
    mlir::MLIRContext *context,
    llvm::ArrayRef<loom::fabric::FabricModuleDomainSlotRef> slots) {
  return encodeRefRange(context, slots);
}

llvm::Expected<std::vector<loom::fabric::FabricModuleDomainSlotRef>>
decodeModuleDomainSlots(mlir::ArrayAttr encoded) {
  return decodeRefRange<loom::fabric::FabricModuleDomainSlotRef>(
      encoded, "Module domain slot inventory");
}

mlir::ArrayAttr encodeModuleDomainAssignments(
    mlir::MLIRContext *context,
    llvm::ArrayRef<loom::fabric::ModuleDomainAssignment> assignments) {
  return encodeRefRange(context, assignments);
}

llvm::Expected<std::vector<loom::fabric::ModuleDomainAssignment>>
decodeModuleDomainAssignments(mlir::ArrayAttr encoded) {
  return decodeRefRange<loom::fabric::ModuleDomainAssignment>(
      encoded, "Module domain assignment relation");
}

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
  if (counts.clocks == 0 || counts.resets == 0)
    return invalid("Module domain relation requires Clock and Reset slots");

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
  if (child.clocks == 0 || child.resets == 0 || parent.clocks == 0 ||
      parent.resets == 0)
    return invalid("Module instance requires Clock and Reset slot inventories");
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
  if (defaultAssignments_)
    return invalid("default Module domain assignments are already active");
  switch (kind) {
  case loom::fabric::FabricClockResetKind::Clock:
    if (clockSlots_ == std::numeric_limits<loom::fabric::FabricOrdinal>::max())
      return invalid("Clock slot inventory overflows the ordinal domain");
    return clockSlots_++;
  case loom::fabric::FabricClockResetKind::Reset:
    if (resetSlots_ == std::numeric_limits<loom::fabric::FabricOrdinal>::max())
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
    mlir::Operation *instance, const ModuleDomainAuthoringRelation &child) {
  if (!instance)
    return invalid("instance domain slot binding has no draft operation");
  auto rows = decodeInstanceBindings(instance);
  if (!rows)
    return rows.takeError();
  const ModuleDomainSlotCounts childCounts{
      child.declaredSlotCount(loom::fabric::FabricClockResetKind::Clock),
      child.declaredSlotCount(loom::fabric::FabricClockResetKind::Reset)};
  const ModuleDomainSlotCounts parentCounts{
      declaredSlotCount(loom::fabric::FabricClockResetKind::Clock),
      declaredSlotCount(loom::fabric::FabricClockResetKind::Reset)};
  if (llvm::Error error = validateModuleInstanceDomainSlotBindings(
          childCounts, parentCounts, *rows))
    return error;
  for (const InstanceBindingRecord &record : instanceBindings_)
    if (record.instance == instance)
      return invalid("instance domain slot binding is already recorded");
  instanceBindings_.push_back(
      {instance, std::make_shared<ModuleDomainAuthoringRelation>(child)});
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::ensureDefaultAssignments(
    loom::fabric::FabricOrdinal inputCount,
    loom::fabric::FabricOrdinal outputCount) {
  if (defaultAssignments_)
    return llvm::Error::success();
  if (clockSlots_ != 0 || resetSlots_ != 0 || !assignments_.empty())
    return llvm::Error::success();

  defaultAssignments_ = true;
  clockSlots_ = 1;
  resetSlots_ = 1;
  for (loom::fabric::FabricOrdinal ordinal = 0; ordinal < inputCount;
       ++ordinal) {
    if (llvm::Error error =
            assignBoundary(loom::fabric::FabricPortDirection::Input, ordinal,
                           loom::fabric::FabricClockResetKind::Clock, 0))
      return error;
    if (llvm::Error error =
            assignBoundary(loom::fabric::FabricPortDirection::Input, ordinal,
                           loom::fabric::FabricClockResetKind::Reset, 0))
      return error;
  }
  for (loom::fabric::FabricOrdinal ordinal = 0; ordinal < outputCount;
       ++ordinal) {
    if (llvm::Error error =
            assignBoundary(loom::fabric::FabricPortDirection::Output, ordinal,
                           loom::fabric::FabricClockResetKind::Clock, 0))
      return error;
    if (llvm::Error error =
            assignBoundary(loom::fabric::FabricPortDirection::Output, ordinal,
                           loom::fabric::FabricClockResetKind::Reset, 0))
      return error;
  }
  for (const MemberKey &member : internalMembers_) {
    if (llvm::Error error =
            assignOne(member, loom::fabric::FabricClockResetKind::Clock, 0))
      return error;
    if (llvm::Error error =
            assignOne(member, loom::fabric::FabricClockResetKind::Reset, 0))
      return error;
  }
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
  if (subOrdinal != 0 && (role == InternalMemberRole::Occurrence ||
                          role == InternalMemberRole::FuNode ||
                          role == InternalMemberRole::LocalMemoryService))
    return invalid("internal domain member role does not take a sub-ordinal");
  MemberKey key;
  key.internal = true;
  key.owner = owner;
  key.role = role;
  key.ordinal = subOrdinal;
  for (const MemberKey &existing : internalMembers_)
    if (existing == key)
      return invalid("internal domain member is already registered");
  internalMembers_.push_back(key);
  if (defaultAssignments_) {
    if (llvm::Error error =
            assignOne(key, loom::fabric::FabricClockResetKind::Clock, 0))
      return error;
    if (llvm::Error error =
            assignOne(key, loom::fabric::FabricClockResetKind::Reset, 0))
      return error;
  }
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
    return invalid("boundary domain member direction is outside the catalog");
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
         assignments_.empty() && instanceBindings_.empty() &&
         !defaultAssignments_;
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

llvm::Expected<ModuleDomainAuthoringRelation>
ModuleDomainAuthoringRelation::remap(const mlir::IRMapping &mapping) const {
  ModuleDomainAuthoringRelation result;
  result.clockSlots_ = clockSlots_;
  result.resetSlots_ = resetSlots_;
  result.defaultAssignments_ = defaultAssignments_;
  result.internalMembers_.reserve(internalMembers_.size());
  result.assignments_.reserve(assignments_.size());
  result.instanceBindings_.reserve(instanceBindings_.size());

  const auto remapMember = [&](MemberKey member) -> llvm::Expected<MemberKey> {
    if (!member.internal)
      return member;
    member.owner = mapping.lookupOrNull(member.owner);
    if (!member.owner)
      return invalid("domain member is missing from the canonical clone map");
    return member;
  };
  for (const MemberKey &member : internalMembers_) {
    auto mapped = remapMember(member);
    if (!mapped)
      return mapped.takeError();
    result.internalMembers_.push_back(*mapped);
  }
  for (const AssignmentRow &assignment : assignments_) {
    auto mapped = remapMember(assignment.member);
    if (!mapped)
      return mapped.takeError();
    result.assignments_.push_back(
        {*mapped, assignment.slotKind, assignment.slotOrdinal});
  }
  for (const InstanceBindingRecord &binding : instanceBindings_) {
    mlir::Operation *instance = mapping.lookupOrNull(binding.instance);
    if (!instance)
      return invalid("Module instance is missing from the canonical clone map");
    if (!binding.child)
      return invalid("Module instance has no child domain relation");
    auto child = binding.child->remap(mapping);
    if (!child)
      return child.takeError();
    result.instanceBindings_.push_back(
        {instance,
         std::make_shared<ModuleDomainAuthoringRelation>(std::move(*child))});
  }
  return result;
}

void ModuleDomainAuthoringRelation::remapMappedOperations(
    const mlir::IRMapping &mapping) {
  const auto remap = [&](MemberKey &member) {
    if (!member.internal)
      return;
    if (mlir::Operation *mapped = mapping.lookupOrNull(member.owner))
      member.owner = mapped;
  };
  for (MemberKey &member : internalMembers_)
    remap(member);
  for (AssignmentRow &assignment : assignments_)
    remap(assignment.member);
  for (InstanceBindingRecord &binding : instanceBindings_)
    if (mlir::Operation *mapped = mapping.lookupOrNull(binding.instance))
      binding.instance = mapped;
}

llvm::Error ModuleDomainAuthoringRelation::replicateMappedOperations(
    const mlir::IRMapping &mapping) {
  std::vector<MemberKey> replicatedMembers;
  for (const MemberKey &member : internalMembers_) {
    mlir::Operation *mapped =
        member.internal ? mapping.lookupOrNull(member.owner) : nullptr;
    if (!mapped)
      continue;
    MemberKey replicated = member;
    replicated.owner = mapped;
    if (llvm::is_contained(internalMembers_, replicated) ||
        llvm::is_contained(replicatedMembers, replicated))
      return invalid("replicated Module domain member already exists");
    replicatedMembers.push_back(replicated);
  }

  std::vector<AssignmentRow> replicatedAssignments;
  for (const AssignmentRow &assignment : assignments_) {
    mlir::Operation *mapped =
        assignment.member.internal
            ? mapping.lookupOrNull(assignment.member.owner)
            : nullptr;
    if (!mapped)
      continue;
    AssignmentRow replicated = assignment;
    replicated.member.owner = mapped;
    replicatedAssignments.push_back(replicated);
  }

  internalMembers_.insert(internalMembers_.end(), replicatedMembers.begin(),
                          replicatedMembers.end());
  assignments_.insert(assignments_.end(), replicatedAssignments.begin(),
                      replicatedAssignments.end());
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::eraseOperations(
    llvm::ArrayRef<mlir::Operation *> operations) {
  if (llvm::is_contained(operations, nullptr))
    return invalid("erased Module domain operation is null");
  const auto erased = [&](mlir::Operation *operation) {
    return llvm::is_contained(operations, operation);
  };
  internalMembers_.erase(
      std::remove_if(internalMembers_.begin(), internalMembers_.end(),
                     [&](const MemberKey &member) {
                       return member.internal && erased(member.owner);
                     }),
      internalMembers_.end());
  assignments_.erase(std::remove_if(assignments_.begin(), assignments_.end(),
                                    [&](const AssignmentRow &row) {
                                      return row.member.internal &&
                                             erased(row.member.owner);
                                    }),
                     assignments_.end());
  instanceBindings_.erase(
      std::remove_if(instanceBindings_.begin(), instanceBindings_.end(),
                     [&](const InstanceBindingRecord &record) {
                       return erased(record.instance);
                     }),
      instanceBindings_.end());
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::resizeInternalMembers(
    mlir::Operation *owner, InternalMemberRole role,
    loom::fabric::FabricOrdinal oldCount, loom::fabric::FabricOrdinal newCount,
    loom::fabric::FabricOrdinal prototypeOrdinal) {
  if (!owner)
    return invalid("resized Module domain owner is null");
  if (oldCount == 0)
    return invalid("resized Module domain inventory has no source member");
  if (prototypeOrdinal >= oldCount)
    return invalid("Module domain prototype member is out of range");

  for (loom::fabric::FabricOrdinal ordinal = 0; ordinal < oldCount; ++ordinal) {
    MemberKey member;
    member.internal = true;
    member.owner = owner;
    member.role = role;
    member.ordinal = ordinal;
    if (!llvm::is_contained(internalMembers_, member))
      return invalid("resized Module domain inventory is not dense");
  }

  MemberKey prototype;
  prototype.internal = true;
  prototype.owner = owner;
  prototype.role = role;
  prototype.ordinal = prototypeOrdinal;
  std::vector<AssignmentRow> prototypeAssignments;
  for (const AssignmentRow &row : assignments_)
    if (row.member == prototype)
      prototypeAssignments.push_back(row);
  if (prototypeAssignments.empty())
    return invalid("Module domain prototype member has no assignments");

  if (newCount < oldCount) {
    internalMembers_.erase(
        std::remove_if(internalMembers_.begin(), internalMembers_.end(),
                       [&](const MemberKey &member) {
                         return member.internal && member.owner == owner &&
                                member.role == role &&
                                member.ordinal >= newCount;
                       }),
        internalMembers_.end());
    assignments_.erase(std::remove_if(assignments_.begin(), assignments_.end(),
                                      [&](const AssignmentRow &row) {
                                        return row.member.internal &&
                                               row.member.owner == owner &&
                                               row.member.role == role &&
                                               row.member.ordinal >= newCount;
                                      }),
                       assignments_.end());
  }
  for (loom::fabric::FabricOrdinal ordinal = oldCount; ordinal < newCount;
       ++ordinal) {
    MemberKey member = prototype;
    member.ordinal = ordinal;
    internalMembers_.push_back(member);
    for (const AssignmentRow &row : prototypeAssignments)
      assignments_.push_back({member, row.slotKind, row.slotOrdinal});
  }
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::truncateBoundaryMembers(
    loom::fabric::FabricPortDirection direction,
    loom::fabric::FabricOrdinal oldCount,
    loom::fabric::FabricOrdinal newCount) {
  if (newCount > oldCount)
    return invalid("Module boundary growth requires explicit domain rows");
  assignments_.erase(std::remove_if(assignments_.begin(), assignments_.end(),
                                    [&](const AssignmentRow &row) {
                                      return !row.member.internal &&
                                             row.member.direction ==
                                                 direction &&
                                             row.member.ordinal >= newCount;
                                    }),
                     assignments_.end());
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::composeInstance(
    mlir::Operation *instance, const mlir::IRMapping &childCloneMapping) {
  auto record = llvm::find_if(instanceBindings_, [&](const auto &candidate) {
    return candidate.instance == instance;
  });
  if (record == instanceBindings_.end())
    return invalid("elaborated Module instance has no domain binding record");
  const std::size_t recordIndex =
      static_cast<std::size_t>(record - instanceBindings_.begin());
  const InstanceBindingRecord active = *record;
  if (!active.child)
    return invalid("elaborated Module instance has no child domain relation");
  auto activeRows = decodeInstanceBindings(instance);
  if (!activeRows)
    return activeRows.takeError();
  const ModuleDomainSlotCounts activeChildCounts{
      active.child->declaredSlotCount(
          loom::fabric::FabricClockResetKind::Clock),
      active.child->declaredSlotCount(
          loom::fabric::FabricClockResetKind::Reset)};
  const ModuleDomainSlotCounts activeParentCounts{
      declaredSlotCount(loom::fabric::FabricClockResetKind::Clock),
      declaredSlotCount(loom::fabric::FabricClockResetKind::Reset)};
  if (llvm::Error error = validateModuleInstanceDomainSlotBindings(
          activeChildCounts, activeParentCounts, *activeRows))
    return error;

  const auto mapSlot = [&](loom::fabric::FabricClockResetKind kind,
                           loom::fabric::FabricOrdinal childOrdinal)
      -> llvm::Expected<loom::fabric::FabricOrdinal> {
    for (const ModuleInstanceDomainSlotBinding &binding : *activeRows)
      if (binding.kind == kind && binding.childSlotOrdinal == childOrdinal)
        return binding.parentSlotOrdinal;
    return invalid("child Module assignment names an unbound domain slot");
  };
  const auto mapMember = [&](MemberKey member) -> llvm::Expected<MemberKey> {
    if (!member.internal)
      return member;
    member.owner = childCloneMapping.lookupOrNull(member.owner);
    if (!member.owner)
      return invalid("child domain member is absent from instance clone map");
    return member;
  };

  const ModuleDomainAuthoringRelation &child = *active.child;
  for (const MemberKey &member : child.internalMembers_) {
    auto mapped = mapMember(member);
    if (!mapped)
      return mapped.takeError();
    internalMembers_.push_back(*mapped);
  }
  for (const AssignmentRow &assignment : child.assignments_) {
    if (!assignment.member.internal)
      continue;
    auto member = mapMember(assignment.member);
    if (!member)
      return member.takeError();
    auto slot = mapSlot(assignment.slotKind, assignment.slotOrdinal);
    if (!slot)
      return slot.takeError();
    assignments_.push_back({*member, assignment.slotKind, *slot});
  }
  for (const InstanceBindingRecord &nested : child.instanceBindings_) {
    mlir::Operation *mappedInstance =
        childCloneMapping.lookupOrNull(nested.instance);
    if (!mappedInstance)
      return invalid("nested Module instance is absent from clone map");
    if (!nested.child)
      return invalid("nested Module instance has no child domain relation");
    auto nestedRows = decodeInstanceBindings(mappedInstance);
    if (!nestedRows)
      return nestedRows.takeError();
    const ModuleDomainSlotCounts nestedChildCounts{
        nested.child->declaredSlotCount(
            loom::fabric::FabricClockResetKind::Clock),
        nested.child->declaredSlotCount(
            loom::fabric::FabricClockResetKind::Reset)};
    const ModuleDomainSlotCounts nestedParentCounts{
        child.declaredSlotCount(loom::fabric::FabricClockResetKind::Clock),
        child.declaredSlotCount(loom::fabric::FabricClockResetKind::Reset)};
    if (llvm::Error error = validateModuleInstanceDomainSlotBindings(
            nestedChildCounts, nestedParentCounts, *nestedRows))
      return error;
    std::vector<ModuleInstanceDomainSlotBinding> remappedRows;
    remappedRows.reserve(nestedRows->size());
    for (const ModuleInstanceDomainSlotBinding &row : *nestedRows) {
      auto parent = mapSlot(row.kind, row.parentSlotOrdinal);
      if (!parent)
        return parent.takeError();
      remappedRows.push_back({row.kind, row.childSlotOrdinal, *parent});
    }
    auto mappedOp = mlir::cast<fabric::InstantiateOp>(mappedInstance);
    mappedOp.setDomainSlotBindingsAttr(encodeModuleInstanceDomainSlotBindings(
        mappedOp.getContext(), remappedRows));
    instanceBindings_.push_back({mappedInstance, nested.child});
  }
  instanceBindings_.erase(instanceBindings_.begin() + recordIndex);
  return llvm::Error::success();
}

llvm::Error ModuleDomainAuthoringRelation::visitAssignments(
    BoundaryAssignmentVisitor boundary,
    InternalAssignmentVisitor internal) const {
  if (!instanceBindings_.empty())
    return invalid("Module instance domain rows were not composed before "
                   "assignment visitation");
  for (const AssignmentRow &row : assignments_) {
    llvm::Error error =
        row.member.internal
            ? internal(row.member.owner, row.member.role, row.member.ordinal,
                       row.slotKind, row.slotOrdinal)
            : boundary(row.member.direction, row.member.ordinal, row.slotKind,
                       row.slotOrdinal);
    if (error)
      return error;
  }
  return llvm::Error::success();
}

} // namespace fabric
