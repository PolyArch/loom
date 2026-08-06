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

} // namespace fabric
