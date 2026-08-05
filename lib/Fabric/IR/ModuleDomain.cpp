#include "Fabric/IR/ModuleDomain.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <limits>
#include <system_error>

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_module_domain_invalid: " +
                                     message.str());
}

std::uint32_t slotCount(ModuleDomainSlotCounts counts,
                        loom::fabric::FabricClockResetKind kind) {
  switch (kind) {
  case loom::fabric::FabricClockResetKind::Clock:
    return counts.clocks;
  case loom::fabric::FabricClockResetKind::Reset:
    return counts.resets;
  }
  return 0;
}

} // namespace

llvm::Error validateModuleInstanceDomainSlotBindings(
    ModuleDomainSlotCounts child, ModuleDomainSlotCounts parent,
    llvm::ArrayRef<ModuleInstanceDomainSlotBinding> bindings) {
  const std::uint64_t expectedCount =
      static_cast<std::uint64_t>(child.clocks) + child.resets;
  if (bindings.size() != expectedCount)
    return invalid("binding count does not equal the child slot count");

  std::size_t index = 0;
  for (loom::fabric::FabricClockResetKind kind :
       {loom::fabric::FabricClockResetKind::Clock,
        loom::fabric::FabricClockResetKind::Reset}) {
    for (std::uint32_t childOrdinal = 0; childOrdinal < slotCount(child, kind);
         ++childOrdinal, ++index) {
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
    fields.push_back(static_cast<std::uint32_t>(binding.kind));
    fields.push_back(binding.childSlotOrdinal);
    fields.push_back(binding.parentSlotOrdinal);
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
    for (std::size_t field = index; field < index + 3; ++field)
      if (fields[field] < 0 || static_cast<std::uint64_t>(fields[field]) >
                                   std::numeric_limits<std::uint32_t>::max())
        return invalid("binding property field is outside uint32 range");

    const std::uint32_t kind = static_cast<std::uint32_t>(fields[index]);
    if (kind >=
        loom::fabric::fabricClosedBound(loom::fabric::FabricClockResetKind{}))
      return invalid("binding property contains an unknown slot kind");
    bindings.push_back({
        static_cast<loom::fabric::FabricClockResetKind>(kind),
        static_cast<std::uint32_t>(fields[index + 1]),
        static_cast<std::uint32_t>(fields[index + 2]),
    });
  }
  return bindings;
}

} // namespace fabric
