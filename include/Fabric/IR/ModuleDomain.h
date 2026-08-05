#ifndef LOOM_FABRIC_IR_MODULEDOMAIN_H
#define LOOM_FABRIC_IR_MODULEDOMAIN_H

#include "Fabric/Identity/FabricRefs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace fabric {

/// The cardinality of one Module's dense symbolic Clock and Reset slot
/// inventories. This is a transient projection of the root-owned relation.
struct ModuleDomainSlotCounts final {
  std::uint32_t clocks = 0;
  std::uint32_t resets = 0;
};

/// One authoring-only child-to-parent slot row owned by a Module instance.
struct ModuleInstanceDomainSlotBinding final {
  loom::fabric::FabricClockResetKind kind =
      loom::fabric::FabricClockResetKind::Clock;
  std::uint32_t childSlotOrdinal = 0;
  std::uint32_t parentSlotOrdinal = 0;

  friend bool operator==(const ModuleInstanceDomainSlotBinding &left,
                         const ModuleInstanceDomainSlotBinding &right) {
    return left.kind == right.kind &&
           left.childSlotOrdinal == right.childSlotOrdinal &&
           left.parentSlotOrdinal == right.parentSlotOrdinal;
  }
  friend bool operator!=(const ModuleInstanceDomainSlotBinding &left,
                         const ModuleInstanceDomainSlotBinding &right) {
    return !(left == right);
  }
};

/// Validates one finalized Module's only symbolic domain relation and returns
/// the dense slot counts consumed by instance-edge validation. `members` is a
/// transient projection mechanically derived from the Module boundary and
/// canonical physical topology; it is not a second persistent inventory.
llvm::Expected<ModuleDomainSlotCounts> validateModuleDomainRelation(
    loom::fabric::FabricModuleTemplateRef module,
    llvm::ArrayRef<loom::fabric::FabricModuleDomainSlotRef> slots,
    llvm::ArrayRef<loom::fabric::FabricModuleDomainMemberRef> members,
    llvm::ArrayRef<loom::fabric::ModuleDomainAssignment> assignments);

/// Validates the exact canonical total relation for one Module instance.
/// Rows are ordered and unique by (kind, childSlotOrdinal). Several child
/// slots may select the same parent slot.
llvm::Error validateModuleInstanceDomainSlotBindings(
    ModuleDomainSlotCounts child, ModuleDomainSlotCounts parent,
    llvm::ArrayRef<ModuleInstanceDomainSlotBinding> bindings);

/// Encodes each binding as the flat, ordered triple
/// (kind, childSlotOrdinal, parentSlotOrdinal). The surrounding MLIR bytecode
/// remains the only persistent framing authority.
mlir::DenseI64ArrayAttr encodeModuleInstanceDomainSlotBindings(
    mlir::MLIRContext *context,
    llvm::ArrayRef<ModuleInstanceDomainSlotBinding> bindings);

/// Strictly decodes the flat binding property without canonicalizing it.
/// Relation order and totality are validated separately against both Modules.
llvm::Expected<std::vector<ModuleInstanceDomainSlotBinding>>
decodeModuleInstanceDomainSlotBindings(mlir::DenseI64ArrayAttr encoded);

} // namespace fabric

#endif // LOOM_FABRIC_IR_MODULEDOMAIN_H
