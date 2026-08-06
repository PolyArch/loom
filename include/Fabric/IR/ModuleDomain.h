#ifndef LOOM_FABRIC_IR_MODULEDOMAIN_H
#define LOOM_FABRIC_IR_MODULEDOMAIN_H

#include "Fabric/Identity/FabricRefs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace mlir {
class Operation;
}

namespace fabric {

/// The cardinality of one Module's dense symbolic Clock and Reset slot
/// inventories. This is a transient projection of the root-owned relation.
struct ModuleDomainSlotCounts final {
  loom::fabric::FabricOrdinal clocks = 0;
  loom::fabric::FabricOrdinal resets = 0;
};

/// One authoring-only child-to-parent slot row owned by a Module instance.
struct ModuleInstanceDomainSlotBinding final {
  loom::fabric::FabricClockResetKind kind =
      loom::fabric::FabricClockResetKind::Clock;
  loom::fabric::FabricOrdinal childSlotOrdinal = 0;
  loom::fabric::FabricOrdinal parentSlotOrdinal = 0;

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

/// One phase-local typed authoring relation for a Module's symbolic Clock and
/// Reset slots and their member assignments. It is the sole authoring
/// representation before canonical labeling: the persistent
/// `domain_slots`/`domain_assignments` wire does not exist yet, and the
/// relation is consumed exactly once inside the canonical-candidate pipeline
/// and destroyed. Boundary members are identified by direction and endpoint
/// ordinal; internal owners by their exact live draft operation and typed
/// subrole. Operation pointers are Builder-lifetime-only: they exist only
/// inside this in-process relation and never enter bytes, attributes, or
/// persistent references. The canonical consumer must remap owner operations
/// across clone and instance remapping before consumption, and draft
/// operations are never erased while the relation is live.
class ModuleDomainAuthoringRelation final {
public:
  enum class InternalMemberRole : std::uint8_t {
    Occurrence,
    InstructionContext,
    FuNode,
    MemoryOperationPort,
    LocalMemoryService,
  };

  /// Declares one slot and returns its dense ordinal within its kind.
  llvm::Expected<loom::fabric::FabricOrdinal>
  declareSlot(loom::fabric::FabricClockResetKind kind);

  /// The number of declared slots of one kind.
  loom::fabric::FabricOrdinal
  declaredSlotCount(loom::fabric::FabricClockResetKind kind) const;

  /// Records one validated total slot correspondence for one Module instance
  /// edge, keyed by the live draft `fabric.instantiate` operation. The rows
  /// are the canonical relation validated against both slot inventories; the
  /// keyed operation is remapped across clone and instance remapping before
  /// the single canonical consumption materializes it.
  llvm::Error
  noteInstanceBindings(mlir::Operation *instance,
                       std::vector<ModuleInstanceDomainSlotBinding> rows);

  struct InstanceBindingRecord final {
    mlir::Operation *instance = nullptr;
    std::vector<ModuleInstanceDomainSlotBinding> rows;
  };
  llvm::ArrayRef<InstanceBindingRecord> instanceBindings() const {
    return instanceBindings_;
  }

  /// True once any slot, assignment, or instance binding has been authored.
  /// Internal member registration alone is authoring support and does not
  /// activate the domain carrier.
  bool hasDomainAuthoring() const {
    return clockSlots_ != 0 || resetSlots_ != 0 || !assignments_.empty() ||
           !instanceBindings_.empty();
  }

  /// Registers one internal owner member created by a construction call.
  llvm::Error
  noteInternalMember(mlir::Operation *owner, InternalMemberRole role,
                     loom::fabric::FabricOrdinal subOrdinal);

  llvm::Error
  assignBoundary(loom::fabric::FabricPortDirection direction,
                 loom::fabric::FabricOrdinal endpointOrdinal,
                 loom::fabric::FabricClockResetKind slotKind,
                 loom::fabric::FabricOrdinal slotOrdinal);
  llvm::Error assignInternal(mlir::Operation *owner, InternalMemberRole role,
                             loom::fabric::FabricOrdinal subOrdinal,
                             loom::fabric::FabricClockResetKind slotKind,
                             loom::fabric::FabricOrdinal slotOrdinal);

  /// No slots declared, no members registered, no assignments made, and no
  /// instance bindings recorded.
  bool empty() const;

  /// Validates exact totality over the boundary members implied by the
  /// signature and every registered internal member: exactly one Clock and
  /// one Reset assignment per member, dense slots, and no extra assignment.
  llvm::Error
  validateTotality(loom::fabric::FabricOrdinal inputCount,
                   loom::fabric::FabricOrdinal outputCount) const;

private:
  struct MemberKey final {
    bool internal = false;
    loom::fabric::FabricPortDirection direction =
        loom::fabric::FabricPortDirection::Input;
    mlir::Operation *owner = nullptr;
    InternalMemberRole role = InternalMemberRole::Occurrence;
    loom::fabric::FabricOrdinal ordinal = 0;

    friend bool operator==(const MemberKey &lhs, const MemberKey &rhs) {
      return lhs.internal == rhs.internal && lhs.direction == rhs.direction &&
             lhs.owner == rhs.owner && lhs.role == rhs.role &&
             lhs.ordinal == rhs.ordinal;
    }
  };
  struct AssignmentRow final {
    MemberKey member;
    loom::fabric::FabricClockResetKind slotKind =
        loom::fabric::FabricClockResetKind::Clock;
    loom::fabric::FabricOrdinal slotOrdinal = 0;
  };

  llvm::Error assignOne(MemberKey member,
                        loom::fabric::FabricClockResetKind slotKind,
                        loom::fabric::FabricOrdinal slotOrdinal);

  loom::fabric::FabricOrdinal clockSlots_ = 0;
  loom::fabric::FabricOrdinal resetSlots_ = 0;
  std::vector<MemberKey> internalMembers_;
  std::vector<AssignmentRow> assignments_;
  std::vector<InstanceBindingRecord> instanceBindings_;
};

} // namespace fabric

#endif // LOOM_FABRIC_IR_MODULEDOMAIN_H
