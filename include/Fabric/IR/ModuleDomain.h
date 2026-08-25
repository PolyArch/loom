#ifndef LOOM_FABRIC_IR_MODULEDOMAIN_H
#define LOOM_FABRIC_IR_MODULEDOMAIN_H

#include "Fabric/Identity/FabricRefs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
class Operation;
class IRMapping;
} // namespace mlir

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

/// The one canonical persistent relation materialized after Fabric entity
/// labeling. MLIR attributes only frame the canonical typed reference bytes.
struct CanonicalModuleDomainRelation final {
  std::vector<loom::fabric::FabricModuleDomainSlotRef> slots;
  std::vector<loom::fabric::ModuleDomainAssignment> assignments;
};

mlir::ArrayAttr encodeModuleDomainSlots(
    mlir::MLIRContext *context,
    llvm::ArrayRef<loom::fabric::FabricModuleDomainSlotRef> slots);
llvm::Expected<std::vector<loom::fabric::FabricModuleDomainSlotRef>>
decodeModuleDomainSlots(mlir::ArrayAttr encoded);
mlir::ArrayAttr encodeModuleDomainAssignments(
    mlir::MLIRContext *context,
    llvm::ArrayRef<loom::fabric::ModuleDomainAssignment> assignments);
llvm::Expected<std::vector<loom::fabric::ModuleDomainAssignment>>
decodeModuleDomainAssignments(mlir::ArrayAttr encoded);

/// One phase-local typed authoring relation for a Module's symbolic Clock and
/// Reset slots and their member assignments. It is the sole authoring
/// representation before canonical labeling. The relation is consumed exactly
/// once to materialize the persistent `domain_slots`/`domain_assignments`
/// carrier, then destroyed. Boundary members are identified by direction and
/// endpoint ordinal; internal owners by their exact live draft operation and
/// typed subrole. Operation pointers are Builder-lifetime-only: they exist only
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

  /// Records one Module instance edge after validating the canonical rows
  /// owned by its `fabric.instantiate` operation against both slot
  /// inventories. The keyed operation is remapped across clone and instance
  /// remapping before the single canonical consumption materializes it.
  llvm::Error noteInstanceBindings(mlir::Operation *instance,
                                   const ModuleDomainAuthoringRelation &child);

  struct InstanceBindingRecord final {
    mlir::Operation *instance = nullptr;
    std::shared_ptr<const ModuleDomainAuthoringRelation> child;
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
  llvm::Error ensureDefaultAssignments(loom::fabric::FabricOrdinal inputCount,
                                       loom::fabric::FabricOrdinal outputCount);

  /// Registers one internal owner member created by a construction call.
  llvm::Error noteInternalMember(mlir::Operation *owner,
                                 InternalMemberRole role,
                                 loom::fabric::FabricOrdinal subOrdinal);

  /// Registers one physical owner that is materialized only after a finite
  /// non-Module instance path is elaborated. `instancePath` is ordered from
  /// the root-local use to the innermost use; `targetOwner` belongs to the
  /// final named target. The path is transient Builder identity and is
  /// consumed by the existing Fabric elaborator.
  llvm::Error
  noteInstantiatedMember(llvm::ArrayRef<mlir::Operation *> instancePath,
                         mlir::Operation *targetOwner, InternalMemberRole role,
                         loom::fabric::FabricOrdinal subOrdinal);

  llvm::Error assignBoundary(loom::fabric::FabricPortDirection direction,
                             loom::fabric::FabricOrdinal endpointOrdinal,
                             loom::fabric::FabricClockResetKind slotKind,
                             loom::fabric::FabricOrdinal slotOrdinal);
  llvm::Error assignInternal(mlir::Operation *owner, InternalMemberRole role,
                             loom::fabric::FabricOrdinal subOrdinal,
                             loom::fabric::FabricClockResetKind slotKind,
                             loom::fabric::FabricOrdinal slotOrdinal);
  llvm::Error assignInstantiated(llvm::ArrayRef<mlir::Operation *> instancePath,
                                 mlir::Operation *targetOwner,
                                 InternalMemberRole role,
                                 loom::fabric::FabricOrdinal subOrdinal,
                                 loom::fabric::FabricClockResetKind slotKind,
                                 loom::fabric::FabricOrdinal slotOrdinal);

  /// No slots declared, no members registered, no assignments made, and no
  /// instance bindings recorded.
  bool empty() const;

  /// Validates exact totality over the boundary members implied by the
  /// signature and every registered internal member: exactly one Clock and
  /// one Reset assignment per member, dense slots, and no extra assignment.
  llvm::Error validateTotality(loom::fabric::FabricOrdinal inputCount,
                               loom::fabric::FabricOrdinal outputCount) const;

  /// Remaps every Builder-lifetime operation identity through one exact IR
  /// clone. Missing mappings fail closed.
  llvm::Expected<ModuleDomainAuthoringRelation>
  remap(const mlir::IRMapping &mapping) const;

  /// Replicates every member and assignment selected by a partial clone map.
  /// Existing members remain unchanged; this is used when an ordinary Builder
  /// draft clones a finalized physical occurrence as another occurrence.
  llvm::Error replicateMappedOperations(const mlir::IRMapping &mapping);

  /// Removes all member and assignment rows owned by one exact operation set.
  /// Callers pass the complete erased subtree, so no stale draft pointer can
  /// survive into finalization.
  llvm::Error eraseOperations(llvm::ArrayRef<mlir::Operation *> operations);

  /// Changes the dense submember inventory of one owner. Added members inherit
  /// the exact Clock and Reset assignments of `prototypeOrdinal`; removed
  /// members and their assignments are discarded.
  llvm::Error
  resizeInternalMembers(mlir::Operation *owner, InternalMemberRole role,
                        loom::fabric::FabricOrdinal oldCount,
                        loom::fabric::FabricOrdinal newCount,
                        loom::fabric::FabricOrdinal prototypeOrdinal = 0);

  /// Removes trailing boundary members after a Builder changes one root
  /// signature. Boundary growth requires explicit domain authoring and is not
  /// inferred by this operation.
  llvm::Error
  truncateBoundaryMembers(loom::fabric::FabricPortDirection direction,
                          loom::fabric::FabricOrdinal oldCount,
                          loom::fabric::FabricOrdinal newCount);

  llvm::Error composeInstance(mlir::Operation *instance,
                              const mlir::IRMapping &childCloneMapping);

  /// Resolves one selected use in every matching non-Module instance path.
  /// The remaining prefix stays deferred and the nested suffix is remapped
  /// through the target clone; a fully resolved path becomes an owner row.
  llvm::Error materializePhysicalInstance(
      mlir::Operation *instance, mlir::Operation *target,
      mlir::Operation *occurrence, const mlir::IRMapping &targetCloneMapping);

  using BoundaryAssignmentVisitor = llvm::function_ref<llvm::Error(
      loom::fabric::FabricPortDirection, loom::fabric::FabricOrdinal,
      loom::fabric::FabricClockResetKind, loom::fabric::FabricOrdinal)>;
  using InternalAssignmentVisitor = llvm::function_ref<llvm::Error(
      mlir::Operation *, InternalMemberRole, loom::fabric::FabricOrdinal,
      loom::fabric::FabricClockResetKind, loom::fabric::FabricOrdinal)>;

  /// Visits the validated authoring rows without exposing their storage.
  llvm::Error visitAssignments(BoundaryAssignmentVisitor boundary,
                               InternalAssignmentVisitor internal) const;

private:
  struct MemberKey final {
    bool internal = false;
    loom::fabric::FabricPortDirection direction =
        loom::fabric::FabricPortDirection::Input;
    mlir::Operation *owner = nullptr;
    std::vector<mlir::Operation *> instancePath;
    InternalMemberRole role = InternalMemberRole::Occurrence;
    loom::fabric::FabricOrdinal ordinal = 0;

    friend bool operator==(const MemberKey &lhs, const MemberKey &rhs) {
      return lhs.internal == rhs.internal && lhs.direction == rhs.direction &&
             lhs.owner == rhs.owner && lhs.instancePath == rhs.instancePath &&
             lhs.role == rhs.role && lhs.ordinal == rhs.ordinal;
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
  llvm::Error noteMember(MemberKey member);

  loom::fabric::FabricOrdinal clockSlots_ = 0;
  loom::fabric::FabricOrdinal resetSlots_ = 0;
  bool defaultAssignments_ = false;
  std::vector<MemberKey> internalMembers_;
  std::vector<AssignmentRow> assignments_;
  std::vector<InstanceBindingRecord> instanceBindings_;
};

} // namespace fabric

#endif // LOOM_FABRIC_IR_MODULEDOMAIN_H
