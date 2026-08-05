#include "Fabric/IR/Elaboration.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ModuleDomain.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static bool acceptsDomainBindings(
    fabric::ModuleDomainSlotCounts child, fabric::ModuleDomainSlotCounts parent,
    ArrayRef<fabric::ModuleInstanceDomainSlotBinding> bindings) {
  if (llvm::Error error = fabric::validateModuleInstanceDomainSlotBindings(
          child, parent, bindings)) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}

static bool verifyDomainBindingRelation() {
  using loom::fabric::FabricClockResetKind;
  using Binding = fabric::ModuleInstanceDomainSlotBinding;

  const fabric::ModuleDomainSlotCounts child{2, 1};
  const fabric::ModuleDomainSlotCounts parent{2, 1};
  const Binding manyToOne[] = {
      {FabricClockResetKind::Clock, 0, 1},
      {FabricClockResetKind::Clock, 1, 1},
      {FabricClockResetKind::Reset, 0, 0},
  };
  if (!acceptsDomainBindings(child, parent, manyToOne))
    return false;

  const Binding wrongOrder[] = {
      {FabricClockResetKind::Clock, 1, 1},
      {FabricClockResetKind::Clock, 0, 1},
      {FabricClockResetKind::Reset, 0, 0},
  };
  const Binding duplicateChild[] = {
      {FabricClockResetKind::Clock, 0, 0},
      {FabricClockResetKind::Clock, 0, 1},
      {FabricClockResetKind::Reset, 0, 0},
  };
  const Binding missingReset[] = {
      {FabricClockResetKind::Clock, 0, 0},
      {FabricClockResetKind::Clock, 1, 1},
  };
  const Binding childOutOfRange[] = {
      {FabricClockResetKind::Clock, 0, 0},
      {FabricClockResetKind::Clock, 2, 1},
      {FabricClockResetKind::Reset, 0, 0},
  };
  const Binding parentOutOfRange[] = {
      {FabricClockResetKind::Clock, 0, 0},
      {FabricClockResetKind::Clock, 1, 2},
      {FabricClockResetKind::Reset, 0, 0},
  };
  const Binding unknownKind[] = {
      {static_cast<FabricClockResetKind>(2), 0, 0},
  };

  MLIRContext context(MLIRContext::Threading::DISABLED);
  DenseI64ArrayAttr encoded =
      fabric::encodeModuleInstanceDomainSlotBindings(&context, manyToOne);
  auto decoded = fabric::decodeModuleInstanceDomainSlotBindings(encoded);
  if (!decoded || ArrayRef<Binding>(*decoded) != ArrayRef<Binding>(manyToOne))
    return false;

  auto rejectsEncoding = [&](ArrayRef<int64_t> fields) {
    auto decoded = fabric::decodeModuleInstanceDomainSlotBindings(
        DenseI64ArrayAttr::get(&context, fields));
    if (decoded)
      return false;
    llvm::consumeError(decoded.takeError());
    return true;
  };
  if (!rejectsEncoding({0, 0}) || !rejectsEncoding({2, 0, 0}) ||
      !rejectsEncoding({0, -1, 0}) || !rejectsEncoding({0, 4294967296LL, 0}))
    return false;

  return !acceptsDomainBindings(child, parent, wrongOrder) &&
         !acceptsDomainBindings(child, parent, duplicateChild) &&
         !acceptsDomainBindings(child, parent, missingReset) &&
         !acceptsDomainBindings(child, parent, childOutOfRange) &&
         !acceptsDomainBindings(child, parent, parentOutOfRange) &&
         !acceptsDomainBindings({1, 0}, {1, 0}, unknownKind) &&
         acceptsDomainBindings({0, 0}, {0, 0}, {});
}

static bool acceptsModuleDomainRelation(
    loom::fabric::FabricModuleTemplateRef module,
    ArrayRef<loom::fabric::FabricModuleDomainSlotRef> slots,
    ArrayRef<loom::fabric::FabricModuleDomainMemberRef> members,
    ArrayRef<loom::fabric::ModuleDomainAssignment> assignments,
    fabric::ModuleDomainSlotCounts expectedCounts) {
  auto counts =
      fabric::validateModuleDomainRelation(module, slots, members, assignments);
  if (!counts) {
    llvm::consumeError(counts.takeError());
    return false;
  }
  return counts->clocks == expectedCounts.clocks &&
         counts->resets == expectedCounts.resets;
}

static bool rejectsModuleDomainRelation(
    loom::fabric::FabricModuleTemplateRef module,
    ArrayRef<loom::fabric::FabricModuleDomainSlotRef> slots,
    ArrayRef<loom::fabric::FabricModuleDomainMemberRef> members,
    ArrayRef<loom::fabric::ModuleDomainAssignment> assignments) {
  auto counts =
      fabric::validateModuleDomainRelation(module, slots, members, assignments);
  if (counts)
    return false;
  llvm::consumeError(counts.takeError());
  return true;
}

static bool verifyModuleDomainRelation() {
  using loom::fabric::FabricClockResetKind;
  using loom::fabric::FabricModuleBoundaryEndpointRef;
  using loom::fabric::FabricModuleDomainMemberRef;
  using loom::fabric::FabricModuleDomainSlotRef;
  using loom::fabric::FabricModulePhysicalOwnerRef;
  using loom::fabric::FabricModuleTemplateRef;
  using loom::fabric::FabricPeOccurrenceRef;
  using loom::fabric::FabricPortDirection;
  using loom::fabric::ModuleDomainAssignment;

  const FabricModuleTemplateRef module(10);
  const FabricModuleDomainSlotRef clock0{module, FabricClockResetKind::Clock,
                                         0};
  const FabricModuleDomainSlotRef clock1{module, FabricClockResetKind::Clock,
                                         1};
  const FabricModuleDomainSlotRef reset0{module, FabricClockResetKind::Reset,
                                         0};
  const FabricModuleDomainSlotRef slots[] = {clock0, clock1, reset0};

  const FabricModuleDomainMemberRef boundary = FabricModuleDomainMemberRef::of(
      FabricModuleBoundaryEndpointRef{module, FabricPortDirection::Input, 0});
  auto owner = FabricModulePhysicalOwnerRef::create(FabricPeOccurrenceRef(11));
  if (!owner)
    return false;
  const FabricModuleDomainMemberRef internal =
      FabricModuleDomainMemberRef::of(*owner);
  const FabricModuleDomainMemberRef members[] = {boundary, internal};
  const ModuleDomainAssignment assignments[] = {
      {boundary, clock1},
      {boundary, reset0},
      {internal, clock0},
      {internal, reset0},
  };
  if (!acceptsModuleDomainRelation(module, slots, members, assignments, {2, 1}))
    return false;

  const FabricModuleDomainSlotRef sparseSlots[] = {
      clock0,
      FabricModuleDomainSlotRef{module, FabricClockResetKind::Clock, 2},
      reset0,
  };
  const FabricModuleDomainSlotRef foreignSlots[] = {
      FabricModuleDomainSlotRef{FabricModuleTemplateRef(99),
                                FabricClockResetKind::Clock, 0},
      reset0,
  };
  const FabricModuleDomainSlotRef duplicateSlots[] = {clock0, clock0, reset0};
  const FabricModuleDomainSlotRef crossKindUnsortedSlots[] = {clock0, reset0,
                                                              clock1};
  const FabricModuleDomainSlotRef unknownKindSlots[] = {
      FabricModuleDomainSlotRef{module, static_cast<FabricClockResetKind>(2),
                                0},
  };
  const FabricModuleDomainMemberRef foreignBoundary =
      FabricModuleDomainMemberRef::of(FabricModuleBoundaryEndpointRef{
          FabricModuleTemplateRef(99), FabricPortDirection::Input, 0});
  const FabricModuleDomainMemberRef unsortedMembers[] = {internal, boundary};
  const FabricModuleDomainMemberRef duplicateMembers[] = {boundary, boundary};
  const ModuleDomainAssignment missingReset[] = {
      {boundary, clock1},
      {internal, clock0},
      {internal, reset0},
  };
  const ModuleDomainAssignment duplicateClock[] = {
      {boundary, clock0},
      {boundary, clock1},
      {internal, clock0},
      {internal, reset0},
  };
  const ModuleDomainAssignment wrongModule[] = {
      {boundary, clock1},
      {boundary, FabricModuleDomainSlotRef{FabricModuleTemplateRef(99),
                                           FabricClockResetKind::Reset, 0}},
      {internal, clock0},
      {internal, reset0},
  };
  const ModuleDomainAssignment outOfRangeSlot[] = {
      {boundary,
       FabricModuleDomainSlotRef{module, FabricClockResetKind::Clock, 2}},
      {boundary, reset0},
      {internal, clock0},
      {internal, reset0},
  };
  const ModuleDomainAssignment unknownMember[] = {
      {boundary, clock1},
      {boundary, reset0},
      {foreignBoundary, clock0},
      {foreignBoundary, reset0},
  };
  const ModuleDomainAssignment unsortedAssignments[] = {
      {boundary, reset0},
      {boundary, clock1},
      {internal, clock0},
      {internal, reset0},
  };

  return rejectsModuleDomainRelation(module, sparseSlots, members,
                                     assignments) &&
         rejectsModuleDomainRelation(module, foreignSlots, members,
                                     assignments) &&
         rejectsModuleDomainRelation(module, duplicateSlots, {}, {}) &&
         rejectsModuleDomainRelation(module, crossKindUnsortedSlots, {}, {}) &&
         rejectsModuleDomainRelation(module, unknownKindSlots, {}, {}) &&
         rejectsModuleDomainRelation(module, slots, unsortedMembers,
                                     assignments) &&
         rejectsModuleDomainRelation(module, slots, duplicateMembers,
                                     assignments) &&
         rejectsModuleDomainRelation(module, slots, {foreignBoundary}, {}) &&
         rejectsModuleDomainRelation(module, slots, members, missingReset) &&
         rejectsModuleDomainRelation(module, slots, members, duplicateClock) &&
         rejectsModuleDomainRelation(module, slots, members, wrongModule) &&
         rejectsModuleDomainRelation(module, slots, members, outOfRangeSlot) &&
         rejectsModuleDomainRelation(module, slots, members, unknownMember) &&
         rejectsModuleDomainRelation(module, slots, members,
                                     unsortedAssignments) &&
         acceptsModuleDomainRelation(module, slots, {}, {}, {2, 1});
}

static constexpr StringLiteral input = R"mlir(
module {
  fabric.module @producer(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
    fabric.switch @SOURCE [spatial]
        (!fabric.bits<8>) -> (!fabric.bits<8>)
        [{connectivity_table = ["1"]}]
    %result = fabric.instantiate @SOURCE(
        %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
    fabric.yield %result : !fabric.bits<8>
  }

  fabric.module @consumer(%arg : !fabric.bits<8>) -> (!fabric.bits<16>) {
    fabric.switch @WIDE [spatial]
        (!fabric.bits<16>) -> (!fabric.bits<16>)
        [{connectivity_table = ["1"]}]
    %result = fabric.instantiate @WIDE(
        %arg : !fabric.bits<8> to !fabric.bits<16>)
        -> (!fabric.bits<16>)
    fabric.yield %result : !fabric.bits<16>
  }

  fabric.module @callee(%arg : !fabric.bits<8>) -> (!fabric.bits<16>) {
    %consumed = fabric.instantiate @consumer(
        %produced : !fabric.bits<8>) -> (!fabric.bits<16>)
    %produced = fabric.instantiate @producer(
        %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
    fabric.yield %consumed : !fabric.bits<16>
  }

  fabric.module @selected(%arg : !fabric.bits<8>) -> (!fabric.bits<16>) {
    %result = fabric.instantiate @callee(
        %arg : !fabric.bits<8>) -> (!fabric.bits<16>)
    fabric.yield %result : !fabric.bits<16>
  }
}
)mlir";

static constexpr StringLiteral failureInput = R"mlir(
module {
  fabric.module @sibling(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
    fabric.switch @IDENTITY [spatial]
        (!fabric.bits<8>) -> (!fabric.bits<8>)
        [{connectivity_table = ["1"]}]
    %result = fabric.instantiate @IDENTITY(
        %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
    fabric.yield %result : !fabric.bits<8>
  }

  fabric.module @selected(%arg : !fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>) {
    %pe = fabric.pe [temporal] (
        %pe_arg = %arg : !fabric.bits_tag<8, 2> to !fabric.bits<4>)
        -> !fabric.bits_tag<8, 2>
        attributes {
          tag_width = 2 : i32,
          num_instruction = 1 : i32,
          fu_config_mode = "per_fu_config",
          operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
          operand_buffer_size = 2 : i32
        } {
      fabric.fu @FU (!fabric.bits<8>) -> (!fabric.bits<8>) {
      ^bb0(%fu_arg : !fabric.bits<8>):
        %sum = fabric.op [@arith.addi] (%fu_arg, %fu_arg)
            {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32]}}
            : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
        fabric.yield %sum : !fabric.bits<8>
      }
      %unused = fabric.instantiate @FU(
          %pe_arg : !fabric.bits<4> to !fabric.bits<8>)
          -> (!fabric.bits<8>)
    }
    fabric.yield %pe : !fabric.bits_tag<8, 2>
  }
}
)mlir";

static unsigned countInstances(Operation *root) {
  unsigned count = 0;
  root->walk([&](fabric::InstantiateOp) { ++count; });
  return count;
}

static std::string print(Operation *op) {
  std::string text;
  llvm::raw_string_ostream(text) << *op;
  return text;
}

static bool verifyFailureAtomicity(MLIRContext &context) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(failureInput, &context);
  if (!module || failed(verify(*module)))
    return false;

  fabric::ModuleOp selected =
      module->lookupSymbol<fabric::ModuleOp>("selected");
  fabric::ModuleOp sibling = module->lookupSymbol<fabric::ModuleOp>("sibling");
  Operation *moduleIdentity = module->getOperation();
  Operation *selectedIdentity = selected.getOperation();
  Operation *siblingIdentity = sibling.getOperation();
  std::string moduleBefore = print(module->getOperation());
  std::string selectedBefore = print(selected);
  std::string siblingBefore = print(sibling);

  if (succeeded(fabric::elaborateInstances(selected)))
    return false;
  return module->getOperation() == moduleIdentity &&
         selected.getOperation() == selectedIdentity &&
         sibling.getOperation() == siblingIdentity &&
         print(module->getOperation()) == moduleBefore &&
         print(selected) == selectedBefore && print(sibling) == siblingBefore;
}

int main() {
  if (!verifyDomainBindingRelation() || !verifyModuleDomainRelation())
    return 1;

  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.getOrLoadDialect<fabric::FabricDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(input, &context);
  OwningOpRef<ModuleOp> passModule =
      parseSourceString<ModuleOp>(input, &context);
  if (!module || !passModule || failed(verify(*module)) ||
      failed(verify(*passModule)))
    return 1;

  fabric::ModuleOp selected =
      module->lookupSymbol<fabric::ModuleOp>("selected");
  fabric::ModuleOp callee = module->lookupSymbol<fabric::ModuleOp>("callee");
  Operation *selectedIdentity = selected.getOperation();
  Operation *calleeIdentity = callee.getOperation();

  if (failed(fabric::elaborateInstances(selected)))
    return 1;
  if (selected.getOperation() != selectedIdentity ||
      callee.getOperation() != calleeIdentity)
    return 1;
  if (countInstances(selected) != 0 || countInstances(callee) != 2)
    return 1;
  if (failed(verify(*module)))
    return 1;

  PassManager manager(&context);
  manager.addPass(fabric::createElaborateInstancesPass());
  if (failed(manager.run(*passModule)))
    return 1;
  fabric::ModuleOp passSelected =
      passModule->lookupSymbol<fabric::ModuleOp>("selected");

  std::string apiText;
  std::string passText;
  llvm::raw_string_ostream(apiText) << selected;
  llvm::raw_string_ostream(passText) << passSelected;
  if (apiText != passText)
    return 1;
  if (!verifyFailureAtomicity(context))
    return 1;

  llvm::outs() << "fabric elaboration API ok\n";
  return 0;
}
