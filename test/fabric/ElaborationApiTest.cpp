#include "Fabric/IR/Elaboration.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

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
          operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>
        } {
      fabric.fu @FU (!fabric.bits<8>) -> (!fabric.bits<8>) {
      ^bb0(%fu_arg : !fabric.bits<8>):
        %sum = fabric.op [@arith.addi] (%fu_arg, %fu_arg)
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
  MLIRContext context;
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
