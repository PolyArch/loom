// Regression test for SCF-to-DFG dominance-sensitive loop lowering.

#include "loom/Conversion/Passes.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>

using namespace mlir;

static void registerDialects(MLIRContext &ctx) {
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

static bool testNestedLoopDominance() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<loom::dataflow::DataflowDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  registerDialects(ctx);

  const char *ir = R"mlir(
module {
  func.func @nested_stencil(%in: memref<?xf32>, %out: memref<?xf32>,
                            %rows: index, %cols: index) attributes { loom.dfg_candidate } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %span = arith.muli %rows, %cols : index
    scf.for %row = %c0 to %rows step %c1 {
      scf.for %col = %c0 to %cols step %c1 {
        %base = arith.muli %row, %span : index
        %idx = arith.addi %base, %col : index
        %val = memref.load %in[%idx] : memref<?xf32>
        memref.store %val, %out[%idx] : memref<?xf32>
      }
    }
    func.return
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(ir, &ctx);
  if (!module) {
    std::cerr << "FAIL: parser rejected the regression module\n";
    return false;
  }

  PassManager pm(&ctx);
  pm.addPass(loom::createConvertSCFToDFGPass());
  if (failed(pm.run(*module))) {
    std::cerr << "FAIL: ConvertSCFToDFGPass failed on nested loop stencil\n";
    return false;
  }

  if (failed(verify(*module))) {
    std::cerr << "FAIL: lowered module does not verify\n";
    return false;
  }

  std::string printed;
  llvm::raw_string_ostream os(printed);
  module->print(os);
  os.flush();

  if (printed.find("handshake.func") == std::string::npos) {
    std::cerr << "FAIL: lowered module missing handshake.func\n";
    return false;
  }

  std::cout << "PASS: testNestedLoopDominance\n";
  return true;
}

int main() {
  if (!testNestedLoopDominance())
    return 1;
  return 0;
}
