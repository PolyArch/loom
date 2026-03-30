// LLVM-to-CF regression test for pointer/null compares.

#include "loom/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
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
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
}

static bool testPointerNullIcmpLowering() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<memref::MemRefDialect>();

  MLIRContext ctx(registry);
  registerDialects(ctx);

  const char *ir = R"mlir(
module {
  llvm.func @null_check(%arg0: !llvm.ptr) -> i1 {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr
    llvm.return %1 : i1
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(ir, &ctx);
  if (!module) {
    std::cerr << "FAIL: parser rejected the regression module\n";
    return false;
  }

  PassManager pm(&ctx);
  pm.addPass(loom::createConvertLLVMToCFPass());
  if (failed(pm.run(*module))) {
    std::cerr << "FAIL: ConvertLLVMToCFPass failed on pointer/null icmp\n";
    return false;
  }

  std::string printed;
  llvm::raw_string_ostream os(printed);
  module->print(os);
  os.flush();

  if (printed.find("arith.cmpi") != std::string::npos) {
    std::cerr << "FAIL: lowered module still contains arith.cmpi\n";
    return false;
  }

  std::cout << "PASS: testPointerNullIcmpLowering\n";
  return true;
}

int main() {
  if (!testPointerNullIcmpLowering())
    return 1;
  return 0;
}
