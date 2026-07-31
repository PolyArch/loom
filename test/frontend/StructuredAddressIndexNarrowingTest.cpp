#include "StructuredAddressIndexNarrowing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "structuredAddressIndexNarrowing: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void normalizesAsymmetricPointerInduction() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n32:64-S128"
} {
  llvm.func @kernel(%base: !llvm.ptr) {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c4_i64 = arith.constant 4 : i64
    %result:3 = scf.while (%remaining = %c4_i32, %cursor = %base)
        : (i32, !llvm.ptr) -> (i32, !llvm.ptr, i32) {
      %value = llvm.load %cursor : !llvm.ptr -> i32
      %next_cursor = llvm.getelementptr inbounds %cursor[%c4_i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %next_remaining = arith.addi %remaining, %c-1_i32 : i32
      %more = arith.cmpi ne, %next_remaining, %c0_i32 : i32
      scf.condition(%more) %next_remaining, %next_cursor, %value
          : i32, !llvm.ptr, i32
    } do {
    ^bb0(%remaining: i32, %cursor: !llvm.ptr, %last: i32):
      scf.yield %remaining, %cursor : i32, !llvm.ptr
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the asymmetric pointer-induction fixture");
  auto function = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("kernel");
  if (!function)
    fail("fixture omitted kernel");

  auto normalized = loom::frontend::detail::materializeAddressIndexContract(
      *module, function.getOperation(), 64,
      [](mlir::Block *, mlir::Block *) { return llvm::Error::success(); });
  if (!normalized)
    fail(llvm::toString(normalized.takeError()));

  mlir::scf::WhileOp loop;
  function.walk([&](mlir::scf::WhileOp candidate) { loop = candidate; });
  if (!loop || loop.getInits().size() != 2 || loop.getNumResults() != 3)
    fail("pointer induction did not preserve the asymmetric loop contract");
  auto initOffset =
      llvm::dyn_cast<mlir::IntegerType>(loop.getInits()[1].getType());
  auto resultOffset =
      llvm::dyn_cast<mlir::IntegerType>(loop.getResult(1).getType());
  if (!initOffset || initOffset.getWidth() != 64 || !resultOffset ||
      resultOffset.getWidth() != 64)
    fail("pointer induction did not materialize the selected offset width");
}

} // namespace

int main() {
  normalizesAsymmetricPointerInduction();
  llvm::outs() << "structured address index narrowing anchor passed\n";
  return EXIT_SUCCESS;
}
