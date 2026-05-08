// loom-raise-opt: an mlir-opt-style driver that registers all upstream
// passes plus the Loom raising passes (loom-llvm-cf-to-cf,
// loom-llvm-func-to-func, loom-llvm-arith-to-arith,
// loom-scf-while-to-for). Used by lit-style hand-written .mlir
// regression tests under test/raise/.

#include "Frontend/Raising/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  ::mlir::registerAllExtensions(registry);
  ::mlir::registerAllPasses();
  loom::raising::registerRaisingPasses();
  return ::mlir::asMainReturnCode(
      ::mlir::MlirOptMain(argc, argv,
                          "Loom raising-pass MLIR optimizer driver\n",
                          registry));
}
