// loom-raise-opt: an mlir-opt-style driver that registers all upstream
// passes plus the Loom raising and lowering passes
// (loom-llvm-cf-to-cf, loom-llvm-func-to-func,
// loom-llvm-arith-to-arith, loom-scf-while-to-for,
// loom-scf-for-to-forall, loom-lower-forall-to-thread,
// loom-lower-for-to-graph, loom-lower-scf-to-dfg) and the optional
// typed Dataflow rewrite pass (dataflow-rewrite). The Loom dataflow
// and fabric dialects are also registered so hand-written .mlir lit
// tests can exercise dataflow.thread / dataflow.graph op shapes.

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "Fabric/IR/FabricDialect.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/Lowering/Passes.h"
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
  registry.insert<::fabric::FabricDialect, ::dataflow::DataflowDialect,
                  ::loom::LoomDialect>();
  loom::raising::registerRaisingPasses();
  loom::lowering::registerLoweringPasses();
  dataflow::registerDataflowTransformsPasses();
  return ::mlir::asMainReturnCode(::mlir::MlirOptMain(
      argc, argv, "Loom raising-pass MLIR optimizer driver\n", registry));
}
