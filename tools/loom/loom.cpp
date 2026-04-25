#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Tech/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  registry.insert<fabric::FabricDialect, dataflow::DataflowDialect>();
  fabric::registerFabricTechPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Loom dialects optimizer\n", registry));
}
