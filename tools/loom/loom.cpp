#include "Fabric/IR/FabricDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<fabric::FabricDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Fabric dialect optimizer\n", registry));
}
