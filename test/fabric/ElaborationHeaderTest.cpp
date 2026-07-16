#include "Fabric/IR/Elaboration.h"

int main() {
  std::unique_ptr<mlir::Pass> pass = fabric::createElaborateInstancesPass();
  return pass ? 0 : 1;
}
