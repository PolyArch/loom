#ifndef FABRIC_IR_ELABORATION_H
#define FABRIC_IR_ELABORATION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace fabric {
class ModuleOp;

// Canonicalize every fabric.instantiate nested under `root`. Named
// declarations remain declarations; every concrete use becomes fresh physical
// IR local to the root. `root` must be directly nested under builtin.module.
// The operation preserves the root operation identity and publishes a verified
// scratch body only after successful semantic preflight and elaboration.
::mlir::LogicalResult elaborateInstances(ModuleOp root);

std::unique_ptr<::mlir::Pass> createElaborateInstancesPass();
void registerFabricIRPasses();

} // namespace fabric

#endif // FABRIC_IR_ELABORATION_H
