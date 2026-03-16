#ifndef FCC_ADG_ADGVERIFIER_H
#define FCC_ADG_ADGVERIFIER_H

#include "mlir/IR/BuiltinOps.h"

namespace fcc {

/// Verify fabric.module compliance:
/// - No dangling ports: every instance result must be used,
///   every instance operand must come from a defined value.
/// Returns success if valid, failure with diagnostics if not.
mlir::LogicalResult verifyFabricModule(mlir::ModuleOp topModule);

} // namespace fcc

#endif // FCC_ADG_ADGVERIFIER_H
