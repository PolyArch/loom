#ifndef FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H
#define FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H

#include "Fabric/IR/ConfiguredFunction.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace fabric {

::mlir::LogicalResult configuredFunctionFromFunc(::mlir::func::FuncOp source,
                                                 ConfiguredFunction &function,
                                                 std::string &error);

} // namespace fabric

#endif // FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H
