#ifndef FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H
#define FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H

#include "Fabric/IR/ConfiguredFunction.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace fabric {

::mlir::LogicalResult configuredFunctionFromFunc(::mlir::func::FuncOp source,
                                                 ConfiguredFunction &function,
                                                 std::string &error);

struct MaterializedConfiguredFunction {
  ::mlir::func::FuncOp wrapper;
};

::mlir::LogicalResult materializeConfiguredFunction(
    const ConfiguredFunction &function, ::mlir::ModuleOp module,
    ::llvm::StringRef symbolName, MaterializedConfiguredFunction &materialized,
    std::string &error);

} // namespace fabric

#endif // FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H
