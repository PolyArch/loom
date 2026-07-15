#ifndef FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H
#define FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/ConfiguredFunction.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace fabric {

::mlir::LogicalResult
configuredFunctionFromSubgraph(::dataflow::SubgraphOp subgraph,
                               ConfiguredFunction &function,
                               std::string &error);

struct MaterializedSubgraph {
  ::mlir::func::FuncOp wrapper;
  ::dataflow::SubgraphOp subgraph;
};

::mlir::LogicalResult materializeConfiguredFunction(
    const ConfiguredFunction &function, ::mlir::ModuleOp module,
    ::llvm::StringRef symbolName, MaterializedSubgraph &materialized,
    std::string &error);

} // namespace fabric

#endif // FABRIC_TECH_CONFIGUREDFUNCTIONADAPTERS_H
