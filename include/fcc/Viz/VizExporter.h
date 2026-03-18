#ifndef FCC_VIZ_VIZEXPORTER_H
#define FCC_VIZ_VIZEXPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace fcc {

/// Export a visualization-only HTML (no mapping, just ADG + DFG side-by-side).
mlir::LogicalResult exportVizOnly(const std::string &outputPath,
                                  mlir::ModuleOp adgModule,
                                  mlir::ModuleOp dfgModule,
                                  const std::string &adgSourcePath,
                                  mlir::MLIRContext *ctx);

/// Export visualization HTML with mapping data (from .map.json file).
/// The mapping JSON is embedded in the HTML for cross-highlighting.
mlir::LogicalResult exportVizWithMapping(const std::string &outputPath,
                                         mlir::ModuleOp adgModule,
                                         mlir::ModuleOp dfgModule,
                                         const std::string &mapJsonPath,
                                         const std::string &adgSourcePath,
                                         mlir::MLIRContext *ctx);

} // namespace fcc

#endif // FCC_VIZ_VIZEXPORTER_H
