#ifndef FCC_SVGEN_SVGEN_H
#define FCC_SVGEN_SVGEN_H

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace fcc {
namespace svgen {

/// Options controlling SystemVerilog generation.
struct SVGenOptions {
  /// Path to the pre-written src/rtl/ directory in the source tree.
  std::string rtlSourceDir;
  /// Output directory where generated rtl/ collateral is placed.
  std::string outputDir;
  /// Optional floating-point IP profile name. When empty, Tier 3
  /// transcendental FP ops (sin, cos, exp, log2) are rejected.
  std::string fpIpProfile;
};

/// Generate self-contained synthesizable SystemVerilog collateral from a
/// fabric MLIR module. The output directory will contain all required SV
/// files and a filelist.f for compilation.
///
/// Returns true on success, false on error (diagnostics emitted to
/// llvm::errs()).
bool generateSV(mlir::ModuleOp adgModule, mlir::MLIRContext *ctx,
                const SVGenOptions &options);

} // namespace svgen
} // namespace fcc

#endif // FCC_SVGEN_SVGEN_H
