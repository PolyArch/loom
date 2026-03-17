#ifndef FCC_CONVERSION_HOSTCODEGEN_H
#define FCC_CONVERSION_HOSTCODEGEN_H

#include "mlir/IR/BuiltinOps.h"
#include <memory>
#include <string>

namespace mlir {
class Pass;
} // namespace mlir

namespace fcc {

// Generate host-facing sources for the accelerated program.
//
// This pass examines the MLIR module (after DFG conversion) and generates:
// - a host C file that replaces accelerated calls with fcc_accel_* MMIO calls
// - fcc_accel.h with the runtime-facing API declarations
// - fcc_accel.c with the thin MMIO runtime implementation
//
// The generated sources can be cross-compiled for RISC-V or built for local
// host-side smoke checking.
//
// Options:
//   outputPath - path to write the generated .c file
//   originalSource - path to the original C source (for reference comments)
std::unique_ptr<mlir::Pass>
createHostCodeGenPass(const std::string &outputPath,
                      const std::string &originalSource = "");

} // namespace fcc

#endif
