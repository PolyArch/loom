#ifndef FCC_CONVERSION_HOSTCODEGEN_H
#define FCC_CONVERSION_HOSTCODEGEN_H

#include "mlir/IR/BuiltinOps.h"
#include <memory>
#include <string>

namespace mlir {
class Pass;
} // namespace mlir

namespace fcc {

// Generate a host C source file for the accelerated program.
//
// This pass examines the MLIR module (after DFG conversion) and generates a
// host C file that replaces calls to accelerated functions with MMIO driver
// calls using the fcc_accel_* API.
//
// The generated host file can be cross-compiled to RISC-V for gem5 simulation.
//
// Options:
//   outputPath - path to write the generated .c file
//   originalSource - path to the original C source (for reference comments)
std::unique_ptr<mlir::Pass>
createHostCodeGenPass(const std::string &outputPath,
                      const std::string &originalSource = "");

} // namespace fcc

#endif
