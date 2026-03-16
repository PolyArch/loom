#ifndef FCC_CONVERSION_PASSES_H
#define FCC_CONVERSION_PASSES_H

#include <memory>
#include <string>

namespace mlir {
class Pass;
} // namespace mlir

namespace fcc {

// Convert LLVM dialect to func/cf/arith/memref/math (CF stage).
// Handles: ptr->memref, GEP->index, LLVM ops->arith/cf ops, debug stripping.
// Functions with unsupported constructs (varargs, etc.) are left as-is.
std::unique_ptr<mlir::Pass> createConvertLLVMToCFPass();

// SCF post-processing: uplift scf.while to scf.for where possible.
// Detects canonical induction patterns (add/sub with constant step,
// comparison against bound) and converts to scf.for.
std::unique_ptr<mlir::Pass> createUpliftWhileToForPass();

// SCF post-processing: eliminate redundant subview bumps.
std::unique_ptr<mlir::Pass> createEliminateSubviewBumpsPass();

// DFG domain exploration: enumerate candidate regions, check quick ADG
// feasibility, and select the best region for DFG conversion.
// Replaces simple heuristic marking with structured exploration.
// ADG capacity is read from module attributes (fcc.adg_total_pes, etc.)
// set by the pipeline when --adg is provided.
std::unique_ptr<mlir::Pass> createMarkDFGDomainPass();

// Convert SCF-stage IR to handshake+dataflow DFG IR.
// Functions with fcc.dfg_candidate are lowered to handshake.func with
// dataflow.stream/gate/carry for loops, handshake.load/store/extmemory
// for memory operations. Non-candidate functions remain as-is.
std::unique_ptr<mlir::Pass> createConvertSCFToDFGPass();

// Generate host C source with MMIO driver calls for accelerated functions.
// Writes a host .c file and fcc_accel.h to the output directory.
// The host code replaces calls to DFG-converted functions with
// fcc_accel_* API invocations for gem5/RISC-V execution.
std::unique_ptr<mlir::Pass>
createHostCodeGenPass(const std::string &outputPath,
                      const std::string &originalSource);

} // namespace fcc

#endif
