#include "tapestry/compile.h"
#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace tapestry {

// ---------------------------------------------------------------------------
// compile() -- full implementation
// ---------------------------------------------------------------------------

CompileResult compile(TaskGraph &tdg, const CompileOptions &opts) {
  CompileResult result;
  std::ostringstream diag;

  if (opts.verbose)
    diag << "Starting compilation of graph '" << tdg.name() << "'\n";

  // 1. Create MLIRContext and register dialects.
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::tdg::TDGDialect>();

  // 2. Per-kernel compilation (C08 interface).
  //    kernel_compiler transforms each C function -> DFG (handshake.func).
  //    The implementation is provided by a later plan (C08). For now, we
  //    record provenance and skip DFG body generation.
  if (opts.verbose) {
    diag << "Kernel compilation: " << tdg.numKernels() << " kernels";
    if (!opts.sourcePaths.empty()) {
      diag << " from sources:";
      for (const auto &p : opts.sourcePaths)
        diag << " " << p;
    }
    diag << "\n";
  }

  // 3. Emit TDG MLIR from the TaskGraph.
  auto module = emitTDG(tdg, ctx);
  if (!module) {
    result.success = false;
    result.diagnostics = "Failed to emit TDG MLIR module";
    return result;
  }

  if (opts.verbose)
    diag << "TDG MLIR emitted successfully\n";

  // 4. Co-optimization loop (C13 interface).
  //    Alternates SW optimization (TDGOptimizer from C10) with HW DSE.
  //    Stub: single iteration, no optimization yet.
  unsigned iters = 0;
  for (unsigned round = 0; round < opts.maxCoOptRounds; ++round) {
    ++iters;
    // Co-optimization body will be implemented in C13.
    // For now, just break after one pass.
    break;
  }

  if (opts.verbose)
    diag << "Co-optimization: " << iters << " iteration(s)\n";

  // 5. Write artifacts.
  if (!opts.outputDir.empty()) {
    std::error_code ec;
    llvm::sys::fs::create_directories(opts.outputDir);

    // Write TDG MLIR.
    std::string mlirPath =
        (llvm::Twine(opts.outputDir) + "/" + tdg.name() + ".tdg.mlir").str();
    if (writeTDGToFile(*module, mlirPath)) {
      if (opts.verbose)
        diag << "Wrote TDG MLIR to " << mlirPath << "\n";
    } else {
      diag << "Warning: could not write TDG MLIR to " << mlirPath << "\n";
    }

    result.reportPath =
        (llvm::Twine(opts.outputDir) + "/" + tdg.name() + ".report.json").str();
  }

  result.success = true;
  result.iterations = iters;
  result.systemThroughput = 0.0; // placeholder until C13
  result.diagnostics = diag.str();
  return result;
}

// ---------------------------------------------------------------------------
// Convenience overload
// ---------------------------------------------------------------------------

CompileResult compile(TaskGraph &tdg, const std::string &archPath,
                      const std::string &outputDir) {
  CompileOptions opts;
  opts.systemArchPath = archPath;
  opts.outputDir = outputDir;
  return compile(tdg, opts);
}

} // namespace tapestry
