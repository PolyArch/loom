#ifndef TAPESTRY_COMPILE_H
#define TAPESTRY_COMPILE_H

#include <string>
#include <vector>

namespace tapestry {

class TaskGraph;

// ============================================================================
// CompileOptions
// ============================================================================

struct CompileOptions {
  /// Path to JSON or MLIR system architecture description.
  /// JSON is auto-converted to MLIR internally.
  /// If empty, co-optimization starts from a spectral-clustering initial arch.
  std::string systemArchPath;

  /// Output directory for compilation artifacts.
  std::string outputDir = "output/";

  /// C/C++ source paths containing the kernel function bodies.
  /// Used by kernel_compiler (C08) to locate functions in LLVM IR.
  std::vector<std::string> sourcePaths;

  /// Maximum co-optimization rounds (SW <-> HW feedback loop).
  unsigned maxCoOptRounds = 5;

  /// Verbose diagnostic output.
  bool verbose = false;
};

// ============================================================================
// CompileResult
// ============================================================================

struct CompileResult {
  /// Whether compilation succeeded.
  bool success = false;

  /// Path to the generated report (JSON).
  std::string reportPath;

  /// Number of co-optimization iterations completed.
  unsigned iterations = 0;

  /// System throughput (ops/cycle) achieved by the final mapping.
  double systemThroughput = 0.0;

  /// Human-readable diagnostics / warnings.
  std::string diagnostics;
};

// ============================================================================
// Top-level compilation entry points
// ============================================================================

/// Compile a TaskGraph to a multi-core CGRA mapping.
///
/// Internally:
///   1. Constructs an MLIRContext and registers all required dialects.
///   2. Calls kernel_compiler for each kernel (C08 -- interface defined here,
///      implementation in a later plan).
///   3. Calls tdg_emitter to generate TDG MLIR.
///   4. Invokes the co-optimization loop (C13 -- stub here).
///   5. Writes artifacts to opts.outputDir.
///
/// \param tdg   The user-constructed TaskGraph.
/// \param opts  Compilation options.
/// \returns CompileResult summarizing the outcome.
CompileResult compile(TaskGraph &tdg, const CompileOptions &opts);

/// Convenience overload that constructs CompileOptions from common arguments.
CompileResult compile(TaskGraph &tdg, const std::string &archPath,
                      const std::string &outputDir);

} // namespace tapestry

#endif // TAPESTRY_COMPILE_H
