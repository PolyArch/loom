//===-- auto_analyze.h - Automatic TDG construction from C source --*- C++ -*-===//
//
// Tier 2 automatic analysis: given a C/C++ source file and entry function name,
// identifies parallelizable kernel regions, extracts inter-kernel data
// dependencies, and constructs a tapestry::TaskGraph with inferred contracts.
//
// No pragmas, no MLIR, no manual annotation required from the programmer.
//
//===----------------------------------------------------------------------===//

#ifndef TAPESTRY_AUTO_ANALYZE_H
#define TAPESTRY_AUTO_ANALYZE_H

#include "loom/SystemCompiler/Contract.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tapestry {

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

/// Options controlling the automatic analysis pass.
struct AutoAnalyzeOptions {
  /// Optional path to an ADG file for feasibility checking.
  std::string adgPath;

  /// Maximum number of kernels to extract from the entry function.
  unsigned maxKernels = 32;

  /// Enable verbose diagnostic output.
  bool verbose = false;
};

//===----------------------------------------------------------------------===//
// Kernel target designation
//===----------------------------------------------------------------------===//

/// Where a kernel should execute.
enum class KernelTarget {
  CGRA,   // Accelerate on CGRA fabric
  HOST,   // Execute on host processor
  AUTO,   // Let the compiler decide
};

//===----------------------------------------------------------------------===//
// Data dependency between two call sites
//===----------------------------------------------------------------------===//

/// Describes a data dependency between two kernel call sites in the entry
/// function, derived from shared pointer/buffer analysis.
struct DataDependency {
  /// Whether a dependency actually exists.
  bool exists = false;

  /// Inferred element data type name (e.g. "f32", "i32").
  std::string dataType;

  /// Whether the access pattern is sequential (implies FIFO ordering).
  bool isSequential = true;

  /// Estimated number of elements transferred, if known.
  std::optional<uint64_t> elementCount;

  /// Name of the shared argument (for diagnostics).
  std::string sharedArgName;
};

//===----------------------------------------------------------------------===//
// Call site binding (provenance for host scheduler generation)
//===----------------------------------------------------------------------===//

/// Records how a kernel call site in the entry function maps to the
/// constructed TaskGraph, preserving argument names and call order so
/// the host scheduler generator can emit correct driver code.
struct CallSiteBinding {
  /// The kernel function name.
  std::string kernelName;

  /// Argument references at the call site (SSA names or parameter names).
  std::vector<std::string> argNames;

  /// Sequential position of this call in the entry function body.
  unsigned callOrder = 0;

  /// Target designation for this kernel.
  KernelTarget target = KernelTarget::AUTO;
};

//===----------------------------------------------------------------------===//
// Inferred edge contract
//===----------------------------------------------------------------------===//

/// An edge in the automatically constructed task graph, representing a
/// data-flow dependency between a producer and consumer kernel.
struct InferredEdge {
  /// Index of the producer kernel in the callBindings vector.
  unsigned producerIndex = 0;

  /// Index of the consumer kernel in the callBindings vector.
  unsigned consumerIndex = 0;

  /// The underlying data dependency analysis result.
  DataDependency dependency;

  /// Inferred ordering (FIFO for sequential, UNORDERED otherwise).
  loom::Ordering ordering = loom::Ordering::FIFO;
};

//===----------------------------------------------------------------------===//
// AutoAnalyzeResult
//===----------------------------------------------------------------------===//

/// Complete result of automatic source analysis.
///
/// Contains everything needed to:
///   (a) feed into tapestry::compile() for full compilation, or
///   (b) let the programmer inspect/override before compilation (Tier 1.5).
struct AutoAnalyzeResult {
  /// Whether analysis succeeded.
  bool success = false;

  /// Diagnostic messages (warnings, errors, informational).
  std::string diagnostics;

  /// Provenance: which source file was analyzed.
  std::string sourcePath;

  /// Provenance: which entry function was analyzed.
  std::string entryFunc;

  /// Per-kernel call site bindings, in call order.
  std::vector<CallSiteBinding> callBindings;

  /// Inferred data-flow edges between kernels.
  std::vector<InferredEdge> edges;

  /// Number of kernels detected.
  unsigned numKernels() const {
    return static_cast<unsigned>(callBindings.size());
  }

  /// Number of edges detected.
  unsigned numEdges() const { return static_cast<unsigned>(edges.size()); }

  /// Print a human-readable summary to the given stream.
  void dump(llvm::raw_ostream &os) const;

  /// Print to llvm::outs().
  void dump() const;
};

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//

/// Automatically analyze a C/C++ source file to construct a task graph.
///
/// This is a compile-driver function (not a runtime operation). It:
///   1. Invokes Clang (in-process) to compile the source to LLVM bitcode
///   2. Converts bitcode to MLIR (LLVM -> CF -> SCF)
///   3. Locates the entry function by name
///   4. Identifies called sub-functions as candidate kernels
///   5. Checks each candidate for CGRA accelerability
///   6. Analyzes pointer/buffer sharing for data dependencies
///   7. Constructs the result with inferred edges and contracts
///
/// The returned AutoAnalyzeResult can be inspected, modified, and then
/// passed to the compilation pipeline.
AutoAnalyzeResult autoAnalyze(const std::string &sourcePath,
                              const std::string &entryFunc,
                              const AutoAnalyzeOptions &opts = {});

//===----------------------------------------------------------------------===//
// Bridge utilities (auto_analyze_bridge.cpp)
//===----------------------------------------------------------------------===//

class TaskGraph; // forward declaration

/// Convert an AutoAnalyzeResult into a TaskGraph for the MLIR pipeline.
TaskGraph buildTaskGraphFromAnalysis(const AutoAnalyzeResult &result);

/// Map a data type name string to its byte size (e.g. "f32" -> 4).
unsigned sizeOfType(const std::string &typeName);

} // namespace tapestry

#endif // TAPESTRY_AUTO_ANALYZE_H
