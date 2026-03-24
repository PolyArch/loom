//===-- KernelCompiler.h - C/C++ source to DFG compiler -----------*- C++ -*-===//
//
// Compiles C/C++ kernel functions into handshake.func DFG modules by
// reusing the Loom single-core frontend pipeline:
//   C -> Clang -> LLVM IR -> MLIR LLVM dialect -> CF -> SCF -> DFG
//
// Provides source-level caching (LLVM IR compiled once per source file)
// and per-function extraction/compilation.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_KERNELCOMPILER_H
#define LOOM_SYSTEMCOMPILER_KERNELCOMPILER_H

#include "loom/SystemCompiler/SystemTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace loom {
namespace tapestry {

/// Provenance information for a compiled kernel.
struct KernelProvenance {
  std::string functionName;
  std::string sourcePath;
  /// Pointer to the func::FuncOp in the cached SCF module (non-owning).
  mlir::Operation *funcPtr = nullptr;
};

/// Result of a single kernel compilation.
struct KernelCompileResult {
  bool success = false;

  /// The DFG module containing a handshake.func for this kernel.
  /// Only valid when success == true.
  mlir::OwningOpRef<mlir::ModuleOp> dfgModule;

  /// Name of the compiled function.
  std::string functionName;

  /// Diagnostic messages (errors or warnings).
  std::string diagnostics;

  /// Resource estimates from the DFG domain analysis.
  unsigned estimatedPECount = 0;
  unsigned estimatedMemPorts = 0;
};

/// Compiles C/C++ kernel functions into DFG (handshake.func) MLIR modules.
///
/// Usage:
///   KernelCompiler compiler(ctx);
///   compiler.loadSource("kernels.cpp");
///   auto result = compiler.compile("vecadd");
///   if (result.success) { /* use result.dfgModule */ }
///
/// Source files are compiled to LLVM IR once and cached. Multiple functions
/// from the same source share the cached SCF-stage MLIR module.
class KernelCompiler {
public:
  /// Construct a kernel compiler with the given MLIR context.
  /// Additional include paths can be provided for the Clang frontend.
  explicit KernelCompiler(mlir::MLIRContext &ctx,
                          std::vector<std::string> includePaths = {});

  ~KernelCompiler();

  /// Compile a C/C++ source file and cache its SCF-stage MLIR module.
  ///
  /// The source is compiled through:
  ///   Clang -> LLVM IR -> MLIR LLVM dialect -> CF -> SCF
  ///
  /// Returns true on success. On failure, diagnostics are printed to stderr.
  /// If the source was already loaded, this is a no-op and returns true.
  bool loadSource(const std::string &sourcePath);

  /// Compile a single function to a DFG (handshake.func) module.
  ///
  /// The function must exist in a previously loaded source file.
  /// The function is extracted, cloned into a standalone module, and
  /// run through MarkDFGDomain + SCFToDFG.
  ///
  /// Returns a KernelCompileResult with success == true on success.
  KernelCompileResult compile(const std::string &functionName);

  /// Quick accelerability check without full DFG compilation.
  ///
  /// Returns true if the function has candidate regions that could
  /// potentially be converted to a DFG. Does NOT guarantee that full
  /// compilation will succeed.
  bool isAccelerable(const std::string &functionName);

  /// List all function names available in loaded sources.
  std::vector<std::string> listFunctions() const;

  /// Convert a KernelCompileResult into a KernelDesc for use by the
  /// multi-core pipeline. Transfers ownership of the DFG module.
  static KernelDesc toKernelDesc(KernelCompileResult &result);

private:
  mlir::MLIRContext &ctx_;
  std::vector<std::string> includePaths_;

  /// Cached SCF-stage modules keyed by source file path.
  /// Each module has been lowered through LLVM -> CF -> SCF.
  std::unordered_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>
      scfModules_;

  /// Map function name -> source path for quick lookup.
  std::unordered_map<std::string, std::string> funcToSource_;

  /// The LLVM context used for Clang compilation (kept alive for the
  /// lifetime of the compiler since MLIR import may reference it).
  std::unique_ptr<llvm::LLVMContext> llvmCtx_;

  /// Register all MLIR dialects needed for the frontend pipeline.
  void registerDialects();

  /// Find a func::FuncOp by name in the cached modules.
  /// Returns nullptr if not found.
  mlir::Operation *findFunction(const std::string &functionName) const;
};

} // namespace tapestry
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_KERNELCOMPILER_H
