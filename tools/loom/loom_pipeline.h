//===-- loom_pipeline.h - Compilation pipeline helpers ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal helper functions for the Loom driver compilation pipeline:
// annotation extraction, output path derivation, clang invocation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_TOOLS_LOOM_PIPELINE_H
#define LOOM_TOOLS_LOOM_PIPELINE_H

#include "clang/Driver/Job.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"

#include <memory>
#include <string>

namespace loom {
namespace pipeline {

using AnnotationMap =
    llvm::StringMap<llvm::SmallVector<std::string, 4>>;

/// Collect global annotations from an LLVM module's
/// llvm.global.annotations metadata.
AnnotationMap CollectGlobalAnnotations(const llvm::Module &module);

/// Derive MLIR output path from the user-specified output path.
std::string DeriveMlirOutputPath(llvm::StringRef output_path);
std::string DeriveScfOutputPath(llvm::StringRef output_path);
std::string DeriveHandshakeOutputPath(llvm::StringRef output_path);

/// Apply collected symbol annotations as loom.annotations attributes.
void ApplySymbolAnnotations(mlir::ModuleOp module,
                            const AnnotationMap &annotations);

/// Scan module for __loom_loop_* calls and attach loom.loop.* annotations.
void ApplyLoopMarkerAnnotations(mlir::ModuleOp module);

/// Scan module for llvm.var.annotation / llvm.ptr.annotation intrinsics
/// and propagate annotation strings to target operations.
void ApplyIntrinsicAnnotations(mlir::ModuleOp module);

/// Check whether a clang driver command is a -cc1 invocation.
bool IsCC1Command(const clang::driver::Command &cmd);

/// Compile a single clang CompilerInvocation to an LLVM module.
std::unique_ptr<llvm::Module> CompileInvocation(
    const std::shared_ptr<clang::CompilerInvocation> &invocation,
    llvm::LLVMContext &context);

/// Ensure the parent directory of output_path exists.
bool EnsureOutputDirectory(llvm::StringRef output_path);

/// Strip unsupported LLVM attributes that may cause import failures.
void StripUnsupportedAttributes(llvm::Module &module);

} // namespace pipeline
} // namespace loom

#endif // LOOM_TOOLS_LOOM_PIPELINE_H
