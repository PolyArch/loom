//===-- VariantGenerator.h - Kernel variant generation driver ------*- C++ -*-===//
//
// Given a kernel's source and a list of VariantOptions, invokes the lowering
// pipeline once per variant with different compiler options and collects the
// resulting DFG modules into a variant-name-to-module map.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_VARIANTGENERATOR_H
#define LOOM_SYSTEMCOMPILER_VARIANTGENERATOR_H

#include "tapestry/task_graph.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

/// Result of generating a single DFG variant.
struct VariantResult {
  std::string variantName;
  ::tapestry::VariantOptions options;

  /// The lowered DFG module (null if lowering failed).
  mlir::ModuleOp dfgModule;

  /// Whether lowering succeeded for this variant.
  bool success = false;

  /// Diagnostic message (populated on failure).
  std::string diagnostic;
};

/// Configuration for the variant generation driver.
struct VariantGeneratorConfig {
  /// Maximum number of DFG nodes before a variant is considered too large
  /// and silently dropped.
  unsigned maxDFGNodes = 512;

  /// Whether to emit verbose diagnostics.
  bool verbose = false;
};

/// Generate DFG module variants for a single kernel.
///
/// For each VariantOptions entry, clones the kernel's base module,
/// configures the lowering pipeline with the variant's unroll factor and
/// domain rank, and runs lowerModuleToDFG(). Successfully lowered variants
/// are returned; failed variants are silently dropped (with a diagnostic
/// in verbose mode).
///
/// \param kernelName   Name of the kernel (for diagnostics).
/// \param baseModule   The kernel's SCF/CF-form module (will be cloned,
///                     not modified).
/// \param variants     List of variant entries to generate.
/// \param ctx          MLIR context (must have required dialects loaded).
/// \param config       Generation configuration.
/// \returns Vector of VariantResult, one per attempted variant. Only
///          entries with success==true have valid dfgModule fields.
std::vector<VariantResult>
generateVariants(const std::string &kernelName,
                 mlir::ModuleOp baseModule,
                 const std::vector<::tapestry::VariantEntry> &variants,
                 mlir::MLIRContext &ctx,
                 const VariantGeneratorConfig &config = {});

/// Build a variant-name-to-module map from generation results.
/// Only includes successful variants.
std::unordered_map<std::string, mlir::ModuleOp>
buildVariantModuleMap(const std::vector<VariantResult> &results);

/// Get the list of successfully generated variant names from results.
std::vector<std::string>
successfulVariantNames(const std::vector<VariantResult> &results);

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_VARIANTGENERATOR_H
