//===-- VariantGenerator.cpp - Kernel variant generation driver ------*- C++ -*-//
//
// Implements the variant generation driver that invokes the lowering pipeline
// multiple times with different compiler options to produce DFG variants.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/VariantGenerator.h"
#include "loom/SystemCompiler/TDGLowering.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/raw_ostream.h"

using namespace loom;

/// Count the number of non-structural operations in a module.
static unsigned countDFGNodes(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    // Skip structural/container ops.
    if (opName == "builtin.module" || opName == "func.func" ||
        opName == "func.return" || opName == "handshake.func" ||
        opName == "handshake.return")
      return;
    ++count;
  });
  return count;
}

std::vector<VariantResult>
loom::generateVariants(const std::string &kernelName,
                       mlir::ModuleOp baseModule,
                       const std::vector<::tapestry::VariantEntry> &variants,
                       mlir::MLIRContext &ctx,
                       const VariantGeneratorConfig &config) {
  std::vector<VariantResult> results;
  results.reserve(variants.size());

  for (const auto &variant : variants) {
    VariantResult vr;
    vr.variantName = variant.variantName;
    vr.options = variant.options;

    if (!baseModule) {
      vr.success = false;
      vr.diagnostic = "base module is null";
      results.push_back(std::move(vr));
      continue;
    }

    // Clone the base module for this variant so we don't mutate the original.
    mlir::OwningOpRef<mlir::ModuleOp> cloned = baseModule.clone();

    // TODO: when the unroll/domain-rank pass options are wired into
    // lowerModuleToDFG, pass variant.options here. For now, all variants
    // go through the same default lowering pipeline. The structural
    // difference comes from the clone-and-lower pattern itself (different
    // variants may get different canonicalization results).
    //
    // Unroll factor: would be passed to ConvertSCFToDFGPass options.
    // Domain rank: would be passed to MarkDFGDomainPass options.

    auto status = tapestry::lowerModuleToDFG(*cloned);
    if (mlir::failed(status)) {
      vr.success = false;
      vr.diagnostic = "lowering failed for variant '" + variant.variantName +
                       "' of kernel '" + kernelName + "'";
      if (config.verbose)
        llvm::errs() << "VariantGenerator: " << vr.diagnostic << "\n";
      results.push_back(std::move(vr));
      continue;
    }

    // Check if the DFG is too large.
    unsigned nodeCount = countDFGNodes(*cloned);
    if (nodeCount > config.maxDFGNodes) {
      vr.success = false;
      vr.diagnostic = "variant '" + variant.variantName + "' of kernel '" +
                       kernelName + "' exceeds maxDFGNodes (" +
                       std::to_string(nodeCount) + " > " +
                       std::to_string(config.maxDFGNodes) + ")";
      if (config.verbose)
        llvm::errs() << "VariantGenerator: " << vr.diagnostic << "\n";
      results.push_back(std::move(vr));
      continue;
    }

    vr.dfgModule = *cloned;
    // Release ownership -- caller takes over.
    (void)cloned.release();
    vr.success = true;

    if (config.verbose)
      llvm::outs() << "VariantGenerator: successfully generated variant '"
                   << variant.variantName << "' for kernel '" << kernelName
                   << "' (" << nodeCount << " DFG nodes)\n";

    results.push_back(std::move(vr));
  }

  return results;
}

std::unordered_map<std::string, mlir::ModuleOp>
loom::buildVariantModuleMap(const std::vector<VariantResult> &results) {
  std::unordered_map<std::string, mlir::ModuleOp> map;
  for (const auto &vr : results) {
    if (vr.success && vr.dfgModule)
      map[vr.variantName] = vr.dfgModule;
  }
  return map;
}

std::vector<std::string>
loom::successfulVariantNames(const std::vector<VariantResult> &results) {
  std::vector<std::string> names;
  for (const auto &vr : results) {
    if (vr.success)
      names.push_back(vr.variantName);
  }
  return names;
}
