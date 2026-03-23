#include "loom/SystemCompiler/KernelProfiler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include <algorithm>
#include <string>

namespace loom {

//===----------------------------------------------------------------------===//
// Helper: estimate memory footprint from operation types
//===----------------------------------------------------------------------===//

namespace {

/// Check if an operation name suggests memory access.
bool isMemoryOp(llvm::StringRef opName) {
  return opName.contains("load") || opName.contains("store") ||
         opName.contains("mem") || opName.contains("buffer") ||
         opName.contains("alloc");
}

/// Estimate bytes per memory operation.
uint64_t estimateMemOpBytes(llvm::StringRef opName) {
  // Default estimate: 4 bytes per memory access (32-bit).
  (void)opName;
  return 4;
}

/// Categorize an operation into a resource type for FU counting.
/// Returns the canonical operation type name.
std::string categorizeOp(llvm::StringRef opName) {
  // Strip dialect prefix to get the base op name.
  auto colonPos = opName.find('.');
  if (colonPos != llvm::StringRef::npos)
    return opName.substr(colonPos + 1).str();
  return opName.str();
}

} // namespace

//===----------------------------------------------------------------------===//
// KernelProfiler
//===----------------------------------------------------------------------===//

KernelProfile KernelProfiler::profile(mlir::ModuleOp kernelDFG,
                                      mlir::MLIRContext *ctx) {
  (void)ctx;
  KernelProfile result;

  // Extract kernel name from module name or first function.
  if (auto nameAttr = kernelDFG.getSymNameAttr())
    result.name = nameAttr.str();

  unsigned totalOps = 0;
  unsigned memOps = 0;
  uint64_t estimatedMemBytes = 0;

  // Walk all operations in the module to count resource usage.
  kernelDFG.walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();

    // Skip module/func-level structural operations.
    if (opName == "builtin.module" || opName == "func.func" ||
        opName == "func.return" || opName == "handshake.func" ||
        opName == "handshake.return" || opName == "cf.br" ||
        opName == "cf.cond_br")
      return;

    std::string category = categorizeOp(opName);
    result.requiredOps[category] += 1;
    totalOps += 1;

    if (isMemoryOp(opName)) {
      memOps += 1;
      estimatedMemBytes += estimateMemOpBytes(opName);
    }
  });

  // Estimate SPM: memory ops times an access granularity estimate.
  // Use a working set heuristic: unique memory ops * cache line size.
  result.estimatedSPMBytes = estimatedMemBytes * 64; // 64x working set factor

  // Estimate MinII: simple heuristic based on operation count.
  // In practice this would use recurrence analysis.
  result.estimatedMinII = std::max(1u, totalOps / 4);

  // Estimate compute cycles.
  result.estimatedComputeCycles =
      static_cast<double>(totalOps) * result.estimatedMinII;

  return result;
}

std::vector<KernelProfile>
KernelProfiler::profileAll(mlir::ModuleOp tdgModule, mlir::MLIRContext *ctx) {
  std::vector<KernelProfile> profiles;

  // Look for nested modules (each representing a kernel).
  tdgModule.walk([&](mlir::ModuleOp nestedModule) {
    // Skip the top-level module itself.
    if (nestedModule == tdgModule)
      return;

    KernelProfile profile = this->profile(nestedModule, ctx);
    if (!profile.name.empty() && profile.totalOpCount() > 0)
      profiles.push_back(std::move(profile));
  });

  // If no nested modules found, profile the top-level module directly.
  if (profiles.empty()) {
    KernelProfile profile = this->profile(tdgModule, ctx);
    if (!profile.name.empty() || profile.totalOpCount() > 0)
      profiles.push_back(std::move(profile));
  }

  return profiles;
}

} // namespace loom
