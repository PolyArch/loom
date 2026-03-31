#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/TypeAdapters.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <iostream>

using namespace loom;

static bool testNoSyntheticFURepertoireFromCountOnlyMetadata() {
  mlir::MLIRContext ctx;
  auto emptyModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  tapestry::SystemArchitecture tapArch;
  tapestry::CoreTypeDesc coreType;
  coreType.name = "count_only";
  coreType.numInstances = 1;
  coreType.adgModule = emptyModule;
  coreType.totalPEs = 4;
  coreType.totalFUs = 8;
  tapArch.coreTypes.push_back(coreType);

  SystemArchitecture arch = toL1Architecture(tapArch, &ctx);
  if (arch.coreTypes.size() != 1) {
    std::cerr << "FAIL: expected exactly one adapted core type\n";
    return false;
  }

  const CoreTypeSpec &adapted = arch.coreTypes.front();
  if (adapted.numFUs != 8) {
    std::cerr << "FAIL: expected numFUs metadata to be preserved\n";
    return false;
  }
  if (!adapted.fuTypeCounts.empty()) {
    std::cerr << "FAIL: count-only metadata must not fabricate FU repertoire\n";
    return false;
  }

  KernelProfile kernel;
  kernel.name = "needs_add";
  kernel.requiredOps["arith.addi"] = 1;
  if (isKernelCompatible(kernel, adapted)) {
    std::cerr << "FAIL: empty repertoire must not accept arithmetic kernels\n";
    return false;
  }

  std::cout << "PASS: testNoSyntheticFURepertoireFromCountOnlyMetadata\n";
  return true;
}

int main() {
  bool ok = true;
  ok &= testNoSyntheticFURepertoireFromCountOnlyMetadata();

  if (!ok)
    return 1;

  std::cout << "TypeAdapters tests: PASS\n";
  return 0;
}
