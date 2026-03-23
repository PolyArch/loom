//===-- TDGLowering.cpp - Kernel DFG lowering implementation ------*- C++ -*-===//
//
// Lowers kernel modules from SCF/CF form to DFG (handshake.func) form using
// the existing Loom conversion passes.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/TDGLowering.h"
#include "loom/Conversion/Passes.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace loom::tapestry;

/// Check if a module already contains handshake.func operations.
static bool hasHandshakeFuncs(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](circt::handshake::FuncOp) { found = true; });
  return found;
}

/// Check if a module contains func.func operations (SCF/CF form).
static bool hasFuncOps(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::func::FuncOp) { found = true; });
  return found;
}

/// Check if a module contains cf.br or cf.cond_br ops (CF form that needs
/// CF-to-SCF lifting before SCF-to-DFG lowering).
static bool hasCFOps(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "cf.br" ||
        op->getName().getStringRef() == "cf.cond_br")
      found = true;
  });
  return found;
}

mlir::LogicalResult loom::tapestry::lowerModuleToDFG(mlir::ModuleOp module) {
  // If already in handshake form, nothing to do.
  if (hasHandshakeFuncs(module))
    return mlir::success();

  // If it has CF ops, lift to SCF first.
  if (hasCFOps(module)) {
    mlir::PassManager cfToSCF(module.getContext());
    cfToSCF.addPass(mlir::createLiftControlFlowToSCFPass());
    cfToSCF.addPass(mlir::createCanonicalizerPass());
    cfToSCF.addPass(mlir::createCSEPass());
    cfToSCF.addPass(loom::createUpliftWhileToForPass());
    cfToSCF.addPass(mlir::createCanonicalizerPass());
    cfToSCF.addPass(mlir::createCSEPass());
    if (mlir::failed(cfToSCF.run(module))) {
      llvm::errs() << "TDGLowering: CF-to-SCF conversion failed\n";
      return mlir::failure();
    }
  }

  // If it has func.func ops, run SCF-to-DFG lowering.
  if (hasFuncOps(module)) {
    mlir::PassManager scfToDFG(module.getContext());
    scfToDFG.addPass(loom::createMarkDFGDomainPass());
    scfToDFG.addPass(loom::createConvertSCFToDFGPass());
    scfToDFG.addPass(mlir::createCanonicalizerPass());
    scfToDFG.addPass(mlir::createCSEPass());
    if (mlir::failed(scfToDFG.run(module))) {
      llvm::errs() << "TDGLowering: SCF-to-DFG conversion failed\n";
      return mlir::failure();
    }
  }

  // Verify we now have handshake.func ops.
  if (!hasHandshakeFuncs(module)) {
    llvm::errs() << "TDGLowering: module has no handshake.func after lowering "
                    "(no DFG candidates found)\n";
    return mlir::failure();
  }

  return mlir::success();
}

bool loom::tapestry::lowerKernelsToDFG(std::vector<KernelDesc> &kernels,
                                        mlir::MLIRContext &ctx) {
  // Ensure required dialects are loaded.
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();

  for (auto &kernel : kernels) {
    if (!kernel.dfgModule) {
      llvm::errs() << "TDGLowering: kernel '" << kernel.name
                   << "' has no module\n";
      return false;
    }

    // Skip kernels already in handshake form (precompiled DFGs).
    if (hasHandshakeFuncs(kernel.dfgModule)) {
      llvm::outs() << "TDGLowering: kernel '" << kernel.name
                   << "' already in DFG form, skipping\n";
      continue;
    }

    llvm::outs() << "TDGLowering: lowering kernel '" << kernel.name << "'...\n";
    if (mlir::failed(lowerModuleToDFG(kernel.dfgModule))) {
      llvm::errs() << "TDGLowering: failed to lower kernel '" << kernel.name
                   << "'\n";
      return false;
    }

    // Update resource estimates after lowering.
    unsigned opCount = 0;
    kernel.dfgModule.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::arith::ArithDialect>(op->getDialect()) ||
          op->getName().getStringRef().starts_with("handshake."))
        ++opCount;
    });
    kernel.requiredPEs = std::max(1u, (opCount + 2) / 3);
    kernel.requiredFUs = opCount;
  }

  return true;
}
