//===-- PrecompiledKernelLoader.cpp - Load pre-lowered DFG files ---*- C++ -*-===//
//
// Loads pre-compiled DFG MLIR files from a directory. Each file should
// contain a module with a handshake.func.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace loom::tapestry;

/// Ensure the context has all dialects needed to parse DFG MLIR.
static void ensureDFGDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

/// Count handshake.func operations in a module.
static unsigned countHandshakeFuncs(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](circt::handshake::FuncOp) { ++count; });
  return count;
}

/// Estimate the number of PE resources needed from a DFG module.
static unsigned estimateRequiredPEs(mlir::ModuleOp module) {
  unsigned opCount = 0;
  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::arith::ArithDialect>(op->getDialect()) ||
        op->getName().getStringRef().starts_with("handshake."))
      ++opCount;
  });
  // Rough estimate: one PE per 2-3 ops
  return std::max(1u, (opCount + 2) / 3);
}

std::vector<KernelDesc>
loom::tapestry::loadPrecompiledKernels(const std::string &directory,
                                       mlir::MLIRContext &ctx) {
  std::vector<KernelDesc> kernels;

  std::error_code ec;
  for (llvm::sys::fs::directory_iterator dirIt(directory, ec), dirEnd;
       dirIt != dirEnd && !ec; dirIt.increment(ec)) {
    llvm::StringRef path = dirIt->path();
    if (!path.ends_with(".mlir"))
      continue;

    KernelDesc kernel = loadSingleKernel(path.str(), ctx);
    if (!kernel.dfgModule) {
      llvm::errs() << "PrecompiledKernelLoader: failed to load " << path
                   << "\n";
      return {};
    }
    kernels.push_back(std::move(kernel));
  }

  if (ec) {
    llvm::errs() << "PrecompiledKernelLoader: error scanning directory '"
                 << directory << "': " << ec.message() << "\n";
    return {};
  }

  return kernels;
}

KernelDesc loom::tapestry::loadSingleKernel(const std::string &filePath,
                                             mlir::MLIRContext &ctx) {
  KernelDesc kernel;

  // Derive kernel name from filename stem
  llvm::StringRef stem = llvm::sys::path::stem(filePath);
  kernel.name = stem.str();

  ensureDFGDialects(ctx);

  llvm::SourceMgr srcMgr;
  auto buf = llvm::MemoryBuffer::getFile(filePath);
  if (!buf) {
    llvm::errs() << "PrecompiledKernelLoader: cannot open " << filePath << "\n";
    return kernel;
  }
  srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &ctx);
  if (!module) {
    llvm::errs() << "PrecompiledKernelLoader: parse failed for " << filePath
                 << "\n";
    return kernel;
  }

  unsigned funcCount = countHandshakeFuncs(*module);
  if (funcCount == 0) {
    llvm::errs() << "PrecompiledKernelLoader: no handshake.func in " << filePath
                 << "\n";
    return kernel;
  }

  kernel.dfgModule = *module;
  kernel.requiredPEs = estimateRequiredPEs(*module);
  kernel.requiredFUs = kernel.requiredPEs * 2;
  module.release();

  return kernel;
}

KernelDesc loom::tapestry::createSyntheticAddKernel(const std::string &name,
                                                     mlir::MLIRContext &ctx) {
  ensureDFGDialects(ctx);

  // Build a simple handshake.func: result = a + b
  // Uses the same format as working unit tests (no none control signals).
  std::string mlirText =
      "module {\n"
      "  handshake.func @" +
      name +
      "(%a: i32, %b: i32, %c: i32) -> (i32) "
      "attributes {argNames = [\"a\", \"b\", \"c\"], "
      "resNames = [\"result\"]} {\n"
      "    %sum = arith.addi %a, %b : i32\n"
      "    handshake.return %sum : i32\n"
      "  }\n"
      "}\n";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &ctx);
  if (!module) {
    llvm::errs() << "createSyntheticAddKernel: failed to parse synthetic DFG\n";
    return KernelDesc{name, mlir::ModuleOp(), 1, 1, 0};
  }

  KernelDesc kernel;
  kernel.name = name;
  kernel.dfgModule = *module;
  kernel.requiredPEs = 1;
  kernel.requiredFUs = 1;
  module.release();

  return kernel;
}

KernelDesc loom::tapestry::createSyntheticMacKernel(const std::string &name,
                                                     mlir::MLIRContext &ctx) {
  ensureDFGDialects(ctx);

  // Build: result = a * b + c
  // Uses 3 inputs matching the ADG boundary configuration.
  std::string mlirText =
      "module {\n"
      "  handshake.func @" +
      name +
      "(%a: i32, %b: i32, %c: i32) -> (i32) "
      "attributes {argNames = [\"a\", \"b\", \"c\"], "
      "resNames = [\"result\"]} {\n"
      "    %prod = arith.muli %a, %b : i32\n"
      "    %sum = arith.addi %prod, %c : i32\n"
      "    handshake.return %sum : i32\n"
      "  }\n"
      "}\n";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &ctx);
  if (!module) {
    llvm::errs() << "createSyntheticMacKernel: failed to parse synthetic DFG\n";
    return KernelDesc{name, mlir::ModuleOp(), 2, 2, 0};
  }

  KernelDesc kernel;
  kernel.name = name;
  kernel.dfgModule = *module;
  kernel.requiredPEs = 2;
  kernel.requiredFUs = 2;
  module.release();

  return kernel;
}
