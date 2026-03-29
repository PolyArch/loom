//===-- SystemADGBuilder.cpp - System-level ADG Builder -----------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the SystemADGBuilder which composes per-core fabric.module
// definitions into a system-level fabric.module with NoC connectivity and
// shared memory hierarchy. Uses the MLIR builder API via SystemADGMLIRBuilder
// to generate proper typed ops.
//
// Core types are now stored as mlir::ModuleOp references instead of strings,
// and build() returns an mlir::ModuleOp directly, eliminating the string
// intermediary.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/SystemADGBuilder.h"
#include "loom/ADG/SystemADGMLIRBuilder.h"
#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Internal data structures
//===----------------------------------------------------------------------===//

struct CoreTypeDef {
  std::string typeName;
  mlir::ModuleOp coreModule;
  unsigned id;
};

struct SystemADGBuilder::Impl {
  mlir::MLIRContext *ctx;
  std::unique_ptr<mlir::MLIRContext> ownedCtx; // non-null if builder owns ctx
  std::string systemName;
  std::vector<CoreTypeDef> coreTypes;
  std::vector<SystemCoreInstance> coreInstances;
  NoCSpec nocSpec;
  SharedMemorySpec sharedMemSpec;
  mlir::ModuleOp builtModule; // null until build() is called

  /// Initialize context with all required dialects.
  static void initContext(mlir::MLIRContext *ctx) {
    mlir::DialectRegistry registry;
    registry.insert<loom::fabric::FabricDialect>();
    registry.insert<loom::dataflow::DataflowDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<circt::handshake::HandshakeDialect>();
    ctx->appendDialectRegistry(registry);
    ctx->loadAllAvailableDialects();
  }
};

//===----------------------------------------------------------------------===//
// SystemADGBuilder public API
//===----------------------------------------------------------------------===//

SystemADGBuilder::SystemADGBuilder(mlir::MLIRContext *ctx,
                                   const std::string &systemName)
    : impl_(std::make_unique<Impl>()) {
  impl_->ctx = ctx;
  impl_->systemName = systemName;
  Impl::initContext(ctx);
}

SystemADGBuilder::SystemADGBuilder(const std::string &systemName)
    : impl_(std::make_unique<Impl>()) {
  impl_->ownedCtx = std::make_unique<mlir::MLIRContext>();
  impl_->ctx = impl_->ownedCtx.get();
  impl_->systemName = systemName;
  Impl::initContext(impl_->ctx);
}

SystemADGBuilder::~SystemADGBuilder() = default;

CoreTypeHandle
SystemADGBuilder::registerCoreType(const std::string &typeName,
                                   mlir::ModuleOp coreModule) {
  if (!coreModule) {
    llvm::report_fatal_error(
        "SystemADGBuilder::registerCoreType(): coreModule must not be null");
  }
  unsigned id = impl_->coreTypes.size();
  impl_->coreTypes.push_back({typeName, coreModule, id});
  return CoreTypeHandle{id};
}

SystemCoreInstance SystemADGBuilder::instantiateCore(CoreTypeHandle type,
                                                     const std::string &name,
                                                     int row, int col) {
  if (type.id >= impl_->coreTypes.size()) {
    llvm::report_fatal_error("SystemADGBuilder: invalid CoreTypeHandle");
  }

  SystemCoreInstance inst;
  inst.instanceName = name;
  inst.coreType = type;
  inst.row = row;
  inst.col = col;

  impl_->coreInstances.push_back(inst);
  return inst;
}

void SystemADGBuilder::setNoCSpec(const NoCSpec &spec) {
  impl_->nocSpec = spec;
}

void SystemADGBuilder::setSharedMemorySpec(const SharedMemorySpec &spec) {
  impl_->sharedMemSpec = spec;
}

mlir::ModuleOp SystemADGBuilder::build() {
  if (impl_->coreInstances.empty()) {
    llvm::report_fatal_error(
        "SystemADGBuilder::build(): no core instances registered");
  }

  // Build CoreType descriptors for the MLIR builder
  std::vector<SystemADGMLIRBuilder::CoreType> mlirCoreTypes;
  for (const auto &ct : impl_->coreTypes) {
    mlirCoreTypes.push_back({ct.typeName, ct.coreModule, ct.id});
  }

  // Build the system module using the MLIR builder
  impl_->builtModule = SystemADGMLIRBuilder::build(
      impl_->ctx, impl_->systemName, mlirCoreTypes,
      impl_->coreInstances, impl_->nocSpec, impl_->sharedMemSpec);

  return impl_->builtModule;
}

void SystemADGBuilder::exportMLIR(const std::string &path) {
  if (!impl_->builtModule) {
    llvm::report_fatal_error(
        "SystemADGBuilder::exportMLIR(): must call build() first");
  }

  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::report_fatal_error(
        llvm::Twine("SystemADGBuilder: cannot write output file: ") + path +
        "\n" + ec.message());
  }

  impl_->builtModule->print(output);
  output.flush();
}

const std::vector<SystemCoreInstance> &
SystemADGBuilder::getCoreInstances() const {
  return impl_->coreInstances;
}

const NoCSpec &SystemADGBuilder::getNoCSpec() const {
  return impl_->nocSpec;
}

const SharedMemorySpec &SystemADGBuilder::getSharedMemorySpec() const {
  return impl_->sharedMemSpec;
}

} // namespace adg
} // namespace loom
