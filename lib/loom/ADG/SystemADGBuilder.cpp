//===-- SystemADGBuilder.cpp - System-level ADG Builder -----------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the SystemADGBuilder which composes per-core fabric.module
// definitions into a system-level fabric.module with NoC connectivity and
// shared memory hierarchy. Uses the MLIR builder API via SystemADGMLIRBuilder
// to generate proper typed ops instead of string concatenation.
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
#include "mlir/IR/AsmState.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Internal data structures
//===----------------------------------------------------------------------===//

struct CoreTypeDef {
  std::string typeName;
  std::string mlirText;
  unsigned id;
};

struct SystemADGBuilder::Impl {
  std::string systemName;
  std::vector<CoreTypeDef> coreTypes;
  std::vector<SystemCoreInstance> coreInstances;
  NoCSpec nocSpec;
  SharedMemorySpec sharedMemSpec;
  std::string builtMLIR;
  bool isBuilt = false;

  /// Generate the system MLIR text using the MLIR builder API.
  std::string generateSystemMLIR() const;
};

//===----------------------------------------------------------------------===//
// MLIR generation via builder API
//===----------------------------------------------------------------------===//

std::string SystemADGBuilder::Impl::generateSystemMLIR() const {
  // Set up MLIR context with all dialects needed to parse core type MLIR
  // and print the system module in custom assembly format.
  // Core types contain fabric/dataflow/arith/math/memref/handshake ops.
  mlir::DialectRegistry registry;
  registry.insert<loom::fabric::FabricDialect>();
  registry.insert<loom::dataflow::DataflowDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // Build CoreType descriptors for the MLIR builder
  std::vector<SystemADGMLIRBuilder::CoreType> mlirCoreTypes;
  for (const auto &ct : coreTypes) {
    mlirCoreTypes.push_back({ct.typeName, ct.mlirText, ct.id});
  }

  // Build the system module using the MLIR builder
  mlir::ModuleOp sysModule = SystemADGMLIRBuilder::build(
      &ctx, systemName, mlirCoreTypes, coreInstances, nocSpec, sharedMemSpec);

  // Print the module to a string.
  // The output uses MLIR generic format -- this is valid and parseable.
  // Custom assembly format would require printing ops individually with
  // a properly configured AsmState, but the generic format is semantically
  // equivalent and fully round-trippable.
  std::string result;
  llvm::raw_string_ostream os(result);
  sysModule->print(os);
  os.flush();

  return result;
}

//===----------------------------------------------------------------------===//
// SystemADGBuilder public API
//===----------------------------------------------------------------------===//

SystemADGBuilder::SystemADGBuilder(const std::string &systemName)
    : impl_(std::make_unique<Impl>()) {
  impl_->systemName = systemName;
}

SystemADGBuilder::~SystemADGBuilder() = default;

CoreTypeHandle
SystemADGBuilder::registerCoreType(const std::string &typeName,
                                   const std::string &mlirText) {
  unsigned id = impl_->coreTypes.size();
  impl_->coreTypes.push_back({typeName, mlirText, id});
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

void SystemADGBuilder::build() {
  if (impl_->coreInstances.empty()) {
    llvm::report_fatal_error(
        "SystemADGBuilder::build(): no core instances registered");
  }

  impl_->builtMLIR = impl_->generateSystemMLIR();
  impl_->isBuilt = true;
}

void SystemADGBuilder::exportSystemMLIR(const std::string &path) {
  if (!impl_->isBuilt) {
    llvm::report_fatal_error(
        "SystemADGBuilder::exportSystemMLIR(): must call build() first");
  }

  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::report_fatal_error(
        llvm::Twine("SystemADGBuilder: cannot write output file: ") + path +
        "\n" + ec.message());
  }

  output << impl_->builtMLIR;
  output.flush();
}

std::string SystemADGBuilder::getSystemMLIR() const {
  if (!impl_->isBuilt) {
    llvm::report_fatal_error(
        "SystemADGBuilder::getSystemMLIR(): must call build() first");
  }
  return impl_->builtMLIR;
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
