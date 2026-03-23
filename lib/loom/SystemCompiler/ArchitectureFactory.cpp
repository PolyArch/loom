//===-- ArchitectureFactory.cpp - Build SystemArchitectures --------*- C++ -*-===//
//
// Factory for constructing SystemArchitecture instances with real ADG modules.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/ADG/ADGBuilder.h"
#include "loom/ADG/ADGVerifier.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace loom::tapestry;

/// Ensure the context has all dialects needed to parse ADG MLIR.
static void ensureDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

/// Build an ADG for one core type using the ADGBuilder, export to a temp
/// file, and parse back into a ModuleOp owned by the context.
static mlir::OwningOpRef<mlir::ModuleOp>
buildCoreADG(const CoreTypeSpec &spec, mlir::MLIRContext &ctx) {
  const std::string moduleName = spec.name + "_adg";
  loom::adg::ADGBuilder builder(moduleName);

  constexpr unsigned dataWidth = 64;

  // Define function units
  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");

  std::vector<loom::adg::FUHandle> fuList;
  fuList.push_back(fuAdd);

  if (spec.includeMultiplier) {
    auto fuMul = builder.defineBinaryFU("fu_mul", "arith.muli", "i32", "i32");
    fuList.push_back(fuMul);
  }

  if (spec.includeComparison) {
    auto fuCmpi = builder.defineCmpiFU("fu_cmpi", "i32", "slt");
    fuList.push_back(fuCmpi);
  }

  // Define PE with all function units
  auto pe = builder.defineSpatialPE(
      spec.name + "_pe",
      /*numInputs=*/4,
      /*numOutputs=*/4,
      /*bitsWidth=*/dataWidth,
      fuList);

  // Build chessboard mesh topology
  unsigned topLeftInputs = 3;
  unsigned bottomRightOutputs = 1;

  if (spec.includeMemory) {
    // Extra boundary ports for memory
    topLeftInputs += 1;
    bottomRightOutputs += 1;
  }

  auto mesh = builder.buildChessMesh(spec.meshRows, spec.meshCols, pe,
                                     /*decomposableBits=*/-1,
                                     topLeftInputs,
                                     bottomRightOutputs);

  // Add scalar I/O boundary ports
  auto in0 = builder.addScalarInput("in0", dataWidth);
  auto in1 = builder.addScalarInput("in1", dataWidth);
  auto in2 = builder.addScalarInput("in2", dataWidth);

  builder.connectInputToPort(in0, mesh.ingressPorts[0]);
  builder.connectInputToPort(in1, mesh.ingressPorts[1]);
  builder.connectInputToPort(in2, mesh.ingressPorts[2]);

  auto out0 = builder.addScalarOutput("out0", dataWidth);
  builder.connectPortToOutput(mesh.egressPorts[0], out0);

  if (spec.includeMemory && mesh.ingressPorts.size() > 3 &&
      mesh.egressPorts.size() > 1) {
    auto in3 = builder.addScalarInput("in3", dataWidth);
    builder.connectInputToPort(in3, mesh.ingressPorts[3]);
    auto out1 = builder.addScalarOutput("out1", dataWidth);
    builder.connectPortToOutput(mesh.egressPorts[1], out1);
  }

  // Export to temp file
  llvm::SmallString<128> tempPath;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(moduleName, "fabric.mlir", tempPath);
  if (ec) {
    llvm::errs() << "ArchitectureFactory: failed to create temp file: "
                 << ec.message() << "\n";
    return nullptr;
  }

  builder.exportMLIR(std::string(tempPath));

  // Parse back into the caller's context
  ensureDialects(ctx);
  llvm::SourceMgr srcMgr;
  auto buf = llvm::MemoryBuffer::getFile(tempPath);
  if (!buf) {
    llvm::errs() << "ArchitectureFactory: failed to read temp file\n";
    return nullptr;
  }
  srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &ctx);

  // Clean up temp file
  llvm::sys::fs::remove(tempPath);
  // Also remove the viz sidecar if it was generated
  llvm::SmallString<128> vizPath(tempPath);
  llvm::sys::path::replace_extension(vizPath, "viz.json");
  llvm::sys::fs::remove(vizPath);

  return module;
}

SystemArchitecture
loom::tapestry::buildArchitecture(const std::string &systemName,
                                  const std::vector<CoreTypeSpec> &specs,
                                  mlir::MLIRContext &ctx) {
  SystemArchitecture arch;
  arch.name = systemName;

  for (const auto &spec : specs) {
    auto adgModule = buildCoreADG(spec, ctx);
    if (!adgModule) {
      llvm::errs() << "ArchitectureFactory: failed to build ADG for core type '"
                   << spec.name << "'\n";
      arch.coreTypes.clear();
      return arch;
    }

    CoreTypeDesc coreType;
    coreType.name = spec.name;
    coreType.numInstances = spec.numInstances;
    coreType.adgModule = *adgModule;
    coreType.totalPEs = spec.meshRows * spec.meshCols;
    unsigned fusPerPE = 1; // addi is always present
    if (spec.includeMultiplier)
      fusPerPE++;
    if (spec.includeComparison)
      fusPerPE++;
    coreType.totalFUs = coreType.totalPEs * fusPerPE;
    coreType.spmSizeBytes = spec.spmSizeBytes;

    // Release ownership: the module is now owned by the context's
    // operation data structure, accessed via the raw ModuleOp pointer.
    adgModule.release();

    arch.coreTypes.push_back(std::move(coreType));
  }

  return arch;
}

SystemArchitecture
loom::tapestry::buildStandardArchitecture(const std::string &systemName,
                                          unsigned numCoreTypes,
                                          unsigned instancesPerType,
                                          unsigned meshRows,
                                          unsigned meshCols,
                                          mlir::MLIRContext &ctx) {
  std::vector<CoreTypeSpec> specs;
  for (unsigned idx = 0; idx < numCoreTypes; ++idx) {
    CoreTypeSpec spec;
    spec.name = "core_type_" + std::to_string(idx);
    spec.meshRows = meshRows;
    spec.meshCols = meshCols;
    spec.numInstances = instancesPerType;
    spec.spmSizeBytes = 4096;
    // Make core types slightly different
    spec.includeMultiplier = true;
    spec.includeComparison = (idx % 2 == 0);
    spec.includeMemory = true;
    specs.push_back(spec);
  }
  return buildArchitecture(systemName, specs, ctx);
}
