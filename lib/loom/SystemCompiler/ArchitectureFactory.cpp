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

  if (spec.includeMemory) {
    auto fuConstI32 =
        builder.defineConstantFU("fu_const_i32_0", "i32", "0 : i32");
    auto fuConstI32One =
        builder.defineConstantFU("fu_const_i32_1", "i32", "1 : i32");
    auto fuConstIndex =
        builder.defineConstantFU("fu_const_index_0", "index", "0 : index");
    auto fuConstIndexOne =
        builder.defineConstantFU("fu_const_index_1", "index", "1 : index");
    auto fuIndexToI32 =
        builder.defineIndexCastFU("fu_index_to_i32", "index", "i32");
    auto fuI32ToIndex =
        builder.defineIndexCastFU("fu_i32_to_index", "i32", "index");
    auto fuStream = builder.defineStreamFU("fu_stream");
    auto fuMuxI32 = builder.defineMuxFU("fu_mux_i32", "i32");
    auto fuMuxIndex = builder.defineMuxFU("fu_mux_index", "index");
    auto fuMuxNone = builder.defineMuxFU("fu_mux_none", "none");
    auto fuJoin = builder.defineJoinFU("fu_join", 4);
    auto fuGateI32 = builder.defineGateFU("fu_gate_i32", "i32");
    auto fuGateIndex = builder.defineGateFU("fu_gate_index", "index");
    auto fuGateI1 = builder.defineGateFU("fu_gate_i1", "i1");
    auto fuCarryI32 = builder.defineCarryFU("fu_carry_i32", "i32");
    auto fuCarryIndex = builder.defineCarryFU("fu_carry_index", "index");
    auto fuCarryNone = builder.defineCarryFU("fu_carry_none", "none");
    auto fuCondBrI32 = builder.defineCondBrFU("fu_cond_br_i32", "i32");
    auto fuCondBrIndex = builder.defineCondBrFU("fu_cond_br_index", "index");
    auto fuCondBrNone = builder.defineCondBrFU("fu_cond_br_none", "none");
    auto fuInvariantI32 =
        builder.defineInvariantFU("fu_invariant_i32", "i32");
    auto fuInvariantIndex =
        builder.defineInvariantFU("fu_invariant_index", "index");
    auto fuInvariantI1 = builder.defineInvariantFU("fu_invariant_i1", "i1");
    auto fuInvariantNone =
        builder.defineInvariantFU("fu_invariant_none", "none");
    auto fuLoad = builder.defineLoadFU("fu_load_i32", "index", "i32");
    auto fuStore = builder.defineStoreFU("fu_store_i32", "index", "i32");

    fuList.push_back(fuConstI32);
    fuList.push_back(fuConstI32One);
    fuList.push_back(fuConstIndex);
    fuList.push_back(fuConstIndexOne);
    fuList.push_back(fuIndexToI32);
    fuList.push_back(fuI32ToIndex);
    fuList.push_back(fuStream);
    fuList.push_back(fuMuxI32);
    fuList.push_back(fuMuxIndex);
    fuList.push_back(fuMuxNone);
    fuList.push_back(fuJoin);
    fuList.push_back(fuGateI32);
    fuList.push_back(fuGateIndex);
    fuList.push_back(fuGateI1);
    fuList.push_back(fuCarryI32);
    fuList.push_back(fuCarryIndex);
    fuList.push_back(fuCarryNone);
    fuList.push_back(fuCondBrI32);
    fuList.push_back(fuCondBrIndex);
    fuList.push_back(fuCondBrNone);
    fuList.push_back(fuInvariantI32);
    fuList.push_back(fuInvariantIndex);
    fuList.push_back(fuInvariantI1);
    fuList.push_back(fuInvariantNone);
    fuList.push_back(fuLoad);
    fuList.push_back(fuStore);
  }

  // Define PE with all function units
  auto pe = builder.defineSpatialPE(
      spec.name + "_pe",
      /*numInputs=*/4,
      /*numOutputs=*/4,
      /*bitsWidth=*/dataWidth,
      fuList);

  // Build chessboard mesh topology.
  loom::adg::ChessMeshOptions meshOpts;
  meshOpts.decomposableBits = -1;

  if (spec.includeMemory) {
    // Reserve a left-side ingress block for extmem outputs and scalar inputs,
    // and a bottom-side egress block for extmem inputs plus the scalar output.
    meshOpts.topLeftExtraInputs = 6;
    meshOpts.bottomLeftExtraOutputs = 3;
    meshOpts.bottomRightExtraOutputs = 1;
  } else {
    meshOpts.topLeftExtraInputs = 3;
    meshOpts.bottomRightExtraOutputs = 1;
  }

  auto mesh = builder.buildChessMesh(
      spec.meshRows, spec.meshCols,
      [&](unsigned, unsigned) { return pe; }, meshOpts);

  // Add scalar I/O boundary ports
  auto in0 = builder.addScalarInput("in0", dataWidth);
  auto in1 = builder.addScalarInput("in1", dataWidth);
  auto in2 = builder.addScalarInput("in2", dataWidth);

  auto out0 = builder.addScalarOutput("out0", dataWidth);
  if (spec.includeMemory) {
    // External memory uses the first three ingress ports and the first three
    // left-side egress ports. The scalar input/output is placed after them.
    auto extMem = builder.defineExtMemory(spec.name + "_extmem", 1, 1);
    auto extMems = builder.instantiateExtMemArray(1, extMem, "extmem");
    auto memrefs = builder.addMemrefInputs("buffer", 1, "memref<?xi32>");
    builder.connectMemrefToExtMem(memrefs[0], extMems[0]);

    builder.connect(extMems[0], 0, mesh.ingressPorts[0].instance,
                    mesh.ingressPorts[0].port);
    builder.connect(extMems[0], 1, mesh.ingressPorts[1].instance,
                    mesh.ingressPorts[1].port);
    builder.connect(extMems[0], 2, mesh.ingressPorts[2].instance,
                    mesh.ingressPorts[2].port);

    builder.connect(mesh.egressPorts[0].instance, mesh.egressPorts[0].port,
                    extMems[0], 1);
    builder.connect(mesh.egressPorts[1].instance, mesh.egressPorts[1].port,
                    extMems[0], 2);
    builder.connect(mesh.egressPorts[2].instance, mesh.egressPorts[2].port,
                    extMems[0], 3);

    builder.connectInputToPort(in0, mesh.ingressPorts[3]);
    builder.connectInputToPort(in1, mesh.ingressPorts[4]);
    builder.connectInputToPort(in2, mesh.ingressPorts[5]);
    builder.connectPortToOutput(mesh.egressPorts[3], out0);
  } else {
    builder.connectInputToPort(in0, mesh.ingressPorts[0]);
    builder.connectInputToPort(in1, mesh.ingressPorts[1]);
    builder.connectInputToPort(in2, mesh.ingressPorts[2]);
    builder.connectPortToOutput(mesh.egressPorts[0], out0);
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
