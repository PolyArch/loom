#include "fcc_args.h"
#include "fcc_pipeline.h"

#include "fcc/Dialect/Dataflow/DataflowDialect.h"
#include "fcc/Dialect/Dataflow/DataflowOps.h"
#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/ADG/ADGVerifier.h"
#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/DFGBuilder.h"
#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/TypeCompat.h"
#include "fcc/Simulator/SimArtifactWriter.h"
#include "fcc/Simulator/SimSession.h"
#include "fcc/Viz/VizExporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;
using namespace fcc;

static OwningOpRef<ModuleOp>
loadMLIR(const std::string &path, MLIRContext &context) {
  llvm::SourceMgr srcMgr;
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf) {
    llvm::errs() << "fcc: cannot open " << path << "\n";
    return {};
  }
  srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(srcMgr, &context);
}

static void attachADGCapacityAttrs(ModuleOp module, const std::string &adgPath,
                                   MLIRContext &context) {
  if (adgPath.empty())
    return;

  auto adgModule = loadMLIR(adgPath, context);
  if (!adgModule) {
    llvm::errs() << "fcc: warning: cannot read ADG for capacity summary: "
                 << adgPath << "\n";
    return;
  }

  fcc::ADGFlattener flattener;
  if (!flattener.flatten(*adgModule, &context)) {
    llvm::errs() << "fcc: warning: cannot flatten ADG for capacity summary: "
                 << adgPath << "\n";
    return;
  }

  const Graph &adg = flattener.getADG();
  unsigned totalPEs = flattener.getPEContainment().size();
  unsigned totalFUs = 0;
  for (const auto &pe : flattener.getPEContainment())
    totalFUs += pe.fuNodeIds.size();

  unsigned totalMemModules = 0;
  unsigned maxDataWidth = 0;
  for (const Node *node : adg.nodeRange()) {
    if (getNodeAttrStr(node, "resource_class") == "memory")
      totalMemModules++;
  }
  for (const Port *port : adg.portRange()) {
    if (auto info = fcc::detail::getPortTypeInfo(port->type)) {
      maxDataWidth = std::max(maxDataWidth, info->valueWidth);
      continue;
    }
    if (auto memWidth = fcc::detail::getMemRefElementWidth(port->type))
      maxDataWidth = std::max(maxDataWidth, *memWidth);
  }

  Builder builder(&context);
  module->setAttr("fcc.adg_total_pes",
                  builder.getI64IntegerAttr(static_cast<int64_t>(totalPEs)));
  module->setAttr("fcc.adg_total_fus",
                  builder.getI64IntegerAttr(static_cast<int64_t>(totalFUs)));
  module->setAttr(
      "fcc.adg_total_mem_modules",
      builder.getI64IntegerAttr(static_cast<int64_t>(totalMemModules)));
  module->setAttr(
      "fcc.adg_max_data_width",
      builder.getI64IntegerAttr(static_cast<int64_t>(maxDataWidth)));

  llvm::outs() << "fcc: ADG capacity summary: PEs=" << totalPEs
               << ", FUs=" << totalFUs
               << ", mem=" << totalMemModules
               << ", maxWidth=" << maxDataWidth << "\n";
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse arguments
  FccArgs args;
  if (!parseArgs(argc, argv, args))
    return 1;

  // Set up MLIR context with all needed dialects
  DialectRegistry registry;
  registry.insert<DLTIDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<ub::UBDialect>();
  registry.insert<fcc::dataflow::DataflowDialect>();
  registry.insert<fcc::fabric::FabricDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string base = args.outputDir + "/" + args.baseName;

  // Helper: load ADG, build DFG, run mapper, generate viz
  auto runMappingPipeline = [&](OwningOpRef<ModuleOp> &dfgModule) -> int {
    llvm::outs() << "fcc: loading ADG from " << args.adgPath << "...\n";

    llvm::SourceMgr adgSourceMgr;
    auto adgBuf = llvm::MemoryBuffer::getFile(args.adgPath);
    if (!adgBuf) {
      llvm::errs() << "fcc: cannot open ADG file: " << args.adgPath << "\n";
      return 1;
    }
    adgSourceMgr.AddNewSourceBuffer(std::move(*adgBuf), llvm::SMLoc());
    auto adgModule = parseSourceFile<ModuleOp>(adgSourceMgr, &context);
    if (!adgModule) {
      llvm::errs() << "fcc: failed to parse ADG MLIR\n";
      return 1;
    }

    // Verify fabric.module compliance (no dangling ports)
    if (failed(fcc::verifyFabricModule(*adgModule))) {
      llvm::errs() << "fcc: ADG fabric.module verification failed\n";
      return 1;
    }

    llvm::outs() << "fcc: flattening ADG...\n";
    fcc::ADGFlattener flattener;
    if (!flattener.flatten(*adgModule, &context)) {
      llvm::errs() << "fcc: ADG flattening failed\n";
      return 1;
    }

    llvm::outs() << "fcc: building DFG...\n";
    fcc::DFGBuilder dfgBuilder;
    if (!dfgBuilder.build(*dfgModule, &context)) {
      llvm::errs() << "fcc: DFG building failed\n";
      return 1;
    }

    llvm::outs() << "fcc: running mapper...\n";
    fcc::Mapper mapper;
    fcc::Mapper::Options mapOpts;
    mapOpts.budgetSeconds = static_cast<double>(args.mapperBudget);
    mapOpts.seed = static_cast<int>(args.mapperSeed);
    mapOpts.verbose = true;

    auto mapResult =
        mapper.run(dfgBuilder.getDFG(), flattener.getADG(), flattener, mapOpts);

    if (!mapResult.success) {
      llvm::errs() << "fcc: mapping failed: " << mapResult.diagnostics << "\n";
    }

    llvm::outs() << "fcc: generating config...\n";
    fcc::ConfigGen configGen;
    if (!configGen.generate(mapResult.state, dfgBuilder.getDFG(),
                            flattener.getADG(), flattener, base,
                            static_cast<int>(args.mapperSeed))) {
      llvm::errs() << "fcc: config generation failed\n";
      return 1;
    }
    llvm::outs() << "fcc: mapping output:\n";
    llvm::outs() << "  " << base << ".config.bin\n";
    llvm::outs() << "  " << base << ".config.json\n";
    llvm::outs() << "  " << base << ".config.h\n";
    llvm::outs() << "  " << base << ".map.json\n";
    llvm::outs() << "  " << base << ".map.txt\n";
    if (!configGen.isConfigComplete()) {
      llvm::outs() << "fcc: warning: config artifacts currently cover "
                      "routing/tag/memory state but not all PE/FU runtime "
                      "configuration\n";
    }

    // Generate visualization with mapping data
    std::string vizPath = base + ".viz.html";
    std::string mapJsonPath = base + ".map.json";
    llvm::outs() << "fcc: generating visualization...\n";

    // We need the original MLIR modules for viz serialization.
    // adgModule is already loaded above. dfgModule is the parameter.
    if (failed(fcc::exportVizWithMapping(vizPath, *adgModule, *dfgModule,
                                          mapJsonPath, &context))) {
      llvm::errs() << "fcc: warning: visualization generation failed\n";
    } else {
      llvm::outs() << "  " << vizPath << "\n";
    }

    if (args.simulate) {
      llvm::outs() << "fcc: running standalone simulation...\n";
      fcc::sim::SimSession session;
      fcc::sim::SimArtifactWriter artifactWriter;
      std::string tracePath = base + ".sim.trace";
      std::string statPath = base + ".sim.stat";

      if (std::string err = session.connect(); !err.empty()) {
        llvm::errs() << "fcc: simulation setup failed: " << err << "\n";
        return 1;
      }
      if (std::string err = session.buildFromMappedState(
              dfgBuilder.getDFG(), flattener.getADG(), mapResult.state);
          !err.empty()) {
        llvm::errs() << "fcc: simulation graph build failed: " << err << "\n";
        return 1;
      }
      if (std::string err = session.loadConfig(configGen.getConfigBlob());
          !err.empty()) {
        llvm::errs() << "fcc: simulation config load failed: " << err << "\n";
        return 1;
      }

      auto [simResult, invokeErr] = session.invoke();
      if (!artifactWriter.writeTrace(simResult, tracePath) ||
          !artifactWriter.writeStat(simResult, statPath)) {
        llvm::errs() << "fcc: failed to write simulation artifacts\n";
        return 1;
      }

      llvm::outs() << "  " << tracePath << "\n";
      llvm::outs() << "  " << statPath << "\n";

      if (!invokeErr.empty()) {
        llvm::errs() << "fcc: simulation invocation failed: " << invokeErr
                     << "\n";
        return 1;
      }
      if (!simResult.success) {
        llvm::errs() << "fcc: simulation failed: " << simResult.errorMessage
                     << "\n";
        return 1;
      }
    }
    return 0;
  };

  // ===== Viz-only mode: just visualize, no mapping =====
  if (args.vizOnly) {
    OwningOpRef<ModuleOp> adgMod, dfgMod;
    if (!args.adgPath.empty()) {
      adgMod = loadMLIR(args.adgPath, context);
      if (!adgMod) return 1;
      if (failed(fcc::verifyFabricModule(*adgMod))) {
        llvm::errs() << "fcc: ADG fabric.module verification failed\n";
        return 1;
      }
      llvm::outs() << "fcc: loaded ADG from " << args.adgPath << "\n";
    }
    if (!args.dfgPath.empty()) {
      dfgMod = loadMLIR(args.dfgPath, context);
      if (!dfgMod) return 1;
      llvm::outs() << "fcc: loaded DFG from " << args.dfgPath << "\n";
    }
    std::string vizPath = base + ".viz.html";
    llvm::outs() << "fcc: generating viz-only...\n";
    if (failed(fcc::exportVizOnly(
            vizPath, adgMod ? *adgMod : ModuleOp(),
            dfgMod ? *dfgMod : ModuleOp(), &context))) {
      llvm::errs() << "fcc: viz generation failed\n";
      return 1;
    }
    llvm::outs() << "  " << vizPath << "\n";
    return 0;
  }

  // ===== DFG-direct mode: skip frontend, load pre-built DFG =====
  if (!args.dfgPath.empty()) {
    llvm::outs() << "fcc: loading DFG from " << args.dfgPath << "...\n";
    llvm::SourceMgr dfgSourceMgr;
    auto dfgBuf = llvm::MemoryBuffer::getFile(args.dfgPath);
    if (!dfgBuf) {
      llvm::errs() << "fcc: cannot open DFG file: " << args.dfgPath << "\n";
      return 1;
    }
    dfgSourceMgr.AddNewSourceBuffer(std::move(*dfgBuf), llvm::SMLoc());
    auto dfgModule = parseSourceFile<ModuleOp>(dfgSourceMgr, &context);
    if (!dfgModule) {
      llvm::errs() << "fcc: failed to parse DFG MLIR\n";
      return 1;
    }

    return runMappingPipeline(dfgModule);
  }

  // ===== Full pipeline: C -> LLVM -> CF -> SCF -> DFG =====
  std::string llPath = base + ".ll";
  llvm::outs() << "fcc: compiling and importing...\n";
  auto module = compileAndImport(args, context, llPath);
  if (!module)
    return 1;

  std::string llvmMlirPath = base + ".llvm.mlir";
  if (failed(writeMLIR(*module, llvmMlirPath)))
    return 1;

  llvm::outs() << "fcc: converting LLVM to CF...\n";
  if (failed(runLLVMToCF(*module)))
    return 1;

  std::string cfPath = base + ".cf.mlir";
  if (failed(writeMLIR(*module, cfPath)))
    return 1;

  llvm::outs() << "fcc: lifting CF to SCF...\n";
  if (failed(runCFToSCF(*module)))
    return 1;

  std::string scfPath = base + ".scf.mlir";
  if (failed(writeMLIR(*module, scfPath)))
    return 1;

  attachADGCapacityAttrs(*module, args.adgPath, context);

  llvm::outs() << "fcc: converting SCF to DFG...\n";
  if (failed(runSCFToDFG(*module)))
    return 1;

  std::string dfgPath = base + ".dfg.mlir";
  if (failed(writeMLIR(*module, dfgPath)))
    return 1;

  std::string hostPath = base + "_host.c";
  std::string accelHeaderPath = args.outputDir + "/fcc_accel.h";
  std::string accelRuntimePath = args.outputDir + "/fcc_accel.c";
  std::string origSource =
      args.sources.empty() ? "" : args.sources[0];
  llvm::outs() << "fcc: generating host code...\n";
  if (failed(runHostCodeGen(*module, hostPath, origSource)))
    return 1;

  llvm::outs() << "fcc: compilation complete.\n";
  llvm::outs() << "  " << llPath << "\n";
  llvm::outs() << "  " << llvmMlirPath << "\n";
  llvm::outs() << "  " << cfPath << "\n";
  llvm::outs() << "  " << scfPath << "\n";
  llvm::outs() << "  " << dfgPath << "\n";
  llvm::outs() << "  " << accelHeaderPath << "\n";
  llvm::outs() << "  " << accelRuntimePath << "\n";
  llvm::outs() << "  " << hostPath << "\n";

  if (!args.adgPath.empty()) {
    int rc = runMappingPipeline(module);
    if (rc != 0)
      return rc;
  }

  return 0;
}
