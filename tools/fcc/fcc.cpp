#include "fcc_args.h"
#include "fcc_pipeline.h"

#include "fcc/Dialect/Dataflow/DataflowDialect.h"
#include "fcc/Dialect/Dataflow/DataflowOps.h"
#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/DFGBuilder.h"
#include "fcc/Mapper/Mapper.h"
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

using namespace mlir;
using namespace fcc;

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

  // Stage 1+2: C -> LLVM IR -> MLIR LLVM dialect (in one step)
  std::string llPath = base + ".ll";
  llvm::outs() << "fcc: compiling and importing...\n";
  auto module = compileAndImport(args, context, llPath);
  if (!module)
    return 1;

  std::string llvmMlirPath = base + ".llvm.mlir";
  if (failed(writeMLIR(*module, llvmMlirPath)))
    return 1;

  // Stage 3: LLVM dialect -> CF stage
  llvm::outs() << "fcc: converting LLVM to CF...\n";
  if (failed(runLLVMToCF(*module)))
    return 1;

  std::string cfPath = base + ".cf.mlir";
  if (failed(writeMLIR(*module, cfPath)))
    return 1;

  // Stage 4: CF -> SCF
  llvm::outs() << "fcc: lifting CF to SCF...\n";
  if (failed(runCFToSCF(*module)))
    return 1;

  std::string scfPath = base + ".scf.mlir";
  if (failed(writeMLIR(*module, scfPath)))
    return 1;

  // Stage 5: SCF -> DFG (handshake + dataflow)
  llvm::outs() << "fcc: converting SCF to DFG...\n";
  if (failed(runSCFToDFG(*module)))
    return 1;

  std::string dfgPath = base + ".dfg.mlir";
  if (failed(writeMLIR(*module, dfgPath)))
    return 1;

  // Stage 6: Host code generation
  std::string hostPath = base + "_host.c";
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
  llvm::outs() << "  " << hostPath << "\n";

  // If ADG is provided, continue to mapping
  if (!args.adgPath.empty()) {
    llvm::outs() << "fcc: loading ADG from " << args.adgPath << "...\n";

    // Load and parse the ADG MLIR file.
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

    // Flatten the ADG.
    llvm::outs() << "fcc: flattening ADG...\n";
    fcc::ADGFlattener flattener;
    if (!flattener.flatten(*adgModule, &context)) {
      llvm::errs() << "fcc: ADG flattening failed\n";
      return 1;
    }

    // Build the DFG from the compiled handshake.func.
    llvm::outs() << "fcc: building DFG...\n";
    fcc::DFGBuilder dfgBuilder;
    if (!dfgBuilder.build(*module, &context)) {
      llvm::errs() << "fcc: DFG building failed\n";
      return 1;
    }

    // Run the mapper.
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
      // Still generate partial output.
    }

    // Generate config files.
    llvm::outs() << "fcc: generating config...\n";
    fcc::ConfigGen configGen;
    configGen.generate(mapResult.state, dfgBuilder.getDFG(),
                       flattener.getADG(), flattener, base,
                       static_cast<int>(args.mapperSeed));

    llvm::outs() << "fcc: mapping output:\n";
    llvm::outs() << "  " << base << ".map.json\n";
    llvm::outs() << "  " << base << ".map.txt\n";

    // Generate visualization HTML.
    std::string vizPath = base + ".viz.html";
    llvm::outs() << "fcc: generating visualization...\n";
    if (failed(fcc::exportVisualization(vizPath, flattener.getADG(),
                                        dfgBuilder.getDFG(), mapResult.state,
                                        flattener))) {
      llvm::errs() << "fcc: warning: visualization generation failed\n";
    } else {
      llvm::outs() << "  " << vizPath << "\n";
    }
  }

  // If simulation requested
  if (args.simulate) {
    llvm::outs() << "fcc: simulation (not yet implemented)...\n";
    // TODO: Batch 5 - standalone simulator
  }

  return 0;
}
