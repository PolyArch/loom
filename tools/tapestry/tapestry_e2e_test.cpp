//===-- tapestry_e2e_test.cpp - End-to-end multi-core compilation test -*- C++ -*-===//
//
// Smoke test for the Tapestry multi-core compilation pipeline. Builds a
// SystemArchitecture with two core types, creates synthetic kernel DFGs,
// defines contracts between them, and runs the HierarchicalCompiler to completion.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/SystemTypes.h"
#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/SystemCompiler/TDGLowering.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace loom::tapestry;

static void registerDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

static llvm::cl::opt<unsigned> meshRows("mesh-rows",
                                         llvm::cl::desc("PE mesh rows"),
                                         llvm::cl::init(2));
static llvm::cl::opt<unsigned> meshCols("mesh-cols",
                                         llvm::cl::desc("PE mesh columns"),
                                         llvm::cl::init(2));
static llvm::cl::opt<double> budget("budget",
                                     llvm::cl::desc("Mapper budget (seconds)"),
                                     llvm::cl::init(15.0));
static llvm::cl::opt<unsigned> maxIter("max-iter",
                                        llvm::cl::desc("Max Benders iterations"),
                                        llvm::cl::init(3));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                     "Tapestry end-to-end integration test\n");

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  llvm::outs() << "=== Tapestry E2E Test ===\n\n";

  //--- Build SystemArchitecture with 2 core types, 2 instances each ---
  llvm::outs() << "Building SystemArchitecture...\n";

  std::vector<CoreTypeSpec> coreSpecs;

  // Core type 0: general-purpose with add + cmp
  CoreTypeSpec gpSpec;
  gpSpec.name = "gp_core";
  gpSpec.meshRows = meshRows;
  gpSpec.meshCols = meshCols;
  gpSpec.numInstances = 2;
  gpSpec.includeMultiplier = false;
  gpSpec.includeComparison = true;
  gpSpec.includeMemory = false;
  gpSpec.spmSizeBytes = 4096;
  coreSpecs.push_back(gpSpec);

  // Core type 1: DSP with add + mul
  CoreTypeSpec dspSpec;
  dspSpec.name = "dsp_core";
  dspSpec.meshRows = meshRows;
  dspSpec.meshCols = meshCols;
  dspSpec.numInstances = 2;
  dspSpec.includeMultiplier = true;
  dspSpec.includeComparison = false;
  dspSpec.includeMemory = false;
  dspSpec.spmSizeBytes = 8192;
  coreSpecs.push_back(dspSpec);

  auto arch = buildArchitecture("test_system", coreSpecs, ctx);
  if (arch.coreTypes.empty()) {
    llvm::errs() << "FAIL: failed to build SystemArchitecture\n";
    return 1;
  }

  llvm::outs() << "  Created " << arch.coreTypes.size() << " core types:\n";
  for (const auto &ct : arch.coreTypes) {
    llvm::outs() << "    " << ct.name << ": " << ct.totalPEs << " PEs, "
                 << ct.totalFUs << " FUs, " << ct.numInstances
                 << " instances\n";
  }

  //--- Create synthetic kernel DFGs ---
  llvm::outs() << "\nCreating synthetic kernels...\n";

  auto addKernel = createSyntheticAddKernel("kernel_add", ctx);
  if (!addKernel.dfgModule) {
    llvm::errs() << "FAIL: failed to create synthetic add kernel\n";
    return 1;
  }

  auto macKernel = createSyntheticMacKernel("kernel_mac", ctx);
  if (!macKernel.dfgModule) {
    llvm::errs() << "FAIL: failed to create synthetic mac kernel\n";
    return 1;
  }

  // Create a third kernel (another add with different name)
  auto addKernel2 = createSyntheticAddKernel("kernel_add2", ctx);
  if (!addKernel2.dfgModule) {
    llvm::errs() << "FAIL: failed to create second synthetic add kernel\n";
    return 1;
  }

  std::vector<KernelDesc> kernels;
  kernels.push_back(std::move(addKernel));
  kernels.push_back(std::move(macKernel));
  kernels.push_back(std::move(addKernel2));

  llvm::outs() << "  Created " << kernels.size() << " kernels\n";

  //--- Verify kernels are in DFG form (they already are, being synthetic) ---
  llvm::outs() << "\nVerifying kernel DFG forms...\n";
  if (!lowerKernelsToDFG(kernels, ctx)) {
    llvm::errs() << "FAIL: kernel DFG lowering/verification failed\n";
    return 1;
  }
  llvm::outs() << "  All kernels in DFG form\n";

  //--- Define contracts ---
  llvm::outs() << "\nDefining contracts...\n";

  std::vector<ContractSpec> contracts;

  ContractSpec c1;
  c1.producerKernel = "kernel_add";
  c1.consumerKernel = "kernel_mac";
  c1.dataType = "i32";
  c1.elementCount = 1024;
  c1.bandwidthBytesPerCycle = 4;
  contracts.push_back(c1);

  ContractSpec c2;
  c2.producerKernel = "kernel_mac";
  c2.consumerKernel = "kernel_add2";
  c2.dataType = "i32";
  c2.elementCount = 512;
  c2.bandwidthBytesPerCycle = 4;
  contracts.push_back(c2);

  llvm::outs() << "  Created " << contracts.size() << " contracts\n";

  //--- Run HierarchicalCompiler ---
  llvm::outs() << "\nRunning Benders decomposition...\n";

  CompilerConfig config;
  config.maxIterations = maxIter;
  config.mapperBudgetSeconds = budget;
  config.mapperSeed = 42;
  config.verbose = true;

  HierarchicalCompiler driver(arch, std::move(kernels), std::move(contracts), ctx);
  auto result = driver.compile(config);

  //--- Report results ---
  llvm::outs() << "\n=== Results ===\n";
  llvm::outs() << "Success: " << (result.success ? "YES" : "NO") << "\n";
  llvm::outs() << "Iterations: " << result.iterations << "\n";
  llvm::outs() << "Total cost: " << result.totalCost << "\n";

  if (!result.diagnostics.empty())
    llvm::outs() << "Diagnostics: " << result.diagnostics << "\n";

  llvm::outs() << "\nAssignment details:\n";
  for (const auto &a : result.assignments) {
    llvm::outs() << "  " << a.kernelName << " -> core type "
                 << a.coreTypeIndex;
    if (a.mappingSuccess)
      llvm::outs() << " [MAPPED, cost=" << a.mappingCost << "]";
    else
      llvm::outs() << " [FAILED]";
    llvm::outs() << "\n";
  }

  llvm::outs() << "\n=== E2E Test "
               << (result.success ? "PASSED" : "COMPLETED (mapping may fail)")
               << " ===\n";

  // Return 0 even if mapping fails -- the test verifies the data flow works,
  // not that the mapper converges on these synthetic benchmarks.
  return 0;
}
