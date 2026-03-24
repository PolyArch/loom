//===-- KernelCompiler.cpp - C/C++ source to DFG compiler ---------*- C++ -*-===//
//
// Implements the KernelCompiler class that wraps Loom's existing single-core
// frontend pipeline for use by the Tapestry multi-core layer.
//
// The pipeline stages are:
//   1. C/C++ -> LLVM IR (via in-process Clang)
//   2. LLVM IR -> MLIR LLVM dialect (via translateLLVMIRToModule)
//   3. LLVM dialect -> CF (func/arith/memref/cf)
//   4. CF -> SCF (structured control flow)
//   5. Per-function extraction + MarkDFGDomain + SCFToDFG
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/KernelCompiler.h"
#include "loom/Conversion/Passes.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/Passes.h"

// Clang in-process frontend
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

using namespace mlir;
using namespace loom::tapestry;

// ---------------------------------------------------------------------------
// Internal helpers (mirror the logic in tools/loom/loom_pipeline.cpp)
// ---------------------------------------------------------------------------

namespace {

/// Detect the Clang input language from the source file extension.
llvm::StringRef detectClangInputLanguage(const std::string &srcPath) {
  llvm::StringRef ext = llvm::sys::path::extension(srcPath);
  if (ext.equals_insensitive(".cc") || ext.equals_insensitive(".cpp") ||
      ext.equals_insensitive(".cxx") || ext.equals_insensitive(".c++") ||
      ext == ".C")
    return "c++";
  return "c";
}

/// Compile a single C/C++ source to an LLVM Module using in-process Clang.
std::unique_ptr<llvm::Module>
compileSourceToLLVM(const std::string &srcPath, llvm::LLVMContext &llvmCtx,
                    const std::vector<std::string> &includePaths) {
  clang::DiagnosticOptions diagOpts;
  auto diagIDs = llvm::makeIntrusiveRefCnt<clang::DiagnosticIDs>();
  auto *diagPrinter =
      new clang::TextDiagnosticPrinter(llvm::errs(), diagOpts);
  clang::DiagnosticsEngine diags(diagIDs, diagOpts, diagPrinter);

  std::string triple = llvm::sys::getDefaultTargetTriple();
  clang::driver::Driver driver("clang", triple, diags);
  driver.setCheckInputsExist(true);

  llvm::SmallVector<const char *> driverArgs;
  driverArgs.push_back("clang");
  driverArgs.push_back("-x");
  llvm::StringRef inputLang = detectClangInputLanguage(srcPath);
  driverArgs.push_back(inputLang.data());
  driverArgs.push_back("-O1");
  driverArgs.push_back("-c");
  driverArgs.push_back("-emit-llvm");
  driverArgs.push_back("-fno-discard-value-names");
  driverArgs.push_back("-fno-math-errno");
  driverArgs.push_back("-gline-tables-only");
  driverArgs.push_back("-resource-dir");
  driverArgs.push_back(LOOM_CLANG_RESOURCE_DIR);

  llvm::SmallVector<std::string> incFlags;
  for (const auto &inc : includePaths)
    incFlags.push_back("-I" + inc);
  for (const auto &f : incFlags)
    driverArgs.push_back(f.c_str());

  driverArgs.push_back(srcPath.c_str());

  auto compilation = driver.BuildCompilation(driverArgs);
  if (!compilation)
    return nullptr;

  const clang::driver::JobList &jobs = compilation->getJobs();
  const clang::driver::Command *cc1Cmd = nullptr;
  for (const auto &job : jobs) {
    if (llvm::StringRef(job.getCreator().getName()) == "clang") {
      cc1Cmd = &job;
      break;
    }
  }
  if (!cc1Cmd) {
    llvm::errs() << "KernelCompiler: no cc1 job found\n";
    return nullptr;
  }

  auto invocation = std::make_shared<clang::CompilerInvocation>();
  if (!clang::CompilerInvocation::CreateFromArgs(
          *invocation, cc1Cmd->getArguments(), diags, "clang"))
    return nullptr;

  if (invocation->getTargetOpts().Triple.empty())
    invocation->getTargetOpts().Triple = triple;

  clang::CompilerInstance compiler(invocation);
  compiler.createDiagnostics();
  compiler.setVirtualFileSystem(llvm::vfs::getRealFileSystem());
  compiler.createFileManager();
  compiler.createSourceManager();

  auto action = std::make_unique<clang::EmitLLVMOnlyAction>(&llvmCtx);
  if (!compiler.ExecuteAction(*action)) {
    llvm::errs() << "KernelCompiler: clang failed for " << srcPath << "\n";
    return nullptr;
  }

  return action->takeModule();
}

/// Remove intrinsic calls that carry metadata operands (crash the MLIR
/// importer).
void stripProblematicIntrinsics(llvm::Module &mod) {
  llvm::SmallVector<llvm::CallInst *> toErase;
  for (auto &fn : mod) {
    for (auto &bb : fn) {
      for (auto &inst : bb) {
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          if (auto *callee = call->getCalledFunction()) {
            llvm::StringRef name = callee->getName();
            if (name.starts_with("llvm.experimental.") ||
                name.starts_with("llvm.var.annotation") ||
                name.starts_with("llvm.ptr.annotation") ||
                name.starts_with("llvm.annotation") ||
                name.starts_with("llvm.assume"))
              toErase.push_back(call);
          }
        }
      }
    }
  }
  for (auto *call : toErase)
    call->eraseFromParent();
}

/// Run LLVM optimization passes to decompose aggregate types into scalars.
void runStructDecompositionPasses(llvm::Module &mod) {
  llvm::PassBuilder PB;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::FunctionPassManager FPM;
  FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  FPM.addPass(llvm::InstCombinePass());
  FPM.addPass(llvm::EarlyCSEPass());
  FPM.addPass(llvm::SimplifyCFGPass());
  FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  FPM.addPass(llvm::InstCombinePass());

  llvm::ModulePassManager MPM;
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(mod, MAM);
}

/// Run the LLVM-to-CF conversion pass pipeline.
LogicalResult runLLVMToCF(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(loom::createConvertLLVMToCFPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(module);
}

/// Run CF-to-SCF conversion pipeline.
LogicalResult runCFToSCF(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createLiftControlFlowToSCFPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(loom::createUpliftWhileToForPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(module);
}

/// Run the MarkDFGDomain + SCFToDFG pipeline.
LogicalResult runSCFToDFG(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(loom::createMarkDFGDomainPass());
  pm.addPass(loom::createConvertSCFToDFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(module);
}

/// Check if a func::FuncOp has any DFG candidate regions.
/// This is a lightweight check that mirrors the logic in MarkDFGDomain.cpp
/// without running the full pass.
bool hasDFGCandidates(func::FuncOp func) {
  if (func.isDeclaration())
    return false;
  if (func.getName() == "main")
    return false;

  // Check for unsupported ops (external function calls)
  bool hasUnsupported = false;
  func.walk([&](func::CallOp) { hasUnsupported = true; });
  if (hasUnsupported)
    return false;

  // Check for loops (required for DFG candidacy)
  bool hasLoop = false;
  func.walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::WhileOp>(op))
      hasLoop = true;
  });

  return hasLoop;
}

/// Count handshake.func operations in a module.
unsigned countHandshakeFuncs(ModuleOp module) {
  unsigned count = 0;
  module.walk([&](circt::handshake::FuncOp) { ++count; });
  return count;
}

/// Estimate PE count from a DFG module by counting compute operations.
unsigned estimatePECount(ModuleOp module) {
  unsigned opCount = 0;
  module.walk([&](Operation *op) {
    if (llvm::isa<arith::ArithDialect>(op->getDialect()) ||
        op->getName().getStringRef().starts_with("handshake."))
      ++opCount;
  });
  return std::max(1u, (opCount + 2) / 3);
}

/// Estimate memory port count from a DFG module.
unsigned estimateMemPorts(ModuleOp module) {
  unsigned portCount = 0;
  module.walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp>(op))
      ++portCount;
  });
  return portCount;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// KernelCompiler implementation
// ---------------------------------------------------------------------------

KernelCompiler::KernelCompiler(MLIRContext &ctx,
                               std::vector<std::string> includePaths)
    : ctx_(ctx), includePaths_(std::move(includePaths)),
      llvmCtx_(std::make_unique<llvm::LLVMContext>()) {
  registerDialects();

  // Initialize LLVM targets (needed for Clang)
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
}

KernelCompiler::~KernelCompiler() = default;

void KernelCompiler::registerDialects() {
  ctx_.getOrLoadDialect<LLVM::LLVMDialect>();
  ctx_.getOrLoadDialect<arith::ArithDialect>();
  ctx_.getOrLoadDialect<cf::ControlFlowDialect>();
  ctx_.getOrLoadDialect<func::FuncDialect>();
  ctx_.getOrLoadDialect<math::MathDialect>();
  ctx_.getOrLoadDialect<memref::MemRefDialect>();
  ctx_.getOrLoadDialect<scf::SCFDialect>();
  ctx_.getOrLoadDialect<DLTIDialect>();
  ctx_.getOrLoadDialect<ub::UBDialect>();
  ctx_.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx_.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx_.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

bool KernelCompiler::loadSource(const std::string &sourcePath) {
  // Check cache
  if (scfModules_.count(sourcePath))
    return true;

  // Compile C/C++ to LLVM IR via in-process Clang
  auto llvmMod = compileSourceToLLVM(sourcePath, *llvmCtx_, includePaths_);
  if (!llvmMod) {
    llvm::errs() << "KernelCompiler: failed to compile " << sourcePath << "\n";
    return false;
  }

  // Clean up problematic intrinsics before MLIR import
  stripProblematicIntrinsics(*llvmMod);

  // Decompose struct types into scalars (SRoA)
  runStructDecompositionPasses(*llvmMod);

  // Import LLVM IR into MLIR LLVM dialect
  auto mlirModule = translateLLVMIRToModule(std::move(llvmMod), &ctx_);
  if (!mlirModule) {
    llvm::errs() << "KernelCompiler: MLIR import failed for " << sourcePath
                 << "\n";
    return false;
  }

  // Run LLVM-to-CF conversion
  if (failed(runLLVMToCF(*mlirModule))) {
    llvm::errs() << "KernelCompiler: LLVM-to-CF failed for " << sourcePath
                 << "\n";
    return false;
  }

  // Run CF-to-SCF conversion
  if (failed(runCFToSCF(*mlirModule))) {
    llvm::errs() << "KernelCompiler: CF-to-SCF failed for " << sourcePath
                 << "\n";
    return false;
  }

  // Register all functions from this source
  mlirModule->walk([&](func::FuncOp func) {
    if (!func.isDeclaration()) {
      funcToSource_[func.getName().str()] = sourcePath;
    }
  });

  // Cache the SCF-stage module
  scfModules_[sourcePath] = std::move(mlirModule);
  return true;
}

KernelCompileResult KernelCompiler::compile(const std::string &functionName) {
  KernelCompileResult result;
  result.functionName = functionName;

  // Find the function in cached modules
  Operation *funcOp = findFunction(functionName);
  if (!funcOp) {
    result.diagnostics =
        "function '" + functionName + "' not found in any loaded source";
    return result;
  }

  auto func = cast<func::FuncOp>(funcOp);
  auto sourceIt = funcToSource_.find(functionName);
  if (sourceIt == funcToSource_.end()) {
    result.diagnostics =
        "internal error: function found but source path unknown";
    return result;
  }

  // Clone the function into a standalone module for independent compilation
  OpBuilder builder(&ctx_);
  auto kernelModule = ModuleOp::create(builder.getUnknownLoc());

  // Clone all function declarations (not just the target) so that
  // references to other functions (e.g., helper declarations) resolve.
  // Only clone the target function's body.
  auto &sourceModule = scfModules_[sourceIt->second];
  IRMapping mapping;

  sourceModule->walk([&](func::FuncOp srcFunc) {
    if (srcFunc.getName() == functionName) {
      // Clone the full function (with body)
      auto cloned = srcFunc.clone(mapping);
      kernelModule.push_back(cloned);
    } else if (srcFunc.isDeclaration()) {
      // Clone declarations that might be referenced
      auto cloned = srcFunc.clone(mapping);
      kernelModule.push_back(cloned);
    }
  });

  // Run the DFG conversion pipeline (MarkDFGDomain + SCFToDFG)
  if (failed(runSCFToDFG(kernelModule))) {
    result.diagnostics =
        "DFG conversion failed for function '" + functionName + "'";
    kernelModule.erase();
    return result;
  }

  // Verify that a handshake.func was produced
  unsigned hsFuncCount = countHandshakeFuncs(kernelModule);
  if (hsFuncCount == 0) {
    result.diagnostics =
        "no DFG candidate found for function '" + functionName +
        "' (function may lack loops or use unsupported operations)";
    kernelModule.erase();
    return result;
  }

  // Populate resource estimates
  result.estimatedPECount = estimatePECount(kernelModule);
  result.estimatedMemPorts = estimateMemPorts(kernelModule);

  result.success = true;
  result.dfgModule = kernelModule;
  return result;
}

bool KernelCompiler::isAccelerable(const std::string &functionName) {
  Operation *funcOp = findFunction(functionName);
  if (!funcOp)
    return false;

  auto func = dyn_cast<func::FuncOp>(funcOp);
  if (!func)
    return false;

  return hasDFGCandidates(func);
}

std::vector<std::string> KernelCompiler::listFunctions() const {
  std::vector<std::string> names;
  names.reserve(funcToSource_.size());
  for (const auto &entry : funcToSource_)
    names.push_back(entry.first);
  return names;
}

KernelDesc KernelCompiler::toKernelDesc(KernelCompileResult &result) {
  KernelDesc desc;
  desc.name = result.functionName;
  if (result.success && result.dfgModule) {
    desc.dfgModule = *result.dfgModule;
    desc.requiredPEs = result.estimatedPECount;
    desc.requiredFUs = result.estimatedPECount * 2;
    desc.requiredMemoryBytes = 0;
    result.dfgModule.release();
  }
  return desc;
}

Operation *KernelCompiler::findFunction(const std::string &functionName) const {
  auto it = funcToSource_.find(functionName);
  if (it == funcToSource_.end())
    return nullptr;

  auto modIt = scfModules_.find(it->second);
  if (modIt == scfModules_.end())
    return nullptr;

  // OwningOpRef::operator->() is non-const, so access the underlying
  // ModuleOp via get() and use it directly.
  ModuleOp mod = const_cast<OwningOpRef<ModuleOp> &>(modIt->second).get();
  Operation *found = nullptr;
  mod.walk([&](func::FuncOp func) {
    if (func.getName() == functionName)
      found = func.getOperation();
  });
  return found;
}
