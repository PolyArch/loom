#include "loom_pipeline.h"
#include "loom/Conversion/HostCodeGen.h"
#include "loom/Conversion/Passes.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/Passes.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

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
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

using namespace mlir;

namespace loom {

static llvm::StringRef detectClangInputLanguage(const std::string &srcPath) {
  llvm::StringRef ext = llvm::sys::path::extension(srcPath);
  if (ext.equals_insensitive(".cc") || ext.equals_insensitive(".cpp") ||
      ext.equals_insensitive(".cxx") || ext.equals_insensitive(".c++") ||
      ext == ".C")
    return "c++";
  return "c";
}

// Compile a single C/C++ source to LLVM Module using in-process clang.
static std::unique_ptr<llvm::Module>
compileOneSource(const std::string &srcPath, llvm::LLVMContext &llvmCtx,
                 const LoomArgs &args) {
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
  // Allow math functions to be converted to LLVM intrinsics (no errno)
  driverArgs.push_back("-fno-math-errno");
  // Preserve source locations as line-table debug info
  driverArgs.push_back("-gline-tables-only");
  // Use built-in headers from project LLVM
  driverArgs.push_back("-resource-dir");
  driverArgs.push_back(LOOM_CLANG_RESOURCE_DIR);

  llvm::SmallVector<std::string> incFlags;
  for (const auto &inc : args.includePaths)
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
    llvm::errs() << "loom: no cc1 job found\n";
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
    llvm::errs() << "loom: clang failed for " << srcPath << "\n";
    return nullptr;
  }

  return action->takeModule();
}

// Remove experimental/annotation intrinsic calls that carry metadata
// operands (which crash the MLIR importer).
static void stripProblematicIntrinsics(llvm::Module &mod) {
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

// Run LLVM optimization passes to decompose aggregate types into scalars.
// SRoA (Scalar Replacement of Aggregates) turns struct alloca/GEP patterns
// into individual scalar variables, which avoids struct-typed memrefs in MLIR.
static void runStructDecompositionPasses(llvm::Module &mod) {
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
  // SRoA decomposes struct allocas into individual scalar SSA values
  FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  // InstCombine cleans up after SRoA (fold redundant ops, simplify GEPs)
  FPM.addPass(llvm::InstCombinePass());
  // EarlyCSE eliminates trivially redundant instructions
  FPM.addPass(llvm::EarlyCSEPass());
  // SimplifyCFG merges/cleans up basic blocks
  FPM.addPass(llvm::SimplifyCFGPass());
  // Run SRoA again to catch anything exposed by previous passes
  FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  FPM.addPass(llvm::InstCombinePass());

  llvm::ModulePassManager MPM;
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(mod, MAM);
}

OwningOpRef<ModuleOp> compileAndImport(const LoomArgs &args,
                                        MLIRContext &ctx,
                                        const std::string &llOutputPath) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> linkedModule;

  for (const auto &src : args.sources) {
    auto mod = compileOneSource(src, llvmCtx, args);
    if (!mod)
      return nullptr;

    if (!linkedModule) {
      linkedModule = std::move(mod);
    } else {
      if (llvm::Linker::linkModules(*linkedModule, std::move(mod))) {
        llvm::errs() << "loom: failed to link modules\n";
        return nullptr;
      }
    }
  }

  // Clean up intrinsics that carry metadata operands (crash MLIR importer)
  stripProblematicIntrinsics(*linkedModule);

  // Run SRoA and supporting passes to decompose struct types into scalars.
  // This must happen before MLIR import because MLIR memref cannot hold
  // struct-typed elements.
  runStructDecompositionPasses(*linkedModule);

  // Write LLVM IR to file for inspection (after optimization)
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(llOutputPath, ec);
    if (!ec)
      linkedModule->print(os, nullptr);
  }

  // Import to MLIR LLVM dialect.
  // Source locations from -gline-tables-only become MLIR FileLineColLoc.
  auto mlirModule = translateLLVMIRToModule(std::move(linkedModule), &ctx);
  if (!mlirModule) {
    llvm::errs() << "loom: MLIR import failed\n";
    return nullptr;
  }

  return mlirModule;
}

LogicalResult runLLVMToCF(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(loom::createConvertLLVMToCFPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(module);
}

LogicalResult runCFToSCF(ModuleOp module) {
  PassManager pm(module.getContext());
  // Lift CF (cond_br/br) to SCF (if/while)
  pm.addPass(createLiftControlFlowToSCFPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Uplift scf.while to scf.for where induction pattern is detected
  pm.addPass(loom::createUpliftWhileToForPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(module);
}

LogicalResult runSCFToDFG(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(loom::createMarkDFGDomainPass());
  pm.addPass(loom::createConvertSCFToDFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(module);
}

LogicalResult runHostCodeGen(ModuleOp module, const std::string &outputPath,
                             const std::string &originalSource) {
  PassManager pm(module.getContext());
  pm.addPass(loom::createHostCodeGenPass(outputPath, originalSource));
  return pm.run(module);
}

LogicalResult writeMLIR(ModuleOp module, const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "loom: cannot open " << path << ": " << ec.message()
                 << "\n";
    return failure();
  }
  // Print with source locations preserved (from -gline-tables-only)
  OpPrintingFlags flags;
  flags.enableDebugInfo(/*pretty=*/true);
  module.print(os, flags);
  return success();
}

} // namespace loom
