//===-- auto_analyze.cpp - Automatic TDG construction from C source --------===//
//
// Implements tapestry::autoAnalyze(): given a C/C++ source file and entry
// function name, compiles to LLVM IR, analyzes the call graph and data
// dependencies at the LLVM IR level, then uses the Loom SCF pipeline for
// per-function accelerability checking.
//
//===----------------------------------------------------------------------===//

#include "tapestry/auto_analyze.h"

#include "loom/Conversion/Passes.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
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
#include "mlir/IR/MLIRContext.h"
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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

#define DEBUG_TYPE "tapestry-auto-analyze"

namespace tapestry {

//===----------------------------------------------------------------------===//
// Internal data types
//===----------------------------------------------------------------------===//

namespace {

/// Describes a single function call observed in the entry function body,
/// analyzed at the LLVM IR level.
struct LLVMCallSite {
  std::string calleeName;
  unsigned callOrder = 0;
  std::vector<std::string> argNames;
  std::vector<bool> argIsPointer;

  /// For pointer arguments, the LLVM Value* used at the call site.
  /// Shared pointers between calls indicate data dependencies.
  std::vector<llvm::Value *> argValues;
};

/// Per-function analysis result.
struct FunctionAnalysis {
  std::string funcName;
  bool isDefined = false;       // Has a body in this module
  bool hasExternalCalls = false; // Calls functions not in this module
  bool hasLoops = false;
  bool hasMemoryAccess = false;
  unsigned estimatedOps = 0;

  /// For each parameter: READ_ONLY, WRITE_ONLY, READ_WRITE, or NONE.
  enum ParamAccess { NONE = 0, READ_ONLY, WRITE_ONLY, READ_WRITE };
  std::vector<ParamAccess> paramAccess;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Source compilation: C/C++ -> LLVM IR
//===----------------------------------------------------------------------===//

static llvm::StringRef detectInputLanguage(const std::string &srcPath) {
  llvm::StringRef ext = llvm::sys::path::extension(srcPath);
  if (ext.equals_insensitive(".cc") || ext.equals_insensitive(".cpp") ||
      ext.equals_insensitive(".cxx") || ext.equals_insensitive(".c++") ||
      ext == ".C")
    return "c++";
  return "c";
}

static std::unique_ptr<llvm::Module>
compileSourceToLLVM(const std::string &srcPath, llvm::LLVMContext &llvmCtx,
                    bool verbose) {
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
  llvm::StringRef inputLang = detectInputLanguage(srcPath);
  driverArgs.push_back(inputLang.data());
  driverArgs.push_back("-O1");
  driverArgs.push_back("-c");
  driverArgs.push_back("-emit-llvm");
  driverArgs.push_back("-fno-discard-value-names");
  driverArgs.push_back("-fno-math-errno");
  driverArgs.push_back("-gline-tables-only");
  driverArgs.push_back("-resource-dir");
  driverArgs.push_back(LOOM_CLANG_RESOURCE_DIR);
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
    llvm::errs() << "auto-analyze: no cc1 job found\n";
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
    llvm::errs() << "auto-analyze: clang failed for " << srcPath << "\n";
    return nullptr;
  }

  return action->takeModule();
}

//===----------------------------------------------------------------------===//
// LLVM IR analysis: call graph, parameter access, accelerability
//===----------------------------------------------------------------------===//

/// Get a human-readable name for an LLVM value.
static std::string getLLVMValueName(llvm::Value *val) {
  if (val->hasName())
    return val->getName().str();
  return "anon";
}

/// Infer data type string from an LLVM pointer element type.
static std::string inferDataTypeFromLLVM(llvm::Type *pointeeTy) {
  if (!pointeeTy)
    return "unknown";
  if (pointeeTy->isFloatTy())
    return "f32";
  if (pointeeTy->isDoubleTy())
    return "f64";
  if (pointeeTy->isHalfTy())
    return "f16";
  if (auto *intTy = llvm::dyn_cast<llvm::IntegerType>(pointeeTy))
    return "i" + std::to_string(intTy->getBitWidth());
  return "unknown";
}

/// Collect all function call sites from the entry function at the LLVM IR level.
static void collectLLVMCallSites(llvm::Function &entryFunc,
                                 const llvm::Module &mod,
                                 std::vector<LLVMCallSite> &callSites) {
  unsigned order = 0;
  for (auto &bb : entryFunc) {
    for (auto &inst : bb) {
      auto *callInst = llvm::dyn_cast<llvm::CallInst>(&inst);
      if (!callInst)
        continue;

      llvm::Function *callee = callInst->getCalledFunction();
      if (!callee)
        continue; // Indirect call: skip

      llvm::StringRef name = callee->getName();
      // Skip LLVM intrinsics and debug calls
      if (name.starts_with("llvm."))
        continue;

      LLVMCallSite site;
      site.calleeName = name.str();
      site.callOrder = order++;

      for (unsigned idx = 0; idx < callInst->arg_size(); ++idx) {
        llvm::Value *arg = callInst->getArgOperand(idx);
        site.argNames.push_back(getLLVMValueName(arg));
        site.argIsPointer.push_back(arg->getType()->isPointerTy());
        site.argValues.push_back(arg);
      }

      callSites.push_back(std::move(site));
    }
  }
}

/// Analyze a function at the LLVM IR level for accelerability.
static FunctionAnalysis analyzeLLVMFunction(llvm::Function &func,
                                            const llvm::Module &mod) {
  FunctionAnalysis result;
  result.funcName = func.getName().str();
  result.isDefined = !func.isDeclaration();

  if (!result.isDefined) {
    result.hasExternalCalls = true;
    return result;
  }

  // Initialize parameter access tracking
  unsigned numArgs = func.arg_size();
  result.paramAccess.resize(numArgs, FunctionAnalysis::NONE);

  // Map function arguments to indices (for pointer arguments)
  llvm::DenseMap<llvm::Value *, unsigned> argToIndex;
  for (unsigned idx = 0; idx < numArgs; ++idx) {
    llvm::Argument *arg = func.getArg(idx);
    if (arg->getType()->isPointerTy())
      argToIndex[arg] = idx;
  }

  for (auto &bb : func) {
    // Check for back-edges (loops): a block that has a predecessor that
    // dominates it or a successor that it dominates is a loop indicator.
    // Simple heuristic: any branch to a previously-visited block.
    for (auto &inst : bb) {
      // Check for function calls
      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&inst)) {
        if (llvm::Function *callee = callInst->getCalledFunction()) {
          if (!callee->getName().starts_with("llvm.")) {
            result.hasExternalCalls = true;
          }
        }
      }

      // Check for loads (memory read)
      if (auto *loadInst = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
        result.hasMemoryAccess = true;
        llvm::Value *ptr = loadInst->getPointerOperand();
        // Trace back to function argument
        while (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(ptr))
          ptr = gep->getPointerOperand();
        while (auto *bc = llvm::dyn_cast<llvm::BitCastInst>(ptr))
          ptr = bc->getOperand(0);

        auto it = argToIndex.find(ptr);
        if (it != argToIndex.end()) {
          unsigned idx = it->second;
          if (result.paramAccess[idx] == FunctionAnalysis::NONE)
            result.paramAccess[idx] = FunctionAnalysis::READ_ONLY;
          else if (result.paramAccess[idx] == FunctionAnalysis::WRITE_ONLY)
            result.paramAccess[idx] = FunctionAnalysis::READ_WRITE;
        }
      }

      // Check for stores (memory write)
      if (auto *storeInst = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
        result.hasMemoryAccess = true;
        llvm::Value *ptr = storeInst->getPointerOperand();
        // Trace back to function argument
        while (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(ptr))
          ptr = gep->getPointerOperand();
        while (auto *bc = llvm::dyn_cast<llvm::BitCastInst>(ptr))
          ptr = bc->getOperand(0);

        auto it = argToIndex.find(ptr);
        if (it != argToIndex.end()) {
          unsigned idx = it->second;
          if (result.paramAccess[idx] == FunctionAnalysis::NONE)
            result.paramAccess[idx] = FunctionAnalysis::WRITE_ONLY;
          else if (result.paramAccess[idx] == FunctionAnalysis::READ_ONLY)
            result.paramAccess[idx] = FunctionAnalysis::READ_WRITE;
        }
      }

      // Count arithmetic ops
      if (llvm::isa<llvm::BinaryOperator>(&inst))
        result.estimatedOps++;
      if (llvm::isa<llvm::ICmpInst, llvm::FCmpInst>(&inst))
        result.estimatedOps++;

      // Loop detection: branch back-edge (targets an earlier block)
      if (auto *brInst = llvm::dyn_cast<llvm::BranchInst>(&inst)) {
        for (unsigned sIdx = 0; sIdx < brInst->getNumSuccessors(); ++sIdx) {
          // If successor appears before this block in layout order, it is
          // likely a loop back-edge.
          llvm::BasicBlock *succ = brInst->getSuccessor(sIdx);
          if (&bb != &func.getEntryBlock()) {
            // Heuristic: if we branch to a block that is the same as any
            // block we already walked through, it is a loop.
            bool isBackEdge = false;
            for (auto &prevBB : func) {
              if (&prevBB == succ) {
                isBackEdge = true;
                break;
              }
              if (&prevBB == &bb)
                break;
            }
            if (isBackEdge)
              result.hasLoops = true;
          }
        }
      }
    }
  }

  return result;
}

/// Check if a function is CGRA-accelerable.
static bool isAccelerable(const FunctionAnalysis &analysis) {
  if (!analysis.isDefined)
    return false;
  if (analysis.hasExternalCalls)
    return false;
  if (!analysis.hasLoops)
    return false;
  if (!analysis.hasMemoryAccess)
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Data dependency analysis (at LLVM IR level)
//===----------------------------------------------------------------------===//

/// Trace an LLVM value through GEP/bitcast chains to find the base pointer.
static llvm::Value *traceToBase(llvm::Value *val) {
  while (true) {
    if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(val)) {
      val = gep->getPointerOperand();
      continue;
    }
    if (auto *bc = llvm::dyn_cast<llvm::BitCastInst>(val)) {
      val = bc->getOperand(0);
      continue;
    }
    break;
  }
  return val;
}

/// Analyze data dependency between two call sites at the LLVM IR level.
static DataDependency analyzeDataDependency(
    const LLVMCallSite &callA, const FunctionAnalysis &analysisA,
    const LLVMCallSite &callB, const FunctionAnalysis &analysisB) {

  DataDependency dep;

  // Check directed dependency: callA writes, callB reads
  for (unsigned aIdx = 0; aIdx < callA.argValues.size(); ++aIdx) {
    if (!callA.argIsPointer[aIdx])
      continue;

    // Check if function A writes to this parameter
    if (aIdx < analysisA.paramAccess.size()) {
      auto accessA = analysisA.paramAccess[aIdx];
      if (accessA != FunctionAnalysis::WRITE_ONLY &&
          accessA != FunctionAnalysis::READ_WRITE)
        continue;
    }

    llvm::Value *baseA = traceToBase(callA.argValues[aIdx]);

    for (unsigned bIdx = 0; bIdx < callB.argValues.size(); ++bIdx) {
      if (!callB.argIsPointer[bIdx])
        continue;

      // Check if function B reads from this parameter
      if (bIdx < analysisB.paramAccess.size()) {
        auto accessB = analysisB.paramAccess[bIdx];
        if (accessB != FunctionAnalysis::READ_ONLY &&
            accessB != FunctionAnalysis::READ_WRITE)
          continue;
      }

      llvm::Value *baseB = traceToBase(callB.argValues[bIdx]);

      if (baseA == baseB) {
        dep.exists = true;
        dep.isSequential = true;
        dep.sharedArgName = callA.argNames[aIdx];

        // Try to infer element type from GEP source element type
        if (auto *gep =
                llvm::dyn_cast<llvm::GetElementPtrInst>(callA.argValues[aIdx]))
          dep.dataType = inferDataTypeFromLLVM(gep->getSourceElementType());
        else
          dep.dataType = "unknown";
        return dep;
      }
    }
  }

  // Conservative fallback: check if ANY pointer args share the same base
  for (unsigned aIdx = 0; aIdx < callA.argValues.size(); ++aIdx) {
    if (!callA.argIsPointer[aIdx])
      continue;
    llvm::Value *baseA = traceToBase(callA.argValues[aIdx]);

    for (unsigned bIdx = 0; bIdx < callB.argValues.size(); ++bIdx) {
      if (!callB.argIsPointer[bIdx])
        continue;
      llvm::Value *baseB = traceToBase(callB.argValues[bIdx]);

      if (baseA == baseB) {
        dep.exists = true;
        dep.isSequential = true;
        dep.sharedArgName = callA.argNames[aIdx];
        dep.dataType = "unknown";
        return dep;
      }
    }
  }

  return dep;
}

//===----------------------------------------------------------------------===//
// Result construction
//===----------------------------------------------------------------------===//

void AutoAnalyzeResult::dump(llvm::raw_ostream &os) const {
  os << "AutoAnalyzeResult:\n";
  os << "  source: " << sourcePath << "\n";
  os << "  entry:  " << entryFunc << "\n";
  os << "  status: " << (success ? "success" : "failed") << "\n";

  if (!diagnostics.empty())
    os << "  diagnostics: " << diagnostics << "\n";

  os << "  kernels: " << numKernels() << "\n";
  for (const auto &binding : callBindings) {
    os << "    [" << binding.callOrder << "] " << binding.kernelName;
    if (binding.target == KernelTarget::HOST)
      os << " (HOST)";
    else if (binding.target == KernelTarget::CGRA)
      os << " (CGRA)";
    os << " args=(";
    for (unsigned idx = 0; idx < binding.argNames.size(); ++idx) {
      if (idx > 0)
        os << ", ";
      os << binding.argNames[idx];
    }
    os << ")\n";
  }

  os << "  edges: " << numEdges() << "\n";
  for (const auto &edge : edges) {
    os << "    " << callBindings[edge.producerIndex].kernelName << " -> "
       << callBindings[edge.consumerIndex].kernelName;
    os << " [type=" << edge.dependency.dataType;
    if (edge.dependency.elementCount.has_value())
      os << ", count=" << edge.dependency.elementCount.value();
    os << ", ordering="
       << (edge.ordering == loom::Ordering::FIFO ? "FIFO" : "UNORDERED");
    if (!edge.dependency.sharedArgName.empty())
      os << ", via=" << edge.dependency.sharedArgName;
    os << "]\n";
  }
}

void AutoAnalyzeResult::dump() const { dump(llvm::outs()); }

//===----------------------------------------------------------------------===//
// Main entry point: autoAnalyze()
//===----------------------------------------------------------------------===//

AutoAnalyzeResult autoAnalyze(const std::string &sourcePath,
                              const std::string &entryFunc,
                              const AutoAnalyzeOptions &opts) {
  AutoAnalyzeResult result;
  result.sourcePath = sourcePath;
  result.entryFunc = entryFunc;

  if (opts.verbose)
    llvm::outs() << "auto-analyze: compiling " << sourcePath
                 << " to LLVM IR\n";

  // Compile C/C++ source to LLVM IR
  llvm::LLVMContext llvmCtx;
  auto llvmModule = compileSourceToLLVM(sourcePath, llvmCtx, opts.verbose);
  if (!llvmModule) {
    result.diagnostics = "Failed to compile source to LLVM IR";
    return result;
  }

  // Locate the entry function in the LLVM module
  llvm::Function *entryFn = llvmModule->getFunction(entryFunc);
  if (!entryFn || entryFn->isDeclaration()) {
    result.diagnostics =
        "Entry function '" + entryFunc + "' not found or has no body";
    return result;
  }

  if (opts.verbose)
    llvm::outs() << "auto-analyze: found entry function '" << entryFunc
                 << "'\n";

  // Collect all function call sites in the entry function (at LLVM IR level)
  std::vector<LLVMCallSite> callSites;
  collectLLVMCallSites(*entryFn, *llvmModule, callSites);

  if (callSites.empty()) {
    result.diagnostics =
        "No function calls found in entry function '" + entryFunc +
        "' (functions may have been inlined at -O1; "
        "use __attribute__((noinline)) to preserve call boundaries)";
    return result;
  }

  if (opts.verbose)
    llvm::outs() << "auto-analyze: found " << callSites.size()
                 << " call site(s)\n";

  // Enforce max kernels limit
  if (callSites.size() > opts.maxKernels) {
    result.diagnostics =
        "Too many call sites (" + std::to_string(callSites.size()) +
        "), limit is " + std::to_string(opts.maxKernels);
    return result;
  }

  // Analyze each called function at LLVM IR level
  llvm::StringMap<FunctionAnalysis> funcAnalyses;
  for (const auto &site : callSites) {
    if (funcAnalyses.count(site.calleeName))
      continue;

    llvm::Function *callee = llvmModule->getFunction(site.calleeName);
    if (callee) {
      funcAnalyses[site.calleeName] =
          analyzeLLVMFunction(*callee, *llvmModule);
    } else {
      FunctionAnalysis ext;
      ext.funcName = site.calleeName;
      ext.hasExternalCalls = true;
      funcAnalyses[site.calleeName] = std::move(ext);
    }
  }

  // Build call bindings and determine targets
  for (const auto &site : callSites) {
    CallSiteBinding binding;
    binding.kernelName = site.calleeName;
    binding.argNames = site.argNames;
    binding.callOrder = site.callOrder;

    auto it = funcAnalyses.find(site.calleeName);
    if (it != funcAnalyses.end() && isAccelerable(it->second)) {
      binding.target = KernelTarget::CGRA;
    } else {
      binding.target = KernelTarget::HOST;
    }

    if (opts.verbose) {
      llvm::outs() << "auto-analyze: kernel '" << binding.kernelName
                   << "' -> "
                   << (binding.target == KernelTarget::CGRA ? "CGRA" : "HOST")
                   << "\n";
    }

    result.callBindings.push_back(std::move(binding));
  }

  // Analyze pairwise data dependencies between kernel call sites
  for (unsigned srcIdx = 0; srcIdx < callSites.size(); ++srcIdx) {
    for (unsigned dstIdx = srcIdx + 1; dstIdx < callSites.size(); ++dstIdx) {
      const auto &siteA = callSites[srcIdx];
      const auto &siteB = callSites[dstIdx];

      auto itA = funcAnalyses.find(siteA.calleeName);
      auto itB = funcAnalyses.find(siteB.calleeName);

      FunctionAnalysis dummyA, dummyB;
      dummyA.funcName = siteA.calleeName;
      dummyB.funcName = siteB.calleeName;

      const FunctionAnalysis &analysisA =
          (itA != funcAnalyses.end()) ? itA->second : dummyA;
      const FunctionAnalysis &analysisB =
          (itB != funcAnalyses.end()) ? itB->second : dummyB;

      DataDependency dep =
          analyzeDataDependency(siteA, analysisA, siteB, analysisB);

      if (dep.exists) {
        InferredEdge edge;
        edge.producerIndex = srcIdx;
        edge.consumerIndex = dstIdx;
        edge.dependency = std::move(dep);
        edge.ordering = edge.dependency.isSequential
                            ? loom::Ordering::FIFO
                            : loom::Ordering::UNORDERED;
        result.edges.push_back(std::move(edge));

        if (opts.verbose) {
          llvm::outs()
              << "auto-analyze: edge " << siteA.calleeName << " -> "
              << siteB.calleeName << " (shared: "
              << result.edges.back().dependency.sharedArgName << ")\n";
        }
      }
    }
  }

  // Log HOST node warnings for edges crossing host/CGRA boundary
  for (auto &edge : result.edges) {
    bool producerIsHost =
        result.callBindings[edge.producerIndex].target == KernelTarget::HOST;
    bool consumerIsHost =
        result.callBindings[edge.consumerIndex].target == KernelTarget::HOST;

    if ((producerIsHost || consumerIsHost) && opts.verbose) {
      llvm::outs() << "auto-analyze: edge "
                   << result.callBindings[edge.producerIndex].kernelName
                   << " -> "
                   << result.callBindings[edge.consumerIndex].kernelName
                   << " crosses host/CGRA boundary (EXTERNAL_DRAM)\n";
    }
  }

  result.success = true;

  if (opts.verbose) {
    llvm::outs() << "\nauto-analyze: summary\n";
    result.dump(llvm::outs());
  }

  return result;
}

} // namespace tapestry
