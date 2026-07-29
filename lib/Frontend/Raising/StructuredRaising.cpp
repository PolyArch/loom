#include "Frontend/Raising/StructuredRaising.h"

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#include <memory>

namespace loom::raising {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_raising_invalid: " + message);
}

} // namespace

llvm::Error normalizeProvenConstantCallbacks(llvm::Module &module) {
  if (llvm::verifyModule(module))
    return invalid("LLVM module failed verification");

  llvm::SmallVector<llvm::CallBase *, 8> indirectCalls;
  for (llvm::Function &function : module)
    for (llvm::Instruction &instruction : llvm::instructions(function))
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
          call && !call->isInlineAsm() && !call->getCalledFunction())
        indirectCalls.push_back(call);
  if (indirectCalls.empty())
    return llvm::Error::success();

  llvm::ValueToValueMapTy mapping;
  std::unique_ptr<llvm::Module> probe = llvm::CloneModule(module, mapping);
  constexpr llvm::StringLiteral probeMetadata = "loom.constant_callback_probe";
  for (llvm::Function &function : *probe)
    for (llvm::Instruction &instruction : llvm::instructions(function))
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
        call->setMetadata(probeMetadata, nullptr);
  for (auto [ordinal, call] : llvm::enumerate(indirectCalls)) {
    auto *cloned = llvm::dyn_cast_or_null<llvm::CallBase>(mapping.lookup(call));
    if (!cloned)
      return invalid("constant-callback probe lost an indirect call");
    llvm::Constant *value = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(module.getContext()), ordinal);
    cloned->setMetadata(
        probeMetadata, llvm::MDNode::get(module.getContext(),
                                         llvm::ConstantAsMetadata::get(value)));
  }

  llvm::LoopAnalysisManager loopAnalyses;
  llvm::FunctionAnalysisManager functionAnalyses;
  llvm::CGSCCAnalysisManager cgsccAnalyses;
  llvm::ModuleAnalysisManager moduleAnalyses;
  llvm::PassBuilder builder;
  builder.registerModuleAnalyses(moduleAnalyses);
  builder.registerCGSCCAnalyses(cgsccAnalyses);
  builder.registerFunctionAnalyses(functionAnalyses);
  builder.registerLoopAnalyses(loopAnalyses);
  builder.crossRegisterProxies(loopAnalyses, functionAnalyses, cgsccAnalyses,
                               moduleAnalyses);

  llvm::ModulePassManager pipeline;
  pipeline.addPass(llvm::IPSCCPPass(llvm::IPSCCPOptions().setFuncSpec(false)));
  pipeline.run(*probe, moduleAnalyses);

  for (llvm::Function &function : *probe) {
    for (llvm::Instruction &instruction : llvm::instructions(function)) {
      auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (!call)
        continue;
      llvm::MDNode *tag = call->getMetadata(probeMetadata);
      if (!tag || tag->getNumOperands() != 1)
        continue;
      auto *metadata =
          llvm::dyn_cast<llvm::ConstantAsMetadata>(tag->getOperand(0));
      auto *ordinal =
          metadata ? llvm::dyn_cast<llvm::ConstantInt>(metadata->getValue())
                   : nullptr;
      auto *target = llvm::dyn_cast<llvm::Function>(
          call->getCalledOperand()->stripPointerCasts());
      if (!ordinal || !target ||
          ordinal->getZExtValue() >= indirectCalls.size())
        continue;
      llvm::CallBase *original = indirectCalls[ordinal->getZExtValue()];
      llvm::Function *originalTarget = module.getFunction(target->getName());
      if (!originalTarget ||
          original->getFunctionType() != originalTarget->getFunctionType())
        continue;
      original->setCalledOperand(originalTarget);
    }
  }
  if (llvm::verifyModule(module))
    return invalid("constant-callback normalization produced invalid LLVM IR");
  return llvm::Error::success();
}

namespace {

void normalizeBulkMemoryIntrinsics(llvm::Module &module) {
  llvm::TargetTransformInfo target(module.getDataLayout());
  llvm::SmallVector<llvm::MemIntrinsic *, 16> intrinsics;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block)
        if (auto *intrinsic = llvm::dyn_cast<llvm::MemIntrinsic>(&instruction))
          intrinsics.push_back(intrinsic);

  for (llvm::MemIntrinsic *intrinsic : intrinsics) {
    if (auto *copy = llvm::dyn_cast<llvm::MemCpyInst>(intrinsic)) {
      llvm::expandMemCpyAsLoop(copy, target);
      copy->eraseFromParent();
      continue;
    }
    if (auto *move = llvm::dyn_cast<llvm::MemMoveInst>(intrinsic)) {
      if (llvm::expandMemMoveAsLoop(move, target))
        move->eraseFromParent();
      continue;
    }
    if (auto *set = llvm::dyn_cast<llvm::MemSetInst>(intrinsic)) {
      llvm::expandMemSetAsLoop(set, nullptr);
      set->eraseFromParent();
    }
  }
}

} // namespace

llvm::Expected<frontend::StructuredProgramCandidate>
raiseLlvmModuleToStructuredProgram(std::unique_ptr<llvm::Module> module,
                                   StructuredRaisingOptions options) {
  if (!module)
    return invalid("missing LLVM module");
  if (llvm::Error error = normalizeProvenConstantCallbacks(*module))
    return std::move(error);
  if (llvm::Error error = specializeExactConstantCallbackCallSites(*module))
    return std::move(error);

  normalizeBulkMemoryIntrinsics(*module);
  if (llvm::verifyModule(*module))
    return invalid("bulk-memory normalization produced invalid LLVM IR");
  const std::string llvmDataLayout = module->getDataLayoutStr();
  if (llvmDataLayout.empty())
    return invalid("LLVM module has no DataLayout");
  if (auto parsed = llvm::DataLayout::parse(llvmDataLayout); !parsed)
    return invalid("LLVM module has an invalid DataLayout: " +
                   llvm::toString(parsed.takeError()));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllFromLLVMIRTranslations(registry);
  mlir::registerAllToLLVMIRTranslations(registry);

  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.allowUnregisteredDialects(options.allowUnregisteredDialects);
  context.loadAllAvailableDialects();
  context.loadDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                      mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                      mlir::math::MathDialect, mlir::memref::MemRefDialect,
                      mlir::scf::SCFDialect, mlir::ub::UBDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> raised =
      mlir::translateLLVMIRToModule(std::move(module), &context);
  if (!raised)
    return invalid("LLVM IR import failed");
  raised->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, llvmDataLayout));

  static const bool passesRegistered = [] {
    mlir::registerAllPasses();
    registerRaisingPasses();
    return true;
  }();
  (void)passesRegistered;
  mlir::PassManager pipeline(&context);
  pipeline.enableVerifier(options.verifyEach);
  if (options.applyPassManagerCommandLineOptions &&
      failed(mlir::applyPassManagerCLOptions(pipeline)))
    return invalid("cannot apply pass-manager command-line options");
  buildRaisingPipeline(pipeline);
  if (failed(pipeline.run(*raised)))
    return invalid("mechanical LLVM-to-SCF raising failed");

  return frontend::finalizeStructuredProgram(raised.get());
}

} // namespace loom::raising
