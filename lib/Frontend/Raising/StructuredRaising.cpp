#include "Frontend/Raising/StructuredRaising.h"

#include "Frontend/Raising/CandidateHints.h"
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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
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
#include <string>
#include <utility>
#include <vector>

namespace loom::raising {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_raising_invalid: " + message);
}

struct LlvmFunctionCandidateHint final {
  std::string symbol;
  FunctionCandidateAnnotation source;
};

llvm::GlobalVariable *referencedGlobal(llvm::Value *value) {
  return llvm::dyn_cast<llvm::GlobalVariable>(value->stripPointerCasts());
}

llvm::Expected<llvm::StringRef> annotationString(llvm::Value *value) {
  llvm::GlobalVariable *global = referencedGlobal(value);
  if (!global || !global->hasInitializer())
    return invalid("candidate annotation does not reference a string global");
  auto *data =
      llvm::dyn_cast<llvm::ConstantDataSequential>(global->getInitializer());
  if (!data || !data->isCString())
    return invalid("candidate annotation is not a C string");
  return data->getAsCString();
}

llvm::Error removeProjectedAnnotationEntries(
    llvm::Module &module, llvm::GlobalVariable *annotations,
    llvm::ArrayRef<llvm::Constant *> retained,
    llvm::ArrayRef<llvm::GlobalVariable *> projectedStrings) {
  if (!retained.empty()) {
    auto *entryType =
        llvm::dyn_cast<llvm::StructType>(retained.front()->getType());
    if (!entryType)
      return invalid("retained global annotation has a non-struct type");
    auto *arrayType = llvm::ArrayType::get(entryType, retained.size());
    llvm::Constant *initializer = llvm::ConstantArray::get(arrayType, retained);
    auto *replacement = new llvm::GlobalVariable(
        module, arrayType, annotations->isConstant(), annotations->getLinkage(),
        initializer, "", annotations, annotations->getThreadLocalMode(),
        annotations->getAddressSpace(), annotations->isExternallyInitialized());
    replacement->copyAttributesFrom(annotations);
    replacement->takeName(annotations);
    annotations->replaceAllUsesWith(replacement);
  } else if (!annotations->use_empty()) {
    return invalid("projected global annotations have unexpected users");
  }
  annotations->eraseFromParent();

  llvm::SmallPtrSet<llvm::GlobalVariable *, 8> visited;
  for (llvm::GlobalVariable *string : projectedStrings) {
    if (!visited.insert(string).second)
      continue;
    string->removeDeadConstantUsers();
    if (string->use_empty() && string->hasLocalLinkage())
      string->eraseFromParent();
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<LlvmFunctionCandidateHint>>
extractFunctionCandidateHints(llvm::Module &module) {
  llvm::GlobalVariable *annotations =
      module.getNamedGlobal("llvm.global.annotations");
  if (!annotations)
    return std::vector<LlvmFunctionCandidateHint>{};
  if (!annotations->hasInitializer())
    return invalid("llvm.global.annotations has no initializer");
  auto *initializer =
      llvm::dyn_cast<llvm::Constant>(annotations->getInitializer());
  if (!initializer)
    return invalid("llvm.global.annotations has a non-constant initializer");

  std::vector<LlvmFunctionCandidateHint> result;
  llvm::SmallVector<llvm::Constant *> retained;
  llvm::SmallVector<llvm::GlobalVariable *> projectedStrings;
  llvm::StringSet<> selectedFunctions;
  for (llvm::Value *operand : initializer->operand_values()) {
    auto *entry = llvm::dyn_cast<llvm::ConstantStruct>(operand);
    if (!entry || entry->getNumOperands() < 2) {
      auto *constant = llvm::dyn_cast<llvm::Constant>(operand);
      if (!constant)
        return invalid("global annotation entry is not constant");
      retained.push_back(constant);
      continue;
    }

    auto encoded = annotationString(entry->getOperand(1));
    if (!encoded) {
      llvm::consumeError(encoded.takeError());
      retained.push_back(entry);
      continue;
    }
    if (!encoded->starts_with("loom.candidate.")) {
      retained.push_back(entry);
      continue;
    }
    if (entry->getNumOperands() != 5)
      return invalid("candidate annotation entry does not have five fields");
    auto source = decodeFunctionCandidateAnnotation(*encoded);
    if (!source)
      return source.takeError();
    auto *function = llvm::dyn_cast<llvm::Function>(
        entry->getOperand(0)->stripPointerCasts());
    if (!function || function->isDeclaration())
      return invalid("candidate annotation does not target a function "
                     "definition");
    if (!selectedFunctions.insert(function->getName()).second)
      return invalid("function has duplicate candidate annotations");

    auto file = annotationString(entry->getOperand(2));
    auto *line = llvm::dyn_cast<llvm::ConstantInt>(entry->getOperand(3));
    if (!file || !line || !line->getType()->isIntegerTy(32) ||
        *file != source->sourceFile ||
        line->getZExtValue() != source->targetBegin.line) {
      if (!file)
        llvm::consumeError(file.takeError());
      return invalid("candidate annotation source anchor disagrees with its "
                     "payload");
    }
    result.push_back({function->getName().str(), std::move(*source)});
    projectedStrings.push_back(referencedGlobal(entry->getOperand(1)));
    projectedStrings.push_back(referencedGlobal(entry->getOperand(2)));
  }

  if (result.empty())
    return result;
  if (llvm::Error error = removeProjectedAnnotationEntries(
          module, annotations, retained, projectedStrings))
    return std::move(error);
  return result;
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

llvm::Expected<frontend::FinalizedStructuredProgramProjection>
raiseLlvmModuleToStructuredProgramWithProjection(
    std::unique_ptr<llvm::Module> module, StructuredRaisingOptions options) {
  if (!module)
    return invalid("missing LLVM module");
  if (llvm::Error error = normalizeProvenConstantCallbacks(*module))
    return std::move(error);
  if (llvm::Error error = specializeExactConstantCallbackCallSites(*module))
    return std::move(error);
  auto candidateHints = extractFunctionCandidateHints(*module);
  if (!candidateHints)
    return candidateHints.takeError();

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

  llvm::SmallVector<mlir::Operation *> trackedCandidateTargets;
  trackedCandidateTargets.reserve(candidateHints->size());
  for (const LlvmFunctionCandidateHint &hint : *candidateHints) {
    mlir::LLVM::LLVMFuncOp function =
        raised->lookupSymbol<mlir::LLVM::LLVMFuncOp>(hint.symbol);
    if (!function)
      return invalid("LLVM import lost a candidate-hinted function");
    function->setAttr(frontend::structuredCandidateHintAttrName,
                      mlir::UnitAttr::get(&context));
    trackedCandidateTargets.push_back(function.getOperation());
  }

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

  auto finalized = frontend::finalizeStructuredProgramWithTrackedEntities(
      raised.get(), {}, trackedCandidateTargets);
  if (!finalized)
    return finalized.takeError();
  if (finalized->trackedOperations.size() != candidateHints->size())
    return invalid("candidate hint projection changed cardinality");
  std::vector<frontend::StructuredFunctionCandidateHintProjection>
      projectedHints;
  projectedHints.reserve(candidateHints->size());
  for (auto [index, hint] : llvm::enumerate(*candidateHints))
    projectedHints.push_back(
        {finalized->trackedOperations[index],
         std::move(hint.source.sourceFile),
         {hint.source.pragma.line, hint.source.pragma.column},
         {hint.source.targetBegin.line, hint.source.targetBegin.column},
         {hint.source.targetEnd.line, hint.source.targetEnd.column}});
  finalized->candidateHints = std::move(projectedHints);
  return std::move(*finalized);
}

llvm::Expected<frontend::StructuredProgramCandidate>
raiseLlvmModuleToStructuredProgram(std::unique_ptr<llvm::Module> module,
                                   StructuredRaisingOptions options) {
  auto finalized = raiseLlvmModuleToStructuredProgramWithProjection(
      std::move(module), options);
  if (!finalized)
    return finalized.takeError();
  return std::move(finalized->artifact);
}

} // namespace loom::raising
