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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::raising {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_raising_invalid: " + message);
}

llvm::Error candidateError(CandidateHintErrorKind kind,
                           const llvm::Twine &message) {
  return llvm::make_error<CandidateHintError>(kind, message.str());
}

struct LlvmCandidateHint final {
  struct LoopLineage final {
    std::uint64_t latchBlockOrdinal = 0;
    std::uint64_t headerBlockOrdinal = 0;
    std::vector<std::vector<std::uint64_t>> blockSuccessors;
  };

  std::string symbol;
  std::string sourceFile;
  SourcePosition pragma;
  SourcePosition targetBegin;
  SourcePosition targetEnd;
  std::optional<LoopLineage> loopLineage;
  llvm::BasicBlock *pendingLoopLatch = nullptr;
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

llvm::Expected<LlvmCandidateHint::LoopLineage>
captureLoopLineage(llvm::Function &function, llvm::BasicBlock *latch) {
  llvm::DominatorTree dominance(function);
  llvm::LoopInfo loops(dominance);
  llvm::Loop *target = nullptr;
  for (llvm::Loop *loop : loops.getLoopsInPreorder()) {
    if (loop->getLoopLatch() != latch)
      continue;
    if (target)
      return candidateError(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "candidate latch closes more than one LLVM loop");
    target = loop;
  }
  if (!target)
    return candidateError(CandidateHintErrorKind::ProjectionProofNotEstablished,
                          "candidate latch no longer closes an LLVM loop");

  llvm::DenseMap<llvm::BasicBlock *, std::uint64_t> ordinals;
  std::uint64_t ordinal = 0;
  for (llvm::BasicBlock &block : function)
    ordinals.try_emplace(&block, ordinal++);
  auto latchOrdinal = ordinals.find(latch);
  auto headerOrdinal = ordinals.find(target->getHeader());
  if (latchOrdinal == ordinals.end() || headerOrdinal == ordinals.end())
    return candidateError(CandidateHintErrorKind::ProjectionProofNotEstablished,
                          "candidate loop left its LLVM function");

  LlvmCandidateHint::LoopLineage lineage;
  lineage.latchBlockOrdinal = latchOrdinal->second;
  lineage.headerBlockOrdinal = headerOrdinal->second;
  lineage.blockSuccessors.reserve(ordinals.size());
  for (llvm::BasicBlock &block : function) {
    std::vector<std::uint64_t> successors;
    llvm::Instruction *terminator = block.getTerminator();
    successors.reserve(terminator->getNumSuccessors());
    for (unsigned index = 0; index != terminator->getNumSuccessors(); ++index) {
      auto successor = ordinals.find(terminator->getSuccessor(index));
      if (successor == ordinals.end())
        return candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "LLVM candidate CFG has a foreign successor");
      successors.push_back(successor->second);
    }
    lineage.blockSuccessors.push_back(std::move(successors));
  }
  return lineage;
}

llvm::Error
verifyImportedLoopLineage(mlir::LLVM::LLVMFuncOp function,
                          const LlvmCandidateHint::LoopLineage &lineage) {
  llvm::SmallVector<mlir::Block *> blocks;
  llvm::DenseMap<mlir::Block *, std::uint64_t> ordinals;
  for (mlir::Block &block : function.getBody()) {
    ordinals.try_emplace(&block, blocks.size());
    blocks.push_back(&block);
  }
  if (blocks.size() != lineage.blockSuccessors.size() ||
      lineage.latchBlockOrdinal >= blocks.size() ||
      lineage.headerBlockOrdinal >= blocks.size())
    return candidateError(
        CandidateHintErrorKind::ProjectionProofNotEstablished,
        "LLVM import changed the candidate loop CFG cardinality");

  for (auto [block, expected] :
       llvm::zip_equal(blocks, lineage.blockSuccessors)) {
    mlir::Operation *terminator = block->getTerminator();
    if (terminator->getNumSuccessors() != expected.size())
      return candidateError(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "LLVM import changed a candidate loop CFG edge count");
    for (auto [successor, expectedOrdinal] :
         llvm::zip_equal(terminator->getSuccessors(), expected)) {
      auto importedOrdinal = ordinals.find(successor);
      if (importedOrdinal == ordinals.end() ||
          importedOrdinal->second != expectedOrdinal)
        return candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "LLVM import changed a candidate loop CFG successor");
    }
  }

  mlir::Block *latch = blocks[lineage.latchBlockOrdinal];
  mlir::Block *header = blocks[lineage.headerBlockOrdinal];
  mlir::DominanceInfo dominance(function.getOperation());
  unsigned latchCount = 0;
  for (mlir::Block *block : blocks) {
    if (!dominance.dominates(header, block))
      continue;
    for (mlir::Block *successor : block->getTerminator()->getSuccessors())
      if (successor == header) {
        ++latchCount;
        if (block != latch)
          return candidateError(
              CandidateHintErrorKind::ProjectionProofNotEstablished,
              "LLVM import changed the candidate loop latch");
      }
  }
  if (latchCount != 1)
    return candidateError(CandidateHintErrorKind::ProjectionProofNotEstablished,
                          "imported candidate loop has no unique latch");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> exactMetadataString(llvm::Metadata *metadata,
                                                    const llvm::Twine &owner) {
  auto *value = llvm::dyn_cast_or_null<llvm::MDString>(metadata);
  if (!value)
    return candidateError(CandidateHintErrorKind::InvalidEncoding,
                          owner + " does not contain a metadata string");
  return value->getString();
}

llvm::Expected<std::optional<std::string>>
stripLoopCandidateMetadata(llvm::Instruction &terminator) {
  llvm::MDNode *loopID = terminator.getMetadata(llvm::LLVMContext::MD_loop);
  if (!loopID)
    return std::nullopt;

  std::optional<std::string> encoded;
  llvm::SmallVector<llvm::Metadata *> retained;
  retained.push_back(nullptr);
  for (unsigned index = 1; index != loopID->getNumOperands(); ++index) {
    llvm::Metadata *operand = loopID->getOperand(index).get();
    auto *property = llvm::dyn_cast_or_null<llvm::MDNode>(operand);
    auto *name = property && property->getNumOperands() != 0
                     ? llvm::dyn_cast_or_null<llvm::MDString>(
                           property->getOperand(0).get())
                     : nullptr;
    if (!name || name->getString() != loopCandidateMetadataName) {
      retained.push_back(operand);
      continue;
    }
    if (encoded)
      return candidateError(CandidateHintErrorKind::InvalidPlacement,
                            "loop has duplicate candidate metadata");
    if (property->getNumOperands() != 2)
      return candidateError(
          CandidateHintErrorKind::InvalidEncoding,
          "candidate loop metadata does not have two operands");
    auto payload = exactMetadataString(property->getOperand(1).get(),
                                       "candidate loop metadata");
    if (!payload)
      return payload.takeError();
    encoded = payload->str();
  }
  if (!encoded)
    return std::nullopt;
  if (loopID->getNumOperands() == 0 || loopID->getOperand(0).get() != loopID)
    return candidateError(CandidateHintErrorKind::InvalidEncoding,
                          "candidate loop metadata is not self-referential");

  if (retained.size() == 1) {
    terminator.setMetadata(llvm::LLVMContext::MD_loop, nullptr);
  } else {
    llvm::MDNode *replacement =
        llvm::MDNode::getDistinct(terminator.getContext(), retained);
    replacement->replaceOperandWith(0, replacement);
    terminator.setMetadata(llvm::LLVMContext::MD_loop, replacement);
  }
  return encoded;
}

llvm::Error extractProjectedCandidateMetadata(
    llvm::Module &module, std::vector<LlvmCandidateHint> &result,
    llvm::StringSet<> &selectedFunctions,
    std::set<std::pair<std::string, std::uint64_t>> &selectedLoops,
    llvm::StringSet<> &selectedPayloads) {
  struct ExpectedLoop final {
    std::string carrier;
    std::string encoded;
    LoopCandidateAnnotation source;
  };
  struct ProjectedLoopTarget final {
    std::string carrier;
    llvm::BasicBlock *latch = nullptr;
  };
  std::vector<ExpectedLoop> expectedLoops;
  std::map<std::string, ProjectedLoopTarget> projectedLoops;

  for (llvm::Function &function : module) {
    if (llvm::MDNode *metadata =
            function.getMetadata(functionCandidateMetadataName)) {
      if (function.isDeclaration())
        return candidateError(
            CandidateHintErrorKind::UnsupportedConstruct,
            "candidate function metadata targets a declaration");
      if (metadata->getNumOperands() != 1)
        return candidateError(
            CandidateHintErrorKind::InvalidEncoding,
            "candidate function metadata does not have one operand");
      auto encoded = exactMetadataString(metadata->getOperand(0).get(),
                                         "candidate function metadata");
      if (!encoded)
        return encoded.takeError();
      auto source = decodeFunctionCandidateAnnotation(*encoded);
      if (!source)
        return source.takeError();
      if (!selectedFunctions.insert(function.getName()).second)
        return candidateError(CandidateHintErrorKind::InvalidPlacement,
                              "function has duplicate candidate metadata");
      if (!selectedPayloads.insert(*encoded).second)
        return candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "one source function candidate reached more than one LLVM "
            "function");
      result.push_back({function.getName().str(), std::move(source->sourceFile),
                        source->pragma, source->targetBegin, source->targetEnd,
                        std::nullopt});
      function.setMetadata(functionCandidateMetadataName, nullptr);
    }

    if (llvm::MDNode *manifest =
            function.getMetadata(loopCandidateManifestMetadataName)) {
      if (function.isDeclaration())
        return candidateError(CandidateHintErrorKind::UnsupportedConstruct,
                              "candidate loop manifest targets a declaration");
      if (manifest->getNumOperands() == 0)
        return candidateError(CandidateHintErrorKind::InvalidEncoding,
                              "candidate loop manifest is empty");
      for (const llvm::MDOperand &operand : manifest->operands()) {
        auto encoded =
            exactMetadataString(operand.get(), "candidate loop manifest");
        if (!encoded)
          return encoded.takeError();
        auto source = decodeLoopCandidateAnnotation(*encoded);
        if (!source)
          return source.takeError();
        if (!selectedLoops.insert({function.getName().str(), source->marker})
                 .second)
          return candidateError(CandidateHintErrorKind::InvalidPlacement,
                                "candidate loop manifest has a duplicate "
                                "carrier-local marker");
        expectedLoops.push_back(
            {function.getName().str(), encoded->str(), std::move(*source)});
      }
      function.setMetadata(loopCandidateManifestMetadataName, nullptr);
    }

    if (function.isDeclaration())
      continue;
    for (llvm::BasicBlock &block : function) {
      auto encoded = stripLoopCandidateMetadata(*block.getTerminator());
      if (!encoded)
        return encoded.takeError();
      if (!*encoded)
        continue;
      auto source = decodeLoopCandidateAnnotation(**encoded);
      if (!source)
        return source.takeError();
      if (!projectedLoops
               .try_emplace(
                   **encoded,
                   ProjectedLoopTarget{function.getName().str(), &block})
               .second)
        return candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "one source loop candidate reached more than one LLVM loop");
    }
  }

  for (ExpectedLoop &expected : expectedLoops) {
    auto target = projectedLoops.find(expected.encoded);
    if (target == projectedLoops.end())
      return candidateError(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "optimization erased or rewrote a candidate loop target");
    if (target->second.carrier != expected.carrier)
      return candidateError(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "candidate loop target moved to another LLVM function");
    if (!selectedPayloads.insert(expected.encoded).second)
      return candidateError(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "one source loop candidate has duplicate LLVM projections");
    result.push_back({expected.carrier, std::move(expected.source.sourceFile),
                      expected.source.pragma, expected.source.targetBegin,
                      expected.source.targetEnd, std::nullopt,
                      target->second.latch});
    projectedLoops.erase(target);
  }
  if (!projectedLoops.empty())
    return candidateError(CandidateHintErrorKind::InvalidEncoding,
                          "candidate loop target has no carrier manifest");
  return llvm::Error::success();
}

llvm::Error rejectUnprojectedCandidateAnnotations(llvm::Module &module) {
  llvm::GlobalVariable *annotations =
      module.getNamedGlobal("llvm.global.annotations");
  if (annotations) {
    if (!annotations->hasInitializer())
      return invalid("llvm.global.annotations has no initializer");
    auto *initializer =
        llvm::dyn_cast<llvm::Constant>(annotations->getInitializer());
    if (!initializer)
      return invalid("llvm.global.annotations has a non-constant initializer");

    for (llvm::Value *operand : initializer->operand_values()) {
      auto *entry = llvm::dyn_cast<llvm::ConstantStruct>(operand);
      if (!entry || entry->getNumOperands() < 2)
        continue;
      auto encoded = annotationString(entry->getOperand(1));
      if (!encoded) {
        llvm::consumeError(encoded.takeError());
        continue;
      }
      if (encoded->starts_with("loom.candidate."))
        return candidateError(
            CandidateHintErrorKind::UnsupportedConstruct,
            "raw candidate annotations require the LLVM candidate projection "
            "pass before structured raising");
    }
  }

  for (llvm::Function &function : module)
    for (llvm::Instruction &instruction : llvm::instructions(function)) {
      auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
      if (!intrinsic ||
          intrinsic->getIntrinsicID() != llvm::Intrinsic::annotation)
        continue;
      auto encoded = annotationString(intrinsic->getArgOperand(1));
      if (!encoded) {
        llvm::consumeError(encoded.takeError());
        continue;
      }
      if (encoded->starts_with("loom.candidate."))
        return candidateError(
            CandidateHintErrorKind::UnsupportedConstruct,
            "raw candidate loop markers require the LLVM candidate projection "
            "pass before structured raising");
    }
  return llvm::Error::success();
}

llvm::Expected<std::vector<LlvmCandidateHint>>
extractCandidateHints(llvm::Module &module) {
  std::vector<LlvmCandidateHint> result;
  llvm::StringSet<> selectedFunctions;
  std::set<std::pair<std::string, std::uint64_t>> selectedLoops;
  llvm::StringSet<> selectedPayloads;
  if (llvm::Error error = extractProjectedCandidateMetadata(
          module, result, selectedFunctions, selectedLoops, selectedPayloads))
    return std::move(error);
  if (llvm::Error error = rejectUnprojectedCandidateAnnotations(module))
    return std::move(error);

  for (LlvmCandidateHint &hint : result) {
    if (!hint.pendingLoopLatch)
      continue;
    llvm::Function *function = module.getFunction(hint.symbol);
    if (!function)
      return candidateError(
          CandidateHintErrorKind::ProjectionProofNotEstablished,
          "candidate loop carrier left its LLVM module");
    auto lineage = captureLoopLineage(*function, hint.pendingLoopLatch);
    if (!lineage)
      return lineage.takeError();
    hint.loopLineage = std::move(*lineage);
    hint.pendingLoopLatch = nullptr;
  }
  if (llvm::Error error = removeCandidateTemporaryRetention(module))
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

  normalizeBulkMemoryIntrinsics(*module);
  if (llvm::verifyModule(*module))
    return invalid("bulk-memory normalization produced invalid LLVM IR");
  auto candidateHints = extractCandidateHints(*module);
  if (!candidateHints)
    return candidateHints.takeError();
  if (llvm::verifyModule(*module))
    return invalid("candidate projection produced invalid LLVM IR");
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

  for (auto [index, hint] : llvm::enumerate(*candidateHints)) {
    mlir::LLVM::LLVMFuncOp function =
        raised->lookupSymbol<mlir::LLVM::LLVMFuncOp>(hint.symbol);
    if (!function)
      return invalid("LLVM import lost a candidate-hinted function");
    auto lineage =
        mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), index);
    if (!hint.loopLineage) {
      function->setAttr(frontend::structuredCandidateHintAttrName, lineage);
      continue;
    }
    if (llvm::Error error =
            verifyImportedLoopLineage(function, *hint.loopLineage))
      return std::move(error);
    auto block = function.getBody().begin();
    std::advance(block, hint.loopLineage->latchBlockOrdinal);
    block->getTerminator()->setAttr(
        frontend::structuredCandidateLoopHintAttrName, lineage);
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

  llvm::SmallVector<mlir::Operation *> trackedCandidateTargets(
      candidateHints->size(), nullptr);
  llvm::Error targetError = llvm::Error::success();
  raised->walk([&](mlir::Operation *operation) {
    if (targetError)
      return mlir::WalkResult::interrupt();
    for (llvm::StringRef name :
         {frontend::structuredCandidateHintAttrName,
          frontend::structuredCandidateLoopHintAttrName}) {
      mlir::Attribute attribute = operation->getAttr(name);
      if (!attribute)
        continue;
      const bool loopTarget =
          name == frontend::structuredCandidateLoopHintAttrName;
      if (loopTarget &&
          !mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp>(operation)) {
        targetError = candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "mechanical raising did not recover a candidate loop target");
        return mlir::WalkResult::interrupt();
      }
      if (!loopTarget && !mlir::isa<mlir::LLVM::LLVMFuncOp>(operation)) {
        targetError = candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "mechanical raising did not retain a candidate function target");
        return mlir::WalkResult::interrupt();
      }
      auto lineage = llvm::dyn_cast<mlir::IntegerAttr>(attribute);
      if (!lineage || lineage.getValue().isNegative() ||
          lineage.getValue().getActiveBits() > 64 ||
          lineage.getValue().getZExtValue() >= trackedCandidateTargets.size()) {
        targetError = candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "raised candidate target has an invalid lineage ordinal");
        return mlir::WalkResult::interrupt();
      }
      const std::uint64_t index = lineage.getValue().getZExtValue();
      if (trackedCandidateTargets[index]) {
        targetError = candidateError(
            CandidateHintErrorKind::ProjectionProofNotEstablished,
            "one source candidate resolved to more than one raised target");
        return mlir::WalkResult::interrupt();
      }
      trackedCandidateTargets[index] = operation;
    }
    return mlir::WalkResult::advance();
  });
  if (targetError)
    return std::move(targetError);
  if (llvm::any_of(trackedCandidateTargets,
                   [](mlir::Operation *operation) { return !operation; }))
    return candidateError(CandidateHintErrorKind::ProjectionProofNotEstablished,
                          "mechanical raising lost a candidate target");

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
         std::move(hint.sourceFile),
         {hint.pragma.line, hint.pragma.column},
         {hint.targetBegin.line, hint.targetBegin.column},
         {hint.targetEnd.line, hint.targetEnd.column},
         hint.loopLineage
             ? frontend::StructuredCandidateHintTargetKind::Loop
             : frontend::StructuredCandidateHintTargetKind::Function});
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
