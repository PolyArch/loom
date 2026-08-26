#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Raising/CandidateHints.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>

namespace {

llvm::Error candidateError(loom::raising::CandidateHintErrorKind kind,
                           const llvm::Twine &message) {
  return llvm::make_error<loom::raising::CandidateHintError>(kind,
                                                             message.str());
}

llvm::GlobalVariable *referencedGlobal(llvm::Value *value) {
  return llvm::dyn_cast<llvm::GlobalVariable>(value->stripPointerCasts());
}

llvm::Expected<llvm::StringRef> annotationString(llvm::Value *value) {
  llvm::GlobalVariable *global = referencedGlobal(value);
  if (!global || !global->hasInitializer())
    return candidateError(
        loom::raising::CandidateHintErrorKind::InvalidEncoding,
        "candidate annotation does not reference a string global");
  auto *data =
      llvm::dyn_cast<llvm::ConstantDataSequential>(global->getInitializer());
  if (!data || !data->isCString())
    return candidateError(
        loom::raising::CandidateHintErrorKind::InvalidEncoding,
        "candidate annotation is not a C string");
  return data->getAsCString();
}

bool hasLoopMetadataProperty(const llvm::MDNode *loopID,
                             llvm::StringRef propertyName) {
  if (!loopID)
    return false;
  for (unsigned index = 1; index != loopID->getNumOperands(); ++index) {
    auto *property =
        llvm::dyn_cast_or_null<llvm::MDNode>(loopID->getOperand(index).get());
    if (!property || property->getNumOperands() == 0)
      continue;
    auto *name =
        llvm::dyn_cast_or_null<llvm::MDString>(property->getOperand(0).get());
    if (name && name->getString() == propertyName)
      return true;
  }
  return false;
}

struct FunctionProjection final {
  llvm::Function *function = nullptr;
  std::string encoded;
};

struct LoopProjection final {
  llvm::Function *carrier = nullptr;
  llvm::CallBase *markerCall = nullptr;
  llvm::SwitchInst *wrapper = nullptr;
  llvm::BasicBlock *wrapperBody = nullptr;
  llvm::BasicBlock *wrapperExit = nullptr;
  llvm::BasicBlock *latch = nullptr;
  llvm::Function *markerIntrinsic = nullptr;
  llvm::GlobalVariable *markerString = nullptr;
  llvm::GlobalVariable *sourceFileString = nullptr;
  std::string encoded;
};

struct TemporaryRetentionProjection final {
  llvm::Function *function = nullptr;
  bool sourceRequired = false;
};

struct CandidateProjectionPlan final {
  llvm::GlobalVariable *annotations = nullptr;
  llvm::SmallVector<llvm::Constant *> retainedAnnotations;
  llvm::SmallVector<llvm::GlobalVariable *> projectedStrings;
  llvm::SmallVector<FunctionProjection> functions;
  llvm::SmallVector<LoopProjection> loops;
  llvm::SmallVector<TemporaryRetentionProjection> temporaryRetentions;
};

llvm::Expected<LoopProjection>
planLoopProjection(llvm::Function &function,
                   const loom::raising::LoopCandidateAnnotation &source,
                   llvm::StringRef encoded) {
  const std::string expectedMarker =
      loom::raising::encodeLoopCandidateMarker(source.marker);
  llvm::CallBase *markerCall = nullptr;
  for (llvm::Instruction &instruction : llvm::instructions(function)) {
    auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction);
    if (!intrinsic ||
        intrinsic->getIntrinsicID() != llvm::Intrinsic::annotation)
      continue;
    auto marker = annotationString(intrinsic->getArgOperand(1));
    if (!marker) {
      llvm::consumeError(marker.takeError());
      continue;
    }
    if (*marker != expectedMarker)
      continue;
    if (markerCall)
      return candidateError(
          loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
          "loop marker has more than one LLVM occurrence in its carrier");
    markerCall = intrinsic;
  }
  if (!markerCall)
    return candidateError(
        loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
        "loop marker has no LLVM occurrence in its carrier");

  auto *annotatedValue =
      llvm::dyn_cast<llvm::ConstantInt>(markerCall->getArgOperand(0));
  auto markerFile = annotationString(markerCall->getArgOperand(2));
  auto *markerLine =
      llvm::dyn_cast<llvm::ConstantInt>(markerCall->getArgOperand(3));
  if (!annotatedValue || !annotatedValue->isZero() || !markerFile ||
      *markerFile != source.sourceFile || !markerLine ||
      !markerLine->getType()->isIntegerTy(32) ||
      markerLine->getZExtValue() != source.pragma.line) {
    if (!markerFile)
      llvm::consumeError(markerFile.takeError());
    return candidateError(
        loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
        "loop marker source anchor disagrees with its candidate payload");
  }

  llvm::BasicBlock *markerBlock = markerCall->getParent();
  auto *wrapper =
      llvm::dyn_cast<llvm::SwitchInst>(markerBlock->getTerminator());
  if (!wrapper || wrapper->getNumCases() != 0 ||
      wrapper->getCondition() != markerCall || !markerCall->hasOneUse())
    return candidateError(
        loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
        "loop marker is not carried by the exact compiler switch wrapper");
  for (llvm::Instruction *next = markerCall->getNextNode(); next != wrapper;
       next = next ? next->getNextNode() : nullptr)
    if (!next || !llvm::isa<llvm::DbgInfoIntrinsic>(next))
      return candidateError(
          loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
          "loop marker switch wrapper contains an unexpected instruction");

  llvm::BasicBlock *wrapperBody = wrapper->getDefaultDest();
  if (wrapperBody == markerBlock ||
      wrapperBody->getUniquePredecessor() != markerBlock ||
      wrapperBody->hasAddressTaken())
    return candidateError(
        loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
        "loop marker switch wrapper has no exact removable boundary");

  llvm::DominatorTree dominance(function);
  llvm::LoopInfo loops(dominance);
  llvm::Loop *target = nullptr;
  for (llvm::Loop *loop : loops.getLoopsInPreorder()) {
    if (loop->getLoopPreheader() != wrapperBody)
      continue;
    if (target)
      return candidateError(
          loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
          "loop marker reaches more than one LLVM loop from one preheader");
    target = loop;
  }
  if (!target)
    return candidateError(
        loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
        "loop marker does not reach a preserved LLVM loop");
  llvm::BasicBlock *latch = target->getLoopLatch();
  if (!latch)
    return candidateError(
        loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
        "candidate loop has no unique LLVM latch");
  llvm::BasicBlock *loopExit = target->getUniqueExitBlock();
  llvm::BasicBlock *wrapperExit =
      loopExit ? loopExit->getSingleSuccessor() : nullptr;
  if (!loopExit || !wrapperExit ||
      wrapperExit->getUniquePredecessor() != loopExit ||
      wrapperExit->hasAddressTaken())
    return candidateError(
        loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
        "candidate loop has no exact removable switch exit boundary");
  if (hasLoopMetadataProperty(
          latch->getTerminator()->getMetadata(llvm::LLVMContext::MD_loop),
          loom::raising::loopCandidateMetadataName))
    return candidateError(
        loom::raising::CandidateHintErrorKind::InvalidPlacement,
        "candidate loop already has projected metadata");

  return LoopProjection{&function,
                        markerCall,
                        wrapper,
                        wrapperBody,
                        wrapperExit,
                        latch,
                        markerCall->getCalledFunction(),
                        referencedGlobal(markerCall->getArgOperand(1)),
                        referencedGlobal(markerCall->getArgOperand(2)),
                        encoded.str()};
}

llvm::Expected<CandidateProjectionPlan>
buildProjectionPlan(llvm::Module &module) {
  CandidateProjectionPlan plan;
  plan.annotations = module.getNamedGlobal("llvm.global.annotations");
  if (!plan.annotations)
    return plan;
  if (!plan.annotations->hasInitializer())
    return candidateError(
        loom::raising::CandidateHintErrorKind::InvalidEncoding,
        "llvm.global.annotations has no initializer");
  auto *initializer =
      llvm::dyn_cast<llvm::Constant>(plan.annotations->getInitializer());
  if (!initializer)
    return candidateError(
        loom::raising::CandidateHintErrorKind::InvalidEncoding,
        "llvm.global.annotations has a non-constant initializer");

  llvm::SmallPtrSet<llvm::Function *, 4> selectedFunctions;
  llvm::SmallPtrSet<llvm::Function *, 8> selectedCarriers;
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> selectedLatches;
  llvm::SmallPtrSet<llvm::Function *, 4> selectedTemporaryRetentions;
  std::set<std::pair<llvm::Function *, std::uint64_t>> selectedLoops;
  for (llvm::Value *operand : initializer->operand_values()) {
    auto *entry = llvm::dyn_cast<llvm::ConstantStruct>(operand);
    if (!entry || entry->getNumOperands() < 2) {
      auto *constant = llvm::dyn_cast<llvm::Constant>(operand);
      if (!constant)
        return candidateError(
            loom::raising::CandidateHintErrorKind::InvalidEncoding,
            "global annotation entry is not constant");
      plan.retainedAnnotations.push_back(constant);
      continue;
    }

    auto encoded = annotationString(entry->getOperand(1));
    if (!encoded) {
      llvm::consumeError(encoded.takeError());
      plan.retainedAnnotations.push_back(entry);
      continue;
    }
    if (!encoded->starts_with("loom.candidate.")) {
      plan.retainedAnnotations.push_back(entry);
      continue;
    }
    if (entry->getNumOperands() != 5)
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "candidate annotation entry does not have five fields");
    auto *function = llvm::dyn_cast<llvm::Function>(
        entry->getOperand(0)->stripPointerCasts());
    if (!function || function->isDeclaration())
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidPlacement,
          "candidate annotation does not target a function definition");
    auto file = annotationString(entry->getOperand(2));
    auto *line = llvm::dyn_cast<llvm::ConstantInt>(entry->getOperand(3));
    if (!file || !line || !line->getType()->isIntegerTy(32)) {
      if (!file)
        llvm::consumeError(file.takeError());
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "candidate annotation has an invalid LLVM source anchor");
    }

    if (*encoded == loom::raising::candidateTemporaryRetentionAnnotation ||
        *encoded == loom::raising::
                        candidateSourceRequiredTemporaryRetentionAnnotation) {
      if (!selectedTemporaryRetentions.insert(function).second)
        return candidateError(
            loom::raising::CandidateHintErrorKind::InvalidPlacement,
            "function has duplicate candidate retention annotations");
      plan.temporaryRetentions.push_back(
          {function,
           *encoded ==
               loom::raising::
                   candidateSourceRequiredTemporaryRetentionAnnotation});
      plan.projectedStrings.push_back(referencedGlobal(entry->getOperand(1)));
      plan.projectedStrings.push_back(referencedGlobal(entry->getOperand(2)));
      continue;
    }

    if (encoded->starts_with(
            loom::raising::functionCandidateAnnotationSchema)) {
      auto source = loom::raising::decodeFunctionCandidateAnnotation(*encoded);
      if (!source)
        return source.takeError();
      if (!selectedFunctions.insert(function).second ||
          function->getMetadata(loom::raising::functionCandidateMetadataName))
        return candidateError(
            loom::raising::CandidateHintErrorKind::InvalidPlacement,
            "function has duplicate candidate annotations");
      if (*file != source->sourceFile ||
          line->getZExtValue() != source->carrier.line)
        return candidateError(
            loom::raising::CandidateHintErrorKind::
                ProjectionProofNotEstablished,
            "candidate function carrier disagrees with its payload");
      plan.functions.push_back({function, encoded->str()});
      selectedCarriers.insert(function);
    } else if (encoded->starts_with(
                   loom::raising::loopCandidateAnnotationSchema)) {
      auto source = loom::raising::decodeLoopCandidateAnnotation(*encoded);
      if (!source)
        return source.takeError();
      if (function->getMetadata(
              loom::raising::loopCandidateManifestMetadataName))
        return candidateError(
            loom::raising::CandidateHintErrorKind::InvalidPlacement,
            "loop candidate carrier already has a projected manifest");
      if (!selectedLoops.insert({function, source->marker}).second)
        return candidateError(
            loom::raising::CandidateHintErrorKind::InvalidPlacement,
            "loop has duplicate candidate annotations");
      if (*file != source->sourceFile ||
          line->getZExtValue() != source->carrier.line)
        return candidateError(
            loom::raising::CandidateHintErrorKind::
                ProjectionProofNotEstablished,
            "loop candidate carrier disagrees with its payload");
      auto loop = planLoopProjection(*function, *source, *encoded);
      if (!loop)
        return loop.takeError();
      if (!selectedLatches.insert(loop->latch).second)
        return candidateError(
            loom::raising::CandidateHintErrorKind::InvalidPlacement,
            "LLVM loop has more than one candidate annotation");
      plan.projectedStrings.push_back(loop->markerString);
      plan.projectedStrings.push_back(loop->sourceFileString);
      plan.loops.push_back(std::move(*loop));
      selectedCarriers.insert(function);
    } else {
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "candidate annotation has an unsupported schema");
    }
    plan.projectedStrings.push_back(referencedGlobal(entry->getOperand(1)));
    plan.projectedStrings.push_back(referencedGlobal(entry->getOperand(2)));
  }
  for (const TemporaryRetentionProjection &retention : plan.temporaryRetentions)
    if (!selectedCarriers.contains(retention.function))
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidPlacement,
          "candidate retention annotation has no candidate carrier");

  llvm::SmallPtrSet<llvm::Function *, 4> cleanupRequired;
  for (const TemporaryRetentionProjection &retention : plan.temporaryRetentions)
    if (!retention.sourceRequired)
      cleanupRequired.insert(retention.function);
  for (llvm::Function *function : cleanupRequired)
    if (function->getMetadata(llvm::getPGOFuncNameMetadataName()) ||
        module.getNamedMetadata("llvm.gcov"))
      return candidateError(
          loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
          "a temporary candidate cannot preserve profiling or coverage "
          "support data");
  for (llvm::Constant *constant : plan.retainedAnnotations) {
    auto *entry = llvm::dyn_cast<llvm::ConstantStruct>(constant);
    auto *target = entry && entry->getNumOperands() != 0
                       ? llvm::dyn_cast<llvm::Function>(
                             entry->getOperand(0)->stripPointerCasts())
                       : nullptr;
    if (!target || !cleanupRequired.contains(target) ||
        entry->getNumOperands() < 5 || entry->getOperand(4)->isNullValue())
      continue;
    return candidateError(
        loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
        "a temporary candidate has an unrelated annotation with arguments");
  }
  return plan;
}

void appendLoopCandidateMetadata(llvm::Instruction &latchTerminator,
                                 llvm::StringRef encoded) {
  llvm::LLVMContext &context = latchTerminator.getContext();
  llvm::MDNode *oldLoopID =
      latchTerminator.getMetadata(llvm::LLVMContext::MD_loop);
  llvm::SmallVector<llvm::Metadata *> operands;
  operands.push_back(nullptr);
  if (oldLoopID)
    for (unsigned index = 1; index != oldLoopID->getNumOperands(); ++index)
      operands.push_back(oldLoopID->getOperand(index).get());
  operands.push_back(llvm::MDNode::get(
      context,
      {llvm::MDString::get(context, loom::raising::loopCandidateMetadataName),
       llvm::MDString::get(context, encoded)}));
  llvm::MDNode *newLoopID = llvm::MDNode::getDistinct(context, operands);
  newLoopID->replaceOperandWith(0, newLoopID);
  latchTerminator.setMetadata(llvm::LLVMContext::MD_loop, newLoopID);
}

void attachLoopCandidateManifests(llvm::ArrayRef<LoopProjection> projections) {
  llvm::DenseMap<llvm::Function *, llvm::SmallVector<llvm::Metadata *>>
      byCarrier;
  for (const LoopProjection &projection : projections)
    byCarrier[projection.carrier].push_back(llvm::MDString::get(
        projection.carrier->getContext(), projection.encoded));
  for (auto &[carrier, payloads] : byCarrier)
    carrier->setMetadata(loom::raising::loopCandidateManifestMetadataName,
                         llvm::MDNode::get(carrier->getContext(), payloads));
}

llvm::Error
projectCandidateTemporaryRetentions(llvm::Module &module,
                                    const CandidateProjectionPlan &plan) {
  llvm::SmallVector<llvm::GlobalValue *> compilerUsed;
  llvm::collectUsedGlobalVariables(module, compilerUsed,
                                   /*CompilerUsed=*/true);
  llvm::SmallPtrSet<llvm::GlobalValue *, 16> compilerRetained(
      compilerUsed.begin(), compilerUsed.end());
  for (const TemporaryRetentionProjection &retention : plan.temporaryRetentions)
    if (!compilerRetained.contains(retention.function))
      return candidateError(
          loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
          "candidate retention annotation lost llvm.compiler.used");
  for (const TemporaryRetentionProjection &retention :
       plan.temporaryRetentions) {
    llvm::Function *function = retention.function;
    function->setMetadata(
        loom::raising::candidateTemporaryRetentionMetadataName,
        llvm::MDNode::get(function->getContext(),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                              llvm::Type::getInt1Ty(function->getContext()),
                              retention.sourceRequired))));
  }
  return llvm::Error::success();
}

llvm::Error
replaceGlobalAnnotations(llvm::Module &module,
                         llvm::GlobalVariable &annotations,
                         llvm::ArrayRef<llvm::Constant *> retained) {
  if (!retained.empty()) {
    auto *entryType =
        llvm::dyn_cast<llvm::StructType>(retained.front()->getType());
    if (!entryType)
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "retained global annotation has a non-struct type");
    for (llvm::Constant *entry : retained)
      if (entry->getType() != entryType)
        return candidateError(
            loom::raising::CandidateHintErrorKind::InvalidEncoding,
            "retained global annotations have inconsistent types");
    auto *arrayType = llvm::ArrayType::get(entryType, retained.size());
    llvm::Constant *initializer = llvm::ConstantArray::get(arrayType, retained);
    auto *replacement = new llvm::GlobalVariable(
        module, arrayType, annotations.isConstant(), annotations.getLinkage(),
        initializer, "", &annotations, annotations.getThreadLocalMode(),
        annotations.getAddressSpace(), annotations.isExternallyInitialized());
    replacement->copyAttributesFrom(&annotations);
    replacement->takeName(&annotations);
    annotations.replaceAllUsesWith(replacement);
  } else if (!annotations.use_empty()) {
    return candidateError(
        loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
        "projected global annotations have unexpected users");
  }
  annotations.eraseFromParent();
  return llvm::Error::success();
}

llvm::Error applyProjectionPlan(llvm::Module &module,
                                CandidateProjectionPlan &plan) {
  if (plan.functions.empty() && plan.loops.empty())
    return llvm::Error::success();

  for (const FunctionProjection &projection : plan.functions) {
    llvm::LLVMContext &context = projection.function->getContext();
    projection.function->setMetadata(
        loom::raising::functionCandidateMetadataName,
        llvm::MDNode::get(context,
                          llvm::MDString::get(context, projection.encoded)));
  }
  for (const LoopProjection &projection : plan.loops)
    appendLoopCandidateMetadata(*projection.latch->getTerminator(),
                                projection.encoded);
  attachLoopCandidateManifests(plan.loops);

  if (llvm::Error error = projectCandidateTemporaryRetentions(module, plan))
    return error;

  llvm::SmallPtrSet<llvm::Function *, 4> markerIntrinsics;
  for (const LoopProjection &projection : plan.loops) {
    llvm::UncondBrInst *branch =
        llvm::UncondBrInst::Create(projection.wrapperBody);
    branch->setDebugLoc(projection.wrapper->getDebugLoc());
    llvm::ReplaceInstWithInst(projection.wrapper, branch);
    projection.markerCall->eraseFromParent();
    if (!llvm::MergeBlockIntoPredecessor(projection.wrapperBody))
      return candidateError(
          loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
          "loop marker switch wrapper could not be removed exactly");
    if (!llvm::MergeBlockIntoPredecessor(projection.wrapperExit))
      return candidateError(
          loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
          "loop marker switch exit could not be removed exactly");
    markerIntrinsics.insert(projection.markerIntrinsic);
  }

  if (llvm::Error error = replaceGlobalAnnotations(module, *plan.annotations,
                                                   plan.retainedAnnotations))
    return error;

  for (llvm::Function *intrinsic : markerIntrinsics)
    if (intrinsic && intrinsic->use_empty())
      intrinsic->eraseFromParent();

  llvm::SmallPtrSet<llvm::GlobalVariable *, 16> visited;
  for (llvm::GlobalVariable *string : plan.projectedStrings) {
    if (!string)
      return candidateError(
          loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
          "candidate projection lost a generated string global");
    if (!visited.insert(string).second)
      continue;
    string->removeDeadConstantUsers();
    if (string->use_empty() && string->hasLocalLinkage())
      string->eraseFromParent();
  }

  if (llvm::verifyModule(module, &llvm::errs()))
    return candidateError(
        loom::raising::CandidateHintErrorKind::ProjectionProofNotEstablished,
        "candidate projection produced an invalid LLVM module");
  return llvm::Error::success();
}

class CandidateProjectionPass
    : public llvm::PassInfoMixin<CandidateProjectionPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &) {
    auto plan = buildProjectionPlan(module);
    if (!plan) {
      module.getContext().emitError("cannot project candidate annotations: " +
                                    llvm::toString(plan.takeError()));
      return llvm::PreservedAnalyses::all();
    }
    if (plan->functions.empty() && plan->loops.empty())
      return llvm::PreservedAnalyses::all();
    if (llvm::Error error = applyProjectionPlan(module, *plan)) {
      module.getContext().emitError("cannot project candidate annotations: " +
                                    llvm::toString(std::move(error)));
      return llvm::PreservedAnalyses::none();
    }
    return llvm::PreservedAnalyses::none();
  }
};

class RemoveCandidateRetentionPass
    : public llvm::PassInfoMixin<RemoveCandidateRetentionPass> {
  struct TemporaryCandidate final {
    llvm::Function *function = nullptr;
    bool sourceRequired = false;
  };

public:
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &) {
    auto carrier = loom::hasGeneratedRelocatablePayloadCarrier(module);
    if (!carrier) {
      module.getContext().emitError(
          "cannot inspect relocatable payload before releasing candidate "
          "retention: " +
          llvm::toString(carrier.takeError()));
      return llvm::PreservedAnalyses::none();
    }
    if (!*carrier)
      return llvm::PreservedAnalyses::all();
    llvm::SmallVector<TemporaryCandidate, 8> temporary;
    for (llvm::Function &function : module) {
      llvm::MDNode *metadata = function.getMetadata(
          loom::raising::candidateTemporaryRetentionMetadataName);
      if (!metadata)
        continue;
      auto *sourceRequired =
          metadata->getNumOperands() == 1
              ? llvm::mdconst::dyn_extract<llvm::ConstantInt>(
                    metadata->getOperand(0))
              : nullptr;
      if (!sourceRequired || !sourceRequired->getType()->isIntegerTy(1)) {
        module.getContext().emitError(
            "cannot release candidate retention: " +
            llvm::toString(candidateError(
                loom::raising::CandidateHintErrorKind::InvalidEncoding,
                "temporary candidate retention has an invalid source "
                "emission fact")));
        return llvm::PreservedAnalyses::none();
      }
      temporary.push_back({&function, !sourceRequired->getValue().isZero()});
    }
    if (llvm::Error error =
            loom::raising::removeCandidateTemporaryRetention(module)) {
      module.getContext().emitError("cannot release candidate retention: " +
                                    llvm::toString(std::move(error)));
      return llvm::PreservedAnalyses::none();
    }
    if (llvm::Error error = eraseUnreferencedTemporaryCandidates(temporary)) {
      module.getContext().emitError(
          "cannot erase unreferenced temporary candidates: " +
          llvm::toString(std::move(error)));
      return llvm::PreservedAnalyses::none();
    }
    return llvm::PreservedAnalyses::none();
  }

private:
  static llvm::Expected<llvm::SmallVector<llvm::GlobalValue *, 16>>
  collectTemporaryDependencyClosure(
      llvm::ArrayRef<TemporaryCandidate> temporary) {
    llvm::SmallVector<llvm::GlobalValue *, 16> closure;
    llvm::SmallPtrSet<llvm::GlobalValue *, 16> selected;
    for (const TemporaryCandidate &candidate : temporary)
      if (selected.insert(candidate.function).second)
        closure.push_back(candidate.function);

    std::optional<std::string> unsupported;
    auto collectValue = [&](const auto &self, llvm::Value *value) -> void {
      auto *constant = llvm::dyn_cast<llvm::Constant>(value);
      if (!constant)
        return;
      auto *global =
          llvm::dyn_cast<llvm::GlobalValue>(constant->stripPointerCasts());
      if (global) {
        if (!global->isDiscardableIfUnused() || selected.contains(global))
          return;
        if (!llvm::isa<llvm::Function, llvm::GlobalVariable>(global)) {
          unsupported = "a temporary candidate has an unsupported "
                        "GlobalValue dependency";
          return;
        }
        selected.insert(global);
        closure.push_back(global);
        return;
      }
      for (llvm::Value *operand : constant->operand_values())
        self(self, operand);
    };

    for (std::size_t index = 0; index != closure.size(); ++index) {
      if (unsupported)
        break;
      llvm::GlobalValue *global = closure[index];
      if (auto *function = llvm::dyn_cast<llvm::Function>(global)) {
        for (llvm::Instruction &instruction : llvm::instructions(*function))
          for (llvm::Value *operand : instruction.operand_values())
            collectValue(collectValue, operand);
        continue;
      }
      auto *variable = llvm::cast<llvm::GlobalVariable>(global);
      if (variable->hasInitializer())
        collectValue(collectValue, variable->getInitializer());
    }
    if (unsupported)
      return candidateError(
          loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
          *unsupported);
    return closure;
  }

  static llvm::Error rejectAnnotatedTemporaryDependencies(
      llvm::Module &module,
      const llvm::SmallPtrSetImpl<llvm::GlobalValue *> &dependencies) {
    llvm::GlobalVariable *annotations =
        module.getNamedGlobal("llvm.global.annotations");
    if (!annotations)
      return llvm::Error::success();
    if (!annotations->hasInitializer())
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "llvm.global.annotations has no initializer");
    auto *initializer =
        llvm::dyn_cast<llvm::Constant>(annotations->getInitializer());
    if (!initializer)
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "llvm.global.annotations has a non-constant initializer");
    for (llvm::Value *operand : initializer->operand_values()) {
      auto *entry = llvm::dyn_cast<llvm::ConstantStruct>(operand);
      auto *target = entry && entry->getNumOperands() != 0
                         ? llvm::dyn_cast<llvm::GlobalValue>(
                               entry->getOperand(0)->stripPointerCasts())
                         : nullptr;
      if (target && dependencies.contains(target))
        return candidateError(
            loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
            "a temporary candidate dependency has an unrelated annotation");
    }
    return llvm::Error::success();
  }

  static void classifyUse(
      llvm::GlobalValue &target, llvm::User &user,
      const llvm::SmallPtrSetImpl<llvm::GlobalValue *> &candidates,
      const llvm::GlobalVariable *annotations, bool sourceRequired,
      llvm::DenseMap<llvm::GlobalValue *,
                     llvm::SmallVector<llvm::GlobalValue *, 4>> &dependencies,
      llvm::SmallPtrSetImpl<llvm::GlobalValue *> &roots,
      llvm::SmallPtrSetImpl<llvm::User *> &visited) {
    if (!visited.insert(&user).second)
      return;
    if (auto *instruction = llvm::dyn_cast<llvm::Instruction>(&user)) {
      llvm::Function *owner = instruction->getFunction();
      if (owner && candidates.contains(owner))
        dependencies[owner].push_back(&target);
      else
        roots.insert(&target);
      return;
    }
    if (auto *global = llvm::dyn_cast<llvm::GlobalVariable>(&user)) {
      if (candidates.contains(global)) {
        dependencies[global].push_back(&target);
        return;
      }
      if (global == annotations && !sourceRequired)
        return;
      roots.insert(&target);
      return;
    }
    if (llvm::isa<llvm::Constant>(&user) &&
        !llvm::isa<llvm::GlobalValue>(&user)) {
      for (llvm::User *nested : user.users())
        classifyUse(target, *nested, candidates, annotations, sourceRequired,
                    dependencies, roots, visited);
      return;
    }
    roots.insert(&target);
  }

  static llvm::Error
  eraseAnnotationsForDeadCandidates(llvm::Module &module,
                                    llvm::ArrayRef<llvm::Function *> dead) {
    llvm::GlobalVariable *annotations =
        module.getNamedGlobal("llvm.global.annotations");
    if (!annotations)
      return llvm::Error::success();
    if (!annotations->hasInitializer())
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "llvm.global.annotations has no initializer");
    auto *initializer =
        llvm::dyn_cast<llvm::Constant>(annotations->getInitializer());
    if (!initializer)
      return candidateError(
          loom::raising::CandidateHintErrorKind::InvalidEncoding,
          "llvm.global.annotations has a non-constant initializer");

    llvm::SmallPtrSet<llvm::Function *, 8> deadSet(dead.begin(), dead.end());
    llvm::SmallVector<llvm::Constant *> retained;
    llvm::SmallVector<llvm::GlobalVariable *> removedSupport;
    llvm::SmallPtrSet<llvm::GlobalVariable *, 8> collectedSupport;
    auto collectSupport = [&](const auto &self, llvm::Value *value) -> void {
      if (llvm::GlobalVariable *global = referencedGlobal(value)) {
        if (!collectedSupport.insert(global).second)
          return;
        removedSupport.push_back(global);
        if (global->hasInitializer())
          self(self, global->getInitializer());
        return;
      }
      auto *constant = llvm::dyn_cast<llvm::Constant>(value);
      if (!constant)
        return;
      for (llvm::Value *operand : constant->operand_values())
        self(self, operand);
    };
    bool removed = false;
    for (llvm::Value *operand : initializer->operand_values()) {
      auto *constant = llvm::dyn_cast<llvm::Constant>(operand);
      auto *entry = llvm::dyn_cast<llvm::ConstantStruct>(operand);
      auto *target = entry && entry->getNumOperands() != 0
                         ? llvm::dyn_cast<llvm::Function>(
                               entry->getOperand(0)->stripPointerCasts())
                         : nullptr;
      if (!target || !deadSet.contains(target)) {
        if (!constant)
          return candidateError(
              loom::raising::CandidateHintErrorKind::InvalidEncoding,
              "global annotation entry is not constant");
        retained.push_back(constant);
        continue;
      }
      removed = true;
      for (unsigned index = 1; index != entry->getNumOperands(); ++index)
        collectSupport(collectSupport, entry->getOperand(index));
    }
    if (!removed)
      return llvm::Error::success();
    if (llvm::Error error =
            replaceGlobalAnnotations(module, *annotations, retained))
      return error;

    for (llvm::GlobalVariable *global : removedSupport) {
      global->removeDeadConstantUsers();
      if (global->use_empty() && global->hasLocalLinkage())
        global->eraseFromParent();
    }
    return llvm::Error::success();
  }

  static llvm::Error eraseUnreferencedTemporaryCandidates(
      llvm::ArrayRef<TemporaryCandidate> temporary) {
    if (temporary.empty())
      return llvm::Error::success();
    auto closure = collectTemporaryDependencyClosure(temporary);
    if (!closure)
      return closure.takeError();
    llvm::SmallPtrSet<llvm::GlobalValue *, 16> candidates(closure->begin(),
                                                          closure->end());
    llvm::SmallPtrSet<llvm::GlobalValue *, 8> original;
    llvm::DenseMap<llvm::GlobalValue *, bool> sourceRequired;
    for (const TemporaryCandidate &candidate : temporary)
      original.insert(candidate.function);
    for (const TemporaryCandidate &candidate : temporary)
      sourceRequired[candidate.function] = candidate.sourceRequired;

    llvm::SmallPtrSet<llvm::GlobalValue *, 16> dependent;
    for (llvm::GlobalValue *global : *closure)
      if (!original.contains(global))
        dependent.insert(global);
    if (!dependent.empty())
      if (llvm::Error error = rejectAnnotatedTemporaryDependencies(
              *temporary.front().function->getParent(), dependent))
        return error;

    llvm::DenseMap<llvm::GlobalValue *,
                   llvm::SmallVector<llvm::GlobalValue *, 4>>
        dependencies;
    llvm::SmallPtrSet<llvm::GlobalValue *, 16> roots;
    llvm::GlobalVariable *annotations =
        temporary.empty()
            ? nullptr
            : temporary.front().function->getParent()->getNamedGlobal(
                  "llvm.global.annotations");
    for (llvm::GlobalValue *global : *closure) {
      if (!global->isDiscardableIfUnused())
        roots.insert(global);
      llvm::SmallPtrSet<llvm::User *, 16> visited;
      for (llvm::User *user : global->users())
        classifyUse(*global, *user, candidates, annotations,
                    sourceRequired.lookup(global), dependencies, roots,
                    visited);
    }

    llvm::SmallPtrSet<llvm::GlobalValue *, 16> live;
    llvm::SmallVector<llvm::GlobalValue *, 16> worklist(roots.begin(),
                                                        roots.end());
    while (!worklist.empty()) {
      llvm::GlobalValue *global = worklist.pop_back_val();
      if (!live.insert(global).second)
        continue;
      llvm::append_range(worklist, dependencies[global]);
    }

    for (const TemporaryCandidate &candidate : temporary)
      if (!candidate.sourceRequired && live.contains(candidate.function))
        return candidateError(
            loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
            "a temporary candidate has an unsupported generated liveness "
            "root");

    llvm::SmallVector<llvm::GlobalValue *, 16> dead;
    llvm::SmallVector<llvm::Function *, 8> deadCandidates;
    for (llvm::GlobalValue *global : *closure) {
      if (live.contains(global))
        continue;
      dead.push_back(global);
      if (original.contains(global))
        deadCandidates.push_back(llvm::cast<llvm::Function>(global));
    }
    if (!deadCandidates.empty())
      if (llvm::Error error = eraseAnnotationsForDeadCandidates(
              *deadCandidates.front()->getParent(), deadCandidates))
        return error;
    llvm::SmallPtrSet<llvm::GlobalValue *, 16> deadSet(dead.begin(),
                                                       dead.end());
    llvm::Module &module = *temporary.front().function->getParent();
    for (llvm::GlobalValue *global : dead) {
      auto *object = llvm::dyn_cast<llvm::GlobalObject>(global);
      if (!object || !object->hasComdat())
        continue;
      for (llvm::GlobalObject &member : module.global_objects())
        if (member.getComdat() == object->getComdat() &&
            !deadSet.contains(&member))
          return candidateError(
              loom::raising::CandidateHintErrorKind::UnsupportedConstruct,
              "a temporary candidate has a live COMDAT member");
    }
    for (llvm::GlobalValue *global : dead) {
      if (auto *function = llvm::dyn_cast<llvm::Function>(global))
        function->dropAllReferences();
      else
        llvm::cast<llvm::GlobalVariable>(global)->dropAllReferences();
    }
    for (llvm::GlobalValue *global : dead) {
      global->removeDeadConstantUsers();
      if (!global->use_empty())
        return candidateError(loom::raising::CandidateHintErrorKind::
                                  ProjectionProofNotEstablished,
                              llvm::Twine("temporary candidate dependency '") +
                                  global->getName() +
                                  "' has a residual live use");
    }
    for (llvm::GlobalValue *global : dead) {
      if (auto *function = llvm::dyn_cast<llvm::Function>(global))
        function->eraseFromParent();
      else
        llvm::cast<llvm::GlobalVariable>(global)->eraseFromParent();
    }
    return llvm::Error::success();
  }
};

llvm::PassPluginLibraryInfo pluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoomCandidateProjection",
          LLVM_VERSION_STRING, [](llvm::PassBuilder &builder) {
            builder.registerPipelineStartEPCallback(
                [](llvm::ModulePassManager &manager, llvm::OptimizationLevel) {
                  manager.addPass(CandidateProjectionPass());
                });
            builder.registerOptimizerLastEPCallback(
                [](llvm::ModulePassManager &manager, llvm::OptimizationLevel,
                   llvm::ThinOrFullLTOPhase) {
                  manager.addPass(RemoveCandidateRetentionPass());
                });
          }};
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return pluginInfo();
}
