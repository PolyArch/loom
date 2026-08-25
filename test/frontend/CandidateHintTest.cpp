#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Raising/CandidateHints.h"
#include "Frontend/Raising/StructuredRaising.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "candidateHint: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

llvm::GlobalVariable *stringGlobal(llvm::Module &module, llvm::StringRef value,
                                   llvm::StringRef name) {
  llvm::Constant *initializer =
      llvm::ConstantDataArray::getString(module.getContext(), value);
  auto *global = new llvm::GlobalVariable(module, initializer->getType(), true,
                                          llvm::GlobalValue::PrivateLinkage,
                                          initializer, name);
  global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  global->setSection("llvm.metadata");
  return global;
}

std::unique_ptr<llvm::Module>
moduleWithAnnotations(llvm::LLVMContext &context,
                      llvm::ArrayRef<std::string> annotations,
                      llvm::StringRef sourceFile, std::uint32_t sourceLine) {
  auto module = std::make_unique<llvm::Module>("candidate-hint", context);
  module->setDataLayout("e-p:64:64-i64:64-n8:16:32:64-S128");
  module->setTargetTriple(llvm::Triple("x86_64-unknown-linux-gnu"));
  auto *functionType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
  llvm::Function *function = llvm::Function::Create(
      functionType, llvm::GlobalValue::ExternalLinkage, "kernel", *module);
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(context, "entry", function);
  llvm::IRBuilder<>(entry).CreateRetVoid();

  if (annotations.empty())
    return module;
  llvm::Type *pointerType = llvm::PointerType::get(context, 0);
  llvm::StructType *entryType =
      llvm::StructType::get(pointerType, pointerType, pointerType,
                            llvm::Type::getInt32Ty(context), pointerType);
  llvm::GlobalVariable *file = stringGlobal(*module, sourceFile, ".hint.file");
  std::vector<llvm::Constant *> entries;
  entries.reserve(annotations.size());
  for (auto [index, annotation] : llvm::enumerate(annotations)) {
    llvm::GlobalVariable *text = stringGlobal(
        *module, annotation, ".hint.annotation." + std::to_string(index));
    entries.push_back(llvm::ConstantStruct::get(
        entryType, function, text, file,
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), sourceLine),
        llvm::ConstantPointerNull::get(llvm::PointerType::get(context, 0))));
  }
  llvm::ArrayType *arrayType = llvm::ArrayType::get(entryType, entries.size());
  auto *global = new llvm::GlobalVariable(
      *module, arrayType, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(arrayType, entries), "llvm.global.annotations");
  global->setSection("llvm.metadata");
  return module;
}

void attachFunctionCandidateMetadata(llvm::Module &module,
                                     llvm::StringRef encoded) {
  llvm::Function *function = module.getFunction("kernel");
  if (!function)
    fail("candidate fixture lost its function");
  llvm::LLVMContext &context = module.getContext();
  function->setMetadata(
      loom::raising::functionCandidateMetadataName,
      llvm::MDNode::get(context, {llvm::MDString::get(context, encoded)}));
}

void attachRawLoopMarker(llvm::Module &module, std::uint64_t marker) {
  llvm::Function *function = module.getFunction("kernel");
  if (!function)
    fail("loop marker fixture lost its function");
  llvm::LLVMContext &context = module.getContext();
  llvm::GlobalVariable *text = stringGlobal(
      module, loom::raising::encodeLoopCandidateMarker(marker), ".loop.marker");
  llvm::GlobalVariable *file =
      stringGlobal(module, "marker-only.c", ".loop.file");
  llvm::Type *i32 = llvm::Type::getInt32Ty(context);
  llvm::Type *pointer = llvm::PointerType::get(context, 0);
  llvm::Function *annotation = llvm::Intrinsic::getOrInsertDeclaration(
      &module, llvm::Intrinsic::annotation, {i32, pointer});
  llvm::IRBuilder<> builder(function->getEntryBlock().getTerminator());
  builder.CreateCall(annotation, {llvm::ConstantInt::get(i32, 0), text, file,
                                  llvm::ConstantInt::get(i32, 7)});
}

std::unique_ptr<llvm::Module> moduleWithProjectedLoopHint(
    llvm::LLVMContext &context,
    const std::optional<loom::raising::LoopCandidateAnnotation> &hint,
    bool weighted = false, bool bulkMemory = false) {
  auto module = std::make_unique<llvm::Module>("candidate-loop-hint", context);
  module->setDataLayout("e-p:64:64-i64:64-n8:16:32:64-S128");
  module->setTargetTriple(llvm::Triple("x86_64-unknown-linux-gnu"));
  auto *functionType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                              {llvm::PointerType::get(context, 0)}, false);
  llvm::Function *function = llvm::Function::Create(
      functionType, llvm::GlobalValue::ExternalLinkage, "loop_kernel", *module);
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(context, "entry", function);
  llvm::BasicBlock *header =
      llvm::BasicBlock::Create(context, "header", function);
  llvm::BasicBlock *body = llvm::BasicBlock::Create(context, "body", function);
  llvm::BasicBlock *latch =
      llvm::BasicBlock::Create(context, "latch", function);
  llvm::BasicBlock *exit = llvm::BasicBlock::Create(context, "exit", function);

  llvm::IRBuilder<> builder(entry);
  builder.CreateBr(header);

  builder.SetInsertPoint(header);
  llvm::PHINode *iteration = builder.CreatePHI(builder.getInt32Ty(), 2);
  iteration->addIncoming(builder.getInt32(0), entry);
  llvm::Value *condition =
      builder.CreateICmpSLT(iteration, builder.getInt32(4));
  auto *conditionBranch = builder.CreateCondBr(condition, body, exit);
  if (weighted) {
    llvm::Metadata *weights[] = {
        llvm::MDString::get(context, "branch_weights"),
        llvm::ConstantAsMetadata::get(builder.getInt32(4)),
        llvm::ConstantAsMetadata::get(builder.getInt32(1))};
    conditionBranch->setMetadata(llvm::LLVMContext::MD_prof,
                                 llvm::MDNode::get(context, weights));
  }

  builder.SetInsertPoint(body);
  builder.CreateStore(iteration, function->getArg(0));
  builder.CreateBr(latch);

  builder.SetInsertPoint(latch);
  llvm::Value *next = builder.CreateAdd(iteration, builder.getInt32(1));
  iteration->addIncoming(next, latch);
  auto *latchBranch = builder.CreateBr(header);

  builder.SetInsertPoint(exit);
  if (bulkMemory)
    builder.CreateMemSet(function->getArg(0), builder.getInt8(0), 4,
                         llvm::Align(1));
  builder.CreateRetVoid();

  if (!hint)
    return module;
  const std::string encoded =
      take(loom::raising::encodeLoopCandidateAnnotation(*hint));
  llvm::MDNode *property = llvm::MDNode::get(
      context,
      {llvm::MDString::get(context, loom::raising::loopCandidateMetadataName),
       llvm::MDString::get(context, encoded)});
  llvm::MDNode *loopID =
      llvm::MDNode::getDistinct(context, {nullptr, property});
  loopID->replaceOperandWith(0, loopID);
  latchBranch->setMetadata(llvm::LLVMContext::MD_loop, loopID);
  function->setMetadata(
      loom::raising::loopCandidateManifestMetadataName,
      llvm::MDNode::get(context, {llvm::MDString::get(context, encoded)}));
  return module;
}

loom::raising::FunctionCandidateAnnotation sourceHint() {
  return {"candidate.c", {2, 5}, {1, 1}, {2, 1}, {4, 2}};
}

void codecIsExact() {
  const loom::raising::FunctionCandidateAnnotation hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  if (!(take(loom::raising::decodeFunctionCandidateAnnotation(encoded)) ==
        hint))
    fail("candidate annotation did not round-trip");
  auto malformed =
      loom::raising::decodeFunctionCandidateAnnotation(encoded + "|unexpected");
  if (malformed)
    fail("candidate annotation accepted trailing fields");
  llvm::consumeError(malformed.takeError());

  const loom::raising::FunctionCandidateAnnotation remapped{
      "mapped.c", {1, 5}, {100, 1}, {1, 1}, {1, 24}};
  std::string remappedEncoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(remapped));
  if (!(take(loom::raising::decodeFunctionCandidateAnnotation(
            remappedEncoded)) == remapped))
    fail("candidate annotation rejected presumed source remapping");

  const loom::raising::LoopCandidateAnnotation loop{17,     "loop.c", {4, 5},
                                                    {6, 1}, {7, 3},   {9, 4}};
  std::string loopEncoded =
      take(loom::raising::encodeLoopCandidateAnnotation(loop));
  if (!(take(loom::raising::decodeLoopCandidateAnnotation(loopEncoded)) ==
        loop))
    fail("loop candidate annotation did not round-trip");
  std::string marker = loom::raising::encodeLoopCandidateMarker(loop.marker);
  if (take(loom::raising::decodeLoopCandidateMarker(marker)) != loop.marker)
    fail("loop candidate marker did not round-trip");
  auto malformedLoop =
      loom::raising::decodeLoopCandidateAnnotation(loopEncoded + "|extra");
  if (malformedLoop)
    fail("loop candidate annotation accepted trailing fields");
  bool typed = false;
  llvm::handleAllErrors(
      malformedLoop.takeError(),
      [&](const loom::raising::CandidateHintError &error) {
        typed = error.kind() ==
                loom::raising::CandidateHintErrorKind::InvalidEncoding;
      });
  if (!typed)
    fail("malformed loop candidate lost its typed diagnostic");
}

void raisingProjectsOneTransientHint() {
  llvm::LLVMContext hintedContext;
  llvm::LLVMContext baselineContext;
  const auto hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto hintedModule = moduleWithAnnotations(hintedContext, {}, hint.sourceFile,
                                            hint.targetBegin.line);
  attachFunctionCandidateMetadata(*hintedModule, encoded);
  auto hinted =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          std::move(hintedModule)));
  auto baseline =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithAnnotations(baselineContext, {}, hint.sourceFile,
                                hint.targetBegin.line)));

  if (hinted.artifact.identity() != baseline.artifact.identity() ||
      hinted.artifact.canonicalBytes().bytes() !=
          baseline.artifact.canonicalBytes().bytes())
    fail("nonbinding candidate hint changed Structured Program identity");
  if (hinted.candidateHints.size() != 1 ||
      hinted.candidateHints.front().sourceFile != hint.sourceFile ||
      hinted.candidateHints.front().pragma.line != hint.pragma.line ||
      hinted.candidateHints.front().pragma.column != hint.pragma.column ||
      hinted.candidateHints.front().targetBegin.line != hint.targetBegin.line ||
      hinted.candidateHints.front().targetBegin.column !=
          hint.targetBegin.column ||
      hinted.candidateHints.front().targetEnd.line != hint.targetEnd.line ||
      hinted.candidateHints.front().targetEnd.column != hint.targetEnd.column)
    fail("raising lost the candidate source range");
  auto view = take(hinted.artifact.view());
  auto target = take(view.resolve(hinted.candidateHints.front().target));
  auto function =
      llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(target.operation);
  if (!function || function.getSymName() != "kernel")
    fail("candidate hint did not resolve to the exact Structured function");
  if (target.operation->getAttr(
          loom::frontend::structuredCandidateHintAttrName))
    fail("candidate hint leaked into canonical Structured semantics");
  if (hinted.artifact.module().lookupSymbol<mlir::LLVM::GlobalOp>(
          "llvm.global.annotations"))
    fail("projected LLVM annotation global survived raising");
}

void raisingProjectsExactLoopHint() {
  const loom::raising::LoopCandidateAnnotation hint{
      17, "loop-candidate.c", {2, 5}, {3, 1}, {4, 3}, {8, 4}};
  llvm::LLVMContext hintedContext;
  llvm::LLVMContext baselineContext;
  auto hinted =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithProjectedLoopHint(hintedContext, hint, /*weighted=*/false,
                                      /*bulkMemory=*/true)));
  auto baseline =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithProjectedLoopHint(baselineContext, std::nullopt,
                                      /*weighted=*/false,
                                      /*bulkMemory=*/true)));
  if (hinted.artifact.identity() != baseline.artifact.identity() ||
      hinted.artifact.canonicalBytes().bytes() !=
          baseline.artifact.canonicalBytes().bytes())
    fail("nonbinding loop hint changed Structured Program identity");
  if (hinted.candidateHints.size() != 1)
    fail("raising did not project exactly one loop candidate");
  const auto &projected = hinted.candidateHints.front();
  if (projected.targetKind !=
          loom::frontend::StructuredCandidateHintTargetKind::Loop ||
      projected.sourceFile != hint.sourceFile ||
      projected.pragma.line != hint.pragma.line ||
      projected.pragma.column != hint.pragma.column ||
      projected.targetBegin.line != hint.targetBegin.line ||
      projected.targetBegin.column != hint.targetBegin.column ||
      projected.targetEnd.line != hint.targetEnd.line ||
      projected.targetEnd.column != hint.targetEnd.column ||
      projected.target.parent != hinted.artifact.identity())
    fail("raising changed exact loop candidate lineage");
  auto view = take(hinted.artifact.view());
  auto target = take(view.resolve(projected.target));
  if (!llvm::isa_and_nonnull<mlir::scf::WhileOp, mlir::scf::ForOp>(
          target.operation))
    fail("loop candidate did not resolve to one exact Structured loop");
  if (target.operation->getAttr(
          loom::frontend::structuredCandidateLoopHintAttrName))
    fail("loop candidate hint leaked into canonical Structured semantics");
}

void unstructuredCandidateLoopFailsClosed() {
  const loom::raising::LoopCandidateAnnotation hint{
      17, "weighted-loop.c", {2, 5}, {3, 1}, {4, 3}, {8, 4}};
  llvm::LLVMContext context;
  auto raised = loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
      moduleWithProjectedLoopHint(context, hint, /*weighted=*/true));
  if (raised)
    fail("raising published an unstructured candidate loop target");
  bool typed = false;
  std::string unexpected;
  llvm::handleAllErrors(
      raised.takeError(),
      [&](const loom::raising::CandidateHintError &error) {
        typed = error.kind() == loom::raising::CandidateHintErrorKind::
                                    ProjectionProofNotEstablished;
      },
      [&](const llvm::ErrorInfoBase &error) { unexpected = error.message(); });
  if (!typed)
    fail("unstructured candidate loop lost its typed refusal: " + unexpected);
}

void malformedAndDuplicateHintsFailClosed() {
  llvm::LLVMContext malformedContext;
  auto malformedModule = moduleWithAnnotations(malformedContext, {},
                                               "candidate.c", /*sourceLine=*/2);
  attachFunctionCandidateMetadata(*malformedModule,
                                  "loom.candidate.function.2.0|broken");
  auto malformed =
      loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          std::move(malformedModule));
  if (malformed)
    fail("raising accepted a malformed candidate annotation");
  std::string message = llvm::toString(malformed.takeError());
  if (!llvm::StringRef(message).contains("candidate_hint_invalid"))
    fail("malformed candidate annotation lost its typed diagnostic");

  llvm::LLVMContext duplicateContext;
  const loom::raising::LoopCandidateAnnotation hint{
      17, "duplicate-loop.c", {2, 5}, {3, 1}, {4, 3}, {8, 4}};
  auto duplicateModule = moduleWithProjectedLoopHint(duplicateContext, hint);
  llvm::Function *carrier = duplicateModule->getFunction("loop_kernel");
  llvm::MDNode *manifest =
      carrier ? carrier->getMetadata(
                    loom::raising::loopCandidateManifestMetadataName)
              : nullptr;
  if (!manifest || manifest->getNumOperands() != 1)
    fail("duplicate candidate fixture lost its loop manifest");
  llvm::Metadata *encoded = manifest->getOperand(0).get();
  carrier->setMetadata(loom::raising::loopCandidateManifestMetadataName,
                       llvm::MDNode::get(duplicateContext, {encoded, encoded}));
  auto duplicate =
      loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          std::move(duplicateModule));
  if (duplicate)
    fail("raising accepted duplicate candidate annotations");
  message = llvm::toString(duplicate.takeError());
  if (!llvm::StringRef(message).contains("duplicate"))
    fail("duplicate candidate annotation lost its diagnostic");
}

void unprojectedCandidateFailsClosed() {
  const auto hint = sourceHint();
  const std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto requireUnsupported = [](auto raised, llvm::StringRef description) {
    if (raised)
      fail(llvm::Twine("raising accepted ") + description);
    bool typed = false;
    llvm::handleAllErrors(
        raised.takeError(),
        [&](const loom::raising::CandidateHintError &error) {
          typed = error.kind() ==
                  loom::raising::CandidateHintErrorKind::UnsupportedConstruct;
        });
    if (!typed)
      fail(llvm::Twine(description) + " lost its typed refusal");
  };

  llvm::LLVMContext annotationContext;
  requireUnsupported(
      loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithAnnotations(annotationContext, {encoded}, hint.sourceFile,
                                hint.targetBegin.line)),
      "a candidate annotation without LLVM projection");

  llvm::LLVMContext markerContext;
  auto markerModule =
      moduleWithAnnotations(markerContext, {}, "marker-only.c", 7);
  attachRawLoopMarker(*markerModule, 17);
  requireUnsupported(
      loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          std::move(markerModule)),
      "a candidate loop marker without LLVM projection");
}

void unrelatedAnnotationsRemainOwnedByLlvm() {
  llvm::LLVMContext context;
  const auto hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto module = moduleWithAnnotations(context, {"other.annotation"},
                                      hint.sourceFile, hint.targetBegin.line);
  attachFunctionCandidateMetadata(*module, encoded);
  auto raised =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          std::move(module)));
  if (raised.candidateHints.size() != 1)
    fail("unrelated annotation changed the candidate hint projection");
  if (!raised.artifact.module().lookupSymbol<mlir::LLVM::GlobalOp>(
          "llvm.global.annotations"))
    fail("raising consumed an unrelated LLVM annotation");
}

void temporaryRetentionIsExactlyOwned() {
  llvm::LLVMContext context;
  llvm::Module module("candidate-retention", context);
  auto *functionType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
  auto createFunction = [&](llvm::StringRef name) {
    llvm::Function *function = llvm::Function::Create(
        functionType, llvm::GlobalValue::InternalLinkage, name, module);
    llvm::BasicBlock *entry =
        llvm::BasicBlock::Create(context, "entry", function);
    llvm::IRBuilder<>(entry).CreateRetVoid();
    return function;
  };
  llvm::Function *candidate = createFunction("candidate");
  llvm::Function *unrelated = createFunction("unrelated");
  candidate->setMetadata(loom::raising::candidateTemporaryRetentionMetadataName,
                         llvm::MDNode::get(context, {}));
  llvm::Type *pointerType = llvm::PointerType::get(context, 0);
  llvm::ArrayType *usedType = llvm::ArrayType::get(pointerType, 2);
  auto *used = new llvm::GlobalVariable(
      module, usedType, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(usedType, {candidate, unrelated}),
      "llvm.compiler.used");
  used->setSection("llvm.metadata");

  if (llvm::Error error =
          loom::raising::removeCandidateTemporaryRetention(module))
    fail(llvm::toString(std::move(error)));
  if (candidate->getMetadata(
          loom::raising::candidateTemporaryRetentionMetadataName))
    fail("candidate temporary retention metadata survived release");
  used = module.getNamedGlobal("llvm.compiler.used");
  auto *initializer =
      used ? llvm::dyn_cast<llvm::ConstantArray>(used->getInitializer())
           : nullptr;
  if (!initializer || initializer->getNumOperands() != 1 ||
      initializer->getOperand(0)->stripPointerCasts() != unrelated)
    fail("candidate retention release changed an unrelated used entry");

  candidate->setMetadata(loom::raising::candidateTemporaryRetentionMetadataName,
                         llvm::MDNode::get(context, {}));
  used->eraseFromParent();
  llvm::Error missingOwner =
      loom::raising::removeCandidateTemporaryRetention(module);
  if (!missingOwner)
    fail("candidate retention accepted a missing owner");
  bool typed = false;
  llvm::handleAllErrors(std::move(missingOwner),
                        [&](const loom::raising::CandidateHintError &error) {
                          typed = error.kind() ==
                                  loom::raising::CandidateHintErrorKind::
                                      ProjectionProofNotEstablished;
                        });
  if (!typed)
    fail("candidate retention lost its typed missing-owner refusal");
}

void preMappingRetainsSourceHintLineage() {
  llvm::LLVMContext context;
  const auto hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto module = moduleWithAnnotations(context, {}, hint.sourceFile,
                                      hint.targetBegin.line);
  attachFunctionCandidateMetadata(*module, encoded);
  auto raised =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          std::move(module)));
  const loom::ArtifactIdentity sourceIdentity = raised.artifact.identity();
  const loom::ArtifactRootReference fabric{
      "test.fabric", {1, 0}, sourceIdentity};
  loom::frontend::StructuredCompilation source{
      fabric,
      {},
      std::move(raised.artifact),
      std::move(raised.sourceProvenance),
      std::move(raised.candidateHints)};
  auto preMapping = take(loom::frontend::lowerStructuredCompilationToPreMapping(
      std::move(source)));
  if (preMapping.candidateHints.size() != 1 ||
      preMapping.candidateHints.front().target.parent != sourceIdentity)
    fail("pre-Mapping lowering dropped source candidate lineage");
}

} // namespace

int main() {
  codecIsExact();
  raisingProjectsOneTransientHint();
  raisingProjectsExactLoopHint();
  unstructuredCandidateLoopFailsClosed();
  malformedAndDuplicateHintsFailClosed();
  unprojectedCandidateFailsClosed();
  unrelatedAnnotationsRemainOwnedByLlvm();
  temporaryRetentionIsExactlyOwned();
  preMappingRetainsSourceHintLineage();
  return EXIT_SUCCESS;
}
