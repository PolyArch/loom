#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Raising/CandidateHints.h"
#include "Frontend/Raising/StructuredRaising.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
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

loom::raising::FunctionCandidateAnnotation sourceHint() {
  return {"candidate.c", {1, 1}, {2, 1}, {4, 2}};
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
      "mapped.c", {100, 1}, {1, 1}, {1, 24}};
  std::string remappedEncoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(remapped));
  if (!(take(loom::raising::decodeFunctionCandidateAnnotation(
            remappedEncoded)) == remapped))
    fail("candidate annotation rejected presumed source remapping");
}

void raisingProjectsOneTransientHint() {
  llvm::LLVMContext hintedContext;
  llvm::LLVMContext baselineContext;
  const auto hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto hinted =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithAnnotations(hintedContext, {encoded}, hint.sourceFile,
                                hint.targetBegin.line)));
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

void malformedAndDuplicateHintsFailClosed() {
  llvm::LLVMContext malformedContext;
  auto malformed =
      loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithAnnotations(malformedContext,
                                {"loom.candidate.function.1.0|broken"},
                                "candidate.c", 2));
  if (malformed)
    fail("raising accepted a malformed candidate annotation");
  std::string message = llvm::toString(malformed.takeError());
  if (!llvm::StringRef(message).contains("candidate_hint_invalid"))
    fail("malformed candidate annotation lost its typed diagnostic");

  llvm::LLVMContext duplicateContext;
  const auto hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto duplicate =
      loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithAnnotations(duplicateContext, {encoded, encoded},
                                hint.sourceFile, hint.targetBegin.line));
  if (duplicate)
    fail("raising accepted duplicate candidate annotations");
  message = llvm::toString(duplicate.takeError());
  if (!llvm::StringRef(message).contains("duplicate"))
    fail("duplicate candidate annotation lost its diagnostic");
}

void unrelatedAnnotationsRemainOwnedByLlvm() {
  llvm::LLVMContext context;
  const auto hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto raised =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithAnnotations(context, {encoded, "other.annotation"},
                                hint.sourceFile, hint.targetBegin.line)));
  if (raised.candidateHints.size() != 1)
    fail("unrelated annotation changed the candidate hint projection");
  if (!raised.artifact.module().lookupSymbol<mlir::LLVM::GlobalOp>(
          "llvm.global.annotations"))
    fail("raising consumed an unrelated LLVM annotation");
}

void preMappingRetainsSourceHintLineage() {
  llvm::LLVMContext context;
  const auto hint = sourceHint();
  std::string encoded =
      take(loom::raising::encodeFunctionCandidateAnnotation(hint));
  auto raised =
      take(loom::raising::raiseLlvmModuleToStructuredProgramWithProjection(
          moduleWithAnnotations(context, {encoded}, hint.sourceFile,
                                hint.targetBegin.line)));
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
  malformedAndDuplicateHintsFailClosed();
  unrelatedAnnotationsRemainOwnedByLlvm();
  preMappingRetainsSourceHintLineage();
  return EXIT_SUCCESS;
}
