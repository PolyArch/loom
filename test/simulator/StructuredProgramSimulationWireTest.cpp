#include "Common/ArtifactStore.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

bool rejected(llvm::Expected<loom::sim::CanonicalSimulationWorkload> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

bool rejected(
    llvm::Expected<loom::sim::CanonicalSimulationRuntimeInput> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<mlir::LLVM::LLVMDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

loom::frontend::StructuredProgramCandidate sourceProgram(llvm::StringRef test) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%n: i32, %input: !llvm.ptr, %output: !llvm.ptr) -> i32 {
    llvm.return %n : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "cannot parse the source Structured Program");
  return take(test, loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate
differentSourceProgram(llvm::StringRef test) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%n: i32, %input: !llvm.ptr, %output: !llvm.ptr) -> i32 {
    %one = llvm.mlir.constant(1 : i32) : i32
    %result = llvm.add %n, %one : i32
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "cannot parse the distinct Structured Program");
  return take(test, loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredEntityRef
entryRef(llvm::StringRef test,
         const loom::frontend::StructuredProgramCandidateView &view) {
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getName() == "kernel")
      return entity.reference;
  }
  fail(test, "cannot find the exact entry reference");
}

loom::sim::CanonicalValueSequence fixedI32(std::uint64_t value) {
  loom::sim::CanonicalValueSequence sequence;
  sequence.tokenCount = 1;
  sequence.lanes.push_back(
      loom::sim::SemanticLane::defined(llvm::APInt(32, value)));
  return sequence;
}

loom::sim::StructuredProgramSimulationWorkload
makeWorkload(const loom::frontend::StructuredEntityRef &entry,
             std::uint64_t value) {
  loom::sim::StructuredProgramSimulationWorkload workload{entry};
  workload.argumentPlan = {fixedI32(value),
                           loom::sim::StructuredRuntimeMemoryInput{},
                           loom::sim::StructuredRuntimeMemoryInput{}};
  workload.observableContract.returnValue = true;
  workload.observableContract.memories.push_back(
      {loom::sim::EntryPointerArgumentTarget{1},
       loom::sim::MemoryObservationForm::DiffFromRuntimeInput});
  return workload;
}

void structuredRootRoundTripsAgainstExactSource() {
  const char *test = __func__;
  loom::frontend::StructuredProgramCandidate source = sourceProgram(test);
  auto view = take(test, source.view());
  auto model = makeWorkload(entryRef(test, view), 7);

  loom::sim::CanonicalSimulationWorkload workload =
      take(test, loom::sim::finalizeSimulationWorkload(model, view));
  require(test,
          workload.kind() ==
              loom::sim::SimulationWorkloadKind::StructuredProgram,
          "finalization changed the workload root kind");
  require(test, workload.structuredProgram() != nullptr,
          "the canonical workload lost its Structured Program model");
  auto bytes = workload.canonicalBytes().bytes();
  require(test,
          bytes.size() >= 4 && bytes[0] == 0 && bytes[1] == 0 &&
              bytes[2] == 0 && bytes[3] == 2,
          "the appended Structured Program root does not use tag two");

  loom::sim::CanonicalSimulationWorkload imported =
      take(test, loom::sim::importSimulationWorkload(bytes, view,
                                                     workload.identity()));
  require(test, imported.identity() == workload.identity(),
          "strict workload import changed identity");

  auto wrongWidth = model;
  wrongWidth.argumentPlan[0] = loom::sim::CanonicalValueSequence{
      1, {loom::sim::SemanticLane::defined(llvm::APInt(16, 7))}};
  require(test,
          rejected(loom::sim::finalizeSimulationWorkload(wrongWidth, view)),
          "a copied value width overrode the exact entry ABI");

  loom::frontend::StructuredProgramCandidate foreign =
      differentSourceProgram(test);
  auto foreignView = take(test, foreign.view());
  auto foreignEntry = entryRef(test, foreignView);
  foreignEntry.parent = source.identity();
  auto wrongOwner = makeWorkload(foreignEntry, 7);
  require(
      test,
      rejected(loom::sim::finalizeSimulationWorkload(wrongOwner, foreignView)),
      "a foreign entry reference was accepted");
}

void runtimeInputBindsExactWorkloadAndPointerObject() {
  const char *test = __func__;
  loom::frontend::StructuredProgramCandidate source = sourceProgram(test);
  auto view = take(test, source.view());
  auto entry = entryRef(test, view);
  loom::sim::CanonicalSimulationWorkload workload =
      take(test,
           loom::sim::finalizeSimulationWorkload(makeWorkload(entry, 7), view));

  loom::sim::StructuredProgramSimulationRuntimeInputDraft draft{
      workload.identity()};
  draft.memoryObjects.push_back(
      loom::sim::RuntimeMemoryObject{std::vector<loom::sim::SemanticMemoryByte>(
          32, loom::sim::SemanticMemoryByte{loom::sim::SemanticState::Defined,
                                            0})});
  draft.pointerBindings.push_back(
      loom::sim::StructuredPointerBindingDraft{1, 0, 4});
  draft.pointerBindings.push_back(
      loom::sim::StructuredPointerBindingDraft{2, 0, 8});

  loom::sim::CanonicalSimulationRuntimeInput input = take(
      test, loom::sim::finalizeSimulationRuntimeInput(draft, workload, view));
  require(test,
          input.kind() ==
                  loom::sim::SimulationWorkloadKind::StructuredProgram &&
              input.structuredProgram() != nullptr,
          "the canonical runtime input lost its Structured Program model");
  require(
      test,
      input.structuredProgram()->memoryObjects.size() == 1 &&
          input.structuredProgram()->pointerBindings.size() == 2 &&
          input.structuredProgram()->pointerBindings[0].binding.objectOrdinal ==
              input.structuredProgram()
                  ->pointerBindings[1]
                  .binding.objectOrdinal,
      "shared pointer inputs did not preserve one aliasing object");
  auto imported = take(test, loom::sim::importSimulationRuntimeInput(
                                 input.canonicalBytes().bytes(), workload, view,
                                 input.identity()));
  require(test, imported.identity() == input.identity(),
          "strict runtime-input import changed identity");

  auto otherWorkload =
      take(test,
           loom::sim::finalizeSimulationWorkload(makeWorkload(entry, 8), view));
  draft.workloadIdentity = otherWorkload.identity();
  require(test,
          rejected(
              loom::sim::finalizeSimulationRuntimeInput(draft, workload, view)),
          "runtime input accepted a different workload identity");

  draft.workloadIdentity = workload.identity();
  draft.pointerBindings.clear();
  require(test,
          rejected(
              loom::sim::finalizeSimulationRuntimeInput(draft, workload, view)),
          "runtime input accepted a missing pointer binding");
}

void storedInputsRecoverTheirExactStructuredOwner() {
  const char *test = __func__;
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-simulation-store", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  loom::frontend::StructuredProgramCandidate source = sourceProgram(test);
  auto sourceReference =
      take(test, loom::frontend::publishStructuredProgram(source, store));
  auto view = take(test, source.view());
  loom::sim::CanonicalSimulationWorkload workload =
      take(test, loom::sim::finalizeSimulationWorkload(
                     makeWorkload(entryRef(test, view), 7), view));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft draft{
      workload.identity()};
  draft.memoryObjects.push_back(
      loom::sim::RuntimeMemoryObject{std::vector<loom::sim::SemanticMemoryByte>(
          32, loom::sim::SemanticMemoryByte{loom::sim::SemanticState::Defined,
                                            0})});
  draft.pointerBindings = {loom::sim::StructuredPointerBindingDraft{1, 0, 4},
                           loom::sim::StructuredPointerBindingDraft{2, 0, 8}};
  loom::sim::CanonicalSimulationRuntimeInput input = take(
      test, loom::sim::finalizeSimulationRuntimeInput(draft, workload, view));

  auto workloadReference =
      take(test, loom::sim::publishSimulationWorkload(workload, store));
  auto inputReference =
      take(test, loom::sim::publishSimulationRuntimeInput(input, store));
  auto imported = take(test, loom::sim::importStructuredProgramSimulationInputs(
                                 workloadReference, inputReference, store));
  require(test,
          imported.structuredProgram.identity() == sourceReference.artifact &&
              imported.workload.identity() == workload.identity() &&
              imported.runtimeInput.identity() == input.identity(),
          "stored simulation inputs did not recover their exact owners");

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail(test, "cannot remove artifact store: " + error.message());
}

} // namespace

int main() {
  structuredRootRoundTripsAgainstExactSource();
  runtimeInputBindsExactWorkloadAndPointerObject();
  storedInputsRecoverTheirExactStructuredOwner();
  llvm::outs() << "structured program simulation wire anchors passed\n";
  return EXIT_SUCCESS;
}
