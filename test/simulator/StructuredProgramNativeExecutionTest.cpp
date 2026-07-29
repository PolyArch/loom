#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/IR/LoomDialect.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
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

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                    mlir::LLVM::LLVMDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

struct SourceProgram {
  loom::frontend::StructuredProgramCandidate candidate;
  llvm::DataLayout layout;
};

SourceProgram sourceProgram(llvm::StringRef test) {
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail(test, "cannot initialize the native target");
  auto target = take(test, llvm::orc::JITTargetMachineBuilder::detectHost());
  llvm::DataLayout layout = take(test, target.getDefaultDataLayoutForTarget());
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%value: i32, %written: !llvm.ptr, %observed: !llvm.ptr) -> i32 {
    llvm.store %value, %written : i32, !llvm.ptr
    %zero = llvm.mlir.constant(0 : i32) : i32
    %nonzero = llvm.icmp "ne" %value, %zero : i32
    llvm.cond_br %nonzero, ^load, ^empty
  ^load:
    %loaded = llvm.load %observed : !llvm.ptr -> i32
    llvm.br ^exit(%loaded : i32)
  ^empty:
    llvm.br ^exit(%zero : i32)
  ^exit(%returned: i32):
    llvm.return %returned : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "cannot parse the source Structured Program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), layout.getStringRepresentation()));
  return {take(test, loom::frontend::finalizeStructuredProgram(module.get())),
          std::move(layout)};
}

loom::frontend::StructuredProgramCandidate
selectedProgram(llvm::StringRef test, const llvm::DataLayout &layout) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %value: i32, %written: !llvm.ptr) ctrl (%ctrl: none) {
    "loom.spatial_region"(%value, %written)
        <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%payload: i32, %target: !llvm.ptr):
        llvm.store %payload, %target : i32, !llvm.ptr
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "selected_graph", source_maps = []} :
        (i32, !llvm.ptr) -> ()
    dataflow.thread.yield
  }

  llvm.func @kernel(%value: i32, %written: !llvm.ptr,
                    %observed: !llvm.ptr) -> i32 {
    %token = dataflow.thread.launch @selected(%value, %written) :
        (i32, !llvm.ptr) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    %loaded = llvm.load %observed : !llvm.ptr -> i32
    llvm.return %loaded : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "cannot parse the selected Structured Program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), layout.getStringRepresentation()));
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

loom::sim::CanonicalValueSequence definedI32(std::uint32_t value) {
  loom::sim::CanonicalValueSequence sequence;
  sequence.tokenCount = 1;
  sequence.lanes.push_back(
      loom::sim::SemanticLane::defined(llvm::APInt(32, value)));
  return sequence;
}

loom::sim::CanonicalValueSequence poisonI32() {
  loom::sim::CanonicalValueSequence sequence;
  sequence.tokenCount = 1;
  sequence.lanes.push_back(loom::sim::SemanticLane::poison());
  return sequence;
}

loom::sim::StructuredProgramSimulationWorkload
makeWorkload(const loom::frontend::StructuredEntityRef &entry,
             loom::sim::CanonicalValueSequence value) {
  loom::sim::StructuredProgramSimulationWorkload workload{entry};
  workload.argumentPlan = {std::move(value),
                           loom::sim::StructuredRuntimeMemoryInput{},
                           loom::sim::StructuredRuntimeMemoryInput{}};
  workload.observableContract.returnValue = true;
  workload.observableContract.memories = {
      {loom::sim::EntryPointerArgumentTarget{1},
       loom::sim::MemoryObservationForm::DiffFromRuntimeInput},
      {loom::sim::EntryPointerArgumentTarget{2},
       loom::sim::MemoryObservationForm::FullState}};
  return workload;
}

loom::sim::StructuredProgramSimulationRuntimeInputDraft
makeRuntimeInput(const loom::ArtifactIdentity &workloadIdentity) {
  loom::sim::StructuredProgramSimulationRuntimeInputDraft input{
      workloadIdentity};
  input.memoryObjects.push_back(
      loom::sim::RuntimeMemoryObject{std::vector<loom::sim::SemanticMemoryByte>(
          4, {loom::sim::SemanticState::Defined, 0})});
  input.pointerBindings = {loom::sim::StructuredPointerBindingDraft{1, 0, 0},
                           loom::sim::StructuredPointerBindingDraft{2, 0, 0}};
  return input;
}

std::array<std::uint8_t, 4> bytesOf(std::uint32_t value, bool littleEndian) {
  std::array<std::uint8_t, 4> result{};
  for (std::uint32_t index = 0; index < result.size(); ++index) {
    const std::uint32_t addressed = littleEndian ? index : 3 - index;
    result[index] = static_cast<std::uint8_t>(value >> (addressed * 8));
  }
  return result;
}

void exactEntryPreservesAliasingAndObservations() {
  const char *test = __func__;
  SourceProgram source = sourceProgram(test);
  auto view = take(test, source.candidate.view());
  auto workload =
      take(test, loom::sim::finalizeSimulationWorkload(
                     makeWorkload(entryRef(test, view), definedI32(0x12345678)),
                     view));
  auto input =
      take(test, loom::sim::finalizeSimulationRuntimeInput(
                     makeRuntimeInput(workload.identity()), workload, view));

  loom::sim::NativeStructuredProgramObservations execution =
      take(test, loom::sim::executeNativeStructuredProgram(source.candidate,
                                                           workload, input));
  require(test, execution.returnValue.has_value(),
          "the selected return value was not observed");
  require(test,
          execution.returnValue->tokenCount == 1 &&
              execution.returnValue->lanes.size() == 1 &&
              execution.returnValue->lanes[0].state ==
                  loom::sim::SemanticState::Defined &&
              execution.returnValue->lanes[0].bits ==
                  llvm::APInt(32, 0x12345678),
          "the aliased read did not observe the preceding write");
  require(test, execution.memories.size() == 2,
          "memory observations do not align with the workload contract");
  require(test, execution.blockActivations.size() == 4,
          "block activation projection is not total");
  std::size_t activeBlocks = 0;
  std::size_t inactiveBlocks = 0;
  for (const auto &activation : execution.blockActivations) {
    require(test,
            activation.block.parent == source.candidate.identity() &&
                activation.block.kind ==
                    loom::frontend::StructuredEntityKind::Block,
            "block activation has a foreign or mistyped reference");
    activeBlocks += activation.activations == 1;
    inactiveBlocks += activation.activations == 0;
  }
  require(test, activeBlocks == 3 && inactiveBlocks == 1,
          "block activation counts do not preserve the executed path");

  const auto expected = bytesOf(0x12345678, source.layout.isLittleEndian());
  const auto *diff =
      std::get_if<loom::sim::DiffMemoryObservation>(&execution.memories[0]);
  require(test,
          diff && diff->byteCount == 4 && diff->runs.size() == 1 &&
              diff->runs[0].byteOffset == 0 &&
              diff->runs[0].changedBytes.size() == expected.size(),
          "the diff observation is not one maximal changed run");
  for (std::size_t index = 0; index < expected.size(); ++index)
    require(test,
            diff->runs[0].changedBytes[index].state ==
                    loom::sim::SemanticState::Defined &&
                diff->runs[0].changedBytes[index].value == expected[index],
            "the diff observation changed the target byte order");

  const auto *full =
      std::get_if<loom::sim::FullMemoryObservation>(&execution.memories[1]);
  require(test, full && full->bytes.size() == expected.size(),
          "the full observation does not cover the backing object");
  for (std::size_t index = 0; index < expected.size(); ++index)
    require(test,
            full->bytes[index].state == loom::sim::SemanticState::Defined &&
                full->bytes[index].value == expected[index],
            "aliased observables disagree on final object bytes");
}

void nonDefinedInputsFailClosed() {
  const char *test = __func__;
  SourceProgram source = sourceProgram(test);
  auto view = take(test, source.candidate.view());
  auto workload =
      take(test, loom::sim::finalizeSimulationWorkload(
                     makeWorkload(entryRef(test, view), poisonI32()), view));
  auto input =
      take(test, loom::sim::finalizeSimulationRuntimeInput(
                     makeRuntimeInput(workload.identity()), workload, view));
  auto execution = loom::sim::executeNativeStructuredProgram(source.candidate,
                                                             workload, input);
  require(test, !execution, "the native provider concretized Poison input");
  const std::string error = llvm::toString(execution.takeError());
  require(test, error.find("_unsupported:") != std::string::npos,
          "unsupported semantic input used the wrong failure class");
}

void selectedOwnershipCarriersExecuteAsWholeProgram() {
  const char *test = __func__;
  SourceProgram source = sourceProgram(test);
  auto sourceView = take(test, source.candidate.view());
  auto workload = take(
      test, loom::sim::finalizeSimulationWorkload(
                makeWorkload(entryRef(test, sourceView), definedI32(0x12345678)),
                sourceView));
  auto input = take(test, loom::sim::finalizeSimulationRuntimeInput(
                              makeRuntimeInput(workload.identity()), workload,
                              sourceView));
  auto selected = selectedProgram(test, source.layout);

  const auto reference = take(
      test, loom::sim::executeNativeStructuredProgram(source.candidate,
                                                       workload, input));
  const auto candidate = take(
      test, loom::sim::executeSelectedStructuredProgram(
                selected, source.candidate, workload, input));
  require(test,
          loom::sim::haveEquivalentFunctionalObservations(reference,
                                                          candidate),
          "selected ownership carriers changed whole-program semantics");
  require(test, candidate.blockActivations.empty(),
          "selected candidate execution invented a source coverage profile");
}

} // namespace

int main() {
  exactEntryPreservesAliasingAndObservations();
  nonDefinedInputsFailClosed();
  selectedOwnershipCarriersExecuteAsWholeProgram();
  llvm::outs() << "structured program native execution anchors passed\n";
  return EXIT_SUCCESS;
}
