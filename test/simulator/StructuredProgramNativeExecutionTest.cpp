#include "Dataflow/IR/DataflowDialect.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Runtime/OrderedChannelABI.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include <optional>
#include <string>
#include <system_error>
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
                    mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                    mlir::math::MathDialect, mlir::scf::SCFDialect>();
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
        <{operandSegmentSizes = array<i32: 2, 0, 0, 0>,
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

loom::frontend::StructuredProgramCandidate
selectedForallProgram(llvm::StringRef test, const llvm::DataLayout &layout) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %value: i32, %written: !llvm.ptr) ctrl (%ctrl: none) {
    "loom.spatial_region"(%value, %written)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%payload: i32, %target: !llvm.ptr):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.forall (%i) = (%c0) to (%c1) step (%c1) {
          llvm.store %payload, %target : i32, !llvm.ptr
          scf.forall.in_parallel {}
        }
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
    fail(test, "cannot parse the selected forall Structured Program");
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

void typedCosineUsesCanonicalNativeSemantics() {
  const char *test = __func__;
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail(test, "cannot initialize the native target");
  auto target = take(test, llvm::orc::JITTargetMachineBuilder::detectHost());
  llvm::DataLayout layout = take(test, target.getDefaultDataLayoutForTarget());
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel() -> f32 {
    %input = arith.constant 0x3E3D4952 : f32
    %result = math.cos %input : f32
    llvm.return %result : f32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "cannot parse the typed cosine program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), layout.getStringRepresentation()));

  auto program =
      take(test, loom::frontend::finalizeStructuredProgram(module.get()));
  auto view = take(test, program.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      entryRef(test, view)};
  workloadDraft.observableContract.returnValue = true;
  auto workload =
      take(test, loom::sim::finalizeSimulationWorkload(workloadDraft, view));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  auto input = take(test, loom::sim::finalizeSimulationRuntimeInput(
                              inputDraft, workload, view));

  auto execution =
      take(test,
           loom::sim::executeNativeStructuredProgram(program, workload, input));
  require(test,
          execution.returnValue && execution.returnValue->tokenCount == 1 &&
              execution.returnValue->lanes.size() == 1 &&
              execution.returnValue->lanes.front().state ==
                  loom::sim::SemanticState::Defined &&
              execution.returnValue->lanes.front().bits ==
                  llvm::APInt(32, 0x3f7ba384),
          "typed cosine used the ambient host libm result");
}

void selectedOwnershipCarriersExecuteAsWholeProgram() {
  const char *test = __func__;
  SourceProgram source = sourceProgram(test);
  auto sourceView = take(test, source.candidate.view());
  auto workload =
      take(test,
           loom::sim::finalizeSimulationWorkload(
               makeWorkload(entryRef(test, sourceView), definedI32(0x12345678)),
               sourceView));
  auto input = take(
      test, loom::sim::finalizeSimulationRuntimeInput(
                makeRuntimeInput(workload.identity()), workload, sourceView));
  auto selected = selectedProgram(test, source.layout);

  const auto reference = take(test, loom::sim::executeNativeStructuredProgram(
                                        source.candidate, workload, input));
  const auto candidate =
      take(test, loom::sim::executeSelectedStructuredProgram(
                     selected, source.candidate, workload, input));
  require(test,
          loom::sim::haveEquivalentFunctionalObservations(reference, candidate),
          "selected ownership carriers changed whole-program semantics");
  require(test, candidate.blockActivations.empty(),
          "selected candidate execution invented a source coverage profile");
}

void orderedLogicalChannelGenerationsCompleteMulticastSequences() {
  const char *test = __func__;
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail(test, "cannot initialize the native target");
  auto target = take(test, llvm::orc::JITTargetMachineBuilder::detectHost());
  llvm::DataLayout layout = take(test, target.getDefaultDataLayoutForTarget());

  auto parse = [&](llvm::StringRef text, llvm::StringRef description) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context());
    if (!module)
      fail(test, ("cannot parse the " + description).str());
    module->getOperation()->setAttr(
        "llvm.target_triple",
        mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
    module->getOperation()->setAttr(
        "llvm.data_layout",
        mlir::StringAttr::get(&context(), layout.getStringRepresentation()));
    return take(test, loom::frontend::finalizeStructuredProgram(module.get()));
  };

  auto source = parse(R"mlir(
module {
  llvm.func @kernel() -> i32 {
    %expected = llvm.mlir.constant(36 : i32) : i32
    llvm.return %expected : i32
  }
}
)mlir",
                      "logical-channel source program");
  auto selected = parse(R"mlir(
module {
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %first: i32, %second: i32)
      ctrl (%ctrl: none) {
    dataflow.channel.send %channel, %first : !dataflow.channel<i32>
    dataflow.channel.send %channel, %second : !dataflow.channel<i32>
    dataflow.thread.yield
  }

  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %output: !llvm.ptr)
      ctrl (%ctrl: none) {
    %first = dataflow.channel.receive %channel : !dataflow.channel<i32>
    %second = dataflow.channel.receive %channel : !dataflow.channel<i32>
    %sum = llvm.add %first, %second : i32
    llvm.store %sum, %output : i32, !llvm.ptr
    dataflow.thread.yield
  }

  llvm.func @channel_epoch(%first: i32, %second: i32,
                           %left: !llvm.ptr, %right: !llvm.ptr) {
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    %producer = dataflow.thread.launch @producer(%channel, %first, %second)
        : (!dataflow.channel<i32>, i32, i32) -> !dataflow.thread_token
    dataflow.thread.wait %producer : !dataflow.thread_token
    %consumer0 = dataflow.thread.launch @consumer(%channel, %left)
        : (!dataflow.channel<i32>, !llvm.ptr) -> !dataflow.thread_token
    dataflow.thread.wait %consumer0 : !dataflow.thread_token
    %consumer1 = dataflow.thread.launch @consumer(%channel, %right)
        : (!dataflow.channel<i32>, !llvm.ptr) -> !dataflow.thread_token
    dataflow.thread.wait %consumer1 : !dataflow.thread_token
    llvm.return
  }

  llvm.func @kernel() -> i32 {
    %one = llvm.mlir.constant(1 : i64) : i64
    %left = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %right = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %seven = llvm.mlir.constant(7 : i32) : i32
    %eleven = llvm.mlir.constant(11 : i32) : i32
    llvm.call @channel_epoch(%seven, %eleven, %left, %right)
        : (i32, i32, !llvm.ptr, !llvm.ptr) -> ()
    %thirteen = llvm.mlir.constant(13 : i32) : i32
    %five = llvm.mlir.constant(5 : i32) : i32
    llvm.call @channel_epoch(%thirteen, %five, %left, %right)
        : (i32, i32, !llvm.ptr, !llvm.ptr) -> ()
    %left_value = llvm.load %left : !llvm.ptr -> i32
    %right_value = llvm.load %right : !llvm.ptr -> i32
    %result = llvm.add %left_value, %right_value : i32
    llvm.return %result : i32
  }
}
)mlir",
                        "logical-channel selected program");

  auto sourceView = take(test, source.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      entryRef(test, sourceView)};
  workloadDraft.observableContract.returnValue = true;
  auto workload = take(
      test, loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  auto input = take(test, loom::sim::finalizeSimulationRuntimeInput(
                              inputDraft, workload, sourceView));

  const auto reference = take(
      test, loom::sim::executeNativeStructuredProgram(source, workload, input));
  const auto candidate = take(test, loom::sim::executeSelectedStructuredProgram(
                                        selected, source, workload, input));
  require(test,
          loom::sim::haveEquivalentFunctionalObservations(reference, candidate),
          "reused ordered multicast generations changed whole-program "
          "observations");

  auto consumerFirst = parse(R"mlir(
module {
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %message: i32) ctrl (%ctrl: none) {
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    dataflow.thread.yield
  }

  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %output: !llvm.ptr)
      ctrl (%ctrl: none) {
    %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
    llvm.store %message, %output : i32, !llvm.ptr
    dataflow.thread.yield
  }

  llvm.func @kernel() -> i32 {
    %one = llvm.mlir.constant(1 : i64) : i64
    %output = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    %seven = llvm.mlir.constant(7 : i32) : i32
    %consumer = dataflow.thread.launch @consumer(%channel, %output)
        : (!dataflow.channel<i32>, !llvm.ptr) -> !dataflow.thread_token
    %producer = dataflow.thread.launch @producer(%channel, %seven)
        : (!dataflow.channel<i32>, i32) -> !dataflow.thread_token
    dataflow.thread.wait %consumer : !dataflow.thread_token
    dataflow.thread.wait %producer : !dataflow.thread_token
    %result = llvm.load %output : !llvm.ptr -> i32
    llvm.return %result : i32
  }
}
)mlir",
                             "consumer-first logical-channel program");
  auto blocked = loom::sim::executeSelectedStructuredProgram(
      consumerFirst, source, workload, input);
  require(test, !blocked,
          "consumer-first serial projection executed an unwritten receive");
  const std::error_code blockedCode =
      llvm::errorToErrorCode(blocked.takeError());
  require(test, blockedCode == std::make_error_code(std::errc::not_supported),
          "consumer-first serial projection was not typed Unsupported");

  auto rateMismatch = parse(R"mlir(
module {
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %message: i32) ctrl (%ctrl: none) {
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    dataflow.thread.yield
  }

  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>, %output: !llvm.ptr)
      ctrl (%ctrl: none) {
    %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
    llvm.store %message, %output : i32, !llvm.ptr
    dataflow.thread.yield
  }

  llvm.func @kernel() -> i32 {
    %one = llvm.mlir.constant(1 : i64) : i64
    %output = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    %seven = llvm.mlir.constant(7 : i32) : i32
    %producer = dataflow.thread.launch @producer(%channel, %seven)
        : (!dataflow.channel<i32>, i32) -> !dataflow.thread_token
    dataflow.thread.wait %producer : !dataflow.thread_token
    %consumer = dataflow.thread.launch @consumer(%channel, %output)
        : (!dataflow.channel<i32>, !llvm.ptr) -> !dataflow.thread_token
    dataflow.thread.wait %consumer : !dataflow.thread_token
    %result = llvm.load %output : !llvm.ptr -> i32
    llvm.return %result : i32
  }
}
)mlir",
                            "logical-channel rate mismatch");
  auto mismatched = loom::sim::executeSelectedStructuredProgram(
      rateMismatch, source, workload, input);
  require(test, !mismatched,
          "mismatched ordered-channel rates reached collective join");
  bool typedPendingConsumer = false;
  llvm::handleAllErrors(
      mismatched.takeError(),
      [&](const loom::runtime::OrderedChannelABIError &error) {
        typedPendingConsumer =
            error.kind() ==
            loom::runtime::OrderedChannelABIError::Kind::PendingConsumer;
      },
      [](const llvm::ErrorInfoBase &) {});
  require(test, typedPendingConsumer,
          "mismatched ordered-channel rates lost their typed rejection");
}

void forallAggregationRegionsAreNotProfileBlocks() {
  const char *test = __func__;
  SourceProgram source = sourceProgram(test);
  auto sourceView = take(test, source.candidate.view());
  auto workload =
      take(test,
           loom::sim::finalizeSimulationWorkload(
               makeWorkload(entryRef(test, sourceView), definedI32(0x12345678)),
               sourceView));
  auto input = take(
      test, loom::sim::finalizeSimulationRuntimeInput(
                makeRuntimeInput(workload.identity()), workload, sourceView));
  auto selected = selectedForallProgram(test, source.layout);

  const auto reference = take(test, loom::sim::executeNativeStructuredProgram(
                                        source.candidate, workload, input));
  const auto candidate =
      take(test, loom::sim::executeProfiledSelectedStructuredProgram(
                     selected, source.candidate, workload, input));
  require(test,
          loom::sim::haveEquivalentFunctionalObservations(reference, candidate),
          "profiled forall ownership changed whole-program semantics");
  require(test, !candidate.blockActivations.empty(),
          "profiled forall ownership produced no block activations");
  auto selectedView = take(test, selected.view());
  for (const auto &activation : candidate.blockActivations) {
    auto entity = take(test, selectedView.resolve(activation.block));
    require(test,
            entity.block && !llvm::isa<mlir::scf::InParallelOp>(
                                entity.block->getParentOp()),
            "forall aggregation region entered the executable profile");
  }
}

void storedPointerPayloadExecutesThroughNativeObjectRegistry() {
  const char *test = __func__;
  SourceProgram host = sourceProgram(test);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%descriptor: !llvm.ptr, %target: !llvm.ptr) -> i32 {
    %loaded = llvm.load %descriptor : !llvm.ptr -> !llvm.ptr
    %address = llvm.getelementptr %loaded[2]
        : (!llvm.ptr) -> !llvm.ptr, i32
    %result = llvm.load %address : !llvm.ptr -> i32
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "cannot parse the stored-pointer Structured Program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), host.layout.getStringRepresentation()));
  auto program =
      take(test, loom::frontend::finalizeStructuredProgram(module.get()));
  auto view = take(test, program.view());

  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      entryRef(test, view)};
  workloadDraft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{},
                                loom::sim::StructuredRuntimeMemoryInput{}};
  workloadDraft.observableContract.returnValue = true;
  auto workload =
      take(test, loom::sim::finalizeSimulationWorkload(workloadDraft, view));

  const unsigned pointerBytes = host.layout.getPointerSize(0);
  loom::sim::RuntimeMemoryObject descriptor(
      std::vector<loom::sim::SemanticMemoryByte>(
          pointerBytes, {loom::sim::SemanticState::Defined, 0}));
  descriptor.initialBytes[1].value = 0x10;
  descriptor.pointerValues = {loom::sim::RuntimeMemoryPointer{
      0, 0,
      loom::sim::PointerTarget{
          1, llvm::APInt(host.layout.getIndexSizeInBits(0), 0)}}};
  loom::sim::RuntimeMemoryObject target(
      std::vector<loom::sim::SemanticMemoryByte>(
          16, {loom::sim::SemanticState::Defined, 0}));
  target.initialBytes[8].value = 0x78;
  target.initialBytes[9].value = 0x56;
  target.initialBytes[10].value = 0x34;
  target.initialBytes[11].value = 0x12;

  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  inputDraft.memoryObjects = {std::move(descriptor), std::move(target)};
  inputDraft.pointerBindings = {{0, 0, 0}, {1, 1, 0}};
  auto alternateDraft = inputDraft;
  alternateDraft.memoryObjects.front().initialBytes[6].value = 0x7f;
  auto input = take(test, loom::sim::finalizeSimulationRuntimeInput(
                              inputDraft, workload, view));
  auto alternateInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                       alternateDraft, workload, view));
  require(test,
          input.identity() == alternateInput.identity() &&
              input.canonicalBytes().bytes() ==
                  alternateInput.canonicalBytes().bytes(),
          "native stored-pointer bytes changed canonical runtime identity");
  auto execution =
      take(test,
           loom::sim::executeNativeStructuredProgram(program, workload, input));
  require(test,
          execution.returnValue && execution.returnValue->lanes.size() == 1 &&
              execution.returnValue->lanes.front().bits ==
                  llvm::APInt(32, 0x12345678),
          "native execution did not restore stored pointer provenance");
}

loom::frontend::StructuredProgramCandidate
denseThreadProgram(llvm::StringRef test, const llvm::DataLayout &layout,
                   bool selected) {
  const char *source = selected ? R"mlir(
module {
  dataflow.thread private @dense domain(#dataflow.thread_domain<dense>)(
      %base: !llvm.ptr) ctrl (%ctrl: none) iv (%coord: index) {
    "loom.spatial_region"(%coord, %base)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%i: index, %target: !llvm.ptr):
        %i64 = arith.index_cast %i : index to i64
        %ptr = llvm.getelementptr %target[%i64]
            : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %ten = arith.constant 10 : i32
        %i32 = arith.index_cast %i : index to i32
        %value = arith.addi %ten, %i32 : i32
        llvm.store %value, %ptr : i32, !llvm.ptr
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "dense_graph", source_maps = []} :
        (index, !llvm.ptr) -> ()
    dataflow.thread.yield
  }

  llvm.func @kernel(%base: !llvm.ptr) -> i32 {
    %extent = arith.constant 4 : index
    %token = dataflow.thread.launch @dense(%base) grid(%extent) :
        (!llvm.ptr) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    %last = llvm.getelementptr %base[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %result = llvm.load %last : !llvm.ptr -> i32
    llvm.return %result : i32
  }
}
)mlir"
                                : R"mlir(
module {
  llvm.func @kernel(%base: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %ptr = llvm.getelementptr %base[%i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %ten = arith.constant 10 : i32
      %i32 = arith.index_cast %i : index to i32
      %value = arith.addi %ten, %i32 : i32
      llvm.store %value, %ptr : i32, !llvm.ptr
    }
    %last = llvm.getelementptr %base[3] : (!llvm.ptr) -> !llvm.ptr, i32
    %result = llvm.load %last : !llvm.ptr -> i32
    llvm.return %result : i32
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "cannot parse the dense thread-domain program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), layout.getStringRepresentation()));
  return take(test, loom::frontend::finalizeStructuredProgram(module.get()));
}

void denseThreadDomainsPreserveWholeProgramSemantics() {
  const char *test = __func__;
  SourceProgram host = sourceProgram(test);
  auto source = denseThreadProgram(test, host.layout, false);
  auto sourceView = take(test, source.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      entryRef(test, sourceView)};
  workloadDraft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{}};
  workloadDraft.observableContract.returnValue = true;
  workloadDraft.observableContract.memories = {
      {loom::sim::EntryPointerArgumentTarget{0},
       loom::sim::MemoryObservationForm::FullState}};
  auto workload = take(
      test, loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  inputDraft.memoryObjects.push_back(
      loom::sim::RuntimeMemoryObject{std::vector<loom::sim::SemanticMemoryByte>(
          16, {loom::sim::SemanticState::Defined, 0})});
  inputDraft.pointerBindings.push_back({0, 0, 0});
  auto input = take(test, loom::sim::finalizeSimulationRuntimeInput(
                              inputDraft, workload, sourceView));
  auto selected = denseThreadProgram(test, host.layout, true);

  const auto reference = take(
      test, loom::sim::executeNativeStructuredProgram(source, workload, input));
  const auto candidate = take(test, loom::sim::executeSelectedStructuredProgram(
                                        selected, source, workload, input));
  require(test,
          loom::sim::haveEquivalentFunctionalObservations(reference, candidate),
          "dense logical thread projection changed whole-program semantics");
}

loom::frontend::StructuredProgramCandidate
dynamicDenseThreadProgram(llvm::StringRef test, const llvm::DataLayout &layout,
                          bool selected) {
  const char *source = selected ? R"mlir(
module {
  dataflow.thread private @dense domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) iv (%coord: index) {
    "loom.spatial_region"(%coord)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%i: index):
        %unused = arith.index_cast %i : index to i64
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "dense_graph", source_maps = []} : (index) -> ()
    dataflow.thread.yield
  }

  llvm.func @kernel(%extent64: i64) -> i32 {
    %extent = arith.index_cast %extent64 : i64 to index
    %token = dataflow.thread.launch @dense() grid(%extent) :
        () -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    %zero = arith.constant 0 : i32
    llvm.return %zero : i32
  }
}
)mlir"
                                : R"mlir(
module {
  llvm.func @kernel(%extent64: i64) -> i32 {
    %extent = arith.index_cast %extent64 : i64 to index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %extent step %one {
      %unused = arith.index_cast %i : index to i64
    }
    %result = arith.constant 0 : i32
    llvm.return %result : i32
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "cannot parse the dynamic dense thread-domain program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), layout.getStringRepresentation()));
  return take(test, loom::frontend::finalizeStructuredProgram(module.get()));
}

void negativeDynamicThreadExtentFailsExecution() {
  const char *test = __func__;
  SourceProgram host = sourceProgram(test);
  auto source = dynamicDenseThreadProgram(test, host.layout, false);
  auto sourceView = take(test, source.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      entryRef(test, sourceView)};
  workloadDraft.argumentPlan = {loom::sim::StructuredRuntimeValueInput{}};
  workloadDraft.observableContract.returnValue = true;
  auto workload = take(
      test, loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  loom::sim::CanonicalValueSequence extent;
  extent.tokenCount = 1;
  extent.lanes.push_back(
      loom::sim::SemanticLane::defined(llvm::APInt(64, -1, /*isSigned=*/true)));
  inputDraft.runtimeValues = {{0, std::move(extent)}};
  auto input = take(test, loom::sim::finalizeSimulationRuntimeInput(
                              inputDraft, workload, sourceView));
  auto selected = dynamicDenseThreadProgram(test, host.layout, true);

  auto execution = loom::sim::executeSelectedStructuredProgram(selected, source,
                                                               workload, input);
  require(test, !execution, "a negative dynamic thread extent was accepted");
  const std::string error = llvm::toString(execution.takeError());
  require(test,
          error.find("_execution_failed:") != std::string::npos &&
              error.find("logical thread extent is negative") !=
                  std::string::npos,
          "negative dynamic thread extent used the wrong failure class");
}

void runtimeAllocationsEnterTheCaptureRegistry() {
  const char *test = __func__;
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail(test, "cannot initialize the native target");
  auto target = take(test, llvm::orc::JITTargetMachineBuilder::detectHost());
  llvm::DataLayout layout = take(test, target.getDefaultDataLayoutForTarget());

  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext localContext(registry,
                                 mlir::MLIRContext::Threading::DISABLED);

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @malloc(%size: i64) -> !llvm.ptr
  llvm.func @free(%object: !llvm.ptr)

  llvm.func @runtime_allocate(%size: i64) -> !llvm.ptr
      attributes {allocsize = array<i32: 0>} {
    %object = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    llvm.return %object : !llvm.ptr
  }
  llvm.func @runtime_release(%object: !llvm.ptr) {
    llvm.call @free(%object) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @main() -> i32 {
    %size = llvm.mlir.constant(16 : i64) : i64
    %object = llvm.call @runtime_allocate(%size) : (i64) -> !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %three = llvm.mlir.constant(3 : i32) : i32
    %four = llvm.mlir.constant(4 : i32) : i32
    %selected = llvm.mlir.constant(42 : i32) : i32
    %p1 = llvm.getelementptr %object[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %p2 = llvm.getelementptr %object[2] : (!llvm.ptr) -> !llvm.ptr, i32
    %p3 = llvm.getelementptr %object[3] : (!llvm.ptr) -> !llvm.ptr, i32
    llvm.store %one, %object : i32, !llvm.ptr
    llvm.store %two, %p1 : i32, !llvm.ptr
    llvm.store %three, %p2 : i32, !llvm.ptr
    llvm.store %four, %p3 : i32, !llvm.ptr
    llvm.store %selected, %object : i32, !llvm.ptr
    llvm.call @runtime_release(%object) : (!llvm.ptr) -> ()
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
}
)mlir",
                                                        &localContext);
  if (!module)
    fail(test, "cannot parse the runtime-allocation program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&localContext, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&localContext, layout.getStringRepresentation()));

  loom::frontend::StructuredProgramCandidate source =
      take(test, loom::frontend::finalizeStructuredProgram(module.get()));
  auto view = take(test, source.view());
  std::optional<loom::frontend::StructuredEntityRef> mainRef;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getName() == "main") {
      mainRef = entity.reference;
      break;
    }
  }
  require(test, mainRef.has_value(), "cannot find the exact main reference");

  loom::sim::StructuredProgramSimulationWorkload workloadDraft{*mainRef};
  workloadDraft.observableContract.returnValue = true;
  auto workload =
      take(test, loom::sim::finalizeSimulationWorkload(workloadDraft, view));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  auto runtimeInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                     runtimeDraft, workload, view));

  mlir::LLVM::StoreOp selectedStore;
  module->walk([&](mlir::LLVM::StoreOp store) { selectedStore = store; });
  require(test, static_cast<bool>(selectedStore),
          "runtime-allocation program has no selected store");
  const loom::ArtifactIdentity &identity = source.identity();
  dataflow::RootedGraphLaunchRef launch{
      dataflow::RootThreadLaunchRef{identity, dataflow::RootThreadLaunchId(0)},
      dataflow::StaticGraphLaunchRef{identity,
                                     dataflow::StaticGraphLaunchId(0)}};
  loom::sim::WorkloadBackedSimulationInputCapturePlan plan{launch, {}, {}, {},
                                                           {},     {}, {}};
  plan.memoryRoots.push_back({dataflow::LogicalMemoryRootRef{
                                  identity, dataflow::LogicalMemoryRootId(0)},
                              selectedStore.getAddr()});

  std::vector<loom::sim::NativeSimulationCallCapture> calls;
  if (llvm::Error error = loom::sim::visitWorkloadBackedSimulationInputCaptures(
          std::move(module), selectedStore.getOperation(), plan, source,
          workload, runtimeInput, 1024 * 1024,
          [&](loom::sim::NativeSimulationCallCapture &&capture) {
            calls.push_back(std::move(capture));
            return llvm::Error::success();
          }))
    fail(test, llvm::toString(std::move(error)));
  require(test,
          calls.size() == 1 && calls.front().objects.size() == 1 &&
              calls.front().objects.front().initialBytes.size() == 16 &&
              calls.front().objects.front().finalBytes.size() == 16 &&
              calls.front().memoryRootObjectOrdinals ==
                  std::vector<std::uint64_t>{0} &&
              calls.front().memoryRootByteOffsets ==
                  std::vector<std::uint64_t>{0},
          "runtime allocation did not resolve to one finite capture object");
}

void pointerCaptureFollowsSelectedMemoryEffects() {
  const char *test = __func__;
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail(test, "cannot initialize the native target");
  auto target = take(test, llvm::orc::JITTargetMachineBuilder::detectHost());
  llvm::DataLayout layout = take(test, target.getDefaultDataLayoutForTarget());

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%root: !llvm.ptr) {
    %value = llvm.mlir.constant(42 : i32) : i32
    llvm.store %value, %root : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @pointer_kernel(%scratch: !llvm.ptr) -> i32 {
    %loaded = llvm.load %scratch : !llvm.ptr -> !llvm.ptr
    %bits = llvm.ptrtoint %loaded : !llvm.ptr to i64
    %value = llvm.trunc %bits : i64 to i32
    llvm.return %value : i32
  }

  llvm.func @main() -> i32 {
    %two = llvm.mlir.constant(2 : i64) : i64
    %storage = llvm.alloca %two x !llvm.ptr : (i64) -> !llvm.ptr
    %address = llvm.mlir.constant(4096 : i64) : i64
    %external = llvm.inttoptr %address : i64 to !llvm.ptr
    llvm.store %external, %storage : !llvm.ptr, !llvm.ptr
    %root = llvm.getelementptr %storage[1]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.call @kernel(%root) : (!llvm.ptr) -> ()
    %observed = llvm.call @pointer_kernel(%storage)
        : (!llvm.ptr) -> i32
    %result = llvm.load %root : !llvm.ptr -> i32
    %combined = llvm.xor %result, %observed : i32
    llvm.return %combined : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "cannot parse the unrelated-pointer-store program");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), layout.getStringRepresentation()));

  loom::frontend::StructuredProgramCandidate source =
      take(test, loom::frontend::finalizeStructuredProgram(module.get()));
  auto view = take(test, source.view());
  std::optional<loom::frontend::StructuredEntityRef> mainRef;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getName() == "main") {
      mainRef = entity.reference;
      break;
    }
  }
  require(test, mainRef.has_value(), "cannot find the exact main reference");

  loom::sim::StructuredProgramSimulationWorkload workloadDraft{*mainRef};
  workloadDraft.observableContract.returnValue = true;
  auto workload =
      take(test, loom::sim::finalizeSimulationWorkload(workloadDraft, view));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  auto runtimeInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                     runtimeDraft, workload, view));

  auto kernel = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("kernel");
  require(test, static_cast<bool>(kernel), "cannot find the selected kernel");
  const loom::ArtifactIdentity &identity = source.identity();
  dataflow::RootedGraphLaunchRef launch{
      dataflow::RootThreadLaunchRef{identity, dataflow::RootThreadLaunchId(0)},
      dataflow::StaticGraphLaunchRef{identity,
                                     dataflow::StaticGraphLaunchId(0)}};
  loom::sim::WorkloadBackedSimulationInputCapturePlan plan{launch, {}, {}, {},
                                                           {},     {}, {}};
  plan.memoryRoots.push_back({dataflow::LogicalMemoryRootRef{
                                  identity, dataflow::LogicalMemoryRootId(0)},
                              kernel.getArgument(0)});
  mlir::OwningOpRef<mlir::ModuleOp> selectedScratchModule(
      llvm::cast<mlir::ModuleOp>(module->clone()));

  std::vector<loom::sim::NativeSimulationCallCapture> calls;
  if (llvm::Error error = loom::sim::visitWorkloadBackedSimulationInputCaptures(
          std::move(module), kernel.getOperation(), plan, source, workload,
          runtimeInput, 1024 * 1024,
          [&](loom::sim::NativeSimulationCallCapture &&capture) {
            calls.push_back(std::move(capture));
            return llvm::Error::success();
          }))
    fail(test, llvm::toString(std::move(error)));
  require(test,
          calls.size() == 1 && calls.front().objects.size() == 1 &&
              calls.front().objects.front().finalBytes.size() == 16 &&
              calls.front().objects.front().finalBytes[8] == 42 &&
              calls.front().objects.front().finalBytes[9] == 0 &&
              calls.front().objects.front().finalBytes[10] == 0 &&
              calls.front().objects.front().finalBytes[11] == 0 &&
              calls.front().memoryRootByteOffsets ==
                  std::vector<std::uint64_t>{8},
          "an unobserved pointer field polluted the selected memory effect");

  auto scratchKernel =
      selectedScratchModule->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "pointer_kernel");
  require(test, static_cast<bool>(scratchKernel),
          "the cloned program lost the pointer-reading kernel");
  plan.memoryRoots.front().boundaryPointer = scratchKernel.getArgument(0);
  llvm::Error error = loom::sim::visitWorkloadBackedSimulationInputCaptures(
      std::move(selectedScratchModule), scratchKernel.getOperation(), plan,
      source, workload, runtimeInput, 1024 * 1024,
      [&](loom::sim::NativeSimulationCallCapture &&) {
        return llvm::Error::success();
      });
  require(test, static_cast<bool>(error),
          "an observed pointer target outside the registry was accepted");
  const std::string message = llvm::toString(std::move(error));
  require(test,
          message.find("captured pointer target is outside the runtime object "
                       "registry") != std::string::npos,
          "an observed unknown pointer target used the wrong failure");
}

} // namespace

int main() {
  exactEntryPreservesAliasingAndObservations();
  nonDefinedInputsFailClosed();
  typedCosineUsesCanonicalNativeSemantics();
  selectedOwnershipCarriersExecuteAsWholeProgram();
  orderedLogicalChannelGenerationsCompleteMulticastSequences();
  forallAggregationRegionsAreNotProfileBlocks();
  storedPointerPayloadExecutesThroughNativeObjectRegistry();
  denseThreadDomainsPreserveWholeProgramSemantics();
  negativeDynamicThreadExtentFailsExecution();
  runtimeAllocationsEnterTheCaptureRegistry();
  pointerCaptureFollowsSelectedMemoryEffects();
  llvm::outs() << "structured program native execution anchors passed\n";
  return EXIT_SUCCESS;
}
