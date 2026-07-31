#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationInputCapture.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <vector>

using namespace dataflow;
using namespace loom;
using namespace loom::sim;

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFGPointerExecutionTest: " << message << '\n';
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::DLTIDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *instance;
}

const char *pointerLoadProgram() {
  return R"mlir(
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<
    "dlti.endianness" = "little",
    index = 32 : i64
  >
} {
  dataflow.graph private @pointer_load(%ctrl: none, %pointer: !llvm.ptr,
                                       %service: memref<?xi32>)
      -> i32 attributes {
        input_segments = array<i32: 1, 0, 1>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %two = arith.constant 2 : i32
    %address = llvm.getelementptr inbounds %pointer[%two]
        : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %value, %done = dataflow.load %service[%address] %ctrl
        : memref<?xi32>, !llvm.ptr
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.thread private @pointer_thread
      domain(#dataflow.thread_domain<dense>)(%pointer: !llvm.ptr)
      ctrl (%ctrl: none) {
    %service = dataflow.memory.service %pointer
        : !llvm.ptr -> memref<?xi32>
    %value, %done = dataflow.graph.launch @pointer_load deps(%ctrl)
        values(%pointer) stream_inputs() memories(%service) stream_outputs()
        : (none, !llvm.ptr, memref<?xi32>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%pointer: !llvm.ptr) {
    %thread = dataflow.thread.launch @pointer_thread(%pointer)
        : (!llvm.ptr) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

const char *pointerCaptureProgram() {
  return R"mlir(
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<
    "dlti.endianness" = "little",
    index = 64 : i64
  >
} {
  llvm.mlir.global external @storage() : !llvm.array<4 x i32>
  dataflow.graph private @pointer_capture(
      %ctrl: none, %pointer: !llvm.ptr, %service: memref<?xi32>)
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value, %done = dataflow.load %service[%pointer] %ctrl
        : memref<?xi32>, !llvm.ptr
    dataflow.graph.return values() streams() memories()
        complete(%done : none)
  }
  dataflow.thread private @pointer_thread
      domain(#dataflow.thread_domain<dense>)(%pointer: !llvm.ptr)
      ctrl (%ctrl: none) {
    %service = dataflow.memory.service %pointer
        : !llvm.ptr -> memref<?xi32>
    %done = dataflow.graph.launch @pointer_capture deps(%ctrl)
        values(%pointer) stream_inputs() memories(%service) stream_outputs()
        : (none, !llvm.ptr, memref<?xi32>) -> none
    dataflow.thread.yield %done : none
  }
  llvm.func @kernel(%pointer: !llvm.ptr) {
    %thread = dataflow.thread.launch @pointer_thread(%pointer)
        : (!llvm.ptr) -> !dataflow.thread_token
    llvm.return
  }
  llvm.func @main() -> i32 {
    %pointer = llvm.mlir.addressof @storage : !llvm.ptr
    llvm.call @kernel(%pointer) : (!llvm.ptr) -> ()
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.return %zero : i32
  }
}
)mlir";
}

const char *pointerPayloadProgram() {
  return R"mlir(
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<
    "dlti.endianness" = "little",
    index = 32 : i64
  >
} {
  dataflow.graph private @pointer_payload(
      %ctrl: none, %pointer: !llvm.ptr, %descriptor: memref<1xi64>,
      %target: memref<4xi32>) -> i32 attributes {
        input_segments = array<i32: 1, 0, 2>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : i32
    %stored = dataflow.store %descriptor[%zero] %pointer %ctrl
        : memref<1xi64>, index, !llvm.ptr
    %loaded, %loaded_done = dataflow.load %descriptor[%zero] %stored
        : memref<1xi64>, index, !llvm.ptr
    %address = llvm.getelementptr inbounds %loaded[%two]
        : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %value, %done = dataflow.load %target[%address] %loaded_done
        : memref<4xi32>, !llvm.ptr
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.thread private @pointer_thread
      domain(#dataflow.thread_domain<dense>)(
          %pointer: !llvm.ptr, %descriptor: memref<1xi64>,
          %target: memref<4xi32>) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @pointer_payload deps(%ctrl)
        values(%pointer) stream_inputs() memories(%descriptor, %target)
        stream_outputs()
        : (none, !llvm.ptr, memref<1xi64>, memref<4xi32>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%pointer: !llvm.ptr, %descriptor: memref<1xi64>,
                          %target: memref<4xi32>) {
    %thread = dataflow.thread.launch @pointer_thread(
        %pointer, %descriptor, %target)
        : (!llvm.ptr, memref<1xi64>, memref<4xi32>)
          -> !dataflow.thread_token
    return
  }
}
)mlir";
}

const char *initialPointerPayloadProgram() {
  return R"mlir(
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<
    "dlti.endianness" = "little",
    index = 32 : i64
  >
} {
  dataflow.graph private @initial_pointer_payload(
      %ctrl: none, %descriptor: memref<1xi64>, %target: memref<4xi32>)
      -> i32 attributes {
        input_segments = array<i32: 0, 0, 2>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : i32
    %loaded, %loaded_done = dataflow.load %descriptor[%zero] %ctrl
        : memref<1xi64>, index, !llvm.ptr
    %address = llvm.getelementptr inbounds %loaded[%two]
        : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %value, %done = dataflow.load %target[%address] %loaded_done
        : memref<4xi32>, !llvm.ptr
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.thread private @pointer_thread
      domain(#dataflow.thread_domain<dense>)(
          %descriptor: memref<1xi64>, %target: memref<4xi32>) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @initial_pointer_payload deps(%ctrl)
        values() stream_inputs() memories(%descriptor, %target)
        stream_outputs()
        : (none, memref<1xi64>, memref<4xi32>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%descriptor: memref<1xi64>,
                          %target: memref<4xi32>) {
    %thread = dataflow.thread.launch @pointer_thread(%descriptor, %target)
        : (memref<1xi64>, memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

CanonicalDataflowArtifact finalizeProgram(const char *source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("failed to parse pointer-load fixture");
  llvm::Expected<CanonicalDataflowArtifact> artifact =
      finalizeCanonicalDataflow(module.get());
  if (!artifact)
    fail("failed to finalize pointer-load fixture: " +
         llvm::toString(artifact.takeError()));
  return std::move(*artifact);
}

LogicalMemoryRootRef rootAtFormal(const CanonicalDataflowProgramView &view,
                                  unsigned formal) {
  for (const CanonicalLogicalMemoryRootView &root : view.logicalMemoryRoots())
    if (root.formalArgIndex && *root.formalArgIndex == formal)
      return root.ref;
  fail("pointer fixture has no requested memory root");
}

LogicalMemoryRootRef rootAtService(const CanonicalDataflowProgramView &view) {
  for (const CanonicalLogicalMemoryRootView &root : view.logicalMemoryRoots())
    if (!root.formalArgIndex && llvm::isa_and_nonnull<MemoryServiceOp>(root.op))
      return root.ref;
  fail("pointer fixture has no memory-service root");
}

void pointerGepLoadPreservesObjectProvenance() {
  CanonicalDataflowArtifact artifact = finalizeProgram(pointerLoadProgram());
  llvm::Expected<CanonicalDataflowProgramView> imported = artifact.view();
  if (!imported)
    fail("failed to import pointer-load fixture: " +
         llvm::toString(imported.takeError()));
  CanonicalDataflowProgramView view = std::move(*imported);
  require(view.rootThreadLaunches().size() == 1 &&
              view.staticGraphLaunches().size() == 1,
          "pointer-load fixture does not have one rooted launch");

  SpatialSimulationWorkload workloadModel{
      RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                           view.staticGraphLaunches().front().ref}};
  workloadModel.valueInputPlan = {RuntimeValueInput{}};
  workloadModel.observableContract.valueResults = {0};
  llvm::Expected<CanonicalSimulationWorkload> workload =
      finalizeSimulationWorkload(workloadModel, view);
  if (!workload)
    fail("failed to finalize pointer-load workload: " +
         llvm::toString(workload.takeError()));

  RuntimeMemoryObject object;
  object.initialBytes.assign(16, SemanticMemoryByte{SemanticState::Defined, 0});
  object.initialBytes[8].value = 0x78;
  object.initialBytes[9].value = 0x56;
  object.initialBytes[10].value = 0x34;
  object.initialBytes[11].value = 0x12;

  SpatialSimulationRuntimeInputDraft inputDraft{workload->identity()};
  inputDraft.runtimeValues = {RuntimeValueEntry{
      0, CanonicalValueSequence{
             1,
             {SemanticLane::definedPointer(llvm::APInt(64, 0x1000), 0,
                                           llvm::APInt(64, 0))}}}};
  inputDraft.memoryObjects = {std::move(object)};
  inputDraft.memoryRootBindings = {
      RuntimeMemoryBindingDraft{rootAtService(view), 0, 0}};
  llvm::Expected<CanonicalSimulationRuntimeInput> input =
      finalizeSimulationRuntimeInput(inputDraft, *workload, view);
  if (!input)
    fail("failed to finalize pointer-load input: " +
         llvm::toString(input.takeError()));

  llvm::Expected<RetiredDFGSimulation> execution =
      simulateRetiredDfgWorkload(artifact, *workload, *input);
  if (!execution)
    fail("pointer-load execution failed: " +
         llvm::toString(execution.takeError()));
  require(execution->observations.valueResults.size() == 1,
          "pointer-load execution returned the wrong result count");
  const auto *published = std::get_if<PublishedValueResult>(
      &execution->observations.valueResults.front());
  require(published && published->value.lanes.size() == 1 &&
              published->value.lanes.front().state == SemanticState::Defined &&
              published->value.lanes.front().bits ==
                  llvm::APInt(32, 0x12345678),
          "pointer-addressed load returned the wrong value");
}

void pointerServiceCaptureResolvesSharedObject() {
  CanonicalDataflowArtifact artifact = finalizeProgram(pointerCaptureProgram());
  llvm::Expected<CanonicalDataflowProgramView> imported = artifact.view();
  if (!imported)
    fail("failed to import pointer-capture fixture: " +
         llvm::toString(imported.takeError()));
  CanonicalDataflowProgramView view = std::move(*imported);
  require(view.rootThreadLaunches().size() == 1 &&
              view.staticGraphLaunches().size() == 1,
          "pointer-capture fixture does not have one rooted launch");

  mlir::LLVM::CallOp call;
  artifact.module().walk([&](mlir::LLVM::CallOp candidate) {
    if (candidate.getCalleeAttr() &&
        candidate.getCalleeAttr().getValue() == "kernel")
      call = candidate;
  });
  require(static_cast<bool>(call),
          "pointer-capture fixture has no exact host call");
  RootedGraphLaunchRef launch{view.rootThreadLaunches().front().ref,
                              view.staticGraphLaunches().front().ref};
  llvm::Expected<DirectCallSimulationInputCapturePlan> capture =
      deriveSimulationInputCapturePlan(view, launch, call);
  if (!capture)
    fail("failed to derive pointer service capture: " +
         llvm::toString(capture.takeError()));
  require(capture->input.objects.size() == 1 &&
              capture->input.memoryRootBindings.size() == 1 &&
              capture->input.valueInputs.size() == 1,
          "pointer service capture did not recover one shared object");
  const auto &target = capture->input.valueInputs.front().pointerTarget;
  require(target && target->memoryRootBindingOrdinal == 0 &&
              target->addressBitWidth == 64,
          "pointer service capture lost its object-provenance relation");

  constexpr llvm::StringLiteral nativeSource = R"llvm(
@storage = global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4

define void @kernel(ptr %pointer) {
entry:
  ret void
}

define i32 @main() {
entry:
  %pointer = getelementptr [4 x i32], ptr @storage, i64 0, i64 0
  call void @kernel(ptr %pointer)
  ret i32 0
}
)llvm";
  static const bool targetInitializationFailed = [] {
    return llvm::InitializeNativeTarget() ||
           llvm::InitializeNativeTargetAsmPrinter();
  }();
  require(!targetInitializationFailed, "cannot initialize native target");
  auto nativeContext = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> nativeModule = llvm::parseIR(
      llvm::MemoryBuffer::getMemBuffer(nativeSource, "<pointer-capture>")
          ->getMemBufferRef(),
      diagnostic, *nativeContext);
  require(static_cast<bool>(nativeModule),
          "cannot parse native pointer-capture module");
  llvm::orc::JITTargetMachineBuilder targetMachine =
      take(llvm::orc::JITTargetMachineBuilder::detectHost());
  nativeModule->setTargetTriple(targetMachine.getTargetTriple());
  nativeModule->setDataLayout(
      take(targetMachine.getDefaultDataLayoutForTarget()));
  NativeSimulationInputCapture native =
      take(executeNativeSimulationInputCapture(
          llvm::orc::ThreadSafeModule(std::move(nativeModule),
                                      std::move(nativeContext)),
          *capture));
  require(native.entryResult == 0 && native.calls.size() == 1 &&
              native.calls.front().runtimeValues.size() == 1,
          "native pointer service capture is incomplete");
  const SemanticLane &lane =
      native.calls.front().runtimeValues.front().value.lanes.front();
  require(lane.pointerTarget && lane.pointerTarget->objectOrdinal == 0 &&
              lane.pointerTarget->byteOffset == llvm::APInt(64, 0),
          "native pointer service capture lost canonical provenance");
}

void pointerPayloadRoundtripPreservesProvenance() {
  CanonicalDataflowArtifact artifact = finalizeProgram(pointerPayloadProgram());
  llvm::Expected<CanonicalDataflowProgramView> imported = artifact.view();
  if (!imported)
    fail("failed to import pointer-payload fixture: " +
         llvm::toString(imported.takeError()));
  CanonicalDataflowProgramView view = std::move(*imported);

  SpatialSimulationWorkload workloadModel{
      RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                           view.staticGraphLaunches().front().ref}};
  workloadModel.valueInputPlan = {RuntimeValueInput{}};
  workloadModel.observableContract.valueResults = {0};
  auto workload = finalizeSimulationWorkload(workloadModel, view);
  if (!workload)
    fail("failed to finalize pointer-payload workload: " +
         llvm::toString(workload.takeError()));

  RuntimeMemoryObject target;
  target.initialBytes.assign(16, SemanticMemoryByte{SemanticState::Defined, 0});
  target.initialBytes[8].value = 0x78;
  target.initialBytes[9].value = 0x56;
  target.initialBytes[10].value = 0x34;
  target.initialBytes[11].value = 0x12;
  RuntimeMemoryObject descriptor;
  descriptor.initialBytes.assign(8,
                                 SemanticMemoryByte{SemanticState::Defined, 0});

  SpatialSimulationRuntimeInputDraft inputDraft{workload->identity()};
  inputDraft.runtimeValues = {RuntimeValueEntry{
      0, CanonicalValueSequence{
             1,
             {SemanticLane::definedPointer(llvm::APInt(64, 0x1000), 0,
                                           llvm::APInt(64, 0))}}}};
  inputDraft.memoryObjects = {std::move(target), std::move(descriptor)};
  inputDraft.memoryRootBindings = {
      RuntimeMemoryBindingDraft{rootAtFormal(view, 1), 1, 0},
      RuntimeMemoryBindingDraft{rootAtFormal(view, 2), 0, 0}};
  auto input = finalizeSimulationRuntimeInput(inputDraft, *workload, view);
  if (!input)
    fail("failed to finalize pointer-payload input: " +
         llvm::toString(input.takeError()));

  auto execution = simulateRetiredDfgWorkload(artifact, *workload, *input);
  if (!execution)
    fail("pointer-payload execution failed: " +
         llvm::toString(execution.takeError()));
  const auto *published = std::get_if<PublishedValueResult>(
      &execution->observations.valueResults.front());
  require(published && published->value.lanes.size() == 1 &&
              published->value.lanes.front().state == SemanticState::Defined &&
              published->value.lanes.front().bits ==
                  llvm::APInt(32, 0x12345678),
          "pointer payload lost provenance across descriptor memory");
}

void initialPointerPayloadUsesCanonicalObjectRegistry() {
  CanonicalDataflowArtifact artifact =
      finalizeProgram(initialPointerPayloadProgram());
  auto imported = artifact.view();
  if (!imported)
    fail("failed to import initial pointer-payload fixture: " +
         llvm::toString(imported.takeError()));
  CanonicalDataflowProgramView view = std::move(*imported);

  SpatialSimulationWorkload workloadModel{
      RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                           view.staticGraphLaunches().front().ref}};
  workloadModel.observableContract.valueResults = {0};
  auto workload = finalizeSimulationWorkload(workloadModel, view);
  if (!workload)
    fail("failed to finalize initial pointer-payload workload: " +
         llvm::toString(workload.takeError()));

  RuntimeMemoryObject target;
  target.initialBytes.assign(16, SemanticMemoryByte{SemanticState::Defined, 0});
  target.initialBytes[8].value = 0x78;
  target.initialBytes[9].value = 0x56;
  target.initialBytes[10].value = 0x34;
  target.initialBytes[11].value = 0x12;
  RuntimeMemoryObject descriptor;
  descriptor.initialBytes.assign(8,
                                 SemanticMemoryByte{SemanticState::Defined, 0});
  descriptor.initialBytes[1].value = 0x10;
  descriptor.pointerValues = {
      RuntimeMemoryPointer{0, 0, PointerTarget{0, llvm::APInt(64, 0)}}};

  SpatialSimulationRuntimeInputDraft inputDraft{workload->identity()};
  inputDraft.memoryObjects = {std::move(target), std::move(descriptor)};
  inputDraft.memoryRootBindings = {
      RuntimeMemoryBindingDraft{rootAtFormal(view, 0), 1, 0},
      RuntimeMemoryBindingDraft{rootAtFormal(view, 1), 0, 0}};
  auto input = finalizeSimulationRuntimeInput(inputDraft, *workload, view);
  if (!input)
    fail("failed to finalize initial pointer-payload input: " +
         llvm::toString(input.takeError()));
  auto roundtrip = importSimulationRuntimeInput(
      input->canonicalBytes().bytes(), *workload, view, input->identity());
  if (!roundtrip)
    fail("failed to import initial pointer-payload input: " +
         llvm::toString(roundtrip.takeError()));

  auto execution = simulateRetiredDfgWorkload(artifact, *workload, *roundtrip);
  if (!execution)
    fail("initial pointer-payload execution failed: " +
         llvm::toString(execution.takeError()));
  const auto *published = std::get_if<PublishedValueResult>(
      &execution->observations.valueResults.front());
  require(published && published->value.lanes.size() == 1 &&
              published->value.lanes.front().state == SemanticState::Defined &&
              published->value.lanes.front().bits ==
                  llvm::APInt(32, 0x12345678),
          "initial pointer payload lost canonical object provenance");
}

} // namespace

int main() {
  pointerServiceCaptureResolvesSharedObject();
  pointerGepLoadPreservesObjectProvenance();
  pointerPayloadRoundtripPreservesProvenance();
  initialPointerPayloadUsesCanonicalObjectRegistry();
  llvm::outs() << "DFG pointer execution anchor passed\n";
  return 0;
}
