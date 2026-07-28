#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Compilation/StaticMemoryBinding.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationInputCapture.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<llvm::Module> parseVecadd(const char *test,
                                          llvm::LLVMContext &context,
                                          bool riscvTarget = true) {
  constexpr llvm::StringLiteral source = R"llvm(
define void @vecadd(ptr %a, ptr %b, ptr %c) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %pa = getelementptr float, ptr %a, i64 %i
  %pb = getelementptr float, ptr %b, i64 %i
  %pc = getelementptr float, ptr %c, i64 %i
  %va = load float, ptr %pa, align 4
  %vb = load float, ptr %pb, align 4
  %sum = fadd float %va, %vb
  store float %sum, ptr %pc, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define i32 @main() {
entry:
  %a = alloca [64 x float], align 4
  %b = alloca [64 x float], align 4
  %c = alloca [64 x float], align 4
  br label %init

init:
  %j = phi i64 [ 0, %entry ], [ %jnext, %init ]
  %fa = uitofp i64 %j to float
  %fb = fmul float %fa, 5.000000e-01
  %pa.init = getelementptr [64 x float], ptr %a, i64 0, i64 %j
  %pb.init = getelementptr [64 x float], ptr %b, i64 0, i64 %j
  %pc.init = getelementptr [64 x float], ptr %c, i64 0, i64 %j
  store float %fa, ptr %pa.init, align 4
  store float %fb, ptr %pb.init, align 4
  store float 0.000000e+00, ptr %pc.init, align 4
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, 64
  br i1 %jdone, label %invoke, label %init

invoke:
  call void @vecadd(ptr %a, ptr %b, ptr %c)
  ret i32 0
}

define i32 @slice_main() {
entry:
  %ab = alloca [128 x float], align 4
  %b = getelementptr [128 x float], ptr %ab, i64 0, i64 64
  %c = alloca [64 x float], align 4
  call void @vecadd(ptr %ab, ptr %b, ptr %c)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<vecadd>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  if (riscvTarget) {
    module->setDataLayout("e-m:e-p:64:64-i64:64-n32:64-S128");
    module->setTargetTriple(llvm::Triple("riscv64-unknown-unknown-elf"));
  }
  return module;
}

std::unique_ptr<llvm::Module> parseTableLookup(const char *test,
                                               llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

@lookup = private constant [4 x i32]
    [i32 287454020, i32 1432778632, i32 -1, i32 7], align 16

define void @table_lookup(ptr %output) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %source = getelementptr [4 x i32], ptr @lookup, i64 0, i64 %i
  %value = load i32, ptr %source, align 4
  %destination = getelementptr i32, ptr %output, i64 %i
  store i32 %value, ptr %destination, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<table-lookup>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

loom::frontend::StructuredEntityRef
findVecaddLoop(const char *test,
               const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(test, candidate.view());
  auto scopes =
      take(test,
           loom::frontend::enumerateOperationSpatialOwnershipScopes(candidate));
  for (const loom::frontend::StructuredEntityRef &scope : scopes) {
    auto entity = take(test, view.resolve(scope));
    auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(entity.operation);
    if (!loop)
      continue;
    auto callable = loop->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (callable && callable.getSymName() == "vecadd")
      return scope;
  }
  fail(test, "raised vecadd has no eligible structured loop");
}

dataflow::RootedGraphLaunchRef
onlyLaunch(const char *test,
           const dataflow::CanonicalDataflowProgramView &view) {
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail(test, "materialized vecadd must have one rooted graph launch");
  return dataflow::RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                                        view.staticGraphLaunches().front().ref};
}

dataflow::LogicalMemoryRootRef
memoryRoot(const char *test, const dataflow::CanonicalDataflowProgramView &view,
           unsigned threadFormal) {
  for (const dataflow::CanonicalLogicalMemoryRootView &root :
       view.logicalMemoryRoots())
    if (root.formalArgIndex && *root.formalArgIndex == threadFormal)
      return root.ref;
  fail(test, "materialized vecadd is missing an imported memory root");
}

mlir::LLVM::CallOp
findVecaddCall(const char *test,
               const dataflow::CanonicalDataflowArtifact &artifact,
               llvm::StringRef caller) {
  mlir::LLVM::CallOp result;
  artifact.module().walk([&](mlir::LLVM::CallOp call) {
    auto function = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (function && function.getSymName() == caller && call.getCalleeAttr() &&
        call.getCalleeAttr().getValue() == "vecadd")
      result = call;
  });
  if (!result)
    fail(test, "materialized vecadd has no requested host call site");
  return result;
}

const loom::sim::SimulationMemoryRootCapture &
captureBinding(const char *test,
               const loom::sim::SimulationInputCapturePlan &plan,
               dataflow::LogicalMemoryRootRef root) {
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       plan.memoryRootBindings)
    if (binding.root == root)
      return binding;
  fail(test, "capture plan is missing a logical memory root");
}

loom::frontend::StructuredEntityRef
findCallable(const char *test,
             const loom::frontend::StructuredProgramCandidate &candidate,
             llvm::StringRef name) {
  auto view = take(test, candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto callable =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (callable && callable.getSymName() == name)
      return entity.reference;
  }
  fail(test, "structured callable does not resolve");
}

loom::sim::RuntimeMemoryObject
definedByteObject(llvm::ArrayRef<std::uint8_t> bytes) {
  loom::sim::RuntimeMemoryObject object;
  object.initialBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    object.initialBytes.push_back({loom::sim::SemanticState::Defined, byte});
  return object;
}

std::vector<loom::sim::SemanticMemoryByte>
applyMemoryDiff(const char *test,
                llvm::ArrayRef<loom::sim::SemanticMemoryByte> baseline,
                const loom::sim::DiffMemoryObservation &diff) {
  if (diff.byteCount != baseline.size())
    fail(test, "typed memory diff has the wrong byte count");
  std::vector<loom::sim::SemanticMemoryByte> result(baseline.begin(),
                                                    baseline.end());
  std::uint64_t previousEnd = 0;
  for (const loom::sim::MemoryDiffRun &run : diff.runs) {
    if (run.changedBytes.empty() || run.byteOffset < previousEnd ||
        run.byteOffset + run.changedBytes.size() > result.size())
      fail(test, "typed memory diff has a malformed run");
    if (previousEnd != 0 && run.byteOffset == previousEnd)
      fail(test, "typed memory diff has adjacent non-maximal runs");
    std::copy(run.changedBytes.begin(), run.changedBytes.end(),
              result.begin() + run.byteOffset);
    previousEnd = run.byteOffset + run.changedBytes.size();
  }
  return result;
}

void sourceCandidateExecutesThroughTypedDfgInput() {
  const char *test = "sourceCandidateExecutesThroughTypedDfgInput";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-frontend-dfg-integration", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  auto nativeContext = std::make_unique<llvm::LLVMContext>();
  auto nativeModule = parseVecadd(test, *nativeContext, false);
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseVecadd(test, context),
                                 design.roots().front().reference(), store));

  loom::frontend::OperationSpatialOwnershipOptions ownership;
  ownership.canonicalIndexWidth = 32;
  auto candidate =
      take(test, loom::frontend::materializeOperationSpatialOwnership(
                     compiled.structuredProgram,
                     findVecaddLoop(test, compiled.structuredProgram),
                     design.roots().front(), ownership));
  auto view = take(test, candidate.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().size() < 20)
    fail(test, "source candidate did not produce a substantive Dataflow graph");

  const dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto capturePlan = take(
      test, loom::sim::deriveSimulationInputCapturePlan(
                view, launch,
                findVecaddCall(test, candidate.canonicalDataflow, "main")));
  if (capturePlan.input.objects.size() != 3 ||
      capturePlan.input.memoryRootBindings.size() != 3)
    fail(test, "vecadd capture plan did not recover three memory objects");
  for (const loom::sim::SimulationMemoryCaptureObject &object :
       capturePlan.input.objects)
    if (object.byteCount != 64 * sizeof(float))
      fail(test, "vecadd capture object has the wrong byte extent");
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       capturePlan.input.memoryRootBindings)
    if (binding.byteOffset != 0 || binding.objectIndex >= 3)
      fail(test, "vecadd capture binding has the wrong object projection");

  loom::sim::NativeSimulationInputCapture nativeCapture =
      take(test, loom::sim::executeNativeSimulationInputCapture(
                     llvm::orc::ThreadSafeModule(std::move(nativeModule),
                                                 std::move(nativeContext)),
                     capturePlan));
  if (nativeCapture.entryResult != 0 || nativeCapture.calls.size() != 1 ||
      nativeCapture.calls.front().objects.size() != 3)
    fail(test, "native vecadd oracle did not capture one complete call");

  auto mismatchedPlan = capturePlan;
  mismatchedPlan.input.objects.front().byteCount -= sizeof(float);
  auto mismatchContext = std::make_unique<llvm::LLVMContext>();
  auto mismatchedCapture = loom::sim::executeNativeSimulationInputCapture(
      llvm::orc::ThreadSafeModule(parseVecadd(test, *mismatchContext, false),
                                  std::move(mismatchContext)),
      mismatchedPlan);
  if (mismatchedCapture)
    fail(test, "native oracle accepted a mismatched host allocation extent");
  llvm::consumeError(mismatchedCapture.takeError());

  auto slicePlan = take(
      test,
      loom::sim::deriveSimulationInputCapturePlan(
          view, launch,
          findVecaddCall(test, candidate.canonicalDataflow, "slice_main")));
  const auto &sliceA =
      captureBinding(test, slicePlan.input, memoryRoot(test, view, 0));
  const auto &sliceB =
      captureBinding(test, slicePlan.input, memoryRoot(test, view, 1));
  const auto &sliceC =
      captureBinding(test, slicePlan.input, memoryRoot(test, view, 2));
  if (slicePlan.input.objects.size() != 2 ||
      slicePlan.input.objects[sliceA.objectIndex].byteCount !=
          128 * sizeof(float) ||
      sliceA.objectIndex != sliceB.objectIndex || sliceA.byteOffset != 0 ||
      sliceB.byteOffset != 64 * sizeof(float) ||
      sliceC.objectIndex == sliceA.objectIndex || sliceC.byteOffset != 0)
    fail(test, "vecadd slice capture did not preserve host aliasing");

  loom::sim::SpatialSimulationWorkload workload{launch};
  workload.observableContract.memories.push_back(
      loom::sim::SpatialMemoryObservable{
          dataflow::LogicalMemoryRootOrViewRef{memoryRoot(test, view, 2)},
          loom::sim::MemoryObservationForm::DiffFromRuntimeInput});
  auto finalizedWorkload =
      take(test, loom::sim::finalizeSimulationWorkload(workload, view));

  loom::sim::SpatialSimulationRuntimeInputDraft input{
      finalizedWorkload.identity()};
  for (const loom::sim::NativeCapturedMemoryObject &object :
       nativeCapture.calls.front().objects)
    input.memoryObjects.push_back(definedByteObject(object.initialBytes));
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       capturePlan.input.memoryRootBindings)
    input.memoryRootBindings.push_back(loom::sim::RuntimeMemoryBindingDraft{
        binding.root, binding.objectIndex, binding.byteOffset});
  auto finalizedInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                       input, finalizedWorkload, view));

  loom::sim::RetiredDFGSimulation execution =
      take(test,
           loom::sim::simulateRetiredDfgWorkload(
               candidate.canonicalDataflow, finalizedWorkload, finalizedInput));
  loom::sim::DFGSimulationReport &report = execution.report;
  if (report.status != "pass")
    fail(test, "typed DFG execution did not retire: " + report.status);
  if (report.operationFireCounts[dataflow::OperationSchemaId::ArithAddF] !=
          64 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowLoad] !=
          128 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowStore] !=
          64)
    fail(test, "typed DFG execution did not run the vecadd workload");

  std::string destinationPort;
  for (const auto &[port, root] : report.finalMemoryRoots)
    if (root == "memory_root2") {
      destinationPort = port;
      break;
    }
  auto destination = report.finalMemoryState.find(destinationPort);
  if (destination == report.finalMemoryState.end() ||
      destination->second.size() != 64)
    fail(test, "typed DFG execution lost the destination state");

  if (execution.observations.valueResults.size() != 0 ||
      execution.observations.streamOutputs.size() != 0 ||
      execution.observations.memories.size() != 1)
    fail(test, "typed DFG observations do not match the workload contract");
  const auto *diff = std::get_if<loom::sim::DiffMemoryObservation>(
      &execution.observations.memories.front());
  if (!diff)
    fail(test, "typed DFG execution did not preserve the requested diff form");
  const dataflow::LogicalMemoryRootRef destinationRoot =
      memoryRoot(test, view, 2);
  const loom::sim::MemoryRootBindingEntry *destinationBinding = nullptr;
  for (const loom::sim::MemoryRootBindingEntry &binding :
       finalizedInput.model().memoryRootBindings)
    if (binding.root == destinationRoot)
      destinationBinding = &binding;
  if (!destinationBinding)
    fail(test, "typed runtime input lost the destination root binding");
  std::vector<loom::sim::SemanticMemoryByte> reconstructed = applyMemoryDiff(
      test,
      finalizedInput.model()
          .memoryObjects[destinationBinding->binding.objectOrdinal]
          .initialBytes,
      *diff);
  const loom::sim::SimulationMemoryRootCapture &capturedDestination =
      captureBinding(test, capturePlan.input, destinationRoot);
  llvm::ArrayRef<std::uint8_t> expected(
      nativeCapture.calls.front()
          .objects[capturedDestination.objectIndex]
          .finalBytes);
  expected = expected.drop_front(capturedDestination.byteOffset);
  if (reconstructed.size() != expected.size())
    fail(test, "typed DFG memory result has the wrong extent");
  for (std::size_t index = 0; index < expected.size(); ++index)
    if (reconstructed[index].state != loom::sim::SemanticState::Defined ||
        reconstructed[index].value != expected[index])
      fail(test, "typed DFG memory diff reconstructs the wrong result");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void staticTableExecutesThroughTypedDfgInput() {
  const char *test = "staticTableExecutesThroughTypedDfgInput";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-static-table-dfg", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseTableLookup(test, context),
                                 design.roots().front().reference(), store));

  loom::frontend::WholeCallableSpatialOwnershipOptions ownership;
  ownership.canonicalIndexWidth = 64;
  auto candidate = take(
      test, loom::frontend::materializeWholeCallableSpatialOwnership(
                compiled.structuredProgram,
                findCallable(test, compiled.structuredProgram, "table_lookup"),
                design.roots().front(), ownership));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto sources = take(test, loom::frontend::deriveRootedLogicalMemorySources(
                                compiled.staticGlobalMemory, view, launch));
  if (sources.size() != 2)
    fail(test, "table lookup did not retain two logical memory roots");

  loom::sim::RuntimeMemoryObject table;
  loom::sim::RuntimeMemoryObject output =
      definedByteObject(std::vector<std::uint8_t>(16, 0));
  std::optional<dataflow::LogicalMemoryRootRef> tableRoot;
  std::optional<dataflow::LogicalMemoryRootRef> outputRoot;
  for (const loom::frontend::RootedLogicalMemorySource &source : sources) {
    if (!source.globalOrdinal) {
      outputRoot = source.root;
      continue;
    }
    if (*source.globalOrdinal >= compiled.staticGlobalMemory.globals.size())
      fail(test, "static global ordinal is out of range");
    const loom::frontend::StaticGlobalMemory &global =
        compiled.staticGlobalMemory.globals[*source.globalOrdinal];
    if (global.symbol != "lookup" ||
        global.provision != loom::frontend::StaticGlobalProvision::Image)
      fail(test, "lookup table has no exact static image");
    table = definedByteObject(global.bytes);
    tableRoot = source.root;
  }
  if (!tableRoot || !outputRoot)
    fail(test, "table and runtime output roots were not distinguished");

  loom::sim::SpatialSimulationWorkload workload{launch};
  workload.observableContract.memories.push_back(
      loom::sim::SpatialMemoryObservable{
          dataflow::LogicalMemoryRootOrViewRef{*outputRoot},
          loom::sim::MemoryObservationForm::FullState});
  auto finalizedWorkload =
      take(test, loom::sim::finalizeSimulationWorkload(workload, view));

  loom::sim::SpatialSimulationRuntimeInputDraft input{
      finalizedWorkload.identity()};
  input.memoryObjects = {std::move(table), std::move(output)};
  input.memoryRootBindings = {
      loom::sim::RuntimeMemoryBindingDraft{*tableRoot, 0, 0},
      loom::sim::RuntimeMemoryBindingDraft{*outputRoot, 1, 0}};
  auto finalizedInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                       input, finalizedWorkload, view));

  std::optional<std::uint64_t> outputObject;
  for (const loom::sim::MemoryRootBindingEntry &binding :
       finalizedInput.model().memoryRootBindings)
    if (binding.root == *outputRoot)
      outputObject = binding.binding.objectOrdinal;
  if (!outputObject)
    fail(test, "finalized runtime input lost the output root binding");

  loom::sim::DFGSimulationReport report = take(
      test, loom::sim::simulateDfgWorkload(candidate.canonicalDataflow,
                                           finalizedWorkload, finalizedInput));
  if (report.status != "pass" ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowLoad] !=
          4 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowStore] !=
          4)
    fail(test, "table workload did not execute real memory actors");

  const std::string expectedRoot =
      "memory_root" + std::to_string(*outputObject);
  std::string outputPort;
  for (const auto &[port, root] : report.finalMemoryRoots)
    if (root == expectedRoot)
      outputPort = port;
  auto finalOutput = report.finalMemoryState.find(outputPort);
  if (finalOutput == report.finalMemoryState.end() ||
      finalOutput->second !=
          llvm::SmallVector<std::string>{"i32:287454020", "i32:1432778632",
                                         "i32:4294967295", "i32:7"}) {
    std::string roots;
    for (const auto &[port, root] : report.finalMemoryRoots)
      roots += port + "=" + root + ";";
    std::string values;
    if (finalOutput != report.finalMemoryState.end())
      for (const std::string &value : finalOutput->second)
        values += value + ";";
    fail(test, "table workload produced the wrong output memory for " +
                   expectedRoot + " at port '" + outputPort +
                   "' roots=" + roots + " values=" + values);
  }

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

} // namespace

int main() {
  sourceCandidateExecutesThroughTypedDfgInput();
  staticTableExecutesThroughTypedDfgInput();
  llvm::outs() << "frontend to typed DFG integration anchor passed\n";
  return EXIT_SUCCESS;
}
