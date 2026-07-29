#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Raising/StructuredRaising.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationInputCapture.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

llvm::cl::opt<std::string>
    targetModulePath(llvm::cl::Positional,
                     llvm::cl::desc("<target LLVM IR module (.ll/.bc)>"),
                     llvm::cl::Required);

llvm::cl::opt<std::string>
    nativeModulePath("native-llvm",
                     llvm::cl::desc("host LLVM IR used as the native oracle"),
                     llvm::cl::value_desc("path"), llvm::cl::Required);

llvm::cl::opt<std::string>
    builtinName("builtin", llvm::cl::desc("builtin Fabric target preset"),
                llvm::cl::value_desc("small|default|large"),
                llvm::cl::Required);

llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("ArtifactStore directory"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);

llvm::cl::opt<std::string> outputPath("output",
                                      llvm::cl::desc("comparison report JSON"),
                                      llvm::cl::value_desc("path"),
                                      llvm::cl::Required);

llvm::cl::opt<std::string> canonicalOutputPath(
    "canonical-output",
    llvm::cl::desc("optional finalized Canonical Dataflow MLIR projection"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<unsigned>
    candidateJobs("candidate-jobs",
                  llvm::cl::desc("parallel ownership-candidate workers"),
                  llvm::cl::value_desc("count"), llvm::cl::init(1));

llvm::cl::opt<std::uint64_t>
    maxEventSteps("max-event-steps",
                  llvm::cl::desc("maximum DFG event wavefronts"),
                  llvm::cl::init(100000));

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("source_backed_dfg_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("source_backed_dfg_unsupported: ") + message);
}

int reportError(llvm::Error error) {
  llvm::errs() << "loom-dfg-run: " << llvm::toString(std::move(error)) << '\n';
  return 1;
}

llvm::Expected<std::unique_ptr<llvm::Module>>
readModule(llvm::LLVMContext &context, llvm::StringRef path) {
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(path, diagnostic, context);
  if (module)
    return std::move(module);
  std::string message;
  llvm::raw_string_ostream stream(message);
  diagnostic.print("loom-dfg-run", stream);
  return invalid(stream.str());
}

llvm::Expected<llvm::orc::ThreadSafeModule>
readNativeModule(llvm::StringRef path) {
  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      readModule(*context, path);
  if (!module)
    return module.takeError();
  if (llvm::Error error =
          loom::raising::normalizeProvenConstantCallbacks(**module))
    return std::move(error);
  if (llvm::Error error =
          loom::raising::specializeExactConstantCallbackCallSites(**module))
    return std::move(error);
  return llvm::orc::ThreadSafeModule(std::move(*module), std::move(context));
}

llvm::Expected<loom::dse::SelectedPreMappingCompilation>
compileTarget(std::unique_ptr<llvm::Module> module,
              const loom::fabric::FinalizedFabricRoot &fabric,
              const loom::ArtifactStore &store) {
  loom::frontend::PreMappingCompilationOptions compilation;
  loom::dse::PreMappingExplorationOptions exploration{
      compilation.raising,
      {compilation.lowering,
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::dse::ObjectiveDirection::Minimize, 1},
       candidateJobs}};
  llvm::Expected<loom::dse::PreMappingExplorationOutcome> outcome =
      loom::dse::exploreLlvmModuleToPreMapping(std::move(module), fabric,
                                               loom::defaultResolvedConfig(),
                                               exploration, store);
  if (!outcome)
    return outcome.takeError();
  if (const auto *incomplete =
          std::get_if<loom::dse::IncompleteSelection>(&*outcome))
    return unsupported(
        "central DSE did not complete for candidate " +
        loom::formatArtifactIdentityHex(incomplete->candidate.artifact));
  if (std::holds_alternative<loom::dse::CompletedNoFeasibleCandidate>(*outcome))
    return unsupported("central DSE found no feasible ownership candidate");
  auto completed =
      std::get<loom::dse::CompletedPreMappingSelection>(std::move(*outcome));
  if (completed.selected.size() != 1)
    return invalid("TopK(1) did not select exactly one candidate");
  return std::move(completed.selected.front());
}

llvm::Expected<mlir::LLVM::LLVMFuncOp>
requireIsolatedCallable(const dataflow::CanonicalDataflowProgramView &view,
                        dataflow::RootedGraphLaunchRef launch) {
  llvm::Expected<dataflow::CanonicalRootThreadLaunchView> root =
      view.resolve(launch.rootThreadLaunch);
  if (!root)
    return root.takeError();
  llvm::Expected<dataflow::CanonicalStaticGraphLaunchView> graphLaunch =
      view.resolve(launch.staticGraphLaunch);
  if (!graphLaunch)
    return graphLaunch.takeError();

  auto rootLaunch = llvm::dyn_cast_or_null<dataflow::ThreadLaunchOp>(root->op);
  auto thread = llvm::dyn_cast_or_null<dataflow::ThreadOp>(root->callee);
  auto staticLaunch =
      llvm::dyn_cast_or_null<dataflow::GraphLaunchOp>(graphLaunch->op);
  if (!rootLaunch || !thread || !staticLaunch)
    return invalid("rooted launch does not resolve to canonical launch ops");
  if (thread.getDomain().getKind() !=
      dataflow::ThreadDomainKind::DenseRectangular)
    return unsupported("DynamicWork workloads have no schema-1.0 capture");
  auto callable = rootLaunch->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!callable || callable.getBody().getBlocks().size() != 1)
    return unsupported("root launch is not an isolated single-block callable");

  unsigned rootLaunchCount = 0;
  unsigned resultSlotCount = 0;
  unsigned resultLoadCount = 0;
  unsigned waitCount = 0;
  for (mlir::Operation &operation : callable.getBody().front()) {
    if (&operation == rootLaunch.getOperation()) {
      ++rootLaunchCount;
      continue;
    }
    if (mlir::isa<mlir::LLVM::AddressOfOp, mlir::LLVM::UndefOp,
                  mlir::LLVM::ConstantOp, mlir::LLVM::ReturnOp>(operation))
      continue;
    if (mlir::isa<mlir::LLVM::AllocaOp>(operation)) {
      ++resultSlotCount;
      continue;
    }
    if (mlir::isa<mlir::LLVM::LoadOp>(operation)) {
      ++resultLoadCount;
      continue;
    }
    if (mlir::isa<dataflow::ThreadWaitOp>(operation)) {
      ++waitCount;
      continue;
    }
    return unsupported("root callable contains work outside the selected "
                       "Spatial launch");
  }
  if (rootLaunchCount != 1)
    return invalid("root callable does not contain its exact launch once");

  unsigned staticLaunchCount = 0;
  unsigned resultStoreCount = 0;
  for (mlir::Operation &operation : thread.getBody().front()) {
    if (&operation == staticLaunch.getOperation()) {
      ++staticLaunchCount;
      continue;
    }
    if (mlir::isa<dataflow::ThreadYieldOp>(operation))
      continue;
    if (mlir::isa<mlir::LLVM::StoreOp>(operation)) {
      ++resultStoreCount;
      continue;
    }
    return unsupported("root thread contains work outside the selected graph");
  }
  if (staticLaunchCount != 1)
    return invalid("root thread does not contain its exact graph launch once");
  const unsigned valueResultCount = staticLaunch.getValueResults().size();
  if (waitCount != 1 || resultSlotCount != valueResultCount ||
      resultLoadCount != valueResultCount ||
      resultStoreCount != valueResultCount)
    return invalid("whole-callable result wrapper is not mechanically total");

  const std::size_t functionInputs = thread.getFunctionType().getNumInputs();
  const std::size_t entryArguments = thread.getBody().front().getNumArguments();
  if (entryArguments < functionInputs + 1)
    return invalid("root thread entry arguments are malformed");
  if (entryArguments != functionInputs + 1)
    return unsupported("non-rank-zero Spatial workloads are not yet captured");
  return callable;
}

using DirectCallPath = std::vector<mlir::LLVM::CallOp>;

void collectDirectCallPaths(
    mlir::LLVM::LLVMFuncOp caller, llvm::StringRef target,
    DirectCallPath &activePath,
    llvm::SmallPtrSetImpl<mlir::Operation *> &activeCallables,
    std::vector<DirectCallPath> &paths) {
  if (!activeCallables.insert(caller.getOperation()).second)
    return;
  caller.walk([&](mlir::LLVM::CallOp call) {
    if (call->getParentOfType<mlir::LLVM::LLVMFuncOp>() != caller ||
        !call.getCalleeAttr())
      return;
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
            call, call.getCalleeAttr());
    if (!callee || callee.isExternal())
      return;
    activePath.push_back(call);
    if (callee.getSymName() == target) {
      paths.push_back(activePath);
    } else {
      collectDirectCallPaths(callee, target, activePath, activeCallables,
                             paths);
    }
    activePath.pop_back();
  });
  activeCallables.erase(caller.getOperation());
}

std::vector<DirectCallPath> findHostCallPaths(mlir::ModuleOp module,
                                              llvm::StringRef entry,
                                              llvm::StringRef target) {
  auto entryFunction = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      mlir::SymbolTable::lookupSymbolIn(module, entry));
  if (!entryFunction || entryFunction.isExternal())
    return {};
  DirectCallPath activePath;
  llvm::SmallPtrSet<mlir::Operation *, 8> activeCallables;
  std::vector<DirectCallPath> paths;
  collectDirectCallPaths(entryFunction, target, activePath, activeCallables,
                         paths);
  return paths;
}

loom::sim::RuntimeMemoryObject
capturedMemoryObject(llvm::ArrayRef<std::uint8_t> bytes) {
  loom::sim::RuntimeMemoryObject object;
  object.initialBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    object.initialBytes.push_back({loom::sim::SemanticState::Defined, byte});
  return object;
}

std::optional<std::size_t>
findCaptureBindingOrdinal(const loom::sim::SimulationInputCapturePlan &plan,
                          dataflow::LogicalMemoryRootRef root) {
  for (auto [ordinal, binding] : llvm::enumerate(plan.memoryRootBindings))
    if (binding.root == root)
      return ordinal;
  return std::nullopt;
}

struct ExecutionTotals {
  std::uint64_t dynamicCalls = 0;
  std::uint64_t valueLanesCompared = 0;
  std::uint64_t memoryBytesCompared = 0;
  std::uint64_t floatingVarianceBytes = 0;
  std::uint64_t wavefrontSteps = 0;
  std::uint64_t eventCount = 0;
  double simulationSeconds = 0.0;
  std::map<dataflow::OperationSchemaId, std::uint64_t> operationFirings;
};

bool equalSemanticLane(const loom::sim::SemanticLane &lhs,
                       const loom::sim::SemanticLane &rhs) {
  return lhs.state == rhs.state &&
         (lhs.state != loom::sim::SemanticState::Defined ||
          lhs.bits == rhs.bits);
}

bool equalValueSequence(const loom::sim::CanonicalValueSequence &lhs,
                        const loom::sim::CanonicalValueSequence &rhs) {
  return lhs.tokenCount == rhs.tokenCount &&
         lhs.lanes.size() == rhs.lanes.size() &&
         llvm::equal(lhs.lanes, rhs.lanes, equalSemanticLane);
}

bool equalRuntimeValues(llvm::ArrayRef<loom::sim::RuntimeValueEntry> lhs,
                        llvm::ArrayRef<loom::sim::RuntimeValueEntry> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs, [](const auto &left, const auto &right) {
           return left.valueInputOrdinal == right.valueInputOrdinal &&
                  equalValueSequence(left.value, right.value);
         });
}

bool equalNativeCapture(const loom::sim::NativeSimulationInputCapture &lhs,
                        const loom::sim::NativeSimulationInputCapture &rhs,
                        bool includeFinalMemory,
                        const loom::sim::SimulationInputCapturePlan &plan) {
  std::vector<bool> requiresInitialState(plan.objects.size(), false);
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       plan.memoryRootBindings) {
    if (binding.objectIndex >= requiresInitialState.size())
      return false;
    requiresInitialState[binding.objectIndex] =
        requiresInitialState[binding.objectIndex] ||
        binding.requiresInitialState;
  }
  if (lhs.entryResult != rhs.entryResult ||
      lhs.calls.size() != rhs.calls.size())
    return false;
  for (auto [leftCall, rightCall] : llvm::zip_equal(lhs.calls, rhs.calls)) {
    if (!equalRuntimeValues(leftCall.runtimeValues, rightCall.runtimeValues) ||
        leftCall.memoryRootByteOffsets !=
            rightCall.memoryRootByteOffsets ||
        leftCall.objects.size() != rightCall.objects.size())
      return false;
    for (auto [objectOrdinal, objects] : llvm::enumerate(
             llvm::zip_equal(leftCall.objects, rightCall.objects))) {
      const auto &[leftObject, rightObject] = objects;
      if (requiresInitialState[objectOrdinal] &&
          leftObject.initialBytes != rightObject.initialBytes)
        return false;
      if (includeFinalMemory && leftObject.finalBytes != rightObject.finalBytes)
        return false;
    }
  }
  return true;
}

llvm::Error compareValueObservations(
    const loom::sim::SpatialSimulationWorkload &workload,
    const loom::sim::SpatialFunctionalObservations &observations,
    const loom::sim::SimulationInputCapturePlan &plan,
    const loom::sim::NativeSimulationCallCapture &selectedNative,
    ExecutionTotals &totals) {
  if (workload.observableContract.valueResults.size() !=
          plan.valueResults.size() ||
      observations.valueResults.size() != plan.valueResults.size() ||
      selectedNative.valueResults.size() != plan.valueResults.size())
    return invalid("DFG and native value-result observation counts differ");
  for (auto [position, projected] : llvm::enumerate(plan.valueResults)) {
    if (projected.valueResultOrdinal !=
        workload.observableContract.valueResults[position])
      return invalid("value-result capture order differs from its workload");
    const auto *published = std::get_if<loom::sim::PublishedValueResult>(
        &observations.valueResults[position]);
    if (!published)
      return invalid("retired DFG execution did not publish a selected value");
    if (!equalValueSequence(published->value,
                            selectedNative.valueResults[position]))
      return invalid(llvm::formatv(
          "DFG value result {0} differs from selected-decision native "
          "execution",
          projected.valueResultOrdinal));
    totals.valueLanesCompared += published->value.lanes.size();
  }
  return llvm::Error::success();
}

llvm::Error compareMemoryObservations(
    const loom::sim::SpatialSimulationWorkload &workload,
    const loom::sim::SpatialFunctionalObservations &observations,
    const loom::sim::SimulationInputCapturePlan &plan,
    const loom::sim::NativeSimulationCallCapture &selectedNative,
    const loom::sim::NativeSimulationCallCapture *sourceNative,
    bool allowFloatingVariance, ExecutionTotals &totals) {
  if (observations.memories.size() !=
      workload.observableContract.memories.size())
    return invalid("DFG execution returned the wrong memory observation count");
  std::vector<std::vector<bool>> floatingCoverage;
  floatingCoverage.reserve(selectedNative.objects.size());
  for (const loom::sim::NativeCapturedMemoryObject &object :
       selectedNative.objects)
    floatingCoverage.emplace_back(object.finalBytes.size(), false);

  for (auto [observable, payload] : llvm::zip_equal(
           workload.observableContract.memories, observations.memories)) {
    const auto *rootOrView =
        std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&observable.target);
    if (!rootOrView)
      return invalid("native comparison received a memory exposure");
    const auto *root = std::get_if<dataflow::LogicalMemoryRootRef>(rootOrView);
    const auto *full = std::get_if<loom::sim::FullMemoryObservation>(&payload);
    if (!root || !full)
      return invalid("native comparison requires full logical-root state");
    std::optional<std::size_t> bindingOrdinal =
        findCaptureBindingOrdinal(plan, *root);
    if (!bindingOrdinal ||
        *bindingOrdinal >= selectedNative.memoryRootByteOffsets.size())
      return invalid("memory observation has no native capture binding");
    const loom::sim::SimulationMemoryRootCapture &binding =
        plan.memoryRootBindings[*bindingOrdinal];
    if (binding.objectIndex >= selectedNative.objects.size())
      return invalid("memory observation references an absent native object");
    const std::uint64_t byteOffset =
        selectedNative.memoryRootByteOffsets[*bindingOrdinal];
    const std::vector<std::uint8_t> &expected =
        selectedNative.objects[binding.objectIndex].finalBytes;
    if (byteOffset > expected.size() ||
        full->bytes.size() > expected.size() - byteOffset)
      return invalid("memory observation exceeds its native captured object");
    llvm::ArrayRef<std::uint8_t> expectedRoot(expected);
    expectedRoot = expectedRoot.slice(byteOffset, full->bytes.size());
    for (auto [ordinal, actualByte] : llvm::enumerate(full->bytes)) {
      if (actualByte.state == loom::sim::SemanticState::Defined &&
          actualByte.value == expectedRoot[ordinal])
        continue;
      return invalid(llvm::formatv(
          "DFG memory observation differs from selected-decision native "
          "execution at "
          "capture object {0}, root-relative byte {1}, object byte {2}: "
          "state={3}, actual={4:X2}, expected={5:X2}",
          binding.objectIndex, ordinal, byteOffset + ordinal,
          static_cast<std::uint32_t>(actualByte.state), actualByte.value,
          expectedRoot[ordinal]));
    }
    if (binding.floatingWriteLaneType) {
      std::vector<bool> &coverage = floatingCoverage[binding.objectIndex];
      for (std::uint64_t ordinal = 0; ordinal < full->bytes.size(); ++ordinal)
        coverage[byteOffset + ordinal] = true;
    }
    totals.memoryBytesCompared += full->bytes.size();
  }

  if (!sourceNative)
    return llvm::Error::success();
  if (sourceNative->objects.size() != selectedNative.objects.size())
    return invalid(
        "source and selected native captures have different objects");
  for (auto [objectOrdinal, pair] : llvm::enumerate(
           llvm::zip_equal(sourceNative->objects, selectedNative.objects))) {
    const auto &[sourceObject, selectedObject] = pair;
    if (sourceObject.initialBytes.size() != sourceObject.finalBytes.size() ||
        selectedObject.initialBytes.size() !=
            selectedObject.finalBytes.size() ||
        sourceObject.finalBytes.size() != selectedObject.finalBytes.size())
      return invalid("source and selected native object extents differ");
    for (std::uint64_t byte = 0; byte < sourceObject.finalBytes.size();
         ++byte) {
      if (sourceObject.finalBytes[byte] == selectedObject.finalBytes[byte])
        continue;
      const bool sourceChanged =
          sourceObject.initialBytes[byte] != sourceObject.finalBytes[byte];
      const bool selectedChanged =
          selectedObject.initialBytes[byte] != selectedObject.finalBytes[byte];
      // Output-only objects may start with different concrete stack bytes in
      // two independent JIT executions. A byte unchanged by both executions
      // is an environment difference, not a compiler semantic difference.
      if (!sourceChanged && !selectedChanged)
        continue;
      if (!allowFloatingVariance)
        return invalid(
            "selected native execution differs without a typed floating "
            "execution decision");
      if (!floatingCoverage[objectOrdinal][byte])
        return invalid(llvm::formatv(
            "selected floating decision changed non-floating memory at "
            "capture object {0}, byte {1}",
            objectOrdinal, byte));
      ++totals.floatingVarianceBytes;
    }
  }
  return llvm::Error::success();
}

llvm::Error
validateCapturableGraph(const dataflow::CanonicalDataflowProgramView &view,
                        dataflow::RootedGraphLaunchRef launch) {
  llvm::Expected<dataflow::GraphRef> graphRef = view.resolve(launch);
  if (!graphRef)
    return graphRef.takeError();
  llvm::Expected<dataflow::CanonicalGraphView> graphView =
      view.resolve(*graphRef);
  if (!graphView)
    return graphView.takeError();
  auto graph = mlir::cast<dataflow::GraphOp>(graphView->op);
  llvm::ArrayRef<std::int32_t> inputSegments = graph.getInputSegmentSizes();
  if (inputSegments.size() != 3)
    return invalid("canonical graph input segments are malformed");
  if (inputSegments[1] != 0)
    return unsupported(
        "native comparison does not yet capture graph stream inputs");
  return llvm::Error::success();
}

llvm::Error
executeCapturedInputPlan(const loom::frontend::PreMappingCompilation &compiled,
                         const dataflow::CanonicalDataflowProgramView &view,
                         dataflow::RootedGraphLaunchRef launch,
                         const loom::sim::SimulationInputCapturePlan &plan,
                         const loom::sim::NativeSimulationInputCapture &capture,
                         const loom::sim::NativeSimulationInputCapture *source,
                         bool allowFloatingVariance, ExecutionTotals &totals) {
  if (plan.valueResults.empty() && plan.memoryRootBindings.empty())
    return unsupported(
        "Spatial execution has no externally observable value or memory");
  if (capture.entryResult != 0)
    return invalid("native oracle entry returned a nonzero status");
  if (source && !equalNativeCapture(*source, capture, false, plan))
    return invalid(
        "source and selected-decision native executions have different inputs");
  if (capture.calls.empty())
    return llvm::Error::success();

  loom::sim::SpatialSimulationWorkload workload{launch};
  for (const loom::sim::SimulationValueInputCapture &value : plan.valueInputs) {
    if (value.valueInputOrdinal != workload.valueInputPlan.size())
      return invalid("capture value inputs are not dense in graph ABI order");
    if (value.fixedValue)
      workload.valueInputPlan.push_back(*value.fixedValue);
    else
      workload.valueInputPlan.push_back(loom::sim::RuntimeValueInput{});
  }
  for (const loom::sim::SimulationValueResultCapture &value :
       plan.valueResults) {
    if (value.valueResultOrdinal !=
        workload.observableContract.valueResults.size())
      return invalid("capture value results are not dense in graph ABI order");
    workload.observableContract.valueResults.push_back(
        value.valueResultOrdinal);
  }
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       plan.memoryRootBindings)
    workload.observableContract.memories.push_back(
        loom::sim::SpatialMemoryObservable{
            dataflow::LogicalMemoryRootOrViewRef{binding.root},
            loom::sim::MemoryObservationForm::FullState});
  llvm::Expected<loom::sim::CanonicalSimulationWorkload> finalizedWorkload =
      loom::sim::finalizeSimulationWorkload(workload, view);
  if (!finalizedWorkload)
    return finalizedWorkload.takeError();

  for (auto [callOrdinal, dynamicCall] : llvm::enumerate(capture.calls)) {
    if (dynamicCall.objects.size() != plan.objects.size() ||
        dynamicCall.memoryRootByteOffsets.size() !=
            plan.memoryRootBindings.size())
      return invalid("native capture object count changed during execution");
    loom::sim::SpatialSimulationRuntimeInputDraft input{
        finalizedWorkload->identity()};
    input.runtimeValues = dynamicCall.runtimeValues;
    for (const loom::sim::NativeCapturedMemoryObject &object :
         dynamicCall.objects)
      input.memoryObjects.push_back(capturedMemoryObject(object.initialBytes));
    for (auto [bindingOrdinal, binding] :
         llvm::enumerate(plan.memoryRootBindings))
      input.memoryRootBindings.push_back(loom::sim::RuntimeMemoryBindingDraft{
          binding.root, binding.objectIndex,
          dynamicCall.memoryRootByteOffsets[bindingOrdinal]});
    llvm::Expected<loom::sim::CanonicalSimulationRuntimeInput> finalizedInput =
        loom::sim::finalizeSimulationRuntimeInput(input, *finalizedWorkload,
                                                  view);
    if (!finalizedInput)
      return finalizedInput.takeError();

    const auto started = std::chrono::steady_clock::now();
    llvm::Expected<loom::sim::RetiredDFGSimulation> execution =
        loom::sim::simulateRetiredDfgWorkload(compiled.canonicalDataflow,
                                              *finalizedWorkload,
                                              *finalizedInput, maxEventSteps);
    const auto stopped = std::chrono::steady_clock::now();
    if (!execution)
      return execution.takeError();
    totals.simulationSeconds +=
        std::chrono::duration<double>(stopped - started).count();
    totals.wavefrontSteps += execution->report.wavefrontSteps;
    totals.eventCount += execution->report.eventCount;
    for (const auto &[schema, count] : execution->report.operationFireCounts)
      totals.operationFirings[schema] += count;
    if (llvm::Error error = compareValueObservations(finalizedWorkload->model(),
                                                     execution->observations,
                                                     plan, dynamicCall, totals))
      return error;
    if (llvm::Error error = compareMemoryObservations(
            finalizedWorkload->model(), execution->observations, plan,
            dynamicCall, source ? &source->calls[callOrdinal] : nullptr,
            allowFloatingVariance, totals))
      return error;
    ++totals.dynamicCalls;
  }
  return llvm::Error::success();
}

llvm::Expected<loom::sim::NativeSimulationInputCapture>
executeSelectedDirectCallCapture(
    const loom::sim::DirectCallSimulationInputCapturePlan &plan,
    const loom::dse::StructuredOwnershipDerivation &derivation) {
  llvm::LLVMContext context;
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      readModule(context, nativeModulePath);
  if (!module)
    return module.takeError();
  llvm::Expected<loom::frontend::StructuredProgramCandidate> host =
      loom::raising::raiseLlvmModuleToStructuredProgram(std::move(*module));
  if (!host)
    return host.takeError();
  llvm::Expected<loom::frontend::StructuredProgramCandidateView> hostView =
      host->view();
  if (!hostView)
    return hostView.takeError();

  std::optional<loom::frontend::StructuredEntityRef> callable;
  for (const loom::frontend::StructuredEntity &entity :
       hostView->entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (!function || plan.invocationPath.empty() ||
        function.getSymName() != plan.invocationPath.back().hostCalleeSymbol)
      continue;
    if (callable)
      return invalid("native module has duplicate selected callable symbols");
    callable = entity.reference;
  }
  if (!callable)
    return invalid("native module has no selected callable symbol");

  llvm::Expected<loom::frontend::PreparedSpatialOwnershipSelection> prepared =
      loom::frontend::prepareSpatialOwnershipSelection(*host, {*callable},
                                                       derivation.decision);
  if (!prepared)
    return prepared.takeError();
  llvm::Expected<llvm::orc::ThreadSafeModule> hostModule =
      readNativeModule(nativeModulePath);
  if (!hostModule)
    return hostModule.takeError();
  return loom::sim::executeStructuredDirectCallSimulationInputCapture(
      std::move(*hostModule), std::move(prepared->module), plan);
}

llvm::Error executeDirectCallLaunch(
    const loom::frontend::PreMappingCompilation &compiled,
    const dataflow::CanonicalDataflowProgramView &view,
    dataflow::RootedGraphLaunchRef launch,
    const loom::dse::StructuredOwnershipDerivation &derivation,
    ExecutionTotals &totals) {
  llvm::Expected<mlir::LLVM::LLVMFuncOp> callable =
      requireIsolatedCallable(view, launch);
  if (!callable)
    return callable.takeError();

  std::vector<DirectCallPath> invocationPaths = findHostCallPaths(
      compiled.canonicalDataflow.module(), "main", callable->getSymName());
  if (invocationPaths.empty())
    return unsupported(
        "isolated Spatial callable has no direct path from main");

  for (const DirectCallPath &invocationPath : invocationPaths) {
    llvm::Expected<loom::sim::DirectCallSimulationInputCapturePlan> plan =
        loom::sim::deriveSimulationInputCapturePlan(view, launch,
                                                    invocationPath);
    if (!plan)
      return plan.takeError();
    llvm::Expected<llvm::orc::ThreadSafeModule> nativeModule =
        readNativeModule(nativeModulePath);
    if (!nativeModule)
      return nativeModule.takeError();
    llvm::Expected<loom::sim::NativeSimulationInputCapture> sourceCapture =
        loom::sim::executeNativeSimulationInputCapture(std::move(*nativeModule),
                                                       *plan);
    if (!sourceCapture)
      return sourceCapture.takeError();
    llvm::Expected<loom::sim::NativeSimulationInputCapture> selectedCapture =
        executeSelectedDirectCallCapture(*plan, derivation);
    if (!selectedCapture)
      return selectedCapture.takeError();

    if (llvm::Error error = executeCapturedInputPlan(
            compiled, view, launch, plan->input, *selectedCapture,
            &*sourceCapture,
            derivation.decision.fmuladdExecutionShape.has_value(), totals))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error executeOperationLaunch(
    const loom::frontend::PreMappingCompilation &compiled,
    const dataflow::CanonicalDataflowProgramView &view,
    dataflow::RootedGraphLaunchRef launch,
    const loom::dse::StructuredOwnershipDerivation &derivation,
    const loom::frontend::StructuredProgramCandidate &parent,
    ExecutionTotals &totals) {
  llvm::Expected<loom::frontend::PreparedSpatialOwnershipSelection> prepared =
      loom::frontend::prepareSpatialOwnershipSelection(parent, derivation.scope,
                                                       derivation.decision);
  if (!prepared)
    return prepared.takeError();
  auto callable =
      prepared->operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!callable)
    return invalid("selected operation has no enclosing LLVM callable");

  if (callable.getSymName() == "main") {
    llvm::Expected<loom::sim::OperationSimulationInputCapturePlan> plan =
        loom::sim::deriveOperationSimulationInputCapturePlan(
            view, launch, prepared->liveIns, prepared->liveOuts);
    if (!plan)
      return plan.takeError();
    llvm::Expected<loom::sim::NativeSimulationInputCapture> capture =
        loom::sim::executeStructuredSimulationInputCapture(
            std::move(prepared->module), prepared->operation, *plan);
    if (!capture)
      return capture.takeError();
    return executeCapturedInputPlan(compiled, view, launch, plan->input,
                                    *capture, nullptr, false, totals);
  }

  std::vector<DirectCallPath> invocationPaths =
      findHostCallPaths(*prepared->module, "main", callable.getSymName());
  if (invocationPaths.empty())
    return unsupported(
        "operation-owned Spatial callable has no direct path from main");

  for (const DirectCallPath &invocationPath : invocationPaths) {
    mlir::IRMapping mapping;
    mlir::OwningOpRef<mlir::ModuleOp> clone(llvm::cast<mlir::ModuleOp>(
        prepared->module->getOperation()->clone(mapping)));
    mlir::Operation *operation = mapping.lookup(prepared->operation);
    std::vector<mlir::Value> liveIns;
    liveIns.reserve(prepared->liveIns.size());
    for (mlir::Value value : prepared->liveIns)
      liveIns.push_back(mapping.lookup(value));
    std::vector<mlir::Value> liveOuts;
    liveOuts.reserve(prepared->liveOuts.size());
    for (mlir::Value value : prepared->liveOuts)
      liveOuts.push_back(mapping.lookup(value));
    DirectCallPath clonedPath;
    clonedPath.reserve(invocationPath.size());
    for (mlir::LLVM::CallOp call : invocationPath)
      clonedPath.push_back(
          llvm::cast<mlir::LLVM::CallOp>(mapping.lookup(call.getOperation())));

    llvm::Expected<loom::sim::OperationSimulationInputCapturePlan> plan =
        loom::sim::deriveOperationSimulationInputCapturePlan(
            view, launch, liveIns, liveOuts, clonedPath);
    if (!plan)
      return plan.takeError();
    llvm::Expected<loom::sim::NativeSimulationInputCapture> capture =
        loom::sim::executeStructuredSimulationInputCapture(std::move(clone),
                                                           operation, *plan);
    if (!capture)
      return capture.takeError();
    if (llvm::Error error =
            executeCapturedInputPlan(compiled, view, launch, plan->input,
                                     *capture, nullptr, false, totals))
      return llvm::Error(std::move(error));
  }
  return llvm::Error::success();
}

llvm::Error
executeLaunch(const loom::dse::SelectedPreMappingCompilation &selected,
              const dataflow::CanonicalDataflowProgramView &view,
              dataflow::RootedGraphLaunchRef launch,
              const loom::ArtifactStore &store, ExecutionTotals &totals) {
  if (llvm::Error error = validateCapturableGraph(view, launch))
    return error;
  if (selected.derivations.size() != 1)
    return unsupported(
        "selected Spatial graph has no unique ownership lineage");
  const loom::dse::StructuredOwnershipDerivation &derivation =
      selected.derivations.front();
  const loom::ArtifactRootReference parentReference{
      loom::frontend::structuredProgramArtifactSchema.identity.str(),
      loom::frontend::structuredProgramArtifactSchema.version,
      derivation.scope.selection.parent};
  llvm::Expected<loom::frontend::StructuredProgramCandidate> parent =
      loom::frontend::importStructuredProgram(parentReference, store);
  if (!parent)
    return parent.takeError();
  auto parentView = parent->view();
  if (!parentView)
    return parentView.takeError();
  auto selection = parentView->resolve(derivation.scope.selection);
  if (!selection)
    return selection.takeError();
  if (llvm::isa_and_nonnull<mlir::LLVM::LLVMFuncOp>(selection->operation))
    return executeDirectCallLaunch(selected.compilation, view, launch,
                                   derivation, totals);
  return executeOperationLaunch(selected.compilation, view, launch, derivation,
                                *parent, totals);
}

llvm::Error writeReport(llvm::StringRef path, std::uint64_t graphCount,
                        std::uint64_t actorCount,
                        const ExecutionTotals &totals) {
  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty())
    if (std::error_code error = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(error, "cannot create %s", parent.c_str());

  llvm::json::Object firings;
  for (const auto &[schema, count] : totals.operationFirings)
    firings[dataflow::operationSchemaSpelling(schema)] = count;
  llvm::json::Object root;
  root["kind"] = "source_backed_dfg_comparison";
  root["status"] = "pass";
  root["graphs"] = graphCount;
  root["actors"] = actorCount;
  root["dynamic_calls"] = totals.dynamicCalls;
  root["value_lanes_compared"] = totals.valueLanesCompared;
  root["memory_bytes_compared"] = totals.memoryBytesCompared;
  root["floating_variance_bytes"] = totals.floatingVarianceBytes;
  root["floating_variance_kind"] =
      totals.floatingVarianceBytes == 0 ? "none" : "selected_decision_replay";
  root["wavefront_steps"] = totals.wavefrontSteps;
  root["event_count"] = totals.eventCount;
  root["simulation_seconds"] = totals.simulationSeconds;
  root["wavefront_steps_per_second"] =
      totals.simulationSeconds > 0.0
          ? static_cast<double>(totals.wavefrontSteps) /
                totals.simulationSeconds
          : 0.0;
  root["operation_firings"] = std::move(firings);

  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "cannot open %s", path.str().c_str());
  output << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << '\n';
  return llvm::Error::success();
}

llvm::Error
writeCanonicalDataflow(llvm::StringRef path,
                       const dataflow::CanonicalDataflowArtifact &canonical) {
  if (path.empty())
    return llvm::Error::success();
  std::string message;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(path, &message);
  if (!output)
    return invalid("cannot open canonical output: " + message);
  canonical.module()->print(output->os());
  output->os() << '\n';
  output->keep();
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Compile one LLVM module to Canonical Dataflow and compare each exact "
      "capturable Spatial invocation with an independent native execution.\n");
  if (candidateJobs == 0)
    return reportError(invalid("candidate-jobs must be positive"));

  llvm::Expected<loom::adg::BuiltinTargetPreset> preset =
      loom::adg::parseBuiltinTargetPreset(builtinName);
  if (!preset)
    return reportError(preset.takeError());
  loom::ArtifactStore store(artifactStorePath);
  auto design = loom::adg::buildBuiltinTarget(store, *preset);
  if (!design)
    return reportError(design.takeError());
  if (design->roots().size() != 1)
    return reportError(invalid("builtin target has no unique Fabric root"));

  llvm::LLVMContext targetContext;
  llvm::Expected<std::unique_ptr<llvm::Module>> target =
      readModule(targetContext, targetModulePath);
  if (!target)
    return reportError(target.takeError());
  llvm::Expected<loom::dse::SelectedPreMappingCompilation> selected =
      compileTarget(std::move(*target), design->roots().front(), store);
  if (!selected)
    return reportError(selected.takeError());
  loom::frontend::PreMappingCompilation &compiled = selected->compilation;
  llvm::Expected<dataflow::CanonicalDataflowProgramView> view =
      compiled.canonicalDataflow.view();
  if (!view)
    return reportError(view.takeError());
  if (llvm::Error error = writeCanonicalDataflow(canonicalOutputPath,
                                                 compiled.canonicalDataflow))
    return reportError(std::move(error));

  std::vector<dataflow::RootedGraphLaunchRef> launches;
  view->forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
    launches.push_back(launch);
  });
  if (launches.empty())
    return reportError(unsupported("selected program is graph-free"));

  ExecutionTotals totals;
  for (dataflow::RootedGraphLaunchRef launch : launches)
    if (llvm::Error error =
            executeLaunch(*selected, *view, launch, store, totals))
      return reportError(std::move(error));
  if (totals.dynamicCalls == 0 ||
      (totals.valueLanesCompared == 0 && totals.memoryBytesCompared == 0) ||
      totals.eventCount == 0)
    return reportError(invalid("execution produced no substantive workload"));
  if (llvm::Error error = writeReport(outputPath, view->graphs().size(),
                                      view->actors().size(), totals))
    return reportError(std::move(error));
  return 0;
}
