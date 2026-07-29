#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "DSE/Promotion.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Evaluation/StandardFindings.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredOwnershipEvaluation: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<llvm::Module> parseModule(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
define void @kernel(ptr %a, ptr %b, ptr %c) {
entry:
  %lhs = load float, ptr %a, align 4
  %rhs = load float, ptr %b, align 4
  %sum = fadd float %lhs, %rhs
  store float %sum, ptr %c, align 4
  ret void
}

define void @cold(ptr %a, ptr %b, ptr %c) {
entry:
  %lhs = load float, ptr %a, align 4
  %rhs = load float, ptr %b, align 4
  %sum = fadd float %lhs, %rhs
  store float %sum, ptr %c, align 4
  ret void
}

define i32 @tiny() {
entry:
  ret i32 7
}

define i32 @main(ptr %a, ptr %b, ptr %c) {
entry:
  call void @kernel(ptr %a, ptr %b, ptr %c)
  %ignored = call i32 @tiny()
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<evaluation>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("structuredOwnershipEvaluation", stream);
    fail(stream.str());
  }
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail("cannot initialize the native target");
  auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
  module->setDataLayout(take(target.getDefaultDataLayoutForTarget()));
  module->setTargetTriple(llvm::Triple("riscv64-unknown-unknown-elf"));
  return module;
}

std::unique_ptr<llvm::Module>
parseFunctionallyIncorrectModule(llvm::LLVMContext &context) {
  std::unique_ptr<llvm::Module> module = parseModule(context);
  llvm::Function *kernel = module->getFunction("kernel");
  if (!kernel)
    fail("incorrect candidate lost kernel");
  for (llvm::BasicBlock &block : *kernel) {
    for (llvm::Instruction &instruction : llvm::make_early_inc_range(block)) {
      auto *add = llvm::dyn_cast<llvm::BinaryOperator>(&instruction);
      if (!add || add->getOpcode() != llvm::Instruction::FAdd)
        continue;
      llvm::IRBuilder<> builder(add);
      llvm::Value *subtract =
          builder.CreateFSub(add->getOperand(0), add->getOperand(1));
      add->replaceAllUsesWith(subtract);
      add->eraseFromParent();
      return module;
    }
  }
  fail("incorrect candidate found no floating addition");
}

loom::frontend::StructuredEntityRef
findCallable(const loom::frontend::StructuredProgramCandidate &candidate,
             llvm::StringRef name) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getSymName() == name)
      return entity.reference;
  }
  fail("callable is absent from the Structured Program: " + name.str());
}

loom::sim::RuntimeMemoryObject zeroedMemory(std::size_t byteCount) {
  return loom::sim::RuntimeMemoryObject{
      std::vector<loom::sim::SemanticMemoryByte>(
          byteCount, {loom::sim::SemanticState::Defined, std::uint8_t{0}})};
}

loom::sim::RuntimeMemoryObject f32Memory(float value) {
  llvm::APInt bits = llvm::APFloat(value).bitcastToAPInt();
  std::vector<loom::sim::SemanticMemoryByte> bytes;
  bytes.reserve(4);
  for (unsigned byte = 0; byte < 4; ++byte)
    bytes.push_back(
        {loom::sim::SemanticState::Defined,
         static_cast<std::uint8_t>(bits.extractBitsAsZExtValue(8, byte * 8))});
  return loom::sim::RuntimeMemoryObject{std::move(bytes)};
}

struct SourceSimulationInputs final {
  loom::sim::CanonicalSimulationWorkload workload;
  loom::sim::CanonicalSimulationRuntimeInput runtimeInput;
  loom::ArtifactRootReference workloadReference;
  loom::ArtifactRootReference runtimeInputReference;
  loom::sim::NativeStructuredProgramObservations observations;
};

SourceSimulationInputs makeSourceSimulationInputs(
    const loom::frontend::StructuredProgramCandidate &source,
    const loom::ArtifactStore &store) {
  auto view = take(source.view());
  loom::sim::StructuredProgramSimulationWorkload draft{
      findCallable(source, "main")};
  draft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{},
                        loom::sim::StructuredRuntimeMemoryInput{},
                        loom::sim::StructuredRuntimeMemoryInput{}};
  draft.observableContract.returnValue = true;
  draft.observableContract.memories.push_back(
      {loom::sim::EntryPointerArgumentTarget{2},
       loom::sim::MemoryObservationForm::FullState});
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));

  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtime{
      workload.identity()};
  runtime.memoryObjects = {f32Memory(3.0F), f32Memory(2.0F), zeroedMemory(4)};
  runtime.pointerBindings = {{0, 0, 0}, {1, 1, 0}, {2, 2, 0}};
  auto runtimeInput =
      take(loom::sim::finalizeSimulationRuntimeInput(runtime, workload, view));
  auto workloadReference =
      take(loom::sim::publishSimulationWorkload(workload, store));
  auto runtimeInputReference =
      take(loom::sim::publishSimulationRuntimeInput(runtimeInput, store));
  auto observations = take(loom::sim::executeNativeStructuredProgram(
      source, workload, runtimeInput));
  return {std::move(workload), std::move(runtimeInput),
          std::move(workloadReference), std::move(runtimeInputReference),
          std::move(observations)};
}

loom::evaluation::DecimalValue
metricResult(const loom::evaluation::EvaluationRequest &request,
             const loom::evaluation::EvaluationEvidence &evidence,
             loom::evaluation::MetricKind kind) {
  const auto *completed =
      std::get_if<loom::evaluation::CompletedEvidence>(&evidence.outcome());
  if (!completed ||
      completed->metricResults.size() != request.metricRequests().size())
    fail("analytic model did not return a total metric result vector");
  std::optional<std::size_t> ordinal;
  for (std::size_t index = 0; index < request.metricRequests().size(); ++index)
    if (request.metricRequests()[index].query().metric == kind) {
      ordinal = index;
      break;
    }
  if (!ordinal)
    fail("analytic model request omitted " +
         loom::evaluation::toString(kind).str());
  const loom::evaluation::MetricResult &result =
      completed->metricResults[*ordinal];
  if (result.uncertainty != loom::evaluation::UncertaintyKind::Unquantified)
    fail("analytic model presented its estimate as ground truth");
  const auto *point =
      std::get_if<loom::evaluation::PointObservation>(&result.observation);
  if (!point)
    fail("analytic model did not return a point estimate");
  const auto *value =
      std::get_if<loom::evaluation::DecimalValue>(&point->value);
  if (!value)
    fail("analytic metric result used the wrong numeric domain");
  return *value;
}

struct EvaluatedRuntime final {
  loom::evaluation::DecimalValue value;
  loom::evaluation::EvaluationRequest request;
  loom::evaluation::EvaluationEvidence evidence;
};

struct EvaluatedFunctional final {
  loom::evaluation::EvaluationRequest request;
  loom::evaluation::EvaluationEvidence evidence;
  loom::evaluation::FindingRequestOrdinal functionalMismatchRequest;
};

EvaluatedRuntime
evaluateStructuredRuntime(const loom::ArtifactRootReference &structuredProgram,
                          const loom::ArtifactRootReference &fabric,
                          const loom::ArtifactRootReference &workload,
                          const loom::ArtifactRootReference &runtimeInput,
                          const loom::ArtifactStore &store) {
  auto prepared =
      take(loom::evaluation::models::prepareStructuredFabricEvaluation(
          structuredProgram, fabric, workload, runtimeInput,
          loom::defaultResolvedConfig(), store));
  auto evidence = take(loom::evaluation::evaluateRequest(
      prepared.request, prepared.resolution, store));
  return EvaluatedRuntime{metricResult(prepared.request, evidence,
                                       loom::evaluation::MetricKind::Runtime),
                          std::move(prepared.request), std::move(evidence)};
}

EvaluatedFunctional evaluateStructuredFunctional(
    const loom::ArtifactRootReference &structuredProgram,
    const loom::ArtifactRootReference &workload,
    const loom::ArtifactRootReference &runtimeInput,
    const loom::ArtifactStore &store) {
  auto prepared = take(
      loom::evaluation::models::prepareStructuredProgramFunctionalEvaluation(
          structuredProgram, workload, runtimeInput,
          loom::defaultResolvedConfig(), store));
  auto evidence = take(loom::evaluation::evaluateRequest(
      prepared.request, prepared.resolution, store));
  return {std::move(prepared.request), std::move(evidence),
          prepared.functionalMismatchRequest};
}

loom::evaluation::FindingResultForm
functionalMismatchResult(const loom::evaluation::EvaluationRequest &request,
                         const loom::evaluation::EvaluationEvidence &evidence) {
  const auto *completed =
      std::get_if<loom::evaluation::CompletedEvidence>(&evidence.outcome());
  if (!completed ||
      completed->findingResults.size() != request.findingRequests().size())
    fail("structured model did not return total finding results");
  for (std::size_t index = 0; index < request.findingRequests().size(); ++index)
    if (request.findingRequests()[index].query().kind ==
        loom::evaluation::standard_findings::FunctionalMismatch)
      return loom::evaluation::findingResultForm(
          completed->findingResults[index].result);
  fail("structured model omitted functional mismatch");
}

loom::evaluation::DecimalValue
evaluateCanonicalDataflowRuntime(const loom::ArtifactRootReference &program,
                                 const loom::ArtifactRootReference &fabric,
                                 const loom::ArtifactStore &store) {
  auto prepared =
      take(loom::evaluation::models::prepareCanonicalDataflowFabricEvaluation(
          program, fabric, loom::defaultResolvedConfig(), store));
  auto evidence = take(loom::evaluation::evaluateRequest(
      prepared.request, prepared.resolution, store));
  return metricResult(prepared.request, evidence,
                      loom::evaluation::MetricKind::Runtime);
}

void verifyStagedOwnershipEvidence(
    const loom::dse::CompletedPreMappingSelection &selection,
    const loom::ArtifactRootReference &source,
    const loom::ArtifactRootReference &profitableCandidate,
    llvm::ArrayRef<loom::ArtifactRootReference> unprofitableCandidates,
    llvm::ArrayRef<loom::ArtifactRootReference> inapplicableCandidates,
    const loom::ArtifactRootReference &fabric,
    const loom::ArtifactRootReference &workload,
    const loom::ArtifactRootReference &runtimeInput,
    const loom::ArtifactStore &store) {
  std::map<loom::ArtifactRootReference,
           std::vector<loom::ArtifactRootReference>,
           decltype(&loom::artifactRootReferenceLess)>
      closures(&loom::artifactRootReferenceLess);
  closures[source];
  closures[fabric];
  closures[workload] = {source};
  closures[runtimeInput] = {source, workload};
  for (const loom::dse::StructuredOwnershipCandidateDisposition &disposition :
       selection.dispositions)
    if (const auto *candidate =
            std::get_if<loom::ArtifactRootReference>(&disposition.result))
      closures[*candidate];

  std::vector<loom::evaluation::CaseArtifactResolution::Entry> entries;
  entries.reserve(closures.size());
  for (auto &[reference, closure] : closures)
    entries.push_back({reference, std::move(closure)});
  const loom::evaluation::CaseArtifactResolution resolution =
      take(loom::evaluation::CaseArtifactResolution::get(std::move(entries)));

  struct EvidenceCounts final {
    std::size_t cost = 0;
    std::size_t functional = 0;
  };
  std::map<loom::ArtifactRootReference, EvidenceCounts,
           decltype(&loom::artifactRootReferenceLess)>
      counts(&loom::artifactRootReferenceLess);
  for (const loom::ArtifactRootReference &evidenceReference :
       selection.satisfiedEvidence) {
    const loom::evaluation::EvaluationEvidence evidence =
        take(loom::evaluation::importEvaluationEvidence(evidenceReference,
                                                        resolution, store));
    const loom::evaluation::EvaluationRequest request =
        take(loom::evaluation::importEvaluationRequest(evidence.requestRef(),
                                                       resolution, store));
    llvm::ArrayRef<loom::ArtifactRootReference> candidates =
        request.subjectBindings().subjects(
            loom::evaluation::CaseSubjectRoleRef(0));
    if (candidates.size() != 1)
      fail("ownership Evidence lost its singular candidate binding");
    EvidenceCounts &candidateCounts = counts[candidates.front()];
    if (!request.metricRequests().empty() &&
        request.findingRequests().empty()) {
      ++candidateCounts.cost;
      continue;
    }
    if (request.metricRequests().empty() &&
        request.findingRequests().size() == 1) {
      ++candidateCounts.functional;
      continue;
    }
    fail("ownership Evidence has an unexpected obligation shape");
  }

  if (counts[source].cost != 1 || counts[source].functional != 1 ||
      counts[profitableCandidate].cost != 1 ||
      counts[profitableCandidate].functional != 1)
    fail("ownership DSE acquired expensive functional Evidence before the "
         "resolved benefit gate");
  for (const loom::ArtifactRootReference &candidate : unprofitableCandidates)
    if (counts[candidate].cost != 1 || counts[candidate].functional != 0)
      fail("ownership DSE acquired expensive functional Evidence before the "
           "resolved benefit gate");
  for (const loom::ArtifactRootReference &candidate : inapplicableCandidates)
    if (counts[candidate].cost != 0 || counts[candidate].functional != 0)
      fail("ownership DSE materialized a workload-inapplicable scope");
}

void runEvaluationAnchor() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-fabric-evaluation", directory);
  if (error)
    fail("cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front().reference(), store));
  SourceSimulationInputs inputs =
      makeSourceSimulationInputs(compiled.structuredProgram, store);
  const loom::frontend::SpatialOwnershipScope spatialScope{
      findCallable(compiled.structuredProgram, "kernel")};
  const loom::frontend::SpatialOwnershipScope coldScope{
      findCallable(compiled.structuredProgram, "cold")};
  const loom::frontend::SpatialOwnershipScope tinyScope{
      findCallable(compiled.structuredProgram, "tiny")};
  auto spatialDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, spatialScope.selection));
  auto coldDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, coldScope.selection));
  auto tinyDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, tinyScope.selection));
  if (spatialDecisions.size() != 1 || coldDecisions.size() != 1 ||
      tinyDecisions.size() != 1)
    fail("functional replay anchor has a non-singleton decision domain");
  const loom::frontend::SpatialOwnershipDecisionPoint spatialDecision =
      spatialDecisions.front();
  const loom::frontend::SpatialOwnershipDecisionPoint coldDecision =
      coldDecisions.front();
  const loom::frontend::SpatialOwnershipDecisionPoint tinyDecision =
      tinyDecisions.front();
  auto spatial = take(loom::frontend::materializeSpatialOwnershipDecision(
      compiled.structuredProgram, spatialScope, spatialDecision,
      design.roots().front()));
  auto cold = take(loom::frontend::materializeSpatialOwnershipDecision(
      compiled.structuredProgram, coldScope, coldDecision,
      design.roots().front()));
  auto tiny = take(loom::frontend::materializeSpatialOwnershipDecision(
      compiled.structuredProgram, tinyScope, tinyDecision,
      design.roots().front()));
  auto incorrect = take(loom::frontend::raiseLlvmModuleToStructured(
      parseFunctionallyIncorrectModule(context),
      design.roots().front().reference(), store));

  const loom::ArtifactRootReference baselineRef =
      take(loom::frontend::publishStructuredProgram(compiled.structuredProgram,
                                                    store));
  const loom::ArtifactRootReference spatialRef =
      take(loom::frontend::publishStructuredProgram(spatial.structuredProgram,
                                                    store));
  const loom::ArtifactRootReference coldRef = take(
      loom::frontend::publishStructuredProgram(cold.structuredProgram, store));
  const loom::ArtifactRootReference tinyRef = take(
      loom::frontend::publishStructuredProgram(tiny.structuredProgram, store));
  const loom::ArtifactRootReference incorrectRef =
      take(loom::frontend::publishStructuredProgram(incorrect.structuredProgram,
                                                    store));
  const loom::ArtifactRootReference dataflowRef = take(
      dataflow::publishCanonicalDataflow(spatial.canonicalDataflow, store));
  auto spatialReplay = take(loom::sim::validateSourceBackedDfgReplay(
      compiled.structuredProgram, spatialScope, spatialDecision, spatial,
      inputs.workload, inputs.runtimeInput,
      {100000, 1000000, 256ULL * 1024ULL * 1024ULL}));
  if (spatialReplay.status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      spatialReplay.dynamicActivations != 1 ||
      spatialReplay.wavefrontSteps == 0 || spatialReplay.eventCount == 0)
    fail("functional replay did not execute the selected graph activation");
  auto coldReplay = take(loom::sim::validateSourceBackedDfgReplay(
      compiled.structuredProgram, coldScope, coldDecision, cold,
      inputs.workload, inputs.runtimeInput,
      {100000, 1000000, 256ULL * 1024ULL * 1024ULL}));
  if (coldReplay.status !=
          loom::sim::SourceBackedDfgValidationStatus::Inapplicable ||
      coldReplay.dynamicActivations != 0 || coldReplay.wavefrontSteps != 0 ||
      coldReplay.eventCount != 0)
    fail("functional replay treated an unexecuted graph as passing");
  llvm::Error limitedReplay =
      loom::evaluation::models::primeStructuredProgramFunctionalReplay(
          spatialRef,
          {inputs.workloadReference,
           inputs.runtimeInputReference,
           compiled.structuredProgram,
           spatialScope,
           spatialDecision,
           spatial,
           inputs.workload,
           inputs.runtimeInput,
           inputs.observations,
           {1, 1, 256ULL * 1024ULL * 1024ULL}},
          store);
  if (!limitedReplay)
    fail("functional replay execution limit was ignored");
  if (llvm::errorToErrorCode(std::move(limitedReplay)) !=
      std::make_error_code(std::errc::timed_out))
    fail("functional replay execution limit used the wrong failure kind");
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredProgramFunctionalReplay(
              spatialRef,
              {inputs.workloadReference,
               inputs.runtimeInputReference,
               compiled.structuredProgram,
               spatialScope,
               spatialDecision,
               spatial,
               inputs.workload,
               inputs.runtimeInput,
               inputs.observations,
               {100000, 1000000, 256ULL * 1024ULL * 1024ULL}},
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredProgramFunctionalReplay(
              tinyRef,
              {inputs.workloadReference,
               inputs.runtimeInputReference,
               compiled.structuredProgram,
               tinyScope,
               tinyDecision,
               tiny,
               inputs.workload,
               inputs.runtimeInput,
               inputs.observations,
               {100000, 1000000, 256ULL * 1024ULL * 1024ULL}},
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredProgramFunctionalReplay(
              coldRef,
              {inputs.workloadReference,
               inputs.runtimeInputReference,
               compiled.structuredProgram,
               coldScope,
               coldDecision,
               cold,
               inputs.workload,
               inputs.runtimeInput,
               inputs.observations,
               {100000, 1000000, 256ULL * 1024ULL * 1024ULL}},
              store))
    fail(llvm::toString(std::move(error)));
  const loom::evaluation::models::StructuredFabricAnalyticInvocation invocation{
      inputs.workloadReference, inputs.runtimeInputReference,
      compiled.structuredProgram, inputs.observations};
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              baselineRef, {compiled.structuredProgram, nullptr, std::nullopt},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              coldRef,
              {cold.structuredProgram, &cold.canonicalDataflow,
               findCallable(compiled.structuredProgram, "cold")},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              tinyRef,
              {tiny.structuredProgram, &tiny.canonicalDataflow,
               findCallable(compiled.structuredProgram, "tiny")},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              spatialRef,
              {spatial.structuredProgram, &spatial.canonicalDataflow,
               findCallable(compiled.structuredProgram, "kernel")},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  EvaluatedRuntime baseline = evaluateStructuredRuntime(
      baselineRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store);
  EvaluatedRuntime spatialEvaluation = evaluateStructuredRuntime(
      spatialRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store);
  EvaluatedRuntime coldEvaluation = evaluateStructuredRuntime(
      coldRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store);
  EvaluatedRuntime tinyEvaluation = evaluateStructuredRuntime(
      tinyRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store);
  EvaluatedFunctional baselineFunctional =
      evaluateStructuredFunctional(baselineRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store);
  EvaluatedFunctional spatialFunctional =
      evaluateStructuredFunctional(spatialRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store);
  EvaluatedFunctional coldFunctional = evaluateStructuredFunctional(
      coldRef, inputs.workloadReference, inputs.runtimeInputReference, store);
  EvaluatedFunctional incorrectFunctional =
      evaluateStructuredFunctional(incorrectRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store);
  if (baseline.request.workload() != inputs.workloadReference ||
      baseline.request.runtimeInput() != inputs.runtimeInputReference)
    fail("Structured Evaluation Request lost its exact source inputs");
  if (baseline.request.metricRequests().size() != 5 ||
      spatialEvaluation.request.metricRequests().size() != 5)
    fail("low-confidence model did not expose the complete metric set");
  if (!baseline.request.findingRequests().empty() ||
      !baselineFunctional.request.metricRequests().empty())
    fail("functional and cost semantics share one model authority");
  if (loom::evaluation::compareDecimalValue(spatialEvaluation.value,
                                            baseline.value) >= 0)
    fail("Fabric-aware Evaluation did not prefer Spatial ownership");
  if (coldEvaluation.value != baseline.value)
    fail("an unexecuted candidate changed whole-workload Runtime");
  if (loom::evaluation::compareDecimalValue(tinyEvaluation.value,
                                            baseline.value) < 0)
    fail("launch overhead did not reject a trivial executed candidate");
  if (functionalMismatchResult(baselineFunctional.request,
                               baselineFunctional.evidence) !=
          loom::evaluation::FindingResultForm::Absent ||
      functionalMismatchResult(spatialFunctional.request,
                               spatialFunctional.evidence) !=
          loom::evaluation::FindingResultForm::Absent ||
      functionalMismatchResult(coldFunctional.request,
                               coldFunctional.evidence) !=
          loom::evaluation::FindingResultForm::NotApplicable ||
      functionalMismatchResult(incorrectFunctional.request,
                               incorrectFunctional.evidence) !=
          loom::evaluation::FindingResultForm::Present)
    fail("functional semantic Evidence did not distinguish the wrong "
         "candidate");

  auto semanticCandidates = take(loom::dse::CandidateSet::get(
      loom::frontend::structuredProgramArtifactSchema,
      {baselineRef, spatialRef, incorrectRef}));
  auto semanticPromotion = take(loom::dse::promoteFindingAbsenceAllPassing(
      semanticCandidates, loom::evaluation::CaseSubjectRoleRef(0),
      {{baselineFunctional.request, baselineFunctional.evidence},
       {spatialFunctional.request, spatialFunctional.evidence},
       {incorrectFunctional.request, incorrectFunctional.evidence}},
      baselineFunctional.functionalMismatchRequest, store));
  const auto *semanticSelection =
      std::get_if<loom::dse::CompletedSelection>(&semanticPromotion);
  if (!semanticSelection || semanticSelection->selected.size() != 2 ||
      llvm::is_contained(semanticSelection->selected, incorrectRef) ||
      !llvm::is_contained(semanticSelection->selected, baselineRef) ||
      !llvm::is_contained(semanticSelection->selected, spatialRef))
    fail("AllPassing did not enforce functional finding absence");

  auto coldCandidateSet = take(loom::dse::CandidateSet::get(
      loom::frontend::structuredProgramArtifactSchema, {baselineRef, coldRef}));
  auto coldPromotion = take(loom::dse::promoteMetricTopKAgainstBaseline(
      coldCandidateSet, loom::evaluation::CaseSubjectRoleRef(0), baselineRef,
      {{baseline.request, baseline.evidence},
       {coldEvaluation.request, coldEvaluation.evidence}},
      {loom::evaluation::MetricRequestOrdinal(0),
       loom::dse::ObjectiveDirection::Minimize, 1},
      store));
  const auto *coldSelection =
      std::get_if<loom::dse::CompletedSelection>(&coldPromotion);
  if (!coldSelection ||
      coldSelection->selected !=
          std::vector<loom::ArtifactRootReference>{baselineRef})
    fail("an unexecuted candidate satisfied the accelerator benefit gate");

  for (loom::evaluation::MetricKind metric :
       {loom::evaluation::MetricKind::LimitingClockFrequency,
        loom::evaluation::MetricKind::TotalArea,
        loom::evaluation::MetricKind::LeakagePower}) {
    const auto baselineValue =
        metricResult(baseline.request, baseline.evidence, metric);
    const auto spatialValue = metricResult(spatialEvaluation.request,
                                           spatialEvaluation.evidence, metric);
    if (baselineValue != spatialValue || baselineValue.coefficient() <= 0)
      fail("static Fabric metric did not remain a populated target fact");
  }
  const auto baselineDynamic =
      metricResult(baseline.request, baseline.evidence,
                   loom::evaluation::MetricKind::DynamicPower);
  const auto spatialDynamic =
      metricResult(spatialEvaluation.request, spatialEvaluation.evidence,
                   loom::evaluation::MetricKind::DynamicPower);
  if (baselineDynamic.coefficient() != 0 || spatialDynamic.coefficient() <= 0)
    fail("dynamic power did not follow Spatial workload activity");

  auto candidates = take(loom::dse::CandidateSet::get(
      loom::frontend::structuredProgramArtifactSchema,
      {baselineRef, spatialRef}));
  auto incomplete = take(loom::dse::promoteMetricTopK(
      candidates, loom::evaluation::CaseSubjectRoleRef(0),
      {{baseline.request, baseline.evidence}},
      {loom::evaluation::MetricRequestOrdinal(0),
       loom::dse::ObjectiveDirection::Minimize, 1},
      store));
  const auto *missing =
      std::get_if<loom::dse::IncompleteSelection>(&incomplete);
  if (!missing ||
      missing->reason != loom::dse::IncompleteSelectionReason::MissingEvidence)
    fail("central DSE treated missing Evidence as a ranking value");

  std::vector<loom::dse::PromotionEvidence> evidence;
  evidence.push_back({std::move(spatialEvaluation.request),
                      std::move(spatialEvaluation.evidence)});
  evidence.push_back(
      {std::move(baseline.request), std::move(baseline.evidence)});
  auto promoted = take(loom::dse::promoteMetricTopK(
      candidates, loom::evaluation::CaseSubjectRoleRef(0), evidence,
      {loom::evaluation::MetricRequestOrdinal(0),
       loom::dse::ObjectiveDirection::Minimize, 1},
      store));
  const auto *selection = std::get_if<loom::dse::CompletedSelection>(&promoted);
  if (!selection || selection->selected.size() != 1 ||
      selection->selected.front() != spatialRef)
    fail("central DSE TopK did not promote the best exact candidate");

  loom::dse::PreMappingExplorationOptions exploration{
      {{},
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::dse::ObjectiveDirection::Minimize, 1}}};
  auto exploredSource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  auto explored = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(exploredSource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(), exploration,
      store));
  const auto *exploredSelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&explored);
  if (!exploredSelection || exploredSelection->selected.size() != 1)
    fail("central ownership exploration did not select one survivor");
  if (llvm::any_of(exploredSelection->dispositions,
                   [&](const loom::dse::StructuredOwnershipCandidateDisposition
                           &disposition) {
                     return disposition.coordinate.scope == coldScope;
                   }))
    fail("ownership DSE attempted a workload-inapplicable scope");
  verifyStagedOwnershipEvidence(
      *exploredSelection, baselineRef, spatialRef, {tinyRef}, {coldRef},
      design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store);
  auto exploredView = take(
      exploredSelection->selected.front().compilation.canonicalDataflow.view());
  if (exploredView.actors().empty() ||
      exploredSelection->selected.front().derivations.size() != 1)
    fail("central ownership exploration lost Spatial work or lineage");

  auto parallelExploration = exploration;
  parallelExploration.ownership.candidateWorkerCount = 2;
  auto parallelSource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  auto parallel = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(parallelSource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(),
      parallelExploration, store));
  const auto *parallelSelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&parallel);
  if (!parallelSelection || parallelSelection->selected.size() != 1)
    fail("parallel ownership exploration did not select one survivor");
  if (parallelSelection->selected.front()
              .compilation.structuredProgram.identity() !=
          exploredSelection->selected.front()
              .compilation.structuredProgram.identity() ||
      parallelSelection->selected.front()
              .compilation.canonicalDataflow.identity() !=
          exploredSelection->selected.front()
              .compilation.canonicalDataflow.identity() ||
      parallelSelection->selected.front().derivations !=
          exploredSelection->selected.front().derivations ||
      parallelSelection->satisfiedEvidence !=
          exploredSelection->satisfiedEvidence ||
      parallelSelection->dispositions != exploredSelection->dispositions)
    fail("candidate worker count changed the formal DSE result");

  if (evaluateCanonicalDataflowRuntime(
          dataflowRef, design.roots().front().reference(), store)
          .coefficient() <= 0)
    fail("Dataflow/Fabric Evaluation returned no spatial work");

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove artifact store directory: " + error.message());
}

} // namespace

int main() {
  if (llvm::Error error =
          loom::evaluation::models::registerStructuredFabricAnalyticModel())
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = loom::evaluation::models::
          registerCanonicalDataflowFabricAnalyticModel())
    fail(llvm::toString(std::move(error)));
  runEvaluationAnchor();
  return EXIT_SUCCESS;
}
