#include "StructuredOwnershipEvaluationTestSupport.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/PreMappingExploration.h"
#include "DSE/PreMappingFrontier.h"
#include "DSE/Promotion.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/StructuredEvaluationAcquisition.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocation.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/CanonicalDataflowFunctional.h"
#include "Evaluation/Models/StructuredEvaluationInvocationCache.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Evaluation/StandardFindings.h"
#include "Frontend/Analysis/StructuredProtocolDependencies.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

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
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <cstdint>
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

using loom::test::structured_ownership::centralPlanEvaluatesScheduleChildren;
using loom::test::structured_ownership::
    exactUniformCallArgumentsAreCandidateLocal;
using loom::test::structured_ownership::fail;
using loom::test::structured_ownership::findCallable;
using loom::test::structured_ownership::hasGeneratorInvocation;
using loom::test::structured_ownership::
    ownershipLineageRejectsAnOutOfRangeScope;
using loom::test::structured_ownership::SourceSimulationInputs;
using loom::test::structured_ownership::take;
using loom::test::structured_ownership::zeroedMemory;

/// The evaluation kernels combine two vectors element by element as
/// straight-line code, each element through a dependent chain of multiply-adds
/// long enough that offloading a kernel pays for a physically priced launch
/// and its outstanding-bound memory requests, while `tiny` does not, and no
/// loop so every scope keeps exactly one ownership decision.
constexpr std::size_t kVectorLength = 64;
constexpr std::size_t kMultiplyAddChainLength = 12;

std::string vectorAddKernel(llvm::StringRef name) {
  std::string body;
  llvm::raw_string_ostream stream(body);
  stream << "define void @" << name << "(ptr %a, ptr %b, ptr %c) {\nentry:\n";
  for (std::size_t element = 0; element < kVectorLength; ++element) {
    stream << "  %pa" << element << " = getelementptr float, ptr %a, i64 "
           << element << "\n"
           << "  %pb" << element << " = getelementptr float, ptr %b, i64 "
           << element << "\n"
           << "  %pc" << element << " = getelementptr float, ptr %c, i64 "
           << element << "\n"
           << "  %lhs" << element << " = load float, ptr %pa" << element
           << ", align 4\n"
           << "  %rhs" << element << " = load float, ptr %pb" << element
           << ", align 4\n"
           << "  %sum" << element << "_0 = fadd float %lhs" << element
           << ", %rhs" << element << "\n";
    for (std::size_t step = 0; step < kMultiplyAddChainLength; ++step)
      stream << "  %prod" << element << "_" << step << " = fmul float %sum"
             << element << "_" << step << ", %rhs" << element << "\n"
             << "  %sum" << element << "_" << step + 1 << " = fadd float %prod"
             << element << "_" << step << ", %lhs" << element << "\n";
    stream << "  store float %sum" << element << "_"
           << kMultiplyAddChainLength << ", ptr %pc" << element
           << ", align 4\n";
  }
  stream << "  ret void\n}\n\n";
  return stream.str();
}

std::unique_ptr<llvm::Module> parseModule(llvm::LLVMContext &context) {
  const std::string source = vectorAddKernel("kernel") +
                             vectorAddKernel("cold") + vectorAddKernel("warm") +
                             R"llvm(
define i32 @tiny() {
entry:
  ret i32 7
}

define i32 @main(ptr %a, ptr %b, ptr %c, ptr %d) {
entry:
  call void @kernel(ptr %a, ptr %b, ptr %c)
  call void @warm(ptr %a, ptr %b, ptr %d)
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
parseProtocolDependencyModule(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
define internal void @produce(ptr %out) {
entry:
  store i32 7, ptr %out, align 4
  ret void
}

define internal i32 @consume(ptr %in) {
entry:
  %value = load i32, ptr %in, align 4
  ret i32 %value
}

define i32 @main() {
entry:
  %buffer = alloca [4 x i32], align 16
  %element = getelementptr inbounds [4 x i32], ptr %buffer, i64 0, i64 0
  call void @produce(ptr %element)
  %value = call i32 @consume(ptr %element)
  ret i32 %value
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<protocol-dependency>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("structuredOwnershipEvaluation", stream);
    fail(stream.str());
  }
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

loom::sim::RuntimeMemoryObject f32VectorMemory(float value) {
  llvm::APInt bits = llvm::APFloat(value).bitcastToAPInt();
  std::vector<loom::sim::SemanticMemoryByte> bytes;
  bytes.reserve(4 * kVectorLength);
  for (std::size_t element = 0; element < kVectorLength; ++element)
    for (unsigned byte = 0; byte < 4; ++byte)
      bytes.push_back({loom::sim::SemanticState::Defined,
                       static_cast<std::uint8_t>(
                           bits.extractBitsAsZExtValue(8, byte * 8))});
  return loom::sim::RuntimeMemoryObject{std::move(bytes)};
}

SourceSimulationInputs makeSourceSimulationInputs(
    const loom::frontend::StructuredProgramCandidate &source,
    const loom::ArtifactStore &store) {
  auto view = take(source.view());
  loom::sim::StructuredProgramSimulationWorkload draft{
      findCallable(source, "main")};
  draft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{},
                        loom::sim::StructuredRuntimeMemoryInput{},
                        loom::sim::StructuredRuntimeMemoryInput{},
                        loom::sim::StructuredRuntimeMemoryInput{}};
  draft.observableContract.returnValue = true;
  draft.observableContract.memories.push_back(
      {loom::sim::EntryPointerArgumentTarget{2},
       loom::sim::MemoryObservationForm::FullState});
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));

  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtime{
      workload.identity()};
  runtime.memoryObjects = {f32VectorMemory(3.0F), f32VectorMemory(2.0F),
                           zeroedMemory(4 * kVectorLength),
                           zeroedMemory(4 * kVectorLength)};
  runtime.pointerBindings = {{0, 0, 0}, {1, 1, 0}, {2, 2, 0}, {3, 3, 0}};
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
                          const loom::ArtifactStore &store,
                          const loom::BlobStore &blobs) {
  auto prepared =
      take(loom::evaluation::models::prepareStructuredFabricEvaluation(
          structuredProgram, fabric, workload, runtimeInput,
          loom::defaultResolvedConfig(), store, blobs));
  auto evidence = take(loom::evaluation::evaluateRequest(
      prepared.request, prepared.resolution, store, blobs));
  return EvaluatedRuntime{metricResult(prepared.request, evidence,
                                       loom::evaluation::MetricKind::Runtime),
                          std::move(prepared.request), std::move(evidence)};
}

EvaluatedFunctional evaluateStructuredFunctional(
    const loom::ArtifactRootReference &structuredProgram,
    const loom::ArtifactRootReference &workload,
    const loom::ArtifactRootReference &runtimeInput,
    const loom::ArtifactStore &store, const loom::BlobStore &blobs) {
  auto prepared = take(
      loom::evaluation::models::prepareStructuredProgramFunctionalEvaluation(
          structuredProgram, workload, runtimeInput,
          loom::defaultResolvedConfig(), store, blobs));
  auto evidence = take(loom::evaluation::evaluateRequest(
      prepared.request, prepared.resolution, store, blobs));
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
                                 const loom::ArtifactStore &store,
                                 const loom::BlobStore &blobs) {
  auto prepared =
      take(loom::evaluation::models::prepareCanonicalDataflowFabricEvaluation(
          program, fabric, loom::defaultResolvedConfig(), store, blobs));
  auto evidence = take(loom::evaluation::evaluateRequest(
      prepared.request, prepared.resolution, store, blobs));
  return metricResult(prepared.request, evidence,
                      loom::evaluation::MetricKind::Runtime);
}

void verifyStagedOwnershipEvidence(
    const loom::dse::CompletedPreMappingSelection &selection,
    const loom::ArtifactRootReference &source,
    const loom::ArtifactRootReference &selectedCandidate,
    llvm::ArrayRef<loom::ArtifactRootReference> costOnlyCandidates,
    llvm::ArrayRef<loom::ArtifactRootReference> inapplicableCandidates,
    const loom::ArtifactRootReference &fabric,
    const loom::ArtifactRootReference &workload,
    const loom::ArtifactRootReference &runtimeInput,
    const loom::ArtifactStore &store, const loom::BlobStore &blobs) {
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
  for (const loom::dse::DsePlanGenerateInvocationRecords &planInvocation :
       selection.planGenerateInvocations)
    for (const loom::dse::GenerateInvocationRecord &record :
         planInvocation.completed())
      for (const loom::dse::CandidateGeneratorOutputBinding &binding :
           record.outputBindings)
        for (const loom::ArtifactRootReference &candidate : binding.artifacts)
          closures[candidate];
  for (const loom::dse::SelectedPreMappingCompilation &selected :
       selection.selected) {
    const auto structured = take(loom::frontend::publishStructuredProgram(
        selected.compilation.structuredProgram, store));
    const auto dataflow = take(dataflow::publishCanonicalDataflow(
        selected.compilation.canonicalDataflow, store));
    closures[structured];
    closures[dataflow];
    for (const dataflow::DataflowRewriteDerivation &derivation :
         selected.dataflowRewriteDerivations) {
      closures[derivation.parent];
      closures[derivation.child];
    }
  }

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
        take(loom::evaluation::importEvaluationEvidence(
            evidenceReference, resolution, store, blobs));
    const loom::evaluation::EvaluationRequest request =
        take(loom::evaluation::importEvaluationRequest(
            evidence.requestRef(), resolution, store, blobs));
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

  if (counts[source].cost > 1 || counts[source].functional != 0 ||
      counts[selectedCandidate].cost != 1 ||
      counts[selectedCandidate].functional != 1)
    fail("bounded ownership planner lost selected candidate Evidence");
  for (const loom::ArtifactRootReference &candidate : costOnlyCandidates)
    if (counts[candidate].cost != 1 || counts[candidate].functional > 1)
      fail("bounded ownership planner duplicated candidate Evidence");
  for (const loom::ArtifactRootReference &candidate : inapplicableCandidates)
    if (counts[candidate].cost != 0 || counts[candidate].functional != 0)
      fail("ownership DSE materialized a workload-inapplicable scope");
}

void runEvaluationAnchor() {
  loom::evaluation::models::StructuredEvaluationInvocationCache evaluationCache;
  loom::evaluation::models::StructuredEvaluationInvocationCacheScope
      evaluationCacheScope(evaluationCache);
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-fabric-evaluation", directory);
  if (error)
    fail("cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto dependencyProgram = take(loom::frontend::raiseLlvmModuleToStructured(
      parseProtocolDependencyModule(context),
      design.roots().front().reference(), store));
  const auto producer =
      findCallable(dependencyProgram.structuredProgram, "produce");
  const auto consumer =
      findCallable(dependencyProgram.structuredProgram, "consume");
  const auto dependencies =
      take(loom::frontend::analysis::projectStructuredProtocolDependencies(
          dependencyProgram.structuredProgram, {producer, consumer}));
  if (dependencies.size() != 1 || dependencies.front().producer != producer ||
      dependencies.front().consumer != consumer ||
      dependencies.front().sharedMemoryObjectCount != 1 ||
      dependencies.front().knownSharedMemoryBytes != 16 ||
      dependencies.front().unknownSharedMemoryObjectCount != 0)
    fail("protocol dependency projection lost its exact fixed payload");

  auto compiled = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front().reference(), store));
  SourceSimulationInputs inputs =
      makeSourceSimulationInputs(compiled.structuredProgram, store);
  const loom::frontend::SpatialOwnershipScope spatialScope{
      findCallable(compiled.structuredProgram, "kernel")};
  const loom::frontend::SpatialOwnershipScope coldScope{
      findCallable(compiled.structuredProgram, "cold")};
  const loom::frontend::SpatialOwnershipScope warmScope{
      findCallable(compiled.structuredProgram, "warm")};
  const loom::frontend::SpatialOwnershipScope tinyScope{
      findCallable(compiled.structuredProgram, "tiny")};
  auto spatialDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, spatialScope.selection));
  auto coldDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, coldScope.selection));
  auto warmDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, warmScope.selection));
  auto tinyDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, tinyScope.selection));
  if (spatialDecisions.size() != 1 || coldDecisions.size() != 1 ||
      warmDecisions.size() != 1 || tinyDecisions.size() != 1)
    fail("functional replay anchor has a non-singleton decision domain");
  const loom::frontend::SpatialOwnershipDecisionPoint spatialDecision =
      spatialDecisions.front();
  const loom::frontend::SpatialOwnershipDecisionPoint coldDecision =
      coldDecisions.front();
  const loom::frontend::SpatialOwnershipDecisionPoint warmDecision =
      warmDecisions.front();
  const loom::frontend::SpatialOwnershipDecisionPoint tinyDecision =
      tinyDecisions.front();
  auto spatial = take(loom::frontend::materializeSpatialOwnershipDecision(
      compiled.structuredProgram, spatialScope, spatialDecision,
      design.roots().front()));
  auto cold = take(loom::frontend::materializeSpatialOwnershipDecision(
      compiled.structuredProgram, coldScope, coldDecision,
      design.roots().front()));
  auto warm = take(loom::frontend::materializeSpatialOwnershipDecision(
      compiled.structuredProgram, warmScope, warmDecision,
      design.roots().front()));
  const loom::frontend::SpatialOwnershipScope combinedWarmScope{
      findCallable(spatial.structuredProgram, "warm")};
  auto combinedWarmDecisions =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          spatial.structuredProgram, combinedWarmScope.selection));
  if (combinedWarmDecisions.size() != 1)
    fail("independent ownership composition changed the decision domain");
  auto combined = take(loom::frontend::materializeSpatialOwnershipDecision(
      spatial.structuredProgram, combinedWarmScope,
      combinedWarmDecisions.front(), design.roots().front()));
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
  const loom::ArtifactRootReference warmRef = take(
      loom::frontend::publishStructuredProgram(warm.structuredProgram, store));
  const loom::ArtifactRootReference combinedRef =
      take(loom::frontend::publishStructuredProgram(combined.structuredProgram,
                                                    store));
  const loom::ArtifactRootReference tinyRef = take(
      loom::frontend::publishStructuredProgram(tiny.structuredProgram, store));
  const loom::ArtifactRootReference incorrectRef =
      take(loom::frontend::publishStructuredProgram(incorrect.structuredProgram,
                                                    store));
  const loom::ArtifactRootReference dataflowRef = take(
      dataflow::publishCanonicalDataflow(spatial.canonicalDataflow, store));

  std::array<std::uint8_t, loom::ComponentViewDigest::byteSize>
      weakerProjectionBytes{};
  std::array<std::uint8_t, loom::ComponentViewDigest::byteSize>
      strongerProjectionBytes{};
  strongerProjectionBytes.front() = 1;
  loom::dse::PreMappingCandidateProjection weakerProjection(
      take(loom::ComponentViewDigest::fromBytes(weakerProjectionBytes)));
  weakerProjection.ownedRegionCount = 1;
  loom::dse::PreMappingCandidateProjection strongerProjection(
      take(loom::ComponentViewDigest::fromBytes(strongerProjectionBytes)));
  strongerProjection.ownedRegionCount = 1;
  strongerProjection.estimateSupport =
      loom::dse::PreMappingEstimateSupport::Supported;
  strongerProjection.estimateConfidence =
      loom::dse::PreMappingEstimateConfidence::Low;
  strongerProjection.estimatedCutTrafficBytes = 0;
  const std::array<loom::dse::PreMappingFrontierCandidate, 2>
      convergedCoordinates = {
          loom::dse::PreMappingFrontierCandidate{
              spatialRef, weakerProjection, std::nullopt,
              loom::dse::PreMappingScheduleIntent::Unconstrained, {},
              std::nullopt},
          loom::dse::PreMappingFrontierCandidate{
              spatialRef, strongerProjection, 1,
              loom::dse::PreMappingScheduleIntent::Unconstrained, {},
              std::nullopt}};
  auto convergedSelection = take(loom::dse::selectPreMappingFrontier(
      convergedCoordinates, 2, 1));
  if (convergedSelection.preferenceOrder !=
          std::vector<loom::ArtifactRootReference>{spatialRef} ||
      convergedSelection.preferenceProjectionIdentities !=
          std::vector<loom::ComponentViewDigest>{strongerProjection.identity})
    fail("central frontier did not canonicalize converged coordinates");

  std::array<std::uint8_t, loom::ComponentViewDigest::byteSize>
      broaderProjectionBytes{};
  broaderProjectionBytes.front() = 3;
  loom::dse::PreMappingCandidateProjection broaderProjection(
      take(loom::ComponentViewDigest::fromBytes(broaderProjectionBytes)));
  broaderProjection.ownedRegionCount = 2;
  broaderProjection.estimateSupport =
      loom::dse::PreMappingEstimateSupport::Supported;
  broaderProjection.estimateConfidence =
      loom::dse::PreMappingEstimateConfidence::Low;
  broaderProjection.estimatedCutTrafficBytes = 0;
  const std::array<loom::dse::PreMappingFrontierCandidate, 2>
      convergedOwnershipCoordinates = {
          loom::dse::PreMappingFrontierCandidate{
              spatialRef, broaderProjection, std::nullopt,
              loom::dse::PreMappingScheduleIntent::Unconstrained, {},
              std::nullopt},
          loom::dse::PreMappingFrontierCandidate{
              spatialRef, strongerProjection, std::nullopt,
              loom::dse::PreMappingScheduleIntent::Unconstrained, {},
              std::nullopt}};
  auto ownershipSelection = take(loom::dse::selectPreMappingFrontier(
      convergedOwnershipCoordinates, 1, 1));
  if (ownershipSelection.preferenceProjectionIdentities !=
      std::vector<loom::ComponentViewDigest>{strongerProjection.identity})
    fail("central frontier representative ranking omitted ownership size");

  std::array<std::uint8_t, loom::ComponentViewDigest::byteSize>
      temporalProjectionBytes{};
  temporalProjectionBytes.front() = 2;
  loom::dse::PreMappingCandidateProjection temporalProjection(
      take(loom::ComponentViewDigest::fromBytes(temporalProjectionBytes)));
  temporalProjection.ownedRegionCount = 1;
  const std::array<loom::dse::PreMappingFrontierCandidate, 3>
      spectrumCoordinates = {
          loom::dse::PreMappingFrontierCandidate{
              baselineRef, weakerProjection, std::nullopt,
              loom::dse::PreMappingScheduleIntent::Unconstrained, {},
              std::nullopt},
          loom::dse::PreMappingFrontierCandidate{
              spatialRef, strongerProjection, 1,
              loom::dse::PreMappingScheduleIntent::Unconstrained, {},
              std::nullopt},
          loom::dse::PreMappingFrontierCandidate{
              coldRef, temporalProjection, std::nullopt,
              loom::dse::PreMappingScheduleIntent::TemporalReuse, {},
              loom::dse::PreMappingSpectrumClass::MaxTemporal}};
  auto spectrumSelection = take(loom::dse::selectPreMappingFrontier(
      spectrumCoordinates, 2, 2));
  if (!llvm::is_contained(spectrumSelection.preferenceOrder, coldRef))
    fail("frontier diversity dropped the verified temporal spectrum point");
  auto unverifiedEndpoint = loom::dse::selectPreMappingFrontier(
      spectrumCoordinates, 2, 2,
      loom::dse::PreMappingSpectrumEndpoint::MaxSpatial);
  if (unverifiedEndpoint)
    fail("endpoint selection accepted a schedule hint without a verified "
         "SystemMapping classification");
  const std::string unverifiedDiagnostic =
      llvm::toString(unverifiedEndpoint.takeError());
  if (!llvm::StringRef(unverifiedDiagnostic).contains(
          "pre_mapping_spectrum_endpoint_unsupported"))
    fail("endpoint selection lost its typed unsupported disposition");

  auto generatorConfig =
      take(loom::dse::projectResolvedStructuredOwnershipGeneratorConfigView(
          loom::defaultResolvedConfig(),
          {warmScope.selection, spatialScope.selection}));
  auto reversedGeneratorConfig =
      take(loom::dse::projectResolvedStructuredOwnershipGeneratorConfigView(
          loom::defaultResolvedConfig(),
          {spatialScope.selection, warmScope.selection}));
  if (generatorConfig.canonicalViewBytes() !=
          reversedGeneratorConfig.canonicalViewBytes() ||
      generatorConfig.digest() != reversedGeneratorConfig.digest())
    fail("ownership generator config retained protocol-root input order");
  auto adoptedGeneratorConfig =
      take(loom::dse::adoptResolvedStructuredOwnershipGeneratorConfigView(
          loom::dse::resolvedStructuredOwnershipGeneratorConfigSchemaBytes(),
          generatorConfig.canonicalViewBytes(), generatorConfig.digest()));
  if (adoptedGeneratorConfig.scopeExpansionLimit() !=
          loom::defaultResolvedConfig()
              .dse.structuredOwnership.scopeExpansionLimit ||
      adoptedGeneratorConfig.protocolCallableRoots().size() != 2)
    fail("ownership generator config did not round-trip typed fields");

  auto generatorInputs =
      take(loom::dse::bindStructuredOwnershipCandidateGeneratorInputs(
          baselineRef, design.roots().front().reference(),
          inputs.workloadReference, inputs.runtimeInputReference));
  auto generatorBinding =
      take(loom::dse::resolveStructuredOwnershipCandidateGeneratorBinding(
          generatorConfig));
  auto generated = take(loom::dse::invokeCandidateGenerator(
      generatorInputs, generatorBinding, store, blobs));
  const auto *completedGeneration =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &generated.outcome);
  std::vector<loom::ArtifactRootReference> expectedGenerated = {
      baselineRef, spatialRef, warmRef};
  llvm::sort(expectedGenerated, loom::artifactRootReferenceLess);
  std::vector<loom::ArtifactRootReference> expectedAccelerators = {spatialRef,
                                                                   warmRef};
  llvm::sort(expectedAccelerators, loom::artifactRootReferenceLess);
  if (!completedGeneration || completedGeneration->outputBindings.size() != 2 ||
      completedGeneration->outputBindings[0].artifacts != expectedGenerated ||
      completedGeneration->outputBindings[1].artifacts != expectedAccelerators)
    fail("central ownership generator changed the exact candidate set");
  if (completedGeneration->lineageEdges.size() < 4)
    fail("central ownership generator lost typed decision lineage");
  for (const loom::dse::CandidateGeneratorLineageEdge &edge :
       completedGeneration->lineageEdges) {
    if (edge.kind !=
            loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
        edge.parents != std::vector<loom::ArtifactRootReference>{baselineRef})
      fail("ownership lineage changed its exact parent relation");
    auto decision =
        take(loom::frontend::adoptSpatialOwnershipDecision(edge.ownerPayload));
    if (decision.scope.selection.parent != baselineRef.artifact)
      fail("ownership lineage decision belongs to a foreign parent");
  }

  auto foreignGeneratorConfig =
      take(loom::dse::projectResolvedStructuredOwnershipGeneratorConfigView(
          loom::defaultResolvedConfig(),
          {findCallable(incorrect.structuredProgram, "kernel")}));
  auto foreignBinding =
      take(loom::dse::resolveStructuredOwnershipCandidateGeneratorBinding(
          foreignGeneratorConfig));
  auto foreignGeneration = loom::dse::invokeCandidateGenerator(
      generatorInputs, foreignBinding, store, blobs);
  if (foreignGeneration)
    fail("ownership generator accepted a foreign protocol root");
  llvm::consumeError(foreignGeneration.takeError());

  auto analyticInvocation =
      take(loom::evaluation::models::prepareStructuredFabricAnalyticInvocation(
          {baselineRef, spatialRef}, design.roots().front().reference(),
          inputs.workloadReference, inputs.runtimeInputReference, store));
  auto strictSpatialEvaluation =
      take(loom::evaluation::models::prepareStructuredFabricEvaluation(
          spatialRef, design.roots().front().reference(),
          inputs.workloadReference, inputs.runtimeInputReference,
          loom::defaultResolvedConfig(), store, blobs));
  auto reusedSpatialEvaluation =
      take(loom::evaluation::models::prepareStructuredFabricEvaluation(
          spatialRef, analyticInvocation, loom::defaultResolvedConfig(), store,
          blobs));
  if (loom::evaluation::evaluationRequestReference(
          strictSpatialEvaluation.request) !=
      loom::evaluation::evaluationRequestReference(
          reusedSpatialEvaluation.request))
    fail("invocation-local analytic resolution changed Request identity");

  auto analyticObligation =
      take(loom::dse::prepareStructuredFabricAnalyticEvidenceObligationTemplate(
          baselineRef, design.roots().front().reference(),
          inputs.workloadReference, inputs.runtimeInputReference,
          loom::defaultResolvedConfig(), store, blobs));
  auto acquisitionConfig =
      take(loom::dse::projectResolvedEvidenceObligationSetConfigView(
          {loom::dse::EvidenceObligationTemplateRef(0)}));
  loom::ResolvedConfig centralConfig = loom::defaultResolvedConfig();
  centralConfig.dse.modelAuthorizations = {
      {analyticObligation.modelBinding().descriptorRef()}};
  centralConfig.dse.evidenceObligationTemplates = {analyticObligation};
  centralConfig.dse.qualityGatePolicies = {
      take(loom::dse::QualityGatePolicy::get({}))};
  std::vector<loom::ArtifactRootReference> centralCandidates{baselineRef,
                                                              spatialRef};
  llvm::sort(centralCandidates, loom::artifactRootReferenceLess);
  centralConfig.dse.planNodes = {loom::dse::PromotePlanNodeDefinition{
      loom::dse::structuredEvaluationPromotionAcquisitionDescriptor()
          .reference(),
      {loom::dse::ExactPlanArtifacts{std::move(centralCandidates)},
       loom::dse::ExactPlanArtifacts{{design.roots().front().reference()}},
       loom::dse::ExactPlanArtifacts{{inputs.workloadReference}},
       loom::dse::ExactPlanArtifacts{{inputs.runtimeInputReference}}},
      acquisitionConfig.canonicalViewBytes().vec(),
      acquisitionConfig.digest(),
      loom::dse::QualityGatePolicyRef(0),
      loom::dse::AllPassingSelection{},
      loom::dse::PromotePurpose::CandidateSelection}};
  auto centralView =
      take(loom::dse::projectResolvedDseConfigView(centralConfig));
  auto centralOutcome =
      take(loom::dse::executeDsePlan(centralView, store, blobs));
  const auto *centralCompleted =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&centralOutcome);
  if (!centralCompleted || centralCompleted->resolve({0, 0}).size() != 2 ||
      centralCompleted->resolve({0, 1}).size() != 2)
    fail("Structured acquisition did not produce total central Evidence");
  for (const loom::ArtifactRootReference &evidenceRef :
       centralCompleted->resolve({0, 1})) {
    auto evidence = take(loom::evaluation::importEvaluationEvidence(
        evidenceRef, analyticInvocation.caseResolution(), store, blobs));
    auto request = take(loom::evaluation::importEvaluationRequest(
        evidence.requestRef(), analyticInvocation.caseResolution(), store,
        blobs));
    const auto candidateSubjects = request.subjectBindings().subjects(
        loom::evaluation::CaseSubjectRoleRef(0));
    const auto fabricSubjects = request.subjectBindings().subjects(
        loom::evaluation::CaseSubjectRoleRef(1));
    if (request.metricRequests().size() != 5 || candidateSubjects.size() != 1 ||
        fabricSubjects.size() != 1 ||
        fabricSubjects.front() != design.roots().front().reference())
      fail("Structured acquisition changed the exact analytic Request");
  }

  auto functionalObligation = take(
      loom::dse::prepareStructuredProgramFunctionalEvidenceObligationTemplate(
          baselineRef, inputs.workloadReference, inputs.runtimeInputReference,
          loom::defaultResolvedConfig(), store, blobs));
  auto functionalAcquisitionConfig =
      take(loom::dse::projectResolvedEvidenceObligationSetConfigView(
          {loom::dse::EvidenceObligationTemplateRef(0)}));
  auto functionalGeneratorConfig =
      take(loom::dse::projectResolvedStructuredOwnershipGeneratorConfigView(
          loom::defaultResolvedConfig(), {spatialScope.selection}));
  auto functionalGate = take(loom::dse::QualityGatePolicy::get(
      {{{loom::dse::FindingGate{0, loom::evaluation::FindingRequestOrdinal(0),
                                loom::dse::RequiredFindingState::Absent}}}}));
  loom::ResolvedConfig functionalPlanConfig = loom::defaultResolvedConfig();
  functionalPlanConfig.dse.modelAuthorizations = {
      {functionalObligation.modelBinding().descriptorRef()}};
  functionalPlanConfig.dse.evidenceObligationTemplates = {functionalObligation};
  functionalPlanConfig.dse.qualityGatePolicies = {functionalGate};
  functionalPlanConfig.dse.planNodes = {
      loom::dse::GeneratePlanNodeDefinition{
          loom::dse::structuredOwnershipCandidateGeneratorDescriptor()
              .reference(),
          {loom::dse::ExactPlanArtifacts{{baselineRef}},
           loom::dse::ExactPlanArtifacts{{design.roots().front().reference()}},
           loom::dse::ExactPlanArtifacts{{inputs.workloadReference}},
           loom::dse::ExactPlanArtifacts{{inputs.runtimeInputReference}}},
          functionalGeneratorConfig.canonicalViewBytes().vec(),
          functionalGeneratorConfig.digest()},
      loom::dse::PromotePlanNodeDefinition{
          loom::dse::structuredEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {loom::dse::PlanOutputRef{0, 0},
           loom::dse::ExactPlanArtifacts{{design.roots().front().reference()}},
           loom::dse::ExactPlanArtifacts{{inputs.workloadReference}},
           loom::dse::ExactPlanArtifacts{{inputs.runtimeInputReference}}},
          functionalAcquisitionConfig.canonicalViewBytes().vec(),
          functionalAcquisitionConfig.digest(),
          loom::dse::QualityGatePolicyRef(0),
          loom::dse::AllPassingSelection{},
          loom::dse::PromotePurpose::CandidateSelection}};
  auto functionalPlanView =
      take(loom::dse::projectResolvedDseConfigView(functionalPlanConfig));
  {
    loom::dse::StructuredOwnershipInvocation functionalInvocation(
        compiled.structuredProgram, compiled.structuredProgram, inputs.workload,
        inputs.runtimeInput, design.roots().front(),
        loom::defaultResolvedConfig(), {}, 1,
        {100000, 1000000, 256ULL * 1024ULL * 1024ULL},
        compiled.sourceProvenance);
    loom::dse::StructuredOwnershipInvocationScope functionalInvocationScope(
        functionalInvocation);
    auto functionalPlanOutcome =
        take(loom::dse::executeDsePlan(functionalPlanView, store, blobs));
    const auto *functionalPlanCompleted =
        std::get_if<loom::dse::CompletedDsePlanExecution>(
            &functionalPlanOutcome);
    std::vector<loom::ArtifactRootReference> expectedFunctionalCandidates{
        baselineRef, spatialRef};
    llvm::sort(expectedFunctionalCandidates,
               loom::artifactRootReferenceLess);
    if (!functionalPlanCompleted ||
        functionalPlanCompleted->resolve({1, 0}) !=
            llvm::ArrayRef<loom::ArtifactRootReference>(
                expectedFunctionalCandidates) ||
        functionalPlanCompleted->resolve({1, 1}).size() != 2)
      fail(
          "central functional Promote did not replay the generated candidates");
    if (functionalInvocation.sourceNativeExecutionCount() != 1)
      fail("central Generate/Promote repeated source native execution");
    const auto functionalCache =
        functionalInvocation.evaluationCacheStatistics();
    if (functionalCache.sourceObservationPrimeCount != 1 ||
        functionalCache.sourceObservationMissCount != 0)
      fail("central Generate/Promote source observation cache counts are " +
           std::to_string(functionalCache.sourceObservationPrimeCount) + "/" +
           std::to_string(functionalCache.sourceObservationHitCount) + "/" +
           std::to_string(functionalCache.sourceObservationMissCount));

    const auto preparedD0 =
        take(functionalInvocation.prepareDataflowGeneration(spatialRef, store));
    if (preparedD0 != dataflowRef)
      fail("Dataflow generation changed the selected Structured D0 identity");

    auto dataflowAnalytic = take(
        loom::dse::
            prepareCanonicalDataflowFabricAnalyticEvidenceObligationTemplate(
                dataflowRef, design.roots().front().reference(),
                loom::defaultResolvedConfig(), store, blobs));
    auto dataflowFunctional = take(
        loom::dse::prepareCanonicalDataflowFunctionalEvidenceObligationTemplate(
            dataflowRef, spatialRef, inputs.workloadReference,
            inputs.runtimeInputReference, loom::defaultResolvedConfig(), store,
            blobs));
    std::vector<loom::dse::EvidenceObligationTemplate> dataflowObligations = {
        dataflowAnalytic, dataflowFunctional};
    auto dataflowAcquisitionConfig =
        take(loom::dse::projectResolvedEvidenceObligationSetConfigView(
            {loom::dse::EvidenceObligationTemplateRef(0),
             loom::dse::EvidenceObligationTemplateRef(1)}));
    auto dataflowAcquisitionBinding =
        take(loom::dse::resolveDataflowEvaluationPromotionAcquisitionBinding(
            dataflowAcquisitionConfig));
    auto dataflowAcquisitionInputs =
        take(loom::dse::bindDataflowEvaluationPromotionInputs(
            {dataflowRef}, spatialRef, design.roots().front().reference(),
            inputs.workloadReference, inputs.runtimeInputReference));
    const std::array<loom::ArtifactRootReference, 1> dataflowCandidates = {
        dataflowRef};
    const std::array<loom::dse::EvidenceObligationTemplateRef, 2>
        dataflowObligationRefs = {loom::dse::EvidenceObligationTemplateRef(0),
                                  loom::dse::EvidenceObligationTemplateRef(1)};
    auto dataflowAcquisition = take(loom::dse::invokePromotionAcquisition(
        dataflowAcquisitionInputs, dataflowAcquisitionBinding,
        dataflowObligations, {dataflowCandidates, dataflowObligationRefs},
        store, blobs));
    const auto *completedDataflowAcquisition =
        std::get_if<loom::dse::CompletedPromotionAcquisition>(
            &dataflowAcquisition);
    if (!completedDataflowAcquisition ||
        completedDataflowAcquisition->evidence.size() != 2)
      fail("Dataflow acquisition did not produce total central Evidence");
    bool observedDataflowAnalytic = false;
    bool observedDataflowFunctional = false;
    for (const loom::dse::PromotionEvidence &record :
         completedDataflowAcquisition->evidence) {
      if (record.request.modelBinding().descriptorRef() ==
          loom::evaluation::models::
              canonicalDataflowFabricAnalyticModelDescriptorRef()) {
        const auto *completed =
            std::get_if<loom::evaluation::CompletedEvidence>(
                &record.evidence.outcome());
        observedDataflowAnalytic =
            completed && completed->metricResults.size() == 5;
      } else if (record.request.modelBinding().descriptorRef() ==
                 loom::evaluation::models::
                     canonicalDataflowFunctionalModelDescriptorRef()) {
        observedDataflowFunctional =
            functionalMismatchResult(record.request, record.evidence) ==
            loom::evaluation::FindingResultForm::Absent;
      }
    }
    if (!observedDataflowAnalytic || !observedDataflowFunctional)
      fail("Dataflow acquisition changed analytical or functional Evidence");

    auto centrallySelected = take(
        functionalInvocation.materializeSelectedCandidate(spatialRef, store));
    if (centrallySelected.candidate.structuredProgram.identity() !=
            spatialRef.artifact ||
        centrallySelected.derivations.size() != 1 ||
        !centrallySelected.functionalReplay ||
        centrallySelected.functionalReplay->status !=
            loom::sim::SourceBackedDfgValidationStatus::Equivalent)
      fail("central functional Promote lost replay or ownership lineage");
  }

  auto coldSpatialEvidence = take(loom::evaluation::evaluateRequest(
      strictSpatialEvaluation.request, strictSpatialEvaluation.resolution,
      store, blobs));

  auto spatialReplay = take(loom::sim::validateSourceBackedDfgReplay(
      compiled.structuredProgram, spatial, inputs.workload, inputs.runtimeInput,
      {100000, 1000000, 256ULL * 1024ULL * 1024ULL}, &inputs.observations));
  if (spatialReplay.status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      spatialReplay.dynamicActivations != 1 ||
      spatialReplay.wavefrontSteps == 0 || spatialReplay.eventCount == 0)
    fail("functional replay did not execute the selected graph activation");
  auto combinedReplay = take(loom::sim::validateSourceBackedDfgReplay(
      compiled.structuredProgram, combined, inputs.workload,
      inputs.runtimeInput, {100000, 1000000, 256ULL * 1024ULL * 1024ULL},
      &inputs.observations,
      [&](const loom::sim::CanonicalSimulationWorkload &,
          const loom::sim::CanonicalSimulationRuntimeInput &)
          -> llvm::Expected<loom::sim::SourceBackedDfgReplayCaseReference> {
        return loom::sim::SourceBackedDfgReplayCaseReference{
            inputs.workloadReference, inputs.runtimeInputReference};
      }));
  if (combinedReplay.status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      combinedReplay.dynamicActivations != 2 ||
      combinedReplay.replayCaseOccurrences != 2 ||
      combinedReplay.replayCases.size() != 1 ||
      combinedReplay.wavefrontSteps == 0 || combinedReplay.eventCount == 0)
    fail("functional replay did not cover every reachable Spatial region");
  auto coldReplay = take(loom::sim::validateSourceBackedDfgReplay(
      compiled.structuredProgram, cold, inputs.workload, inputs.runtimeInput,
      {100000, 1000000, 256ULL * 1024ULL * 1024ULL}, &inputs.observations));
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
           spatial,
           inputs.workload,
           inputs.runtimeInput,
           inputs.observations,
           {1, 1, 256ULL * 1024ULL * 1024ULL}},
          store);
  if (limitedReplay)
    fail(llvm::toString(std::move(limitedReplay)));
  auto limitedFunctional = take(
      loom::evaluation::models::prepareStructuredProgramFunctionalEvaluation(
          spatialRef, inputs.workloadReference, inputs.runtimeInputReference,
          loom::defaultResolvedConfig(), store, blobs));
  auto limitedEvidence = take(loom::evaluation::evaluateRequest(
      limitedFunctional.request, limitedFunctional.resolution, store, blobs));
  if (limitedEvidence.outcomeKind() !=
      loom::evaluation::EvidenceOutcomeKind::CancelledOrTimeout)
    fail("functional replay execution limit did not become typed Evidence");
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredProgramFunctionalReplay(
              spatialRef,
              {inputs.workloadReference,
               inputs.runtimeInputReference,
               compiled.structuredProgram,
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
               cold,
               inputs.workload,
               inputs.runtimeInput,
               inputs.observations,
               {100000, 1000000, 256ULL * 1024ULL * 1024ULL}},
              store))
    fail(llvm::toString(std::move(error)));
  const loom::evaluation::models::StructuredFabricAnalyticInvocation invocation{
      inputs.workloadReference,
      inputs.runtimeInputReference,
      inputs.workload,
      inputs.runtimeInput,
      compiled.structuredProgram,
      inputs.observations};
  auto combinedObservations =
      take(loom::sim::executeProfiledSelectedStructuredProgram(
          combined.structuredProgram, compiled.structuredProgram,
          inputs.workload, inputs.runtimeInput));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              baselineRef,
              {compiled.structuredProgram,
               nullptr,
               {},
               {},
               &inputs.observations},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              coldRef,
              {cold.structuredProgram, &cold.canonicalDataflow,
               cold.spatialGraphs, cold.blockActivityLineage},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              tinyRef,
              {tiny.structuredProgram, &tiny.canonicalDataflow,
               tiny.spatialGraphs, tiny.blockActivityLineage},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              spatialRef,
              {spatial.structuredProgram, &spatial.canonicalDataflow,
               spatial.spatialGraphs, spatial.blockActivityLineage},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::evaluation::models::primeStructuredFabricAnalyticResult(
              combinedRef,
              {combined.structuredProgram,
               &combined.canonicalDataflow,
               combined.spatialGraphs,
               {},
               &combinedObservations},
              invocation, design.roots().front(), loom::defaultResolvedConfig(),
              store))
    fail(llvm::toString(std::move(error)));
  EvaluatedRuntime baseline = evaluateStructuredRuntime(
      baselineRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store, blobs);
  EvaluatedRuntime spatialEvaluation = evaluateStructuredRuntime(
      spatialRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store, blobs);
  EvaluatedRuntime coldEvaluation = evaluateStructuredRuntime(
      coldRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store, blobs);
  EvaluatedRuntime tinyEvaluation = evaluateStructuredRuntime(
      tinyRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store, blobs);
  EvaluatedRuntime combinedEvaluation = evaluateStructuredRuntime(
      combinedRef, design.roots().front().reference(), inputs.workloadReference,
      inputs.runtimeInputReference, store, blobs);
  if (metricResult(strictSpatialEvaluation.request, coldSpatialEvidence,
                   loom::evaluation::MetricKind::Runtime) !=
      spatialEvaluation.value)
    fail("source-activity projection changed the exact analytical result");
  const auto sourceFunctionalBefore = evaluationCache.statistics();
  EvaluatedFunctional baselineFunctional =
      evaluateStructuredFunctional(baselineRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store, blobs);
  const auto sourceFunctionalAfter = evaluationCache.statistics();
  if (sourceFunctionalAfter.sourceObservationHitCount +
          sourceFunctionalAfter.sourceObservationMissCount !=
      sourceFunctionalBefore.sourceObservationHitCount +
          sourceFunctionalBefore.sourceObservationMissCount + 1)
    fail("exact-source functional Evidence did not consume native source "
         "observations");
  EvaluatedFunctional spatialFunctional =
      evaluateStructuredFunctional(spatialRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store, blobs);
  EvaluatedFunctional coldFunctional =
      evaluateStructuredFunctional(coldRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store, blobs);
  EvaluatedFunctional incorrectFunctional =
      evaluateStructuredFunctional(incorrectRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store, blobs);
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
  if (loom::evaluation::compareDecimalValue(combinedEvaluation.value,
                                            spatialEvaluation.value) >= 0 ||
      loom::evaluation::compareDecimalValue(
          combinedEvaluation.value,
          evaluateStructuredRuntime(warmRef, design.roots().front().reference(),
                                    inputs.workloadReference,
                                    inputs.runtimeInputReference, store, blobs)
              .value) >= 0)
    fail("whole-candidate Evaluation did not compose independent Spatial work");
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
  loom::dse::QualityGateClause semanticClause;
  semanticClause.atoms.push_back(
      loom::dse::FindingGate{0, baselineFunctional.functionalMismatchRequest,
                             loom::dse::RequiredFindingState::Absent});
  const loom::dse::QualityGatePolicy semanticGate =
      take(loom::dse::QualityGatePolicy::get({std::move(semanticClause)}));
  auto semanticPromotion = take(loom::dse::promoteCandidates(
      semanticCandidates, loom::evaluation::CaseSubjectRoleRef(0),
      {{baselineFunctional.request, baselineFunctional.evidence},
       {spatialFunctional.request, spatialFunctional.evidence},
       {incorrectFunctional.request, incorrectFunctional.evidence}},
      semanticGate, loom::dse::AllPassingSelection{}, nullptr, store));
  const auto *semanticSelection =
      std::get_if<loom::dse::CompletedSelection>(&semanticPromotion);
  if (!semanticSelection || semanticSelection->selected.size() != 2 ||
      llvm::is_contained(semanticSelection->selected, incorrectRef) ||
      !llvm::is_contained(semanticSelection->selected, baselineRef) ||
      !llvm::is_contained(semanticSelection->selected, spatialRef))
    fail("AllPassing did not enforce functional finding absence");

  auto inapplicableCandidates = take(loom::dse::CandidateSet::get(
      loom::frontend::structuredProgramArtifactSchema, {coldRef}));
  auto inapplicablePromotion = take(loom::dse::promoteCandidates(
      inapplicableCandidates, loom::evaluation::CaseSubjectRoleRef(0),
      {{coldFunctional.request, coldFunctional.evidence}}, semanticGate,
      loom::dse::AllPassingSelection{}, nullptr, store));
  const auto *indeterminate =
      std::get_if<loom::dse::IncompleteSelection>(&inapplicablePromotion);
  if (!indeterminate ||
      indeterminate->reason !=
          loom::dse::IncompleteSelectionReason::NonComparableEvidence)
    fail("NotApplicable quality Evidence did not make Promotion incomplete");

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

  loom::dse::PreMappingExplorationOptions exploration{
      {{},
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::ResolvedObjectiveDirection::Minimize, 1}}};
  exploration.ownership.selectionMode =
      loom::dse::StructuredOwnershipSelectionMode::BenefitQualified;
  auto exploredSource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  exploration.ownership.protocolCallableRoots = {
      findCallable(exploredSource.structuredProgram, "kernel"),
      findCallable(exploredSource.structuredProgram, "cold"),
      findCallable(exploredSource.structuredProgram, "warm"),
      findCallable(exploredSource.structuredProgram, "tiny")};
  auto explored = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(exploredSource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(), exploration, store,
      blobs));
  const auto *exploredSelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&explored);
  if (!exploredSelection || exploredSelection->selected.size() != 1)
    fail("central ownership exploration did not select one survivor");
  if (exploredSelection->requestedPlannerMode !=
          loom::dse::StructuredOwnershipSelectionMode::BenefitQualified ||
      exploredSelection->resolvedPlannerMode !=
          loom::dse::StructuredOwnershipSelectionMode::SemanticConformance)
    fail("benefit-qualified API spelling bypassed the bounded joint planner");
  const loom::ArtifactRootReference selectedRef =
      take(loom::frontend::publishStructuredProgram(
          exploredSelection->selected.front().compilation.structuredProgram,
          store));
  if (selectedRef != spatialRef && selectedRef != warmRef)
    fail("central ownership exploration selected no profitable kernel");
  const loom::ArtifactRootReference costOnlyProfitable =
      selectedRef == spatialRef ? warmRef : spatialRef;
  if (llvm::any_of(exploredSelection->dispositions,
                   [&](const loom::dse::StructuredOwnershipCandidateDisposition
                           &disposition) {
                     return disposition.coordinate.scope == coldScope;
                   }))
    fail("ownership DSE attempted a workload-inapplicable scope");
  verifyStagedOwnershipEvidence(*exploredSelection, baselineRef, selectedRef,
                                {costOnlyProfitable, tinyRef}, {coldRef},
                                design.roots().front().reference(),
                                inputs.workloadReference,
                                inputs.runtimeInputReference, store, blobs);
  const auto &exploredView =
      exploredSelection->selected.front().compilation.canonicalDataflow.view();
  if (exploredView.actors().empty() ||
      exploredSelection->selected.front().derivations.size() != 1)
    fail("central ownership exploration lost Spatial work or lineage");

  auto benefitOnlySource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  auto benefitOnlyExploration = exploration;
  benefitOnlyExploration.ownership.protocolCallableRoots = {
      findCallable(benefitOnlySource.structuredProgram, "tiny")};
  auto benefitOnly = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(benefitOnlySource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(),
      benefitOnlyExploration, store, blobs));
  const auto *benefitOnlySelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&benefitOnly);
  if (!benefitOnlySelection || benefitOnlySelection->selected.size() != 1 ||
      benefitOnlySelection->selected.front().derivations.size() != 1 ||
      !benefitOnlySelection->selected.front().functionalReplay ||
      benefitOnlySelection->selected.front().functionalReplay->status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent)
    fail("benefit-qualified API spelling bypassed bounded semantic selection");

  auto semanticOnlySource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  auto semanticOnlyExploration = benefitOnlyExploration;
  semanticOnlyExploration.ownership.selectionMode =
      loom::dse::StructuredOwnershipSelectionMode::SemanticConformance;
  semanticOnlyExploration.ownership.protocolCallableRoots = {
      findCallable(semanticOnlySource.structuredProgram, "tiny")};
  auto semanticOnly = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(semanticOnlySource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(),
      semanticOnlyExploration, store, blobs));
  const auto *semanticOnlySelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&semanticOnly);
  if (!semanticOnlySelection || semanticOnlySelection->selected.size() != 1 ||
      semanticOnlySelection->selected.front().derivations.size() != 1 ||
      !semanticOnlySelection->selected.front().functionalReplay ||
      semanticOnlySelection->selected.front().functionalReplay->status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent)
    fail("semantic conformance did not select the executed equivalent graph");
  if (benefitOnlySelection->selected.front()
              .compilation.structuredProgram.identity() !=
          semanticOnlySelection->selected.front()
              .compilation.structuredProgram.identity() ||
      benefitOnlySelection->selected.front()
              .compilation.canonicalDataflow.identity() !=
          semanticOnlySelection->selected.front()
              .compilation.canonicalDataflow.identity() ||
      benefitOnlySelection->candidateInventory !=
          semanticOnlySelection->candidateInventory)
    fail("API selection spelling changed the resolved bounded frontier");
  if (hasGeneratorInvocation(*semanticOnlySelection,
                             "compiler.dataflow_rewrite"))
    fail("semantic conformance rewrote an already admitted D0");
  if (!semanticOnlySelection->selected.front()
           .dataflowRewriteDerivations.empty())
    fail("semantic conformance retained optional Dataflow rewrite lineage");
  const auto &semanticOnlyView = semanticOnlySelection->selected.front()
                                     .compilation.canonicalDataflow.view();
  if (semanticOnlyView.graphs().empty() || semanticOnlyView.actors().empty())
    fail("semantic conformance selected a graph-free candidate");

  auto semanticChainSource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  auto semanticChainExploration = semanticOnlyExploration;
  semanticChainExploration.ownership.selection.k = 2;
  semanticChainExploration.ownership.protocolCallableRoots = {
      findCallable(semanticChainSource.structuredProgram, "kernel"),
      findCallable(semanticChainSource.structuredProgram, "warm")};
  auto semanticChain = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(semanticChainSource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(),
      semanticChainExploration, store, blobs));
  const auto *semanticChainSelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&semanticChain);
  if (!semanticChainSelection || semanticChainSelection->selected.size() != 2)
    fail("semantic conformance did not retain its bounded ownership chain");
  if (semanticChainSelection->frontierAccounting.functionalReplays.consumed !=
          semanticChainSelection->evaluationTiming.functionalReplayCalls ||
      semanticChainSelection->frontierAccounting.analyticEvaluations.consumed <=
          semanticChainSelection->frontierAccounting.functionalReplays
              .consumed)
    fail("semantic conformance did not narrow analytic expansion before "
         "functional replay");
  bool sawUnclassifiedScheduleFact = false;
  for (const auto &record : semanticChainSelection->candidateInventory)
    if (record.temporalWitness && !record.verifiedSpectrum)
      sawUnclassifiedScheduleFact = true;
    else if (record.verifiedSpectrum)
      fail("pre-Mapping logical-domain evidence claimed a spectrum endpoint");
  if (!sawUnclassifiedScheduleFact ||
      !semanticChainSelection->shadowRecall ||
      semanticChainSelection->shadowRecall->eligibleSubsets == 0)
    fail("bounded ownership frontier omitted logical-domain or shadow evidence");
  std::size_t retainedPlanningCandidates = 0;
  std::size_t budgetedPlanningCandidates = 0;
  std::vector<std::uint64_t> planningRanks;
  for (const auto &record : semanticChainSelection->candidateInventory) {
    if (record.disposition ==
        loom::dse::PreMappingCandidatePlanningDisposition::Retained) {
      ++retainedPlanningCandidates;
      if (!record.preferenceRank)
        fail("retained planning candidate has no preference rank");
      planningRanks.push_back(*record.preferenceRank);
    } else {
      ++budgetedPlanningCandidates;
      if (record.preferenceRank)
        fail("budgeted planning candidate retained a preference rank");
    }
  }
  llvm::sort(planningRanks);
  if (retainedPlanningCandidates != 2 || budgetedPlanningCandidates == 0 ||
      planningRanks != std::vector<std::uint64_t>({0, 1}) ||
      semanticChainSelection->protocolRootActivity.size() != 2)
    fail("bounded ownership planning inventory is incomplete");
  if (semanticChainSelection->selected[0].derivations.size() != 2 ||
      semanticChainSelection->selected[1].derivations.size() != 1)
    fail("semantic conformance did not rank ownership coverage first");
  if (semanticChainSelection->selected[0]
          .compilation.structuredProgram.identity() != combinedRef.artifact)
    fail("semantic conformance lost the complete ownership closure");
  const loom::ArtifactIdentity prefixIdentity =
      semanticChainSelection->selected[1]
          .compilation.structuredProgram.identity();
  if (prefixIdentity != spatialRef.artifact &&
      prefixIdentity != warmRef.artifact)
    fail("semantic conformance retained a non-prefix ownership alternative");
  for (const auto &candidate : semanticChainSelection->selected)
    if (!candidate.functionalReplay ||
        candidate.functionalReplay->status !=
            loom::sim::SourceBackedDfgValidationStatus::Equivalent)
      fail("semantic conformance retained an unverified ownership prefix");

  auto rewriteChainSource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  auto rewriteChainExploration = semanticChainExploration;
  rewriteChainExploration.frontier.stoppingPolicy =
      loom::dse::JointDesignStoppingPolicy::BoundedQuality;
  rewriteChainExploration.ownership.protocolCallableRoots = {
      findCallable(rewriteChainSource.structuredProgram, "kernel"),
      findCallable(rewriteChainSource.structuredProgram, "warm")};
  auto rewriteChain = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(rewriteChainSource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(),
      rewriteChainExploration, store, blobs));
  const auto *rewriteChainSelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&rewriteChain);
  if (!rewriteChainSelection || rewriteChainSelection->selected.size() != 2)
    fail("Dataflow rewrites did not fill the bounded Mapping frontier");
  const auto firstParent = rewriteChainSelection->selected.front()
                               .compilation.structuredProgram.identity();
  if (firstParent == rewriteChainSelection->selected.back()
                         .compilation.structuredProgram.identity())
    fail("one parent's Dataflow rewrites displaced another retained parent");
  unsigned firstParentChildren = 0;
  for (const auto &record : rewriteChainSelection->candidateInventory)
    if (record.structuredProgram && record.canonicalDataflow &&
        record.structuredProgram->artifact == firstParent)
      ++firstParentChildren;
  if (firstParentChildren < 2)
    fail("the Mapping frontier did not exercise competing Dataflow siblings");

  auto parallelExploration = exploration;
  parallelExploration.ownership.candidateWorkerCount = 2;
  auto parallelSource = take(loom::frontend::raiseLlvmModuleToStructured(
      parseModule(context), design.roots().front()));
  auto parallel = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(parallelSource), inputs.workload, inputs.runtimeInput,
      design.roots().front(), loom::defaultResolvedConfig(),
      parallelExploration, store, blobs));
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
      parallelSelection->dispositions != exploredSelection->dispositions ||
      parallelSelection->protocolRootActivity !=
          exploredSelection->protocolRootActivity ||
      parallelSelection->protocolDependencies !=
          exploredSelection->protocolDependencies ||
      parallelSelection->candidateInventory !=
          exploredSelection->candidateInventory ||
      parallelSelection->sharedEvaluationStatistics.profileCacheHits !=
          exploredSelection->sharedEvaluationStatistics.profileCacheHits ||
      parallelSelection->sharedEvaluationStatistics.profileCacheMisses !=
          exploredSelection->sharedEvaluationStatistics.profileCacheMisses)
    fail("candidate worker count changed the formal DSE result");

  {
    loom::evaluation::models::StructuredEvaluationInvocationCache isolatedCache;
    loom::evaluation::models::StructuredEvaluationInvocationCacheScope
        isolatedScope(isolatedCache);
    auto leaked =
        loom::evaluation::models::getPrimedStructuredProgramFunctionalReplay(
            spatialRef, inputs.workloadReference, inputs.runtimeInputReference);
    if (leaked)
      fail("a fresh Evaluation invocation observed a prior replay result");
    llvm::consumeError(leaked.takeError());
    if (isolatedCache.statistics().functionalMissCount != 1)
      fail("fresh Evaluation invocation did not account its exact cache miss");
  }
  auto restoredReplay =
      take(loom::evaluation::models::getPrimedStructuredProgramFunctionalReplay(
          spatialRef, inputs.workloadReference, inputs.runtimeInputReference));
  if (restoredReplay.status !=
      loom::sim::SourceBackedDfgValidationStatus::Equivalent)
    fail("nested Evaluation cache scope did not restore its parent binding");

  const auto cacheStatistics = evaluationCache.statistics();
  if (cacheStatistics.analyticPrimeCount == 0 ||
      cacheStatistics.analyticHitCount == 0 ||
      cacheStatistics.functionalPrimeCount == 0 ||
      cacheStatistics.functionalHitCount == 0)
    fail("invocation-local Evaluation cache did not reuse exact typed results");

  if (evaluateCanonicalDataflowRuntime(
          dataflowRef, design.roots().front().reference(), store, blobs)
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
  exactUniformCallArgumentsAreCandidateLocal();
  centralPlanEvaluatesScheduleChildren();
  ownershipLineageRejectsAnOutOfRangeScope();
  runEvaluationAnchor();
  return EXIT_SUCCESS;
}
