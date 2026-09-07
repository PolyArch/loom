#include "StructuredOwnershipEvaluationTestSupport.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/SpecialMathAccuracy.h"
#include "Config/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "DSE/PreMappingFrontier.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
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

namespace loom::test::structured_ownership {
namespace {

loom::frontend::StructuredProgramCandidate makeScheduledLoopProgram() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func internal @loop_kernel(%out: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %floating = arith.constant 1.0 : f32
    %sine = math.sin %floating : f32
    %bits = arith.bitcast %sine : f32 to i32
    %address = llvm.getelementptr inbounds %out[0]
        : (!llvm.ptr) -> !llvm.ptr, i32
    llvm.store %bits, %address : i32, !llvm.ptr
    scf.for %index = %c0 to %c8 step %c1 {
      %wide = arith.index_cast %index : index to i64
      %doubled = arith.addi %wide, %wide : i64
    }
    llvm.return
  }

  llvm.func @main(%out: !llvm.ptr) -> i32 {
    llvm.call @loop_kernel(%out) : (!llvm.ptr) -> ()
    %c0 = arith.constant 0 : i32
    llvm.return %c0 : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the central schedule fixture");
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail("cannot initialize the native target");
  auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context,
                            take(target.getDefaultDataLayoutForTarget())
                                .getStringRepresentation()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

std::unique_ptr<llvm::Module>
parseUniformCallSpecializationModule(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
define internal i32 @uniform_core(ptr %optional, ptr %out, i32 %count) {
entry:
  %is_null = icmp eq ptr %optional, null
  br i1 %is_null, label %without_value, label %with_value

with_value:
  %loaded = load i32, ptr %optional, align 4
  br label %merge

without_value:
  br label %merge

merge:
  %value = phi i32 [ %loaded, %with_value ], [ 0, %without_value ]
  br label %loop

loop:
  %index = phi i32 [ 0, %merge ], [ %next_index, %loop ]
  %acc = phi i32 [ %value, %merge ], [ %next_acc, %loop ]
  %next_acc = add i32 %acc, 1
  %next_index = add i32 %index, 1
  %continue = icmp slt i32 %next_index, %count
  br i1 %continue, label %loop, label %exit

exit:
  store i32 %next_acc, ptr %out, align 4
  ret i32 %next_acc
}

define internal i32 @uniform_forward(ptr %optional, ptr %out, i32 %count) {
entry:
  %value = call i32 @uniform_core(ptr %optional, ptr %out, i32 %count)
  ret i32 %value
}

define internal i32 @conflicting_core(ptr %optional, ptr %out) {
entry:
  %is_null = icmp eq ptr %optional, null
  br i1 %is_null, label %without_value, label %with_value

with_value:
  %loaded = load i32, ptr %optional, align 4
  br label %merge

without_value:
  br label %merge

merge:
  %value = phi i32 [ %loaded, %with_value ], [ 0, %without_value ]
  store i32 %value, ptr %out, align 4
  ret i32 %value
}

define i32 @entry(ptr %unknown, ptr %out, i32 %count) {
entry:
  %uniform = call i32 @uniform_forward(ptr null, ptr %out, i32 %count)
  %null_case = call i32 @conflicting_core(ptr null, ptr %out)
  %unknown_case = call i32 @conflicting_core(ptr %unknown, ptr %out)
  %partial = add i32 %uniform, %null_case
  %result = add i32 %partial, %unknown_case
  ret i32 %result
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<call-specialization>");
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

template <typename... OpTy>
loom::frontend::StructuredEntityRef findTopLevelOperationInCallable(
    const loom::frontend::StructuredProgramCandidate &candidate,
    llvm::StringRef callableName) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    if (!entity.operation || !llvm::isa<OpTy...>(entity.operation))
      continue;
    auto function = entity.operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (function && function.getSymName() == callableName &&
        entity.operation->getParentOp() == function.getOperation())
      return entity.reference;
  }
  fail("operation is absent from Structured callable: " + callableName.str());
}

SourceSimulationInputs makeScheduledLoopInputs(
    const loom::frontend::StructuredProgramCandidate &source,
    const loom::ArtifactStore &store) {
  auto view = take(source.view());
  loom::sim::StructuredProgramSimulationWorkload draft{
      findCallable(source, "main")};
  draft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{}};
  draft.observableContract.returnValue = true;
  draft.observableContract.memories.push_back(
      {loom::sim::EntryPointerArgumentTarget{0},
       loom::sim::MemoryObservationForm::FullState});
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));

  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtime{
      workload.identity()};
  runtime.memoryObjects = {zeroedMemory(8 * sizeof(std::uint32_t))};
  runtime.pointerBindings = {{0, 0, 0}};
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

} // namespace

void centralPlanEvaluatesScheduleChildren() {
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-central-schedule", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Large));
  auto system = take(loom::fabric::importEntireFabricRoot(
      design.roots().front().reference(), store));

  loom::frontend::StructuredCompilation compilation{
      system.reference(), {}, makeScheduledLoopProgram(), {}, {}};
  auto inputs = makeScheduledLoopInputs(compilation.structuredProgram, store);
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.dse.schedule.scopeExpansionLimit = 8;
  loom::dse::PreMappingExplorationOptions options{
      {{},
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::ResolvedObjectiveDirection::Minimize, 64}}};
  options.ownership.selectionMode =
      loom::dse::StructuredOwnershipSelectionMode::SemanticConformance;
  options.ownership.protocolCallableRoots = {
      findCallable(compilation.structuredProgram, "loop_kernel")};

  auto explored = take(loom::dse::exploreStructuredCompilationToPreMapping(
      std::move(compilation), inputs.workload, inputs.runtimeInput, system,
      config, options, store, blobs));
  const auto *selection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&explored);
  if (!selection || selection->selected.empty()) {
    if (const auto *incomplete =
            std::get_if<loom::dse::IncompletePreMappingExploration>(&explored))
      fail("central schedule exploration is incomplete at node " +
           (incomplete->planNodeOrdinal
                ? std::to_string(*incomplete->planNodeOrdinal)
                : std::string("none")) +
           ": " + loom::dse::toString(incomplete->reason).str());
    fail("central schedule exploration selected no feasible candidate");
  }
  for (const loom::dse::SelectedPreMappingCompilation &selected :
       selection->selected) {
    if (!selected.planningRecordOrdinal ||
        *selected.planningRecordOrdinal >= selection->candidateInventory.size())
      fail("central schedule candidate lost its planning record identity");
    const auto &record =
        selection->candidateInventory[*selected.planningRecordOrdinal];
    if (!record.candidateIdentity)
      fail("central schedule candidate did not publish a stable identity");
    if (!record.structuredProgram)
      fail("central schedule candidate lost its planning program");
    auto planningProgram = take(loom::frontend::importStructuredProgram(
        *record.structuredProgram, store));
    auto planningView = take(planningProgram.view());
    for (const loom::frontend::StructuredEntityRef &root :
         record.ownedProtocolRoots)
      (void)take(planningView.resolve(root));
    auto recomputed = take(loom::dse::computePreMappingCandidateIdentity(
        record, selection->sourceProgram, selection->fabric,
        selection->workload, selection->runtimeInput,
        selection->frontierPolicyDigest));
    if (*record.candidateIdentity != recomputed)
      fail("central schedule candidate identity is not reproducible");
    auto differentInvocation =
        take(loom::dse::computePreMappingCandidateIdentity(
            record, selection->fabric, selection->workload,
            selection->runtimeInput, selection->sourceProgram,
            record.projection ? record.projection->identity
                              : selection->frontierPolicyDigest));
    if (differentInvocation != recomputed)
      fail("invocation provenance changed semantic candidate identity");
  }

  bool sawStructuredMemoryCommunication = false;
  bool sawSpecialMathAccuracy = false;
  bool sawCanonicalGeneratorOrder = false;
  const std::vector<std::string> expectedGeneratorOrder = {
      "compiler.structured_ownership", "compiler.structured_execution_shape",
      "compiler.structured_schedule",
      "compiler.structured_memory_communication",
      "compiler.structured_special_math_accuracy"};
  bool sawBoundedGenerate = false;
  for (const loom::dse::DsePlanGenerateInvocationRecords &planInvocation :
       selection->planGenerateInvocations) {
    std::map<std::uint64_t, std::string> generatorByPlanNode;
    for (const auto &record : planInvocation.incomplete()) {
      const auto *descriptor =
          record.generatorBinding.descriptorRef().descriptor();
      if (!descriptor || !record.incompleteReason ||
          *record.incompleteReason !=
              loom::dse::CandidateGeneratorIncompleteReason::
                  SemanticLimitReached)
        fail("bounded pre-Mapping Generate lost its typed exhaustion reason");
      auto [position, inserted] = generatorByPlanNode.try_emplace(
          record.planNodeOrdinal, descriptor->spelling.str());
      if (!inserted && position->second != descriptor->spelling)
        fail("one pre-Mapping plan node used conflicting generators");
      sawBoundedGenerate = true;
    }
    for (const loom::dse::GenerateInvocationRecord &record :
         planInvocation.completed()) {
      const loom::dse::CandidateGeneratorDescriptor *descriptor =
          record.generatorBinding.descriptorRef().descriptor();
      if (!descriptor)
        fail("pre-Mapping Generate provenance lost its exact descriptor");
      auto [position, inserted] = generatorByPlanNode.try_emplace(
          record.planNodeOrdinal, descriptor->spelling.str());
      if (!inserted && position->second != descriptor->spelling)
        fail("one pre-Mapping plan node used conflicting generators");
      sawStructuredMemoryCommunication |=
          descriptor->spelling == "compiler.structured_memory_communication";
      sawSpecialMathAccuracy |=
          descriptor->spelling == "compiler.structured_special_math_accuracy";
    }
    auto first = generatorByPlanNode.find(0);
    if (first == generatorByPlanNode.end() ||
        first->second != expectedGeneratorOrder.front())
      continue;
    for (std::size_t ordinal = 0; ordinal != expectedGeneratorOrder.size();
         ++ordinal) {
      auto found = generatorByPlanNode.find(ordinal);
      if (found == generatorByPlanNode.end() ||
          found->second != expectedGeneratorOrder[ordinal])
        fail("production pre-Mapping generator order is not canonical");
    }
    sawCanonicalGeneratorOrder = true;
  }
  if (!sawStructuredMemoryCommunication || !sawSpecialMathAccuracy ||
      !sawCanonicalGeneratorOrder || !sawBoundedGenerate)
    fail("production pre-Mapping boundary discarded Generate provenance");
  if (hasGeneratorInvocation(*selection, "compiler.dataflow_rewrite"))
    fail("semantic conformance rewrote an already admitted D0");

  bool sawScheduleChild = false;
  loom::frontend::FabricCapabilityIndex capabilities(system.view());
  for (const loom::dse::SelectedPreMappingCompilation &selected :
       selection->selected) {
    if (selected.scheduleDerivations.empty())
      continue;
    for (const loom::dse::StructuredScheduleDerivation &derivation :
         selected.scheduleDerivations) {
      if (derivation.parent == derivation.child)
        fail("selected schedule lineage retained a self edge");
      auto parent = take(
          loom::frontend::importStructuredProgram(derivation.parent, store));
      auto child = take(
          loom::frontend::importStructuredProgram(derivation.child, store));
      auto fabric =
          take(loom::fabric::importEntireFabricRoot(derivation.fabric, store));
      if (llvm::Error error =
              loom::frontend::verifyStructuredScheduleDerivation(
                  parent, fabric, derivation.decision, child))
        fail("selected schedule lineage lost its exact child: " +
             llvm::toString(std::move(error)));
    }
    if (!selected.functionalReplay ||
        selected.functionalReplay->status !=
            loom::sim::SourceBackedDfgValidationStatus::Equivalent)
      fail("selected schedule child lacks equivalent source-backed replay");
    if (!selected.specialMathAccuracyDerivations.empty())
      fail("strict special math created an accuracy decision lineage");
    bool sawCorrectlyRoundedSine = false;
    selected.compilation.structuredProgram.module().walk(
        [&](mlir::math::SinOp operation) {
          auto accuracy = llvm::dyn_cast_or_null<mlir::StringAttr>(
              operation->getDiscardableAttr(
                  loom::kSpecialMathAccuracyAttrName));
          sawCorrectlyRoundedSine |=
              accuracy && accuracy.getValue() == "CorrectlyRounded";
        });
    if (!sawCorrectlyRoundedSine)
      fail("strict special math lost its mechanical accuracy closure");
    auto miss = take(capabilities.firstInadmissibleActor(
        selected.compilation.canonicalDataflow));
    if (miss)
      fail("selected schedule child bypassed exact Fabric admission");
    sawScheduleChild = true;
  }
  if (!sawScheduleChild)
    fail("production central plan did not evaluate a schedule child");
  bool sawUnclassifiedLogicalDomainFact = false;
  for (const auto &record : selection->candidateInventory)
    if (record.temporalWitness && !record.verifiedSpectrum)
      sawUnclassifiedLogicalDomainFact = true;
    else if (record.verifiedSpectrum)
      fail("production central plan claimed an unverified spectrum endpoint");
  if (!sawUnclassifiedLogicalDomainFact)
    fail("production central plan did not retain a logical-domain fact");

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

void exactUniformCallArgumentsAreCandidateLocal() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-uniform-call-specialization", directory);
  if (error)
    fail("cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(loom::frontend::raiseLlvmModuleToStructured(
      parseUniformCallSpecializationModule(context),
      design.roots().front().reference(), store));
  const loom::frontend::StructuredEntityRef uniform =
      findTopLevelOperationInCallable<mlir::scf::ForOp, mlir::scf::WhileOp>(
          compiled.structuredProgram, "uniform_core");
  auto domain = take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
      compiled.structuredProgram, uniform));
  if (domain.size() != 2)
    fail("uniform call arguments did not add one specialization choice");

  using Shape = loom::frontend::DirectCallSpecializationShape;
  const auto specialized = llvm::find_if(
      domain, [](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.directCallSpecializationShape ==
               Shape::UniformExactConstants;
      });
  const auto unspecialized = llvm::find_if(
      domain, [](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return !point.directCallSpecializationShape;
      });
  if (specialized == domain.end() || unspecialized == domain.end())
    fail("uniform call specialization domain is incomplete");

  auto baseline = take(loom::frontend::prepareSpatialOwnershipSelection(
      compiled.structuredProgram, {uniform}, *unspecialized));
  auto selected = take(loom::frontend::prepareSpatialOwnershipSelection(
      compiled.structuredProgram, {uniform}, *specialized));
  auto baselineFunction =
      baseline.operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  auto selectedFunction =
      selected.operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!baselineFunction || !selectedFunction)
    fail("call specialization changed the selected callable kind");
  std::size_t baselineLoads = 0;
  std::size_t selectedLoads = 0;
  baselineFunction.walk([&](mlir::LLVM::LoadOp) { ++baselineLoads; });
  selectedFunction.walk([&](mlir::LLVM::LoadOp) { ++selectedLoads; });
  if (baselineLoads != 1 || selectedLoads != 0 ||
      !selectedFunction.getBody().front().getArgument(0).use_empty())
    fail("uniform null specialization retained the unreachable memory path");

  llvm::DenseSet<mlir::Block *> liveBlocks;
  selected.module->walk([&](mlir::Operation *operation) {
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        liveBlocks.insert(&block);
  });
  llvm::DenseSet<mlir::Block *> trackedBlocks;
  for (const auto &binding : selected.sourceBlocks) {
    if (!liveBlocks.contains(binding.candidateBlock))
      fail("call specialization retained a dead block lineage");
    if (!trackedBlocks.insert(binding.candidateBlock).second)
      fail("call specialization duplicated a live block lineage");
  }
  if (trackedBlocks.size() != liveBlocks.size())
    fail("call specialization did not preserve total live block lineage");

  const loom::frontend::StructuredEntityRef conflicting =
      findCallable(compiled.structuredProgram, "conflicting_core");
  auto conflictingDomain =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          compiled.structuredProgram, conflicting));
  if (conflictingDomain.size() != 1 ||
      conflictingDomain.front().directCallSpecializationShape)
    fail("conflicting call arguments admitted an unsound specialization");

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove artifact store directory: " + error.message());
}

void ownershipLineageRejectsAnOutOfRangeScope() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-ownership-lineage-context", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto parent = makeScheduledLoopProgram();
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  const loom::frontend::SpatialOwnershipScope validScope{
      findCallable(parent, "loop_kernel")};
  auto validDomain =
      take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
          parent, validScope.selection));
  if (validDomain.empty())
    fail("ownership fixture has no typed decision domain");

  loom::frontend::SpatialOwnershipDecisionPoint invalidCallPoint =
      validDomain.front();
  invalidCallPoint.directCallSpecializationShape =
      static_cast<loom::frontend::DirectCallSpecializationShape>(99);
  auto invalidCallEncoding = loom::frontend::encodeSpatialOwnershipDecision(
      {validScope, invalidCallPoint});
  if (invalidCallEncoding)
    fail("ownership encoder accepted an unknown direct-call specialization");
  llvm::consumeError(invalidCallEncoding.takeError());
  auto invalidCallMaterialization =
      loom::frontend::materializeStructuredSpatialOwnershipDecision(
          parent, validScope, invalidCallPoint);
  if (invalidCallMaterialization)
    fail("ownership materializer accepted an unknown direct-call "
         "specialization");
  llvm::consumeError(invalidCallMaterialization.takeError());

  loom::frontend::SpatialOwnershipDecisionPoint invalidWidthPoint =
      validDomain.front();
  invalidWidthPoint.addressProjection =
      loom::frontend::RootRelativeAddressProjection{7};
  auto invalidWidthEncoding = loom::frontend::encodeSpatialOwnershipDecision(
      {validScope, invalidWidthPoint});
  if (invalidWidthEncoding)
    fail("ownership encoder accepted an unsupported root-relative width");
  llvm::consumeError(invalidWidthEncoding.takeError());
  auto invalidWidthMaterialization =
      loom::frontend::materializeStructuredSpatialOwnershipDecision(
          parent, validScope, invalidWidthPoint);
  if (invalidWidthMaterialization)
    fail("ownership materializer accepted an unsupported root-relative width");
  llvm::consumeError(invalidWidthMaterialization.takeError());

  const loom::frontend::SpatialOwnershipDecision invalidDecision{
      {{parent.identity(), loom::frontend::StructuredEntityKind::Operation, 0}},
      {std::nullopt, static_cast<loom::frontend::ForallOwnershipShape>(99),
       std::nullopt}};
  auto invalidEncoding =
      loom::frontend::encodeSpatialOwnershipDecision(invalidDecision);
  if (invalidEncoding)
    fail("ownership encoder accepted an unknown in-memory decision shape");
  llvm::consumeError(invalidEncoding.takeError());
  const loom::frontend::SpatialOwnershipDecision decision{
      {{parent.identity(), loom::frontend::StructuredEntityKind::Operation,
        999999}},
      {}};
  auto encoded = take(loom::frontend::encodeSpatialOwnershipDecision(decision));
  const auto *contract =
      loom::dse::structuredOwnershipCandidateGeneratorDescriptor()
          .ownerLineagePayload;
  if (!contract)
    fail("ownership generator has no owner lineage contract");
  llvm::Error validation = contract->validateCanonical(
      encoded, parentReference, {parentReference}, store);
  if (!validation)
    fail("ownership lineage accepted an out-of-range parent-local scope");
  llvm::consumeError(std::move(validation));
  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

} // namespace loom::test::structured_ownership
