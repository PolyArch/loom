#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/SpecialMathAccuracy.h"
#include "Config/ResolvedConfig.h"
#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/PreMappingExploration.h"
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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
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

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredOwnershipEvaluation: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

bool hasCompletedGenerator(
    const loom::dse::CompletedPreMappingSelection &selection,
    llvm::StringRef spelling) {
  for (const loom::dse::DsePlanGenerateInvocationRecords &planInvocation :
       selection.planGenerateInvocations)
    for (const loom::dse::GenerateInvocationRecord &record :
         planInvocation.completed()) {
      const loom::dse::CandidateGeneratorDescriptor *descriptor =
          record.generatorBinding.descriptorRef().descriptor();
      if (!descriptor)
        fail("pre-Mapping Generate provenance lost its exact descriptor");
      if (descriptor->spelling == spelling)
        return true;
    }
  return false;
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

define void @warm(ptr %a, ptr %b, ptr %c) {
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
                        loom::sim::StructuredRuntimeMemoryInput{},
                        loom::sim::StructuredRuntimeMemoryInput{}};
  draft.observableContract.returnValue = true;
  draft.observableContract.memories.push_back(
      {loom::sim::EntryPointerArgumentTarget{2},
       loom::sim::MemoryObservationForm::FullState});
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));

  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtime{
      workload.identity()};
  runtime.memoryObjects = {f32Memory(3.0F), f32Memory(2.0F), zeroedMemory(4),
                           zeroedMemory(4)};
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
      system.reference(), {}, makeScheduledLoopProgram(), {}};
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

  bool sawStructuredMemoryCommunication = false;
  bool sawSpecialMathAccuracy = false;
  bool sawCanonicalGeneratorOrder = false;
  const std::vector<std::string> expectedGeneratorOrder = {
      "compiler.structured_ownership", "compiler.structured_execution_shape",
      "compiler.structured_schedule",
      "compiler.structured_memory_communication",
      "compiler.structured_special_math_accuracy"};
  for (const loom::dse::DsePlanGenerateInvocationRecords &planInvocation :
       selection->planGenerateInvocations) {
    if (!planInvocation.incomplete().empty())
      fail("completed pre-Mapping selection retained an incomplete Generate");
    std::map<std::uint64_t, std::string> generatorByPlanNode;
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
      !sawCanonicalGeneratorOrder)
    fail("production pre-Mapping boundary discarded Generate provenance");
  if (hasCompletedGenerator(*selection, "compiler.dataflow_rewrite"))
    fail("semantic conformance rewrote an already admitted D0");

  bool sawScheduleChild = false;
  loom::frontend::FabricCapabilityIndex capabilities(system.view());
  for (const loom::dse::SelectedPreMappingCompilation &selected :
       selection->selected) {
    if (selected.scheduleDerivations.empty())
      continue;
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
    for (const loom::dse::DataflowRewriteDerivation &derivation :
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

  if (counts[source].cost != 1 || counts[source].functional != 0 ||
      counts[selectedCandidate].cost != 1 ||
      counts[selectedCandidate].functional != 1)
    fail("ownership DSE acquired expensive functional Evidence before the "
         "resolved benefit gate");
  for (const loom::ArtifactRootReference &candidate : costOnlyCandidates)
    if (counts[candidate].cost != 1 || counts[candidate].functional != 0)
      fail("ownership DSE acquired expensive functional Evidence before the "
           "resolved benefit gate");
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
  centralConfig.dse.planNodes = {loom::dse::PromotePlanNodeDefinition{
      loom::dse::structuredEvaluationPromotionAcquisitionDescriptor()
          .reference(),
      {loom::dse::ExactPlanArtifacts{{baselineRef, spatialRef}},
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
    if (!functionalPlanCompleted ||
        functionalPlanCompleted->resolve({1, 0}) !=
            llvm::ArrayRef<loom::ArtifactRootReference>(
                {baselineRef, spatialRef}) ||
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
  EvaluatedFunctional baselineFunctional =
      evaluateStructuredFunctional(baselineRef, inputs.workloadReference,
                                   inputs.runtimeInputReference, store, blobs);
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
  if (!hasCompletedGenerator(*exploredSelection, "compiler.dataflow_rewrite"))
    fail("benefit-qualified exploration skipped Dataflow rewrites");
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
  auto exploredView = take(
      exploredSelection->selected.front().compilation.canonicalDataflow.view());
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
      !benefitOnlySelection->selected.front().derivations.empty())
    fail("benefit-qualified ownership did not retain the host baseline");

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
  if (hasCompletedGenerator(*semanticOnlySelection,
                            "compiler.dataflow_rewrite"))
    fail("semantic conformance rewrote an already admitted D0");
  if (!semanticOnlySelection->selected.front()
           .dataflowRewriteDerivations.empty())
    fail("semantic conformance retained optional Dataflow rewrite lineage");
  auto semanticOnlyView = take(semanticOnlySelection->selected.front()
                                   .compilation.canonicalDataflow.view());
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
      parallelSelection->dispositions != exploredSelection->dispositions)
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
  llvm::Error validation =
      contract->validateCanonical(encoded, {parentReference}, store);
  if (!validation)
    fail("ownership lineage accepted an out-of-range parent-local scope");
  llvm::consumeError(std::move(validation));
  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
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
