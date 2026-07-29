#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredThreadDomain: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

llvm::StringRef nativeDataLayout() {
  static const std::string layout = [] {
    if (llvm::InitializeNativeTarget() ||
        llvm::InitializeNativeTargetAsmPrinter())
      fail("cannot initialize the native target");
    auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
    return take(target.getDefaultDataLayoutForTarget())
        .getStringRepresentation();
  }();
  return layout;
}

loom::frontend::StructuredProgramCandidate makeSource() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%base: !llvm.ptr) {
    %lb = arith.constant 4 : index
    %ub = arith.constant 20 : index
    %step = arith.constant 2 : index
    scf.forall (%i) = (%lb) to (%ub) step (%step) {
      %i32 = arith.index_cast %i : index to i32
      %ptr = llvm.getelementptr inbounds %base[%i32]
          : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4 x i8>
      %zero = arith.constant 0.0 : f32
      llvm.store %zero, %ptr : f32, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the thread-domain fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredEntityRef
findForall(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation))
    if (llvm::isa_and_nonnull<mlir::scf::ForallOp>(entity.operation))
      return entity.reference;
  fail("Structured Program has no scf.forall ownership scope");
}

loom::frontend::StructuredEntityRef
findKernel(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getName() == "kernel")
      return entity.reference;
  }
  fail("Structured Program has no kernel entry");
}

std::uint64_t constantExtent(dataflow::ThreadLaunchOp launch) {
  if (launch.getGridUpperBounds().size() != 1)
    fail("logical thread launch does not have rank one");
  auto constant = launch.getGridUpperBounds()
                      .front()
                      .getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    fail("static thread extent was not materialized as a constant");
  auto value = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!value)
    fail("thread extent constant is not an integer");
  return value.getValue().getZExtValue();
}

std::int64_t constantIndex(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  auto attribute = constant
                       ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                       : mlir::IntegerAttr{};
  if (!attribute)
    fail("thread launch parameter is not a constant index value");
  return attribute.getValue().getSExtValue();
}

loom::frontend::SpatialOwnershipDecisionPoint
findThreadDecision(const loom::frontend::StructuredProgramCandidate &source,
                   std::optional<unsigned> indexWidth) {
  using Shape = loom::frontend::ForallOwnershipShape;
  auto domain = take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
      source, findForall(source)));
  auto found = llvm::find_if(
      domain, [&](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.forallOwnershipShape == Shape::LogicalThreadDomain &&
               point.canonicalIndexWidth == indexWidth;
      });
  if (found == domain.end())
    fail("logical thread-domain decision is absent");
  return *found;
}

void requireSourceInductionDefUse(
    const loom::frontend::MaterializedOwnershipCandidate &candidate,
    std::size_t lowerInputOrdinal, std::size_t stepInputOrdinal,
    unsigned indexWidth) {
  dataflow::ThreadOp thread;
  loom::SpatialRegionOp spatial;
  candidate.structuredProgram.module().walk([&](mlir::Operation *op) {
    if (auto value = llvm::dyn_cast<dataflow::ThreadOp>(op))
      thread = value;
    if (auto value = llvm::dyn_cast<loom::SpatialRegionOp>(op))
      spatial = value;
  });
  if (!thread || !spatial)
    fail("thread-domain candidate has no complete Spatial boundary");

  mlir::Block &threadEntry = thread.getBody().front();
  const std::size_t inputCount = thread.getFunctionType().getNumInputs();
  if (lowerInputOrdinal >= inputCount || stepInputOrdinal >= inputCount ||
      threadEntry.getNumArguments() != inputCount + 2)
    fail("thread-domain source-IV ABI is malformed");
  mlir::Value expectedLower = threadEntry.getArgument(lowerInputOrdinal);
  mlir::Value expectedStep = threadEntry.getArgument(stepInputOrdinal);
  mlir::Value expectedCoordinate = threadEntry.getArgument(inputCount + 1);

  auto graphInputBinding = [&](mlir::Value value) -> mlir::Value {
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || argument.getOwner() != &spatial.getBody().front())
      return {};
    const std::size_t applicationOrdinal = argument.getArgNumber();
    if (applicationOrdinal >= spatial.getValueInputs().size())
      return {};
    return spatial.getValueInputs()[applicationOrdinal];
  };

  mlir::arith::MulIOp scale;
  mlir::arith::AddIOp offset;
  spatial.walk([&](mlir::Operation *op) {
    if (auto value = llvm::dyn_cast<mlir::arith::MulIOp>(op))
      scale = value;
    if (auto value = llvm::dyn_cast<mlir::arith::AddIOp>(op))
      offset = value;
  });
  if (!scale || !offset)
    fail("source induction reconstruction is absent from the graph");

  const auto scaleType = llvm::dyn_cast<mlir::IntegerType>(scale.getType());
  const auto offsetType = llvm::dyn_cast<mlir::IntegerType>(offset.getType());
  if (!scaleType || !offsetType || scaleType != offsetType ||
      scaleType.getWidth() != indexWidth * 2)
    fail("source induction reconstruction does not use a wide exact domain");

  llvm::SmallVector<mlir::Value, 2> scaleBindings;
  for (mlir::Value operand : scale->getOperands()) {
    auto cast = operand.getDefiningOp<mlir::arith::IndexCastOp>();
    if (mlir::Value binding =
            cast ? graphInputBinding(cast.getIn()) : mlir::Value{})
      scaleBindings.push_back(binding);
  }
  if (!llvm::is_contained(scaleBindings, expectedStep) ||
      !llvm::is_contained(scaleBindings, expectedCoordinate))
    fail("source step and dense coordinate are not graph launch inputs");

  bool lowerIsLaunchInput = false;
  bool scaleFeedsOffset = false;
  for (mlir::Value operand : offset->getOperands()) {
    auto cast = operand.getDefiningOp<mlir::arith::IndexCastOp>();
    lowerIsLaunchInput |=
        cast && graphInputBinding(cast.getIn()) == expectedLower;
    scaleFeedsOffset |= operand == scale.getResult();
  }
  if (!lowerIsLaunchInput || !scaleFeedsOffset)
    fail("source lower bound is not paired with the reconstructed step");

  bool projectedToIndex = false;
  for (mlir::Operation *user : offset->getUsers()) {
    auto cast = llvm::dyn_cast<mlir::arith::IndexCastOp>(user);
    projectedToIndex |= cast && llvm::isa<mlir::IndexType>(cast.getType());
  }
  if (!projectedToIndex)
    fail("wide source induction was not projected to the thread index ABI");
}

void requireThreadDomainChoice(
    const loom::frontend::StructuredProgramCandidate &source,
    const loom::fabric::FinalizedFabricRoot &fabric) {
  using Shape = loom::frontend::ForallOwnershipShape;
  loom::frontend::StructuredEntityRef selected = findForall(source);
  auto domain = take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
      source, selected));
  std::optional<loom::frontend::SpatialOwnershipDecisionPoint> graphDecision;
  for (const loom::frontend::SpatialOwnershipDecisionPoint &decision : domain) {
    if (decision.canonicalIndexWidth != 64)
      continue;
    if (decision.forallOwnershipShape == Shape::GraphParallel)
      graphDecision = decision;
  }
  if (!graphDecision)
    fail("scf.forall ownership shape is missing from the 64-bit domain");

  auto sourceView = take(source.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      findKernel(source)};
  workloadDraft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{}};
  workloadDraft.observableContract.memories = {
      {loom::sim::EntryPointerArgumentTarget{0},
       loom::sim::MemoryObservationForm::FullState}};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  inputDraft.memoryObjects.push_back(
      loom::sim::RuntimeMemoryObject{std::vector<loom::sim::SemanticMemoryByte>(
          80, {loom::sim::SemanticState::Defined, 0xff})});
  inputDraft.pointerBindings.push_back({0, 0, 0});
  auto input = take(loom::sim::finalizeSimulationRuntimeInput(
      inputDraft, workload, sourceView));
  for (unsigned width : {32U, 64U}) {
    auto threadDecision = findThreadDecision(source, width);
    auto prepared = take(loom::frontend::prepareSpatialOwnershipSelection(
        source, {selected}, threadDecision));
    if (prepared.liveIns.size() != 3 || !prepared.sourceInductions ||
        prepared.sourceInductions->size() != 1 ||
        (*prepared.sourceInductions)[0].lowerInputOrdinal != 1 ||
        (*prepared.sourceInductions)[0].stepInputOrdinal != 2 ||
        constantIndex(prepared.liveIns[1]) != 4 ||
        constantIndex(prepared.liveIns[2]) != 2)
      fail("prepared source-IV boundary is not the canonical launch ABI");

    auto threadDomain =
        take(loom::frontend::materializeSpatialOwnershipDecision(
            source, {selected}, threadDecision, fabric));
    dataflow::ThreadOp thread;
    dataflow::ThreadLaunchOp launch;
    bool retainedForall = false;
    threadDomain.structuredProgram.module().walk([&](mlir::Operation *op) {
      if (auto candidate = llvm::dyn_cast<dataflow::ThreadOp>(op))
        thread = candidate;
      if (auto candidate = llvm::dyn_cast<dataflow::ThreadLaunchOp>(op))
        launch = candidate;
      retainedForall |= llvm::isa<mlir::scf::ForallOp>(op);
    });
    if (!thread || !launch)
      fail("logical thread-domain materialization lost its definition or "
           "launch");
    const std::size_t inputCount = thread.getFunctionType().getNumInputs();
    if (inputCount != 3 || launch.getBodyOperands().size() != inputCount ||
        constantIndex(launch.getBodyOperands()[1]) != 4 ||
        constantIndex(launch.getBodyOperands()[2]) != 2)
      fail("thread formals and launch operands diverged from the prepared ABI");
    if (thread.getBody().front().getNumArguments() != inputCount + 2)
      fail("logical thread definition did not materialize one coordinate");
    if (constantExtent(launch) != 8)
      fail("nonzero lower bound and nonunit step produced the wrong extent");
    if (retainedForall)
      fail("thread-domain materialization retained the source scf.forall");
    requireSourceInductionDefUse(threadDomain, 1, 2, width);

    auto replay = take(loom::sim::validateSourceBackedDfgReplay(
        source, {selected}, threadDecision, threadDomain, workload, input,
        {10000, 1000000, 1024 * 1024}));
    if (replay.status !=
            loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
        replay.dynamicActivations != 8 || replay.wavefrontSteps == 0 ||
        replay.eventCount == 0)
      fail("logical thread-domain DFG replay did not prove all activations");
    if (width == 64) {
      auto nonRetiring =
          take(loom::frontend::materializeSpatialOwnershipDecision(
              source, {selected}, threadDecision, fabric));
      auto nonRetiringModule =
          mlir::OwningOpRef<mlir::ModuleOp>(llvm::cast<mlir::ModuleOp>(
              nonRetiring.canonicalDataflow.module()->clone()));
      dataflow::StoreOp selectedStore;
      nonRetiringModule->walk([&](dataflow::StoreOp store) {
        if (selectedStore)
          fail("non-retiring replay fixture has multiple stores");
        selectedStore = store;
      });
      if (!selectedStore)
        fail("non-retiring replay fixture has no store");
      mlir::OpBuilder storeBuilder(selectedStore);
      mlir::Value outOfRange = mlir::arith::ConstantIndexOp::create(
          storeBuilder, selectedStore.getLoc(), 1000);
      selectedStore.getAddrMutable().set(outOfRange);
      auto nonRetiringDataflow =
          take(dataflow::finalizeCanonicalDataflow(nonRetiringModule.get()));
      loom::frontend::MaterializedOwnershipCandidate nonRetiringCandidate{
          std::move(nonRetiring.structuredProgram),
          std::move(nonRetiringDataflow)};
      auto nonRetiringReplay = take(loom::sim::validateSourceBackedDfgReplay(
          source, {selected}, threadDecision, nonRetiringCandidate, workload,
          input, {10000, 1000000, 1024 * 1024}));
      if (nonRetiringReplay.status !=
          loom::sim::SourceBackedDfgValidationStatus::Mismatch)
        fail("non-retiring candidate was not classified as a mismatch");

      auto withExtent = [&](std::uint64_t value) {
        auto selectedModule =
            mlir::OwningOpRef<mlir::ModuleOp>(llvm::cast<mlir::ModuleOp>(
                threadDomain.structuredProgram.module()->clone()));
        dataflow::ThreadLaunchOp selectedLaunch;
        selectedModule->walk(
            [&](dataflow::ThreadLaunchOp launch) { selectedLaunch = launch; });
        auto extent = selectedLaunch.getGridUpperBounds()
                          .front()
                          .getDefiningOp<mlir::arith::ConstantOp>();
        if (!extent)
          fail("selected activation mismatch fixture has no static extent");
        extent.setValueAttr(mlir::IntegerAttr::get(extent.getType(), value));
        auto program = take(
            loom::frontend::finalizeStructuredProgram(selectedModule.get()));
        auto dataflow = take(
            loom::lowering::lowerStructuredProgramToCanonicalDataflow(program));
        return loom::frontend::MaterializedOwnershipCandidate{
            std::move(program), std::move(dataflow)};
      };

      auto shortened = withExtent(7);
      auto mismatched = take(loom::sim::validateSourceBackedDfgReplay(
          source, {selected}, threadDecision, shortened, workload, input,
          {10000, 1000000, 1024 * 1024}));
      if (mismatched.status !=
              loom::sim::SourceBackedDfgValidationStatus::Mismatch ||
          mismatched.dynamicActivations != 8)
        fail("selected launch activation loss was not detected");

      auto expanded = withExtent(9);
      auto mismatchBeforeReplay = take(loom::sim::validateSourceBackedDfgReplay(
          source, {selected}, threadDecision, expanded, workload, input,
          {1, 1, 1024 * 1024}));
      if (mismatchBeforeReplay.status !=
              loom::sim::SourceBackedDfgValidationStatus::Mismatch ||
          mismatchBeforeReplay.dynamicActivations != 8 ||
          mismatchBeforeReplay.wavefrontSteps != 0 ||
          mismatchBeforeReplay.eventCount != 0)
        fail("activation multiplicity was not reconciled before graph replay");

      auto repeatedModule =
          mlir::OwningOpRef<mlir::ModuleOp>(llvm::cast<mlir::ModuleOp>(
              threadDomain.structuredProgram.module()->clone()));
      dataflow::ThreadLaunchOp repeatedLaunch;
      repeatedModule->walk(
          [&](dataflow::ThreadLaunchOp launch) { repeatedLaunch = launch; });
      if (!repeatedLaunch)
        fail("selected repeated-coordinate fixture has no thread launch");
      dataflow::ThreadWaitOp repeatedWait;
      for (mlir::Operation *user : repeatedLaunch->getUsers()) {
        auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(user);
        if (!wait || repeatedWait)
          fail("selected repeated-coordinate fixture has no unique wait");
        repeatedWait = wait;
      }
      if (!repeatedWait)
        fail("selected repeated-coordinate fixture has no thread wait");
      mlir::OpBuilder repeatBuilder(repeatedLaunch);
      mlir::Value lower = mlir::arith::ConstantIndexOp::create(
          repeatBuilder, repeatedLaunch.getLoc(), 0);
      mlir::Value upper = mlir::arith::ConstantIndexOp::create(
          repeatBuilder, repeatedLaunch.getLoc(), 2);
      mlir::Value step = mlir::arith::ConstantIndexOp::create(
          repeatBuilder, repeatedLaunch.getLoc(), 1);
      auto repeat = mlir::scf::ForOp::create(
          repeatBuilder, repeatedLaunch.getLoc(), lower, upper, step);
      repeatedLaunch->moveBefore(repeat.getBody()->getTerminator());
      repeatedWait->moveBefore(repeat.getBody()->getTerminator());
      auto repeatedProgram = take(
          loom::frontend::finalizeStructuredProgram(repeatedModule.get()));
      auto repeatedDataflow = take(
          loom::lowering::lowerStructuredProgramToCanonicalDataflow(
              repeatedProgram));
      loom::frontend::MaterializedOwnershipCandidate repeated{
          std::move(repeatedProgram), std::move(repeatedDataflow)};
      auto repeatedMismatch = take(loom::sim::validateSourceBackedDfgReplay(
          source, {selected}, threadDecision, repeated, workload, input,
          {1, 1, 1024 * 1024}));
      if (repeatedMismatch.status !=
              loom::sim::SourceBackedDfgValidationStatus::Mismatch ||
          repeatedMismatch.dynamicActivations != 8 ||
          repeatedMismatch.wavefrontSteps != 0 ||
          repeatedMismatch.eventCount != 0)
        fail("repeated coordinates were not rejected before graph replay");

      auto limited = loom::sim::validateSourceBackedDfgReplay(
          source, {selected}, threadDecision, threadDomain, workload, input,
          {10000, 1000000, 32});
      if (limited)
        fail("capture retained bytes exceeded an ignored execution limit");
      if (llvm::errorToErrorCode(limited.takeError()) !=
          std::make_error_code(std::errc::timed_out))
        fail("capture byte exhaustion used the wrong failure kind");

      auto requireExecutionLimit =
          [&](loom::sim::SourceBackedDfgValidationLimits limits,
              llvm::StringRef description) {
            auto result = loom::sim::validateSourceBackedDfgReplay(
                source, {selected}, threadDecision, threadDomain, workload,
                input, limits);
            if (result)
              fail((description + " was ignored").str());
            if (llvm::errorToErrorCode(result.takeError()) !=
                std::make_error_code(std::errc::timed_out))
              fail((description + " used the wrong failure kind").str());
          };
      requireExecutionLimit({1, 1000000, 1024 * 1024},
                            "wavefront execution limit");
      requireExecutionLimit({10000, 1, 1024 * 1024}, "event execution limit");
    }
  }
}

loom::frontend::StructuredProgramCandidate makeOverflowSource() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  llvm.func @kernel() {
    %lb = arith.constant -2147483648 : index
    %ub = arith.constant 2147483647 : index
    %step = arith.constant 1 : index
    scf.forall (%i) = (%lb) to (%ub) step (%step) {
      %i32 = arith.index_cast %i : index to i32
      %one = arith.constant 1 : i32
      %unused = arith.addi %i32, %one : i32
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the index-overflow fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

void requireIndexWidthRejection(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  using Shape = loom::frontend::ForallOwnershipShape;
  auto source = makeOverflowSource();
  loom::frontend::StructuredEntityRef selected = findForall(source);
  auto domain = take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
      source, selected));
  auto decision = llvm::find_if(
      domain, [](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.forallOwnershipShape == Shape::LogicalThreadDomain;
      });
  if (decision == domain.end())
    fail("overflow fixture has no logical thread-domain decision");
  auto candidate = loom::frontend::materializeSpatialOwnershipDecision(
      source, {selected}, *decision, fabric);
  if (candidate)
    fail("unrepresentable 32-bit logical extent was accepted");
  const std::string error = llvm::toString(candidate.takeError());
  if (error.find("thread-domain extent exceeds the selected signed index "
                 "width") == std::string::npos)
    fail("unrepresentable logical extent used the wrong rejection");
}

loom::frontend::StructuredProgramCandidate makeCoordinateOverflowSource() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  llvm.func @kernel(%base: !llvm.ptr) {
    %lb = arith.constant -2000000000 : index
    %ub = arith.constant 300000003 : index
    %step = arith.constant 1100000001 : index
    scf.forall (%i) = (%lb) to (%ub) step (%step) {
      %i64 = arith.index_cast %i : index to i64
      %lb64 = arith.constant -2000000000 : i64
      %step64 = arith.constant 1100000001 : i64
      %relative = arith.subi %i64, %lb64 : i64
      %lane64 = arith.divsi %relative, %step64 : i64
      %lane32 = arith.trunci %lane64 : i64 to i32
      %ptr = llvm.getelementptr inbounds %base[%lane32]
          : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4 x i8>
      llvm.store %lane32, %ptr : i32, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the coordinate-overflow fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

void requireWidenedCoordinateRecovery(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  auto source = makeCoordinateOverflowSource();
  auto selected = findForall(source);
  auto decision = findThreadDecision(source, std::nullopt);
  auto candidate = take(loom::frontend::materializeSpatialOwnershipDecision(
      source, {selected}, decision, fabric));
  auto sourceView = take(source.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      findKernel(source)};
  workloadDraft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{}};
  workloadDraft.observableContract.memories = {
      {loom::sim::EntryPointerArgumentTarget{0},
       loom::sim::MemoryObservationForm::FullState}};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  inputDraft.memoryObjects.push_back(
      loom::sim::RuntimeMemoryObject{std::vector<loom::sim::SemanticMemoryByte>(
          12, {loom::sim::SemanticState::Defined, 0})});
  inputDraft.pointerBindings.push_back({0, 0, 0});
  auto input = take(loom::sim::finalizeSimulationRuntimeInput(
      inputDraft, workload, sourceView));
  auto replay = take(loom::sim::validateSourceBackedDfgReplay(
      source, {selected}, decision, candidate, workload, input,
      {10000, 1000000, 1024 * 1024}));
  if (replay.status != loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      replay.dynamicActivations != 3)
    fail("coordinate recovery overflowed before exact division");
}

loom::frontend::StructuredProgramCandidate makeDynamicDomainSource() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%base: !llvm.ptr, %lb16: i16, %ub32: i32, %step16: i16) {
    %lb = arith.index_cast %lb16 : i16 to index
    %ub = arith.index_castui %ub32 : i32 to index
    %raw_step = arith.index_cast %step16 : i16 to index
    %one = arith.constant 1 : index
    %step = arith.maxsi %raw_step, %one : index
    scf.forall (%i) = (%lb) to (%ub) step (%step) {
      %relative = arith.subi %i, %lb : index
      %relative64 = arith.index_cast %relative : index to i64
      %ptr = llvm.getelementptr inbounds %base[%relative64]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      %zero = arith.constant 0.0 : f32
      llvm.store %zero, %ptr : f32, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the dynamic thread-domain fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

void requireDynamicThreadDomain(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  auto source = makeDynamicDomainSource();
  auto selected = findForall(source);
  auto decision = findThreadDecision(source, 64);
  auto candidate = take(loom::frontend::materializeSpatialOwnershipDecision(
      source, {selected}, decision, fabric));
  dataflow::ThreadLaunchOp launch;
  candidate.structuredProgram.module().walk(
      [&](dataflow::ThreadLaunchOp value) { launch = value; });
  if (!launch || launch.getGridUpperBounds().size() != 1 ||
      launch.getGridUpperBounds()
          .front()
          .getDefiningOp<mlir::arith::ConstantOp>())
    fail("dynamic source domain did not retain a dynamic exact extent");

  auto narrowDecision = findThreadDecision(source, 32);
  auto rejected = loom::frontend::prepareSpatialOwnershipSelection(
      source, {selected}, narrowDecision);
  if (rejected)
    fail("unsigned i32 upper domain was accepted by a signed i32 index ABI");
  if (llvm::toString(rejected.takeError())
          .find("complete signed value-domain proof") == std::string::npos)
    fail("dynamic width rejection did not report the missing value proof");

  auto sourceView = take(source.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      findKernel(source)};
  workloadDraft.argumentPlan = {loom::sim::StructuredRuntimeMemoryInput{},
                                loom::sim::StructuredRuntimeValueInput{},
                                loom::sim::StructuredRuntimeValueInput{},
                                loom::sim::StructuredRuntimeValueInput{}};
  workloadDraft.observableContract.memories = {
      {loom::sim::EntryPointerArgumentTarget{0},
       loom::sim::MemoryObservationForm::FullState}};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  inputDraft.memoryObjects.push_back(
      loom::sim::RuntimeMemoryObject{std::vector<loom::sim::SemanticMemoryByte>(
          80, {loom::sim::SemanticState::Defined, 0xff})});
  inputDraft.pointerBindings.push_back({0, 0, 0});
  auto scalar = [](unsigned width, std::int64_t value) {
    loom::sim::CanonicalValueSequence sequence;
    sequence.tokenCount = 1;
    sequence.lanes.push_back(loom::sim::SemanticLane::defined(
        llvm::APInt(width, static_cast<std::uint64_t>(value), true)));
    return sequence;
  };
  inputDraft.runtimeValues = {
      {1, scalar(16, -4)}, {2, scalar(32, 8)}, {3, scalar(16, 3)}};
  auto input = take(loom::sim::finalizeSimulationRuntimeInput(
      inputDraft, workload, sourceView));
  auto replay = take(loom::sim::validateSourceBackedDfgReplay(
      source, {selected}, decision, candidate, workload, input,
      {10000, 1000000, 1024 * 1024}));
  if (replay.status != loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      replay.dynamicActivations != 4)
    fail("dynamic lower and step did not replay the exact source domain");
}

loom::frontend::StructuredProgramCandidate
makeDirectPointerSource(bool conflictingEndianness,
                        bool includeLlvmDataLayout = true) {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%base: !llvm.ptr) {
    %extent = arith.constant 3 : index
    scf.forall (%i) in (%extent) {
      %unused = llvm.load %base : !llvm.ptr -> i32
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the direct-pointer fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  if (includeLlvmDataLayout)
    module->getOperation()->setAttr(
        "llvm.data_layout",
        mlir::StringAttr::get(&context, nativeDataLayout()));
  if (conflictingEndianness || !includeLlvmDataLayout) {
    mlir::StringAttr key = mlir::StringAttr::get(
        &context, mlir::DLTIDialect::kDataLayoutEndiannessKey);
    const bool little = llvm::DataLayout(nativeDataLayout()).isLittleEndian();
    llvm::StringRef endianness =
        little != conflictingEndianness
            ? mlir::DLTIDialect::kDataLayoutEndiannessLittle
            : mlir::DLTIDialect::kDataLayoutEndiannessBig;
    module->getOperation()->setAttr(
        mlir::DLTIDialect::kDataLayoutAttrName,
        mlir::DataLayoutSpecAttr::get(
            &context, {mlir::DataLayoutEntryAttr::get(
                          key, mlir::StringAttr::get(&context, endianness))}));
  }
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

void requireDataLayoutProjectionAndFailureKind() {
  auto source = makeDirectPointerSource(false);
  auto decision = findThreadDecision(source, std::nullopt);
  auto prepared = take(loom::frontend::prepareSpatialOwnershipSelection(
      source, {findForall(source)}, decision));
  auto preservedLayout =
      prepared.module.get().getOperation()->getAttrOfType<mlir::StringAttr>(
          mlir::LLVM::LLVMDialect::getDataLayoutAttrName());
  if (!preservedLayout || preservedLayout.getValue() != nativeDataLayout())
    fail("Structured candidate rewrote the exact LLVM DataLayout spelling");
  mlir::DataLayoutSpecInterface layout = prepared.module->getDataLayoutSpec();
  mlir::StringAttr key =
      mlir::StringAttr::get(prepared.module->getContext(),
                            mlir::DLTIDialect::kDataLayoutEndiannessKey);
  if (!layout || !layout.getSpecForIdentifier(key))
    fail("LLVM DataLayout endianness was not projected for direct memory");

  auto conflicting = makeDirectPointerSource(true);
  auto rejected = loom::frontend::prepareSpatialOwnershipSelection(
      conflicting, {findForall(conflicting)},
      findThreadDecision(conflicting, std::nullopt));
  if (rejected)
    fail("conflicting LLVM and DLTI endianness was accepted");
  bool candidateRejection = false;
  std::string detail;
  llvm::handleAllErrors(
      rejected.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &error) {
        candidateRejection = true;
        detail = error.message();
      },
      [&](const llvm::ErrorInfoBase &error) {
        llvm::raw_string_ostream stream(detail);
        error.log(stream);
      });
  if (candidateRejection ||
      detail.find("DataLayout endianness") == std::string::npos)
    fail("malformed parent DataLayout was classified as candidate pruning");

  auto missingLlvmLayout = makeDirectPointerSource(false, false);
  auto missing = loom::frontend::prepareSpatialOwnershipSelection(
      missingLlvmLayout, {findForall(missingLlvmLayout)},
      findThreadDecision(missingLlvmLayout, std::nullopt));
  if (missing)
    fail("DLTI endianness replaced the LLVM DataLayout authority");
  candidateRejection = false;
  detail.clear();
  llvm::handleAllErrors(
      missing.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &error) {
        candidateRejection = true;
        detail = error.message();
      },
      [&](const llvm::ErrorInfoBase &error) {
        llvm::raw_string_ostream stream(detail);
        error.log(stream);
      });
  if (candidateRejection ||
      detail.find("nonempty LLVM DataLayout") == std::string::npos)
    fail("missing LLVM DataLayout used candidate pruning or a DLTI fallback");
}

void requireOverlappingPlainWriteRejection() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%base: !llvm.ptr) {
    %extent = arith.constant 3 : index
    scf.forall (%i) in (%extent) {
      %one = arith.constant 1 : i32
      llvm.store %one, %base : i32, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the overlapping-write fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  auto plain = take(loom::frontend::finalizeStructuredProgram(module.get()));
  auto rejected = loom::frontend::prepareSpatialOwnershipSelection(
      plain, {findForall(plain)}, findThreadDecision(plain, std::nullopt));
  if (rejected)
    fail("overlapping plain writes were accepted as a logical thread domain");
  if (llvm::toString(rejected.takeError())
          .find("parallel dependence and effect legality") == std::string::npos)
    fail("overlapping plain writes used the wrong candidate rejection");
}

void requireMixedWidthOverlapRejection() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%base: !llvm.ptr) {
    %extent = arith.constant 2 : index
    scf.forall (%i) in (%extent) {
      %i64 = arith.index_cast %i : index to i64
      %wide_ptr = llvm.getelementptr inbounds %base[%i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
      %narrow_ptr = llvm.getelementptr inbounds %base[%i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      %wide = arith.constant 0 : i64
      %narrow = arith.constant 0 : i32
      llvm.store %wide, %wide_ptr : i64, !llvm.ptr
      llvm.store %narrow, %narrow_ptr : i32, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the mixed-width overlap fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  auto source = take(loom::frontend::finalizeStructuredProgram(module.get()));
  auto rejected = loom::frontend::prepareSpatialOwnershipSelection(
      source, {findForall(source)}, findThreadDecision(source, 64));
  if (rejected)
    fail("mixed-width overlapping writes were accepted as independent lanes");
  if (llvm::toString(rejected.takeError())
          .find("parallel dependence and effect legality") == std::string::npos)
    fail("mixed-width overlap used the wrong candidate rejection");
}

void requireNarrowIndexCastAliasRejection() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @kernel(%base: !llvm.ptr) {
    %extent = arith.constant 3 : index
    scf.forall (%i) in (%extent) {
      %narrow = arith.index_cast %i : index to i1
      %ptr = llvm.getelementptr inbounds %base[%narrow]
          : (!llvm.ptr, i1) -> !llvm.ptr, !llvm.array<1 x i8>
      %value = arith.constant 1 : i8
      llvm.store %value, %ptr : i8, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the narrow-index-cast alias fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  auto source = take(loom::frontend::finalizeStructuredProgram(module.get()));
  auto rejected = loom::frontend::prepareSpatialOwnershipSelection(
      source, {findForall(source)}, findThreadDecision(source, 64));
  if (rejected)
    fail("narrow signed index cast hid an inter-lane write alias");
  if (llvm::toString(rejected.takeError())
          .find("parallel dependence and effect legality") == std::string::npos)
    fail("narrow index-cast alias used the wrong candidate rejection");
}

void requireScaledByteAddressBoundary(
    const loom::fabric::FinalizedFabricRoot &fabric) {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  llvm.func @kernel(%base: !llvm.ptr) {
    %extent = arith.constant 5 : index
    scf.forall (%i) in (%extent) {
      %i32 = arith.index_cast %i : index to i32
      %ptr = llvm.getelementptr inbounds %base[%i32]
          : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1073741824 x i8>
      %value = arith.constant 1 : i8
      llvm.store %value, %ptr : i8, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the scaled-byte-overflow fixture");
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  auto source = take(loom::frontend::finalizeStructuredProgram(module.get()));
  auto rejected = loom::frontend::prepareSpatialOwnershipSelection(
      source, {findForall(source)}, findThreadDecision(source, std::nullopt));
  if (rejected)
    fail("scaled byte address overflow was accepted by a narrow index ABI");
  if (llvm::toString(rejected.takeError())
          .find("parallel dependence and effect legality") == std::string::npos)
    fail("scaled byte address overflow used the wrong candidate rejection");

  auto acceptedModule = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  llvm.func @kernel(%base: !llvm.ptr) {
    %extent = arith.constant 5 : index
    scf.forall (%i) in (%extent) {
      %i32 = arith.index_cast %i : index to i32
      %ptr = llvm.getelementptr inbounds %base[%i32]
          : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1073741824 x i8>
      %value = arith.constant 1 : i32
      llvm.store %value, %ptr : i32, !llvm.ptr
      scf.forall.in_parallel {}
    }
    llvm.return
  }
}
)mlir",
                                                               &context);
  if (!acceptedModule)
    fail("cannot parse the scaled-element-address fixture");
  acceptedModule->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  acceptedModule->getOperation()->setAttr(
      "llvm.data_layout", mlir::StringAttr::get(&context, nativeDataLayout()));
  auto acceptedSource =
      take(loom::frontend::finalizeStructuredProgram(acceptedModule.get()));
  auto candidate = take(loom::frontend::materializeSpatialOwnershipDecision(
      acceptedSource, {findForall(acceptedSource)},
      findThreadDecision(acceptedSource, std::nullopt), fabric));

  bool sawElementScale = false;
  candidate.canonicalDataflow.module().walk([&](mlir::Operation *operation) {
    for (mlir::Type type : operation->getResultTypes()) {
      auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
      if (integer && integer.getWidth() > 32)
        fail("scaled element address introduced an over-wide graph actor");
    }
    auto multiply = llvm::dyn_cast<mlir::arith::MulIOp>(operation);
    if (!multiply)
      return;
    for (mlir::Value operand : multiply->getOperands()) {
      mlir::Value source = operand;
      if (auto cast = source.getDefiningOp<mlir::arith::IndexCastOp>())
        source = cast.getIn();
      auto constant = source.getDefiningOp<mlir::arith::ConstantOp>();
      auto value = constant
                       ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                       : mlir::IntegerAttr{};
      if (!value) {
        auto graphConstant = source.getDefiningOp<dataflow::ConstantOp>();
        value = graphConstant
                    ? llvm::dyn_cast<mlir::IntegerAttr>(
                          graphConstant.getConstValue())
                    : mlir::IntegerAttr{};
      }
      sawElementScale |=
          value && value.getValue().getSExtValue() == 268435456;
    }
  });
  if (!sawElementScale)
    fail("scaled element address did not preserve the exact element stride");
}

} // namespace

int main() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-thread-domain", directory);
  if (error)
    fail("cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  loom::frontend::StructuredProgramCandidate source = makeSource();
  requireThreadDomainChoice(source, design.roots().front());
  requireIndexWidthRejection(design.roots().front());
  requireWidenedCoordinateRecovery(design.roots().front());
  requireNarrowIndexCastAliasRejection();
  requireScaledByteAddressBoundary(design.roots().front());
  requireDynamicThreadDomain(design.roots().front());
  requireDataLayoutProjectionAndFailureKind();
  requireOverlappingPlainWriteRejection();
  requireMixedWidthOverlapRejection();
  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove artifact store directory: " + error.message());
  llvm::outs() << "structured thread-domain anchor passed\n";
  return EXIT_SUCCESS;
}
