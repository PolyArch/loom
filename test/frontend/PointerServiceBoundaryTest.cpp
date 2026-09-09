#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Analysis/StoredMemoryProvenance.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "pointerServiceBoundary: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<llvm::Module> parseCursor(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define internal i32 @cursor(ptr %descriptor) {
entry:
  %pointer = load ptr, ptr %descriptor, align 8
  %value = load i16, ptr %pointer, align 1
  %extended = zext i16 %value to i32
  %next = getelementptr inbounds i8, ptr %pointer, i64 2
  store ptr %next, ptr %descriptor, align 8
  ret i32 %extended
}

define i32 @main() {
entry:
  %bytes = alloca [2 x i8], align 1
  %descriptor = alloca ptr, align 8
  store ptr %bytes, ptr %descriptor, align 8
  %value = call i32 @cursor(ptr %descriptor)
  ret i32 %value
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<pointer-cursor>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (module)
    return module;
  std::string message;
  llvm::raw_string_ostream stream(message);
  diagnostic.print("pointerServiceBoundary", stream);
  fail(stream.str());
}

std::unique_ptr<llvm::Module>
parsePostTestedPointerLoop(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define internal void @copy(ptr %source, ptr %destination, i16 %count) {
entry:
  %remaining.initial = zext i16 %count to i64
  br label %loop

loop:
  %remaining = phi i64 [ %remaining.initial, %entry ], [ %remaining.next, %loop ]
  %source.current = phi ptr [ %source, %entry ], [ %source.next, %loop ]
  %destination.current = phi ptr [ %destination, %entry ], [ %destination.next, %loop ]
  %value = load double, ptr %source.current, align 8
  store double %value, ptr %destination.current, align 8
  %source.next = getelementptr inbounds i8, ptr %source.current, i64 8
  %destination.next = getelementptr inbounds i8, ptr %destination.current, i64 8
  %remaining.next = add i64 %remaining, -1
  %more = icmp ne i64 %remaining.next, 0
  br i1 %more, label %loop, label %exit

exit:
  ret void
}

define i32 @main() {
entry:
  %source = alloca [2 x double], align 8
  %destination = alloca [2 x double], align 8
  %source.0 = getelementptr [2 x double], ptr %source, i64 0, i64 0
  %source.1 = getelementptr [2 x double], ptr %source, i64 0, i64 1
  store double 1.250000e+00, ptr %source.0, align 8
  store double 2.500000e+00, ptr %source.1, align 8
  call void @copy(ptr %source.0, ptr %destination, i16 2)
  %destination.1 = getelementptr [2 x double], ptr %destination, i64 0, i64 1
  %result = load double, ptr %destination.1, align 8
  %matches = fcmp oeq double %result, 2.500000e+00
  %status = select i1 %matches, i32 0, i32 1
  ret i32 %status
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<post-tested-pointer-loop>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (module)
    return module;
  std::string message;
  llvm::raw_string_ostream stream(message);
  diagnostic.print("postTestedPointerLoop", stream);
  fail(stream.str());
}

loom::frontend::StructuredEntityRef
findCallable(const loom::frontend::StructuredProgramCandidate &program,
             llvm::StringRef name) {
  auto view = take(program.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getSymName() == name)
      return entity.reference;
  }
  fail("Structured Program omitted " + name.str());
}

void pointerServiceBoundary() {
  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-pointer-service-boundary", directory))
    fail("cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(loom::frontend::compileLlvmModuleToPreMapping(
      parseCursor(context), design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipScope scope{
      findCallable(compiled.structuredProgram, "cursor")};
  auto domain = take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
      compiled.structuredProgram, scope.selection));
  if (domain.size() != 1)
    fail("pointer cursor has an ambiguous ownership decision");
  auto candidate =
      take(loom::frontend::materializeSpatialOwnershipDecision(
          compiled.structuredProgram, scope, domain.front(),
          design.roots().front()));

  auto cursor =
      candidate.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "cursor");
  if (!cursor)
    fail("candidate lost the LLVM callable authority");
  unsigned hostPointerLoads = 0;
  unsigned launches = 0;
  cursor.getBody().walk([&](mlir::Operation *operation) {
    if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(operation))
      hostPointerLoads +=
          llvm::isa<mlir::LLVM::LLVMPointerType>(load.getResult().getType());
    launches += llvm::isa<dataflow::ThreadLaunchOp>(operation);
  });
  if (hostPointerLoads != 1 || launches != 1)
    fail("dynamic pointer service source was not retained as one host prelude");

  unsigned services = 0;
  candidate.canonicalDataflow.module().walk(
      [&](dataflow::MemoryServiceOp) { ++services; });
  if (services != 2)
    fail("descriptor and pointee did not acquire distinct memory services");

  const auto &view = candidate.canonicalDataflow.view();
  if (view.graphs().size() != 1)
    fail("pointer cursor did not publish one canonical graph");
  auto graph = llvm::dyn_cast_or_null<dataflow::GraphOp>(view.graphs()[0].op);
  if (!graph || graph.getFunctionType().getNumResults() != 1 ||
      !graph.getFunctionType().getResult(0).isInteger(32))
    fail("pointer cursor graph lost its scalar result");
  unsigned loads = 0;
  unsigned stores = 0;
  unsigned geps = 0;
  graph.getBody().walk([&](mlir::Operation *operation) {
    loads += llvm::isa<dataflow::LoadOp>(operation);
    stores += llvm::isa<dataflow::StoreOp>(operation);
    geps += llvm::isa<mlir::LLVM::GEPOp>(operation);
    if (llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp>(operation))
      fail("canonical graph retained an imperative memory operation");
  });
  if (loads != 1 || stores != 1 || geps != 1)
    fail("pointer cursor graph lost load, pointer update, or store semantics");

  auto sourceView = take(compiled.structuredProgram.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      findCallable(compiled.structuredProgram, "main")};
  workloadDraft.observableContract.returnValue = true;
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  auto runtimeInput = take(loom::sim::finalizeSimulationRuntimeInput(
      runtimeDraft, workload, sourceView));
  auto replay = take(loom::sim::validateSourceBackedDfgReplay(
      compiled.structuredProgram, candidate, workload, runtimeInput,
      {/*maxWavefrontSteps=*/1000, /*maxEventCount=*/10000,
       /*maxRetainedCaptureBytes=*/1024 * 1024}));
  if (replay.status != loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      replay.dynamicActivations != 1 || replay.eventCount == 0)
    fail("pointer cursor replay did not prove one nonempty equivalent graph: "
         "status=" +
         std::to_string(static_cast<unsigned>(replay.status)) +
         ", activations=" + std::to_string(replay.dynamicActivations) +
         ", value_lanes=" + std::to_string(replay.valueLanesCompared) +
         ", memory_bytes=" + std::to_string(replay.memoryBytesCompared) +
         ", events=" + std::to_string(replay.eventCount));

  std::vector<mlir::OpResult> hostBases;
  candidate.structuredProgram.module().walk(
      [&](mlir::LLVM::AllocaOp allocation) {
        hostBases.push_back(llvm::cast<mlir::OpResult>(allocation.getRes()));
      });
  auto lowered = take(
      loom::lowering::lowerStructuredProgramToCanonicalDataflowWithProjection(
          candidate.structuredProgram, {}, hostBases));
  if (hostBases.empty() || lowered.trackedValues.size() != hostBases.size() ||
      lowered.spatialGraphs.size() != 1)
    fail("pointer cursor fixture lost its host bases or selected launch");
  const auto reference =
      [](const dataflow::CanonicalDataflowArtifact &artifact) {
        return loom::ArtifactRootReference{
            dataflow::canonicalDataflowSchema.identity.str(),
            dataflow::canonicalDataflowSchema.version, artifact.identity()};
      };
  std::vector<dataflow::DataflowRewriteDerivation> derivations;
  std::optional<dataflow::CanonicalDataflowArtifact> child;
  const dataflow::CanonicalDataflowArtifact *parent = &lowered.artifact;
  // Two transactions exercise composition, including rebinding the second
  // transaction's SSA inputs to the first child's owned module.
  for (unsigned step = 0; step != 2; ++step) {
    bool changed = false;
    for (const auto &decision :
         take(dataflow::enumerateFixedDataflowRewriteDecisions(*parent))) {
      auto rewritten =
          take(dataflow::materializeDataflowRewrite(*parent, decision));
      if (!rewritten || rewritten->identity() == lowered.artifact.identity())
        continue;
      derivations.push_back(
          {reference(*parent), reference(*rewritten), decision});
      child.emplace(std::move(*rewritten));
      parent = &*child;
      changed = true;
      break;
    }
    if (!changed)
      fail("pointer cursor has no distinct two-step rewrite path");
  }
  const auto target = reference(*child);
  auto missing = dataflow::replayDataflowRewriteDerivationsWithTrackedEntities(
      take(loom::lowering::lowerStructuredProgramToCanonicalDataflow(
          candidate.structuredProgram)),
      target, {}, {});
  if (missing)
    fail("rewritten pointer cursor accepted missing derivation");
  llvm::consumeError(missing.takeError());
  auto wrongDerivations = derivations;
  wrongDerivations.back().decision =
      dataflow::PackUnpackRoundTripRewrite{dataflow::ActorId(1)};
  auto wrong = dataflow::replayDataflowRewriteDerivationsWithTrackedEntities(
      take(loom::lowering::lowerStructuredProgramToCanonicalDataflow(
          candidate.structuredProgram)),
      target, wrongDerivations, {});
  if (wrong)
    fail("rewritten pointer cursor accepted a foreign rewrite decision");
  llvm::consumeError(wrong.takeError());
  std::reverse(derivations.begin(), derivations.end());
  auto replayed =
      take(dataflow::replayDataflowRewriteDerivationsWithTrackedEntities(
          std::move(lowered.artifact), target, derivations,
          {lowered.spatialGraphs.front().staticGraphLaunch},
          lowered.trackedValues));
  if (replayed.artifact.identity() != child->identity() ||
      replayed.trackedStaticGraphLaunches.size() != 1 ||
      replayed.trackedValues.size() != hostBases.size())
    fail("rewrite replay lost its exact artifact or tracked entities");
  take(replayed.artifact.view().resolve(
      replayed.trackedStaticGraphLaunches.front()));
  for (mlir::Value base : replayed.trackedValues) {
    auto allocation = base.getDefiningOp<mlir::LLVM::AllocaOp>();
    if (!allocation || allocation->getParentOfType<mlir::ModuleOp>() !=
                           replayed.artifact.module())
      fail("rewrite replay retained a foreign host allocation value");
  }

  if (std::error_code error = llvm::sys::fs::remove_directories(directory))
    fail("cannot remove artifact store: " + error.message());
}

void exactPointerAddressingFallback() {
  llvm::SmallString<128> directory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-pointer-addressing-fallback", directory))
    fail("cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(loom::frontend::compileLlvmModuleToPreMapping(
      parsePostTestedPointerLoop(context), design.roots().front().reference(),
      store));
  loom::frontend::SpatialOwnershipScope scope{
      findCallable(compiled.structuredProgram, "copy")};
  auto domain = take(loom::frontend::enumerateSpatialOwnershipDecisionDomain(
      compiled.structuredProgram, scope.selection));

  std::optional<loom::frontend::SpatialOwnershipDecisionPoint> rootRelative;
  std::optional<loom::frontend::SpatialOwnershipDecisionPoint> pointerAddressed;
  for (const auto &decision : domain) {
    if (!decision.addressProjection || decision.directCallSpecializationShape)
      continue;
    if (const auto *root =
            std::get_if<loom::frontend::RootRelativeAddressProjection>(
                &*decision.addressProjection);
        root && root->canonicalIndexWidth == 64)
      rootRelative = decision;
    if (std::holds_alternative<
            loom::frontend::PointerAddressedAddressProjection>(
            *decision.addressProjection))
      pointerAddressed = decision;
  }
  if (!rootRelative || !pointerAddressed)
    fail("address decision domain omitted an exact projection");

  auto normalized = loom::frontend::materializeSpatialOwnershipDecision(
      compiled.structuredProgram, scope, *rootRelative, design.roots().front());
  if (normalized)
    fail("unproven root-relative pointer induction was accepted");
  std::string message = llvm::toString(normalized.takeError());
  if (message.find("pointer induction offset") == std::string::npos)
    fail("root-relative rejection lost its proof failure: " + message);

  auto candidate =
      take(loom::frontend::materializeSpatialOwnershipDecision(
          compiled.structuredProgram, scope, *pointerAddressed,
          design.roots().front()));
  const auto &view = candidate.canonicalDataflow.view();
  if (view.graphs().size() != 1 || view.actors().empty())
    fail("pointer-addressed decision did not publish a nonempty graph");

  bool sawPointerCarry = false;
  bool sawPointerLoad = false;
  bool sawPointerStore = false;
  bool sawTypedGep = false;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(actor.op))
      sawPointerCarry |= dataflow::DataflowDialect::isPointerValueType(
          carry.getOutput().getType());
    if (auto load = llvm::dyn_cast<dataflow::LoadOp>(actor.op))
      sawPointerLoad |= dataflow::DataflowDialect::isPointerValueType(
          load.getAddr().getType());
    if (auto store = llvm::dyn_cast<dataflow::StoreOp>(actor.op))
      sawPointerStore |= dataflow::DataflowDialect::isPointerValueType(
          store.getAddr().getType());
    sawTypedGep |= llvm::isa<mlir::LLVM::GEPOp>(actor.op);
  }
  if (!sawPointerCarry || !sawPointerLoad || !sawPointerStore || !sawTypedGep)
    fail("pointer-addressed graph lost its exact address representation");

  auto sourceView = take(compiled.structuredProgram.view());
  loom::sim::StructuredProgramSimulationWorkload workloadDraft{
      findCallable(compiled.structuredProgram, "main")};
  workloadDraft.observableContract.returnValue = true;
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, sourceView));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  auto runtimeInput = take(loom::sim::finalizeSimulationRuntimeInput(
      runtimeDraft, workload, sourceView));
  auto replay = take(loom::sim::validateSourceBackedDfgReplay(
      compiled.structuredProgram, candidate, workload, runtimeInput,
      {/*maxWavefrontSteps=*/1000, /*maxEventCount=*/10000,
       /*maxRetainedCaptureBytes=*/1024 * 1024}));
  if (replay.status != loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      replay.dynamicActivations != 1 || replay.eventCount == 0)
    fail("pointer-addressed replay was not exactly equivalent");

  if (std::error_code error = llvm::sys::fs::remove_directories(directory))
    fail("cannot remove artifact store: " + error.message());
}

// A complete byte fill initializes a pointer representation. A short or
// strided fill must keep the partial-representation refusal, even when a
// conditional full pointer store supplies the only possible non-null root.
void byteFillPointerRepresentation() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                      mlir::scf::SCFDialect>();
  constexpr llvm::StringLiteral source = R"mlir(
module attributes {llvm.data_layout = "e-p:64:64"} {
  llvm.func @fill(%choose: i1) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %upper = arith.constant 8 : i64
    %step = arith.constant 1 : i64
    %byte = arith.constant 0 : i8
    %slot = llvm.alloca %upper x i8 : (i64) -> !llvm.ptr
    %target = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
    scf.for %index = %zero to %upper step %step : i64 {
      %address = llvm.getelementptr %slot[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      llvm.store %byte, %address : i8, !llvm.ptr
    }
    scf.if %choose {
      llvm.store %target, %slot : !llvm.ptr, !llvm.ptr
    }
    %pointer = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    llvm.return
  }
}
)mlir";
  for (auto [upper, step] : {std::pair<int64_t, int64_t>{8, 1},
                             {7, 1}, {8, 2}}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    if (!module)
      fail("cannot parse byte-fill pointer representation");
    auto callable = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("fill");
    mlir::scf::ForOp loop;
    mlir::LLVM::LoadOp read;
    mlir::Value target;
    callable.walk([&](mlir::scf::ForOp operation) { loop = operation; });
    callable.walk([&](mlir::LLVM::LoadOp operation) { read = operation; });
    callable.walk([&](mlir::LLVM::AllocaOp operation) {
      target = operation.getResult();
    });
    // Keep the allocation at eight bytes while varying only the write domain.
    mlir::OpBuilder builder(loop);
    loop.getUpperBoundMutable().assign(
        mlir::arith::ConstantIntOp::create(builder, loop.getLoc(), upper, 64));
    loop.getStepMutable().assign(
        mlir::arith::ConstantIntOp::create(builder, loop.getLoc(), step, 64));
    loom::frontend::analysis::StoredMemoryProvenance provenance(callable);
    auto outcome = provenance.projectPointerTarget(read.getResult());
    if (upper == 8 && step == 1) {
      auto *root = std::get_if<
          loom::frontend::analysis::StoredPointerTarget>(&outcome);
      if (!root || root->root != target || !root->mayBeNull)
        fail("complete byte fill lost the nullable pointer origin");
    } else {
      auto *refusal = std::get_if<
          loom::frontend::analysis::StoredPointerRefusal>(&outcome);
      if (!refusal || *refusal != loom::frontend::analysis::
                                     StoredPointerRefusal::PartialPointerWrite)
        fail("incomplete byte fill admitted a pointer representation");
    }
  }
}

} // namespace

int main() {
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail("cannot initialize the native target");
  byteFillPointerRepresentation();
  pointerServiceBoundary();
  exactPointerAddressingFallback();
  llvm::outs() << "pointer service boundary anchor passed\n";
  return EXIT_SUCCESS;
}
