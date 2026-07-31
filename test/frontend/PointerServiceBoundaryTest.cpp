#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  auto candidate = take(loom::frontend::materializeSpatialOwnershipDecision(
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

  auto view = take(candidate.canonicalDataflow.view());
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
      compiled.structuredProgram, scope, domain.front(), candidate, workload,
      runtimeInput,
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

  if (std::error_code error = llvm::sys::fs::remove_directories(directory))
    fail("cannot remove artifact store: " + error.message());
}

} // namespace

int main() {
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail("cannot initialize the native target");
  pointerServiceBoundary();
  llvm::outs() << "pointer service boundary anchor passed\n";
  return EXIT_SUCCESS;
}
