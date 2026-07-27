#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
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
#include <cstring>
#include <memory>
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
                                          llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

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

loom::sim::CanonicalValueSequence oneDefinedValue(unsigned width,
                                                  std::uint64_t value) {
  loom::sim::CanonicalValueSequence sequence;
  sequence.tokenCount = 1;
  sequence.lanes.push_back(
      loom::sim::SemanticLane::defined(llvm::APInt(width, value)));
  return sequence;
}

loom::sim::RuntimeMemoryObject floatObject(float scale) {
  loom::sim::RuntimeMemoryObject object;
  object.initialBytes.reserve(64 * sizeof(float));
  for (std::uint32_t index = 0; index < 64; ++index) {
    const float value = static_cast<float>(index) * scale;
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned byte = 0; byte < sizeof(bits); ++byte)
      object.initialBytes.push_back(loom::sim::SemanticMemoryByte{
          loom::sim::SemanticState::Defined,
          static_cast<std::uint8_t>(bits >> (byte * 8))});
  }
  return object;
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

  loom::sim::SpatialSimulationWorkload workload{onlyLaunch(test, view)};
  workload.valueInputPlan = {oneDefinedValue(64, 1), oneDefinedValue(64, 64),
                             oneDefinedValue(64, 0)};
  workload.observableContract.memories.push_back(
      loom::sim::SpatialMemoryObservable{
          dataflow::LogicalMemoryRootOrViewRef{memoryRoot(test, view, 2)},
          loom::sim::MemoryObservationForm::DiffFromRuntimeInput});
  auto finalizedWorkload =
      take(test, loom::sim::finalizeSimulationWorkload(workload, view));

  loom::sim::SpatialSimulationRuntimeInputDraft input{
      finalizedWorkload.identity()};
  input.memoryObjects = {floatObject(1.0F), floatObject(0.5F),
                         floatObject(0.0F)};
  input.memoryRootBindings = {
      loom::sim::RuntimeMemoryBindingDraft{memoryRoot(test, view, 0), 0, 0},
      loom::sim::RuntimeMemoryBindingDraft{memoryRoot(test, view, 1), 1, 0},
      loom::sim::RuntimeMemoryBindingDraft{memoryRoot(test, view, 2), 2, 0}};
  auto finalizedInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                       input, finalizedWorkload, view));

  loom::sim::DFGSimulationReport report = take(
      test, loom::sim::simulateDfgWorkload(candidate.canonicalDataflow,
                                           finalizedWorkload, finalizedInput));
  if (report.status != "pass")
    fail(test, "typed DFG execution did not retire: " + report.status);
  if (report.operationFireCounts[dataflow::OperationSchemaId::ArithAddF] !=
          64 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowLoad] !=
          128 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowStore] !=
          64)
    fail(test, "typed DFG execution did not run the vecadd workload");

  auto destination = report.finalMemoryState.find("arg5");
  if (destination == report.finalMemoryState.end() ||
      destination->second.size() != 64 ||
      destination->second.front() != "f32:0" ||
      destination->second[2] != "f32:3" ||
      destination->second.back() != "f32:94.500000")
    fail(test, "typed DFG execution produced the wrong destination state");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

} // namespace

int main() {
  sourceCandidateExecutesThroughTypedDfgInput();
  llvm::outs() << "frontend to typed DFG integration anchor passed\n";
  return EXIT_SUCCESS;
}
