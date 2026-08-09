#include "RootCompleteSpatialFeedbackTestSupport.h"

#include "Common/ArtifactStore.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <utility>

namespace loom::test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "root-complete Spatial feedback fixture failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

} // namespace

frontend::StructuredEntityRef
findStructuredCallable(const frontend::StructuredProgramCandidate &candidate,
                       llvm::StringRef name) {
  auto view = take(candidate.view());
  for (const frontend::StructuredEntity &entity :
       view.entities(frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getSymName() == name)
      return entity.reference;
  }
  fail("callable is absent from the Structured Program: " + name);
}

frontend::StructuredProgramCandidate
buildWideVectorStructuredSource(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func internal @kernel(%value: vector<4xi64>) -> vector<4xi64> {
    %sum = arith.addi %value, %value : vector<4xi64>
    llvm.return %sum : vector<4xi64>
  }
  llvm.func @main() -> i32 {
    %value = arith.constant dense<[1, 2, 3, 4]> : vector<4xi64>
    %result = llvm.call @kernel(%value)
        : (vector<4xi64>) -> vector<4xi64>
    %zero = arith.constant 0 : i32
    llvm.return %zero : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse wide-vector Structured source fixture");
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
  return take(frontend::finalizeStructuredProgram(module.get()));
}

PublishedStructuredSimulationInputs publishWideVectorStructuredInputs(
    const frontend::StructuredProgramCandidate &source, ArtifactStore &store) {
  auto view = take(source.view());
  sim::StructuredProgramSimulationWorkload workloadDraft{
      findStructuredCallable(source, "main")};
  workloadDraft.observableContract.returnValue = true;
  auto workload = take(sim::finalizeSimulationWorkload(workloadDraft, view));
  sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  auto runtime =
      take(sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  auto workloadReference =
      take(sim::publishSimulationWorkload(workload, store));
  auto runtimeInputReference =
      take(sim::publishSimulationRuntimeInput(runtime, store));
  return {std::move(workload), std::move(runtime), std::move(workloadReference),
          std::move(runtimeInputReference)};
}

} // namespace loom::test
