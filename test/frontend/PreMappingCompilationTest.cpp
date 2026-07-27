#include "Frontend/Compilation/PreMappingCompilation.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <system_error>

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

std::unique_ptr<llvm::Module> parseModule(const char *test,
                                          llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
define i32 @main(i32 %value) {
entry:
  %sum = add i32 %value, %value
  ret i32 %sum
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<pre-mapping>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module> parseSpatialModule(const char *test,
                                                 llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define internal void @kernel(ptr %a, ptr %b, ptr %c) {
entry:
  %lhs = load float, ptr %a, align 4
  %rhs = load float, ptr %b, align 4
  %sum = fadd float %lhs, %rhs
  store float %sum, ptr %c, align 4
  ret void
}

define i32 @main(ptr %a, ptr %b, ptr %c) {
entry:
  call void @kernel(ptr %a, ptr %b, ptr %c)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<spatial-owner>");
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
findCallable(const char *test,
             const loom::frontend::StructuredProgramCandidate &candidate,
             llvm::StringRef name) {
  auto view = take(test, candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getSymName() == name)
      return entity.reference;
  }
  fail(test, "raised Structured Program omitted the requested callable");
}

void exactFabricAndWholeProgramDataflow() {
  const char *test = "exactFabricAndWholeProgramDataflow";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-pre-mapping", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  if (design.roots().size() != 1)
    fail(test, "builtin target did not publish one System Fabric root");
  mlir::MLIRContext actorContext;
  mlir::Type i32 = mlir::IntegerType::get(&actorContext, 32);
  dataflow::CanonicalActorSchemaProjection add{
      dataflow::OperationSchemaId::ArithAddI,
      mlir::FunctionType::get(&actorContext, {i32, i32}, {i32}),
      dataflow::IntegerOverflowPayload{}};
  loom::frontend::FabricCapabilityIndex capabilities(
      design.roots().front().view());
  auto resources = capabilities.admittingOperationResources(add, 32);
  if (resources.empty())
    fail(test, "System Fabric hid its imported operation resources");
  for (const auto &resource : resources)
    if (resource.artifact == design.roots().front().view().identity())
      fail(test, "module-local operation resource was rebound to the System");
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseModule(test, context),
                                 design.roots().front().reference(), store));
  if (compiled.fabric != design.roots().front().reference())
    fail(test, "pre-Mapping result lost exact Fabric target identity");
  if (!compiled.canonicalDataflow.module().lookupSymbol("main"))
    fail(test, "whole-program Dataflow artifact lost LLVM callable envelope");
  auto view = take(test, compiled.canonicalDataflow.view());
  if (!view.graphs().empty())
    fail(test, "mechanical compilation invented a SpatialCore graph");
  auto published =
      take(test, loom::frontend::publishPreMappingCompilation(compiled, store));
  if (published.fabric != design.roots().front().reference())
    fail(test, "published compilation changed its exact Fabric binding");
  auto importedStructured = take(test, loom::frontend::importStructuredProgram(
                                           published.structuredProgram, store));
  auto importedDataflow = take(test, dataflow::importCanonicalDataflow(
                                         published.canonicalDataflow, store));
  if (importedStructured.identity() != compiled.structuredProgram.identity() ||
      importedDataflow.identity() != compiled.canonicalDataflow.identity())
    fail(test, "published artifacts did not round-trip through their owners");
  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void explicitWholeCallableSpatialOwnership() {
  const char *test = "explicitWholeCallableSpatialOwnership";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-spatial-owner", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseSpatialModule(test, context),
                                 design.roots().front().reference(), store));
  const loom::ArtifactIdentity parentIdentity =
      compiled.structuredProgram.identity();
  auto selected =
      take(test, loom::frontend::materializeWholeCallableSpatialOwnership(
                     compiled.structuredProgram,
                     findCallable(test, compiled.structuredProgram, "kernel"),
                     design.roots().front()));

  if (selected.structuredProgram.identity() == parentIdentity)
    fail(test, "ownership materialization did not create a child candidate");
  if (!selected.structuredProgram.module().lookupSymbol("kernel"))
    fail(test, "ownership materialization removed the LLVM ABI authority");

  bool sawLaunch = false;
  bool sawWait = false;
  selected.structuredProgram.module().walk([&](mlir::Operation *operation) {
    sawLaunch |= llvm::isa<dataflow::ThreadLaunchOp>(operation);
    sawWait |= llvm::isa<dataflow::ThreadWaitOp>(operation);
  });
  if (!sawLaunch || !sawWait)
    fail(test, "direct call was not replaced by ordered thread execution");

  auto view = take(test, selected.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().empty())
    fail(test, "selected callable did not publish one nonempty graph");
  std::set<dataflow::OperationSchemaId> schemas;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    auto projection =
        take(test, dataflow::projectRegisteredActorSchemaProjection(actor.op));
    schemas.insert(projection.schema);
  }
  for (dataflow::OperationSchemaId required :
       {dataflow::OperationSchemaId::DataflowLoad,
        dataflow::OperationSchemaId::ArithAddF,
        dataflow::OperationSchemaId::DataflowStore})
    if (schemas.find(required) == schemas.end())
      fail(test, "canonical graph omitted a load-add-store actor");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

} // namespace

int main() {
  exactFabricAndWholeProgramDataflow();
  explicitWholeCallableSpatialOwnership();
  llvm::outs() << "pre-Mapping compilation anchor passed\n";
  return EXIT_SUCCESS;
}
