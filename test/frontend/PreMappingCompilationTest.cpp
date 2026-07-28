#include "Frontend/Compilation/PreMappingCompilation.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/IndexWidth.h"
#include "Common/ResolvedConfig.h"
#include "DSE/PreMappingExploration.h"
#include "DSE/Promotion.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StaticMemoryBinding.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

define void @kernel(ptr %a, ptr %b, ptr %c) {
entry:
  %pa = getelementptr float, ptr %a, i32 0
  %lhs = load float, ptr %pa, align 4
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

std::unique_ptr<llvm::Module>
parseGlobalMemoryModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

@lookup = private constant i32 7

define void @global_lookup(ptr %output) {
entry:
  %value = load i32, ptr @lookup, align 4
  store i32 %value, ptr %output, align 4
  ret void
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<global-memory-owner>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module>
parseFmulAddSpatialModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define void @kernel(ptr %a, ptr %b, ptr %c) {
entry:
  %lhs = load float, ptr %a, align 4
  %rhs = load float, ptr %b, align 4
  %acc = load float, ptr %c, align 4
  %result = call float @llvm.fmuladd.f32(float %lhs, float %rhs, float %acc)
  store float %result, ptr %c, align 4
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float)
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<fmuladd-owner>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module>
parseOperationFmulAddModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define void @fmuladd_loop(ptr %a, ptr %b, ptr %c) {
entry:
  %outside_lhs = load float, ptr %a, align 4
  %outside_rhs = load float, ptr %b, align 4
  %outside_acc = load float, ptr %c, align 4
  %outside = call float @llvm.fmuladd.f32(
      float %outside_lhs, float %outside_rhs, float %outside_acc)
  store float %outside, ptr %c, align 4
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %pa = getelementptr float, ptr %a, i32 %i
  %pb = getelementptr float, ptr %b, i32 %i
  %pc = getelementptr float, ptr %c, i32 %i
  %lhs = load float, ptr %pa, align 4
  %rhs = load float, ptr %pb, align 4
  %acc = load float, ptr %pc, align 4
  %result = call float @llvm.fmuladd.f32(
      float %lhs, float %rhs, float %acc)
  store float %result, ptr %pc, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp ne i32 %next, 4
  br i1 %done, label %loop, label %exit

exit:
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float)
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<operation-fmuladd>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module>
parseWholeCallableLoopModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define void @kernel(ptr %a) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %a, i64 %i
  %value = load float, ptr %p, align 4
  store float %value, ptr %p, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp ne i64 %next, 64
  br i1 %done, label %loop, label %exit

exit:
  ret void
}

define void @unsigned_index(ptr %a) {
entry:
  br label %loop

loop:
  %index = phi i32 [ 0, %entry ], [ %next, %loop ]
  %wide = zext nneg i32 %index to i64
  %p = getelementptr float, ptr %a, i64 %wide
  %value = load float, ptr %p, align 4
  store float %value, ptr %p, align 4
  %next = add nuw nsw i32 %index, 1
  %done = icmp ne i32 %next, 64
  br i1 %done, label %loop, label %exit

exit:
  ret void
}

define void @unsigned_may_not_fit(ptr %a, i32 %index) {
entry:
  %wide = zext i32 %index to i64
  %p = getelementptr float, ptr %a, i64 %wide
  %value = load float, ptr %p, align 4
  store float %value, ptr %p, align 4
  ret void
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<whole-callable-loop>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module>
parseLoopOwnershipModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define i32 @kernel(ptr %a, ptr %b, i32 %n) {
entry:
  %init = load float, ptr %b, align 4
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %a, i64 %i
  store float %init, ptr %p, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp ne i64 %next, 64
  br i1 %done, label %loop, label %exit

exit:
  ret i32 %n
}

define i32 @dynamic(ptr %a, ptr %b, i64 %n, i32 %result) {
entry:
  %init = load float, ptr %b, align 4
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr float, ptr %a, i64 %i
  store float %init, ptr %p, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp ne i64 %next, %n
  br i1 %done, label %loop, label %exit

exit:
  ret i32 %result
}

)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<loop-owner>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module>
parseByteOffsetOwnershipModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define void @byte_offset(ptr %a, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %base = getelementptr inbounds float, ptr %a, i32 %i
  %p = getelementptr inbounds i8, ptr %base, i64 -4
  %value = load float, ptr %p, align 4
  store float %value, ptr %p, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp ult i32 %next, %n
  br i1 %done, label %loop, label %exit

exit:
  ret void
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<byte-offset-owner>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

std::unique_ptr<llvm::Module>
parseEscapedLoopModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define i32 @accum(ptr %a, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %newsum, %loop ]
  %p = getelementptr i32, ptr %a, i32 %i
  %v = load i32, ptr %p, align 4
  %newsum = add i32 %sum, %v
  %next = add nuw i32 %i, 1
  %done = icmp ult i32 %next, %n
  br i1 %done, label %loop, label %exit

exit:
  ret i32 %newsum
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<escaped-loop>");
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

loom::frontend::StructuredEntityRef
findStructuredLoop(const char *test,
                   const loom::frontend::StructuredProgramCandidate &candidate,
                   llvm::StringRef callableName) {
  auto view = take(test, candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(entity.operation);
    if (!loop)
      continue;
    auto callable = loop->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (callable && callable.getSymName() == callableName)
      return entity.reference;
  }
  fail(test, "raised Structured Program omitted the structured loop");
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
  const std::uint64_t systemAddCount =
      take(test, capabilities.admittingOperationResourceCount(add, 32));
  for (const auto &resource : resources)
    if (resource.artifact == design.roots().front().view().identity())
      fail(test, "module-local operation resource was rebound to the System");
  auto importedModule =
      take(test, loom::fabric::importEntireFabricRoot(
                     design.roots().front().directDependencies().front().root,
                     store));
  loom::frontend::FabricCapabilityIndex moduleCapabilities(
      importedModule.view());
  const std::uint64_t moduleAddCount =
      take(test, moduleCapabilities.admittingOperationResourceCount(add, 32));
  if (moduleAddCount == 0 || systemAddCount != moduleAddCount * 4)
    fail(test,
         "System Fabric did not expand each module-local operation resource "
         "through its four SpatialCore occurrences");
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
  const loom::frontend::StructuredEntityRef callable =
      findCallable(test, compiled.structuredProgram, "kernel");
  auto decisionDomain =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, callable));
  if (decisionDomain.size() != 1 ||
      decisionDomain.front().canonicalIndexWidth ||
      decisionDomain.front().fmuladdExecutionShape)
    fail(test, "constant GEP invented a dynamic ownership decision");
  auto selected = take(
      test, loom::frontend::materializeWholeCallableSpatialOwnership(
                compiled.structuredProgram, callable, design.roots().front()));

  if (selected.structuredProgram.identity() == parentIdentity)
    fail(test, "ownership materialization did not create a child candidate");
  auto wrapper =
      selected.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "kernel");
  if (!wrapper || wrapper.getLinkage() != mlir::LLVM::Linkage::External ||
      wrapper.getFunctionType().getNumParams() != 3)
    fail(test, "ownership materialization removed the LLVM ABI authority");

  bool sawLaunch = false;
  bool sawWait = false;
  bool sawOriginalCompute = false;
  wrapper.walk([&](mlir::Operation *operation) {
    sawLaunch |= llvm::isa<dataflow::ThreadLaunchOp>(operation);
    sawWait |= llvm::isa<dataflow::ThreadWaitOp>(operation);
    sawOriginalCompute |=
        llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp>(operation) ||
        operation->getName().getStringRef() == "arith.addf";
  });
  if (!sawLaunch || !sawWait)
    fail(test,
         "ABI callable body was not replaced by ordered thread execution");
  if (sawOriginalCompute)
    fail(test, "ABI callable retained a competing InstructionCore body");

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

void wholeCallableExternalizesGlobalMemoryCapability() {
  const char *test = "wholeCallableExternalizesGlobalMemoryCapability";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-global-memory-owner", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseGlobalMemoryModule(test, context),
                                 design.roots().front().reference(), store));
  auto selected = take(
      test, loom::frontend::materializeWholeCallableSpatialOwnership(
                compiled.structuredProgram,
                findCallable(test, compiled.structuredProgram, "global_lookup"),
                design.roots().front()));

  auto wrapper =
      selected.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "global_lookup");
  unsigned addressCount = 0;
  wrapper.walk([&](mlir::LLVM::AddressOfOp) { ++addressCount; });
  if (addressCount != 1)
    fail(test, "stored-program wrapper does not own the global address");

  auto view = take(test, selected.canonicalDataflow.view());
  if (view.graphs().size() != 1)
    fail(test, "global-memory ownership did not publish exactly one graph");
  auto graph = llvm::cast<dataflow::GraphOp>(view.graphs().front().op);
  if (graph.getInputSegmentSizes()[2] != 2)
    fail(test, "global memory did not cross the graph memory-capability ABI");
  bool sawAddress = false;
  for (const dataflow::CanonicalActorView &actor : view.actors())
    sawAddress |= llvm::isa<mlir::LLVM::AddressOfOp>(actor.op);
  if (sawAddress)
    fail(test, "global address escaped into the SpatialCore actor graph");

  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail(test, "global-memory candidate has no unique rooted launch");
  dataflow::RootedGraphLaunchRef launch{view.rootThreadLaunches().front().ref,
                                        view.staticGraphLaunches().front().ref};
  auto sources = take(test, loom::frontend::deriveRootedLogicalMemorySources(
                                compiled.staticGlobalMemory, view, launch));
  unsigned staticSources = 0;
  for (const loom::frontend::RootedLogicalMemorySource &source : sources) {
    if (!source.globalOrdinal)
      continue;
    ++staticSources;
    if (*source.globalOrdinal >= compiled.staticGlobalMemory.globals.size() ||
        compiled.staticGlobalMemory.globals[*source.globalOrdinal].symbol !=
            "lookup")
      fail(test, "logical memory root resolved the wrong static global");
  }
  if (staticSources != 1)
    fail(test, "static global did not bind exactly one logical memory root");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void explicitFmulAddExecutionShape() {
  const char *test = "explicitFmulAddExecutionShape";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-fmuladd-owner", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseFmulAddSpatialModule(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::StructuredEntityRef callable =
      findCallable(test, compiled.structuredProgram, "kernel");

  loom::frontend::WholeCallableSpatialOwnershipOptions options;
  options.fmuladdExecutionShape = loom::raising::FMulAddExecutionShape::Fused;
  auto selected =
      take(test, loom::frontend::materializeWholeCallableSpatialOwnership(
                     compiled.structuredProgram, callable,
                     design.roots().front(), options));
  auto view = take(test, selected.canonicalDataflow.view());
  bool sawFma = false;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    auto projection =
        take(test, dataflow::projectRegisteredActorSchemaProjection(actor.op));
    sawFma |= projection.schema == dataflow::OperationSchemaId::MathFma;
  }
  if (!sawFma)
    fail(test, "selected Fused shape did not publish math.fma");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void wholeCallableRequiresCanonicalAddressIndexDecision() {
  const char *test = "wholeCallableRequiresCanonicalAddressIndexDecision";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-whole-callable-index", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseWholeCallableLoopModule(test, context),
                                 design.roots().front().reference(), store));
  auto candidate = loom::frontend::materializeWholeCallableSpatialOwnership(
      compiled.structuredProgram,
      findCallable(test, compiled.structuredProgram, "kernel"),
      design.roots().front());
  if (candidate)
    fail(test, "whole-callable ownership silently selected an index width");
  std::string message = llvm::toString(candidate.takeError());
  if (message.find("explicit canonical index width") == std::string::npos)
    fail(test, "missing index decision was not diagnosed: " + message);

  loom::frontend::WholeCallableSpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto selected =
      take(test, loom::frontend::materializeWholeCallableSpatialOwnership(
                     compiled.structuredProgram,
                     findCallable(test, compiled.structuredProgram, "kernel"),
                     design.roots().front(), options));
  unsigned indexWidth =
      take(test, loom::getIndexBitWidth(selected.structuredProgram.module()));
  if (indexWidth != 32)
    fail(test, "whole-callable index decision was not materialized");
  auto view = take(test, selected.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().empty())
    fail(test, "whole-callable index decision did not publish its graph");

  auto unsignedIndex =
      take(test,
           loom::frontend::materializeWholeCallableSpatialOwnership(
               compiled.structuredProgram,
               findCallable(test, compiled.structuredProgram, "unsigned_index"),
               design.roots().front(), options));
  auto unsignedView = take(test, unsignedIndex.canonicalDataflow.view());
  if (unsignedView.graphs().size() != 1 || unsignedView.actors().empty())
    fail(test, "proven nonnegative extended index did not publish its graph");

  auto unprovenUnsigned =
      loom::frontend::materializeWholeCallableSpatialOwnership(
          compiled.structuredProgram,
          findCallable(test, compiled.structuredProgram,
                       "unsigned_may_not_fit"),
          design.roots().front(), options);
  if (unprovenUnsigned)
    fail(test, "unproven unsigned index narrowing was accepted");
  message = llvm::toString(unprovenUnsigned.takeError());
  if (message.find("cannot prove a wide GEP index") == std::string::npos)
    fail(test, "unproven unsigned index was misdiagnosed: " + message);

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void explicitOperationSpatialOwnership() {
  const char *test = "explicitOperationSpatialOwnership";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-operation-owner", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseLoopOwnershipModule(test, context),
                                 design.roots().front().reference(), store));
  const loom::ArtifactIdentity parentIdentity =
      compiled.structuredProgram.identity();
  loom::frontend::StructuredEntityRef loop =
      findStructuredLoop(test, compiled.structuredProgram, "kernel");
  auto implicit = loom::frontend::materializeOperationSpatialOwnership(
      compiled.structuredProgram, loop, design.roots().front());
  if (implicit)
    fail(test, "operation ownership silently selected an index width");
  std::string implicitMessage = llvm::toString(implicit.takeError());
  if (implicitMessage.find("explicit canonical index width") ==
      std::string::npos)
    fail(test, "missing index decision was not diagnosed: " + implicitMessage);

  loom::frontend::OperationSpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto selected =
      take(test, loom::frontend::materializeOperationSpatialOwnership(
                     compiled.structuredProgram, loop, design.roots().front(),
                     options));

  if (selected.structuredProgram.identity() == parentIdentity)
    fail(test, "operation ownership did not create a child candidate");
  unsigned indexWidth =
      take(test, loom::getIndexBitWidth(
                     selected.structuredProgram.module().getOperation()));
  if (indexWidth != 32)
    fail(test, "selected index width was not materialized in Structured IR");

  auto wrapper =
      selected.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "kernel");
  if (!wrapper || wrapper.getLinkage() != mlir::LLVM::Linkage::External)
    fail(test, "operation ownership removed the LLVM ABI authority");
  auto abiReturn = llvm::dyn_cast<mlir::IntegerType>(
      wrapper.getFunctionType().getReturnType());
  if (wrapper.getFunctionType().getNumParams() != 3 || !abiReturn ||
      abiReturn.getWidth() != 32)
    fail(test, "operation ownership changed the LLVM callable ABI");

  mlir::Block &body = wrapper.getBody().front();
  bool sawLoop = false;
  int loadOrder = -1;
  int launchOrder = -1;
  int waitOrder = -1;
  int returnOrder = -1;
  int order = 0;
  dataflow::ThreadLaunchOp launch;
  mlir::LLVM::ReturnOp returnOp;
  for (mlir::Operation &operation : body) {
    sawLoop |= llvm::isa<mlir::scf::WhileOp>(&operation);
    if (llvm::isa<mlir::LLVM::LoadOp>(&operation) && loadOrder < 0)
      loadOrder = order;
    if (auto candidate = llvm::dyn_cast<dataflow::ThreadLaunchOp>(&operation)) {
      if (!launch)
        launch = candidate;
      if (launchOrder < 0)
        launchOrder = order;
    }
    if (llvm::isa<dataflow::ThreadWaitOp>(&operation) && waitOrder < 0)
      waitOrder = order;
    if (auto candidate = llvm::dyn_cast<mlir::LLVM::ReturnOp>(&operation)) {
      returnOp = candidate;
      returnOrder = order;
    }
    ++order;
  }
  if (sawLoop)
    fail(test, "ABI callable retained the selected structured loop");
  if (loadOrder < 0)
    fail(test, "ABI callable lost the surrounding pre-loop load");
  if (!launch || launchOrder < loadOrder || waitOrder < launchOrder ||
      returnOrder < waitOrder)
    fail(test, "launch/wait did not replace the loop at its exact position");
  if (!returnOp || returnOp->getNumOperands() != 1)
    fail(test, "ABI callable lost its surrounding non-void return");
  auto returned = llvm::dyn_cast<mlir::BlockArgument>(returnOp->getOperand(0));
  if (!returned || returned.getOwner() != &body || returned.getArgNumber() != 2)
    fail(test, "surrounding return no longer forwards the ABI i32 argument");

  unsigned threadCount = 0;
  dataflow::ThreadOp thread;
  selected.structuredProgram.module().walk([&](dataflow::ThreadOp candidate) {
    ++threadCount;
    thread = candidate;
  });
  if (threadCount != 1 || !thread)
    fail(test, "operation ownership did not create exactly one thread");
  if (thread.getSymVisibility() != "private")
    fail(test, "operation thread escaped the private visibility policy");
  if (launch.getCallee() != thread.getSymName())
    fail(test, "launch does not resolve to the created thread");
  if (!thread.getFunctionType().getResults().empty())
    fail(test, "operation thread invented SSA data results");
  if (launch.getBodyOperands().size() !=
      thread.getFunctionType().getNumInputs())
    fail(test, "launch operands diverge from the explicit live-in inputs");
  mlir::Block &threadEntry = thread.getBody().front();
  if (threadEntry.getNumArguments() !=
          thread.getFunctionType().getNumInputs() + 1 ||
      !llvm::isa<mlir::NoneType>(threadEntry.getArguments().back().getType()))
    fail(test, "rank-zero thread does not carry exactly one ctrl argument");

  auto view = take(test, selected.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().empty())
    fail(test, "selected operation did not publish one nonempty graph");
  bool sawStore = false;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    auto projection =
        take(test, dataflow::projectRegisteredActorSchemaProjection(actor.op));
    sawStore |= projection.schema == dataflow::OperationSchemaId::DataflowStore;
  }
  if (!sawStore)
    fail(test, "canonical graph dropped the loop's side-effecting store");

  auto unproven = loom::frontend::materializeOperationSpatialOwnership(
      compiled.structuredProgram,
      findStructuredLoop(test, compiled.structuredProgram, "dynamic"),
      design.roots().front(), options);
  if (unproven)
    fail(test, "runtime-bounded wide GEP index was narrowed without proof");
  std::string unprovenMessage;
  bool classifiedNonFinalizable = false;
  llvm::Error unhandled = llvm::handleErrors(
      unproven.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &rejection) {
        classifiedNonFinalizable =
            rejection.kind() ==
            loom::frontend::SpatialOwnershipCandidateRejectionKind::
                NonFinalizable;
        unprovenMessage = rejection.message();
      });
  if (unhandled)
    fail(test, "unproven narrowing returned a non-candidate error: " +
                   llvm::toString(std::move(unhandled)));
  if (!classifiedNonFinalizable)
    fail(test, "unproven narrowing was not a typed non-finalizable candidate");
  if (unprovenMessage.find("cannot prove") == std::string::npos)
    fail(test, "unproven narrowing did not report its proof boundary: " +
                   unprovenMessage);

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void operationOwnershipInternalizesConstants() {
  const char *test = "operationOwnershipInternalizesConstants";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-operation-constants", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseByteOffsetOwnershipModule(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::OperationSpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto selected = take(
      test,
      loom::frontend::materializeOperationSpatialOwnership(
          compiled.structuredProgram,
          findStructuredLoop(test, compiled.structuredProgram, "byte_offset"),
          design.roots().front(), options));

  auto wrapper =
      selected.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "byte_offset");
  if (!wrapper)
    fail(test, "operation ownership removed the LLVM callable envelope");
  dataflow::ThreadLaunchOp launch;
  wrapper.getBody().walk([&](dataflow::ThreadLaunchOp candidate) {
    if (launch)
      fail(test, "operation ownership emitted multiple launches");
    launch = candidate;
  });
  if (!launch || launch.getBodyOperands().size() != 2)
    fail(test, "compile-time constants escaped through the launch ABI");

  dataflow::ThreadOp thread;
  selected.structuredProgram.module().walk([&](dataflow::ThreadOp candidate) {
    if (thread)
      fail(test, "operation ownership emitted multiple threads");
    thread = candidate;
  });
  if (!thread || thread.getFunctionType().getNumInputs() != 2)
    fail(test, "compile-time constants escaped through the thread ABI");

  auto view = take(test, selected.canonicalDataflow.view());
  bool sawConstant = false;
  bool sawLoad = false;
  bool sawStore = false;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    auto projection =
        take(test, dataflow::projectRegisteredActorSchemaProjection(actor.op));
    sawConstant |=
        projection.schema == dataflow::OperationSchemaId::DataflowConstant;
    sawLoad |= projection.schema == dataflow::OperationSchemaId::DataflowLoad;
    sawStore |= projection.schema == dataflow::OperationSchemaId::DataflowStore;
  }
  if (!sawConstant || !sawLoad || !sawStore)
    fail(test, "canonical graph lost its constant or memory actors");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void operationOwnershipScopesFollowCanonicalOrder() {
  const char *test = "operationOwnershipScopesFollowCanonicalOrder";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-operation-scopes", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseLoopOwnershipModule(test, context),
                                 design.roots().front().reference(), store));
  auto scopes =
      take(test, loom::frontend::enumerateOperationSpatialOwnershipScopes(
                     compiled.structuredProgram));
  auto view = take(test, compiled.structuredProgram.view());

  std::vector<loom::frontend::StructuredEntityRef> expected;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    if (llvm::isa_and_nonnull<mlir::scf::WhileOp>(entity.operation))
      expected.push_back(entity.reference);
  }
  if (expected.size() != 2 || scopes != expected)
    fail(test, "ownership scopes do not follow canonical operation order");
  for (const auto &scope : scopes)
    if (scope.parent != compiled.structuredProgram.identity() ||
        scope.kind != loom::frontend::StructuredEntityKind::Operation)
      fail(test, "ownership scope is not parent-local operation identity");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void wholeCallableScopesFollowCanonicalOrder() {
  const char *test = "wholeCallableScopesFollowCanonicalOrder";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-whole-callable-scopes", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseSpatialModule(test, context),
                                 design.roots().front().reference(), store));
  auto scopes =
      take(test, loom::frontend::enumerateWholeCallableSpatialOwnershipScopes(
                     compiled.structuredProgram));
  if (scopes.size() != 1 ||
      scopes.front() !=
          findCallable(test, compiled.structuredProgram, "kernel"))
    fail(test, "whole-callable domain admitted a declaration, non-void "
               "wrapper, or omitted the eligible kernel");
  if (scopes.front().parent != compiled.structuredProgram.identity() ||
      scopes.front().kind != loom::frontend::StructuredEntityKind::Operation)
    fail(test, "whole-callable scope is not parent-local operation identity");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void operationFmulAddDecisionIsCandidateLocal() {
  const char *test = "operationFmulAddDecisionIsCandidateLocal";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-operation-fmuladd", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseOperationFmulAddModule(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::OperationSpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  options.fmuladdExecutionShape = loom::raising::FMulAddExecutionShape::Fused;
  auto selected = take(
      test,
      loom::frontend::materializeOperationSpatialOwnership(
          compiled.structuredProgram,
          findStructuredLoop(test, compiled.structuredProgram, "fmuladd_loop"),
          design.roots().front(), options));

  auto graph = take(test, selected.canonicalDataflow.view());
  bool sawFma = false;
  for (const dataflow::CanonicalActorView &actor : graph.actors()) {
    auto projection =
        take(test, dataflow::projectRegisteredActorSchemaProjection(actor.op));
    sawFma |= projection.schema == dataflow::OperationSchemaId::MathFma;
  }
  if (!sawFma)
    fail(test, "selected operation did not publish its Fused execution shape");

  auto wrapper =
      selected.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "fmuladd_loop");
  bool retainedUnselectedChoice = false;
  wrapper.walk([&](mlir::LLVM::FMulAddOp) {
    retainedUnselectedChoice = true;
    return mlir::WalkResult::interrupt();
  });
  if (!retainedUnselectedChoice)
    fail(test, "operation decision rewrote an unselected fmuladd");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void ownershipDecisionDomainIsScopeLocalAndTyped() {
  const char *test = "ownershipDecisionDomainIsScopeLocalAndTyped";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-ownership-domain", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseOperationFmulAddModule(test, context),
                                 design.roots().front().reference(), store));
  const loom::frontend::StructuredEntityRef scope =
      findStructuredLoop(test, compiled.structuredProgram, "fmuladd_loop");
  auto domain =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, scope));

  using Shape = loom::raising::FMulAddExecutionShape;
  const std::vector<loom::frontend::SpatialOwnershipDecisionPoint> expected = {
      {Shape::Fused, 32},
      {Shape::Split, 32},
      {Shape::Fused, 64},
      {Shape::Split, 64},
  };
  if (domain != expected)
    fail(test, "scope-local decision domain is incomplete or noncanonical");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void unifiedOwnershipDomainMaterializesExplicitDecision() {
  const char *test = "unifiedOwnershipDomainMaterializesExplicitDecision";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-unified-ownership-domain", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseOperationFmulAddModule(test, context),
                                 design.roots().front().reference(), store));
  auto scopes = take(test, loom::frontend::enumerateSpatialOwnershipScopes(
                               compiled.structuredProgram));
  const loom::frontend::StructuredEntityRef callable =
      findCallable(test, compiled.structuredProgram, "fmuladd_loop");
  const loom::frontend::StructuredEntityRef loop =
      findStructuredLoop(test, compiled.structuredProgram, "fmuladd_loop");

  std::optional<loom::frontend::SpatialOwnershipScope> callableScope;
  std::optional<loom::frontend::SpatialOwnershipScope> operationScope;
  for (const loom::frontend::SpatialOwnershipScope &scope : scopes) {
    if (scope.selection == callable)
      callableScope = scope;
    if (scope.selection == loop)
      operationScope = scope;
  }
  if (!callableScope ||
      callableScope->kind !=
          loom::frontend::SpatialOwnershipScopeKind::WholeCallable)
    fail(test, "unified domain omitted the whole-callable ownership scope");
  if (!operationScope ||
      operationScope->kind !=
          loom::frontend::SpatialOwnershipScopeKind::Operation)
    fail(test, "unified domain omitted the operation ownership scope");

  loom::frontend::SpatialOwnershipDecisionPoint decision{
      loom::raising::FMulAddExecutionShape::Fused, 32};
  auto selected =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, *operationScope, decision,
                     design.roots().front()));
  auto view = take(test, selected.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().empty())
    fail(test, "unified materialization did not produce Spatial workload");
  bool sawFma = false;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    auto projection =
        take(test, dataflow::projectRegisteredActorSchemaProjection(actor.op));
    sawFma |= projection.schema == dataflow::OperationSchemaId::MathFma;
  }
  if (!sawFma)
    fail(test, "unified materialization lost the explicit execution shape");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void operationSpatialOwnershipRejectsEscapedResult() {
  const char *test = "operationSpatialOwnershipRejectsEscapedResult";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-operation-reject", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseEscapedLoopModule(test, context),
                                 design.roots().front().reference(), store));
  auto selected = loom::frontend::materializeOperationSpatialOwnership(
      compiled.structuredProgram,
      findStructuredLoop(test, compiled.structuredProgram, "accum"),
      design.roots().front());
  if (selected)
    fail(test, "operation with an externally used result was materialized");
  std::string message = llvm::toString(selected.takeError());
  if (message.find("used outside") == std::string::npos)
    fail(test, "rejection did not name the escaped SSA result: " + message);

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

loom::evaluation::DecimalValue
runtimeResult(const char *test,
              const loom::evaluation::EvaluationEvidence &evidence) {
  const auto *completed =
      std::get_if<loom::evaluation::CompletedEvidence>(&evidence.outcome());
  if (!completed || completed->metricResults.size() != 1)
    fail(test, "analytic model did not return one completed Runtime result");
  const loom::evaluation::MetricResult &result =
      completed->metricResults.front();
  if (result.uncertainty != loom::evaluation::UncertaintyKind::Unknown)
    fail(test, "analytic model presented its estimate as ground truth");
  const auto *point =
      std::get_if<loom::evaluation::PointObservation>(&result.observation);
  if (!point)
    fail(test, "analytic model did not return a point estimate");
  const auto *runtime =
      std::get_if<loom::evaluation::DecimalValue>(&point->value);
  if (!runtime)
    fail(test, "analytic Runtime result used the wrong numeric domain");
  return *runtime;
}

struct EvaluatedRuntime final {
  loom::evaluation::DecimalValue value;
  loom::evaluation::EvaluationRequest request;
  loom::evaluation::EvaluationEvidence evidence;
};

EvaluatedRuntime
evaluateStructuredRuntime(const char *test,
                          const loom::ArtifactRootReference &structuredProgram,
                          const loom::ArtifactRootReference &fabric,
                          const loom::ArtifactStore &store) {
  auto prepared = take(
      test,
      loom::evaluation::models::prepareStructuredFabricRuntimeEvaluation(
          structuredProgram, fabric, loom::defaultResolvedConfig(), store));
  auto evidence = take(test, loom::evaluation::evaluateRequest(
                                 prepared.request, prepared.resolution, store));
  return EvaluatedRuntime{runtimeResult(test, evidence),
                          std::move(prepared.request), std::move(evidence)};
}

loom::evaluation::DecimalValue
evaluateCanonicalDataflowRuntime(const char *test,
                                 const loom::ArtifactRootReference &program,
                                 const loom::ArtifactRootReference &fabric,
                                 const loom::ArtifactStore &store) {
  auto prepared = take(
      test,
      loom::evaluation::models::prepareCanonicalDataflowFabricRuntimeEvaluation(
          program, fabric, loom::defaultResolvedConfig(), store));
  auto evidence = take(test, loom::evaluation::evaluateRequest(
                                 prepared.request, prepared.resolution, store));
  return runtimeResult(test, evidence);
}

void structuredFabricEvaluationRanksMaterializedOwnership() {
  const char *test = "structuredFabricEvaluationRanksMaterializedOwnership";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-fabric-evaluation", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseSpatialModule(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::WholeCallableSpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto spatial =
      take(test, loom::frontend::materializeWholeCallableSpatialOwnership(
                     compiled.structuredProgram,
                     findCallable(test, compiled.structuredProgram, "kernel"),
                     design.roots().front(), options));

  const loom::ArtifactRootReference baselineRef =
      take(test, loom::frontend::publishStructuredProgram(
                     compiled.structuredProgram, store));
  const loom::ArtifactRootReference spatialRef =
      take(test, loom::frontend::publishStructuredProgram(
                     spatial.structuredProgram, store));
  const loom::ArtifactRootReference dataflowRef =
      take(test, dataflow::publishCanonicalDataflow(spatial.canonicalDataflow,
                                                    store));
  EvaluatedRuntime baseline = evaluateStructuredRuntime(
      test, baselineRef, design.roots().front().reference(), store);
  EvaluatedRuntime spatialEvaluation = evaluateStructuredRuntime(
      test, spatialRef, design.roots().front().reference(), store);
  if (loom::evaluation::compareDecimalValue(spatialEvaluation.value,
                                            baseline.value) >= 0)
    fail(test, "Fabric-aware Evaluation did not prefer materialized Spatial "
               "ownership");

  auto candidates =
      take(test, loom::dse::CandidateSet::get(
                     loom::frontend::structuredProgramArtifactSchema,
                     {baselineRef, spatialRef}));
  const auto incomplete =
      take(test, loom::dse::promoteMetricTopK(
                     candidates, loom::evaluation::CaseSubjectRoleRef(0),
                     {{baseline.request, baseline.evidence}},
                     {loom::evaluation::MetricRequestOrdinal(0),
                      loom::dse::ObjectiveDirection::Minimize, 1},
                     store));
  const auto *missing =
      std::get_if<loom::dse::IncompleteSelection>(&incomplete);
  if (!missing ||
      missing->reason != loom::dse::IncompleteSelectionReason::MissingEvidence)
    fail(test, "central DSE treated missing Evidence as a ranking value");

  std::vector<loom::dse::PromotionEvidence> evidence;
  evidence.push_back({std::move(spatialEvaluation.request),
                      std::move(spatialEvaluation.evidence)});
  evidence.push_back(
      {std::move(baseline.request), std::move(baseline.evidence)});
  auto promoted = take(
      test, loom::dse::promoteMetricTopK(
                candidates, loom::evaluation::CaseSubjectRoleRef(0), evidence,
                {loom::evaluation::MetricRequestOrdinal(0),
                 loom::dse::ObjectiveDirection::Minimize, 1},
                store));
  const auto *selection = std::get_if<loom::dse::CompletedSelection>(&promoted);
  if (!selection || selection->selected.size() != 1 ||
      selection->selected.front() != spatialRef)
    fail(test, "central DSE TopK did not promote the best exact candidate");

  loom::dse::PreMappingExplorationOptions exploration{
      {},
      {{},
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::dse::ObjectiveDirection::Minimize, 1}}};
  auto explored =
      take(test, loom::dse::exploreLlvmModuleToPreMapping(
                     parseSpatialModule(test, context), design.roots().front(),
                     loom::defaultResolvedConfig(), exploration, store));
  const auto *exploredSelection =
      std::get_if<loom::dse::CompletedPreMappingSelection>(&explored);
  if (!exploredSelection || exploredSelection->selected.size() != 1)
    fail(test, "central ownership exploration did not select one survivor");
  auto exploredView =
      take(test, exploredSelection->selected.front().canonicalDataflow.view());
  if (exploredView.actors().empty())
    fail(test, "central ownership exploration selected no Spatial workload");
  const loom::evaluation::DecimalValue dataflowRuntime =
      evaluateCanonicalDataflowRuntime(
          test, dataflowRef, design.roots().front().reference(), store);
  if (dataflowRuntime.coefficient() <= 0)
    fail(test, "Dataflow/Fabric Evaluation returned no spatial work");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

} // namespace

int main() {
  if (llvm::Error error =
          loom::evaluation::models::registerStructuredFabricAnalyticModel())
    fail("registration", llvm::toString(std::move(error)));
  if (llvm::Error error = loom::evaluation::models::
          registerCanonicalDataflowFabricAnalyticModel())
    fail("registration", llvm::toString(std::move(error)));
  exactFabricAndWholeProgramDataflow();
  explicitWholeCallableSpatialOwnership();
  wholeCallableExternalizesGlobalMemoryCapability();
  explicitFmulAddExecutionShape();
  wholeCallableRequiresCanonicalAddressIndexDecision();
  explicitOperationSpatialOwnership();
  operationOwnershipInternalizesConstants();
  wholeCallableScopesFollowCanonicalOrder();
  operationOwnershipScopesFollowCanonicalOrder();
  operationFmulAddDecisionIsCandidateLocal();
  ownershipDecisionDomainIsScopeLocalAndTyped();
  unifiedOwnershipDomainMaterializesExplicitDecision();
  operationSpatialOwnershipRejectsEscapedResult();
  structuredFabricEvaluationRanksMaterializedOwnership();
  llvm::outs() << "pre-Mapping compilation anchor passed\n";
  return EXIT_SUCCESS;
}
