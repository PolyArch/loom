#include "Frontend/Compilation/PreMappingCompilation.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/IndexWidth.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

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
  std::string unprovenMessage = llvm::toString(unproven.takeError());
  if (unprovenMessage.find("cannot prove") == std::string::npos)
    fail(test, "unproven narrowing did not report its proof boundary: " +
                   unprovenMessage);

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

} // namespace

int main() {
  exactFabricAndWholeProgramDataflow();
  explicitWholeCallableSpatialOwnership();
  explicitFmulAddExecutionShape();
  explicitOperationSpatialOwnership();
  operationOwnershipScopesFollowCanonicalOrder();
  operationSpatialOwnershipRejectsEscapedResult();
  llvm::outs() << "pre-Mapping compilation anchor passed\n";
  return EXIT_SUCCESS;
}
