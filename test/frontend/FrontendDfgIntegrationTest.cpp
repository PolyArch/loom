#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/PreMappingCompilation.h"
#include "Frontend/Compilation/StaticMemoryBinding.h"
#include "Frontend/Raising/StructuredRaising.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationInputCapture.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
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
                                          llvm::LLVMContext &context,
                                          bool riscvTarget = true) {
  constexpr llvm::StringLiteral source = R"llvm(
define void @vecadd(ptr %a, ptr %b, ptr %c, float %bias) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %pa = getelementptr float, ptr %a, i64 %i
  %pb = getelementptr float, ptr %b, i64 %i
  %pc = getelementptr float, ptr %c, i64 %i
  %va = load float, ptr %pa, align 4
  %vb = load float, ptr %pb, align 4
  %partial = fadd float %va, %vb
  %sum = fadd float %partial, %bias
  store float %sum, ptr %pc, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define i32 @main() {
entry:
  %a = alloca [64 x float], align 4
  %b = alloca [64 x float], align 4
  %c = alloca [64 x float], align 4
  %bias.slot = alloca float, align 4
  store float 2.500000e-01, ptr %bias.slot, align 4
  br label %init

init:
  %j = phi i64 [ 0, %entry ], [ %jnext, %init ]
  %fa = uitofp i64 %j to float
  %fb = fmul float %fa, 5.000000e-01
  %pa.init = getelementptr [64 x float], ptr %a, i64 0, i64 %j
  %pb.init = getelementptr [64 x float], ptr %b, i64 0, i64 %j
  %pc.init = getelementptr [64 x float], ptr %c, i64 0, i64 %j
  store float %fa, ptr %pa.init, align 4
  store float %fb, ptr %pb.init, align 4
  store float 0.000000e+00, ptr %pc.init, align 4
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, 64
  br i1 %jdone, label %invoke, label %init

invoke:
  %bias = load float, ptr %bias.slot, align 4
  call void @vecadd(ptr %a, ptr %b, ptr %c, float %bias)
  ret i32 0
}

define i32 @slice_main() {
entry:
  %ab = alloca [128 x float], align 4
  %b = getelementptr [128 x float], ptr %ab, i64 0, i64 64
  %c = alloca [64 x float], align 4
  call void @vecadd(ptr %ab, ptr %b, ptr %c, float 1.000000e+00)
  ret i32 0
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
  if (riscvTarget) {
    module->setDataLayout("e-m:e-p:64:64-i64:64-n32:64-S128");
    module->setTargetTriple(llvm::Triple("riscv64-unknown-unknown-elf"));
  }
  return module;
}

std::unique_ptr<llvm::Module> parseTableLookup(const char *test,
                                               llvm::LLVMContext &context,
                                               bool riscvTarget = true) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

@lookup = private constant [4 x i32]
    [i32 287454020, i32 1432778632, i32 -1, i32 7], align 16

define void @table_lookup(ptr %output) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %source = getelementptr [4 x i32], ptr @lookup, i64 0, i64 %i
  %value = load i32, ptr %source, align 4
  %destination = getelementptr i32, ptr %output, i64 %i
  store i32 %value, ptr %destination, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @table_lookup_arg(ptr %table, ptr %output) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %source = getelementptr i32, ptr %table, i64 %i
  %value = load i32, ptr %source, align 4
  %destination = getelementptr i32, ptr %output, i64 %i
  store i32 %value, ptr %destination, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @table_lookup_wrapper(ptr %table, ptr %output) {
entry:
  call void @table_lookup_arg(ptr %table, ptr %output)
  ret void
}

define i32 @main() {
entry:
  %output = alloca [4 x i32], align 16
  call void @table_lookup_arg(ptr @lookup, ptr %output)
  ret i32 0
}

define i32 @direct_main() {
entry:
  %output = alloca [4 x i32], align 16
  call void @table_lookup(ptr %output)
  ret i32 0
}

define i32 @nested_main() {
entry:
  %first = alloca [4 x i32], align 16
  %second = alloca [4 x i32], align 16
  call void @table_lookup_wrapper(ptr @lookup, ptr %first)
  call void @table_lookup_wrapper(ptr @lookup, ptr %second)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<table-lookup>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  if (!riscvTarget) {
    module->setDataLayout("");
    module->setTargetTriple(llvm::Triple());
  }
  return module;
}

std::unique_ptr<llvm::Module> parseFmuladd(const char *test,
                                           llvm::LLVMContext &context,
                                           bool riscvTarget = true) {
  constexpr llvm::StringLiteral source = R"llvm(
declare float @llvm.fmuladd.f32(float, float, float)

define void @fma_kernel(ptr %a, ptr %b, ptr %c, ptr %output) {
entry:
  %av = load float, ptr %a, align 4
  %bv = load float, ptr %b, align 4
  %cv = load float, ptr %c, align 4
  %result = call float @llvm.fmuladd.f32(float %av, float %bv, float %cv)
  store float %result, ptr %output, align 4
  ret void
}

define i32 @main() {
entry:
  %a = alloca float, align 4
  %b = alloca float, align 4
  %c = alloca float, align 4
  %output = alloca float, align 4
  %one_plus_epsilon = bitcast i32 1065353217 to float
  %negative_rounded_product = bitcast i32 -1082130430 to float
  store float %one_plus_epsilon, ptr %a, align 4
  store float %one_plus_epsilon, ptr %b, align 4
  store float %negative_rounded_product, ptr %c, align 4
  store float 0.000000e+00, ptr %output, align 4
  call void @fma_kernel(ptr %a, ptr %b, ptr %c, ptr %output)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<fmuladd>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  if (riscvTarget) {
    module->setDataLayout("e-m:e-p:64:64-i64:64-n32:64-S128");
    module->setTargetTriple(llvm::Triple("riscv64-unknown-unknown-elf"));
  }
  return module;
}

std::unique_ptr<llvm::Module> parseScalarReduction(const char *test,
                                                   llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define i32 @accum() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %newsum, %loop ]
  %newsum = add i32 %sum, %i
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %newsum
}

define i32 @main() {
entry:
  %result = call i32 @accum()
  %wrong = icmp ne i32 %result, 28
  %status = zext i1 %wrong to i32
  ret i32 %status
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<scalar-reduction>");
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
parseNestedMemoryViews(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define void @increment_rows(ptr %base) {
entry:
  br label %outer

outer:
  %row = phi i64 [ 0, %entry ], [ %next.row, %inner.exit ]
  %row.base = getelementptr [4 x i32], ptr %base, i64 %row, i64 0
  br label %inner

inner:
  %column = phi i64 [ 0, %outer ], [ %next.column, %inner ]
  %element = getelementptr i32, ptr %row.base, i64 %column
  %value = load i32, ptr %element, align 4
  %incremented = add i32 %value, 1
  store i32 %incremented, ptr %element, align 4
  %next.column = add nuw nsw i64 %column, 1
  %inner.done = icmp eq i64 %next.column, 4
  br i1 %inner.done, label %inner.exit, label %inner

inner.exit:
  %next.row = add nuw nsw i64 %row, 1
  %outer.done = icmp eq i64 %next.row, 2
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

define i32 @main() {
entry:
  %storage = alloca [2 x [4 x i32]], align 16
  call void @increment_rows(ptr %storage)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<nested-memory-views>");
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
parseDescriptorMemoryView(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

@lookup = private constant [6 x i32]
    [i32 287454020, i32 1432778632, i32 -1, i32 7, i32 9, i32 11], align 16
@alternate = private constant [6 x i32]
    [i32 10, i32 1, i32 2, i32 3, i32 4, i32 5], align 16

define void @descriptor_lookup(ptr %descriptor, ptr %output) {
entry:
  %table = load ptr, ptr %descriptor, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %source = getelementptr i32, ptr %table, i64 %i
  %value = load i32, ptr %source, align 4
  %destination = getelementptr i32, ptr %output, i64 %i
  store i32 %value, ptr %destination, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @descriptor_leaf(ptr %table, ptr %output) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %source = getelementptr i32, ptr %table, i64 %i
  %value = load i32, ptr %source, align 4
  %destination = getelementptr i32, ptr %output, i64 %i
  store i32 %value, ptr %destination, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @descriptor_bridge(ptr %descriptor, ptr %output) {
entry:
  %table = load ptr, ptr %descriptor, align 8
  call void @descriptor_leaf(ptr %table, ptr %output)
  ret void
}

define i32 @nested_descriptor_main() {
entry:
  %descriptor = alloca ptr, align 8
  %output = alloca [4 x i32], align 16
  %lookup.view = getelementptr [6 x i32], ptr @lookup, i64 0, i64 1
  store ptr %lookup.view, ptr %descriptor, align 8
  call void @descriptor_bridge(ptr %descriptor, ptr %output)
  ret i32 0
}

define i32 @main() {
entry:
  %descriptor = alloca ptr, align 8
  %output = alloca [4 x i32], align 16
  %lookup.view = getelementptr [6 x i32], ptr @lookup, i64 0, i64 1
  store ptr %lookup.view, ptr %descriptor, align 8
  call void @descriptor_lookup(ptr %descriptor, ptr %output)
  ret i32 0
}

define i32 @ambiguous_main(i1 %condition) {
entry:
  %descriptor = alloca ptr, align 8
  %output = alloca [4 x i32], align 16
  br i1 %condition, label %left, label %right

left:
  %lookup.view = getelementptr [6 x i32], ptr @lookup, i64 0, i64 1
  store ptr %lookup.view, ptr %descriptor, align 8
  br label %invoke

right:
  %alternate.view = getelementptr [6 x i32], ptr @alternate, i64 0, i64 1
  store ptr %alternate.view, ptr %descriptor, align 8
  br label %invoke

invoke:
  call void @descriptor_lookup(ptr %descriptor, ptr %output)
  ret i32 0
}

define i32 @offset_ambiguous_main(i1 %condition) {
entry:
  %descriptor = alloca ptr, align 8
  %output = alloca [4 x i32], align 16
  %first = getelementptr [6 x i32], ptr @lookup, i64 0, i64 1
  %second = getelementptr [6 x i32], ptr @lookup, i64 0, i64 2
  %view = select i1 %condition, ptr %first, ptr %second
  store ptr %view, ptr %descriptor, align 8
  call void @descriptor_lookup(ptr %descriptor, ptr %output)
  ret i32 0
}

define void @descriptor_repeat_select(ptr %descriptor, ptr %output,
                                      i1 %execute) {
entry:
  br label %outer

outer:
  %iteration = phi i64 [ 0, %entry ], [ %outer.next, %latch ]
  %slot = phi ptr [ %descriptor, %entry ], [ %slot.next, %latch ]
  %table = load ptr, ptr %slot, align 8
  br label %inner

inner:
  %index = phi i64 [ 0, %outer ], [ %next, %inner.latch ]
  br i1 %execute, label %selected, label %inner.latch

selected:
  %source = getelementptr i32, ptr %table, i64 %index
  %value = load i32, ptr %source, align 4
  %destination = getelementptr i32, ptr %output, i64 %index
  store i32 %value, ptr %destination, align 4
  br label %inner.latch

inner.latch:
  %next = add nuw nsw i64 %index, 1
  %inner.done = icmp eq i64 %next, 4
  br i1 %inner.done, label %latch, label %inner

latch:
  %slot.next = getelementptr ptr, ptr %slot, i64 0
  %outer.next = add nuw nsw i64 %iteration, 1
  %outer.done = icmp eq i64 %outer.next, 1
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

define i32 @repeat_select_main() {
entry:
  %descriptor = alloca ptr, align 8
  %output = alloca [4 x i32], align 16
  %lookup.view = getelementptr [6 x i32], ptr @lookup, i64 0, i64 1
  store ptr %lookup.view, ptr %descriptor, align 8
  br label %invoke.loop

invoke.loop:
  %iteration = phi i64 [ 0, %entry ], [ %next, %invoke.loop ]
  %slot = phi ptr [ %descriptor, %entry ], [ %slot.next, %invoke.loop ]
  call void @descriptor_repeat_select(ptr %slot, ptr %output, i1 true)
  %slot.next = getelementptr ptr, ptr %slot, i64 0
  %next = add nuw nsw i64 %iteration, 1
  %done = icmp eq i64 %next, 1
  br i1 %done, label %exit, label %invoke.loop

exit:
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<descriptor-memory-view>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

void configureHostModule(const char *test, llvm::Module &module) {
  static const bool initializationFailed = [] {
    return llvm::InitializeNativeTarget() ||
           llvm::InitializeNativeTargetAsmPrinter();
  }();
  if (initializationFailed)
    fail(test, "cannot initialize the native target");
  auto target = take(test, llvm::orc::JITTargetMachineBuilder::detectHost());
  auto layout = take(test, target.getDefaultDataLayoutForTarget());
  module.setTargetTriple(target.getTargetTriple());
  module.setDataLayout(layout);
}

loom::frontend::StructuredEntityRef
findVecaddLoop(const char *test,
               const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(test, candidate.view());
  auto domain = take(
      test, loom::frontend::enumerateSpatialOwnershipScopeDomain(candidate));
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : domain) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    auto entity = take(test, view.resolve(scope->selection));
    auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(entity.operation);
    if (!loop)
      continue;
    auto callable = loop->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (callable && callable.getSymName() == "vecadd")
      return scope->selection;
  }
  fail(test, "raised vecadd has no eligible structured loop");
}

loom::frontend::StructuredEntityRef
findStructuredLoop(const char *test,
                   const loom::frontend::StructuredProgramCandidate &candidate,
                   llvm::StringRef callableName) {
  auto view = take(test, candidate.view());
  auto domain = take(
      test, loom::frontend::enumerateSpatialOwnershipScopeDomain(candidate));
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : domain) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    auto entity = take(test, view.resolve(scope->selection));
    auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(entity.operation);
    if (!loop)
      continue;
    auto callable = loop->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (callable && callable.getSymName() == callableName)
      return scope->selection;
  }
  fail(test, "raised Structured Program has no requested loop");
}

loom::frontend::StructuredEntityRef findNestedStructuredLoop(
    const char *test,
    const loom::frontend::StructuredProgramCandidate &candidate,
    llvm::StringRef callableName) {
  auto view = take(test, candidate.view());
  auto domain = take(
      test, loom::frontend::enumerateSpatialOwnershipScopeDomain(candidate));
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : domain) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    auto entity = take(test, view.resolve(scope->selection));
    auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(entity.operation);
    if (!loop || !loop->getParentOfType<mlir::scf::WhileOp>())
      continue;
    auto callable = loop->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (callable && callable.getSymName() == callableName)
      return scope->selection;
  }
  fail(test, "raised Structured Program has no requested nested loop");
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

mlir::LLVM::CallOp findHostCall(const char *test, mlir::ModuleOp module,
                                llvm::StringRef caller, llvm::StringRef callee,
                                std::uint64_t requestedOrdinal = 0) {
  mlir::LLVM::CallOp result;
  std::uint64_t ordinal = 0;
  module.walk([&](mlir::LLVM::CallOp call) {
    auto function = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (function && function.getSymName() == caller && call.getCalleeAttr() &&
        call.getCalleeAttr().getValue() == callee &&
        ordinal++ == requestedOrdinal)
      result = call;
  });
  if (!result)
    fail(test, "materialized candidate has no requested host call site");
  return result;
}

mlir::LLVM::CallOp
findHostCall(const char *test,
             const dataflow::CanonicalDataflowArtifact &artifact,
             llvm::StringRef caller, llvm::StringRef callee) {
  return findHostCall(test, artifact.module(), caller, callee);
}

const loom::sim::SimulationMemoryRootCapture &
captureBinding(const char *test,
               const loom::sim::SimulationInputCapturePlan &plan,
               dataflow::LogicalMemoryRootRef root) {
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       plan.memoryRootBindings)
    if (binding.root == root)
      return binding;
  fail(test, "capture plan is missing a logical memory root");
}

loom::frontend::StructuredEntityRef
findCallable(const char *test,
             const loom::frontend::StructuredProgramCandidate &candidate,
             llvm::StringRef name) {
  auto view = take(test, candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto callable =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (callable && callable.getSymName() == name)
      return entity.reference;
  }
  fail(test, "structured callable does not resolve");
}

loom::sim::RuntimeMemoryObject
definedByteObject(llvm::ArrayRef<std::uint8_t> bytes) {
  loom::sim::RuntimeMemoryObject object;
  object.initialBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    object.initialBytes.push_back({loom::sim::SemanticState::Defined, byte});
  return object;
}

std::vector<loom::sim::SemanticMemoryByte>
applyMemoryDiff(const char *test,
                llvm::ArrayRef<loom::sim::SemanticMemoryByte> baseline,
                const loom::sim::DiffMemoryObservation &diff) {
  if (diff.byteCount != baseline.size())
    fail(test, "typed memory diff has the wrong byte count");
  std::vector<loom::sim::SemanticMemoryByte> result(baseline.begin(),
                                                    baseline.end());
  std::uint64_t previousEnd = 0;
  for (const loom::sim::MemoryDiffRun &run : diff.runs) {
    if (run.changedBytes.empty() || run.byteOffset < previousEnd ||
        run.byteOffset + run.changedBytes.size() > result.size())
      fail(test, "typed memory diff has a malformed run");
    if (previousEnd != 0 && run.byteOffset == previousEnd)
      fail(test, "typed memory diff has adjacent non-maximal runs");
    std::copy(run.changedBytes.begin(), run.changedBytes.end(),
              result.begin() + run.byteOffset);
    previousEnd = run.byteOffset + run.changedBytes.size();
  }
  return result;
}

void selectedFmuladdShapesRemainObservable() {
  const char *test = "selectedFmuladdShapesRemainObservable";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-fmuladd-oracle", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext targetContext;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseFmuladd(test, targetContext),
                                 design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipOptions ownership;
  ownership.fmuladdExecutionShape = loom::raising::FMulAddExecutionShape::Fused;
  auto candidate = take(
      test, loom::frontend::materializeSpatialOwnership(
                compiled.structuredProgram,
                findCallable(test, compiled.structuredProgram, "fma_kernel"),
                design.roots().front(), ownership));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto plan = take(test, loom::sim::deriveSimulationInputCapturePlan(
                             view, launch,
                             findHostCall(test, candidate.canonicalDataflow,
                                          "main", "fma_kernel")));
  const auto &outputBinding =
      captureBinding(test, plan.input, memoryRoot(test, view, 3));
  if (outputBinding.floatingWriteLaneType !=
      mlir::Float32Type::get(candidate.canonicalDataflow.module().getContext()))
    fail(test, "fmuladd output is not a uniform floating write root");

  llvm::LLVMContext hostContext;
  std::unique_ptr<llvm::Module> hostModule =
      parseFmuladd(test, hostContext, false);
  configureHostModule(test, *hostModule);
  auto host = take(test, loom::raising::raiseLlvmModuleToStructuredProgram(
                             std::move(hostModule)));
  loom::frontend::SpatialOwnershipScope hostScope{
      findCallable(test, host, "fma_kernel")};
  auto captureShape = [&](loom::raising::FMulAddExecutionShape shape) {
    auto prepared =
        take(test, loom::frontend::prepareSpatialOwnershipSelection(
                       host, hostScope,
                       loom::frontend::SpatialOwnershipDecisionPoint{
                           shape, std::nullopt}));
    auto nativeContext = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> nativeModule =
        parseFmuladd(test, *nativeContext, false);
    configureHostModule(test, *nativeModule);
    return take(test,
                loom::sim::executeStructuredDirectCallSimulationInputCapture(
                    llvm::orc::ThreadSafeModule(std::move(nativeModule),
                                                std::move(nativeContext)),
                    std::move(prepared.module), plan));
  };
  loom::sim::NativeSimulationInputCapture fused =
      captureShape(loom::raising::FMulAddExecutionShape::Fused);
  loom::sim::NativeSimulationInputCapture split =
      captureShape(loom::raising::FMulAddExecutionShape::Split);
  if (fused.entryResult != 0 || split.entryResult != 0 ||
      fused.calls.size() != 1 || split.calls.size() != 1 ||
      outputBinding.objectIndex >= fused.calls.front().objects.size() ||
      outputBinding.objectIndex >= split.calls.front().objects.size())
    fail(test, "fmuladd native captures are malformed");
  const auto &fusedOutput =
      fused.calls.front().objects[outputBinding.objectIndex];
  const auto &splitOutput =
      split.calls.front().objects[outputBinding.objectIndex];
  if (fusedOutput.finalBytes == splitOutput.finalBytes)
    fail(test, "typed fmuladd decisions did not produce distinct results");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void scalarLiveOutExecutesWithoutMemoryObjects() {
  const char *test = "scalarLiveOutExecutesWithoutMemoryObjects";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-scalar-liveout", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> source = parseScalarReduction(test, context);
  configureHostModule(test, *source);
  llvm::Triple foreignTarget(source->getTargetTriple());
  foreignTarget.setVendorName(
      foreignTarget.getVendorName() == "unknown" ? "pc" : "unknown");
  source->setTargetTriple(foreignTarget);
  auto compiled = take(
      test, loom::frontend::compileLlvmModuleToPreMapping(
                std::move(source), design.roots().front().reference(), store));
  loom::frontend::StructuredEntityRef loop =
      findStructuredLoop(test, compiled.structuredProgram, "accum");
  auto domain =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, loop));
  if (domain.size() != 1)
    fail(test, "scalar loop did not expose one exact decision");
  loom::frontend::SpatialOwnershipScope scope{loop};
  auto candidate =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, scope, domain.front(),
                     design.roots().front()));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto prepared =
      take(test, loom::frontend::prepareSpatialOwnershipSelection(
                     compiled.structuredProgram, scope, domain.front()));
  mlir::LLVM::CallOp invocation =
      findHostCall(test, *prepared.module, "main", "accum");
  auto plan = take(
      test, loom::sim::deriveOperationSimulationInputCapturePlan(
                view, launch, prepared.liveIns, prepared.liveOuts, invocation));
  if (!plan.input.objects.empty() || !plan.input.memoryRootBindings.empty() ||
      plan.input.valueResults.size() != 1 ||
      plan.input.valueResults.front().valueResultOrdinal != 0)
    fail(test, "scalar result capture invented memory or lost its output");

  loom::sim::NativeSimulationInputCapture native =
      take(test, loom::sim::executeStructuredSimulationInputCapture(
                     std::move(prepared.module), prepared.operation, plan));
  if (native.entryResult != 0 || native.calls.size() != 1 ||
      native.calls.front().valueResults.size() != 1)
    fail(test, "native oracle did not capture one scalar graph result");

  loom::sim::SpatialSimulationWorkload workload{launch};
  workload.observableContract.valueResults = {0};
  auto finalizedWorkload =
      take(test, loom::sim::finalizeSimulationWorkload(workload, view));
  loom::sim::SpatialSimulationRuntimeInputDraft input{
      finalizedWorkload.identity()};
  auto finalizedInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                       input, finalizedWorkload, view));
  loom::sim::RetiredDFGSimulation execution =
      take(test,
           loom::sim::simulateRetiredDfgWorkload(
               candidate.canonicalDataflow, finalizedWorkload, finalizedInput));
  if (execution.observations.valueResults.size() != 1)
    fail(test, "DFG execution did not publish the scalar graph result");
  const auto *published = std::get_if<loom::sim::PublishedValueResult>(
      &execution.observations.valueResults.front());
  if (!published || published->value.tokenCount != 1 ||
      published->value.lanes.size() != 1 ||
      published->value.lanes.front().state !=
          loom::sim::SemanticState::Defined ||
      published->value.lanes.front().bits != llvm::APInt(32, 28) ||
      native.calls.front().valueResults.front().lanes.size() != 1 ||
      native.calls.front().valueResults.front().lanes.front().bits !=
          published->value.lanes.front().bits)
    fail(test, "native and DFG scalar result observations differ");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void operationCandidateCapturesCallerOwnedMemory() {
  const char *test = "operationCandidateCapturesCallerOwnedMemory";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-operation-memory-capture", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseVecadd(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipScope scope{
      findVecaddLoop(test, compiled.structuredProgram)};
  auto decisions =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, scope.selection));
  if (decisions.empty())
    fail(test, "vecadd loop has no ownership decision");
  auto candidate =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, scope, decisions.front(),
                     design.roots().front()));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto prepared =
      take(test, loom::frontend::prepareSpatialOwnershipSelection(
                     compiled.structuredProgram, scope, decisions.front()));
  mlir::LLVM::CallOp invocation =
      findHostCall(test, *prepared.module, "main", "vecadd");
  auto plan = take(
      test, loom::sim::deriveOperationSimulationInputCapturePlan(
                view, launch, prepared.liveIns, prepared.liveOuts, invocation));
  if (plan.input.objects.size() != 3 ||
      plan.input.memoryRootBindings.size() != 3)
    fail(test, "operation capture did not recover caller-owned memory");
  for (const loom::sim::SimulationMemoryCaptureObject &object :
       plan.input.objects)
    if (object.byteCount != 64 * sizeof(float) || object.operandByteOffset != 0)
      fail(test, "operation capture has the wrong backing-object extent");

  loom::sim::NativeSimulationInputCapture capture =
      take(test, loom::sim::executeStructuredSimulationInputCapture(
                     std::move(prepared.module), prepared.operation, plan));
  if (capture.entryResult != 0 || capture.calls.size() != 1 ||
      capture.calls.front().objects.size() != 3)
    fail(test, "operation oracle did not capture the exact host invocation");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void nestedOperationCandidateUsesExactCallPath() {
  const char *test = "nestedOperationCandidateUsesExactCallPath";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-nested-operation-capture", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseTableLookup(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipScope scope{
      findStructuredLoop(test, compiled.structuredProgram, "table_lookup_arg")};
  auto decisions =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, scope.selection));
  if (decisions.empty())
    fail(test, "table lookup loop has no ownership decision");
  auto candidate =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, scope, decisions.front(),
                     design.roots().front()));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto prepared =
      take(test, loom::frontend::prepareSpatialOwnershipSelection(
                     compiled.structuredProgram, scope, decisions.front()));
  llvm::SmallVector<mlir::LLVM::CallOp, 2> path{
      findHostCall(test, *prepared.module, "nested_main",
                   "table_lookup_wrapper", 0),
      findHostCall(test, *prepared.module, "table_lookup_wrapper",
                   "table_lookup_arg")};
  auto plan =
      take(test, loom::sim::deriveOperationSimulationInputCapturePlan(
                     view, launch, prepared.liveIns, prepared.liveOuts, path));
  if (plan.input.objects.size() != 2 ||
      plan.input.memoryRootBindings.size() != 2)
    fail(test, "nested operation capture lost a finite backing object");

  loom::sim::NativeSimulationInputCapture capture =
      take(test, loom::sim::executeStructuredSimulationInputCapture(
                     std::move(prepared.module), prepared.operation, plan,
                     "nested_main"));
  if (capture.entryResult != 0 || capture.calls.size() != 1 ||
      capture.calls.front().objects.size() != 2)
    fail(test, "nested operation capture conflated static call paths");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void operationCandidateCapturesInvocationLocalMemoryViews() {
  const char *test = "operationCandidateCapturesInvocationLocalMemoryViews";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-operation-memory-views", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseNestedMemoryViews(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipScope scope{findNestedStructuredLoop(
      test, compiled.structuredProgram, "increment_rows")};
  auto decisions =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, scope.selection));
  if (decisions.empty())
    fail(test, "nested row loop has no ownership decision");
  auto candidate =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, scope, decisions.front(),
                     design.roots().front()));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto prepared =
      take(test, loom::frontend::prepareSpatialOwnershipSelection(
                     compiled.structuredProgram, scope, decisions.front()));
  mlir::LLVM::CallOp invocation =
      findHostCall(test, *prepared.module, "main", "increment_rows");
  auto plan = take(
      test, loom::sim::deriveOperationSimulationInputCapturePlan(
                view, launch, prepared.liveIns, prepared.liveOuts, invocation));
  if (plan.input.objects.size() != 1 ||
      plan.input.memoryRootBindings.size() != 1 ||
      plan.input.objects.front().byteCount != 8 * sizeof(std::int32_t))
    fail(test, "nested row capture lost its finite backing object");

  loom::sim::NativeSimulationInputCapture capture =
      take(test, loom::sim::executeStructuredSimulationInputCapture(
                     std::move(prepared.module), prepared.operation, plan));
  if (capture.entryResult != 0 || capture.calls.size() != 2 ||
      capture.calls.front().memoryRootByteOffsets.size() != 1 ||
      capture.calls.back().memoryRootByteOffsets.size() != 1 ||
      capture.calls.front().memoryRootByteOffsets.front() != 0 ||
      capture.calls.back().memoryRootByteOffsets.front() !=
          4 * sizeof(std::int32_t))
    fail(test, "dynamic row views did not retain invocation-local offsets");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void operationCandidateCapturesDescriptorLoadedMemory() {
  const char *test = "operationCandidateCapturesDescriptorLoadedMemory";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-operation-descriptor-memory", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseDescriptorMemoryView(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipScope scope{findStructuredLoop(
      test, compiled.structuredProgram, "descriptor_lookup")};
  auto decisions =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, scope.selection));
  if (decisions.empty())
    fail(test, "descriptor lookup loop has no ownership decision");
  auto candidate =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, scope, decisions.front(),
                     design.roots().front()));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto prepared =
      take(test, loom::frontend::prepareSpatialOwnershipSelection(
                     compiled.structuredProgram, scope, decisions.front()));
  mlir::LLVM::CallOp invocation =
      findHostCall(test, *prepared.module, "main", "descriptor_lookup");
  auto plan = take(
      test, loom::sim::deriveOperationSimulationInputCapturePlan(
                view, launch, prepared.liveIns, prepared.liveOuts, invocation));
  if (plan.input.objects.size() != 2 ||
      plan.input.memoryRootBindings.size() != 2)
    fail(test, "descriptor memory capture lost its finite backing object");
  std::optional<std::uint64_t> tableObjectOrdinal;
  std::optional<std::uint64_t> tableRootOrdinal;
  for (auto [ordinal, object] : llvm::enumerate(plan.input.objects))
    if (object.byteCount == 6 * sizeof(std::int32_t))
      tableObjectOrdinal = ordinal;
  if (!tableObjectOrdinal)
    fail(test, "descriptor capture lost the complete table allocation");
  for (auto [ordinal, root] : llvm::enumerate(plan.input.memoryRootBindings))
    if (root.objectIndex == *tableObjectOrdinal)
      tableRootOrdinal = ordinal;
  if (!tableRootOrdinal)
    fail(test, "descriptor table has no logical memory-root binding");

  loom::frontend::SpatialOwnershipScope transitiveScope{
      findStructuredLoop(test, compiled.structuredProgram, "descriptor_leaf")};
  auto transitiveDecisions =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, transitiveScope.selection));
  if (transitiveDecisions.empty())
    fail(test, "transitive descriptor leaf has no ownership decision");
  auto transitiveCandidate =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, transitiveScope,
                     transitiveDecisions.front(), design.roots().front()));
  auto transitiveView =
      take(test, transitiveCandidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef transitiveLaunch =
      onlyLaunch(test, transitiveView);
  auto transitivePrepared =
      take(test, loom::frontend::prepareSpatialOwnershipSelection(
                     compiled.structuredProgram, transitiveScope,
                     transitiveDecisions.front()));
  llvm::SmallVector<mlir::LLVM::CallOp, 2> transitivePath{
      findHostCall(test, *transitivePrepared.module, "nested_descriptor_main",
                   "descriptor_bridge"),
      findHostCall(test, *transitivePrepared.module, "descriptor_bridge",
                   "descriptor_leaf")};
  auto transitivePlan = take(
      test, loom::sim::deriveOperationSimulationInputCapturePlan(
                transitiveView, transitiveLaunch, transitivePrepared.liveIns,
                transitivePrepared.liveOuts, transitivePath));
  if (transitivePlan.input.objects.size() != 2 ||
      transitivePlan.input.memoryRootBindings.size() != 2)
    fail(test, "transitive descriptor capture lost a finite backing object");

  mlir::LLVM::CallOp ambiguousInvocation = findHostCall(
      test, *prepared.module, "ambiguous_main", "descriptor_lookup");
  auto ambiguous = loom::sim::deriveOperationSimulationInputCapturePlan(
      view, launch, prepared.liveIns, prepared.liveOuts, ambiguousInvocation);
  if (ambiguous)
    fail(test, "ambiguous descriptor stores unexpectedly produced a plan");
  std::string ambiguity = llvm::toString(ambiguous.takeError());
  if (ambiguity.find("simulation_input_capture_unsupported") ==
      std::string::npos)
    fail(test, "descriptor ambiguity reported the wrong failure: " + ambiguity);

  mlir::LLVM::CallOp offsetAmbiguousInvocation = findHostCall(
      test, *prepared.module, "offset_ambiguous_main", "descriptor_lookup");
  auto offsetAmbiguous = loom::sim::deriveOperationSimulationInputCapturePlan(
      view, launch, prepared.liveIns, prepared.liveOuts,
      offsetAmbiguousInvocation);
  if (offsetAmbiguous)
    fail(test, "distinct descriptor offsets unexpectedly produced a plan");
  std::string offsetAmbiguity = llvm::toString(offsetAmbiguous.takeError());
  if (offsetAmbiguity.find("simulation_input_capture_unsupported") ==
      std::string::npos)
    fail(test, "descriptor offset ambiguity reported the wrong failure: " +
                   offsetAmbiguity);

  loom::frontend::SpatialOwnershipScope nestedScope{findNestedStructuredLoop(
      test, compiled.structuredProgram, "descriptor_repeat_select")};
  auto nestedDecisions =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, nestedScope.selection));
  if (nestedDecisions.empty())
    fail(test, "repeat/select descriptor loop has no ownership decision");
  auto nestedCandidate =
      take(test, loom::frontend::materializeSpatialOwnershipDecision(
                     compiled.structuredProgram, nestedScope,
                     nestedDecisions.front(), design.roots().front()));
  auto nestedView = take(test, nestedCandidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef nestedLaunch = onlyLaunch(test, nestedView);
  auto nestedPrepared =
      take(test, loom::frontend::prepareSpatialOwnershipSelection(
                     compiled.structuredProgram, nestedScope,
                     nestedDecisions.front()));
  mlir::LLVM::CallOp nestedInvocation =
      findHostCall(test, *nestedPrepared.module, "repeat_select_main",
                   "descriptor_repeat_select");
  auto nestedPlan =
      take(test, loom::sim::deriveOperationSimulationInputCapturePlan(
                     nestedView, nestedLaunch, nestedPrepared.liveIns,
                     nestedPrepared.liveOuts, nestedInvocation));
  if (nestedPlan.input.objects.size() != 2 ||
      nestedPlan.input.memoryRootBindings.size() != 2)
    fail(test, "repeat/select descriptor capture lost a backing object");
  mlir::Operation *nestedOperation = nestedPrepared.operation;
  loom::sim::NativeSimulationInputCapture nestedCapture =
      take(test, loom::sim::executeStructuredSimulationInputCapture(
                     std::move(nestedPrepared.module), nestedOperation,
                     nestedPlan, "repeat_select_main"));
  if (nestedCapture.entryResult != 0 || nestedCapture.calls.size() != 1 ||
      nestedCapture.calls.front().objects.size() != 2)
    fail(test,
         "repeat/select descriptor oracle lost its exact graph activation");

  loom::sim::NativeSimulationInputCapture capture =
      take(test, loom::sim::executeStructuredSimulationInputCapture(
                     std::move(prepared.module), prepared.operation, plan));
  if (capture.entryResult != 0 || capture.calls.size() != 1 ||
      capture.calls.front().objects.size() != 2)
    fail(test, "descriptor memory oracle did not capture one invocation");
  const auto &table = capture.calls.front().objects[*tableObjectOrdinal];
  if (table.initialBytes.size() != 6 * sizeof(std::int32_t) ||
      table.initialBytes[0] != 0x44 || table.initialBytes[1] != 0x33 ||
      table.initialBytes[2] != 0x22 || table.initialBytes[3] != 0x11 ||
      capture.calls.front().memoryRootByteOffsets[*tableRootOrdinal] !=
          sizeof(std::int32_t))
    fail(test, "descriptor interior view lost its backing-object projection");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void wholeCallableScalarResultUsesCallerStorage() {
  const char *test = "wholeCallableScalarResultUsesCallerStorage";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-whole-callable-result", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseScalarReduction(test, context),
                                 design.roots().front().reference(), store));
  auto candidate =
      take(test, loom::frontend::materializeSpatialOwnership(
                     compiled.structuredProgram,
                     findCallable(test, compiled.structuredProgram, "accum"),
                     design.roots().front()));

  auto callable =
      candidate.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "accum");
  if (!callable || callable.getBody().getBlocks().size() != 1 ||
      !llvm::isa<mlir::IntegerType>(callable.getFunctionType().getReturnType()))
    fail(test, "whole-callable ownership changed the LLVM ABI authority");
  dataflow::ThreadLaunchOp launch;
  dataflow::ThreadWaitOp wait;
  mlir::LLVM::ReturnOp returnOp;
  callable.getBody().walk([&](mlir::Operation *operation) {
    if (auto candidate = llvm::dyn_cast<dataflow::ThreadLaunchOp>(operation))
      launch = candidate;
    else if (auto candidate = llvm::dyn_cast<dataflow::ThreadWaitOp>(operation))
      wait = candidate;
    else if (auto candidate = llvm::dyn_cast<mlir::LLVM::ReturnOp>(operation))
      returnOp = candidate;
  });
  if (!launch || !wait || !returnOp || returnOp.getNumOperands() != 1 ||
      !returnOp.getOperand(0).getDefiningOp<mlir::LLVM::LoadOp>())
    fail(test, "whole-callable result did not cross caller-owned storage");

  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef rooted = onlyLaunch(test, view);
  auto graphLaunch = take(test, view.resolve(rooted.staticGraphLaunch));
  auto graphView = take(test, view.resolve(graphLaunch.callee));
  auto graph = llvm::dyn_cast_or_null<dataflow::GraphOp>(graphView.op);
  if (!graph || graph.getFunctionType().getNumResults() != 1)
    fail(test, "whole-callable graph lost its scalar value result");

  auto plan = take(test, loom::sim::deriveSimulationInputCapturePlan(
                             view, rooted,
                             findHostCall(test, candidate.canonicalDataflow,
                                          "main", "accum")));
  if (!plan.input.objects.empty() || !plan.input.memoryRootBindings.empty() ||
      plan.input.valueResults.size() != 1 ||
      plan.input.valueResults.front().valueResultOrdinal != 0)
    fail(test, "whole-callable capture lost its direct scalar result");

  auto capture = [&](bool selected) {
    auto nativeContext = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> nativeModule =
        parseScalarReduction(test, *nativeContext);
    configureHostModule(test, *nativeModule);
    llvm::orc::ThreadSafeModule hostModule(std::move(nativeModule),
                                           std::move(nativeContext));
    if (!selected)
      return take(test, loom::sim::executeNativeSimulationInputCapture(
                            std::move(hostModule), plan));

    llvm::LLVMContext selectedContext;
    std::unique_ptr<llvm::Module> selectedModule =
        parseScalarReduction(test, selectedContext);
    configureHostModule(test, *selectedModule);
    auto selectedProgram =
        take(test, loom::raising::raiseLlvmModuleToStructuredProgram(
                       std::move(selectedModule)));
    loom::frontend::SpatialOwnershipScope scope{
        findCallable(test, selectedProgram, "accum")};
    auto domain =
        take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                       selectedProgram, scope.selection));
    if (domain.size() != 1)
      fail(test, "whole-callable scalar result has an ambiguous decision");
    auto prepared = take(test, loom::frontend::prepareSpatialOwnershipSelection(
                                   selectedProgram, scope, domain.front()));
    return take(test,
                loom::sim::executeStructuredDirectCallSimulationInputCapture(
                    std::move(hostModule), std::move(prepared.module), plan));
  };
  loom::sim::NativeSimulationInputCapture source = capture(false);
  loom::sim::NativeSimulationInputCapture selected = capture(true);
  if (source.entryResult != 0 || selected.entryResult != 0 ||
      source.calls.size() != 1 || selected.calls.size() != 1 ||
      source.calls.front().valueResults.size() != 1 ||
      selected.calls.front().valueResults.size() != 1 ||
      source.calls.front().valueResults.front().lanes.size() != 1 ||
      selected.calls.front().valueResults.front().lanes.size() != 1 ||
      source.calls.front().valueResults.front().lanes.front().bits !=
          llvm::APInt(32, 28) ||
      selected.calls.front().valueResults.front().lanes.front().bits !=
          llvm::APInt(32, 28))
    fail(test, "whole-callable native result capture is not exact");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
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
  auto nativeContext = std::make_unique<llvm::LLVMContext>();
  auto nativeModule = parseVecadd(test, *nativeContext, false);
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseVecadd(test, context),
                                 design.roots().front().reference(), store));

  loom::frontend::SpatialOwnershipOptions ownership;
  ownership.canonicalIndexWidth = 32;
  auto candidate =
      take(test, loom::frontend::materializeSpatialOwnership(
                     compiled.structuredProgram,
                     findVecaddLoop(test, compiled.structuredProgram),
                     design.roots().front(), ownership));
  auto view = take(test, candidate.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().size() < 20)
    fail(test, "source candidate did not produce a substantive Dataflow graph");

  const dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto capturePlan = take(
      test,
      loom::sim::deriveSimulationInputCapturePlan(
          view, launch,
          findHostCall(test, candidate.canonicalDataflow, "main", "vecadd")));
  if (capturePlan.input.objects.size() != 3 ||
      capturePlan.input.memoryRootBindings.size() != 3)
    fail(test, "vecadd capture plan did not recover three memory objects");
  if (!captureBinding(test, capturePlan.input, memoryRoot(test, view, 0))
           .requiresInitialState ||
      !captureBinding(test, capturePlan.input, memoryRoot(test, view, 1))
           .requiresInitialState ||
      captureBinding(test, capturePlan.input, memoryRoot(test, view, 2))
          .requiresInitialState)
    fail(test, "vecadd capture plan has the wrong initial-state relation");
  if (capturePlan.input.valueInputs.size() != 1 ||
      capturePlan.input.valueInputs.front().fixedValue ||
      capturePlan.input.valueInputs.front().boundaryOperandOrdinal != 3)
    fail(test, "vecadd capture plan did not recover its runtime scalar");
  for (const loom::sim::SimulationMemoryCaptureObject &object :
       capturePlan.input.objects)
    if (object.byteCount != 64 * sizeof(float))
      fail(test, "vecadd capture object has the wrong byte extent");
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       capturePlan.input.memoryRootBindings)
    if (binding.byteOffset != 0 || binding.objectIndex >= 3)
      fail(test, "vecadd capture binding has the wrong object projection");
  if (captureBinding(test, capturePlan.input, memoryRoot(test, view, 0))
          .floatingWriteLaneType ||
      captureBinding(test, capturePlan.input, memoryRoot(test, view, 1))
          .floatingWriteLaneType ||
      captureBinding(test, capturePlan.input, memoryRoot(test, view, 2))
              .floatingWriteLaneType !=
          mlir::Float32Type::get(
              candidate.canonicalDataflow.module().getContext()))
    fail(test, "vecadd capture plan has the wrong floating write projection");

  loom::sim::NativeSimulationInputCapture nativeCapture =
      take(test, loom::sim::executeNativeSimulationInputCapture(
                     llvm::orc::ThreadSafeModule(std::move(nativeModule),
                                                 std::move(nativeContext)),
                     capturePlan));
  if (nativeCapture.entryResult != 0 || nativeCapture.calls.size() != 1 ||
      nativeCapture.calls.front().objects.size() != 3 ||
      nativeCapture.calls.front().runtimeValues.size() != 1)
    fail(test, "native vecadd oracle did not capture one complete call");

  auto mismatchedPlan = capturePlan;
  mismatchedPlan.input.objects.front().byteCount -= sizeof(float);
  auto mismatchContext = std::make_unique<llvm::LLVMContext>();
  auto mismatchedCapture = loom::sim::executeNativeSimulationInputCapture(
      llvm::orc::ThreadSafeModule(parseVecadd(test, *mismatchContext, false),
                                  std::move(mismatchContext)),
      mismatchedPlan);
  if (mismatchedCapture)
    fail(test, "native oracle accepted a mismatched host allocation extent");
  llvm::consumeError(mismatchedCapture.takeError());

  auto slicePlan =
      take(test, loom::sim::deriveSimulationInputCapturePlan(
                     view, launch,
                     findHostCall(test, candidate.canonicalDataflow,
                                  "slice_main", "vecadd")));
  if (slicePlan.input.valueInputs.size() != 1 ||
      !slicePlan.input.valueInputs.front().fixedValue)
    fail(test, "constant call operand did not become a fixed graph input");
  const auto &sliceA =
      captureBinding(test, slicePlan.input, memoryRoot(test, view, 0));
  const auto &sliceB =
      captureBinding(test, slicePlan.input, memoryRoot(test, view, 1));
  const auto &sliceC =
      captureBinding(test, slicePlan.input, memoryRoot(test, view, 2));
  if (slicePlan.input.objects.size() != 2 ||
      slicePlan.input.objects[sliceA.objectIndex].byteCount !=
          128 * sizeof(float) ||
      sliceA.objectIndex != sliceB.objectIndex || sliceA.byteOffset != 0 ||
      sliceB.byteOffset != 64 * sizeof(float) ||
      sliceC.objectIndex == sliceA.objectIndex || sliceC.byteOffset != 0)
    fail(test, "vecadd slice capture did not preserve host aliasing");

  loom::sim::SpatialSimulationWorkload workload{launch};
  workload.valueInputPlan.push_back(loom::sim::RuntimeValueInput{});
  workload.observableContract.memories.push_back(
      loom::sim::SpatialMemoryObservable{
          dataflow::LogicalMemoryRootOrViewRef{memoryRoot(test, view, 2)},
          loom::sim::MemoryObservationForm::DiffFromRuntimeInput});
  auto finalizedWorkload =
      take(test, loom::sim::finalizeSimulationWorkload(workload, view));

  loom::sim::SpatialSimulationRuntimeInputDraft input{
      finalizedWorkload.identity()};
  input.runtimeValues = nativeCapture.calls.front().runtimeValues;
  for (const loom::sim::NativeCapturedMemoryObject &object :
       nativeCapture.calls.front().objects)
    input.memoryObjects.push_back(definedByteObject(object.initialBytes));
  for (const loom::sim::SimulationMemoryRootCapture &binding :
       capturePlan.input.memoryRootBindings)
    input.memoryRootBindings.push_back(loom::sim::RuntimeMemoryBindingDraft{
        binding.root, binding.objectIndex, binding.byteOffset});
  auto finalizedInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                       input, finalizedWorkload, view));

  loom::sim::RetiredDFGSimulation execution =
      take(test,
           loom::sim::simulateRetiredDfgWorkload(
               candidate.canonicalDataflow, finalizedWorkload, finalizedInput));
  loom::sim::DFGSimulationReport &report = execution.report;
  if (report.status != "pass")
    fail(test, "typed DFG execution did not retire: " + report.status);
  if (report.operationFireCounts[dataflow::OperationSchemaId::ArithAddF] !=
          128 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowLoad] !=
          128 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowStore] !=
          64)
    fail(test, "typed DFG execution did not run the vecadd workload");

  if (!report.finalMemoryState.empty() || !report.finalMemoryRoots.empty())
    fail(test, "typed DFG execution retained the legacy memory report");

  if (execution.observations.valueResults.size() != 0 ||
      execution.observations.streamOutputs.size() != 0 ||
      execution.observations.memories.size() != 1)
    fail(test, "typed DFG observations do not match the workload contract");
  const auto *diff = std::get_if<loom::sim::DiffMemoryObservation>(
      &execution.observations.memories.front());
  if (!diff)
    fail(test, "typed DFG execution did not preserve the requested diff form");
  const dataflow::LogicalMemoryRootRef destinationRoot =
      memoryRoot(test, view, 2);
  const loom::sim::MemoryRootBindingEntry *destinationBinding = nullptr;
  for (const loom::sim::MemoryRootBindingEntry &binding :
       finalizedInput.spatial()->memoryRootBindings)
    if (binding.root == destinationRoot)
      destinationBinding = &binding;
  if (!destinationBinding)
    fail(test, "typed runtime input lost the destination root binding");
  std::vector<loom::sim::SemanticMemoryByte> reconstructed = applyMemoryDiff(
      test,
      finalizedInput.spatial()
          ->memoryObjects[destinationBinding->binding.objectOrdinal]
          .initialBytes,
      *diff);
  const loom::sim::SimulationMemoryRootCapture &capturedDestination =
      captureBinding(test, capturePlan.input, destinationRoot);
  llvm::ArrayRef<std::uint8_t> expected(
      nativeCapture.calls.front()
          .objects[capturedDestination.objectIndex]
          .finalBytes);
  expected = expected.drop_front(capturedDestination.byteOffset);
  if (reconstructed.size() != expected.size())
    fail(test, "typed DFG memory result has the wrong extent");
  for (std::size_t index = 0; index < expected.size(); ++index)
    if (reconstructed[index].state != loom::sim::SemanticState::Defined ||
        reconstructed[index].value != expected[index])
      fail(test, "typed DFG memory diff reconstructs the wrong result");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

void staticTableExecutesThroughTypedDfgInput() {
  const char *test = "staticTableExecutesThroughTypedDfgInput";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-static-table-dfg", directory);
  if (error)
    fail(test, "cannot create artifact store: " + error.message());
  loom::ArtifactStore store(directory);

  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  auto nativeContext = std::make_unique<llvm::LLVMContext>();
  auto nativeModule = parseTableLookup(test, *nativeContext, false);
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseTableLookup(test, context),
                                 design.roots().front().reference(), store));

  loom::frontend::SpatialOwnershipOptions ownership;
  ownership.canonicalIndexWidth = 64;
  auto candidate = take(
      test, loom::frontend::materializeSpatialOwnership(
                compiled.structuredProgram,
                findCallable(test, compiled.structuredProgram, "table_lookup"),
                design.roots().front(), ownership));
  auto view = take(test, candidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef launch = onlyLaunch(test, view);
  auto sources = take(test, loom::frontend::deriveRootedLogicalMemorySources(
                                compiled.staticGlobalMemory, view, launch));
  if (sources.size() != 2)
    fail(test, "table lookup did not retain two logical memory roots");

  loom::sim::RuntimeMemoryObject table;
  loom::sim::RuntimeMemoryObject output =
      definedByteObject(std::vector<std::uint8_t>(16, 0));
  std::optional<dataflow::LogicalMemoryRootRef> tableRoot;
  std::optional<dataflow::LogicalMemoryRootRef> outputRoot;
  for (const loom::frontend::RootedLogicalMemorySource &source : sources) {
    if (!source.globalOrdinal) {
      outputRoot = source.root;
      continue;
    }
    if (*source.globalOrdinal >= compiled.staticGlobalMemory.globals.size())
      fail(test, "static global ordinal is out of range");
    const loom::frontend::StaticGlobalMemory &global =
        compiled.staticGlobalMemory.globals[*source.globalOrdinal];
    if (global.symbol != "lookup" ||
        global.provision != loom::frontend::StaticGlobalProvision::Image)
      fail(test, "lookup table has no exact static image");
    table = definedByteObject(global.bytes);
    tableRoot = source.root;
  }
  if (!tableRoot || !outputRoot)
    fail(test, "table and runtime output roots were not distinguished");

  auto directPlan =
      take(test, loom::sim::deriveSimulationInputCapturePlan(
                     view, launch,
                     findHostCall(test, candidate.canonicalDataflow,
                                  "direct_main", "table_lookup")));
  if (directPlan.input.objects.size() != 2 ||
      directPlan.input.memoryRootBindings.size() != 2)
    fail(test, "direct table capture lost its global or output object");
  auto directContext = std::make_unique<llvm::LLVMContext>();
  auto directModule = parseTableLookup(test, *directContext, false);
  loom::sim::NativeSimulationInputCapture directCapture =
      take(test, loom::sim::executeNativeSimulationInputCapture(
                     llvm::orc::ThreadSafeModule(std::move(directModule),
                                                 std::move(directContext)),
                     directPlan, "direct_main"));
  if (directCapture.entryResult != 0 || directCapture.calls.size() != 1 ||
      directCapture.calls.front().objects.size() != 2)
    fail(test, "direct table oracle did not capture one complete call");
  const auto &directTable =
      directCapture.calls.front().objects
          [captureBinding(test, directPlan.input, *tableRoot).objectIndex];
  const auto &directOutput =
      directCapture.calls.front().objects
          [captureBinding(test, directPlan.input, *outputRoot).objectIndex];
  if (directTable.initialBytes !=
          compiled.staticGlobalMemory.lookup("lookup")->bytes ||
      directTable.finalBytes != directTable.initialBytes ||
      directOutput.finalBytes != directTable.initialBytes)
    fail(test, "direct table oracle did not preserve the static image");

  auto captureCandidate = take(
      test,
      loom::frontend::materializeSpatialOwnership(
          compiled.structuredProgram,
          findCallable(test, compiled.structuredProgram, "table_lookup_arg"),
          design.roots().front(), ownership));
  auto captureView = take(test, captureCandidate.canonicalDataflow.view());
  dataflow::RootedGraphLaunchRef captureLaunch = onlyLaunch(test, captureView);
  mlir::LLVM::CallOp hostCall;
  captureCandidate.canonicalDataflow.module().walk(
      [&](mlir::LLVM::CallOp call) {
        auto caller = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
        if (caller && caller.getSymName() == "main" && call.getCalleeAttr() &&
            call.getCalleeAttr().getValue() == "table_lookup_arg")
          hostCall = call;
      });
  if (!hostCall)
    fail(test, "table lookup candidate has no direct host call");
  auto capturePlan = take(test, loom::sim::deriveSimulationInputCapturePlan(
                                    captureView, captureLaunch, hostCall));
  if (capturePlan.input.objects.size() != 2 ||
      capturePlan.input.memoryRootBindings.size() != 2)
    fail(test, "table lookup capture did not retain both backing objects");
  for (const loom::sim::SimulationMemoryCaptureObject &object :
       capturePlan.input.objects)
    if (object.byteCount != 4 * sizeof(std::int32_t))
      fail(test, "table lookup capture object has the wrong byte extent");
  loom::sim::NativeSimulationInputCapture nativeCapture =
      take(test, loom::sim::executeNativeSimulationInputCapture(
                     llvm::orc::ThreadSafeModule(std::move(nativeModule),
                                                 std::move(nativeContext)),
                     capturePlan));
  if (nativeCapture.entryResult != 0 || nativeCapture.calls.size() != 1 ||
      nativeCapture.calls.front().objects.size() != 2)
    fail(test, "native table oracle did not capture one complete call");

  llvm::SmallVector<mlir::LLVM::CallOp, 2> nestedPath{
      findHostCall(test, captureCandidate.canonicalDataflow.module(),
                   "nested_main", "table_lookup_wrapper", 0),
      findHostCall(test, captureCandidate.canonicalDataflow.module(),
                   "table_lookup_wrapper", "table_lookup_arg")};
  auto nestedPlan = take(test, loom::sim::deriveSimulationInputCapturePlan(
                                   captureView, captureLaunch, nestedPath));
  auto nestedContext = std::make_unique<llvm::LLVMContext>();
  auto nestedModule = parseTableLookup(test, *nestedContext, false);
  loom::sim::NativeSimulationInputCapture nestedCapture =
      take(test, loom::sim::executeNativeSimulationInputCapture(
                     llvm::orc::ThreadSafeModule(std::move(nestedModule),
                                                 std::move(nestedContext)),
                     nestedPlan, "nested_main"));
  if (nestedCapture.entryResult != 0 || nestedCapture.calls.size() != 1 ||
      nestedCapture.calls.front().objects.size() != 2)
    fail(test, "whole-callable oracle conflated nested call paths");

  const auto &capturedTable =
      nativeCapture.calls.front()
          .objects[captureBinding(test, capturePlan.input,
                                  memoryRoot(test, captureView, 0))
                       .objectIndex];
  const auto &capturedOutput =
      nativeCapture.calls.front()
          .objects[captureBinding(test, capturePlan.input,
                                  memoryRoot(test, captureView, 1))
                       .objectIndex];
  const loom::frontend::StaticGlobalMemory *staticTable =
      compiled.staticGlobalMemory.lookup("lookup");
  if (!staticTable || capturedTable.initialBytes != staticTable->bytes ||
      capturedTable.finalBytes != capturedTable.initialBytes ||
      capturedOutput.finalBytes != capturedTable.initialBytes)
    fail(test, "native table oracle did not preserve the static image");

  loom::sim::SpatialSimulationWorkload workload{launch};
  workload.observableContract.memories.push_back(
      loom::sim::SpatialMemoryObservable{
          dataflow::LogicalMemoryRootOrViewRef{*outputRoot},
          loom::sim::MemoryObservationForm::FullState});
  auto finalizedWorkload =
      take(test, loom::sim::finalizeSimulationWorkload(workload, view));

  loom::sim::SpatialSimulationRuntimeInputDraft input{
      finalizedWorkload.identity()};
  input.memoryObjects = {std::move(table), std::move(output)};
  input.memoryRootBindings = {
      loom::sim::RuntimeMemoryBindingDraft{*tableRoot, 0, 0},
      loom::sim::RuntimeMemoryBindingDraft{*outputRoot, 1, 0}};
  auto finalizedInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                       input, finalizedWorkload, view));

  std::optional<std::uint64_t> outputObject;
  for (const loom::sim::MemoryRootBindingEntry &binding :
       finalizedInput.spatial()->memoryRootBindings)
    if (binding.root == *outputRoot)
      outputObject = binding.binding.objectOrdinal;
  if (!outputObject)
    fail(test, "finalized runtime input lost the output root binding");

  loom::sim::RetiredDFGSimulation execution =
      take(test,
           loom::sim::simulateRetiredDfgWorkload(
               candidate.canonicalDataflow, finalizedWorkload, finalizedInput));
  loom::sim::DFGSimulationReport &report = execution.report;
  if (report.status != "pass" ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowLoad] !=
          4 ||
      report.operationFireCounts[dataflow::OperationSchemaId::DataflowStore] !=
          4)
    fail(test, "table workload did not execute real memory actors");

  if (!report.finalMemoryState.empty() || !report.finalMemoryRoots.empty())
    fail(test, "typed table execution retained the legacy memory report");
  if (execution.observations.memories.size() != 1)
    fail(test, "typed table execution lost its output observation");
  const auto *finalOutput = std::get_if<loom::sim::FullMemoryObservation>(
      &execution.observations.memories.front());
  const std::vector<std::uint8_t> expectedBytes = {
      0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55,
      0xff, 0xff, 0xff, 0xff, 0x07, 0x00, 0x00, 0x00};
  if (!finalOutput || finalOutput->bytes.size() != expectedBytes.size())
    fail(test, "typed table execution produced the wrong output extent");
  for (std::size_t index = 0; index < expectedBytes.size(); ++index)
    if (finalOutput->bytes[index].state != loom::sim::SemanticState::Defined ||
        finalOutput->bytes[index].value != expectedBytes[index])
      fail(test, "typed table execution produced the wrong output bytes");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store: " + cleanup.message());
}

} // namespace

int main() {
  selectedFmuladdShapesRemainObservable();
  scalarLiveOutExecutesWithoutMemoryObjects();
  operationCandidateCapturesCallerOwnedMemory();
  nestedOperationCandidateUsesExactCallPath();
  operationCandidateCapturesInvocationLocalMemoryViews();
  operationCandidateCapturesDescriptorLoadedMemory();
  wholeCallableScalarResultUsesCallerStorage();
  sourceCandidateExecutesThroughTypedDfgInput();
  staticTableExecutesThroughTypedDfgInput();
  llvm::outs() << "frontend to typed DFG integration anchor passed\n";
  return EXIT_SUCCESS;
}
