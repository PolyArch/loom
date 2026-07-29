#include "Frontend/Compilation/PreMappingCompilation.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/IndexWidth.h"
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
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

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
parseConstantCallbackModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define internal void @target() {
entry:
  ret void
}

define internal void @wrong_target() {
entry:
  ret void
}

define internal void @dispatch(ptr %callback) {
entry:
  call void %callback()
  ret void
}

define void @unknown_dispatch(ptr %callback) {
entry:
  call void %callback()
  ret void
}

define void @metadata_spoof() {
entry:
  call void @wrong_target(), !loom.constant_callback_probe !0
  ret void
}

define i32 @main() {
entry:
  call void @dispatch(ptr @target)
  ret i32 0
}

!0 = !{i64 1}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<constant-callback>");
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
parseUndefBoundaryModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define void @undef_store(ptr %output) {
entry:
  store i32 undef, ptr %output, align 4
  ret void
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<undef-boundary-owner>");
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
parsePointerInductionModule(const char *test, llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define void @pointer_induction(ptr %a, ptr %b, ptr %c, i32 %count) {
entry:
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop

loop:
  %remaining = phi i32 [ %count, %entry ], [ %next_remaining, %loop ]
  %pa = phi ptr [ %a, %entry ], [ %next_a, %loop ]
  %pa_alias = phi ptr [ %a, %entry ], [ %next_a, %loop ]
  %pb = phi ptr [ %b, %entry ], [ %next_b, %loop ]
  %pc = phi ptr [ %c, %entry ], [ %next_c, %loop ]
  %current_a = getelementptr inbounds i8, ptr %pa_alias, i64 4
  %lhs = load float, ptr %current_a, align 4
  %rhs = load float, ptr %pb, align 4
  %sum = fadd float %lhs, %rhs
  store float %sum, ptr %pc, align 4
  %next_a = getelementptr inbounds i8, ptr %pa, i64 8
  %next_b = getelementptr inbounds i8, ptr %pb, i64 4
  %next_c = getelementptr inbounds i8, ptr %pc, i64 4
  %next_remaining = add i32 %remaining, -1
  %more = icmp ne i32 %next_remaining, 0
  br i1 %more, label %loop, label %exit

exit:
  ret void
}

define void @runtime_stride_pointer_induction(ptr %base, i32 %count,
                                               i32 %stride) {
entry:
  %empty = icmp eq i32 %count, 0
  %wide_stride = sext i32 %stride to i64
  br i1 %empty, label %exit, label %loop

loop:
  %remaining = phi i32 [ %count, %entry ], [ %next_remaining, %loop ]
  %cursor = phi ptr [ %base, %entry ], [ %next_cursor, %loop ]
  %value = load i8, ptr %cursor, align 1
  store i8 %value, ptr %cursor, align 1
  %next_cursor = getelementptr inbounds i8, ptr %cursor, i64 %wide_stride
  %next_remaining = add i32 %remaining, -1
  %more = icmp ne i32 %next_remaining, 0
  br i1 %more, label %loop, label %exit

exit:
  ret void
}

define void @bounded_pointer_induction(ptr %input, ptr %output, i32 %count) {
entry:
  %has_work = icmp sgt i32 %count, 0
  br i1 %has_work, label %loop, label %exit

loop:
  %index = phi i32 [ 0, %entry ], [ %next_index, %loop ]
  %input_cursor = phi ptr [ %input, %entry ], [ %next_input, %loop ]
  %output_cursor = phi ptr [ %output, %entry ], [ %next_output, %loop ]
  %value = load i16, ptr %input_cursor, align 2
  store i16 %value, ptr %output_cursor, align 2
  %next_input = getelementptr inbounds i8, ptr %input_cursor, i64 2
  %next_output = getelementptr inbounds i8, ptr %output_cursor, i64 2
  %next_index = add nuw nsw i32 %index, 1
  %more = icmp ne i32 %next_index, %count
  br i1 %more, label %loop, label %exit

exit:
  ret void
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer =
      llvm::MemoryBuffer::getMemBuffer(source, "<pointer-induction-owner>");
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
parseNestedPointerInductionModule(const char *test,
                                  llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define void @nested_pointer_induction(ptr %outer_base, i32 %rows,
                                      i32 %columns) {
entry:
  %empty_rows = icmp eq i32 %rows, 0
  %empty_columns = icmp eq i32 %columns, 0
  %empty = or i1 %empty_rows, %empty_columns
  br i1 %empty, label %exit, label %outer

outer:
  %rows_left = phi i32 [ %rows, %entry ], [ %next_rows, %outer_latch ]
  %row = phi ptr [ %outer_base, %entry ], [ %next_row, %outer_latch ]
  %row_value = load i32, ptr %row, align 4
  store i32 %row_value, ptr %row, align 4
  br label %inner

inner:
  %columns_left = phi i32 [ %columns, %outer ], [ %next_columns, %inner ]
  %element = phi ptr [ %row, %outer ], [ %next_element, %inner ]
  %value = load i32, ptr %element, align 4
  store i32 %value, ptr %element, align 4
  %next_element = getelementptr inbounds i8, ptr %element, i64 4
  %next_columns = add i32 %columns_left, -1
  %more_columns = icmp ne i32 %next_columns, 0
  br i1 %more_columns, label %inner, label %outer_latch

outer_latch:
  %next_row = getelementptr inbounds i8, ptr %row, i64 64
  %next_rows = add i32 %rows_left, -1
  %more_rows = icmp ne i32 %next_rows, 0
  br i1 %more_rows, label %outer, label %exit

exit:
  ret void
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(
      source, "<nested-pointer-induction-owner>");
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
  mlir::MLIRContext actorContext(mlir::MLIRContext::Threading::DISABLED);
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

void constantCallbackIsMechanicallyDevirtualized() {
  const char *test = "constantCallbackIsMechanicallyDevirtualized";
  llvm::LLVMContext context;
  auto structured =
      take(test, loom::raising::raiseLlvmModuleToStructuredProgram(
                     parseConstantCallbackModule(test, context)));
  auto dispatch =
      structured.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>("dispatch");
  if (!dispatch)
    fail(test, "mechanical raising lost the callback dispatcher");

  bool sawDirectTarget = false;
  bool sawIndirectCall = false;
  dispatch.walk([&](mlir::LLVM::CallOp call) {
    if (!call.getCalleeAttr()) {
      sawIndirectCall = true;
      return;
    }
    sawDirectTarget |= call.getCalleeAttr().getValue() == "target";
  });
  if (!sawDirectTarget || sawIndirectCall)
    fail(test, "constant callback did not become one exact direct call");

  auto unknown = structured.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      "unknown_dispatch");
  if (!unknown)
    fail(test, "mechanical raising lost the unknown callback dispatcher");
  bool retainedIndirectCall = false;
  unknown.walk([&](mlir::LLVM::CallOp call) {
    retainedIndirectCall |= !call.getCalleeAttr();
  });
  if (!retainedIndirectCall)
    fail(test, "input metadata spoofed an exact callback proof");
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
      test, loom::frontend::materializeSpatialOwnership(
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
      test, loom::frontend::materializeSpatialOwnership(
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

void wholeCallableExternalizesUndefValue() {
  const char *test = "wholeCallableExternalizesUndefValue";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-undef-boundary-owner", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseUndefBoundaryModule(test, context),
                                 design.roots().front().reference(), store));
  auto selected = take(
      test, loom::frontend::materializeSpatialOwnership(
                compiled.structuredProgram,
                findCallable(test, compiled.structuredProgram, "undef_store"),
                design.roots().front()));
  auto view = take(test, selected.canonicalDataflow.view());
  if (view.graphs().size() != 1)
    fail(test, "undef boundary did not produce one graph");
  auto graph = mlir::cast<dataflow::GraphOp>(view.graphs().front().op);
  llvm::ArrayRef<std::int32_t> segments = graph.getInputSegmentSizes();
  if (segments.size() != 3 || segments[0] != 1 || segments[2] != 1)
    fail(test, "undef was not externalized as one graph value input");
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (actor.op->getName().getStringRef() == "llvm.mlir.undef")
      fail(test, "undef remained a canonical actor");

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

  loom::frontend::SpatialOwnershipOptions options;
  options.fmuladdExecutionShape = loom::raising::FMulAddExecutionShape::Fused;
  auto selected = take(test, loom::frontend::materializeSpatialOwnership(
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
  auto candidate = loom::frontend::materializeSpatialOwnership(
      compiled.structuredProgram,
      findCallable(test, compiled.structuredProgram, "kernel"),
      design.roots().front());
  if (candidate)
    fail(test, "whole-callable ownership silently selected an index width");
  std::string message = llvm::toString(candidate.takeError());
  if (message.find("explicit canonical index width") == std::string::npos)
    fail(test, "missing index decision was not diagnosed: " + message);

  loom::frontend::SpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto selected =
      take(test, loom::frontend::materializeSpatialOwnership(
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
           loom::frontend::materializeSpatialOwnership(
               compiled.structuredProgram,
               findCallable(test, compiled.structuredProgram, "unsigned_index"),
               design.roots().front(), options));
  auto unsignedView = take(test, unsignedIndex.canonicalDataflow.view());
  if (unsignedView.graphs().size() != 1 || unsignedView.actors().empty())
    fail(test, "proven nonnegative extended index did not publish its graph");

  auto unprovenUnsigned = loom::frontend::materializeSpatialOwnership(
      compiled.structuredProgram,
      findCallable(test, compiled.structuredProgram, "unsigned_may_not_fit"),
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

void wholeCallableNormalizesPointerInduction() {
  const char *test = "wholeCallableNormalizesPointerInduction";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-pointer-induction", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parsePointerInductionModule(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::StructuredEntityRef callable =
      findCallable(test, compiled.structuredProgram, "pointer_induction");
  auto decisions =
      take(test, loom::frontend::enumerateSpatialOwnershipDecisionDomain(
                     compiled.structuredProgram, callable));
  bool saw64BitAddressDomain = false;
  for (const auto &decision : decisions)
    saw64BitAddressDomain |= decision.canonicalIndexWidth == 64;
  if (!saw64BitAddressDomain)
    fail(test, "pointer induction did not request a canonical address domain");

  loom::frontend::SpatialOwnershipOptions narrowOptions;
  narrowOptions.canonicalIndexWidth = 32;
  auto narrow = loom::frontend::materializeSpatialOwnership(
      compiled.structuredProgram, callable, design.roots().front(),
      narrowOptions);
  if (narrow)
    fail(test, "insufficient pointer induction width was accepted");
  std::string message = llvm::toString(narrow.takeError());
  if (message.find("pointer induction offset") == std::string::npos)
    fail(test,
         "insufficient pointer induction width was misdiagnosed: " + message);

  loom::frontend::SpatialOwnershipOptions options;
  options.canonicalIndexWidth = 64;
  auto selected = take(test, loom::frontend::materializeSpatialOwnership(
                                 compiled.structuredProgram, callable,
                                 design.roots().front(), options));
  auto view = take(test, selected.canonicalDataflow.view());
  bool sawLoad = false;
  bool sawStore = false;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    auto projection =
        take(test, dataflow::projectRegisteredActorSchemaProjection(actor.op));
    sawLoad |= projection.schema == dataflow::OperationSchemaId::DataflowLoad;
    sawStore |= projection.schema == dataflow::OperationSchemaId::DataflowStore;
    if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(actor.op))
      if (dataflow::DataflowDialect::containsMemoryCapability(
              carry.getOutput().getType()))
        fail(test, "pointer induction became dynamic memory carry state");
  }
  if (!sawLoad || !sawStore)
    fail(test, "normalized pointer induction lost its memory transactions");

  auto runtimeStride =
      take(test, loom::frontend::materializeSpatialOwnership(
                     compiled.structuredProgram,
                     findCallable(test, compiled.structuredProgram,
                                  "runtime_stride_pointer_induction"),
                     design.roots().front(), options));
  auto runtimeStrideView = take(test, runtimeStride.canonicalDataflow.view());
  if (runtimeStrideView.graphs().size() != 1 ||
      runtimeStrideView.actors().empty())
    fail(test, "loop-invariant runtime stride did not publish its graph");
  for (const dataflow::CanonicalActorView &actor : runtimeStrideView.actors())
    if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(actor.op))
      if (dataflow::DataflowDialect::containsMemoryCapability(
              carry.getOutput().getType()))
        fail(test, "runtime pointer stride became memory carry state");

  auto bounded = take(test, loom::frontend::materializeSpatialOwnership(
                                compiled.structuredProgram,
                                findCallable(test, compiled.structuredProgram,
                                             "bounded_pointer_induction"),
                                design.roots().front(), options));
  auto boundedView = take(test, bounded.canonicalDataflow.view());
  if (boundedView.graphs().size() != 1 || boundedView.actors().empty())
    fail(test, "runtime-bounded pointer induction did not publish its graph");
  for (const dataflow::CanonicalActorView &actor : boundedView.actors())
    if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(actor.op))
      if (dataflow::DataflowDialect::containsMemoryCapability(
              carry.getOutput().getType()))
        fail(test, "runtime-bounded pointer induction retained pointer state");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void wholeCallableNormalizesNestedPointerInduction() {
  const char *test = "wholeCallableNormalizesNestedPointerInduction";
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-nested-pointer-induction", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled =
      take(test, loom::frontend::compileLlvmModuleToPreMapping(
                     parseNestedPointerInductionModule(test, context),
                     design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipOptions options;
  options.canonicalIndexWidth = 64;
  auto selected = take(test, loom::frontend::materializeSpatialOwnership(
                                 compiled.structuredProgram,
                                 findCallable(test, compiled.structuredProgram,
                                              "nested_pointer_induction"),
                                 design.roots().front(), options));
  auto view = take(test, selected.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().empty())
    fail(test, "nested pointer induction did not publish its graph");
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (auto carry = llvm::dyn_cast<dataflow::CarryOp>(actor.op))
      if (dataflow::DataflowDialect::containsMemoryCapability(
              carry.getOutput().getType()))
        fail(test, "nested pointer induction retained pointer carry state");

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
  auto implicit = loom::frontend::materializeSpatialOwnership(
      compiled.structuredProgram, loop, design.roots().front());
  if (implicit)
    fail(test, "operation ownership silently selected an index width");
  std::string implicitMessage = llvm::toString(implicit.takeError());
  if (implicitMessage.find("explicit canonical index width") ==
      std::string::npos)
    fail(test, "missing index decision was not diagnosed: " + implicitMessage);

  loom::frontend::SpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto selected = take(test, loom::frontend::materializeSpatialOwnership(
                                 compiled.structuredProgram, loop,
                                 design.roots().front(), options));

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

  auto unproven = loom::frontend::materializeSpatialOwnership(
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
  loom::frontend::SpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto selected = take(
      test,
      loom::frontend::materializeSpatialOwnership(
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
  auto domain = take(test, loom::frontend::enumerateSpatialOwnershipScopeDomain(
                               compiled.structuredProgram));
  auto view = take(test, compiled.structuredProgram.view());

  std::vector<loom::frontend::StructuredEntityRef> scopes;
  std::vector<std::uint64_t> scopeOrdinals;
  for (auto [domainOrdinal, entry] : llvm::enumerate(domain)) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    auto entity = take(test, view.resolve(scope->selection));
    if (llvm::isa_and_nonnull<mlir::scf::WhileOp>(entity.operation)) {
      scopes.push_back(scope->selection);
      scopeOrdinals.push_back(domainOrdinal);
    }
  }

  std::vector<loom::frontend::StructuredEntityRef> expected;
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    if (llvm::isa_and_nonnull<mlir::scf::WhileOp>(entity.operation))
      expected.push_back(entity.reference);
  }
  if (expected.size() != 2 || scopes != expected)
    fail(test, "ownership scopes do not follow canonical operation order");
  for (auto [scope, domainOrdinal] : llvm::zip_equal(scopes, scopeOrdinals)) {
    if (scope.parent != compiled.structuredProgram.identity() ||
        scope.kind != loom::frontend::StructuredEntityKind::Operation)
      fail(test, "ownership scope is not parent-local operation identity");
    std::optional<std::uint64_t> parentOrdinal =
        domain.parentScopeOrdinal(domainOrdinal);
    if (!parentOrdinal)
      fail(test, "nested ownership scope has no parent scope");
    const auto *parentScope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(
            &domain[*parentOrdinal]);
    if (!parentScope)
      fail(test, "nested ownership scope parent is not materializable");
    auto parentEntity = take(test, view.resolve(parentScope->selection));
    if (!llvm::isa_and_nonnull<mlir::LLVM::LLVMFuncOp>(parentEntity.operation))
      fail(test, "loop ownership scope is not parented by its callable");
  }

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
  auto domain = take(test, loom::frontend::enumerateSpatialOwnershipScopeDomain(
                               compiled.structuredProgram));
  auto view = take(test, compiled.structuredProgram.view());
  std::vector<loom::frontend::StructuredEntityRef> scopes;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : domain) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    auto entity = take(test, view.resolve(scope->selection));
    if (llvm::isa_and_nonnull<mlir::LLVM::LLVMFuncOp>(entity.operation))
      scopes.push_back(scope->selection);
  }
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
  loom::frontend::SpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  options.fmuladdExecutionShape = loom::raising::FMulAddExecutionShape::Fused;
  auto selected = take(
      test,
      loom::frontend::materializeSpatialOwnership(
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
  auto domain = take(test, loom::frontend::enumerateSpatialOwnershipScopeDomain(
                               compiled.structuredProgram));
  const loom::frontend::StructuredEntityRef callable =
      findCallable(test, compiled.structuredProgram, "fmuladd_loop");
  const loom::frontend::StructuredEntityRef loop =
      findStructuredLoop(test, compiled.structuredProgram, "fmuladd_loop");

  std::optional<loom::frontend::SpatialOwnershipScope> callableScope;
  std::optional<loom::frontend::SpatialOwnershipScope> operationScope;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : domain) {
    const auto *scope =
        std::get_if<loom::frontend::SpatialOwnershipScope>(&entry);
    if (!scope)
      continue;
    if (scope->selection == callable)
      callableScope = *scope;
    if (scope->selection == loop)
      operationScope = *scope;
  }
  if (!callableScope)
    fail(test, "unified domain omitted the whole-callable ownership scope");
  if (!operationScope)
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

void operationSpatialOwnershipExternalizesEscapedResult() {
  const char *test = "operationSpatialOwnershipExternalizesEscapedResult";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-operation-liveout", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseEscapedLoopModule(test, context),
                                 design.roots().front().reference(), store));
  loom::frontend::SpatialOwnershipOptions options;
  options.canonicalIndexWidth = 32;
  auto selected = take(
      test, loom::frontend::materializeSpatialOwnership(
                compiled.structuredProgram,
                findStructuredLoop(test, compiled.structuredProgram, "accum"),
                design.roots().front(), options));

  auto wrapper =
      selected.structuredProgram.module().lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "accum");
  if (!wrapper || wrapper.getFunctionType().getReturnType() !=
                      mlir::IntegerType::get(wrapper.getContext(), 32))
    fail(test, "ownership materialization changed the LLVM result ABI");

  dataflow::ThreadLaunchOp launch;
  dataflow::ThreadWaitOp wait;
  mlir::LLVM::LoadOp resultLoad;
  mlir::LLVM::ReturnOp resultReturn;
  wrapper.walk([&](mlir::Operation *operation) {
    if (auto candidate = llvm::dyn_cast<dataflow::ThreadLaunchOp>(operation))
      launch = candidate;
    else if (auto candidate = llvm::dyn_cast<dataflow::ThreadWaitOp>(operation))
      wait = candidate;
    else if (auto candidate = llvm::dyn_cast<mlir::LLVM::LoadOp>(operation))
      resultLoad = candidate;
    else if (auto candidate = llvm::dyn_cast<mlir::LLVM::ReturnOp>(operation))
      resultReturn = candidate;
  });
  if (!launch || !wait || !resultLoad || !resultReturn ||
      resultReturn.getNumOperands() != 1 ||
      resultReturn.getOperand(0) != resultLoad.getResult())
    fail(test, "escaped result was not loaded after thread completion");

  dataflow::ThreadOp thread;
  selected.structuredProgram.module().walk(
      [&](dataflow::ThreadOp candidate) { thread = candidate; });
  if (!thread || !thread.getFunctionType().getResults().empty())
    fail(test, "escaped result invented a thread data result");
  bool storedResult = false;
  thread.walk([&](mlir::LLVM::StoreOp store) {
    storedResult |= store.getValue().getType().isInteger(32);
  });
  if (!storedResult)
    fail(test,
         "thread did not publish the escaped value through its result slot");

  auto view = take(test, selected.canonicalDataflow.view());
  auto graph = view.graphs().size() == 1
                   ? llvm::dyn_cast<dataflow::GraphOp>(view.graphs().front().op)
                   : dataflow::GraphOp{};
  if (!graph || graph.getFunctionType().getNumResults() != 1 ||
      !graph.getFunctionType().getResult(0).isInteger(32))
    fail(test, "canonical graph did not retain the selected i32 result");

  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

} // namespace

int main() {
  exactFabricAndWholeProgramDataflow();
  constantCallbackIsMechanicallyDevirtualized();
  explicitWholeCallableSpatialOwnership();
  wholeCallableExternalizesGlobalMemoryCapability();
  wholeCallableExternalizesUndefValue();
  explicitFmulAddExecutionShape();
  wholeCallableRequiresCanonicalAddressIndexDecision();
  wholeCallableNormalizesPointerInduction();
  wholeCallableNormalizesNestedPointerInduction();
  explicitOperationSpatialOwnership();
  operationOwnershipInternalizesConstants();
  wholeCallableScopesFollowCanonicalOrder();
  operationOwnershipScopesFollowCanonicalOrder();
  operationFmulAddDecisionIsCandidateLocal();
  ownershipDecisionDomainIsScopeLocalAndTyped();
  unifiedOwnershipDomainMaterializesExplicitDecision();
  operationSpatialOwnershipExternalizesEscapedResult();
  llvm::outs() << "pre-Mapping compilation anchor passed\n";
  return EXIT_SUCCESS;
}
