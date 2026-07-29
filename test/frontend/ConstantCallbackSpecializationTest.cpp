#include "Frontend/Raising/StructuredRaising.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <string>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "constant callback specialization anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<llvm::Module> parseModule(llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

define internal void @target_a() {
entry:
  ret void
}

define internal void @target_b() {
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

define i32 @main() {
entry:
  call void @dispatch(ptr @target_a)
  call void @dispatch(ptr @target_b)
  ret i32 0
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<callback-sites>");
  std::unique_ptr<llvm::Module> module =
      llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print("constant-callback-specialization", stream);
    fail(stream.str());
  }
  return module;
}

bool hasIndirectCall(mlir::LLVM::LLVMFuncOp function) {
  bool indirect = false;
  function.walk(
      [&](mlir::LLVM::CallOp call) { indirect |= !call.getCalleeAttr(); });
  return indirect;
}

void exactCallSitesUseDistinctSpecializations() {
  llvm::LLVMContext context;
  auto structured = take(
      loom::raising::raiseLlvmModuleToStructuredProgram(parseModule(context)));
  mlir::ModuleOp module = structured.module();
  auto main = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main");
  auto dispatch = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("dispatch");
  auto unknown =
      module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("unknown_dispatch");
  if (!main || !dispatch || !unknown)
    fail("raising lost a source callable");

  llvm::SmallVector<std::string, 2> specializedCallees;
  main.walk([&](mlir::LLVM::CallOp call) {
    if (mlir::FlatSymbolRefAttr callee = call.getCalleeAttr())
      specializedCallees.push_back(callee.getValue().str());
  });
  if (specializedCallees.size() != 2 ||
      specializedCallees[0] == specializedCallees[1] ||
      specializedCallees[0] == "dispatch" ||
      specializedCallees[1] == "dispatch")
    fail("constant callback call sites did not select distinct clones");

  constexpr llvm::StringLiteral expectedTargets[] = {"target_a", "target_b"};
  for (auto [ordinal, calleeName] : llvm::enumerate(specializedCallees)) {
    auto clone = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(calleeName);
    if (!clone)
      fail("specialized dispatcher symbol does not resolve");
    llvm::SmallVector<std::string, 2> targets;
    bool indirect = false;
    clone.walk([&](mlir::LLVM::CallOp call) {
      if (mlir::FlatSymbolRefAttr callee = call.getCalleeAttr())
        targets.push_back(callee.getValue().str());
      else
        indirect = true;
    });
    if (indirect || targets.size() != 1 ||
        targets.front() != expectedTargets[ordinal])
      fail("specialized dispatcher does not call its exact target");
  }

  if (!hasIndirectCall(dispatch) || !hasIndirectCall(unknown))
    fail("specialization rewrote an unresolved dispatcher globally");
}

} // namespace

int main() {
  exactCallSitesUseDistinctSpecializations();
  llvm::outs() << "constant callback specialization anchor passed\n";
  return EXIT_SUCCESS;
}
