#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

using loom::frontend::StructuredEntityKind;
using loom::frontend::StructuredProgramCandidate;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect>();
    auto *created = new mlir::MLIRContext(registry);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef test,
                                        llvm::StringRef source) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "cannot parse Structured Program fixture");
  return module;
}

StructuredProgramCandidate finalize(llvm::StringRef test,
                                    llvm::StringRef source) {
  auto module = parse(test, source);
  return take(test, loom::frontend::finalizeStructuredProgram(module.get()));
}

void privateNamesAndLocationsDoNotChangeIdentity() {
  const char *test = __func__;
  const char *first = R"mlir(
module {
  func.func private @one(%arg0: i32) -> i32 {
    %value = arith.addi %arg0, %arg0 : i32 loc("first")
    return %value : i32
  }
  func.func @entry(%arg0: i32) -> i32 {
    %result = call @one(%arg0) : (i32) -> i32
    return %result : i32
  }
}
)mlir";
  const char *second = R"mlir(
module {
  func.func private @different_name(%arg0: i32) -> i32 {
    %value = arith.addi %arg0, %arg0 : i32 loc("second")
    return %value : i32
  }
  func.func @entry(%arg0: i32) -> i32 {
    %result = call @different_name(%arg0) : (i32) -> i32
    return %result : i32
  }
}
)mlir";
  StructuredProgramCandidate a = finalize(test, first);
  StructuredProgramCandidate b = finalize(test, second);
  require(
      test, a.identity() == b.identity(),
      "private symbol spelling or source location changed candidate identity");
}

void semanticOperationChangesIdentity() {
  const char *test = __func__;
  StructuredProgramCandidate add = finalize(test, R"mlir(
module {
  func.func @entry(%arg0: i32) -> i32 {
    %result = arith.addi %arg0, %arg0 : i32
    return %result : i32
  }
}
)mlir");
  StructuredProgramCandidate multiply = finalize(test, R"mlir(
module {
  func.func @entry(%arg0: i32) -> i32 {
    %result = arith.muli %arg0, %arg0 : i32
    return %result : i32
  }
}
)mlir");
  require(test, add.identity() != multiply.identity(),
          "a semantic operation change did not change candidate identity");
}

void referencesAreParentAndKindChecked() {
  const char *test = __func__;
  StructuredProgramCandidate first = finalize(test, R"mlir(
module { func.func @entry(%arg0: i32) -> i32 { return %arg0 : i32 } }
)mlir");
  StructuredProgramCandidate second = finalize(test, R"mlir(
module {
  func.func @entry(%arg0: i32) -> i32 {
    %value = arith.addi %arg0, %arg0 : i32
    return %value : i32
  }
}
)mlir");
  auto firstView = take(test, first.view());
  auto operations = firstView.entities(StructuredEntityKind::Operation);
  require(test, !operations.empty(), "fixture has no operation entity");
  auto reference = operations.back().reference;
  reference.parent = second.identity();
  auto rejected = firstView.resolve(reference);
  require(test, !rejected,
          "a foreign StructuredEntityRef was accepted by its parent view");
  llvm::consumeError(rejected.takeError());
  reference.parent = first.identity();
  reference.kind = StructuredEntityKind::Value;
  rejected = firstView.resolve(reference);
  require(test, !rejected, "a wrong-kind StructuredEntityRef was accepted");
  llvm::consumeError(rejected.takeError());
}

void importedCandidateReencodesExactly() {
  const char *test = __func__;
  StructuredProgramCandidate candidate = finalize(test, R"mlir(
module { func.func @entry(%arg0: i32) -> i32 { return %arg0 : i32 } }
)mlir");
  StructuredProgramCandidate imported =
      take(test, loom::frontend::importStructuredProgram(
                     candidate.identity(), candidate.canonicalBytes()));
  require(test, imported.identity() == candidate.identity(),
          "strict import changed candidate identity");
  require(test,
          imported.canonicalBytes().bytes() ==
              candidate.canonicalBytes().bytes(),
          "strict import did not preserve canonical bytes");
}

void graphFreeInstructionCoreProgramPublishesDataflowArtifact() {
  const char *test = __func__;
  StructuredProgramCandidate structured = finalize(test, R"mlir(
module {
  func.func @main(%arg0: i32) -> i32 {
    %value = arith.addi %arg0, %arg0 : i32
    return %value : i32
  }
}
)mlir");
  auto dataflow = take(
      test,
      loom::lowering::lowerStructuredProgramToCanonicalDataflow(structured));
  require(test, dataflow.module().lookupSymbol<mlir::func::FuncOp>("main"),
          "graph-free Dataflow artifact lost the InstructionCore program");
  auto view = take(test, dataflow.view());
  require(test, view.graphs().empty(),
          "a graph-free candidate unexpectedly acquired a SpatialCore graph");
}

} // namespace

int main() {
  privateNamesAndLocationsDoNotChangeIdentity();
  semanticOperationChangesIdentity();
  referencesAreParentAndKindChecked();
  importedCandidateReencodesExactly();
  graphFreeInstructionCoreProgramPublishesDataflowArtifact();
  llvm::outs() << "structured program artifact anchors passed\n";
  return EXIT_SUCCESS;
}
