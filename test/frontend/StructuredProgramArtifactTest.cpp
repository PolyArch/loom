#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

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
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
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
  func.func @entry(%arg0: i32) -> i32 {
    %result = call @different_name(%arg0) : (i32) -> i32
    return %result : i32
  }
  func.func private @different_name(%arg0: i32) -> i32 {
    %value = arith.addi %arg0, %arg0 : i32 loc("second")
    return %value : i32
  }
}
)mlir";
  StructuredProgramCandidate a = finalize(test, first);
  StructuredProgramCandidate b = finalize(test, second);
  require(
      test, a.identity() == b.identity(),
      "private symbol spelling, order, or source location changed candidate "
      "identity");
}

void loopDebugLocationsDoNotChangeIdentity() {
  const char *test = __func__;
  StructuredProgramCandidate first = finalize(test, R"mlir(
#first_begin = loc("first.c":11:3)
#first_end = loc("first.c":19:5)
#first_start = loc(fused[#first_begin, #first_end])
#first_finish = loc(fused[#first_end, #first_begin])
#loop = #llvm.loop_annotation<mustProgress = true, startLoc = #first_start, endLoc = #first_finish>
module {
  llvm.func @entry() {
    llvm.br ^bb1 {loop_annotation = #loop}
  ^bb1:
    llvm.return
  }
}
)mlir");
  StructuredProgramCandidate second = finalize(test, R"mlir(
#second_begin = loc("second.c":101:7)
#second_end = loc("second.c":137:9)
#second_start = loc(fused[#second_begin, #second_end])
#second_finish = loc(fused[#second_end, #second_begin])
#loop = #llvm.loop_annotation<mustProgress = true, startLoc = #second_start, endLoc = #second_finish>
module {
  llvm.func @entry() {
    llvm.br ^bb1 {loop_annotation = #loop}
  ^bb1:
    llvm.return
  }
}
)mlir");
  require(test, first.identity() == second.identity(),
          "LLVM loop debug locations changed candidate identity");

  mlir::LLVM::LoopAnnotationAttr projected;
  first.module().walk([&](mlir::Operation *operation) {
    auto annotation = operation->getAttrOfType<mlir::LLVM::LoopAnnotationAttr>(
        "loop_annotation");
    if (annotation)
      projected = annotation;
  });
  require(test,
          projected && projected.getMustProgress() &&
              projected.getMustProgress().getValue(),
          "loop debug projection removed the semantic must-progress contract");
  require(test, !projected.getStartLoc() && !projected.getEndLoc(),
          "loop debug projection retained source locations");
}

void finalizationProjectsSourceProvenanceOutsideIdentity() {
  const char *test = __func__;
  auto module = parse(test, R"mlir(
#callee = loc("kernel.c":11:7)
#caller = loc("wrapper.c":19:3)
#inlined = loc(callsite(#callee at #caller))
module {
  func.func @entry(%arg0: i32) -> i32 {
    %sum = arith.addi %arg0, %arg0 : i32 loc(#inlined)
    %product = arith.muli %sum, %arg0 : i32 loc("wrapper.c":19:3)
    return %product : i32
  }
}
)mlir");
  mlir::Operation *inlinedOperation = nullptr;
  mlir::Operation *callerOnlyOperation = nullptr;
  module->walk([&](mlir::arith::AddIOp operation) {
    inlinedOperation = operation.getOperation();
  });
  module->walk([&](mlir::arith::MulIOp operation) {
    callerOnlyOperation = operation.getOperation();
  });
  require(test, inlinedOperation && callerOnlyOperation,
          "source operation is missing");
  mlir::Location inlinedLocation = inlinedOperation->getLoc();
  mlir::Attribute debugFile =
      mlir::LLVM::DIFileAttr::get(module->getContext(), "kernel.c", "/src");
  inlinedOperation->setLoc(
      mlir::FusedLoc::get(module->getContext(), {inlinedLocation}, debugFile));
  callerOnlyOperation->setLoc(mlir::CallSiteLoc::get(
      mlir::UnknownLoc::get(module->getContext()),
      mlir::FileLineColLoc::get(module->getContext(), "caller-only.c", 23, 5)));

  auto finalized =
      take(test, loom::frontend::finalizeStructuredProgramWithTrackedBlocks(
                     module.get(), {}));
  bool foundKernel = false;
  bool foundDebugKernel = false;
  bool foundWrapper = false;
  bool foundCallerOnly = false;
  bool foundInlineCall = false;
  bool inventedCallerOnlyLineage = false;
  for (const auto &provenance : finalized.sourceProvenance) {
    require(test, provenance.operation.parent == finalized.artifact.identity(),
            "source provenance is not keyed by the finalized artifact");
    require(test, provenance.operation.kind == StructuredEntityKind::Operation,
            "source provenance is not keyed by operation references");
    foundKernel |= llvm::is_contained(provenance.sourceFiles, "kernel.c");
    foundDebugKernel |=
        llvm::is_contained(provenance.sourceFiles, "/src/kernel.c");
    foundWrapper |= llvm::is_contained(provenance.sourceFiles, "wrapper.c");
    foundCallerOnly |=
        llvm::is_contained(provenance.sourceFiles, "caller-only.c");
    foundInlineCall |=
        llvm::any_of(provenance.callLineages, [](const auto &lineage) {
          return lineage.frames.size() == 2 &&
                 lineage.frames[0].sourceFile == "kernel.c" &&
                 lineage.frames[0].position ==
                     loom::frontend::StructuredSourcePosition{11, 7} &&
                 lineage.frames[1].sourceFile == "wrapper.c" &&
                 lineage.frames[1].position ==
                     loom::frontend::StructuredSourcePosition{19, 3};
        });
    inventedCallerOnlyLineage |=
        llvm::any_of(provenance.callLineages, [](const auto &lineage) {
          return !lineage.frames.empty() &&
                 lineage.frames.front().sourceFile == "caller-only.c";
        });
  }
  require(test,
          foundKernel && foundDebugKernel && foundWrapper && foundCallerOnly,
          "source provenance did not preserve canonical source paths");
  require(test, foundInlineCall,
          "source provenance did not preserve inline call lineage");
  require(test, !inventedCallerOnlyLineage,
          "unknown operation source was replaced by its caller location");

  mlir::IRMapping mapping;
  auto clone =
      take(test, loom::frontend::cloneStructuredProgramWithSourceLocations(
                     finalized.artifact, finalized.sourceProvenance, mapping));
  auto refinalized =
      take(test, loom::frontend::finalizeStructuredProgramWithTrackedBlocks(
                     clone.get(), {}));
  bool restoredInlineCall =
      llvm::any_of(refinalized.sourceProvenance, [](const auto &provenance) {
        return llvm::any_of(provenance.callLineages, [](const auto &lineage) {
          return lineage.frames.size() == 2 &&
                 lineage.frames[0].sourceFile == "kernel.c" &&
                 lineage.frames[0].position ==
                     loom::frontend::StructuredSourcePosition{11, 7} &&
                 lineage.frames[1].sourceFile == "wrapper.c" &&
                 lineage.frames[1].position ==
                     loom::frontend::StructuredSourcePosition{19, 3};
        });
      });
  require(test, restoredInlineCall,
          "Structured cloning did not restore inline call lineage");

  std::vector<loom::frontend::StructuredOperationSourceProvenance>
      inventoryOnly = finalized.sourceProvenance;
  for (auto &provenance : inventoryOnly)
    provenance.callLineages.clear();
  mlir::IRMapping inventoryMapping;
  auto inventoryClone =
      take(test, loom::frontend::cloneStructuredProgramWithSourceLocations(
                     finalized.artifact, inventoryOnly, inventoryMapping));
  auto inventoryRefinalized =
      take(test, loom::frontend::finalizeStructuredProgramWithTrackedBlocks(
                     inventoryClone.get(), {}));
  require(test,
          llvm::all_of(inventoryRefinalized.sourceProvenance,
                       [](const auto &provenance) {
                         return provenance.callLineages.empty();
                       }),
          "file-only provenance invented an exact source position");
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

void addressProjectionChangesIdentity() {
  const char *test = __func__;
  const char *source = R"mlir(
module {
  llvm.func @entry(%pointer: !llvm.ptr) {
    %value = llvm.mlir.constant(7 : i32) : i32
    llvm.store %value, %pointer : i32, !llvm.ptr
    llvm.return
  }
}
)mlir";
  auto pointerModule = parse(test, source);
  auto rootRelativeModule = parse(test, source);
  rootRelativeModule->walk([](mlir::LLVM::StoreOp store) {
    store->setAttr(loom::rootRelativeAddressAttrName,
                   mlir::UnitAttr::get(store.getContext()));
  });
  StructuredProgramCandidate pointer = take(
      test, loom::frontend::finalizeStructuredProgram(pointerModule.get()));
  StructuredProgramCandidate rootRelative =
      take(test,
           loom::frontend::finalizeStructuredProgram(rootRelativeModule.get()));
  require(test, pointer.identity() != rootRelative.identity(),
          "address projection marker did not change candidate identity");

  auto malformed = parse(test, source);
  malformed->walk([](mlir::LLVM::LLVMFuncOp function) {
    function->setAttr(loom::rootRelativeAddressAttrName,
                      mlir::UnitAttr::get(function.getContext()));
  });
  auto rejected = loom::frontend::finalizeStructuredProgram(malformed.get());
  require(test, !rejected,
          "root-relative address marker was accepted on a non-memory op");
  llvm::consumeError(rejected.takeError());
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

  auto finalizedView = take(test, candidate.view());
  auto importedView = take(test, imported.view());
  constexpr std::array kinds{
      StructuredEntityKind::Operation, StructuredEntityKind::Region,
      StructuredEntityKind::Block, StructuredEntityKind::Value};
  for (StructuredEntityKind kind : kinds) {
    auto finalizedEntities = finalizedView.entities(kind);
    auto importedEntities = importedView.entities(kind);
    require(test, finalizedEntities.size() == importedEntities.size(),
            "strict import changed a canonical entity domain");
    for (auto pair : llvm::zip(finalizedEntities, importedEntities)) {
      const auto &finalized = std::get<0>(pair);
      const auto &reimported = std::get<1>(pair);
      require(test, finalized.reference == reimported.reference,
              "strict import changed a canonical entity reference");
      require(test,
              static_cast<bool>(finalized.operation) ==
                      static_cast<bool>(reimported.operation) &&
                  static_cast<bool>(finalized.region) ==
                      static_cast<bool>(reimported.region) &&
                  static_cast<bool>(finalized.block) ==
                      static_cast<bool>(reimported.block) &&
                  static_cast<bool>(finalized.value) ==
                      static_cast<bool>(reimported.value),
              "strict import changed a canonical entity carrier");
      if (finalized.operation)
        require(test,
                finalized.operation->getName().getStringRef() ==
                    reimported.operation->getName().getStringRef(),
                "strict import changed a canonical operation entity");
      if (finalized.value) {
        std::string finalizedType;
        std::string reimportedType;
        llvm::raw_string_ostream(finalizedType) << finalized.value.getType();
        llvm::raw_string_ostream(reimportedType) << reimported.value.getType();
        require(test, finalizedType == reimportedType,
                "strict import changed a canonical value entity");
      }
    }
  }
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

void finalizedArtifactsDoNotOwnThreadPools() {
  const char *test = __func__;
  StructuredProgramCandidate structured = finalize(test, R"mlir(
module { func.func @main(%arg0: i32) -> i32 { return %arg0 : i32 } }
)mlir");
  require(test, structured.module().getContext()->getNumThreads() == 1,
          "a finalized Structured Program owns an implicit thread pool");

  auto dataflow = take(
      test,
      loom::lowering::lowerStructuredProgramToCanonicalDataflow(structured));
  require(test, dataflow.module().getContext()->getNumThreads() == 1,
          "a finalized Canonical Dataflow Program owns an implicit thread "
          "pool");
}

void moveAssignmentKeepsContextAliveUntilModuleRelease() {
  const char *test = __func__;
  StructuredProgramCandidate replaced = finalize(test, R"mlir(
module { func.func @old(%arg0: i32) -> i32 { return %arg0 : i32 } }
)mlir");
  StructuredProgramCandidate replacement = finalize(test, R"mlir(
module { func.func @new(%arg0: i32) -> i32 { return %arg0 : i32 } }
)mlir");
  const loom::ArtifactIdentity replacementIdentity = replacement.identity();

  replaced = std::move(replacement);

  require(test, replaced.identity() == replacementIdentity,
          "move assignment did not retain the replacement identity");
  require(test, replaced.module().lookupSymbol<mlir::func::FuncOp>("new"),
          "move assignment did not retain the replacement module");
}

} // namespace

int main() {
  privateNamesAndLocationsDoNotChangeIdentity();
  loopDebugLocationsDoNotChangeIdentity();
  finalizationProjectsSourceProvenanceOutsideIdentity();
  semanticOperationChangesIdentity();
  addressProjectionChangesIdentity();
  referencesAreParentAndKindChecked();
  importedCandidateReencodesExactly();
  graphFreeInstructionCoreProgramPublishesDataflowArtifact();
  finalizedArtifactsDoNotOwnThreadPools();
  moveAssignmentKeepsContextAliveUntilModuleRelease();
  llvm::outs() << "structured program artifact anchors passed\n";
  return EXIT_SUCCESS;
}
