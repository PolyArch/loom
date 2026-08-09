#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredMemoryPipeline: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                    mlir::arith::ArithDialect, mlir::DLTIDialect,
                    mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

llvm::StringRef nativeDataLayout() {
  static std::string layout = [] {
    if (llvm::InitializeNativeTarget() ||
        llvm::InitializeNativeTargetAsmPrinter())
      fail("cannot initialize the native target");
    auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
    return take(target.getDefaultDataLayoutForTarget())
        .getStringRepresentation();
  }();
  return layout;
}

llvm::StringRef nativeTargetTriple() {
  static std::string triple = [] {
    auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
    return target.getTargetTriple().str();
  }();
  return triple;
}

loom::frontend::StructuredProgramCandidate
parsePipelineProgram(unsigned tripCount, llvm::StringRef outputIndex,
                     llvm::StringRef extraCompute = {}) {
  std::string source = R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @pipeline_source : memref<4x2xi32> =
      dense<[[1, 2], [3, 4], [5, 6], [7, 8]]>
  memref.global @pipeline_target : memref<4xi32> = dense<0>
  func.func private @opaque()

  dataflow.thread private @pipeline domain(#dataflow.thread_domain<dense>)(
      %source: memref<4x2xi32>, %target: memref<4xi32>)
      ctrl (%start: none) {
    "loom.spatial_region"(%source, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<4x2xi32>, %output: memref<4xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cub = arith.constant )mlir";
  source += std::to_string(tripCount);
  source += R"mlir( : index
        scf.for %i = %c0 to %cub step %c1 {
          %buffer = memref.alloc() : memref<1x2xi32>
          %view = memref.subview %input[%i, 0] [1, 2] [1, 1] :
              memref<4x2xi32> to memref<1x2xi32, strided<[2, 1], offset: ?>>
          memref.copy %view, %buffer :
              memref<1x2xi32, strided<[2, 1], offset: ?>> to memref<1x2xi32>
)mlir";
  source += extraCompute.str();
  source += R"mlir(
          %j0 = arith.constant 0 : index
          %j1 = arith.constant 1 : index
          %lhs = memref.load %buffer[%j0, %j0] : memref<1x2xi32>
          %rhs = memref.load %buffer[%j0, %j1] : memref<1x2xi32>
          %sum = arith.addi %lhs, %rhs : i32
          memref.store %sum, %output[)mlir";
  source += outputIndex.str();
  source += R"mlir(] : memref<4xi32>
          memref.dealloc %buffer : memref<1x2xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "pipeline_graph", source_maps = []} :
        (memref<4x2xi32>, memref<4xi32>) -> ()
    dataflow.thread.yield
  }

  llvm.func @entry() -> i32 {
    %source = memref.get_global @pipeline_source : memref<4x2xi32>
    %target = memref.get_global @pipeline_target : memref<4xi32>
    %token = dataflow.thread.launch @pipeline(%source, %target) :
        (memref<4x2xi32>, memref<4xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    %last = arith.constant )mlir";
  source += std::to_string(tripCount == 0 ? 0 : tripCount - 1);
  source += R"mlir( : index
    %value = memref.load %target[%last] : memref<4xi32>
    llvm.return %value : i32
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("cannot parse the pipeline fixture");
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), nativeDataLayout()));
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), nativeTargetTriple()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate
parseDirectSourceProgram(bool aliasAtLaunch) {
  std::string source = R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @direct_source : memref<2x2xi32> =
      dense<[[1, 2], [3, 4]]>
  memref.global @direct_target : memref<2x2xi32> = dense<0>

  dataflow.thread private @direct_pipeline
      domain(#dataflow.thread_domain<dense>)(
      %source: memref<2x2xi32>, %target: memref<2x2xi32>)
      ctrl (%start: none) {
    "loom.spatial_region"(%source, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<2x2xi32>, %output: memref<2x2xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %c0 to %c2 step %c1 {
          %buffer = memref.alloc() : memref<2x2xi32>
          memref.copy %input, %buffer :
              memref<2x2xi32> to memref<2x2xi32>
          %j0 = arith.constant 0 : index
          %value = memref.load %buffer[%i, %j0] : memref<2x2xi32>
          memref.store %value, %output[%i, %j0] : memref<2x2xi32>
          memref.dealloc %buffer : memref<2x2xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "direct_pipeline_graph", source_maps = []} :
        (memref<2x2xi32>, memref<2x2xi32>) -> ()
    dataflow.thread.yield
  }

  llvm.func @entry() {
    %source = memref.get_global @direct_source : memref<2x2xi32>
    %target = memref.get_global @direct_target : memref<2x2xi32>
    %token = dataflow.thread.launch @direct_pipeline(%source, )mlir";
  source += aliasAtLaunch ? "%source" : "%target";
  source += R"mlir() :
        (memref<2x2xi32>, memref<2x2xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    llvm.return
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("cannot parse the direct-source pipeline fixture");
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), nativeDataLayout()));
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), nativeTargetTriple()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

std::vector<loom::frontend::StructuredMemoryCommunicationDecision>
pipelineDecisions(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto domain =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          candidate, 64));
  std::vector<loom::frontend::StructuredMemoryCommunicationDecision> result;
  for (const auto &decision : domain.decisions)
    if (loom::frontend::structuredMemoryCommunicationDecisionKind(decision) ==
        loom::frontend::StructuredMemoryCommunicationDecisionKind::
            PipelineStagedLoop)
      result.push_back(decision);
  return result;
}

loom::sim::DFGSimulationReport
simulate(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto d0 = take(
      loom::lowering::lowerStructuredProgramToCanonicalDataflow(candidate));
  dataflow::GraphOp graph;
  d0.module().walk([&](dataflow::GraphOp candidateGraph) {
    if (graph)
      fail("pipeline fixture lowered to multiple graphs");
    graph = candidateGraph;
  });
  if (!graph)
    fail("pipeline fixture did not lower to a graph");
  loom::sim::DFGSimulationOptions options;
  options.graphName = graph.getSymName().str();
  options.memories = {{0, 0, "1,2,3,4,5,6,7,8"}, {1, 0, "0,0,0,0"}};
  return take(loom::sim::simulateDataflowGraph(d0.module(), options));
}

loom::frontend::StructuredProgramCandidate nativeReferenceProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  memref.global constant @pipeline_native_source : memref<4x2xi32> =
      dense<[[1, 2], [3, 4], [5, 6], [7, 8]]>
  memref.global @pipeline_native_target : memref<4xi32> = dense<0>

  llvm.func @entry() -> i32 {
    %source = memref.get_global @pipeline_native_source : memref<4x2xi32>
    %target = memref.get_global @pipeline_native_target : memref<4xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %lhs = memref.load %source[%i, %c0] : memref<4x2xi32>
      %rhs = memref.load %source[%i, %c1] : memref<4x2xi32>
      %sum = arith.addi %lhs, %rhs : i32
      memref.store %sum, %target[%i] : memref<4xi32>
    }
    %result = memref.load %target[%c3] : memref<4xi32>
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the native pipeline oracle");
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), nativeDataLayout()));
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), nativeTargetTriple()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredEntityRef
entryRef(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto view = take(candidate.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getName() == "entry")
      return entity.reference;
  }
  fail("native pipeline oracle has no exact entry");
}

void requireNativeOracle() {
  auto reference = nativeReferenceProgram();
  auto view = take(reference.view());
  loom::sim::StructuredProgramSimulationWorkload draft{entryRef(reference)};
  draft.observableContract.returnValue = true;
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));
  loom::sim::StructuredProgramSimulationRuntimeInputDraft inputDraft{
      workload.identity()};
  auto input = take(
      loom::sim::finalizeSimulationRuntimeInput(inputDraft, workload, view));
  auto expected = take(
      loom::sim::executeNativeStructuredProgram(reference, workload, input));
  if (!expected.returnValue || expected.returnValue->lanes.size() != 1 ||
      expected.returnValue->lanes.front().state !=
          loom::sim::SemanticState::Defined ||
      expected.returnValue->lanes.front().bits != llvm::APInt(32, 15))
    fail("the independent native pipeline oracle returned the wrong value");
}

void pipelineMaterializesDerivedDoubleBuffer() {
  auto parent = parsePipelineProgram(4, "%i");
  auto decisions = pipelineDecisions(parent);
  if (decisions.size() != 1)
    fail("one legal staged loop did not expose one pipeline decision");
  auto encoded =
      take(loom::frontend::encodeStructuredMemoryCommunicationDecision(
          decisions.front()));
  if (encoded.size() != 4 + loom::frontend::structuredEntityRefWireSize ||
      encoded[3] != 2 ||
      !(take(loom::frontend::adoptStructuredMemoryCommunicationDecision(
            encoded)) == decisions.front()))
    fail("pipeline decision did not use the canonical parameter-free wire");

  auto selected =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          parent, decisions.front()));
  unsigned ringCount = 0;
  unsigned copyCount = 0;
  selected.structuredProgram.module().walk([&](mlir::memref::AllocOp alloc) {
    if (alloc.getType().getShape() == llvm::ArrayRef<std::int64_t>({2, 1, 2}))
      ++ringCount;
  });
  selected.structuredProgram.module().walk(
      [&](mlir::memref::CopyOp) { ++copyCount; });
  if (ringCount != 1 || copyCount != 0)
    fail("pipeline did not materialize one derived two-slot ring");

  auto transformed = simulate(selected.structuredProgram);
  llvm::SmallVector<std::string, 4> expected = {"i32:3", "i32:7", "i32:11",
                                                "i32:15"};
  auto transformedOutput = transformed.finalMemoryState.find("arg1");
  if (transformed.status != "pass" ||
      transformedOutput == transformed.finalMemoryState.end() ||
      transformedOutput->second != expected)
    fail("pipeline changed the source workflow observables");
  requireNativeOracle();
}

void legalityRejectsUnprovedSchedules() {
  if (pipelineDecisions(parsePipelineProgram(2, "%i")).size() != 1)
    fail("the exact two-trip boundary was not admitted");
  if (!pipelineDecisions(parsePipelineProgram(1, "%i")).empty() ||
      !pipelineDecisions(parsePipelineProgram(0, "%i")).empty())
    fail("a loop shorter than two trips was admitted");
  if (!pipelineDecisions(parsePipelineProgram(4, "%c0")).empty())
    fail("cross-iteration output aliasing was admitted");
  if (!pipelineDecisions(
           parsePipelineProgram(4, "%i", "func.call @opaque() : () -> ()\n"))
           .empty())
    fail("an unknown call in the compute suffix was admitted");

  auto direct = parseDirectSourceProgram(false);
  auto directDecisions = pipelineDecisions(direct);
  if (directDecisions.size() != 1)
    fail("an exact static direct source was not admitted");
  auto selected =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          direct, directDecisions.front()));
  bool foundDirectRing = false;
  selected.structuredProgram.module().walk([&](mlir::memref::AllocOp alloc) {
    foundDirectRing |=
        alloc.getType().getShape() == llvm::ArrayRef<std::int64_t>({2, 2, 2});
  });
  if (!foundDirectRing)
    fail("the exact direct source did not derive a two-slot ring");
  if (!pipelineDecisions(parseDirectSourceProgram(true)).empty())
    fail("caller-visible source/output aliasing was admitted");
}

} // namespace

int main() {
  pipelineMaterializesDerivedDoubleBuffer();
  legalityRejectsUnprovedSchedules();
  return EXIT_SUCCESS;
}
