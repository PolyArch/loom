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
#include "llvm/ADT/ArrayRef.h"
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
  llvm::errs() << "structuredMemoryLayout: " << message << '\n';
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

loom::frontend::StructuredProgramCandidate parseProgram(llvm::StringRef body) {
  std::string source = R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @layout_source : memref<2x3x2xi32> =
      dense<[[[1, 2], [3, 4], [5, 6]],
             [[7, 8], [9, 10], [11, 12]]]>
  memref.global @layout_target : memref<2x3x2xi32> =
      dense<0>

  dataflow.thread private @layout domain(#dataflow.thread_domain<dense>)(
      %source: memref<2x3x2xi32>, %target: memref<2x3x2xi32>)
      ctrl (%start: none) {
    "loom.spatial_region"(%source, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<2x3x2xi32>, %output: memref<2x3x2xi32>):
)mlir";
  source += body.str();
  source += R"mlir(
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "layout_graph", source_maps = []} :
        (memref<2x3x2xi32>, memref<2x3x2xi32>) -> ()
    dataflow.thread.yield
  }

  llvm.func @entry() -> i32 {
    %source = memref.get_global @layout_source : memref<2x3x2xi32>
    %target = memref.get_global @layout_target : memref<2x3x2xi32>
    %token = dataflow.thread.launch @layout(%source, %target) :
        (memref<2x3x2xi32>, memref<2x3x2xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %value = memref.load %target[%c1, %c2, %c1] : memref<2x3x2xi32>
    llvm.return %value : i32
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("cannot parse the layout fixture");
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), nativeDataLayout()));
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context(), nativeTargetTriple()));
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate layoutProgram() {
  return parseProgram(R"mlir(
        %first = memref.alloc() : memref<2x3x2xi32>
        %second = memref.alloc() : memref<2x3x2xi32>
        memref.copy %input, %first :
            memref<2x3x2xi32> to memref<2x3x2xi32>
        memref.copy %first, %second :
            memref<2x3x2xi32> to memref<2x3x2xi32>
        memref.copy %second, %output :
            memref<2x3x2xi32> to memref<2x3x2xi32>
        memref.dealloc %first : memref<2x3x2xi32>
        memref.dealloc %second : memref<2x3x2xi32>
)mlir");
}

loom::frontend::StructuredProgramCandidate nativeReferenceProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  memref.global constant @layout_source : memref<2x3x2xi32> =
      dense<[[[1, 2], [3, 4], [5, 6]],
             [[7, 8], [9, 10], [11, 12]]]>
  memref.global @layout_target : memref<2x3x2xi32> = dense<0>

  llvm.func @entry() -> i32 {
    %source = memref.get_global @layout_source : memref<2x3x2xi32>
    %target = memref.get_global @layout_target : memref<2x3x2xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        scf.for %k = %c0 to %c2 step %c1 {
          %value = memref.load %source[%i, %j, %k] : memref<2x3x2xi32>
          memref.store %value, %target[%i, %j, %k] : memref<2x3x2xi32>
        }
      }
    }
    %result = memref.load %target[%c1, %c2, %c1] : memref<2x3x2xi32>
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse the native layout oracle");
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
  fail("layout fixture has no exact native entry");
}

void requireNativeOracle(const loom::sim::DFGSimulationReport &transformed) {
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
      expected.returnValue->lanes.front().bits != llvm::APInt(32, 12))
    fail("the independent native layout oracle returned the wrong value");

  const llvm::SmallVector<std::string, 12> expectedMemory = {
      "i32:1", "i32:2", "i32:3", "i32:4",  "i32:5",  "i32:6",
      "i32:7", "i32:8", "i32:9", "i32:10", "i32:11", "i32:12"};
  auto found = transformed.finalMemoryState.find("arg1");
  if (found == transformed.finalMemoryState.end() ||
      found->second != expectedMemory)
    fail("permuted D0 memory differs from the independent native oracle");
}

const loom::frontend::PermuteLocalBufferLayoutDecision &
layoutDecision(const loom::frontend::StructuredMemoryCommunicationDecision &d) {
  const auto *layout =
      std::get_if<loom::frontend::PermuteLocalBufferLayoutDecision>(&d);
  if (!layout)
    fail("selected decision is not a layout permutation");
  return *layout;
}

std::vector<loom::frontend::StructuredMemoryCommunicationDecision>
layoutDecisions(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto domain =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          candidate, 64));
  std::vector<loom::frontend::StructuredMemoryCommunicationDecision> result;
  for (const auto &decision : domain.decisions)
    if (loom::frontend::structuredMemoryCommunicationDecisionKind(decision) ==
        loom::frontend::StructuredMemoryCommunicationDecisionKind::
            PermuteLocalBufferLayout)
      result.push_back(decision);
  return result;
}

llvm::SmallVector<mlir::MemRefType, 2>
allocationTypes(const loom::frontend::StructuredProgramCandidate &candidate) {
  llvm::SmallVector<mlir::MemRefType, 2> result;
  candidate.module().walk(
      [&](mlir::memref::AllocOp alloc) { result.push_back(alloc.getType()); });
  return result;
}

std::vector<std::int64_t> strides(mlir::MemRefType type) {
  llvm::SmallVector<std::int64_t, 4> result;
  std::int64_t offset = 0;
  if (mlir::failed(type.getStridesAndOffset(result, offset)))
    fail("allocation has no exact strided layout");
  return {result.begin(), result.end()};
}

loom::sim::DFGSimulationReport
simulate(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto d0 = take(
      loom::lowering::lowerStructuredProgramToCanonicalDataflow(candidate));
  loom::sim::DFGSimulationOptions options;
  dataflow::GraphOp graph;
  d0.module().walk([&](dataflow::GraphOp candidateGraph) {
    if (graph)
      fail("layout fixture lowered to multiple graphs");
    graph = candidateGraph;
  });
  if (!graph)
    fail("layout fixture did not lower to a graph");
  options.graphName = graph.getSymName().str();
  options.memories = {{0, 0, "1,2,3,4,5,6,7,8,9,10,11,12"},
                      {1, 0, "0,0,0,0,0,0,0,0,0,0,0,0"}};
  return take(loom::sim::simulateDataflowGraph(d0.module(), options));
}

void adjacentPermutationsComposeAndRoundTrip() {
  auto parent = layoutProgram();
  auto bounded =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          parent, 3));
  if (bounded.inspectedMemoryScopes != 3 || bounded.decisions.size() != 2 ||
      layoutDecision(bounded.decisions[0]).anchor !=
          layoutDecision(bounded.decisions[1]).anchor)
    fail("scope limiting cut an admitted allocation parameter domain");
  auto firstDomain = layoutDecisions(parent);
  if (firstDomain.size() != 4)
    fail("two rank-three buffers did not expose four adjacent exchanges");
  const auto &firstChoice = layoutDecision(firstDomain.front());
  if (firstChoice.adjacentStoragePosition != 0)
    fail("layout parameters are not in canonical ascending order");

  auto encoded =
      take(loom::frontend::encodeStructuredMemoryCommunicationDecision(
          firstDomain.front()));
  if (encoded.size() != 4 + loom::frontend::structuredEntityRefWireSize + 8 ||
      encoded[0] != 0 || encoded[1] != 0 || encoded[2] != 0 ||
      encoded[3] != 1 ||
      !(take(loom::frontend::adoptStructuredMemoryCommunicationDecision(
            encoded)) == firstDomain.front()))
    fail("layout decision did not use the exact 2.0 wire");

  std::vector<std::uint8_t> legacy =
      loom::frontend::encodeStructuredEntityRef(firstChoice.anchor);
  legacy.insert(legacy.end(), {0, 0, 0, 1});
  for (unsigned index = 0; index != 8; ++index)
    legacy.push_back(0);
  auto rejected =
      loom::frontend::adoptStructuredMemoryCommunicationDecision(legacy);
  if (rejected)
    fail("a legacy anchor-first decision was reinterpreted as 2.0");
  llvm::consumeError(rejected.takeError());

  auto first =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          parent, firstDomain.front()));
  auto afterFirst = allocationTypes(first.structuredProgram);
  if (afterFirst.size() != 2 ||
      strides(afterFirst[0]) != std::vector<std::int64_t>({2, 4, 1}) ||
      !afterFirst[1].getLayout().isIdentity())
    fail("the first adjacent storage exchange wrote the wrong strides");

  auto secondDomain = layoutDecisions(first.structuredProgram);
  auto secondChoice = llvm::find_if(secondDomain, [&](const auto &decision) {
    const auto &layout = layoutDecision(decision);
    auto view = take(first.structuredProgram.view());
    auto entity = take(view.resolve(layout.anchor));
    auto alloc = entity.value.template getDefiningOp<mlir::memref::AllocOp>();
    return layout.adjacentStoragePosition == 1 && alloc &&
           alloc.getType().getLayout().isIdentity();
  });
  if (secondChoice == secondDomain.end())
    fail("the other buffer has no adjacent-position-one choice");
  auto second =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          first.structuredProgram, *secondChoice));
  auto afterSecond = allocationTypes(second.structuredProgram);
  if (afterSecond.size() != 2 ||
      strides(afterSecond[0]) != std::vector<std::int64_t>({2, 4, 1}) ||
      strides(afterSecond[1]) != std::vector<std::int64_t>({6, 1, 3}))
    fail("independent adjacent exchanges did not compose exact layouts");

  auto reference = simulate(parent);
  auto transformed = simulate(second.structuredProgram);
  if (reference.status != "pass" || transformed.status != "pass" ||
      reference.finalMemoryState.find("arg1") ==
          reference.finalMemoryState.end() ||
      transformed.finalMemoryState.find("arg1") ==
          transformed.finalMemoryState.end() ||
      reference.finalMemoryState.find("arg1")->second !=
          transformed.finalMemoryState.find("arg1")->second) {
    llvm::errs() << "reference status: " << reference.status
                 << ", transformed status: " << transformed.status << '\n';
    llvm::errs() << "reference diagnostics:";
    for (const std::string &value : reference.diagnostics)
      llvm::errs() << ' ' << value;
    llvm::errs() << "\ntransformed diagnostics:";
    for (const std::string &value : transformed.diagnostics)
      llvm::errs() << ' ' << value;
    llvm::errs() << '\n';
    fail("different source and target layouts changed the copied values");
  }
  requireNativeOracle(transformed);
}

void invalidAllocationsHaveNoLayoutDecision() {
  auto rankOne = parseProgram(R"mlir(
        %buffer = memref.alloc() : memref<12xi32>
        %c0 = arith.constant 0 : index
        %value = memref.load %buffer[%c0] : memref<12xi32>
)mlir");
  if (!layoutDecisions(rankOne).empty())
    fail("a rank-one allocation exposed a layout permutation");

  auto castEscape = parseProgram(R"mlir(
        %buffer = memref.alloc() : memref<2x3x2xi32>
        %alias = memref.cast %buffer : memref<2x3x2xi32> to memref<?x3x2xi32>
)mlir");
  if (!layoutDecisions(castEscape).empty())
    fail("an allocation with a derived alias exposed a layout permutation");

  auto zeroShape = parseProgram(R"mlir(
        %buffer = memref.alloc() : memref<2x0x2xi32>
)mlir");
  if (!layoutDecisions(zeroShape).empty())
    fail("a zero-extent allocation exposed a layout permutation");

  auto dynamicShape = parseProgram(R"mlir(
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %buffer = memref.alloc(%c2) : memref<?x3x2xi32>
        %value = memref.load %buffer[%c0, %c0, %c0] : memref<?x3x2xi32>
)mlir");
  if (!layoutDecisions(dynamicShape).empty())
    fail("a dynamically shaped allocation exposed a layout permutation");
}

} // namespace

int main() {
  adjacentPermutationsComposeAndRoundTrip();
  invalidAllocationsHaveNoLayoutDecision();
  return EXIT_SUCCESS;
}
