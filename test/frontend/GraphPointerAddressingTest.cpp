#include "GraphPointerAddressing.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Frontend/Lowering/Passes.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

using namespace dataflow;
using namespace loom;
using namespace loom::sim;

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "GraphPointerAddressingTest: " << message << '\n';
  std::exit(1);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext instance(mlir::MLIRContext::Threading::DISABLED);
  instance.loadDialect<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                       mlir::DLTIDialect, mlir::func::FuncDialect,
                       mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return instance;
}

LogicalMemoryRootRef rootAtService(const CanonicalDataflowProgramView &view) {
  for (const auto &root : view.logicalMemoryRoots())
    if (!root.formalArgIndex && llvm::isa_and_nonnull<MemoryServiceOp>(root.op))
      return root.ref;
  fail("address graph has no memory-service root");
}

// The unexpanded GEP simulator is the independent arithmetic oracle. These
// cases distinguish index, scale, prefix-sum, pointer-wrap and object-bound
// poison; preserving the final wrapped offset alone would miss all of them.
void checkAddress(const char *name, unsigned representationBits,
                  unsigned addressBits, const char *indexType,
                  const char *elementType, const char *firstIndex,
                  const char *secondIndex, const char *flags,
                  uint64_t baseOffset, SemanticState expectedState,
                  uint64_t expectedOffset = 0) {
  const bool second = secondIndex != nullptr;
  std::string text =
      "module attributes {llvm.data_layout = \"e-p:" +
      std::to_string(representationBits) + ":" +
      std::to_string(representationBits) + ":" +
      std::to_string(representationBits) + ":" + std::to_string(addressBits) +
      "\", dlti.dl_spec = #dlti.dl_spec<\"dlti.endianness\" = \"little\", "
      "index = 64 : i64>} {\n"
      "dataflow.graph private @address(%ctrl: none, %pointer: !llvm.ptr, "
      "%service: memref<?xi8>) -> !llvm.ptr attributes {"
      "input_segments = array<i32: 1, 0, 1>, result_segments = array<i32: 1, "
      "0, 0>} {\n"
      "%index = arith.constant " +
      firstIndex + " : " + indexType + "\n";
  if (second)
    text += "%inner = arith.constant " + std::string(secondIndex) + " : " +
            indexType + "\n";
  text +=
      "%result = llvm.getelementptr " + std::string(flags) +
      " %pointer[%index" + (second ? ", %inner" : "") + "] : (!llvm.ptr, " +
      indexType + (second ? ", " + std::string(indexType) : "") +
      ") -> !llvm.ptr, " + elementType +
      "\n"
      "%retired:2 = dataflow.sync %ctrl, %result : (none, !llvm.ptr) -> (none, "
      "!llvm.ptr)\n"
      "dataflow.graph.return values(%retired#1 : !llvm.ptr) streams() "
      "memories() complete(%retired#0 : none) }\n"
      "dataflow.thread private @thread "
      "domain(#dataflow.thread_domain<dense>)(%pointer: !llvm.ptr) ctrl "
      "(%ctrl: none) {\n"
      "%service = dataflow.memory.service %pointer : !llvm.ptr -> "
      "memref<?xi8>\n"
      "%value, %done = dataflow.graph.launch @address deps(%ctrl) "
      "values(%pointer) stream_inputs() memories(%service) stream_outputs() : "
      "(none, !llvm.ptr, memref<?xi8>) -> (!llvm.ptr, none)\n"
      "dataflow.thread.yield %done : none }\n"
      "func.func private @host(%pointer: !llvm.ptr) { %thread = "
      "dataflow.thread.launch @thread(%pointer) : (!llvm.ptr) -> "
      "!dataflow.thread_token return } }";
  for (bool normalized : {false, true}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context());
    if (!module)
      fail("cannot parse address probe");
    if (normalized) {
      auto graph = module->lookupSymbol<dataflow::GraphOp>("address");
      if (mlir::failed(loom::lowering::normalizeGraphPointerAddresses(graph)))
        fail("normalization failed");
      mlir::PassManager pm(&context());
      pm.addPass(loom::lowering::createLowerGraphConstantsPass());
      if (mlir::failed(pm.run(*module)))
        fail("constant publication failed");
    }
    auto artifact = take(finalizeCanonicalDataflow(*module));
    const auto &view = artifact.view();
    SpatialSimulationWorkload draft{
        RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                             view.staticGraphLaunches().front().ref}};
    draft.valueInputPlan = {RuntimeValueInput{}};
    draft.observableContract.valueResults = {0};
    auto workload = take(finalizeSimulationWorkload(draft, view));
    RuntimeMemoryObject object;
    object.initialBytes.assign(200,
                               SemanticMemoryByte{SemanticState::Defined, 0});
    SpatialSimulationRuntimeInputDraft input{workload.identity()};
    input.runtimeValues = {RuntimeValueEntry{
        0, CanonicalValueSequence{1,
                                  {SemanticLane::definedPointer(
                                      llvm::APInt(representationBits, 0), 0,
                                      llvm::APInt(addressBits, baseOffset))}}}};
    input.memoryObjects = {std::move(object)};
    input.memoryRootBindings = {
        RuntimeMemoryBindingDraft{rootAtService(view), 0, 0}};
    auto runtime = take(finalizeSimulationRuntimeInput(input, workload, view));
    auto result = take(simulateRetiredDfgWorkload(artifact, workload, runtime));
    const auto &value = std::get<PublishedValueResult>(
        result.observations.valueResults.front());
    const auto &lane = value.value.lanes.front();
    const std::string label =
        std::string(name) +
        (normalized ? " after normalization: " : " before normalization: ");
    require(lane.state == expectedState,
            label + "unexpected address semantic state");
    if (expectedState == SemanticState::Defined)
      require(lane.pointerTarget && lane.pointerTarget->objectOrdinal == 0 &&
                  lane.pointerTarget->byteOffset ==
                      llvm::APInt(addressBits, expectedOffset),
              label + "changed object provenance or offset");
  }
}

void partialPointerLayoutKeepsItsAddressDomain() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {llvm.data_layout = "e-p:64:64:64:32"} {
  dataflow.graph private @partial(%ctrl: none, %base: !llvm.ptr, %index: i64)
      -> !llvm.ptr attributes {input_segments = array<i32: 2, 0, 0>,
                              result_segments = array<i32: 1, 0, 0>} {
    %address = llvm.getelementptr %base[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %retired:2 = dataflow.sync %ctrl, %address : (none, !llvm.ptr) -> (none, !llvm.ptr)
    dataflow.graph.return values(%retired#1 : !llvm.ptr) streams() memories()
        complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  require(bool(module), "cannot parse partial-width pointer graph");
  auto graph = module->lookupSymbol<dataflow::GraphOp>("partial");
  require(
      mlir::succeeded(loom::lowering::normalizeGraphPointerAddresses(graph)),
      "partial-width pointer address normalization failed");
  mlir::LLVM::GEPOp address;
  graph.walk([&](mlir::LLVM::GEPOp op) { address = op; });
  require(
      address && address.getElemType().isInteger(8) &&
          address.getDynamicIndices().size() == 1 &&
          address.getDynamicIndices().front().getType().isInteger(32) &&
          mlir::isa<mlir::LLVM::LLVMPointerType>(address.getType()),
      "address width was replaced by pointer representation or index width");
}

} // namespace

int main() {
  partialPointerLayoutKeepsItsAddressDomain();
  checkAddress("signed-extension", 64, 64, "i32", "i32", "-1", nullptr, "", 8,
               SemanticState::Defined, 4);
  checkAddress("index-truncation", 32, 32, "i64", "i8", "4294967296", nullptr,
               "nusw", 0, SemanticState::Poison);
  checkAddress("scale-overflow", 32, 32, "i32", "i32", "1073741824", nullptr,
               "nusw", 0, SemanticState::Poison);
  checkAddress("accumulation-overflow", 8, 8, "i8", "!llvm.array<64 x i8>", "1",
               "64", "nusw", 0, SemanticState::Poison);
  checkAddress("intermediate-bounds", 64, 64, "i64", "!llvm.array<128 x i8>",
               "2", "-256", "inbounds", 0, SemanticState::Poison);
  checkAddress("unsigned-pointer-wrap", 8, 8, "i8", "i8", "64", nullptr, "nuw",
               192, SemanticState::Poison);
}
