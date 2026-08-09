#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Compilation/StructuredMemoryCommunication.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Simulator/DFGSimulator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "structuredMemoryChannel: " << message << '\n';
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
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

struct FixtureOptions final {
  bool twoConsumers = false;
  bool partialConsumer = false;
  bool reorderedConsumer = false;
  bool visibleBeforeReceive = false;
  bool sharedDefinitions = false;
  bool lateConsumerOperand = false;
};

loom::frontend::StructuredProgramCandidate
parseProgram(FixtureOptions options = {}) {
  const char *consumerLimit = options.partialConsumer ? "%c3" : "%c4";
  const char *consumerIndex = options.reorderedConsumer ? "%reverse" : "%i";
  const char *consumerScalar = options.lateConsumerOperand ? "%late" : "%early";
  std::string reverse =
      options.reorderedConsumer
          ? "          %reverse = arith.subi %c3, %i : index\n"
          : "";
  std::string visible = options.visibleBeforeReceive ? R"mlir(
        %visible = arith.constant 99 : i32
        memref.store %visible, %output[%c0] : memref<4xi32>
)mlir"
                                                     : "";

  std::string source = R"mlir(
#layout = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
module attributes {dlti.dl_spec = #layout} {
  memref.global constant @channel_source : memref<4xi32> =
      dense<[1, 3, 5, 7]>
  memref.global @channel_target : memref<4xi32> = dense<0>

  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %source: memref<4xi32>, %temporary: memref<4xi32>)
      ctrl (%start: none) {
    "loom.spatial_region"(%source, %temporary)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%input: memref<4xi32>, %buffer: memref<4xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %value = memref.load %input[%i] : memref<4xi32>
          memref.store %value, %buffer[%i] : memref<4xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "channel_producer", source_maps = []} :
        (memref<4xi32>, memref<4xi32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %temporary: memref<4xi32>, %target: memref<4xi32>, %unused: i32)
      ctrl (%start: none) {
    "loom.spatial_region"(%temporary, %target)
        <{operandSegmentSizes = array<i32: 0, 0, 2, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%buffer: memref<4xi32>, %output: memref<4xi32>):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
)mlir";
  source += visible;
  source += "        scf.for %i = %c0 to ";
  source += consumerLimit;
  source += " step %c1 {\n";
  source += reverse;
  source += "          %value = memref.load %buffer[";
  source += consumerIndex;
  source += R"mlir(] : memref<4xi32>
          %one = arith.constant 1 : i32
          %adjusted = arith.addi %value, %one : i32
          memref.store %adjusted, %output[%i] : memref<4xi32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "channel_consumer", source_maps = []} :
        (memref<4xi32>, memref<4xi32>) -> ()
    dataflow.thread.yield
  }

  func.func @entry() {
    %source = memref.get_global @channel_source : memref<4xi32>
    %target = memref.get_global @channel_target : memref<4xi32>
    %early = arith.constant 0 : i32
    %temporary = memref.alloc() : memref<4xi32>
    %producer = dataflow.thread.launch @producer(%source, %temporary) :
        (memref<4xi32>, memref<4xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %producer : !dataflow.thread_token
    %late = arith.constant 1 : i32
)mlir";
  source += "    %consumer = dataflow.thread.launch @consumer(%temporary, "
            "%target, ";
  source += consumerScalar;
  source += R"mlir() :
        (memref<4xi32>, memref<4xi32>, i32) -> !dataflow.thread_token
    dataflow.thread.wait %consumer : !dataflow.thread_token
)mlir";
  if (options.twoConsumers)
    source += R"mlir(
    %consumer_again = dataflow.thread.launch @consumer(%temporary, %target, %early) :
        (memref<4xi32>, memref<4xi32>, i32) -> !dataflow.thread_token
    dataflow.thread.wait %consumer_again : !dataflow.thread_token
)mlir";
  source += "    memref.dealloc %temporary : memref<4xi32>\n";
  if (options.sharedDefinitions)
    source += R"mlir(
    %other = memref.alloc() : memref<4xi32>
    %other_producer = dataflow.thread.launch @producer(%source, %other) :
        (memref<4xi32>, memref<4xi32>) -> !dataflow.thread_token
    dataflow.thread.wait %other_producer : !dataflow.thread_token
    %other_consumer = dataflow.thread.launch @consumer(%other, %target, %early) :
        (memref<4xi32>, memref<4xi32>, i32) -> !dataflow.thread_token
    dataflow.thread.wait %other_consumer : !dataflow.thread_token
    memref.dealloc %other : memref<4xi32>
)mlir";
  source += R"mlir(
    return
  }
}
)mlir";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("cannot parse the channel fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

std::vector<loom::frontend::StructuredMemoryCommunicationDecision>
channelDecisions(const loom::frontend::StructuredProgramCandidate &candidate) {
  auto domain =
      take(loom::frontend::enumerateStructuredMemoryCommunicationDecisions(
          candidate, 64));
  std::vector<loom::frontend::StructuredMemoryCommunicationDecision> result;
  for (const auto &decision : domain.decisions)
    if (loom::frontend::structuredMemoryCommunicationDecisionKind(decision) ==
        loom::frontend::StructuredMemoryCommunicationDecisionKind::
            PromoteSpscBufferToChannel)
      result.push_back(decision);
  return result;
}

std::size_t countThreads(mlir::ModuleOp module) {
  std::size_t count = 0;
  module.walk([&](dataflow::ThreadOp) { ++count; });
  return count;
}

void exactSpscMaterializesAndSimulates() {
  auto parent = parseProgram();
  auto decisions = channelDecisions(parent);
  if (decisions.size() != 1)
    fail("one exact SPSC buffer did not expose one channel decision");
  auto encoded =
      take(loom::frontend::encodeStructuredMemoryCommunicationDecision(
          decisions.front()));
  if (encoded.size() != 4 + loom::frontend::structuredEntityRefWireSize ||
      encoded[3] != 3 ||
      !(take(loom::frontend::adoptStructuredMemoryCommunicationDecision(
            encoded)) == decisions.front()))
    fail("channel decision did not use the parameter-free 2.0 wire");

  auto selected =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          parent, decisions.front()));
  std::size_t creates = 0;
  std::size_t sends = 0;
  std::size_t receives = 0;
  std::size_t allocations = 0;
  selected.structuredProgram.module().walk(
      [&](dataflow::ChannelCreateOp) { ++creates; });
  selected.structuredProgram.module().walk(
      [&](dataflow::ChannelSendOp) { ++sends; });
  selected.structuredProgram.module().walk(
      [&](dataflow::ChannelReceiveOp) { ++receives; });
  selected.structuredProgram.module().walk(
      [&](mlir::memref::AllocOp) { ++allocations; });
  if (creates != 1 || sends != 1 || receives != 1 || allocations != 0)
    fail("SPSC promotion did not replace the exact temporary closure");

  dataflow::ThreadLaunchOp producer;
  dataflow::ThreadLaunchOp consumer;
  dataflow::ThreadWaitOp producerWait;
  dataflow::ChannelCreateOp created;
  selected.structuredProgram.module().walk(
      [&](dataflow::ChannelCreateOp candidate) { created = candidate; });
  for (mlir::OpOperand &use : created.getChannel().getUses()) {
    auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(use.getOwner());
    auto thread =
        launch ? mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
                     launch, launch.getCalleeAttr())
               : dataflow::ThreadOp{};
    bool sends = false;
    bool receives = false;
    if (thread)
      thread.walk([&](mlir::Operation *operation) {
        sends |= llvm::isa<dataflow::ChannelSendOp>(operation);
        receives |= llvm::isa<dataflow::ChannelReceiveOp>(operation);
      });
    if (sends && !receives)
      producer = launch;
    if (receives && !sends)
      consumer = launch;
  }
  selected.structuredProgram.module().walk([&](dataflow::ThreadWaitOp wait) {
    if (producer && wait.getAsyncDependencies().size() == 1 &&
        wait.getAsyncDependencies().front() == producer.getAsyncToken())
      producerWait = wait;
  });
  if (!producer || !consumer || !producerWait ||
      !consumer->isBeforeInBlock(producerWait))
    fail("consumer launch was not moved before producer completion");

  auto d0 = take(loom::lowering::lowerStructuredProgramToCanonicalDataflow(
      selected.structuredProgram));
  dataflow::GraphOp producerGraph;
  dataflow::GraphOp consumerGraph;
  d0.module().walk([&](dataflow::GraphOp graph) {
    if (graph.getResultSegmentSizes()[1] == 1)
      producerGraph = graph;
    if (graph.getInputSegmentSizes()[1] == 1)
      consumerGraph = graph;
  });
  if (!producerGraph || !consumerGraph)
    fail("mechanical lowering lost a promoted stream boundary");

  loom::sim::DFGSimulationOptions producerOptions;
  producerOptions.graphName = producerGraph.getSymName().str();
  producerOptions.memories = {{0, 0, "1,3,5,7"}};
  auto producerReport =
      take(loom::sim::simulateDataflowGraph(d0.module(), producerOptions));
  const llvm::SmallVector<std::string, 4> sent = {"i32:1", "i32:3", "i32:5",
                                                  "i32:7"};
  if (producerReport.status != "pass" ||
      producerReport.finalStreamOutputs.size() != 1 ||
      producerReport.finalStreamOutputs.front() != sent)
    fail("producer DFG changed the promoted ordered payload sequence");

  loom::sim::DFGSimulationOptions consumerOptions;
  consumerOptions.graphName = consumerGraph.getSymName().str();
  for (const char *value : {"1", "3", "5", "7"})
    consumerOptions.args.push_back({0, value});
  consumerOptions.memories = {{1, 0, "0,0,0,0"}};
  auto consumerReport =
      take(loom::sim::simulateDataflowGraph(d0.module(), consumerOptions));
  const llvm::SmallVector<std::string, 4> expected = {"i32:2", "i32:4", "i32:6",
                                                      "i32:8"};
  auto output = consumerReport.finalMemoryState.find("arg1");
  if (consumerReport.status != "pass" ||
      output == consumerReport.finalMemoryState.end() ||
      output->second != expected)
    fail("consumer DFG changed the promoted memory observables");
}

void rejectsUnprovedCommunication() {
  if (!channelDecisions(parseProgram({.twoConsumers = true})).empty())
    fail("a temporary with two consumers was promoted");
  if (!channelDecisions(parseProgram({.partialConsumer = true})).empty())
    fail("a partial consumer domain was promoted");
  if (!channelDecisions(parseProgram({.reorderedConsumer = true})).empty())
    fail("a reordered consumer domain was promoted");
  if (!channelDecisions(parseProgram({.visibleBeforeReceive = true})).empty())
    fail("a consumer with a visible pre-receive effect was promoted");
  if (!channelDecisions(parseProgram({.lateConsumerOperand = true})).empty())
    fail("a consumer operand defined after the motion point was promoted");
}

void sharedDefinitionsSpecializeOnlySelectedLaunches() {
  auto parent = parseProgram({.sharedDefinitions = true});
  auto decisions = channelDecisions(parent);
  if (decisions.size() != 2)
    fail("two independent SPSC buffers did not expose two decisions");
  auto selected =
      take(loom::frontend::materializeStructuredMemoryCommunicationDecision(
          parent, decisions.front()));
  std::size_t allocations = 0;
  std::size_t creates = 0;
  selected.structuredProgram.module().walk(
      [&](mlir::memref::AllocOp) { ++allocations; });
  selected.structuredProgram.module().walk(
      [&](dataflow::ChannelCreateOp) { ++creates; });
  if (allocations != 1 || creates != 1 ||
      countThreads(selected.structuredProgram.module()) != 4)
    fail("shared thread definitions were not specialized exactly once");
}

} // namespace

int main() {
  exactSpscMaterializesAndSimulates();
  rejectsUnprovedCommunication();
  sharedDefinitionsSpecializeOnlySelectedLaunches();
  return EXIT_SUCCESS;
}
