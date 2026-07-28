// Anchor tests for the schema-1.0 Spatial SimulationWorkload and
// SimulationRuntimeInput persistent artifacts: rooted-launch ownership,
// total value classification, stream horizon state, canonical object
// ordinals, observable contracts, strict wire parsing, and DFG/CGRA
// admission.

#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationAdmission.h"
#include "Simulator/SimulationArtifacts.h"

#include "Common/ArtifactText.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::sim;
using namespace dataflow;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}
void require(const char *test, bool ok, const std::string &message) {
  if (!ok)
    fail(test, message);
}
template <typename T> bool isRejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}
bool errored(llvm::Error error) {
  bool failed = static_cast<bool>(error);
  llvm::consumeError(std::move(error));
  return failed;
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *ctx = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::DLTIDialect,
                    mlir::func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect>();
    auto *c = new mlir::MLIRContext(registry);
    c->loadAllAvailableDialects();
    return c;
  }();
  return *ctx;
}
CanonicalDataflowArtifact finalizeProgram(const char *test,
                                          llvm::StringRef text) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context());
  if (!module)
    fail(test, "failed to parse fixture");
  llvm::Expected<CanonicalDataflowArtifact> artifact =
      finalizeCanonicalDataflow(module.get());
  if (!artifact)
    fail(test, "finalize failed: " + llvm::toString(artifact.takeError()));
  return std::move(*artifact);
}
CanonicalDataflowProgramView viewOf(const char *test,
                                    const CanonicalDataflowArtifact &artifact) {
  llvm::Expected<CanonicalDataflowProgramView> view = artifact.view();
  if (!view)
    fail(test, "view import failed: " + llvm::toString(view.takeError()));
  return std::move(*view);
}

// The vecadd anchor program: value input N, imported roots A, B, and C, and
// the computation C[i] = A[i] + B[i] reduced to one representative scalar
// access per root.
const char *vecaddProgram() {
  return R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<
  "dlti.endianness" = "little",
  index = 32 : i64
>} {
  dataflow.graph private @vecadd(%ctrl: none, %n: index, %a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) -> () attributes {input_segments = array<i32: 1, 0, 3>, result_segments = array<i32: 0, 0, 0>} {
    %zero = arith.constant 0 : index
    %va, %da = dataflow.load %a[%zero] %ctrl : memref<1024xf32>
    %vb, %db = dataflow.load %b[%zero] %ctrl : memref<1024xf32>
    %sum = arith.addf %va, %vb : f32
    %ds = dataflow.store %c[%zero] %sum %ctrl : memref<1024xf32>
    dataflow.graph.return values() streams() memories() complete(%da, %db, %ds : none, none, none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(%a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) ctrl (%ctrl: none) {
    %n = arith.constant 1024 : index
    %d = dataflow.graph.launch @vecadd deps(%ctrl) values(%n) stream_inputs() memories(%a, %b, %c) stream_outputs() : (none, index, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) -> none
    dataflow.thread.yield %d : none
  }
  func.func private @host(%a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) {
    %t = dataflow.thread.launch @t(%a, %b, %c) : (memref<1024xf32>, memref<1024xf32>, memref<1024xf32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// A different finalized program, used only as a foreign artifact identity.
const char *foreignProgram() {
  return R"mlir(
module {
  dataflow.graph private @g(%ctrl: none) -> i32 attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
    %k = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %r:2 = dataflow.sync %ctrl, %k : (none, i32) -> (none, i32)
    dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none)
  }
}
)mlir";
}

RootedGraphLaunchRef onlyLaunch(const char *test,
                                const CanonicalDataflowProgramView &view) {
  require(test,
          view.rootThreadLaunches().size() == 1 &&
              view.staticGraphLaunches().size() == 1,
          "fixture has exactly one root and one static launch");
  return RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                              view.staticGraphLaunches().front().ref};
}

LogicalMemoryRootRef rootByFormal(const char *test,
                                  const CanonicalDataflowProgramView &view,
                                  unsigned formal) {
  for (const CanonicalLogicalMemoryRootView &root : view.logicalMemoryRoots())
    if (root.formalArgIndex && *root.formalArgIndex == formal)
      return root.ref;
  fail(test, "no imported root for formal");
}

SemanticLane definedLane(unsigned width, uint64_t value) {
  return SemanticLane::defined(llvm::APInt(width, value));
}
CanonicalValueSequence oneToken(std::initializer_list<SemanticLane> lanes) {
  CanonicalValueSequence sequence;
  sequence.tokenCount = 1;
  sequence.lanes = lanes;
  return sequence;
}

// A valid vecadd workload: N fixed to one defined 32-bit index token (the
// configured index width), every root observed through the contract's
// LogicalMemory(Root(C)) diff target.
SpatialSimulationWorkload
makeVecaddWorkload(const CanonicalDataflowProgramView &view) {
  SpatialSimulationWorkload workload{onlyLaunch("makeVecaddWorkload", view)};
  workload.valueInputPlan.push_back(oneToken({definedLane(32, 1024)}));
  workload.observableContract.memories.push_back(SpatialMemoryObservable{
      LogicalMemoryRootOrViewRef{rootByFormal("makeVecaddWorkload", view, 2)},
      MemoryObservationForm::DiffFromRuntimeInput});
  return workload;
}

// (a) Rooted-launch ownership and foreign/stale/wrong-kind rejection.
void rootedLaunchOwnership() {
  const char *test = "rootedLaunchOwnership";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, vecaddProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);

  SpatialSimulationWorkload workload = makeVecaddWorkload(view);
  llvm::Expected<CanonicalSimulationWorkload> finalized =
      finalizeSimulationWorkload(workload, view);
  if (!finalized)
    fail(test, "a valid spatial workload finalizes: " +
                   llvm::toString(finalized.takeError()));
  require(test,
          finalized->completion() ==
              GraphLaunchDoneTransferRef{workload.launchRef},
          "graph completion derives from the rooted launch");

  // A foreign artifact identity on either nested reference is rejected.
  CanonicalDataflowArtifact other = finalizeProgram(test, foreignProgram());
  SpatialSimulationWorkload foreign = workload;
  foreign.launchRef.rootThreadLaunch.artifact = other.identity();
  require(test, isRejected(finalizeSimulationWorkload(foreign, view)),
          "a foreign root thread launch artifact is rejected");
  foreign = workload;
  foreign.launchRef.staticGraphLaunch.artifact = other.identity();
  require(test, isRejected(finalizeSimulationWorkload(foreign, view)),
          "a foreign static graph launch artifact is rejected");

  // A stale entity ID beyond the artifact's entity space is rejected.
  SpatialSimulationWorkload stale = workload;
  stale.launchRef.staticGraphLaunch.entity =
      StaticGraphLaunchId(view.entityCount() + 7);
  require(test, isRejected(finalizeSimulationWorkload(stale, view)),
          "a stale static graph launch ID is rejected");

  // A wrong-kind entity ID is rejected even when in range.
  SpatialSimulationWorkload wrongKind = workload;
  wrongKind.launchRef.staticGraphLaunch.entity =
      StaticGraphLaunchId(workload.launchRef.rootThreadLaunch.entity.value());
  require(test, isRejected(finalizeSimulationWorkload(wrongKind, view)),
          "a wrong-kind static graph launch ID is rejected");
}

// A multi-value program: index, fixed-vector, and f32 value inputs and two
// value results, exercising total classification and exact lane states.
const char *valuesProgram() {
  return R"mlir(
module {
  dataflow.graph private @gv(%ctrl: none, %n: index, %v: vector<2x2xi32>, %f: f32) -> (vector<2x2xi32>, f32) attributes {input_segments = array<i32: 3, 0, 0>, result_segments = array<i32: 2, 0, 0>} {
    %r:3 = dataflow.sync %ctrl, %v, %f : (none, vector<2x2xi32>, f32) -> (none, vector<2x2xi32>, f32)
    dataflow.graph.return values(%r#1, %r#2 : vector<2x2xi32>, f32) streams() memories() complete(%r#0 : none)
  }
  dataflow.thread private @tv domain(#dataflow.thread_domain<dense>)() ctrl (%c: none) {
    %n = arith.constant 8 : index
    %v = arith.constant dense<1> : vector<2x2xi32>
    %f = arith.constant 1.000000e+00 : f32
    %a, %b, %d = dataflow.graph.launch @gv deps(%c) values(%n, %v, %f) stream_inputs() memories() stream_outputs() : (none, index, vector<2x2xi32>, f32) -> (vector<2x2xi32>, f32, none)
    dataflow.thread.yield %d : none
  }
  func.func private @host() {
    %t = dataflow.thread.launch @tv() : () -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// A stream-input program: one producer thread sends on a host channel and one
// consumer thread launches a graph with one i32 stream input.
const char *streamProgram() {
  return R"mlir(
module {
  dataflow.graph private @gs(%start: none, %input: i32) -> () attributes {input_segments = array<i32: 0, 1, 0>, result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(%ch: !dataflow.channel<i32>) ctrl (%c: none) {
    %v = arith.constant 7 : i32
    dataflow.channel.send %ch, %v : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(%ch: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @gs deps(%ctrl) values() stream_inputs(%ch source_map affine_map<() -> ()>) memories() stream_outputs() : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @host(%ch: !dataflow.channel<i32>) {
    %p = dataflow.thread.launch @producer(%ch) : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %c = dataflow.thread.launch @consumer(%ch) : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// A rank-two grid program for dense-coordinate validation.
const char *gridProgram() {
  return R"mlir(
module {
  dataflow.graph private @gg(%start: none) -> () attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @tg domain(#dataflow.thread_domain<dense>)() ctrl (%c: none) iv (%i: index, %j: index) {
    %d = dataflow.graph.launch @gg deps(%c) values() stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %d : none
  }
  func.func private @host() {
    %g0 = arith.constant 4 : index
    %g1 = arith.constant 8 : index
    %t = dataflow.thread.launch @tg() grid(%g0, %g1) : () -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// A thread holding two memory formals whose launch binds only the first, so
// the second formal is an unrelated root for this workload.
const char *bindingRulesProgram() {
  return R"mlir(
module {
  dataflow.graph private @g1(%ctrl: none, %m: memref<4xi32>) -> () attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 0>} {
    %idx = arith.constant 0 : index
    %d, %done = dataflow.load %m[%idx] %ctrl : memref<4xi32>
    dataflow.graph.return values() streams() memories() complete(%done : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(%m0: memref<4xi32>, %m1: memref<4xi32>) ctrl (%ctrl: none) {
    %d = dataflow.graph.launch @g1 deps(%ctrl) values() stream_inputs() memories(%m0) stream_outputs() : (none, memref<4xi32>) -> none
    dataflow.thread.yield %d : none
  }
  func.func private @host(%m0: memref<4xi32>, %m1: memref<4xi32>) {
    %t = dataflow.thread.launch @t(%m0, %m1) : (memref<4xi32>, memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// One thread launching a fresh-allocation exporter and a formal-passthrough
// exporter of the same imported root.
const char *exposureProgram() {
  return R"mlir(
module {
  dataflow.graph private @gf(%ctrl: none, %mem: memref<10xi32>) -> memref<10xi32> attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 1>} {
    %a = memref.alloc() : memref<10xi32>
    %idx = arith.constant 0 : index
    %d, %done = dataflow.load %a[%idx] %ctrl : memref<10xi32>
    dataflow.graph.return values() streams() memories(%a : memref<10xi32>) complete(%done : none)
  }
  dataflow.graph private @gi(%ctrl: none, %mem: memref<10xi32>) -> memref<10xi32> attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 1>} {
    dataflow.graph.return values() streams() memories(%mem : memref<10xi32>) complete(%ctrl : none)
  }
  dataflow.graph private @gc(%ctrl: none, %mem: memref<10xi32>) -> memref<10xi32> attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 1>} {
    dataflow.graph.return values() streams() memories(%mem : memref<10xi32>) complete(%ctrl : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(%m: memref<10xi32>) ctrl (%ctrl: none) {
    %rf, %df = dataflow.graph.launch @gf deps(%ctrl) values() stream_inputs() memories(%m) stream_outputs() : (none, memref<10xi32>) -> (memref<10xi32>, none)
    %ri, %di = dataflow.graph.launch @gi deps(%df) values() stream_inputs() memories(%m) stream_outputs() : (none, memref<10xi32>) -> (memref<10xi32>, none)
    %rc, %dc = dataflow.graph.launch @gc deps(%di) values() stream_inputs() memories(%rf) stream_outputs() : (none, memref<10xi32>) -> (memref<10xi32>, none)
    dataflow.thread.yield %dc : none
  }
  func.func private @host(%m: memref<10xi32>) {
    %t = dataflow.thread.launch @t(%m) : (memref<10xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// Two actor-free graphs launched by one thread, for mapping coverage checks.
const char *twoGraphProgram() {
  return R"mlir(
module {
  dataflow.graph private @ga(%start: none) -> () attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.graph private @gb(%start: none) -> () attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)() ctrl (%c: none) {
    %da = dataflow.graph.launch @ga deps(%c) values() stream_inputs() memories() stream_outputs() : (none) -> none
    %db = dataflow.graph.launch @gb deps(%da) values() stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %db : none
  }
  func.func private @host() {
    %t = dataflow.thread.launch @t() : () -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// A none-payload stream program: pure-signal messages with no lane content.
const char *noneStreamProgram() {
  return R"mlir(
module {
  dataflow.graph private @gn(%start: none, %sig: none) -> () attributes {input_segments = array<i32: 0, 1, 0>, result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(%ch: !dataflow.channel<none>) ctrl (%c: none) {
    %n = dataflow.sync %c : (none) -> (none)
    dataflow.channel.send %ch, %n : !dataflow.channel<none>
    dataflow.thread.yield
  }
  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(%ch: !dataflow.channel<none>) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @gn deps(%ctrl) values() stream_inputs(%ch source_map affine_map<() -> ()>) memories() stream_outputs() : (none, !dataflow.channel<none>) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @host(%ch: !dataflow.channel<none>) {
    %p = dataflow.thread.launch @producer(%ch) : (!dataflow.channel<none>) -> !dataflow.thread_token
    %c = dataflow.thread.launch @consumer(%ch) : (!dataflow.channel<none>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

enum class ExposureLaunchKind {
  FreshAllocation,
  ImportedMemory,
  DerivedMemory
};

RootedGraphLaunchRef launchOf(const char *test,
                              const CanonicalDataflowProgramView &view,
                              ExposureLaunchKind kind) {
  for (const CanonicalStaticGraphLaunchView &site :
       view.staticGraphLaunches()) {
    auto graph =
        llvm::cast<GraphOp>(llvm::cantFail(view.resolve(site.callee)).op);
    auto launch = llvm::cast<GraphLaunchOp>(site.op);
    bool hasAllocation = false;
    graph.walk([&](mlir::memref::AllocOp) { hasAllocation = true; });
    bool importedMemory =
        !launch.getMemoryInputs().empty() &&
        llvm::isa<mlir::BlockArgument>(launch.getMemoryInputs().front());
    bool selected =
        (kind == ExposureLaunchKind::FreshAllocation && hasAllocation) ||
        (kind == ExposureLaunchKind::ImportedMemory && !hasAllocation &&
         importedMemory) ||
        (kind == ExposureLaunchKind::DerivedMemory && !hasAllocation &&
         !importedMemory);
    if (selected) {
      require(test, view.rootThreadLaunches().size() == 1,
              "fixture has one root launch");
      return RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                                  site.ref};
    }
  }
  fail(test, "no static launch matching semantic selector");
}

RootedGraphLaunchRef
launchAtThreadOrder(const char *test, const CanonicalDataflowProgramView &view,
                    unsigned ordinal) {
  require(test, view.rootThreadLaunches().size() == 1,
          "fixture has one root launch");
  llvm::SmallVector<mlir::Operation *> launches;
  view.rootThreadLaunches().front().callee->walk(
      [&](GraphLaunchOp launch) { launches.push_back(launch.getOperation()); });
  if (ordinal >= launches.size())
    fail(test, "thread launch ordinal is out of range");
  for (const CanonicalStaticGraphLaunchView &site : view.staticGraphLaunches())
    if (site.op == launches[ordinal])
      return RootedGraphLaunchRef{view.rootThreadLaunches().front().ref,
                                  site.ref};
  fail(test, "stored graph launch lacks a canonical static launch reference");
}

RootedGraphLaunchRef consumerLaunch(const char *test,
                                    const CanonicalDataflowProgramView &view) {
  for (const CanonicalRootThreadLaunchView &root : view.rootThreadLaunches()) {
    bool launchesGraph = false;
    root.callee->walk([&](GraphLaunchOp) { launchesGraph = true; });
    if (launchesGraph) {
      require(test, view.staticGraphLaunches().size() == 1,
              "fixture has one static launch");
      return RootedGraphLaunchRef{root.ref,
                                  view.staticGraphLaunches().front().ref};
    }
  }
  fail(test, "no graph-launching root thread");
}

std::vector<uint8_t> bytesOf(const CanonicalSemanticBytes &bytes) {
  return std::vector<uint8_t>(bytes.bytes().begin(), bytes.bytes().end());
}

void putU64Be(std::vector<uint8_t> &bytes, std::size_t offset,
              std::uint64_t value) {
  for (unsigned index = 0; index < 8; ++index)
    bytes[offset + index] =
        static_cast<std::uint8_t>(value >> (56 - 8 * index));
}

RuntimeMemoryObject byteObject(std::uint64_t count, std::uint8_t value) {
  return RuntimeMemoryObject{std::vector<SemanticMemoryByte>(
      count, SemanticMemoryByte{SemanticState::Defined, value})};
}

RuntimeMemoryObject littleEndianF32(std::uint64_t count, std::uint32_t bits) {
  RuntimeMemoryObject object;
  object.initialBytes.reserve(count * sizeof(bits));
  for (std::uint64_t element = 0; element < count; ++element)
    for (unsigned byte = 0; byte < sizeof(bits); ++byte)
      object.initialBytes.push_back(
          SemanticMemoryByte{SemanticState::Defined,
                             static_cast<std::uint8_t>(bits >> (byte * 8))});
  return object;
}

// The typed workload/runtime wire must drive the real DFG engine. A and B
// intentionally alias one object, so the expected store also anchors the
// neutral byte-addressed runtime-memory contract.
void typedDfgExecution() {
  const char *test = "typedDfgExecution";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, vecaddProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  LogicalMemoryRootRef rootA = rootByFormal(test, view, 0);
  LogicalMemoryRootRef rootB = rootByFormal(test, view, 1);
  LogicalMemoryRootRef rootC = rootByFormal(test, view, 2);
  SpatialSimulationWorkload workloadModel = makeVecaddWorkload(view);
  workloadModel.observableContract.memories.push_back(SpatialMemoryObservable{
      LogicalMemoryRootOrViewRef{rootA}, MemoryObservationForm::FullState});
  workloadModel.observableContract.memories.push_back(SpatialMemoryObservable{
      LogicalMemoryRootOrViewRef{rootB}, MemoryObservationForm::FullState});
  std::sort(workloadModel.observableContract.memories.begin(),
            workloadModel.observableContract.memories.end(),
            [](const SpatialMemoryObservable &lhs,
               const SpatialMemoryObservable &rhs) {
              return std::get<LogicalMemoryRootRef>(
                         std::get<LogicalMemoryRootOrViewRef>(lhs.target))
                         .entity.value() <
                     std::get<LogicalMemoryRootRef>(
                         std::get<LogicalMemoryRootOrViewRef>(rhs.target))
                         .entity.value();
            });
  llvm::Expected<CanonicalSimulationWorkload> workload =
      finalizeSimulationWorkload(workloadModel, view);
  if (!workload)
    fail(test, "workload finalization failed: " +
                   llvm::toString(workload.takeError()));

  SpatialSimulationRuntimeInputDraft draft{workload->identity()};
  draft.memoryObjects = {littleEndianF32(1025, 0x3f800000U),
                         littleEndianF32(1024, 0)};
  draft.memoryRootBindings = {
      RuntimeMemoryBindingDraft{rootA, 0, 0},
      RuntimeMemoryBindingDraft{rootB, 0, sizeof(float)},
      RuntimeMemoryBindingDraft{rootC, 1, 0}};
  llvm::Expected<CanonicalSimulationRuntimeInput> input =
      finalizeSimulationRuntimeInput(draft, *workload, view);
  if (!input)
    fail(test, "runtime-input finalization failed: " +
                   llvm::toString(input.takeError()));

  llvm::Expected<RetiredDFGSimulation> execution =
      simulateRetiredDfgWorkload(artifact, *workload, *input);
  if (!execution)
    fail(test, "typed DFG execution failed: " +
                   llvm::toString(execution.takeError()));
  DFGSimulationReport &report = execution->report;
  require(test,
          report.workload == formatArtifactIdentityHex(workload->identity()),
          "typed report does not identify its exact workload");
  require(test, report.status == "pass", "typed run did not retire");
  require(test,
          report.operationFireCounts[OperationSchemaId::DataflowLoad] == 2,
          "typed run did not execute both loads");
  require(test,
          report.operationFireCounts[OperationSchemaId::DataflowStore] == 1,
          "typed run did not execute the store");
  require(test, report.operationFireCounts[OperationSchemaId::ArithAddF] == 1,
          "typed run did not execute the add");
  auto output = report.finalMemoryState.find("arg3");
  require(test,
          output != report.finalMemoryState.end() &&
              output->second.size() == 1024 &&
              output->second.front() == "f32:2" &&
              output->second.back() == "f32:0",
          "typed aliased execution produced the wrong destination state");
  require(test, execution->observations.memories.size() == 3,
          "typed execution did not project its selected memories");
  const DiffMemoryObservation *diff = nullptr;
  const FullMemoryObservation *rootAState = nullptr;
  const FullMemoryObservation *rootBState = nullptr;
  for (std::size_t index = 0;
       index < workload->model().observableContract.memories.size(); ++index) {
    const SpatialMemoryObservable &observable =
        workload->model().observableContract.memories[index];
    const LogicalMemoryRootRef root = std::get<LogicalMemoryRootRef>(
        std::get<LogicalMemoryRootOrViewRef>(observable.target));
    if (root == rootA)
      rootAState = std::get_if<FullMemoryObservation>(
          &execution->observations.memories[index]);
    else if (root == rootB)
      rootBState = std::get_if<FullMemoryObservation>(
          &execution->observations.memories[index]);
    else if (root == rootC)
      diff = std::get_if<DiffMemoryObservation>(
          &execution->observations.memories[index]);
  }
  require(test,
          rootAState && rootAState->bytes.size() == 1025 * sizeof(float) &&
              rootBState && rootBState->bytes.size() == 1024 * sizeof(float),
          "typed alias projections did not preserve root-relative offsets");
  require(test,
          diff && diff->byteCount == 1024 * sizeof(float) &&
              diff->runs.size() == 1,
          "typed execution did not produce the canonical sparse diff");
  require(test,
          diff->runs.front().byteOffset == 3 &&
              diff->runs.front().changedBytes.size() == 1 &&
              diff->runs.front().changedBytes.front().state ==
                  SemanticState::Defined &&
              diff->runs.front().changedBytes.front().value == 0x40,
          "typed execution emitted a non-maximal or incorrect diff run");
}

// (b) Total Fixed/Runtime classification and exact lane-state validation.
void valueClassificationAndLanes() {
  const char *test = "valueClassificationAndLanes";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, valuesProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);

  auto base = [&]() {
    SpatialSimulationWorkload workload{onlyLaunch(test, view)};
    // Index input fixed; vector input fixed with four row-major lanes; the f32
    // input delegated to runtime.
    workload.valueInputPlan = {
        oneToken({definedLane(32, 8)}),
        oneToken({definedLane(32, 1), definedLane(32, 2), definedLane(32, 3),
                  definedLane(32, 4)}),
        SpatialValueInputSource{RuntimeValueInput{}}};
    workload.observableContract.valueResults = {0, 1};
    return workload;
  };
  require(test, !isRejected(finalizeSimulationWorkload(base(), view)),
          "a totally classified plan finalizes");

  // Totality: a missing or extra entry breaks the classification.
  {
    SpatialSimulationWorkload workload = base();
    workload.valueInputPlan.pop_back();
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "a non-total plan is rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.valueInputPlan.push_back(RuntimeValueInput{});
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "a plan with an extra entry is rejected");
  }
  // A fixed entry holds exactly one token.
  {
    SpatialSimulationWorkload workload = base();
    CanonicalValueSequence twoTokens;
    twoTokens.tokenCount = 2;
    twoTokens.lanes = {definedLane(32, 1), definedLane(32, 2)};
    workload.valueInputPlan[0] = twoTokens;
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "a two-token fixed value is rejected");
  }
  // Exact lane states: wrong scalar width, wrong vector lane width, and wrong
  // vector lane count are rejected; poison and undef lanes are accepted.
  {
    SpatialSimulationWorkload workload = base();
    workload.valueInputPlan[0] = oneToken({definedLane(16, 8)});
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "a defined lane of the wrong width is rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.valueInputPlan[1] =
        oneToken({definedLane(16, 1), definedLane(16, 2), definedLane(16, 3),
                  definedLane(16, 4)});
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "vector lanes of the wrong width are rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.valueInputPlan[1] =
        oneToken({definedLane(32, 1), definedLane(32, 2), definedLane(32, 3)});
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "a vector token with the wrong lane count is rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.valueInputPlan[1] =
        oneToken({definedLane(32, 1), SemanticLane::poison(),
                  SemanticLane::undef(), definedLane(32, 4)});
    require(test, !isRejected(finalizeSimulationWorkload(workload, view)),
            "poison and undef lanes are accepted");
  }

  // The runtime value table exactly complements the Runtime classifications.
  llvm::Expected<CanonicalSimulationWorkload> finalized =
      finalizeSimulationWorkload(base(), view);
  if (!finalized)
    fail(test, "baseline workload finalizes");
  auto draftFor = [&](std::vector<RuntimeValueEntry> values) {
    SpatialSimulationRuntimeInputDraft draft{finalized->identity()};
    draft.runtimeValues = std::move(values);
    return draft;
  };
  require(test,
          !isRejected(finalizeSimulationRuntimeInput(
              draftFor({RuntimeValueEntry{2, oneToken({definedLane(32, 7)})}}),
              *finalized, view)),
          "the exact runtime complement finalizes");
  llvm::Expected<CanonicalSimulationRuntimeInput> executionInput =
      finalizeSimulationRuntimeInput(
          draftFor({RuntimeValueEntry{2, oneToken({definedLane(32, 7)})}}),
          *finalized, view);
  if (!executionInput)
    fail(test, "value-result runtime input finalization failed");
  llvm::Expected<RetiredDFGSimulation> execution =
      simulateRetiredDfgWorkload(artifact, *finalized, *executionInput);
  if (!execution)
    fail(test, "value-result DFG execution failed: " +
                   llvm::toString(execution.takeError()));
  require(test, execution->observations.valueResults.size() == 2,
          "typed execution did not project both selected value results");
  const auto *vectorResult = std::get_if<PublishedValueResult>(
      &execution->observations.valueResults[0]);
  const auto *floatResult = std::get_if<PublishedValueResult>(
      &execution->observations.valueResults[1]);
  require(test,
          vectorResult && vectorResult->value.tokenCount == 1 &&
              vectorResult->value.lanes.size() == 4 &&
              vectorResult->value.lanes[0].bits == llvm::APInt(32, 1) &&
              vectorResult->value.lanes[3].bits == llvm::APInt(32, 4) &&
              floatResult && floatResult->value.tokenCount == 1 &&
              floatResult->value.lanes.size() == 1 &&
              floatResult->value.lanes.front().bits == llvm::APInt(32, 7),
          "typed execution changed value-result lane order or bits");
  require(test,
          isRejected(
              finalizeSimulationRuntimeInput(draftFor({}), *finalized, view)),
          "a missing runtime value is rejected");
  require(test,
          isRejected(finalizeSimulationRuntimeInput(
              draftFor({RuntimeValueEntry{1, oneToken({definedLane(32, 7)})}}),
              *finalized, view)),
          "a runtime value for a Fixed ordinal is rejected");
  {
    CanonicalValueSequence threeTokens;
    threeTokens.tokenCount = 3;
    threeTokens.lanes = {definedLane(32, 1), definedLane(32, 2),
                         definedLane(32, 3)};
    require(
        test,
        isRejected(finalizeSimulationRuntimeInput(
            draftFor({RuntimeValueEntry{2, threeTokens}}), *finalized, view)),
        "a multi-token runtime value is rejected");
  }
  {
    CanonicalDataflowArtifact other = finalizeProgram(test, foreignProgram());
    SpatialSimulationRuntimeInputDraft draft =
        draftFor({RuntimeValueEntry{2, oneToken({definedLane(32, 7)})}});
    draft.workloadIdentity = other.identity();
    require(test,
            isRejected(finalizeSimulationRuntimeInput(draft, *finalized, view)),
            "a runtime input naming a foreign workload is rejected");
  }
}

// (c) CanonicalStreamSequence and the runtime open/closed horizon state.
void streamHorizonAndCardinality() {
  const char *test = "streamHorizonAndCardinality";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, streamProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);

  SpatialSimulationWorkload workload{consumerLaunch(test, view)};
  llvm::Expected<CanonicalSimulationWorkload> finalized =
      finalizeSimulationWorkload(workload, view);
  if (!finalized)
    fail(test, "stream workload finalizes");

  auto streamOf = [](std::uint64_t tokens, StreamTermination termination) {
    CanonicalStreamSequence stream;
    stream.values.tokenCount = tokens;
    stream.values.lanes.assign(tokens, definedLane(32, 5));
    stream.termination = termination;
    return stream;
  };
  auto draftWith = [&](std::vector<CanonicalStreamSequence> streams) {
    SpatialSimulationRuntimeInputDraft draft{finalized->identity()};
    draft.runtimeStreams = std::move(streams);
    return draft;
  };
  require(test,
          !isRejected(finalizeSimulationRuntimeInput(
              draftWith({streamOf(3, StreamTermination::ClosedAfterLast)}),
              *finalized, view)),
          "a closed stream finalizes");
  require(test,
          !isRejected(finalizeSimulationRuntimeInput(
              draftWith({streamOf(2, StreamTermination::OpenAfterLast)}),
              *finalized, view)),
          "an open stream finalizes: no later token or close exists");
  require(test,
          !isRejected(finalizeSimulationRuntimeInput(
              draftWith({streamOf(0, StreamTermination::OpenAfterLast)}),
              *finalized, view)),
          "an empty open stream finalizes");
  require(test,
          isRejected(
              finalizeSimulationRuntimeInput(draftWith({}), *finalized, view)),
          "a non-total stream table is rejected");
  require(test,
          isRejected(finalizeSimulationRuntimeInput(
              draftWith({streamOf(1, StreamTermination::ClosedAfterLast),
                         streamOf(1, StreamTermination::ClosedAfterLast)}),
              *finalized, view)),
          "an over-total stream table is rejected");
  {
    CanonicalStreamSequence wrongWidth =
        streamOf(1, StreamTermination::ClosedAfterLast);
    wrongWidth.values.lanes = {definedLane(16, 5)};
    require(test,
            isRejected(finalizeSimulationRuntimeInput(draftWith({wrongWidth}),
                                                      *finalized, view)),
            "a stream lane of the wrong width is rejected");
  }

  // The horizon state survives the canonical round trip byte-exactly.
  llvm::Expected<CanonicalSimulationRuntimeInput> input =
      finalizeSimulationRuntimeInput(
          draftWith({streamOf(2, StreamTermination::OpenAfterLast)}),
          *finalized, view);
  if (!input)
    fail(test, "open-stream input finalizes");
  llvm::Expected<CanonicalSimulationRuntimeInput> again =
      importSimulationRuntimeInput(input->canonicalBytes().bytes(), *finalized,
                                   view, input->identity());
  if (!again)
    fail(test, "the open-stream input imports");
  require(test,
          again->model().runtimeStreams[0].termination ==
              StreamTermination::OpenAfterLast,
          "the open horizon state survives the round trip");
  llvm::Expected<DFGSimulationReport> report =
      simulateDfgWorkload(artifact, *finalized, *again);
  if (!report)
    fail(test, "typed stream admission failed: " +
                   llvm::toString(report.takeError()));
  require(test,
          report->status == "unsupported" && report->diagnostics.size() == 1 &&
              report->diagnostics.front() ==
                  "typed runtime stream termination is unsupported",
          "DFG-sim must fail closed until typed stream horizons are modeled");
}

// (d) Canonical object-ordinal invariance under author ordering and
// shared-ordinal aliasing, including legal overlap.
void memoryObjectOrdinals() {
  const char *test = "memoryObjectOrdinals";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, vecaddProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  SpatialSimulationWorkload workload = makeVecaddWorkload(view);
  llvm::Expected<CanonicalSimulationWorkload> finalized =
      finalizeSimulationWorkload(workload, view);
  if (!finalized)
    fail(test, "vecadd workload finalizes");
  LogicalMemoryRootRef rootA = rootByFormal(test, view, 0);
  LogicalMemoryRootRef rootB = rootByFormal(test, view, 1);
  LogicalMemoryRootRef rootC = rootByFormal(test, view, 2);

  auto finalizeDraft = [&](std::vector<RuntimeMemoryObject> objects,
                           std::vector<RuntimeMemoryBindingDraft> bindings) {
    SpatialSimulationRuntimeInputDraft draft{finalized->identity()};
    draft.memoryObjects = std::move(objects);
    draft.memoryRootBindings = std::move(bindings);
    return finalizeSimulationRuntimeInput(draft, *finalized, view);
  };

  // Author ordering of the object array and binding list does not change the
  // canonical bytes or identity.
  llvm::Expected<CanonicalSimulationRuntimeInput> first = finalizeDraft(
      {byteObject(16, 0x0A), byteObject(16, 0x0B), byteObject(16, 0x0C)},
      {RuntimeMemoryBindingDraft{rootA, 0, 0},
       RuntimeMemoryBindingDraft{rootB, 1, 0},
       RuntimeMemoryBindingDraft{rootC, 2, 0}});
  llvm::Expected<CanonicalSimulationRuntimeInput> second = finalizeDraft(
      {byteObject(16, 0x0C), byteObject(16, 0x0A), byteObject(16, 0x0B)},
      {RuntimeMemoryBindingDraft{rootC, 0, 0},
       RuntimeMemoryBindingDraft{rootA, 1, 0},
       RuntimeMemoryBindingDraft{rootB, 2, 0}});
  if (!first || !second)
    fail(test, "both author orderings finalize");
  require(test,
          first->identity() == second->identity() &&
              bytesOf(first->canonicalBytes()) ==
                  bytesOf(second->canonicalBytes()),
          "canonical object ordinals are invariant under author ordering");

  // Two roots aliasing one object, with overlapping ranges, is legal.
  llvm::Expected<CanonicalSimulationRuntimeInput> aliased =
      finalizeDraft({byteObject(64, 0xAB), byteObject(16, 0x0C)},
                    {RuntimeMemoryBindingDraft{rootA, 0, 0},
                     RuntimeMemoryBindingDraft{rootB, 0, 16},
                     RuntimeMemoryBindingDraft{rootC, 1, 0}});
  if (!aliased)
    fail(test, "overlapping same-object aliases finalize");
  const SpatialSimulationRuntimeInput &aliasModel = aliased->model();
  auto bindingOf = [&](const dataflow::LogicalMemoryRootRef &root) {
    for (const MemoryRootBindingEntry &entry : aliasModel.memoryRootBindings)
      if (entry.root == root)
        return entry.binding;
    fail(test, "binding present");
  };
  require(test,
          bindingOf(rootA).objectOrdinal == bindingOf(rootB).objectOrdinal &&
              bindingOf(rootA).objectOrdinal != bindingOf(rootC).objectOrdinal,
          "aliased roots share one canonical object ordinal");

  // Invalid objects and bindings fail closed.
  require(test,
          isRejected(finalizeDraft({byteObject(16, 0), byteObject(16, 0),
                                    byteObject(16, 0), byteObject(16, 0)},
                                   {RuntimeMemoryBindingDraft{rootA, 0, 0},
                                    RuntimeMemoryBindingDraft{rootB, 1, 0},
                                    RuntimeMemoryBindingDraft{rootC, 2, 0}})),
          "an unreferenced object is rejected");
  require(test,
          isRejected(finalizeDraft({byteObject(16, 0), byteObject(16, 0)},
                                   {RuntimeMemoryBindingDraft{rootA, 0, 0},
                                    RuntimeMemoryBindingDraft{rootB, 1, 0}})),
          "a missing root binding is rejected");
  require(test,
          isRejected(finalizeDraft(
              {byteObject(16, 0), byteObject(16, 0), byteObject(16, 0)},
              {RuntimeMemoryBindingDraft{rootA, 0, 0},
               RuntimeMemoryBindingDraft{rootA, 1, 0},
               RuntimeMemoryBindingDraft{rootB, 2, 0},
               RuntimeMemoryBindingDraft{rootC, 0, 0}})),
          "a duplicate root binding is rejected");
  require(test,
          isRejected(finalizeDraft(
              {byteObject(16, 0), byteObject(16, 0), byteObject(16, 0)},
              {RuntimeMemoryBindingDraft{rootA, 7, 0},
               RuntimeMemoryBindingDraft{rootB, 1, 0},
               RuntimeMemoryBindingDraft{rootC, 2, 0}})),
          "an out-of-range object ordinal is rejected");
  require(test,
          isRejected(finalizeDraft(
              {byteObject(16, 0), byteObject(16, 0), byteObject(16, 0)},
              {RuntimeMemoryBindingDraft{rootA, 0, 16},
               RuntimeMemoryBindingDraft{rootB, 1, 0},
               RuntimeMemoryBindingDraft{rootC, 2, 0}})),
          "an out-of-range byte offset is rejected");
  require(test,
          isRejected(finalizeDraft(
              {RuntimeMemoryObject{}, byteObject(16, 0), byteObject(16, 0)},
              {RuntimeMemoryBindingDraft{rootA, 0, 0},
               RuntimeMemoryBindingDraft{rootB, 1, 0},
               RuntimeMemoryBindingDraft{rootC, 2, 0}})),
          "an empty memory object is rejected");

  // An unrelated root (a thread formal the launch never binds) is rejected.
  {
    CanonicalDataflowArtifact rules =
        finalizeProgram(test, bindingRulesProgram());
    CanonicalDataflowProgramView rulesView = viewOf(test, rules);
    SpatialSimulationWorkload rulesWorkload{onlyLaunch(test, rulesView)};
    llvm::Expected<CanonicalSimulationWorkload> rulesFinalized =
        finalizeSimulationWorkload(rulesWorkload, rulesView);
    if (!rulesFinalized)
      fail(test, "binding-rules workload finalizes");
    LogicalMemoryRootRef bound = rootByFormal(test, rulesView, 0);
    LogicalMemoryRootRef unbound = rootByFormal(test, rulesView, 1);
    SpatialSimulationRuntimeInputDraft draft{rulesFinalized->identity()};
    draft.memoryObjects = {byteObject(16, 0), byteObject(16, 0)};
    draft.memoryRootBindings = {RuntimeMemoryBindingDraft{bound, 0, 0},
                                RuntimeMemoryBindingDraft{unbound, 1, 0}};
    require(test,
            isRejected(finalizeSimulationRuntimeInput(draft, *rulesFinalized,
                                                      rulesView)),
            "a binding for an unrelated root is rejected");
  }

  // The parser enforces the canonical ordinal order on the wire: swapping two
  // serialized object ordinals is rejected without repair.
  std::vector<uint8_t> wire = bytesOf(first->canonicalBytes());
  const std::size_t record = 56; // root(40) + object ordinal(8) + offset(8)
  require(test, wire.size() >= 3 * record, "wire holds three bindings");
  const std::size_t base = wire.size() - 3 * record;
  std::vector<uint8_t> swapped = wire;
  std::uint64_t ordinal0 = 0, ordinal1 = 0;
  for (unsigned i = 0; i < 8; ++i) {
    ordinal0 = (ordinal0 << 8) | swapped[base + 40 + i];
    ordinal1 = (ordinal1 << 8) | swapped[base + record + 40 + i];
  }
  putU64Be(swapped, base + 40, ordinal1);
  putU64Be(swapped, base + record + 40, ordinal0);
  require(test,
          isRejected(importSimulationRuntimeInput(swapped, *finalized, view,
                                                  first->identity())),
          "noncanonical object ordinals on the wire are rejected");
  require(test,
          !isRejected(importSimulationRuntimeInput(wire, *finalized, view,
                                                   first->identity())),
          "the untouched wire imports");
}

// (e) Observable sorting, duplicates, and mandatory completion derivation.
void observableContractRules() {
  const char *test = "observableContractRules";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, valuesProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  auto base = [&]() {
    SpatialSimulationWorkload workload{onlyLaunch(test, view)};
    workload.valueInputPlan = {SpatialValueInputSource{RuntimeValueInput{}},
                               SpatialValueInputSource{RuntimeValueInput{}},
                               SpatialValueInputSource{RuntimeValueInput{}}};
    return workload;
  };
  {
    SpatialSimulationWorkload workload = base();
    workload.observableContract.valueResults = {1, 0};
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "unsorted value results are rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.observableContract.valueResults = {1, 1};
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "duplicate value results are rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.observableContract.valueResults = {2};
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "an out-of-range value result is rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.observableContract.streamOutputs = {0};
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "an out-of-range stream output is rejected");
  }
  // Completion is derived, never serialized: two workloads differing only by
  // nothing else have identical bytes, and the derivation names the launch.
  llvm::Expected<CanonicalSimulationWorkload> one =
      finalizeSimulationWorkload(base(), view);
  if (!one)
    fail(test, "observable workload finalizes");
  require(test,
          one->completion() ==
              GraphLaunchDoneTransferRef{one->model().launchRef},
          "mandatory completion derives from the rooted launch");
  require(test,
          !errored(
              view.validate(GraphLaunchBoundaryTransferRef{one->completion()})),
          "the derived completion transfer is valid in the owner view");
}

// (f) Malformed, trailing, unsorted, unknown-variant, stale-reference, and
// System-root wire failures.
void wireRejections() {
  const char *test = "wireRejections";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, vecaddProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);

  SpatialSimulationWorkload workload = makeVecaddWorkload(view);
  LogicalMemoryRootRef rootA = rootByFormal(test, view, 0);
  LogicalMemoryRootRef rootB = rootByFormal(test, view, 1);
  workload.observableContract.memories.push_back(SpatialMemoryObservable{
      LogicalMemoryRootOrViewRef{rootA}, MemoryObservationForm::FullState});
  workload.observableContract.memories.push_back(SpatialMemoryObservable{
      LogicalMemoryRootOrViewRef{rootB}, MemoryObservationForm::FullState});
  std::sort(workload.observableContract.memories.begin(),
            workload.observableContract.memories.end(),
            [](const SpatialMemoryObservable &lhs,
               const SpatialMemoryObservable &rhs) {
              return std::get<LogicalMemoryRootRef>(
                         std::get<LogicalMemoryRootOrViewRef>(lhs.target))
                         .entity.value() <
                     std::get<LogicalMemoryRootRef>(
                         std::get<LogicalMemoryRootOrViewRef>(rhs.target))
                         .entity.value();
            });
  llvm::Expected<CanonicalSimulationWorkload> finalized =
      finalizeSimulationWorkload(workload, view);
  if (!finalized)
    fail(test, "observable vecadd workload finalizes");
  std::vector<uint8_t> wire = bytesOf(finalized->canonicalBytes());

  auto import = [&](std::vector<uint8_t> bytes) {
    return importSimulationWorkload(bytes, view, finalized->identity());
  };
  require(test, !isRejected(import(wire)), "the canonical wire imports");
  {
    std::vector<uint8_t> bad = wire;
    bad.push_back(0);
    require(test, isRejected(import(std::move(bad))),
            "trailing bytes are rejected");
  }
  {
    std::vector<uint8_t> bad = wire;
    bad.pop_back();
    require(test, isRejected(import(std::move(bad))),
            "truncated bytes are rejected");
  }
  {
    std::vector<uint8_t> bad = wire;
    bad[3] = 1;
    require(test, isRejected(import(std::move(bad))),
            "the System root discriminant fails closed");
  }
  {
    std::vector<uint8_t> bad = wire;
    bad[3] = 9;
    require(test, isRejected(import(std::move(bad))),
            "an unknown root discriminant is rejected");
  }
  {
    // The value-input source discriminant of the single plan entry: root tag
    // (4) + launch refs (80) + coordinate count (8) + plan count (8) + entry
    // ordinal (8) = offset 108.
    std::vector<uint8_t> bad = wire;
    bad[108] = 7;
    require(test, isRejected(import(std::move(bad))),
            "an unknown value-input source discriminant is rejected");
  }
  {
    // The static graph launch entity ID: root tag (4) + root identity (32) +
    // root entity (8) + static identity (32) = offset 76; bump the low byte.
    std::vector<uint8_t> bad = wire;
    bad[83] += 1;
    require(test, isRejected(import(std::move(bad))),
            "a stale static launch reference is rejected");
  }
  {
    // The three memory observables are the payload tail, each a 52-byte
    // record (kind 4 + root-or-view tag 4 + root 40 + form 4). Reversing the
    // records breaks the sorted table; duplicating one breaks uniqueness.
    const std::size_t record = 52;
    const std::size_t base = wire.size() - 3 * record;
    std::vector<uint8_t> reversed = wire;
    for (std::size_t k = 0; k < 3; ++k)
      std::memcpy(reversed.data() + base + k * record,
                  wire.data() + base + (2 - k) * record, record);
    require(test, isRejected(import(std::move(reversed))),
            "an unsorted observable table is rejected");
    std::vector<uint8_t> duplicated = wire;
    std::memcpy(duplicated.data() + base + record, wire.data() + base, record);
    require(test, isRejected(import(std::move(duplicated))),
            "a duplicate observable entry is rejected");
  }
  {
    CanonicalDataflowArtifact other = finalizeProgram(test, foreignProgram());
    require(test,
            isRejected(importSimulationWorkload(wire, view, other.identity())),
            "a foreign expected identity is rejected");
  }
}

// (g) Dense coordinate rank and static grid bounds.
void denseCoordinates() {
  const char *test = "denseCoordinates";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, gridProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  auto withCoords = [&](std::vector<std::uint64_t> coords) {
    SpatialSimulationWorkload workload{onlyLaunch(test, view)};
    workload.denseCoordinates = std::move(coords);
    return finalizeSimulationWorkload(workload, view);
  };
  require(test, !isRejected(withCoords({2, 5})),
          "an in-bounds dense point finalizes");
  require(test, !isRejected(withCoords({0, 0})), "the origin finalizes");
  require(test, isRejected(withCoords({})), "a rank-short point is rejected");
  require(test, isRejected(withCoords({2})), "a rank-short point is rejected");
  require(test, isRejected(withCoords({2, 5, 0})),
          "a rank-long point is rejected");
  require(test, isRejected(withCoords({4, 0})),
          "a coordinate at the static bound is rejected");
  require(test, isRejected(withCoords({3, 8})),
          "a coordinate past the static bound is rejected");
}

// (h) Exposure targets and diff-baseline eligibility.
void exposureTargetsAndDiffBaseline() {
  const char *test = "exposureTargetsAndDiffBaseline";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, exposureProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  RootedGraphLaunchRef freshLaunch =
      launchOf(test, view, ExposureLaunchKind::FreshAllocation);
  RootedGraphLaunchRef passthroughLaunch =
      launchOf(test, view, ExposureLaunchKind::ImportedMemory);
  LogicalMemoryRootRef rootM = rootByFormal(test, view, 0);

  auto workloadOn = [&](RootedGraphLaunchRef launch,
                        MemoryObservationForm form) {
    SpatialSimulationWorkload workload{launch};
    workload.observableContract.memories.push_back(
        SpatialMemoryObservable{MemoryExposureTarget{0}, form});
    return workload;
  };
  auto inputFor = [&](const CanonicalSimulationWorkload &finalizedWorkload) {
    SpatialSimulationRuntimeInputDraft draft{finalizedWorkload.identity()};
    draft.memoryObjects = {byteObject(40, 0x11)};
    draft.memoryRootBindings = {RuntimeMemoryBindingDraft{rootM, 0, 0}};
    return finalizeSimulationRuntimeInput(draft, finalizedWorkload, view);
  };

  // A passthrough exposure names the imported root: a diff baseline exists.
  llvm::Expected<CanonicalSimulationWorkload> observedImported =
      finalizeSimulationWorkload(
          workloadOn(passthroughLaunch,
                     MemoryObservationForm::DiffFromRuntimeInput),
          view);
  if (!observedImported)
    fail(test, "the passthrough-exposure workload finalizes");
  require(test, !isRejected(inputFor(*observedImported)),
          "a diff over an imported-root exposure has a baseline");

  // A fresh-allocation exposure has no runtime byte baseline: the diff form
  // is rejected, FullState is accepted.
  llvm::Expected<CanonicalSimulationWorkload> observedFreshDiff =
      finalizeSimulationWorkload(
          workloadOn(freshLaunch, MemoryObservationForm::DiffFromRuntimeInput),
          view);
  if (!observedFreshDiff)
    fail(test, "the fresh-exposure workload finalizes");
  require(test, isRejected(inputFor(*observedFreshDiff)),
          "a diff over a fresh-allocation exposure is rejected");
  llvm::Expected<CanonicalSimulationWorkload> observedFreshFull =
      finalizeSimulationWorkload(
          workloadOn(freshLaunch, MemoryObservationForm::FullState), view);
  if (!observedFreshFull)
    fail(test, "the fresh-exposure FullState workload finalizes");
  require(test, !isRejected(inputFor(*observedFreshFull)),
          "a full-state observation of a fresh exposure is accepted");
}

// (i) DFG admission: same-root validation and owner-mismatch rejection.
void admissionAdapters() {
  const char *test = "admissionAdapters";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, twoGraphProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  RootedGraphLaunchRef launchA = launchAtThreadOrder(test, view, 0);
  GraphRef graphA = llvm::cantFail(view.resolve(launchA));

  SpatialSimulationWorkload workload{launchA};
  llvm::Expected<CanonicalSimulationWorkload> finalizedWorkload =
      finalizeSimulationWorkload(workload, view);
  if (!finalizedWorkload)
    fail(test, "admission workload finalizes");
  SpatialSimulationRuntimeInputDraft draft{finalizedWorkload->identity()};
  llvm::Expected<CanonicalSimulationRuntimeInput> finalizedInput =
      finalizeSimulationRuntimeInput(draft, *finalizedWorkload, view);
  if (!finalizedInput)
    fail(test, "admission runtime input finalizes");

  llvm::Expected<GraphRef> admitted =
      admitDfgSpatialSimulation(*finalizedWorkload, *finalizedInput, view);
  require(test, admitted && *admitted == graphA,
          "DFG admission resolves the called graph through the same rooted "
          "launch");
  {
    CanonicalDataflowArtifact other = finalizeProgram(test, foreignProgram());
    CanonicalDataflowProgramView otherView = viewOf(test, other);
    require(test,
            isRejected(admitDfgSpatialSimulation(*finalizedWorkload,
                                                 *finalizedInput, otherView)),
            "DFG admission rejects a foreign Dataflow owner");
  }
}

// (j) Bounded scale observation: values, streams, and memory bindings with
// elapsed time and peak RSS.
void scaleObservation() {
  const char *test = "scaleObservation";
  using clock = std::chrono::steady_clock;
  const auto begin = clock::now();

  CanonicalDataflowArtifact streamArtifact =
      finalizeProgram(test, streamProgram());
  CanonicalDataflowProgramView streamView = viewOf(test, streamArtifact);
  SpatialSimulationWorkload streamWorkload{consumerLaunch(test, streamView)};
  llvm::Expected<CanonicalSimulationWorkload> finalizedStream =
      finalizeSimulationWorkload(streamWorkload, streamView);
  if (!finalizedStream)
    fail(test, "scale stream workload finalizes");

  const std::uint64_t tokenCount = 1u << 20;
  CanonicalStreamSequence stream;
  stream.values.tokenCount = tokenCount;
  stream.values.lanes.assign(tokenCount, definedLane(32, 42));
  stream.termination = StreamTermination::OpenAfterLast;
  SpatialSimulationRuntimeInputDraft streamDraft{finalizedStream->identity()};
  streamDraft.runtimeStreams = {std::move(stream)};
  llvm::Expected<CanonicalSimulationRuntimeInput> streamInput =
      finalizeSimulationRuntimeInput(streamDraft, *finalizedStream, streamView);
  if (!streamInput)
    fail(test, "scale stream input finalizes");
  require(test,
          !isRejected(importSimulationRuntimeInput(
              streamInput->canonicalBytes().bytes(), *finalizedStream,
              streamView, streamInput->identity())),
          "the scale stream input round-trips");

  CanonicalDataflowArtifact vecaddArtifact =
      finalizeProgram(test, vecaddProgram());
  CanonicalDataflowProgramView vecaddView = viewOf(test, vecaddArtifact);
  SpatialSimulationWorkload vecaddWorkload = makeVecaddWorkload(vecaddView);
  llvm::Expected<CanonicalSimulationWorkload> finalizedVecadd =
      finalizeSimulationWorkload(vecaddWorkload, vecaddView);
  if (!finalizedVecadd)
    fail(test, "scale vecadd workload finalizes");
  SpatialSimulationRuntimeInputDraft vecaddDraft{finalizedVecadd->identity()};
  const std::uint64_t objectBytes = 1u << 20;
  vecaddDraft.memoryObjects = {byteObject(objectBytes, 0x0A),
                               byteObject(objectBytes, 0x0B),
                               byteObject(objectBytes, 0x0C)};
  vecaddDraft.memoryRootBindings = {
      RuntimeMemoryBindingDraft{rootByFormal(test, vecaddView, 0), 0, 0},
      RuntimeMemoryBindingDraft{rootByFormal(test, vecaddView, 1), 1, 0},
      RuntimeMemoryBindingDraft{rootByFormal(test, vecaddView, 2), 2, 0}};
  llvm::Expected<CanonicalSimulationRuntimeInput> vecaddInput =
      finalizeSimulationRuntimeInput(vecaddDraft, *finalizedVecadd, vecaddView);
  if (!vecaddInput)
    fail(test, "scale vecadd input finalizes");
  require(test,
          !isRejected(importSimulationRuntimeInput(
              vecaddInput->canonicalBytes().bytes(), *finalizedVecadd,
              vecaddView, vecaddInput->identity())),
          "the scale vecadd input round-trips");

  const auto end = clock::now();
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  llvm::outs() << "scale: " << tokenCount << " stream lanes, 3x" << objectBytes
               << " object bytes, 3 bindings: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                        begin)
                      .count()
               << " ms, peak RSS " << usage.ru_maxrss / 1024 << " MiB\n";
}

// (k) Finalizers reject out-of-domain enum values and hidden noncanonical
// payloads instead of emitting parser-rejected bytes.
void enumStrictness() {
  const char *test = "enumStrictness";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, valuesProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  auto base = [&]() {
    SpatialSimulationWorkload workload{onlyLaunch(test, view)};
    workload.valueInputPlan = {
        oneToken({definedLane(32, 8)}),
        oneToken({definedLane(32, 1), definedLane(32, 2), definedLane(32, 3),
                  definedLane(32, 4)}),
        SpatialValueInputSource{RuntimeValueInput{}}};
    return workload;
  };
  {
    SpatialSimulationWorkload workload = base();
    workload.valueInputPlan[0] =
        oneToken({SemanticLane{static_cast<SemanticState>(9), llvm::APInt()}});
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "an out-of-domain lane state is rejected at finalization");
  }
  {
    SpatialSimulationWorkload workload = base();
    SemanticLane hidden = SemanticLane::poison();
    hidden.bits = llvm::APInt(32, 7);
    workload.valueInputPlan[0] = oneToken({hidden});
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "a poison lane carrying hidden payload bits is rejected");
  }
  {
    SpatialSimulationWorkload workload = base();
    workload.observableContract.memories.push_back(SpatialMemoryObservable{
        MemoryExposureTarget{0}, static_cast<MemoryObservationForm>(9)});
    require(test, isRejected(finalizeSimulationWorkload(workload, view)),
            "an out-of-domain observation form is rejected at finalization");
  }
  {
    CanonicalDataflowArtifact streamArtifact =
        finalizeProgram(test, streamProgram());
    CanonicalDataflowProgramView streamView = viewOf(test, streamArtifact);
    SpatialSimulationWorkload streamWorkload{consumerLaunch(test, streamView)};
    llvm::Expected<CanonicalSimulationWorkload> finalizedStream =
        finalizeSimulationWorkload(streamWorkload, streamView);
    if (!finalizedStream)
      fail(test, "stream workload finalizes");
    SpatialSimulationRuntimeInputDraft draft{finalizedStream->identity()};
    CanonicalStreamSequence stream;
    stream.values.tokenCount = 1;
    stream.values.lanes = {definedLane(32, 1)};
    stream.termination = static_cast<StreamTermination>(7);
    draft.runtimeStreams = {stream};
    require(test,
            isRejected(finalizeSimulationRuntimeInput(draft, *finalizedStream,
                                                      streamView)),
            "an out-of-domain stream termination is rejected");
  }
  {
    CanonicalDataflowArtifact vecaddArtifact =
        finalizeProgram(test, vecaddProgram());
    CanonicalDataflowProgramView vecaddView = viewOf(test, vecaddArtifact);
    SpatialSimulationWorkload vecaddWorkload = makeVecaddWorkload(vecaddView);
    llvm::Expected<CanonicalSimulationWorkload> finalizedVecadd =
        finalizeSimulationWorkload(vecaddWorkload, vecaddView);
    if (!finalizedVecadd)
      fail(test, "vecadd workload finalizes");
    SpatialSimulationRuntimeInputDraft draft{finalizedVecadd->identity()};
    draft.memoryObjects = {
        byteObject(16, 0x0A),
        RuntimeMemoryObject{{SemanticMemoryByte{SemanticState::Poison, 0xFF}}},
        byteObject(16, 0x0C)};
    draft.memoryRootBindings = {
        RuntimeMemoryBindingDraft{rootByFormal(test, vecaddView, 0), 0, 0},
        RuntimeMemoryBindingDraft{rootByFormal(test, vecaddView, 1), 1, 0},
        RuntimeMemoryBindingDraft{rootByFormal(test, vecaddView, 2), 2, 0}};
    require(test,
            isRejected(finalizeSimulationRuntimeInput(draft, *finalizedVecadd,
                                                      vecaddView)),
            "a poison memory byte carrying a hidden value is rejected");
    SpatialSimulationRuntimeInputDraft badState{finalizedVecadd->identity()};
    badState.memoryObjects = {byteObject(16, 0x0A),
                              RuntimeMemoryObject{{SemanticMemoryByte{
                                  static_cast<SemanticState>(9), 0}}},
                              byteObject(16, 0x0C)};
    badState.memoryRootBindings = draft.memoryRootBindings;
    require(test,
            isRejected(finalizeSimulationRuntimeInput(
                badState, *finalizedVecadd, vecaddView)),
            "an out-of-domain memory-byte state is rejected");
  }
}

// (l) A direct root-or-view observable is valid for any root reachable in the
// rooted launch, including a fresh allocation owned by the called graph.
void directMemoryObservables() {
  const char *test = "directMemoryObservables";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, exposureProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  RootedGraphLaunchRef freshLaunch =
      launchOf(test, view, ExposureLaunchKind::FreshAllocation);
  LogicalMemoryRootRef rootM = rootByFormal(test, view, 0);
  LogicalMemoryRootRef freshRoot = rootM;
  for (const CanonicalLogicalMemoryRootView &root : view.logicalMemoryRoots())
    if (!root.formalArgIndex)
      freshRoot = root.ref;
  require(test, freshRoot != rootM, "fixture has a fresh allocation root");

  auto workloadOn = [&](dataflow::LogicalMemoryRootRef root,
                        MemoryObservationForm form) {
    SpatialSimulationWorkload workload{freshLaunch};
    workload.observableContract.memories.push_back(
        SpatialMemoryObservable{LogicalMemoryRootOrViewRef{root}, form});
    return workload;
  };
  // A fresh allocation owned by the called graph is a legal FullState target.
  require(test,
          !isRejected(finalizeSimulationWorkload(
              workloadOn(freshRoot, MemoryObservationForm::FullState), view)),
          "a fresh graph allocation is a legal direct observable");
  // An imported root of the launch remains legal in both forms.
  require(test,
          !isRejected(finalizeSimulationWorkload(
              workloadOn(rootM, MemoryObservationForm::FullState), view)),
          "an imported root is a legal direct observable");
  // A root unreachable from the rooted launch is rejected.
  {
    CanonicalDataflowArtifact vecaddArtifact =
        finalizeProgram(test, vecaddProgram());
    CanonicalDataflowProgramView vecaddView = viewOf(test, vecaddArtifact);
    SpatialSimulationWorkload unrelated{freshLaunch};
    unrelated.observableContract.memories.push_back(SpatialMemoryObservable{
        LogicalMemoryRootOrViewRef{rootByFormal(test, vecaddView, 0)},
        MemoryObservationForm::FullState});
    require(test, isRejected(finalizeSimulationWorkload(unrelated, view)),
            "a root from an unrelated artifact is rejected");
  }
  // The fresh root has no runtime baseline: the diff form is rejected.
  {
    llvm::Expected<CanonicalSimulationWorkload> diffOnFresh =
        finalizeSimulationWorkload(
            workloadOn(freshRoot, MemoryObservationForm::DiffFromRuntimeInput),
            view);
    if (!diffOnFresh)
      fail(test, "the diff-on-fresh workload finalizes");
    SpatialSimulationRuntimeInputDraft draft{diffOnFresh->identity()};
    draft.memoryObjects = {byteObject(40, 0x11)};
    draft.memoryRootBindings = {RuntimeMemoryBindingDraft{rootM, 0, 0}};
    require(
        test,
        isRejected(finalizeSimulationRuntimeInput(draft, *diffOnFresh, view)),
        "a diff over a fresh root has no baseline and is rejected");
  }
}

// (m) None-payload streams: token count may be nonzero while lanes per token
// is zero and the lane array is empty.
void noneTokenStreams() {
  const char *test = "noneTokenStreams";
  CanonicalDataflowArtifact artifact =
      finalizeProgram(test, noneStreamProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  SpatialSimulationWorkload workload{consumerLaunch(test, view)};
  llvm::Expected<CanonicalSimulationWorkload> finalized =
      finalizeSimulationWorkload(workload, view);
  if (!finalized)
    fail(test, "none-stream workload finalizes");

  auto noneStream = [](std::uint64_t tokens) {
    CanonicalStreamSequence stream;
    stream.values.tokenCount = tokens;
    stream.termination = StreamTermination::ClosedAfterLast;
    return stream;
  };
  auto draftWith = [&](CanonicalStreamSequence stream) {
    SpatialSimulationRuntimeInputDraft draft{finalized->identity()};
    draft.runtimeStreams = {std::move(stream)};
    return draft;
  };
  llvm::Expected<CanonicalSimulationRuntimeInput> input =
      finalizeSimulationRuntimeInput(draftWith(noneStream(3)), *finalized,
                                     view);
  if (!input)
    fail(test, "a three-token none stream finalizes");
  llvm::Expected<CanonicalSimulationRuntimeInput> imported =
      importSimulationRuntimeInput(input->canonicalBytes().bytes(), *finalized,
                                   view, input->identity());
  if (!imported)
    fail(test, "the none stream imports");
  require(test,
          imported->model().runtimeStreams[0].values.tokenCount == 3 &&
              imported->model().runtimeStreams[0].values.lanes.empty(),
          "the none token count survives the round trip with empty lanes");
  require(test,
          !isRejected(finalizeSimulationRuntimeInput(draftWith(noneStream(0)),
                                                     *finalized, view)),
          "an empty none stream finalizes");
  {
    CanonicalStreamSequence withLane = noneStream(1);
    withLane.values.lanes = {definedLane(32, 1)};
    require(test,
            isRejected(finalizeSimulationRuntimeInput(draftWith(withLane),
                                                      *finalized, view)),
            "a none stream carrying lanes is rejected");
  }
}

// (n) A fresh allocation exposed by one launch and bound into the next stays
// unbound: fresh roots never join the imported-root registry, even through an
// exposure chain.
void freshExposureChain() {
  const char *test = "freshExposureChain";
  CanonicalDataflowArtifact artifact = finalizeProgram(test, exposureProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  RootedGraphLaunchRef chainLaunch =
      launchOf(test, view, ExposureLaunchKind::DerivedMemory);
  LogicalMemoryRootRef rootM = rootByFormal(test, view, 0);
  LogicalMemoryRootOrViewRef chainRole =
      llvm::cantFail(view.resolveExposure(MemoryExposureRef{chainLaunch, 0}));
  require(test, std::holds_alternative<LogicalMemoryRootRef>(chainRole),
          "the downstream exposure resolves to a root");
  LogicalMemoryRootRef freshRoot = std::get<LogicalMemoryRootRef>(chainRole);
  require(test, freshRoot != rootM,
          "the downstream exposure resolves to the fresh root");

  SpatialSimulationWorkload workload{chainLaunch};
  workload.observableContract.memories.push_back(SpatialMemoryObservable{
      MemoryExposureTarget{0}, MemoryObservationForm::FullState});
  llvm::Expected<CanonicalSimulationWorkload> finalized =
      finalizeSimulationWorkload(workload, view);
  if (!finalized)
    fail(test, "the chained-launch workload finalizes");

  // The launch binds only fresh storage: the registry is empty.
  SpatialSimulationRuntimeInputDraft empty{finalized->identity()};
  require(test,
          !isRejected(finalizeSimulationRuntimeInput(empty, *finalized, view)),
          "an empty registry is total when no imported root is reachable");
  // A binding for the fresh root or for the unrelated imported root is
  // rejected.
  SpatialSimulationRuntimeInputDraft boundFresh{finalized->identity()};
  boundFresh.memoryObjects = {byteObject(40, 0x11)};
  boundFresh.memoryRootBindings = {RuntimeMemoryBindingDraft{freshRoot, 0, 0}};
  require(
      test,
      isRejected(finalizeSimulationRuntimeInput(boundFresh, *finalized, view)),
      "a fresh root never receives a binding");
  SpatialSimulationRuntimeInputDraft boundImported{finalized->identity()};
  boundImported.memoryObjects = {byteObject(40, 0x11)};
  boundImported.memoryRootBindings = {RuntimeMemoryBindingDraft{rootM, 0, 0}};
  require(test,
          isRejected(
              finalizeSimulationRuntimeInput(boundImported, *finalized, view)),
          "the upstream imported root is unrelated to this launch");
}

} // namespace

int main() {
  typedDfgExecution();
  rootedLaunchOwnership();
  valueClassificationAndLanes();
  streamHorizonAndCardinality();
  memoryObjectOrdinals();
  observableContractRules();
  wireRejections();
  denseCoordinates();
  exposureTargetsAndDiffBaseline();
  admissionAdapters();
  enumStrictness();
  directMemoryObservables();
  noneTokenStreams();
  freshExposureChain();
  scaleObservation();
  llvm::outs() << "simulation wire anchors passed\n";
  return 0;
}
