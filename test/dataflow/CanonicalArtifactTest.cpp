// Six real-MLIR anchor groups cover canonical identity, entity import,
// structural token/channel relations, and memory/service relations without a
// Mapping Artifact.

#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "Common/ArtifactStore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace loom;
using namespace dataflow;

namespace {

// Shared parsing and finalization helpers

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
    registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect, mlir::DLTIDialect,
                    mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
    auto *c =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    c->loadAllAvailableDialects();
    return c;
  }();
  return *ctx;
}
mlir::OwningOpRef<mlir::ModuleOp> parse(const char *test,
                                        llvm::StringRef text) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context());
  if (!module)
    fail(test, "failed to parse fixture");
  return module;
}
CanonicalDataflowArtifact finalize(const char *test, llvm::StringRef text) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(test, text);
  llvm::Expected<CanonicalDataflowArtifact> artifact =
      finalizeCanonicalDataflow(module.get());
  if (!artifact)
    fail(test, "finalize failed: " + llvm::toString(artifact.takeError()));
  return std::move(*artifact);
}
bool finalizeRejected(const char *test, llvm::StringRef text) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(test, text);
  llvm::Expected<CanonicalDataflowArtifact> artifact =
      finalizeCanonicalDataflow(module.get());
  if (artifact)
    return false;
  llvm::consumeError(artifact.takeError());
  return true;
}
ArtifactIdentity identityOf(const char *test, llvm::StringRef text) {
  return finalize(test, text).identity();
}
std::vector<std::uint8_t> bytesOf(const CanonicalDataflowArtifact &artifact) {
  llvm::ArrayRef<std::uint8_t> bytes = artifact.canonicalBytes().bytes();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}
CanonicalDataflowProgramView viewOf(const char *test,
                                    const CanonicalDataflowArtifact &artifact) {
  llvm::Expected<CanonicalDataflowProgramView> view = artifact.view();
  if (!view)
    fail(test, "view import failed: " + llvm::toString(view.takeError()));
  return std::move(*view);
}
CanonicalActorView actorByName(const char *test,
                               const CanonicalDataflowProgramView &view,
                               llvm::StringRef opName) {
  for (const CanonicalActorView &actor : view.actors())
    if (actor.op->getName().getStringRef() == opName)
      return actor;
  fail(test, ("no actor named " + opName).str());
}

// A finalization-valid single-graph compute program parameterized by symbol
// name, the two constant payloads, the compute op, and the SSA spellings.
std::string computeGraph(llvm::StringRef sym, int lhs, int rhs,
                         llvm::StringRef op, llvm::StringRef a,
                         llvm::StringRef b) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "module {\n  dataflow.graph private @" << sym
     << "(%ctrl: none) -> i32 attributes {input_segments = array<i32: 0, 0, "
        "0>, result_segments = array<i32: 1, 0, 0>} {\n    %"
     << a << " = dataflow.constant %ctrl {const_value = " << lhs
     << " : i32} : i32\n    %" << b
     << " = dataflow.constant %ctrl {const_value = " << rhs
     << " : i32} : i32\n    %sum = " << op << " %" << a << ", %" << b
     << " : i32\n    %ret:2 = dataflow.sync %ctrl, %sum : (none, i32) -> "
        "(none, "
        "i32)\n    dataflow.graph.return values(%ret#1 : i32) streams() "
        "memories() complete(%ret#0 : none)\n  }\n}\n";
  return os.str();
}

std::string reconvergentControlGraph(unsigned depth) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << R"mlir(module {
  dataflow.graph private @reconvergent(%start: none, %empty: i1) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i32} : i32
    %one = dataflow.constant %start {const_value = 1 : i32} : i32
    %starts:2 = dataflow.demux %empty, %start
        : (i1, none) -> (none, none)
    %lowers:2 = dataflow.demux %empty, %zero
        : (i1, i32) -> (i32, i32)
    %limits:2 = dataflow.demux %empty, %one
        : (i1, i32) -> (i32, i32)
    %steps:2 = dataflow.demux %empty, %one
        : (i1, i32) -> (i32, i32)
    %iv, %phase = dataflow.stream %lowers#0, %limits#0, %steps#0
        step add while slt : i32
    %control = dataflow.carry %phase, %starts#0, %body : none
    %lanes:2 = dataflow.demux %phase, %control
        : (i1, none) -> (none, none)
    %v0 = dataflow.invariant %phase, %zero : i32
    %v1 = dataflow.invariant %phase, %one : i32
)mlir";
  for (unsigned i = 2; i < depth; ++i)
    os << "    %v" << i << " = arith.addi %v" << i - 1 << ", %v" << i - 2
       << " : i32\n";
  os << "    %selected = arith.cmpi sgt, %v" << depth - 1 << ", %v" << depth - 2
     << " : i32\n"
     << R"mlir(    %branches:2 = dataflow.demux %selected, %lanes#1
        : (i1, none) -> (none, none)
    %body = dataflow.mux %selected, %branches#0, %branches#1
        : (i1, none, none) -> none
    %complete = dataflow.mux %empty, %lanes#0, %starts#1
        : (i1, none, none) -> none
    dataflow.graph.return %complete : none
  }
}
)mlir";
  return os.str();
}

// (a) Canonical invariance

void canonicalInvariance() {
  const char *test = "canonicalInvariance";
  // Private-symbol and SSA renaming.
  require(test,
          identityOf(test, computeGraph("g", 3, 4, "arith.addi", "a", "b")) ==
              identityOf(test, computeGraph("z", 3, 4, "arith.addi", "p", "q")),
          "private-symbol and SSA renaming must not change identity");

  // Location changes.
  const char *located = R"mlir(
module {
  dataflow.graph private @g(%ctrl: none) -> i32 attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
    %c = dataflow.constant %ctrl {const_value = 5 : i32} : i32 loc("x":1:2)
    %r:2 = dataflow.sync %ctrl, %c : (none, i32) -> (none, i32) loc("y":3:4)
    dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none)
  }
}
)mlir";
  const char *plain = R"mlir(
module {
  dataflow.graph private @g(%ctrl: none) -> i32 attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
    %c = dataflow.constant %ctrl {const_value = 5 : i32} : i32
    %r:2 = dataflow.sync %ctrl, %c : (none, i32) -> (none, i32)
    dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none)
  }
}
)mlir";
  require(test, identityOf(test, located) == identityOf(test, plain),
          "location changes must not change identity");

  // Graph-actor textual reordering inside one graph body: two independent
  // constants swapped in program order must not change identity.
  auto twoActorGraph = [](llvm::StringRef first, llvm::StringRef second) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "module { dataflow.graph private @g(%c: none) -> i32 attributes "
          "{input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: "
          "1, 0, 0>} {\n    "
       << first << "\n    " << second
       << "\n    %s = arith.addi %a, %b : i32\n    %r:2 = dataflow.sync %c, %s "
          ": (none, i32) -> (none, i32)\n    dataflow.graph.return values(%r#1 "
          ": i32) streams() memories() complete(%r#0 : none) } }";
    return os.str();
  };
  const char *ka = "%a = dataflow.constant %c {const_value = 1 : i32} : i32";
  const char *kb = "%b = dataflow.constant %c {const_value = 2 : i32} : i32";
  require(test,
          identityOf(test, twoActorGraph(ka, kb)) ==
              identityOf(test, twoActorGraph(kb, ka)),
          "graph-actor textual reordering must not change identity");

  // N identical private graphs: two symmetric isomorphic presentations (forward
  // and reverse module order) have equal identity and valid distinct IDs. This
  // same compact helper is a bounded scale anchor: without orbit pruning the
  // finalizer explores a factorial candidate product and this call would not
  // terminate, with no timing assertion.
  auto identicalGraphs = [](int n, bool reverse) {
    std::string s = "module {\n";
    for (int k = 0; k < n; ++k) {
      std::string i = std::to_string(reverse ? n - 1 - k : k);
      s +=
          "  dataflow.graph private @g" + i +
          "(%c: none) -> i32 attributes {input_segments = array<i32: 0, 0, 0>, "
          "result_segments = array<i32: 1, 0, 0>} { %k = dataflow.constant %c "
          "{const_value = 1 : i32} : i32\n    %r:2 = dataflow.sync %c, %k : "
          "(none, i32) -> (none, i32)\n    dataflow.graph.return values(%r#1 : "
          "i32) streams() memories() complete(%r#0 : none) }\n";
    }
    return s + "}";
  };
  CanonicalDataflowArtifact ten = finalize(test, identicalGraphs(10, false));
  CanonicalDataflowArtifact tenReversed =
      finalize(test, identicalGraphs(10, true));
  require(test,
          ten.identity() == tenReversed.identity() &&
              bytesOf(ten) == bytesOf(tenReversed),
          "symmetric isomorphic presentations have equal canonical bytes and "
          "identity in any order");
  CanonicalDataflowProgramView sym = viewOf(test, ten);
  std::set<std::uint64_t> ids;
  for (const CanonicalGraphView &g : sym.graphs())
    ids.insert(g.ref.entity.value());
  for (const CanonicalActorView &a : sym.actors())
    ids.insert(a.ref.entity.value());
  require(test, ids.size() == sym.graphs().size() + sym.actors().size(),
          "automorphic entities must receive distinct in-range IDs");
  for (std::uint64_t id : ids)
    require(test, id < sym.entityCount(), "entity IDs must be in range");
}

void reconvergentCardinalityRejection() {
  const char *test = "reconvergentCardinalityRejection";
  require(test, finalizeRejected(test, reconvergentControlGraph(40)),
          "a non-one-shot reconvergent graph must fail without path explosion");
}

void artifactStoreRoundTrip() {
  const char *test = "artifactStoreRoundTrip";
  CanonicalDataflowArtifact artifact =
      finalize(test, computeGraph("g", 3, 4, "arith.addi", "a", "b"));
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-dataflow-artifact", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  ArtifactStore store(directory);
  auto reference = publishCanonicalDataflow(artifact, store);
  if (!reference)
    fail(test, "publication failed: " + llvm::toString(reference.takeError()));
  auto imported = importCanonicalDataflow(*reference, store);
  if (!imported)
    fail(test, "strict import failed: " + llvm::toString(imported.takeError()));
  require(test,
          imported->identity() == artifact.identity() &&
              bytesOf(*imported) == bytesOf(artifact),
          "ArtifactStore import must preserve canonical Dataflow bytes");
  require(
      test,
      isRejected(importCanonicalDataflow(
          ArtifactRootReference{"loom.foreign", {1, 0}, artifact.identity()},
          store)),
      "foreign artifact schema must reject before import");
  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

void importerConstructedOperationRoundTrip() {
  const char *test = "importerConstructedOperationRoundTrip";
  const char *assumeProgram = R"mlir(
module {
  llvm.func @entry(%condition: i1) {
    llvm.intr.assume %condition : i1
    llvm.return
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(test, assumeProgram);
  mlir::LLVM::AssumeOp parsedAssume;
  module->walk([&](mlir::LLVM::AssumeOp assume) { parsedAssume = assume; });
  require(test, static_cast<bool>(parsedAssume), "fixture has one assume op");

  mlir::OpBuilder builder(parsedAssume);
  mlir::LLVM::AssumeOp::create(
      builder, parsedAssume.getLoc(), parsedAssume.getCond(),
      llvm::ArrayRef<mlir::ValueRange>{}, builder.getArrayAttr({}));
  parsedAssume.erase();

  llvm::Expected<CanonicalDataflowArtifact> finalized =
      finalizeCanonicalDataflow(module.get());
  if (!finalized)
    fail(test, "finalize failed: " + llvm::toString(finalized.takeError()));
  llvm::Expected<CanonicalDataflowArtifact> imported = importCanonicalDataflow(
      finalized->identity(), finalized->canonicalBytes());
  if (!imported)
    fail(test, "strict import failed: " + llvm::toString(imported.takeError()));
  require(test, imported->identity() == finalized->identity(),
          "strict import preserves an importer-built operation identity");
}

// (b) Semantic differences

void semanticDifferences() {
  const char *test = "semanticDifferences";
  ArtifactIdentity base =
      identityOf(test, computeGraph("g", 3, 4, "arith.addi", "a", "b"));
  require(test,
          base !=
              identityOf(test, computeGraph("g", 3, 4, "arith.muli", "a", "b")),
          "actor kind must change identity");
  require(test,
          base !=
              identityOf(test, computeGraph("g", 9, 4, "arith.addi", "a", "b")),
          "a semantic attribute must change identity");

  // Payload type.
  const char *i32 = R"mlir(
module { dataflow.graph private @g(%c: none, %x: i32) -> i32 attributes {input_segments = array<i32: 1, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
  %r:2 = dataflow.sync %c, %x : (none, i32) -> (none, i32)
  dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none) } }
)mlir";
  const char *i64 = R"mlir(
module { dataflow.graph private @g(%c: none, %x: i64) -> i64 attributes {input_segments = array<i32: 1, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
  %r:2 = dataflow.sync %c, %x : (none, i64) -> (none, i64)
  dataflow.graph.return values(%r#1 : i64) streams() memories() complete(%r#0 : none) } }
)mlir";
  require(test, identityOf(test, i32) != identityOf(test, i64),
          "payload type must change identity");

  // Operand ordinal / edge rewiring.
  const char *base2 = R"mlir(
module { dataflow.graph private @g(%c: none, %x: i32, %y: i32) -> i32 attributes {input_segments = array<i32: 2, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
  %d = arith.subi %x, %y : i32
  %r:2 = dataflow.sync %c, %d : (none, i32) -> (none, i32)
  dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none) } }
)mlir";
  const char *swapped = R"mlir(
module { dataflow.graph private @g(%c: none, %x: i32, %y: i32) -> i32 attributes {input_segments = array<i32: 2, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
  %d = arith.subi %y, %x : i32
  %r:2 = dataflow.sync %c, %d : (none, i32) -> (none, i32)
  dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none) } }
)mlir";
  require(test, identityOf(test, base2) != identityOf(test, swapped),
          "operand ordinal and edge rewiring must change identity");

  // Externally visible linkage: a public host symbol's name is semantic, while
  // a private symbol's name is a canonical label and is redacted.
  auto launchHost = [](llvm::StringRef vis, llvm::StringRef host) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "module {\n"
       << "  dataflow.graph private @g(%c: none) -> i32 attributes "
          "{input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: "
          "1, 0, 0>} { %k = dataflow.constant %c {const_value = 1 : i32} : "
          "i32\n"
          "    %r:2 = dataflow.sync %c, %k : (none, i32) -> (none, i32)\n"
          "    dataflow.graph.return values(%r#1 : i32) streams() memories() "
          "complete(%r#0 : none) }\n"
       << "  dataflow.thread private @t "
          "domain(#dataflow.thread_domain<dense>)() ctrl (%c: none) { %a, %d = "
          "dataflow.graph.launch @g deps(%c) values() stream_inputs() "
          "memories() stream_outputs() : (none) -> (i32, none)\n"
          "    dataflow.thread.yield %d : none }\n"
       << "  func.func " << vis << " @" << host
       << "() { %t = dataflow.thread.launch @t() : () -> "
          "!dataflow.thread_token\n    return } }\n";
    return os.str();
  };
  require(test,
          identityOf(test, launchHost("public", "host")) !=
              identityOf(test, launchHost("public", "entry")),
          "a public linkage-name change must change identity");
  require(test,
          identityOf(test, launchHost("private", "host")) ==
              identityOf(test, launchHost("private", "entry")),
          "a private symbol name must not change identity");

  // Stored-program order in an ordered thread body: two independent launches
  // swapped in textual order must change identity.
  auto twoLaunchThread = [](llvm::StringRef first, llvm::StringRef second) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "module {\n"
       << "  dataflow.graph private @g0(%c: none) -> i32 attributes "
          "{input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: "
          "1, 0, 0>} { %k = dataflow.constant %c {const_value = 1 : i32} : "
          "i32\n"
          "    %r:2 = dataflow.sync %c, %k : (none, i32) -> (none, i32)\n"
          "    dataflow.graph.return values(%r#1 : i32) streams() memories() "
          "complete(%r#0 : none) }\n"
       << "  dataflow.graph private @g1(%c: none) -> i32 attributes "
          "{input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: "
          "1, 0, 0>} { %k = dataflow.constant %c {const_value = 2 : i32} : "
          "i32\n"
          "    %r:2 = dataflow.sync %c, %k : (none, i32) -> (none, i32)\n"
          "    dataflow.graph.return values(%r#1 : i32) streams() memories() "
          "complete(%r#0 : none) }\n"
       << "  dataflow.thread private @t "
          "domain(#dataflow.thread_domain<dense>)() ctrl (%c: none) {\n"
       << "    %a, %da = dataflow.graph.launch @" << first
       << " deps(%c) values() stream_inputs() memories() stream_outputs() : "
          "(none) -> (i32, none)\n"
       << "    %b, %db = dataflow.graph.launch @" << second
       << " deps(%c) values() stream_inputs() memories() stream_outputs() : "
          "(none) -> (i32, none)\n"
       << "    dataflow.graph.wait %da, %db : none, none\n"
       << "    dataflow.thread.yield %da, %db : none, none }\n"
       << "  func.func private @h() { %t = dataflow.thread.launch @t() : () -> "
          "!dataflow.thread_token\n    return } }\n";
    return os.str();
  };
  require(test,
          identityOf(test, twoLaunchThread("g0", "g1")) !=
              identityOf(test, twoLaunchThread("g1", "g0")),
          "stored-program order in a thread body must change identity");
}

void storedProgramOperationsAreNotActors() {
  const char *test = "storedProgramOperationsAreNotActors";
  CanonicalDataflowArtifact artifact = finalize(test, R"mlir(
module {
  llvm.func @update(%aggregate: !llvm.struct<(i16, !llvm.ptr)>,
                    %value: !llvm.ptr)
      -> !llvm.struct<(i16, !llvm.ptr)> {
    %updated = llvm.insertvalue %value, %aggregate[1]
      : !llvm.struct<(i16, !llvm.ptr)>
    llvm.return %updated : !llvm.struct<(i16, !llvm.ptr)>
  }
}
)mlir");
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  require(test, view.graphs().empty() && view.actors().empty(),
          "stored-program LLVM operations must not become Spatial actors");
}

// (c) Finalize, import, and rejections over all five kinds with no Mapping

// All five entity kinds: a graph, actors, a fresh allocation root, a thread
// memory-formal root, a root thread launch, and a static graph launch.
const char *allKindsProgram() {
  return R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @g(%ctrl: none, %mem: memref<10xi32>) -> memref<10xi32> attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 1>} {
    %a = memref.alloc() : memref<4xi32>
    %idx = arith.constant 0 : index
    %d, %done = dataflow.load %a[%idx] %ctrl : memref<4xi32>
    dataflow.graph.return values() streams() memories(%mem : memref<10xi32>) complete(%done : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(%mem: memref<10xi32>) ctrl (%ctrl: none) {
    %m, %d = dataflow.graph.launch @g deps(%ctrl) values() stream_inputs() memories(%mem) stream_outputs() : (none, memref<10xi32>) -> (memref<10xi32>, none)
    dataflow.thread.yield %d : none
  }
  func.func private @host(%mem: memref<10xi32>) {
    %t = dataflow.thread.launch @t(%mem) : (memref<10xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

void finalizeImportRejections() {
  const char *test = "finalizeImportRejections";
  CanonicalDataflowArtifact artifact = finalize(test, allKindsProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  // No Mapping Artifact is consulted: the view resolves the five kinds
  // directly.
  bool sawGraph = view.graphs().size() == 1;
  bool sawActor = !view.actors().empty();
  bool sawRootLaunch = view.rootThreadLaunches().size() == 1;
  bool sawStaticLaunch = view.staticGraphLaunches().size() == 1;
  require(test, sawGraph && sawActor && sawRootLaunch && sawStaticLaunch,
          "graph, actor, root launch, and static launch resolve");
  // Two roots: the imported thread memory formal and the fresh allocation.
  require(test, view.logicalMemoryRoots().size() == 2,
          "a thread memory formal and a fresh allocation are the two roots");
  bool sawFormalRoot = false, sawAllocRoot = false;
  for (const CanonicalLogicalMemoryRootView &r : view.logicalMemoryRoots()) {
    if (r.formalArgIndex) {
      sawFormalRoot = llvm::isa<dataflow::ThreadOp>(r.op);
    } else {
      sawAllocRoot = r.op->getName().getStringRef() == "memref.alloc";
    }
    require(test, !errored(view.resolve(r.ref).takeError()),
            "a logical memory root resolves");
  }
  require(test, sawFormalRoot,
          "an imported memory root is a dataflow.thread formal, not a graph");
  require(test, sawAllocRoot, "a fresh allocation is a root");

  // Authored IDs on carriers are discarded by the finalizer.
  const char *authored = R"mlir(
module {
  dataflow.graph private @g(%ctrl: none) -> i32 attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 1, 0, 0>, dataflow.entity_id = #dataflow.entity_id<7>} {
    %k = dataflow.constant %ctrl {const_value = 1 : i32, dataflow.entity_id = #dataflow.entity_id<3>} : i32
    %r:2 = dataflow.sync %ctrl, %k : (none, i32) -> (none, i32)
    dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none)
  }
}
)mlir";
  const char *cleanTwin = R"mlir(
module {
  dataflow.graph private @g(%ctrl: none) -> i32 attributes {input_segments = array<i32: 0, 0, 0>, result_segments = array<i32: 1, 0, 0>} {
    %k = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %r:2 = dataflow.sync %ctrl, %k : (none, i32) -> (none, i32)
    dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none)
  }
}
)mlir";
  require(test, identityOf(test, authored) == identityOf(test, cleanTwin),
          "authored entity IDs are stripped to the clean identity");
  require(test, !errored(artifact.view().takeError()),
          "a clean finalized artifact re-imports");

  // Import rejects every inconsistent materialized ID: stale/noncanonical,
  // missing, and duplicate.
  auto importAfter = [&](auto mutate) -> bool {
    mlir::OwningOpRef<mlir::ModuleOp> clone(
        llvm::cast<mlir::ModuleOp>(artifact.module().getOperation()->clone()));
    mutate(clone.get());
    return isRejected(CanonicalDataflowProgramView::import(
        clone.get(), artifact.identity(), artifact.canonicalBytes()));
  };
  require(test, importAfter([](mlir::ModuleOp m) {
            m.walk([&](GraphOp g) {
              g->setAttr(kEntityIdAttrName,
                         EntityIdAttr::get(m.getContext(), 4096));
            });
          }),
          "a stale, out-of-range materialized ID is rejected");
  require(test, importAfter([](mlir::ModuleOp m) {
            m.walk([&](GraphOp g) { g->removeAttr(kEntityIdAttrName); });
          }),
          "a missing materialized ID is rejected");
  require(test, importAfter([](mlir::ModuleOp m) {
            llvm::SmallVector<mlir::Operation *> carriers;
            m.walk([&](mlir::Operation *op) {
              if (op->hasAttr(kEntityIdAttrName) && !llvm::isa<GraphOp>(op))
                carriers.push_back(op);
            });
            if (carriers.size() >= 2)
              carriers[1]->setAttr(kEntityIdAttrName,
                                   carriers[0]->getAttr(kEntityIdAttrName));
          }),
          "a duplicate materialized ID is rejected");
  // A foreign artifact identity is rejected.
  {
    CanonicalDataflowArtifact other =
        finalize(test, computeGraph("h", 2, 3, "arith.muli", "a", "b"));
    require(test,
            isRejected(view.resolve(
                GraphRef{other.identity(), view.graphs().front().ref.entity})),
            "a foreign-artifact reference is rejected");
  }
  // A residual, unrooted memory producer as a graph memory input fails
  // finalization before publication.
  const char *residual = R"mlir(
module {
  dataflow.graph private @g(%ctrl: none, %mem: memref<10xi32>) -> memref<10xi32> attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 1>} {
    dataflow.graph.return values() streams() memories(%mem : memref<10xi32>) complete(%ctrl : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
    %raw = memref.alloca() : memref<10xi32>
    %m, %d = dataflow.graph.launch @g deps(%ctrl) values() stream_inputs() memories(%raw) stream_outputs() : (none, memref<10xi32>) -> (memref<10xi32>, none)
    dataflow.thread.yield %d : none
  }
  func.func private @host() { %t = dataflow.thread.launch @t() : () -> !dataflow.thread_token
    return }
}
)mlir";
  require(test, finalizeRejected(test, residual),
          "an unresolved memory root relation fails finalization");
}

// (d) Rooted launch, token endpoints, and the software edge relation

void rootedLaunchTokenEdge() {
  const char *test = "rootedLaunchTokenEdge";
  const char *program = R"mlir(
module {
  dataflow.graph private @g(%ctrl: none, %x: i32, %mem: memref<4xi32>) -> i32 attributes {input_segments = array<i32: 1, 0, 1>, result_segments = array<i32: 1, 0, 0>} {
    %y = arith.addi %x, %x : i32
    %idx = arith.constant 0 : index
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<4xi32>
    %r:2 = dataflow.sync %done, %y : (none, i32) -> (none, i32)
    dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(%x: i32, %mem: memref<4xi32>) ctrl (%ctrl: none) {
    %v, %d = dataflow.graph.launch @g deps(%ctrl) values(%x) stream_inputs() memories(%mem) stream_outputs() : (none, i32, memref<4xi32>) -> (i32, none)
    dataflow.thread.yield %d : none
  }
  func.func private @host(%a: i32, %mem: memref<4xi32>) {
    %t1 = dataflow.thread.launch @t(%a, %mem) : (i32, memref<4xi32>) -> !dataflow.thread_token
    %t2 = dataflow.thread.launch @t(%a, %mem) : (i32, memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
  CanonicalDataflowArtifact artifact = finalize(test, program);
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  // One thread reached from two roots yields two distinct rooted launches.
  unsigned count = 0;
  bool resolves = true;
  view.forEachRootedGraphLaunch([&](RootedGraphLaunchRef r) {
    ++count;
    if (isRejected(view.resolve(r)))
      resolves = false;
  });
  require(test, count == 2 && resolves,
          "two roots of one thread yield two resolvable rooted launches");
  RootThreadLaunchRef r0 = view.rootThreadLaunches()[0].ref;
  RootThreadLaunchRef r1 = view.rootThreadLaunches()[1].ref;
  StaticGraphLaunchRef sg = view.staticGraphLaunches().front().ref;
  require(test, RootedGraphLaunchRef{r0, sg} != RootedGraphLaunchRef{r1, sg},
          "the two rooted launches are distinct");
  // A wrong-kind rooted launch and a stale static launch are rejected.
  RootThreadLaunchRef wrongKind{
      view.identity(),
      RootThreadLaunchId(view.graphs().front().ref.entity.value())};
  require(test, isRejected(view.resolve(RootedGraphLaunchRef{wrongKind, sg})),
          "a wrong-kind rooted launch is rejected");
  require(test,
          isRejected(view.resolve(RootedGraphLaunchRef{
              r0, StaticGraphLaunchRef{view.identity(),
                                       StaticGraphLaunchId(4096)}})),
          "a stale static graph launch is rejected");

  // Token endpoints and the software edge relation.
  ActorRef add = actorByName(test, view, "arith.addi").ref;
  ActorRef sync = actorByName(test, view, "dataflow.sync").ref;
  llvm::Expected<llvm::ArrayRef<CanonicalGraphConsumerEndpointRef>> consumers =
      view.graphConsumers(
          CanonicalGraphProducerEndpointRef{ActorTokenResultRef{add, 0}});
  require(test, static_cast<bool>(consumers) && !consumers->empty(),
          "the adder's result has consumers");
  bool feedsSync = false;
  for (const CanonicalGraphConsumerEndpointRef &c : *consumers)
    if (const auto *o = std::get_if<ActorTokenOperandRef>(&c))
      feedsSync |= o->actor.entity == sync.entity;
  require(test, feedsSync, "the adder feeds the synchronizer");
  llvm::Expected<CanonicalGraphProducerEndpointRef> producer =
      view.graphProducer(
          CanonicalGraphConsumerEndpointRef{ActorTokenOperandRef{sync, 1}});
  require(test,
          producer && std::holds_alternative<ActorTokenResultRef>(*producer) &&
              std::get<ActorTokenResultRef>(*producer).actor.entity ==
                  add.entity,
          "the sync data input is produced by the adder");
  // Token endpoints reject a memory-capability operand, a wrong-kind actor, and
  // an out-of-range ordinal.
  ActorRef load = actorByName(test, view, "dataflow.load").ref;
  require(test,
          errored(view.validate(CanonicalGraphConsumerEndpointRef{
              ActorTokenOperandRef{load, 0}})),
          "a memory-capability actor operand is not a token endpoint");
  ActorRef wrongKindActor{view.identity(),
                          ActorId(view.graphs().front().ref.entity.value())};
  require(test,
          errored(view.validate(CanonicalGraphConsumerEndpointRef{
              ActorTokenOperandRef{wrongKindActor, 0}})),
          "a wrong-kind actor token endpoint is rejected");
  require(test,
          errored(view.validate(CanonicalGraphConsumerEndpointRef{
              ActorTokenOperandRef{sync, 4096}})),
          "an out-of-range actor operand is rejected");
}

// (e) source_map channel multicast, terminals, events, and foreign rejection

// One rank-zero channel produced by a thread send and consumed two ways: by a
// graph stream input carrying an explicit source_map and by a direct receive.
const char *multicastProgram() {
  return R"mlir(
module {
  dataflow.graph private @gc(%start: none, %input: i32) -> () attributes {input_segments = array<i32: 0, 1, 0>, result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(%ch: !dataflow.channel<i32>) ctrl (%c: none) {
    %v = arith.constant 7 : i32
    dataflow.channel.send %ch, %v : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @streamconsumer domain(#dataflow.thread_domain<dense>)(%ch: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @gc deps(%ctrl) values() stream_inputs(%ch source_map affine_map<() -> ()>) memories() stream_outputs() : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  dataflow.thread private @directconsumer domain(#dataflow.thread_domain<dense>)(%ch: !dataflow.channel<i32>) ctrl (%c: none) {
    %m = dataflow.channel.receive %ch : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  func.func private @host(%ch: !dataflow.channel<i32>) {
    %p = dataflow.thread.launch @producer(%ch) : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %c1 = dataflow.thread.launch @streamconsumer(%ch) : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %c2 = dataflow.thread.launch @directconsumer(%ch) : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

RootThreadLaunchRef sendingRoot(const char *test,
                                const CanonicalDataflowProgramView &view) {
  for (const CanonicalRootThreadLaunchView &root : view.rootThreadLaunches()) {
    auto thread = llvm::cast<dataflow::ThreadOp>(root.callee);
    bool hasSend = false;
    thread->walk([&](dataflow::ChannelSendOp) { hasSend = true; });
    if (hasSend)
      return root.ref;
  }
  fail(test, "no sending thread");
}

void channelMulticastTerminals() {
  const char *test = "channelMulticastTerminals";
  CanonicalDataflowArtifact artifact = finalize(test, multicastProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  ChannelProducerRef producer{
      ThreadChannelSendSiteRef{sendingRoot(test, view), 0}};
  llvm::Expected<llvm::ArrayRef<ChannelConsumerBinding>> consumers =
      view.channelConsumers(producer);
  require(test, consumers && consumers->size() == 2,
          "the multicast producer derives two consumers");
  bool streamMap = false, directNoMap = false;
  for (const ChannelConsumerBinding &b : *consumers) {
    if (std::holds_alternative<GraphStreamInputConsumerRef>(b.consumer))
      streamMap = b.sourceMap.has_value();
    else if (std::holds_alternative<ThreadChannelReceiveSiteRef>(b.consumer))
      directNoMap = !b.sourceMap.has_value();
  }
  require(test, streamMap,
          "the graph stream-input consumer carries its exact source_map");
  require(test, directNoMap, "the direct receive carries no source_map");

  // The producer terminal derives its complete sink set by callback.
  CanonicalProducerTerminalRef terminal{ChannelProducerTerminalRef{producer}};
  std::vector<CanonicalSinkTerminalRef> sinks;
  require(test,
          !errored(view.pairedSinks(terminal,
                                    [&](const CanonicalSinkTerminalRef &s) {
                                      sinks.push_back(s);
                                    })) &&
              sinks.size() == 2 && sinks[0] != sinks[1],
          "the producer terminal derives two distinct sinks");
  // MessageTransfer is the member of a channel transfer obligation.
  llvm::Expected<ServiceMemberRef> channelMember =
      view.messageTransferMember(terminal);
  require(test,
          channelMember &&
              std::holds_alternative<MessageTransferMemberRef>(*channelMember),
          "a channel transfer obligation has a MessageTransfer member");
  // Boundary MessageTransfer: a root-thread start boundary is also a transfer.
  CanonicalProducerTerminalRef boundary{
      RootThreadBoundarySourceRef{RootThreadBoundaryTransferRef{
          RootThreadStartTransferRef{sendingRoot(test, view)}}}};
  llvm::Expected<ServiceMemberRef> boundaryMember =
      view.messageTransferMember(boundary);
  require(test,
          boundaryMember &&
              std::holds_alternative<MessageTransferMemberRef>(*boundaryMember),
          "a boundary transfer also has a MessageTransfer member");
  // A static transfer event round-trips with no event entity ID; a foreign
  // producer is rejected.
  StaticTransferEventRef event{ProducedTransferEventRef{terminal}};
  require(test,
          !errored(view.validate(event)) &&
              std::get<ProducedTransferEventRef>(event).terminal == terminal,
          "a static transfer event round trips with no entity ID");
  CanonicalDataflowArtifact other =
      finalize(test, computeGraph("q", 1, 2, "arith.addi", "a", "b"));
  ChannelProducerRef foreign{ThreadChannelSendSiteRef{
      RootThreadLaunchRef{other.identity(), sendingRoot(test, view).entity},
      0}};
  require(test, isRejected(view.channelConsumers(foreign)),
          "a foreign-artifact channel producer is rejected");
}

// (f) Logical-memory view, exposure, and service members

// Two distinct thread memory formals (%m0, %m1); a reusable graph-local view
// (@gv casts its imported formal) instantiated at two static sites bound to the
// two formals; and a graph-launch-result-to-later-launch chain (@gc consumes a
// prior launch's memory result).
const char *memoryCompositionProgram() {
  // @gv returns one graph-body cast in two memory results; @gl casts its formal
  // and consumes it in a load but never exposes it; @gc passes @gv's first
  // result through. The thread binds @gv to two distinct memory formals.
  return R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @gv(%start: none, %mem: memref<10xi32>) -> (memref<?xi32>, memref<?xi32>) attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 2>} {
    %v = memref.cast %mem : memref<10xi32> to memref<?xi32>
    dataflow.graph.return values() streams() memories(%v, %v : memref<?xi32>, memref<?xi32>) complete(%start : none)
  }
  dataflow.graph private @gc(%start: none, %mem: memref<*xi32>) -> memref<*xi32> attributes {input_segments = array<i32: 0, 0, 1>, result_segments = array<i32: 0, 0, 1>} {
    dataflow.graph.return values() streams() memories(%mem : memref<*xi32>) complete(%start : none)
  }
  dataflow.graph private @gl(%ctrl: none, %addr: index, %mem: memref<10xi32>) -> i32 attributes {input_segments = array<i32: 1, 0, 1>, result_segments = array<i32: 1, 0, 0>} {
    %vc = memref.cast %mem : memref<10xi32> to memref<?xi32>
    %data, %done = dataflow.load %vc[%addr] %ctrl : memref<?xi32>
    %y = arith.addi %data, %data : i32
    %r:2 = dataflow.sync %done, %y : (none, i32) -> (none, i32)
    dataflow.graph.return values(%r#1 : i32) streams() memories() complete(%r#0 : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(%m0: memref<10xi32>, %m1: memref<10xi32>) ctrl (%ctrl: none) {
    %r0a, %r0b, %d0 = dataflow.graph.launch @gv deps(%ctrl) values() stream_inputs() memories(%m0) stream_outputs() : (none, memref<10xi32>) -> (memref<?xi32>, memref<?xi32>, none)
    %r1a, %r1b, %d1 = dataflow.graph.launch @gv deps(%ctrl) values() stream_inputs() memories(%m1) stream_outputs() : (none, memref<10xi32>) -> (memref<?xi32>, memref<?xi32>, none)
    %tc = memref.cast %r0a : memref<?xi32> to memref<*xi32>
    %rc, %dc = dataflow.graph.launch @gc deps(%d0) values() stream_inputs() memories(%tc) stream_outputs() : (none, memref<*xi32>) -> (memref<*xi32>, none)
    %addr = arith.constant 0 : index
    %v, %dl = dataflow.graph.launch @gl deps(%d1) values(%addr) stream_inputs() memories(%m0) stream_outputs() : (none, index, memref<10xi32>) -> (i32, none)
    dataflow.graph.wait %dc, %dl : none, none
    dataflow.thread.yield %dc, %dl : none, none
  }
  func.func private @host(%m0: memref<10xi32>, %m1: memref<10xi32>) {
    %t = dataflow.thread.launch @t(%m0, %m1) : (memref<10xi32>, memref<10xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
}

// A MemoryExposure is a capability boundary, structurally neither a contextual
// actor nor a service member -- enforced at compile time, not by a runtime
// path.
static_assert(!std::is_constructible_v<ContextualActorRef, MemoryExposureRef>,
              "a memory exposure is not a contextual actor");
static_assert(!std::is_constructible_v<ServiceMemberRef, MemoryExposureRef>,
              "a memory exposure is not a service member");

void memoryViewExposureService() {
  const char *test = "memoryViewExposureService";
  CanonicalDataflowArtifact artifact =
      finalize(test, memoryCompositionProgram());
  CanonicalDataflowProgramView view = viewOf(test, artifact);
  require(test, view.logicalMemoryRoots().size() == 2,
          "two distinct thread memory formals are two roots");
  RootThreadLaunchRef root = view.rootThreadLaunches().front().ref;
  auto rootByFormal = [&](unsigned f) -> LogicalMemoryRootRef {
    for (const CanonicalLogicalMemoryRootView &r : view.logicalMemoryRoots())
      if (r.formalArgIndex && *r.formalArgIndex == f)
        return r.ref;
    fail(test, "no root for formal");
  };
  LogicalMemoryRootRef rootM0 = rootByFormal(0), rootM1 = rootByFormal(1);

  // Identify sites by their ABI shape, not private graph spellings. Private
  // symbols are deliberately normalized by canonical finalization.
  std::optional<StaticGraphLaunchRef> gvOf[2], glSite, gcSite;
  for (const CanonicalStaticGraphLaunchView &sg : view.staticGraphLaunches()) {
    auto launch = llvm::cast<GraphLaunchOp>(sg.op);
    if (launch.getMemoryResults().size() == 2)
      gvOf[llvm::cast<mlir::BlockArgument>(launch.getMemoryInputs()[0])
               .getArgNumber()] = sg.ref;
    else if (launch.getValueResults().size() == 1)
      glSite = sg.ref;
    else if (launch.getMemoryResults().size() == 1)
      gcSite = sg.ref;
  }
  require(test, gvOf[0] && gvOf[1] && glSite && gcSite,
          "the four static launches are identified");
  auto expose = [&](StaticGraphLaunchRef s,
                    unsigned r) -> LogicalMemoryRootOrViewRef {
    llvm::Expected<LogicalMemoryRootOrViewRef> e =
        view.resolveExposure(MemoryExposureRef{{root, s}, r});
    if (!e)
      fail(test, "exposure did not resolve: " + llvm::toString(e.takeError()));
    return *e;
  };
  auto rootOf = [](const LogicalMemoryRootOrViewRef &role) {
    if (const auto *r = std::get_if<LogicalMemoryRootRef>(&role))
      return *r;
    return std::get<LogicalMemoryViewRef>(role).root;
  };

  // One graph-body cast returned twice is exactly one root-local view.
  LogicalMemoryRootOrViewRef m0v0 = expose(*gvOf[0], 0);
  require(test, m0v0 == expose(*gvOf[0], 1),
          "one cast returned in two ordinals is exactly one view");
  require(test,
          std::holds_alternative<LogicalMemoryViewRef>(m0v0) &&
              rootOf(m0v0) == rootM0,
          "the @gv view over m0 is a root-local view under m0's root");
  LogicalMemoryRootOrViewRef m1v0 = expose(*gvOf[1], 0);
  require(test, m0v0 != m1v0 && rootOf(m1v0) == rootM1,
          "two sites under two roots yield distinct root-local views");
  // @gc consumes a thread-body cast of @gv's first result, so its exposure is
  // exactly that thread-view: a root-local view under m0, distinct from @gv's
  // own graph-body view.
  LogicalMemoryRootOrViewRef gc = expose(*gcSite, 0);
  require(test,
          std::holds_alternative<LogicalMemoryViewRef>(gc) &&
              rootOf(gc) == rootM0 && gc != m0v0,
          "the @gc exposure is a thread-view under m0, distinct from the @gv "
          "view");
  // m0's inventory holds three views (the @gv graph view, the thread cast, and
  // the non-exposed @gl cast), each exactly once; m1 holds only its @gv view.
  llvm::Expected<llvm::ArrayRef<LogicalMemoryViewRef>> invM0 =
      view.views(rootM0);
  llvm::Expected<llvm::ArrayRef<LogicalMemoryViewRef>> invM1 =
      view.views(rootM1);
  require(test, invM0 && invM0->size() == 3,
          "m0 inventory holds the @gv, thread, and non-exposed @gl views");
  std::set<StructuralOrdinal> ord;
  for (const LogicalMemoryViewRef &v : *invM0)
    ord.insert(v.viewOrdinal);
  require(test,
          ord.size() == 3 &&
              ord.count(std::get<LogicalMemoryViewRef>(gc).viewOrdinal) == 1,
          "each m0 view appears once and includes the chained thread view");
  require(test, invM1 && invM1->size() == 1, "m1 inventory has one view");

  // Service members via the exact schema; wrong-owner and non-member rejected.
  RootedGraphLaunchRef glLaunch{root, *glSite};
  ActorRef loadRef = actorByName(test, view, "dataflow.load").ref;
  ActorRef addRef = actorByName(test, view, "arith.addi").ref;
  llvm::Expected<ServiceMemberRef> member =
      view.serviceMemberFor(ContextualActorRef{glLaunch, loadRef});
  if (!member)
    fail(test, "addressed memory service projection failed: " +
                   llvm::toString(member.takeError()));
  require(test, std::holds_alternative<AddressedMemoryActorMemberRef>(*member),
          "an addressed memory actor is an addressed-memory service member");
  require(
      test,
      isRejected(view.serviceMemberFor(ContextualActorRef{glLaunch, addRef})),
      "a compute actor is not a service member");
  require(test,
          isRejected(view.serviceMemberFor(ContextualActorRef{
              RootedGraphLaunchRef{root, *gvOf[0]}, loadRef})),
          "an actor outside the launched graph is a wrong-owner rejection");
}

} // namespace

int main() {
  canonicalInvariance();
  reconvergentCardinalityRejection();
  artifactStoreRoundTrip();
  importerConstructedOperationRoundTrip();
  semanticDifferences();
  storedProgramOperationsAreNotActors();
  finalizeImportRejections();
  rootedLaunchTokenEdge();
  channelMulticastTerminals();
  memoryViewExposureService();
  llvm::outs() << "all canonical dataflow artifact tests passed\n";
  return EXIT_SUCCESS;
}
