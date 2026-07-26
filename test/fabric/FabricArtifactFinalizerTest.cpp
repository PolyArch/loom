#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"

#include "Common/ArtifactFinalizer.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactIdentity;
using loom::ArtifactStore;
using loom::CanonicalSemanticBytes;
using loom::fabric::FinalizedFabricRoot;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

void requireFabricError(llvm::StringRef test, llvm::Error error,
                        loom::fabric::FabricRefErrorKind expected) {
  if (!error)
    fail(test, "accepted an invalid finalized Fabric reference");
  const loom::fabric::FabricRefErrorKind actual =
      loom::fabric::takeFabricRefErrorKind(std::move(error));
  require(test, actual == expected,
          "finalized Fabric reference failure kind changed");
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-finalizer-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

mlir::MLIRContext &context() {
  static mlir::MLIRContext *ctx = [] {
    mlir::DialectRegistry registry;
    registry.insert<::fabric::FabricDialect>();
    auto *result = new mlir::MLIRContext(registry);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *ctx;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef test,
                                        llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric source");
  return module;
}

::fabric::ModuleOp root(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::ModuleOp selected;
  for (::fabric::ModuleOp candidate : module.getOps<::fabric::ModuleOp>()) {
    if (selected)
      fail(test, "fixture has more than one root");
    selected = candidate;
  }
  if (!selected)
    fail(test, "fixture has no Fabric root");
  return selected;
}

loom::fabric::FabricFuTemplateRef
uniqueFuTemplate(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &view) {
  std::optional<loom::fabric::FabricFuTemplateRef> result;
  for (std::uint64_t id = 0;; ++id) {
    std::optional<loom::fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    if (result)
      fail(test, "fixture has more than one canonical FU template");
    result = loom::fabric::FabricFuTemplateRef(id);
  }
  if (!result)
    fail(test, "fixture has no canonical FU template");
  return *result;
}

std::string moduleSource(llvm::StringRef name, bool reverse) {
  const llvm::StringLiteral first = R"mlir(
    %x = fabric.fifo %a [max_depth = 2, bypassable = true]
         : !fabric.bits<32>
    %y = fabric.boundary [s2t] %x, %ta
         : (!fabric.bits<32>, !fabric.bits<4>)
        -> !fabric.bits_tag<32, 4>
  )mlir";
  const llvm::StringLiteral second = R"mlir(
    %u = fabric.fifo %b [max_depth = 3, bypassable = false]
         : !fabric.bits<32>
    %v = fabric.boundary [s2t] %u, %tb
         : (!fabric.bits<32>, !fabric.bits<4>)
        -> !fabric.bits_tag<32, 4>
  )mlir";

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @" << name
         << "(%a: !fabric.bits<32>, %ta: !fabric.bits<4>, "
            "%b: !fabric.bits<32>, %tb: !fabric.bits<4>) "
            "-> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>) {\n";
  if (reverse)
    stream << second << first;
  else
    stream << first << second;
  stream << "fabric.yield %y, %v : !fabric.bits_tag<32, 4>, "
            "!fabric.bits_tag<32, 4>\n} }\n";
  return stream.str();
}

void canonicalPublicationAndStrictImport() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  mlir::OwningOpRef<mlir::ModuleOp> first =
      parse(test, moduleSource("first", false));
  mlir::OwningOpRef<mlir::ModuleOp> second =
      parse(test, moduleSource("second", true));

  FinalizedFabricRoot firstResult =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *first), store));
  FinalizedFabricRoot secondResult =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *second), store));
  require(test,
          firstResult.reference().artifact == secondResult.reference().artifact,
          "source names or graph-region order changed Fabric identity");
  require(test, firstResult.directDependencies().empty(),
          "Module root retained a direct dependency");
  require(test, firstResult.view().pointConnections().size() == 2,
          "fixed FIFO-to-boundary connections were not imported");

  std::optional<loom::fabric::FabricFifoOccurrenceRef> nonBypassableFifo;
  for (const loom::fabric::FabricPhysicalTraversalRef &traversal :
       firstResult.view().admittedTraversals()) {
    if (llvm::Error error =
            loom::fabric::validateFabricRef(firstResult.view(), traversal))
      fail(test, llvm::toString(std::move(error)));
    if (traversal.kind() !=
        loom::fabric::FabricPhysicalTraversalKind::FifoTraversal)
      continue;
    const auto &fifo =
        std::get<loom::fabric::FabricFifoTraversalPayload>(traversal.payload);
    if (fifo.mode == loom::fabric::FabricFifoTraversalMode::Buffered &&
        !firstResult.view().admitsTraversal(
            loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
                fifo.owner, loom::fabric::FabricFifoTraversalMode::Bypass)))
      nonBypassableFifo = fifo.owner;
  }
  require(test, nonBypassableFifo.has_value(),
          "non-bypassable FIFO capability was not preserved");

  const loom::fabric::FabricInventoryOwnerRef fifoOwner =
      loom::fabric::FabricInventoryOwnerRef::of(*nonBypassableFifo);
  const ::fabric::ResourceContract *fifoContract =
      firstResult.view().resourceContract(fifoOwner);
  require(test,
          fifoContract && fifoContract->stateCount() == 1 &&
              fifoContract->usePatternCount() == 3,
          "finalized view did not expose the FIFO owner contract");
  requireFabricError(
      test,
      loom::fabric::validateFabricRef(
          firstResult.view(),
          loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
              *nonBypassableFifo,
              loom::fabric::FabricFifoTraversalMode::Bypass)),
      loom::fabric::FabricRefErrorKind::TraversalNotAdmitted);

  const loom::fabric::FabricPointConnectionPayload connection =
      firstResult.view().pointConnections().front();
  loom::fabric::FabricTransportEndpointRef stale = connection.destination;
  stale.ordinal = firstResult.view().transportEndpointCount(stale.owner);
  requireFabricError(test,
                     loom::fabric::validateFabricRef(firstResult.view(), stale),
                     loom::fabric::FabricRefErrorKind::OrdinalOutOfRange);

  std::vector<std::uint8_t> foreignBytes(ArtifactIdentity::byteSize, 0x5a);
  const ArtifactIdentity foreign =
      take(test, ArtifactIdentity::fromBytes(foreignBytes));
  requireFabricError(test,
                     loom::fabric::checkFabricBinding(
                         firstResult.view(),
                         loom::fabric::FabricImportBinding{
                             foreign, loom::fabric::FabricRootKind::Module}),
                     loom::fabric::FabricRefErrorKind::ForeignArtifact);

  FinalizedFabricRoot imported =
      take(test, loom::fabric::importEntireFabricRoot(firstResult.reference(),
                                                      store));
  require(test,
          imported.reference().artifact == firstResult.reference().artifact,
          "strict import changed Fabric identity");
  require(test,
          imported.canonicalBytes().bytes().equals(
              firstResult.canonicalBytes().bytes()),
          "strict import changed canonical bytes");
}

void malformedStoredPayloadIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  std::vector<std::uint8_t> bytes = {'n', 'o', 't', '-', 'f',
                                     'a', 'b', 'r', 'i', 'c'};
  CanonicalSemanticBytes malformed(bytes);
  ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "fabric_artifact_invalid");
}

void spatialSwitchConnectivityBecomesTraversals() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  mlir::OwningOpRef<mlir::ModuleOp> source = parse(test, R"mlir(
    module {
      fabric.module @switch_root(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>) {
        %x:2 = fabric.switch [spatial] %a, %b
          [{connectivity_table = ["11", "10"]}]
          : (!fabric.bits<32>, !fabric.bits<32>)
         -> (!fabric.bits<32>, !fabric.bits<32>)
        fabric.yield %x#0, %x#1
          : !fabric.bits<32>, !fabric.bits<32>
      }
    }
  )mlir");

  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  std::vector<loom::fabric::FabricSwitchTraversalPayload> traversals;
  for (const loom::fabric::FabricPhysicalTraversalRef &traversal :
       finalized.view().admittedTraversals()) {
    if (traversal.kind() !=
        loom::fabric::FabricPhysicalTraversalKind::SwitchTraversal)
      continue;
    traversals.push_back(std::get<loom::fabric::FabricSwitchTraversalPayload>(
        traversal.payload));
  }
  require(test, traversals.size() == 3,
          "switch connectivity did not produce three physical traversals");
  bool input0Output0 = false;
  bool input1Output0 = false;
  bool input1Output1 = false;
  for (const auto &traversal : traversals) {
    input0Output0 |= traversal.input == 0 && traversal.output == 0;
    input1Output0 |= traversal.input == 1 && traversal.output == 0;
    input1Output1 |= traversal.input == 1 && traversal.output == 1;
  }
  require(test, input0Output0 && input1Output0 && input1Output1,
          "switch traversal ordinals do not follow the MSB-left convention");
}

void fuCapabilityTemplatesComeFromThePhysicalGraph() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());

  mlir::OwningOpRef<mlir::ModuleOp> singleSource = parse(test, R"mlir(
    module {
      fabric.module @single(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %sum = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %sum : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir");
  FinalizedFabricRoot single = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *singleSource), store));
  const loom::fabric::FabricFuTemplateRef singleFu =
      uniqueFuTemplate(test, single.view());
  auto singleTemplates = single.view().fuCapabilityTemplates(singleFu);
  require(test, singleTemplates.size() == 1,
          "single operation FU did not produce one capability template");
  require(test,
          singleTemplates.front().activeNodes.size() == 1 &&
              singleTemplates.front().activeEdges.size() == 3,
          "single operation FU capability template is incomplete");

  mlir::OwningOpRef<mlir::ModuleOp> branchSource = parse(test, R"mlir(
    module {
      fabric.module @branch(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %a0, %a1 = fabric.demux %fa : !fabric.bits<32> -> 2
            %b0, %b1 = fabric.demux %fb : !fabric.bits<32> -> 2
            %sum = fabric.op [@arith.addi] (%a0, %b0)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %product = fabric.op [@arith.muli] (%a1, %b1)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerMultiply>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %selected = fabric.mux %sum, %product : !fabric.bits<32>
            fabric.yield %selected : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir");
  FinalizedFabricRoot branch = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *branchSource), store));
  const loom::fabric::FabricFuTemplateRef branchFu =
      uniqueFuTemplate(test, branch.view());
  auto branchTemplates = branch.view().fuCapabilityTemplates(branchFu);
  require(test, branchTemplates.size() == 2,
          "branch FU did not produce exactly two coherent templates");
  for (const loom::fabric::FabricFuCapabilityTemplateRecord &record :
       branchTemplates) {
    unsigned opCount = 0;
    unsigned muxCount = 0;
    unsigned demuxCount = 0;
    for (const loom::fabric::FabricFuTemplateNodeRef &node :
         record.activeNodes) {
      opCount += node.node == loom::fabric::FabricFuNodeKind::Op;
      muxCount += node.node == loom::fabric::FabricFuNodeKind::Mux;
      demuxCount += node.node == loom::fabric::FabricFuNodeKind::Demux;
    }
    require(test,
            opCount == 1 && muxCount == 1 && demuxCount == 2 &&
                record.activeEdges.size() == 6,
            "branch FU capability template contains a mixed route selection");
  }

  FinalizedFabricRoot imported = take(
      test, loom::fabric::importEntireFabricRoot(branch.reference(), store));
  require(test,
          imported.view().fuCapabilityTemplates(branchFu) == branchTemplates,
          "strict import changed the FU capability-template inventory");
}

} // namespace

int main() {
  canonicalPublicationAndStrictImport();
  malformedStoredPayloadIsRejected();
  spatialSwitchConnectivityBecomesTraversals();
  fuCapabilityTemplatesComeFromThePhysicalGraph();
  return EXIT_SUCCESS;
}
