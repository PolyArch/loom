#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricHandshake.h"

#include "Common/ArtifactFinalizer.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

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

#include <array>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::adg::DesignBuilder;
using loom::adg::FifoSpec;
using loom::adg::PortType;
using loom::adg::SpatialValue;
using loom::adg::SwitchSpec;
using loom::fabric::FabricFuCapabilityTemplateRef;
using loom::fabric::FabricFuNodeKind;
using loom::fabric::FabricFuOccurrenceRef;
using loom::fabric::FabricFuTemplateNodeRef;
using loom::fabric::FabricHandshakeOwnerKind;
using loom::fabric::FabricHandshakeSelection;
using loom::fabric::FabricInventoryKind;
using loom::fabric::FabricInventoryOwnerRef;
using loom::fabric::FabricMemoryHandshakePlacement;
using loom::fabric::FabricMemoryOccurrenceRef;
using loom::fabric::FabricMemoryOperationPortRef;
using loom::fabric::FabricPhysicalTraversalKind;
using loom::fabric::FabricPhysicalTraversalRef;
using loom::fabric::FabricPortDirection;
using loom::fabric::FabricRegisterFifoPathRole;
using loom::fabric::FabricSwitchOccurrenceRef;
using loom::fabric::FinalizedFabricRoot;
using loom::fabric::HandshakeOwnerModel;
using loom::fabric::HandshakeSignalKind;
using loom::fabric::HandshakeSignalRef;
using loom::fabric::ResolvedHandshakeActivation;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireRejected(llvm::StringRef test, llvm::Expected<T> value,
                     llvm::StringRef expected) {
  if (value)
    fail(test, "invalid handshake selection was accepted");
  const std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(expected))
    fail(test, "unexpected rejection: " + message);
}

void requireRejected(llvm::StringRef test, llvm::Error error,
                     llvm::StringRef expected) {
  if (!error)
    fail(test, "invalid handshake selection was accepted");
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(expected))
    fail(test, "unexpected rejection: " + message);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-handshake-model-test", path))
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
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
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
      fail(test, "fixture has more than one Fabric root");
    selected = candidate;
  }
  if (!selected)
    fail(test, "fixture has no Fabric root");
  return selected;
}

void materializeOperationContracts(llvm::StringRef test,
                                   mlir::ModuleOp module) {
  std::vector<std::uint8_t> bytes =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  module.walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(module.getContext(), signedBytes));
  });
}

std::string broadcastSource(std::uint32_t fanout) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @broadcast(%input: !fabric.bits<32>) -> (";
  for (std::uint32_t output = 0; output < fanout; ++output) {
    if (output)
      stream << ", ";
    stream << "!fabric.bits<32>";
  }
  stream << ") {\n  %outputs:" << fanout
         << " = fabric.switch [spatial] %input "
            "[{connectivity_table = [";
  for (std::uint32_t output = 0; output < fanout; ++output) {
    if (output)
      stream << ", ";
    stream << "\"1\"";
  }
  stream << "]}] : (!fabric.bits<32>) -> (";
  for (std::uint32_t output = 0; output < fanout; ++output) {
    if (output)
      stream << ", ";
    stream << "!fabric.bits<32>";
  }
  stream << ")\n  fabric.yield ";
  for (std::uint32_t output = 0; output < fanout; ++output) {
    if (output)
      stream << ", ";
    stream << "%outputs#" << output;
  }
  stream << " : ";
  for (std::uint32_t output = 0; output < fanout; ++output) {
    if (output)
      stream << ", ";
    stream << "!fabric.bits<32>";
  }
  stream << "\n} }\n";
  return source;
}

const HandshakeOwnerModel &
switchModel(llvm::StringRef test, llvm::ArrayRef<HandshakeOwnerModel> models) {
  const HandshakeOwnerModel *result = nullptr;
  for (const HandshakeOwnerModel &model : models) {
    if (model.owner().kind() != FabricHandshakeOwnerKind::SwitchOccurrence)
      continue;
    if (result)
      fail(test, "fixture has more than one switch handshake owner");
    result = &model;
  }
  if (!result)
    fail(test, "fixture has no switch handshake owner");
  return *result;
}

std::uint32_t node(llvm::StringRef test, const HandshakeOwnerModel &model,
                   const HandshakeSignalRef &signal) {
  std::optional<std::uint32_t> result = model.nodeForSignal(signal);
  if (!result)
    fail(test, "owner model omitted a boundary signal");
  return *result;
}

bool hasPath(const HandshakeOwnerModel &model,
             const ResolvedHandshakeActivation &activation,
             std::uint32_t source, std::uint32_t destination) {
  std::vector<std::vector<std::uint32_t>> adjacency(model.nodes().size());
  for (std::uint32_t arcOrdinal : activation.arcOrdinals()) {
    const auto &arc = model.arcs()[arcOrdinal];
    adjacency[arc.source].push_back(arc.destination);
  }
  std::vector<bool> visited(model.nodes().size(), false);
  std::vector<std::uint32_t> worklist = {source};
  visited[source] = true;
  while (!worklist.empty()) {
    const std::uint32_t current = worklist.back();
    worklist.pop_back();
    if (current == destination)
      return true;
    for (std::uint32_t next : adjacency[current]) {
      if (visited[next])
        continue;
      visited[next] = true;
      worklist.push_back(next);
    }
  }
  return false;
}

void selectedPointConnectionConsumesItsWitness() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto source = parse(test, R"mlir(
    module {
      fabric.module @point_connection(
          %data : !fabric.bits<32>,
          %tag : !fabric.bits<4>) -> !fabric.bits_tag<32, 4> {
        %buffered = fabric.fifo %data [max_depth = 2, bypassable = true]
            : !fabric.bits<32>
        %tagged = fabric.boundary [s2t] %buffered, %tag
            : (!fabric.bits<32>, !fabric.bits<4>)
            -> !fabric.bits_tag<32, 4>
        fabric.yield %tagged : !fabric.bits_tag<32, 4>
      }
    }
  )mlir");
  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  const auto traversal = llvm::find_if(
      finalized.view().admittedTraversals(), [](const auto &candidate) {
        return candidate.kind() == FabricPhysicalTraversalKind::PointConnection;
      });
  require(test, traversal != finalized.view().admittedTraversals().end(),
          "fixture has no point-connection traversal");

  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(finalized.view()));
  const HandshakeOwnerModel *model = nullptr;
  for (const HandshakeOwnerModel &candidate : models) {
    if (candidate.owner().kind() != FabricHandshakeOwnerKind::PointConnection)
      continue;
    const auto &payload = std::get<loom::fabric::FabricPointConnectionPayload>(
        candidate.owner().payload());
    const auto &selected = std::get<loom::fabric::FabricPointConnectionPayload>(
        traversal->payload);
    if (payload.source != selected.source ||
        payload.destination != selected.destination)
      continue;
    model = &candidate;
    break;
  }
  require(test, model != nullptr,
          "point connection has no exact handshake owner model");

  FabricHandshakeSelection selection;
  selection.traversals.push_back(*traversal);
  const ResolvedHandshakeActivation activation =
      take(test, loom::fabric::resolveSelectedHandshake(*model, selection));
  require(test, activation.arcOrdinals().size() == 2,
          "selected point connection lost its direct handshake relation");
}

void atomicBroadcastProjectionIsLinear(std::uint32_t fanout) {
  const std::string testName =
      ("atomicBroadcastProjectionIsLinear" + std::to_string(fanout));
  const llvm::StringRef test(testName);
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      parse(test, broadcastSource(fanout));
  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(finalized.view()));
  const HandshakeOwnerModel &model = switchModel(test, models);

  require(test, model.nodes().size() <= 5 * fanout + 8,
          "broadcast owner node count is not linear in fanout");
  require(test, model.arcs().size() <= 8 * fanout + 8,
          "broadcast owner arc count is not linear in fanout");
  require(test, model.traversalWitnesses().size() == 2 * fanout,
          "broadcast activation witness count is not linear in fanout");
  std::size_t anyTraversalFragments = 0;
  for (const auto &fragment : model.fragments()) {
    if (fragment.activationKind !=
        loom::fabric::HandshakeActivationKind::AnyTraversal)
      continue;
    ++anyTraversalFragments;
    require(test, fragment.witnessCount != 0,
            "traversal-selected fragment has no witness");
  }
  require(test, anyTraversalFragments == fanout + 1,
          "broadcast activation conditions lost their factorization");

  std::vector<FabricPhysicalTraversalRef> selected;
  std::vector<loom::fabric::FabricTransportEndpointRef> outputs;
  std::optional<loom::fabric::FabricTransportEndpointRef> input;
  for (const auto &traversal : finalized.view().physicalTraversals()) {
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::SwitchTraversal)
      continue;
    selected.push_back(traversal.reference);
    require(test,
            traversal.sources.size() == 1 && traversal.destinations.size() == 1,
            "switch traversal changed endpoint cardinality");
    if (input && *input != traversal.sources.front())
      fail(test, "single-input switch projected multiple ingress endpoints");
    input = traversal.sources.front();
    outputs.push_back(traversal.destinations.front());
  }
  require(test, input.has_value() && selected.size() == fanout,
          "switch traversal selection is incomplete");

  FabricHandshakeSelection selection;
  selection.traversals = selected;
  ResolvedHandshakeActivation activation =
      take(test, loom::fabric::resolveSelectedHandshake(model, selection));
  require(test, activation.arcOrdinals().size() <= 8 * fanout + 8,
          "selected broadcast expanded a quadratic arc relation");

  const std::uint32_t inputValid =
      node(test, model, {*input, HandshakeSignalKind::Valid});
  const std::uint32_t inputReady =
      node(test, model, {*input, HandshakeSignalKind::Ready});
  for (std::uint32_t output = 0; output < fanout; ++output) {
    const std::uint32_t outputValid =
        node(test, model, {outputs[output], HandshakeSignalKind::Valid});
    const std::uint32_t outputReady =
        node(test, model, {outputs[output], HandshakeSignalKind::Ready});
    const std::uint32_t peer = (output + 1) % fanout;
    const std::uint32_t peerReady =
        node(test, model, {outputs[peer], HandshakeSignalKind::Ready});
    require(test, hasPath(model, activation, inputValid, outputValid),
            "input valid does not reach a selected output valid");
    require(test, hasPath(model, activation, outputReady, inputReady),
            "selected output ready does not reach input ready");
    require(test, !hasPath(model, activation, outputReady, outputValid),
            "output valid incorrectly depends on its own ready");
    require(test, hasPath(model, activation, peerReady, outputValid),
            "output valid does not depend on a selected peer ready");
  }
}

void fifoModeOwnsItsExactCombinationalBreak() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto source = parse(test, R"mlir(
    module {
      fabric.module @fifo_modes(%input: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %output = fabric.fifo %input [max_depth = 2, bypassable = true]
          : !fabric.bits<32>
        fabric.yield %output : !fabric.bits<32>
      }
    }
  )mlir");
  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(finalized.view()));
  const HandshakeOwnerModel *model = nullptr;
  for (const HandshakeOwnerModel &candidate : models) {
    if (candidate.owner().kind() == FabricHandshakeOwnerKind::FifoOccurrence)
      model = &candidate;
  }
  require(test, model != nullptr, "fixture has no FIFO handshake owner");

  std::optional<FabricPhysicalTraversalRef> buffered;
  std::optional<FabricPhysicalTraversalRef> bypass;
  loom::fabric::FabricTransportEndpointRef input;
  loom::fabric::FabricTransportEndpointRef output;
  for (const auto &traversal : finalized.view().physicalTraversals()) {
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::FifoTraversal)
      continue;
    const auto &payload = std::get<loom::fabric::FabricFifoTraversalPayload>(
        traversal.reference.payload);
    if (payload.mode == loom::fabric::FabricFifoTraversalMode::Buffered)
      buffered = traversal.reference;
    else
      bypass = traversal.reference;
    input = traversal.sources.front();
    output = traversal.destinations.front();
  }
  require(test, buffered.has_value() && bypass.has_value(),
          "bypassable FIFO did not expose both exact modes");

  const std::uint32_t inputValid =
      node(test, *model, {input, HandshakeSignalKind::Valid});
  const std::uint32_t inputReady =
      node(test, *model, {input, HandshakeSignalKind::Ready});
  const std::uint32_t outputValid =
      node(test, *model, {output, HandshakeSignalKind::Valid});
  const std::uint32_t outputReady =
      node(test, *model, {output, HandshakeSignalKind::Ready});

  FabricHandshakeSelection bufferedSelection;
  bufferedSelection.traversals.push_back(*buffered);
  ResolvedHandshakeActivation bufferedActivation = take(
      test, loom::fabric::resolveSelectedHandshake(*model, bufferedSelection));
  require(test,
          !hasPath(*model, bufferedActivation, inputValid, outputValid) &&
              !hasPath(*model, bufferedActivation, outputReady, inputReady),
          "buffered FIFO is not isolated in both handshake directions");

  FabricHandshakeSelection bypassSelection;
  bypassSelection.traversals.push_back(*bypass);
  ResolvedHandshakeActivation bypassActivation = take(
      test, loom::fabric::resolveSelectedHandshake(*model, bypassSelection));
  require(test,
          hasPath(*model, bypassActivation, inputValid, outputValid) &&
              hasPath(*model, bypassActivation, outputReady, inputReady),
          "FIFO bypass is not transparent in both directions");
}

void bufferedPhysicalCycleIsAcceptedBeforeSelection() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));
  auto spatial = take(test, design.createSpatialCore("ready-cycle", {}, {}));
  auto backedge = take(test, spatial.createBackedge(bits32));
  auto buffered =
      take(test, spatial.addFifo(backedge.value(), FifoSpec{bits32, 2, false}))
          .value();
  if (llvm::Error error =
          spatial.resolveBackedge(std::move(backedge), buffered))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({}))
    fail(test, llvm::toString(std::move(error)));
  auto completed = take(test, std::move(design).finalize());
  require(test, completed.roots().size() == 1,
          "buffered physical cycle did not publish one Fabric root");
}

void selectedGlobalCycleUsesExactTraversalSelection() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));
  auto spatial = take(
      test, design.createSpatialCore("selected-cycle", {bits32}, {bits32}));
  auto external = take(test, spatial.input(0));
  auto backedge = take(test, spatial.createBackedge(bits32));
  auto routed =
      take(test, spatial.addSwitch({external, backedge.value()},
                                   SwitchSpec::spatial({bits32, bits32},
                                                       {bits32, bits32},
                                                       {{0, 1}, {0, 1}})));
  SpatialValue feedback =
      take(test, spatial.addFifo(routed[0], FifoSpec{bits32, 2, true})).value();
  if (llvm::Error error =
          spatial.resolveBackedge(std::move(backedge), feedback))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({routed[1]}))
    fail(test, llvm::toString(std::move(error)));
  auto completed = take(test, std::move(design).finalize());
  require(test, completed.roots().size() == 1,
          "fixture did not publish exactly one Fabric root");
  const FinalizedFabricRoot &finalized = completed.roots().front();

  std::vector<FabricPhysicalTraversalRef> acyclic;
  std::vector<FabricPhysicalTraversalRef> cyclic;
  std::optional<FabricPhysicalTraversalRef> buffered;
  std::optional<FabricPhysicalTraversalRef> bypass;
  for (const auto &traversal : finalized.view().physicalTraversals()) {
    if (traversal.reference.kind() ==
        FabricPhysicalTraversalKind::SwitchTraversal) {
      const auto &payload =
          std::get<loom::fabric::FabricSwitchTraversalPayload>(
              traversal.reference.payload);
      if (payload.input == 0)
        acyclic.push_back(traversal.reference);
      if ((payload.input == 1 && payload.output == 0) ||
          (payload.input == 0 && payload.output == 1))
        cyclic.push_back(traversal.reference);
      continue;
    }
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::FifoTraversal)
      continue;
    const auto &payload = std::get<loom::fabric::FabricFifoTraversalPayload>(
        traversal.reference.payload);
    if (payload.mode == loom::fabric::FabricFifoTraversalMode::Buffered)
      buffered = traversal.reference;
    else
      bypass = traversal.reference;
  }
  require(test, acyclic.size() == 2 && cyclic.size() == 2 && buffered && bypass,
          "fixture did not expose the exact switch and FIFO alternatives");

  FabricHandshakeSelection legal;
  legal.traversals = acyclic;
  if (llvm::Error error =
          loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
              finalized.view(), legal))
    fail(test, llvm::toString(std::move(error)));

  const auto attachments =
      finalized.view().moduleBoundaryTransportAttachments();
  require(test, attachments.size() == 2,
          "selected-cycle fixture lost its Module boundary attachments");
  std::optional<loom::fabric::FabricTransportEndpointRef> input;
  std::optional<loom::fabric::FabricTransportEndpointRef> output;
  for (const auto &attachment : attachments) {
    if (attachment.boundary.direction == FabricPortDirection::Input)
      input = attachment.endpoint;
    else
      output = attachment.endpoint;
  }
  require(test, input && output,
          "selected-cycle fixture has incomplete boundary directions");
  const std::array<HandshakeSignalRef, 4> terminals = {
      HandshakeSignalRef{*input, HandshakeSignalKind::Valid},
      HandshakeSignalRef{*input, HandshakeSignalKind::Ready},
      HandshakeSignalRef{*output, HandshakeSignalKind::Valid},
      HandshakeSignalRef{*output, HandshakeSignalKind::Ready}};
  const auto reachability =
      take(test, loom::fabric::deriveSelectedHandshakeReachability(
                     finalized.view(), legal, terminals));
  require(test,
          llvm::is_contained(reachability,
                             loom::fabric::HandshakeDependencyArc{
                                 terminals[0], terminals[2]}) &&
              llvm::is_contained(reachability,
                                 loom::fabric::HandshakeDependencyArc{
                                     terminals[3], terminals[1]}),
          "selected reachability lost forward valid or backward ready");

  FabricHandshakeSelection isolatedCycle;
  isolatedCycle.traversals = cyclic;
  isolatedCycle.traversals.push_back(*buffered);
  if (llvm::Error error =
          loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
              finalized.view(), isolatedCycle))
    fail(test, "selected Buffered traversal did not isolate the cycle: " +
                   llvm::toString(std::move(error)));

  FabricHandshakeSelection illegal;
  illegal.traversals = cyclic;
  illegal.traversals.push_back(*bypass);
  requireRejected(test,
                  loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
                      finalized.view(), illegal),
                  "SelectedCombinationalHandshakeCycle");
}

void atomicBoundarySelectionActivatesWholeOwner() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto source = parse(test, R"mlir(
    module {
      fabric.module @split(%input: !fabric.bits_tag<32, 4>)
          -> (!fabric.bits<32>, !fabric.bits<4>) {
        %data, %tag = fabric.boundary [t2s] %input
          : !fabric.bits_tag<32, 4>
          -> (!fabric.bits<32>, !fabric.bits<4>)
        fabric.yield %data, %tag : !fabric.bits<32>, !fabric.bits<4>
      }
    }
  )mlir");
  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(finalized.view()));
  const HandshakeOwnerModel *model = nullptr;
  for (const HandshakeOwnerModel &candidate : models) {
    if (candidate.owner().kind() !=
        FabricHandshakeOwnerKind::BoundaryOccurrence)
      continue;
    if (model)
      fail(test, "fixture has more than one boundary handshake owner");
    model = &candidate;
  }
  require(test, model != nullptr, "fixture has no boundary handshake owner");

  std::vector<FabricPhysicalTraversalRef> legs;
  for (const auto &traversal : finalized.view().physicalTraversals())
    if (traversal.reference.kind() ==
        FabricPhysicalTraversalKind::BoundaryTraversal)
      legs.push_back(traversal.reference);
  require(test, legs.size() == 2,
          "split boundary did not expose two physical legs");

  FabricHandshakeSelection partial;
  partial.traversals.push_back(legs.front());
  ResolvedHandshakeActivation partialActivation =
      take(test, loom::fabric::resolveSelectedHandshake(*model, partial));

  FabricHandshakeSelection complete;
  complete.traversals = legs;
  ResolvedHandshakeActivation activation =
      take(test, loom::fabric::resolveSelectedHandshake(*model, complete));
  require(test,
          !partialActivation.arcOrdinals().empty() &&
              llvm::equal(partialActivation.arcOrdinals(),
                          activation.arcOrdinals()),
          "one selected boundary leg did not activate the whole atomic owner");
}

void oneToOneBoundariesUseDirectHandshake() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto source = parse(test, R"mlir(
    module {
      fabric.module @direct_boundaries(
          %data: !fabric.bits<16>,
          %rewrite_input: !fabric.bits_tag<8, 3>,
          %remove_input: !fabric.bits_tag<64, 5>)
          -> (!fabric.bits_tag<16, 6>, !fabric.bits_tag<8, 7>,
              !fabric.bits<64>) {
        %configured = fabric.boundary [s2t] %data
            : !fabric.bits<16> -> !fabric.bits_tag<16, 6>
        %rewritten = fabric.boundary [t2t] %rewrite_input
            {hw_params = [{lut_size = 5 : i32}]}
            : !fabric.bits_tag<8, 3> -> !fabric.bits_tag<8, 7>
        %removed = fabric.boundary [t2s] %remove_input
            : !fabric.bits_tag<64, 5> -> !fabric.bits<64>
        fabric.yield %configured, %rewritten, %removed
            : !fabric.bits_tag<16, 6>, !fabric.bits_tag<8, 7>,
              !fabric.bits<64>
      }
    }
  )mlir");
  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  const auto unconditional =
      take(test, loom::fabric::deriveUnconditionalHandshakeDependencyArcs(
                     finalized.view()));
  require(test, unconditional.empty(),
          "unconfigured one-to-one boundary became an unconditional arc");
  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(finalized.view()));

  std::uint32_t checked = 0;
  for (const HandshakeOwnerModel &model : models) {
    if (model.owner().kind() != FabricHandshakeOwnerKind::BoundaryOccurrence)
      continue;
    const auto boundary = std::get<loom::fabric::FabricBoundaryOccurrenceRef>(
        model.owner().payload());
    const auto endpointOwner =
        loom::fabric::FabricTransportEndpointOwnerRef::of(boundary);
    std::optional<loom::fabric::FabricTransportEndpointRef> input;
    std::optional<loom::fabric::FabricTransportEndpointRef> output;
    for (std::uint64_t ordinal = 0;
         ordinal < finalized.view().transportEndpointCount(endpointOwner);
         ++ordinal) {
      const loom::fabric::FabricTransportEndpointRef endpoint{endpointOwner,
                                                              ordinal};
      const auto direction =
          finalized.view().transportEndpointDirection(endpoint);
      if (direction == FabricPortDirection::Input) {
        require(test, !input.has_value(),
                "one-to-one boundary has multiple inputs");
        input = endpoint;
      } else if (direction == FabricPortDirection::Output) {
        require(test, !output.has_value(),
                "one-to-one boundary has multiple outputs");
        output = endpoint;
      }
    }
    require(test, input.has_value() && output.has_value(),
            "one-to-one boundary omitted an endpoint");

    std::optional<FabricPhysicalTraversalRef> selectedTraversal;
    for (const FabricPhysicalTraversalRef &traversal :
         finalized.view().admittedTraversals()) {
      if (traversal.kind() != FabricPhysicalTraversalKind::BoundaryTraversal)
        continue;
      if (std::get<loom::fabric::FabricBoundaryTraversalPayload>(
              traversal.payload)
              .owner != boundary)
        continue;
      require(test, !selectedTraversal.has_value(),
              "one-to-one boundary has multiple traversal legs");
      selectedTraversal = traversal;
    }
    require(test, selectedTraversal.has_value(),
            "one-to-one boundary has no traversal leg");

    FabricHandshakeSelection selection;
    selection.traversals.push_back(*selectedTraversal);
    const ResolvedHandshakeActivation activation =
        take(test, loom::fabric::resolveSelectedHandshake(model, selection));
    require(test, activation.arcOrdinals().size() == 2,
            "one-to-one boundary does not own two direct handshake arcs");
    require(test,
            hasPath(model, activation,
                    node(test, model, {*input, HandshakeSignalKind::Valid}),
                    node(test, model, {*output, HandshakeSignalKind::Valid})),
            "one-to-one boundary lost forward valid dependence");
    require(test,
            hasPath(model, activation,
                    node(test, model, {*output, HandshakeSignalKind::Ready}),
                    node(test, model, {*input, HandshakeSignalKind::Ready})),
            "one-to-one boundary lost backward ready dependence");
    ++checked;
  }
  require(test, checked == 3,
          "fixture did not validate all one-to-one boundary forms");
}

void registerFifoPathsAreRegisteredBreaks() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto source = parse(test, R"mlir(
    module {
      fabric.module @temporal_registers(
          %lhs : !fabric.bits_tag<16, 3>,
          %rhs : !fabric.bits_tag<16, 3>) -> !fabric.bits_tag<16, 3> {
        %result = fabric.pe [temporal]
            (%pe_lhs = %lhs : !fabric.bits_tag<16, 3>,
             %pe_rhs = %rhs : !fabric.bits_tag<16, 3>)
            -> !fabric.bits_tag<16, 3>
            attributes {
              tag_width = 3 : i32,
              num_instruction = 2 : i32,
              num_reg_fifo = 2 : i32,
              reg_fifo_depth = 4 : i32,
              reg_fifo_ports = 2 : i32,
              fu_config_mode = "per_fu_config",
              operand_buffer_mode = #fabric.operand_buffer_mode<per_input_port>,
              operand_buffer_size = 4 : i32
            } {
          fabric.fu(%fu_lhs = %pe_lhs : !fabric.bits<16>,
                    %fu_rhs = %pe_rhs : !fabric.bits<16>)
              -> !fabric.bits<16> {
            %sum = fabric.op [@arith.addi] (%fu_lhs, %fu_rhs) {
              implementation_family =
                #fabric.implementation_family<ScalarIntegerAddSub>,
              hw_params = {integer_widths = [16 : i32]}
            } : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
            fabric.yield %sum : !fabric.bits<16>
          }
        }
        fabric.yield %result : !fabric.bits_tag<16, 3>
      }
    }
  )mlir");
  materializeOperationContracts(test, *source);
  FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  require(test, finalized.view().peOccurrences().size() == 1,
          "fixture has no unique temporal PE occurrence");
  const auto pe = finalized.view().peOccurrences().front();
  require(test,
          finalized.view().inventorySize(FabricInventoryOwnerRef::of(pe),
                                         FabricInventoryKind::RegisterFifo) ==
              2,
          "temporal PE omitted its register-FIFO inventory");

  std::vector<FabricPhysicalTraversalRef> paths;
  for (const auto &traversal : finalized.view().physicalTraversals()) {
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::PeRegisterFifoTraversal)
      continue;
    const auto &payload = std::get<loom::fabric::FabricPeRegisterFifoPayload>(
        traversal.reference.payload);
    require(test, payload.owner == pe && payload.registerFifo < 2,
            "register-FIFO traversal has a stale owner or ordinal");
    require(test, traversal.sources.empty() && traversal.destinations.empty(),
            "registered register-FIFO path exposed a combinational endpoint");
    require(test, traversal.resourceStates.size() == 1,
            "register-FIFO path omitted its exact resource state");
    paths.push_back(traversal.reference);
  }
  require(test, paths.size() == 4,
          "temporal PE did not expose one read and one write path per FIFO");
  for (std::uint64_t fifo = 0; fifo != 2; ++fifo) {
    for (FabricRegisterFifoPathRole role :
         {FabricRegisterFifoPathRole::Write, FabricRegisterFifoPathRole::Read})
      require(test,
              llvm::is_contained(
                  paths,
                  FabricPhysicalTraversalRef::peRegisterFifo(pe, fifo, role)),
              "temporal PE omitted a typed register-FIFO path");
  }

  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(finalized.view()));
  const HandshakeOwnerModel *model = nullptr;
  for (const HandshakeOwnerModel &candidate : models) {
    if (candidate.owner().kind() != FabricHandshakeOwnerKind::PeOccurrence)
      continue;
    if (model)
      fail(test, "fixture has more than one PE handshake owner");
    model = &candidate;
  }
  require(test, model != nullptr, "fixture has no PE handshake owner");
  for (const FabricPhysicalTraversalRef &path : paths) {
    FabricHandshakeSelection selection;
    selection.traversals.push_back(path);
    ResolvedHandshakeActivation activation =
        take(test, loom::fabric::resolveSelectedHandshake(*model, selection));
    require(test, activation.arcOrdinals().empty(),
            "registered register-FIFO path created a combinational arc");
  }
}

void memoryOperationPlanOwnsAtomicBoundaryAndRegisteredBreak() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto target = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  require(test,
          target.roots().size() == 1 &&
              target.roots().front().directDependencies().size() == 1,
          "builtin fixture did not publish one SpatialCore dependency");
  FinalizedFabricRoot module =
      take(test, loom::fabric::importEntireFabricRoot(
                     target.roots().front().directDependencies().front().root,
                     store));

  std::optional<FabricMemoryOccurrenceRef> memory;
  for (FabricMemoryOccurrenceRef candidate :
       module.view().memoryOccurrences()) {
    if (module.view().memorySchedule(candidate) ==
        ::fabric::Schedule::Spatial) {
      memory = candidate;
      break;
    }
  }
  require(test, memory.has_value(),
          "builtin fixture has no Spatial memory occurrence");
  const auto ports = module.view().memoryOperationPorts(*memory);
  require(test, !ports.empty(), "Spatial memory has no operation port");
  const FabricMemoryOperationPortRef port = ports.front();
  const loom::fabric::FabricMemoryCapabilityAlternativeRef capability{port, 0};
  const auto *alternative =
      module.view().memoryCapabilityAlternative(capability);
  require(test, alternative && !alternative->admissibleUsePatterns.empty(),
          "memory capability has no admissible operation plan");
  const loom::fabric::FabricUsePatternRef usePattern{
      loom::fabric::FabricUsePatternOwnerRef(FabricInventoryOwnerRef::of(port)),
      alternative->admissibleUsePatterns.front().ordinal()};
  const auto selected = take(
      test, loom::fabric::makeMemoryHandshakeSelection(
                module.view(), FabricMemoryHandshakePlacement(port), capability,
                usePattern, ::dataflow::semantics::MemoryMaskForm::Absent));

  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(module.view()));
  const HandshakeOwnerModel *model = nullptr;
  for (const HandshakeOwnerModel &candidate : models) {
    if (candidate.owner().kind() != FabricHandshakeOwnerKind::MemoryOccurrence)
      continue;
    if (std::get<FabricMemoryOccurrenceRef>(candidate.owner().payload()) !=
        *memory)
      continue;
    model = &candidate;
    break;
  }
  require(test, model != nullptr,
          "memory occurrence has no handshake owner model");

  FabricHandshakeSelection selection;
  selection.memoryOperations.push_back(selected);
  ResolvedHandshakeActivation activation =
      take(test, loom::fabric::resolveSelectedHandshake(*model, selection));
  require(test, !activation.arcOrdinals().empty(),
          "memory operation plan activated no handshake equations");

  auto endpointForRole = [&](::dataflow::semantics::ServiceValueRole role) {
    const auto binding = llvm::find_if(
        alternative->roleToEndpoint,
        [&](const ::fabric::MemoryRoleEndpointBindingRecord &candidate) {
          return candidate.role == role;
        });
    if (binding == alternative->roleToEndpoint.end())
      fail(test, "memory capability omitted an expected service role");
    return loom::fabric::FabricTransportEndpointRef{
        loom::fabric::FabricTransportEndpointOwnerRef::of(*memory),
        binding->endpointOrdinal};
  };
  const auto address =
      endpointForRole(::dataflow::semantics::ServiceValueRole::Address);
  const auto control =
      endpointForRole(::dataflow::semantics::ServiceValueRole::Control);
  const auto data =
      endpointForRole(::dataflow::semantics::ServiceValueRole::Data);
  const auto completion =
      endpointForRole(::dataflow::semantics::ServiceValueRole::Completion);

  const auto addressValid =
      node(test, *model, {address, HandshakeSignalKind::Valid});
  const auto addressReady =
      node(test, *model, {address, HandshakeSignalKind::Ready});
  const auto controlValid =
      node(test, *model, {control, HandshakeSignalKind::Valid});
  const auto controlReady =
      node(test, *model, {control, HandshakeSignalKind::Ready});
  const auto dataValid = node(test, *model, {data, HandshakeSignalKind::Valid});
  const auto dataReady = node(test, *model, {data, HandshakeSignalKind::Ready});
  const auto completionValid =
      node(test, *model, {completion, HandshakeSignalKind::Valid});
  const auto completionReady =
      node(test, *model, {completion, HandshakeSignalKind::Ready});

  require(test,
          hasPath(*model, activation, addressValid, controlReady) &&
              hasPath(*model, activation, controlValid, addressReady),
          "memory request roles are not one atomic rendezvous");
  require(test,
          hasPath(*model, activation, dataReady, completionValid) &&
              hasPath(*model, activation, completionReady, dataValid),
          "memory response roles are not one atomic publication");
  require(test,
          !hasPath(*model, activation, addressValid, dataValid) &&
              !hasPath(*model, activation, dataReady, addressReady),
          "memory holding state failed to cut request-to-response paths");

  requireRejected(test,
                  loom::fabric::makeMemoryHandshakeSelection(
                      module.view(), FabricMemoryHandshakePlacement(port),
                      capability,
                      loom::fabric::FabricUsePatternRef{
                          loom::fabric::FabricUsePatternOwnerRef(
                              FabricInventoryOwnerRef::of(port)),
                          usePattern.ordinal + 1},
                      ::dataflow::semantics::MemoryMaskForm::Absent),
                  "unknown use pattern");
}

void fuSelectionUsesExactActorPortCorrespondence() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  auto target = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  FinalizedFabricRoot module =
      take(test, loom::fabric::importEntireFabricRoot(
                     target.roots().front().directDependencies().front().root,
                     store));

  std::optional<FabricFuOccurrenceRef> occurrence;
  std::optional<FabricFuCapabilityTemplateRef> templateRef;
  std::optional<FabricFuTemplateNodeRef> operation;
  for (FabricFuOccurrenceRef candidate : module.view().fuOccurrences()) {
    const auto definition = module.view().fuTemplateOf(candidate);
    if (!definition)
      continue;
    for (auto [ordinal, row] :
         llvm::enumerate(module.view().fuCapabilityTemplates(*definition))) {
      std::vector<FabricFuTemplateNodeRef> operations;
      for (const FabricFuTemplateNodeRef &node : row.activeNodes)
        if (node.node == FabricFuNodeKind::Op)
          operations.push_back(node);
      if (operations.size() != 1)
        continue;
      const auto *capability =
          module.view().resolvedFabricOpCapability(operations.front());
      if (!capability || capability->implementationFamily !=
                             ::fabric::ImplementationFamilyId::TokenSync)
        continue;
      occurrence = candidate;
      templateRef = FabricFuCapabilityTemplateRef{
          *definition, static_cast<loom::fabric::FabricOrdinal>(ordinal)};
      operation = operations.front();
      break;
    }
    if (occurrence)
      break;
  }
  require(test, occurrence && templateRef && operation,
          "builtin fixture has no TokenSync capability row");

  mlir::Type i32 = mlir::IntegerType::get(&context(), 32);
  dataflow::CanonicalActorSchemaProjection actor{
      dataflow::OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context(), {i32, i32}, {i32, i32}),
      dataflow::NoPayload{}};
  loom::fabric::FabricFuOperationHandshakeBinding binding{
      *operation, actor, 64, std::nullopt, {0, 1}, {0, 1}};
  const auto selected =
      take(test, loom::fabric::makeFuHandshakeSelection(
                     module.view(), *occurrence, *templateRef, {binding}));

  std::vector<HandshakeOwnerModel> models =
      take(test, loom::fabric::compileHandshakeOwnerModels(module.view()));
  const HandshakeOwnerModel *model = nullptr;
  for (const HandshakeOwnerModel &candidate : models) {
    if (candidate.owner().kind() != FabricHandshakeOwnerKind::FuOccurrence ||
        std::get<FabricFuOccurrenceRef>(candidate.owner().payload()) !=
            *occurrence)
      continue;
    model = &candidate;
    break;
  }
  require(test, model != nullptr, "FU occurrence has no handshake model");

  const auto &row = module.view().fuCapabilityTemplates(
      templateRef->fu)[templateRef->ordinal];
  const auto terminalEdges = take(
      test, loom::fabric::projectFabricFuCapabilityTemplateTerminalEdges(row));
  auto boundaryForOperationPort = [&](FabricPortDirection direction,
                                      std::uint64_t ordinal) {
    for (const auto &edge : terminalEdges) {
      const auto *nodePort =
          direction == FabricPortDirection::Input
              ? std::get_if<loom::fabric::FabricFuNodePortRef>(
                    &edge.destination.payload)
              : std::get_if<loom::fabric::FabricFuNodePortRef>(
                    &edge.source.payload);
      const auto *boundary =
          direction == FabricPortDirection::Input
              ? std::get_if<loom::fabric::FabricFuTemplatePortRef>(
                    &edge.source.payload)
              : std::get_if<loom::fabric::FabricFuTemplatePortRef>(
                    &edge.destination.payload);
      if (!nodePort || !boundary || nodePort->node != *operation ||
          nodePort->direction != direction || nodePort->ordinal != ordinal)
        continue;
      auto endpoint = module.view().fuOccurrenceTransportEndpoint(
          {*occurrence, boundary->direction, boundary->ordinal});
      if (!endpoint)
        fail(test, "FU boundary has no occurrence endpoint");
      return *endpoint;
    }
    fail(test, "capability row omitted an operation-port boundary");
  };

  const auto input0 = boundaryForOperationPort(FabricPortDirection::Input, 0);
  const auto input1 = boundaryForOperationPort(FabricPortDirection::Input, 1);
  const auto input2 = boundaryForOperationPort(FabricPortDirection::Input, 2);
  const auto output0 = boundaryForOperationPort(FabricPortDirection::Output, 0);
  const auto output1 = boundaryForOperationPort(FabricPortDirection::Output, 1);
  const auto output2 = boundaryForOperationPort(FabricPortDirection::Output, 2);

  FabricHandshakeSelection selection;
  selection.fuCapabilities.push_back(selected);
  if (llvm::Error error =
          loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
              module.view(), selection))
    fail(test, llvm::toString(std::move(error)));
  const ResolvedHandshakeActivation activation =
      take(test, loom::fabric::resolveSelectedHandshake(*model, selection));
  require(test,
          hasPath(*model, activation,
                  node(test, *model, {input0, HandshakeSignalKind::Valid}),
                  node(test, *model, {input1, HandshakeSignalKind::Ready})),
          "selected sync inputs are not one atomic rendezvous");
  require(test,
          hasPath(*model, activation,
                  node(test, *model, {output0, HandshakeSignalKind::Ready}),
                  node(test, *model, {output1, HandshakeSignalKind::Valid})),
          "selected sync results are not one atomic publication");
  require(test,
          !hasPath(*model, activation,
                   node(test, *model, {input0, HandshakeSignalKind::Valid}),
                   node(test, *model, {output0, HandshakeSignalKind::Valid})),
          "registered sync operation failed to cut forward valid");
  require(
      test,
      !hasPath(*model, activation,
               node(test, *model, {input0, HandshakeSignalKind::Valid}),
               node(test, *model, {input2, HandshakeSignalKind::Ready})) &&
          !hasPath(*model, activation,
                   node(test, *model, {output2, HandshakeSignalKind::Ready}),
                   node(test, *model, {output0, HandshakeSignalKind::Valid})),
      "inactive physical sync lanes created backpressure");

  const auto *capability = module.view().resolvedFabricOpCapability(*operation);
  require(test, capability != nullptr,
          "selected sync operation has no resolved capability");
  std::vector<mlir::Type> fullTypes;
  std::vector<std::uint64_t> fullInputs;
  std::vector<std::uint64_t> fullResults;
  for (const auto &port : capability->physicalPorts) {
    if (port.reference.direction == FabricPortDirection::Input)
      fullInputs.push_back(port.reference.ordinal);
    else
      fullResults.push_back(port.reference.ordinal);
  }
  require(test, fullInputs.size() == fullResults.size(),
          "TokenSync physical lane inventories disagree");
  fullTypes.assign(fullInputs.size(), i32);
  dataflow::CanonicalActorSchemaProjection fullActor{
      dataflow::OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context(), fullTypes, fullTypes),
      dataflow::NoPayload{}};
  loom::fabric::FabricFuOperationHandshakeBinding fullBinding{
      *operation, fullActor, 64, std::nullopt, fullInputs, fullResults};
  const auto fullSelection =
      take(test, loom::fabric::makeFuHandshakeSelection(
                     module.view(), *occurrence, *templateRef, {fullBinding}));
  FabricHandshakeSelection full;
  full.fuCapabilities.push_back(fullSelection);
  if (llvm::Error error =
          loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
              module.view(), full))
    fail(test, "full-width registered sync closed a combinational cycle: " +
                   llvm::toString(std::move(error)));
}

void fullWidthDirectSyncIsAcyclic() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  const PortType bits128 = take(test, PortType::bits(128));
  const std::vector<PortType> types(2, bits128);
  DesignBuilder design(store);
  auto spatial =
      take(test, design.createSpatialCore("direct-sync", types, types));
  auto pe = take(test, spatial.addPe({take(test, spatial.input(0)),
                                      take(test, spatial.input(1))},
                                     loom::adg::PeSpec::spatial(types, types)));
  auto fu =
      take(test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                          loom::adg::FuSpec{types, types}));
  auto operation = take(
      test, fu.addOperation(
                {take(test, fu.input(0)), take(test, fu.input(1))},
                loom::adg::OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::TokenSync,
                    ::fabric::RoutedTokenParams{128, 2},
                    {::dataflow::OperationSchemaId::DataflowSync},
                    types,
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error = fu.addCapabilityTemplate(
          loom::adg::FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close(
          {take(test, operation.output(0)), take(test, operation.output(1))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          spatial.close({take(test, pe.output(0)), take(test, pe.output(1))}))
    fail(test, llvm::toString(std::move(error)));
  FinalizedFabricRoot finalized =
      take(test, std::move(design).finalize()).roots().front();

  const FabricFuOccurrenceRef occurrence =
      finalized.view().fuOccurrences().front();
  const auto definition = finalized.view().fuTemplateOf(occurrence);
  require(test, definition.has_value(), "direct sync FU has no definition");
  const FabricFuCapabilityTemplateRef capability{*definition, 0};
  const auto &row = finalized.view().fuCapabilityTemplates(*definition).front();
  const auto selectedOperation =
      llvm::find_if(row.activeNodes, [](const auto &node) {
        return node.node == FabricFuNodeKind::Op;
      });
  require(test, selectedOperation != row.activeNodes.end(),
          "direct sync capability has no operation");
  mlir::Type none = mlir::NoneType::get(&context());
  mlir::Type i32 = mlir::IntegerType::get(&context(), 32);
  dataflow::CanonicalActorSchemaProjection actor{
      dataflow::OperationSchemaId::DataflowSync,
      mlir::FunctionType::get(&context(), {none, i32}, {none, i32}),
      dataflow::NoPayload{}};
  loom::fabric::FabricFuOperationHandshakeBinding binding{
      *selectedOperation, actor, 64, std::nullopt, {0, 1}, {0, 1}};
  const auto selected =
      take(test, loom::fabric::makeFuHandshakeSelection(
                     finalized.view(), occurrence, capability, {binding}));
  FabricHandshakeSelection selection;
  selection.fuCapabilities.push_back(selected);
  if (llvm::Error error =
          loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
              finalized.view(), selection))
    fail(test, llvm::toString(std::move(error)));
}

} // namespace

int main() {
  selectedPointConnectionConsumesItsWitness();
  atomicBroadcastProjectionIsLinear(64);
  atomicBroadcastProjectionIsLinear(256);
  fifoModeOwnsItsExactCombinationalBreak();
  bufferedPhysicalCycleIsAcceptedBeforeSelection();
  selectedGlobalCycleUsesExactTraversalSelection();
  atomicBoundarySelectionActivatesWholeOwner();
  oneToOneBoundariesUseDirectHandshake();
  registerFifoPathsAreRegisteredBreaks();
  memoryOperationPlanOwnsAtomicBoundaryAndRegisteredBreak();
  fuSelectionUsesExactActorPortCorrespondence();
  fullWidthDirectSyncIsAcyclic();
  return EXIT_SUCCESS;
}
