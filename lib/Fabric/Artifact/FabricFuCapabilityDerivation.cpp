#include "FabricFuCapabilityDerivation.h"

#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "fabric_artifact_invalid: FU capability derivation " + message);
}

FabricFuNodeKind nodeKind(Operation *operation) {
  if (isa<::fabric::MuxOp>(operation))
    return FabricFuNodeKind::Mux;
  if (isa<::fabric::DemuxOp>(operation))
    return FabricFuNodeKind::Demux;
  return FabricFuNodeKind::Op;
}

struct DerivationState {
  llvm::DenseSet<Operation *> activeNodes;
  llvm::DenseSet<Value> activeValues;
  llvm::DenseMap<Operation *, unsigned> muxSelections;
  llvm::DenseMap<Operation *, unsigned> demuxSelections;
  std::vector<FabricFuCapabilityTemplateEdge> edges;
};

using StateList = std::vector<DerivationState>;

class Deriver {
public:
  Deriver(::fabric::FuOp fu, FabricFuTemplateRef owner,
          llvm::ArrayRef<Operation *> canonicalNodeOrder)
      : fu_(fu), owner_(owner), canonicalNodeOrder_(canonicalNodeOrder) {
    for (auto [ordinal, operation] : llvm::enumerate(canonicalNodeOrder)) {
      nodeRefs_.try_emplace(
          operation,
          FabricFuTemplateNodeRef{nodeKind(operation), owner,
                                  static_cast<FabricOrdinal>(ordinal)});
    }
  }

  llvm::Expected<std::vector<FabricFuCapabilityTemplateRecord>> run() {
    std::vector<FabricFuCapabilityTemplateRecord> records;
    for (Operation &operation : fu_.getBody().front().without_terminator()) {
      if (!isa<::fabric::OpOp>(operation))
        continue;
      DerivationState initial;
      StateList states = activateNode(std::move(initial), &operation);
      if (failed_)
        return invalid(failure_);
      for (DerivationState &state : states) {
        FabricFuCapabilityTemplateRecord record;
        record.activeNodes.reserve(state.activeNodes.size());
        for (Operation *active : state.activeNodes)
          record.activeNodes.push_back(nodeRefs_.lookup(active));
        record.activeEdges = std::move(state.edges);
        auto normalized =
            normalizeFabricFuCapabilityTemplateRecord(std::move(record));
        if (!normalized)
          return normalized.takeError();
        records.push_back(std::move(*normalized));
      }
    }
    if (failed_)
      return invalid(failure_);
    if (records.empty())
      return invalid("found no complete physical template");

    using KeyedRecord =
        std::pair<std::vector<std::uint8_t>, FabricFuCapabilityTemplateRecord>;
    std::vector<KeyedRecord> keyed;
    keyed.reserve(records.size());
    for (FabricFuCapabilityTemplateRecord &record : records) {
      auto bytes = canonicalFabricFuCapabilityTemplateBytes(record);
      if (!bytes)
        return bytes.takeError();
      keyed.emplace_back(std::move(*bytes), std::move(record));
    }
    llvm::sort(keyed, [](const KeyedRecord &left, const KeyedRecord &right) {
      return left.first < right.first;
    });

    records.clear();
    records.reserve(keyed.size());
    for (std::size_t index = 0; index < keyed.size(); ++index) {
      if (index != 0 && keyed[index - 1].first == keyed[index].first)
        continue;
      records.push_back(std::move(keyed[index].second));
    }
    return normalizeFabricFuCapabilityTemplateInventory(records);
  }

  llvm::Expected<FabricFuCapabilityTemplateRecord>
  run(const ::fabric::FuCapabilityTemplateSelection &selection) {
    constrained_ = true;
    DerivationState initial;

    for (std::uint64_t ordinal : selection.activeOperationNodeOrdinals) {
      if (ordinal >= canonicalNodeOrder_.size())
        return invalid("names an active operation outside the FU node domain");
      Operation *operation = canonicalNodeOrder_[ordinal];
      if (!isa<::fabric::OpOp>(operation))
        return invalid("names a non-operation as an active operation");
      declaredOperations_.insert(operation);
    }

    for (const ::fabric::FuCapabilityRouteSelection &route : selection.routes) {
      if (route.selectorNodeOrdinal >= canonicalNodeOrder_.size())
        return invalid("names a selector outside the FU node domain");
      Operation *selector = canonicalNodeOrder_[route.selectorNodeOrdinal];
      if (auto mux = dyn_cast<::fabric::MuxOp>(selector)) {
        if (route.selectedPort >= mux.getInputs().size())
          return invalid("selects a mux input outside its port domain");
        initial.muxSelections[selector] = route.selectedPort;
      } else if (auto demux = dyn_cast<::fabric::DemuxOp>(selector)) {
        if (route.selectedPort >= demux.getOutputs().size())
          return invalid("selects a demux output outside its port domain");
        initial.demuxSelections[selector] = route.selectedPort;
      } else {
        return invalid("names a non-selector as a route selection");
      }
      declaredSelectors_.insert(selector);
    }

    StateList states{std::move(initial)};
    for (std::uint64_t ordinal : selection.activeOperationNodeOrdinals) {
      Operation *operation = canonicalNodeOrder_[ordinal];
      states = expand(std::move(states), [&](DerivationState state) {
        return activateNode(std::move(state), operation);
      });
    }
    if (failed_)
      return invalid(failure_);
    if (states.size() != 1)
      return invalid(states.empty()
                         ? "row does not materialize a complete physical graph"
                         : "row materializes more than one physical graph");

    DerivationState &state = states.front();
    for (Operation *operation : state.activeNodes) {
      if (isa<::fabric::OpOp>(operation) &&
          !declaredOperations_.contains(operation))
        return invalid("activates an operation absent from its row");
      if (isa<::fabric::MuxOp, ::fabric::DemuxOp>(operation) &&
          !declaredSelectors_.contains(operation))
        return invalid("activates a selector absent from its row");
    }
    for (Operation *operation : declaredOperations_)
      if (!state.activeNodes.contains(operation))
        return invalid("declares an operation absent from its physical graph");
    for (Operation *selector : declaredSelectors_)
      if (!state.activeNodes.contains(selector))
        return invalid("declares an unused selector route");

    FabricFuCapabilityTemplateRecord record;
    record.activeNodes.reserve(state.activeNodes.size());
    for (Operation *active : state.activeNodes)
      record.activeNodes.push_back(nodeRefs_.lookup(active));
    record.activeEdges = std::move(state.edges);
    return normalizeFabricFuCapabilityTemplateRecord(std::move(record));
  }

private:
  std::optional<FabricFuCapabilityTemplateEndpointRef>
  sourceEndpoint(Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() != &fu_.getBody().front()) {
        fail("encountered a block argument outside the owning FU");
        return std::nullopt;
      }
      return FabricFuCapabilityTemplateEndpointRef::boundaryPort(
          FabricFuTemplatePortRef{owner_, FabricPortDirection::Input,
                                  argument.getArgNumber()});
    }
    auto result = dyn_cast<OpResult>(value);
    if (!result || !nodeRefs_.count(result.getOwner())) {
      fail("encountered a value outside the owning FU graph");
      return std::nullopt;
    }
    return FabricFuCapabilityTemplateEndpointRef::nodePort(FabricFuNodePortRef{
        nodeRefs_.lookup(result.getOwner()), FabricPortDirection::Output,
        result.getResultNumber()});
  }

  FabricFuCapabilityTemplateEndpointRef destinationEndpoint(OpOperand &use) {
    Operation *owner = use.getOwner();
    if (isa<::fabric::YieldOp>(owner))
      return FabricFuCapabilityTemplateEndpointRef::boundaryPort(
          FabricFuTemplatePortRef{owner_, FabricPortDirection::Output,
                                  use.getOperandNumber()});
    auto found = nodeRefs_.find(owner);
    if (found == nodeRefs_.end()) {
      fail("encountered a consumer outside the owning FU graph");
      return FabricFuCapabilityTemplateEndpointRef::boundaryPort(
          FabricFuTemplatePortRef{owner_, FabricPortDirection::Output, 0});
    }
    return FabricFuCapabilityTemplateEndpointRef::nodePort(FabricFuNodePortRef{
        found->second, FabricPortDirection::Input, use.getOperandNumber()});
  }

  bool addEdge(DerivationState &state, Value source, OpOperand &use) {
    auto sourceRef = sourceEndpoint(source);
    if (!sourceRef)
      return false;
    FabricFuCapabilityTemplateEdge edge{*sourceRef, destinationEndpoint(use)};
    if (failed_)
      return false;
    if (std::find(state.edges.begin(), state.edges.end(), edge) ==
        state.edges.end())
      state.edges.push_back(std::move(edge));
    return true;
  }

  StateList activateNode(DerivationState state, Operation *operation) {
    if (auto mux = dyn_cast<::fabric::MuxOp>(operation)) {
      auto selected = state.muxSelections.find(operation);
      if (selected != state.muxSelections.end()) {
        const unsigned input = selected->second;
        return selectMuxInput(std::move(state), mux, input);
      }
      if (constrained_) {
        fail("reaches a mux without a selected input");
        return {};
      }
      StateList result;
      for (unsigned input = 0; input < mux.getInputs().size(); ++input)
        append(result, selectMuxInput(state, mux, input));
      return result;
    }
    if (auto demux = dyn_cast<::fabric::DemuxOp>(operation)) {
      auto selected = state.demuxSelections.find(operation);
      if (selected != state.demuxSelections.end()) {
        const unsigned output = selected->second;
        return selectDemuxOutput(std::move(state), demux, output);
      }
      if (constrained_) {
        fail("reaches a demux without a selected output");
        return {};
      }
      StateList result;
      for (unsigned output = 0; output < demux.getOutputs().size(); ++output) {
        if (demux.getOutputs()[output].use_empty())
          continue;
        append(result, selectDemuxOutput(state, demux, output));
      }
      return result;
    }

    if (!isa<::fabric::OpOp>(operation)) {
      fail("encountered a nonphysical FU node");
      return {};
    }
    if (!state.activeNodes.insert(operation).second)
      return {std::move(state)};

    StateList states{std::move(state)};
    for (Value input : operation->getOperands())
      states = expand(std::move(states), [&](DerivationState current) {
        return activateValue(std::move(current), input);
      });
    for (Value output : operation->getResults())
      states = expand(std::move(states), [&](DerivationState current) {
        return activateValue(std::move(current), output);
      });
    return states;
  }

  StateList selectMuxInput(DerivationState state, ::fabric::MuxOp mux,
                           unsigned input) {
    auto [position, inserted] =
        state.muxSelections.try_emplace(mux.getOperation(), input);
    if (!inserted && position->second != input)
      return {};
    if (!state.activeNodes.insert(mux.getOperation()).second)
      return {std::move(state)};
    if (!addEdge(state, mux.getInputs()[input], mux->getOpOperand(input)))
      return {};
    StateList states = activateValue(std::move(state), mux.getInputs()[input]);
    return expand(std::move(states), [&](DerivationState current) {
      return activateValue(std::move(current), mux.getOutput());
    });
  }

  StateList selectDemuxOutput(DerivationState state, ::fabric::DemuxOp demux,
                              unsigned output) {
    auto [position, inserted] =
        state.demuxSelections.try_emplace(demux.getOperation(), output);
    if (!inserted && position->second != output)
      return {};
    if (!state.activeNodes.insert(demux.getOperation()).second)
      return {std::move(state)};
    if (!addEdge(state, demux.getInput(), demux->getOpOperand(0)))
      return {};
    StateList states = activateValue(std::move(state), demux.getInput());
    return expand(std::move(states), [&](DerivationState current) {
      return activateValue(std::move(current), demux.getOutputs()[output]);
    });
  }

  StateList activateValue(DerivationState state, Value value) {
    if (!state.activeValues.insert(value).second)
      return {std::move(state)};

    StateList states;
    if (auto result = dyn_cast<OpResult>(value)) {
      Operation *producer = result.getOwner();
      if (isa<::fabric::DemuxOp>(producer)) {
        auto demux = cast<::fabric::DemuxOp>(producer);
        states = selectDemuxOutput(std::move(state), demux,
                                   result.getResultNumber());
      } else {
        states = activateNode(std::move(state), producer);
      }
    } else {
      states.push_back(std::move(state));
    }

    return expand(std::move(states), [&](DerivationState current) {
      return processUses(std::move(current), value);
    });
  }

  StateList processUses(DerivationState state, Value value) {
    StateList states{std::move(state)};
    for (OpOperand &use : value.getUses()) {
      Operation *consumer = use.getOwner();
      if (auto mux = dyn_cast<::fabric::MuxOp>(consumer)) {
        states = expand(std::move(states), [&](DerivationState current) {
          auto selected = current.muxSelections.find(consumer);
          if (selected == current.muxSelections.end() ||
              selected->second != use.getOperandNumber())
            return StateList{std::move(current)};
          if (!addEdge(current, value, use))
            return StateList{};
          return activateNode(std::move(current), mux.getOperation());
        });
        continue;
      }

      if (!isa<::fabric::OpOp, ::fabric::DemuxOp, ::fabric::YieldOp>(
              consumer)) {
        fail("encountered an unsupported FU-local consumer");
        return {};
      }
      states = expand(std::move(states), [&](DerivationState current) {
        if (!addEdge(current, value, use))
          return StateList{};
        if (isa<::fabric::YieldOp>(consumer))
          return StateList{std::move(current)};
        return activateNode(std::move(current), consumer);
      });
    }

    return expand(std::move(states), [&](DerivationState current) {
      auto source = sourceEndpoint(value);
      if (!source)
        return StateList{};
      bool hasConsumer = llvm::any_of(
          current.edges, [&](const FabricFuCapabilityTemplateEdge &edge) {
            return edge.source == *source;
          });

      if (constrained_)
        return hasConsumer ? StateList{std::move(current)} : StateList{};

      llvm::SmallVector<std::pair<::fabric::MuxOp, unsigned>, 4> optional;
      for (OpOperand &use : value.getUses()) {
        auto mux = dyn_cast<::fabric::MuxOp>(use.getOwner());
        if (!mux || current.muxSelections.count(mux.getOperation()))
          continue;
        optional.emplace_back(mux, use.getOperandNumber());
      }
      StateList result;
      selectOptionalMuxes(current, optional, 0, hasConsumer, result);
      return result;
    });
  }

  void selectOptionalMuxes(
      const DerivationState &state,
      llvm::ArrayRef<std::pair<::fabric::MuxOp, unsigned>> optional,
      std::size_t index, bool selectedAny, StateList &result) {
    if (index == optional.size()) {
      if (selectedAny)
        result.push_back(state);
      return;
    }

    selectOptionalMuxes(state, optional, index + 1, selectedAny, result);
    StateList selected =
        selectMuxInput(state, optional[index].first, optional[index].second);
    for (const DerivationState &candidate : selected)
      selectOptionalMuxes(candidate, optional, index + 1, true, result);
  }

  template <typename Function>
  StateList expand(StateList states, Function function) {
    StateList result;
    for (DerivationState &state : states)
      append(result, function(std::move(state)));
    return result;
  }

  static void append(StateList &destination, StateList source) {
    destination.insert(destination.end(),
                       std::make_move_iterator(source.begin()),
                       std::make_move_iterator(source.end()));
  }

  void fail(llvm::StringRef message) {
    if (!failed_) {
      failed_ = true;
      failure_ = message.str();
    }
  }

  ::fabric::FuOp fu_;
  FabricFuTemplateRef owner_;
  llvm::ArrayRef<Operation *> canonicalNodeOrder_;
  llvm::DenseMap<Operation *, FabricFuTemplateNodeRef> nodeRefs_;
  llvm::DenseSet<Operation *> declaredOperations_;
  llvm::DenseSet<Operation *> declaredSelectors_;
  bool constrained_ = false;
  bool failed_ = false;
  std::string failure_;
};

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  return bytes;
}

llvm::Expected<::fabric::FuCapabilityTemplateSelection>
projectSingletonDomain(const FabricFuCapabilityTemplateRecord &record) {
  ::fabric::FuCapabilityTemplateSelection selection;
  for (const FabricFuTemplateNodeRef &node : record.activeNodes) {
    if (node.node == FabricFuNodeKind::Op) {
      selection.activeOperationNodeOrdinals.push_back(node.ordinal);
      continue;
    }

    std::optional<FabricOrdinal> selectedPort;
    for (const FabricFuCapabilityTemplateEdge &edge : record.activeEdges) {
      const FabricFuCapabilityTemplateEndpointRef &endpoint =
          node.node == FabricFuNodeKind::Mux ? edge.destination : edge.source;
      auto *port = std::get_if<FabricFuNodePortRef>(&endpoint.payload);
      if (!port || port->node != node)
        continue;
      const FabricPortDirection expected = node.node == FabricFuNodeKind::Mux
                                               ? FabricPortDirection::Input
                                               : FabricPortDirection::Output;
      if (port->direction != expected)
        continue;
      if (selectedPort)
        return invalid("singleton projection found multiple selector choices");
      selectedPort = port->ordinal;
    }
    if (!selectedPort)
      return invalid("singleton projection found no selector choice");
    selection.routes.push_back({node.ordinal, *selectedPort});
  }
  return selection;
}

} // namespace

llvm::Expected<::fabric::FuCapabilityDomainRecord>
canonicalizeFabricFuCapabilityDomain(
    ::fabric::FuOp fu, llvm::ArrayRef<Operation *> canonicalNodeOrder,
    FabricFuCapabilityOrdinalSpace sourceOrdinalSpace) {
  ::fabric::FuCapabilityDomainAttr attribute = fu.getCapabilityTemplatesAttr();
  if (!attribute) {
    auto inferred =
        Deriver(fu, FabricFuTemplateRef(0), canonicalNodeOrder).run();
    if (!inferred)
      return inferred.takeError();
    if (inferred->size() != 1)
      return invalid(
          "a multi-template FU requires an explicit capability domain");
    auto selection = projectSingletonDomain(inferred->front());
    if (!selection)
      return selection.takeError();
    auto canonical = canonicalizeFabricFuCapabilityTemplate(
        fu, *selection, canonicalNodeOrder,
        FabricFuCapabilityOrdinalSpace::CanonicalDefinition);
    if (!canonical)
      return canonical.takeError();
    return ::fabric::FuCapabilityDomainRecord::create({std::move(*canonical)});
  }

  auto decoded = ::fabric::decodeFuCapabilityDomainRecord(
      unsignedBytes(attribute.getRecord()));
  if (!decoded)
    return decoded.takeError();

  std::vector<::fabric::FuCapabilityTemplateSelection> remapped;
  remapped.reserve(decoded->templates().size());
  for (const ::fabric::FuCapabilityTemplateSelection &source :
       decoded->templates()) {
    auto destination = canonicalizeFabricFuCapabilityTemplate(
        fu, source, canonicalNodeOrder, sourceOrdinalSpace);
    if (!destination)
      return destination.takeError();
    remapped.push_back(std::move(*destination));
  }
  return ::fabric::FuCapabilityDomainRecord::create(std::move(remapped));
}

llvm::Expected<::fabric::FuCapabilityTemplateSelection>
canonicalizeFabricFuCapabilityTemplate(
    ::fabric::FuOp fu, const ::fabric::FuCapabilityTemplateSelection &selection,
    llvm::ArrayRef<Operation *> canonicalNodeOrder,
    FabricFuCapabilityOrdinalSpace sourceOrdinalSpace) {
  llvm::SmallVector<Operation *, 16> physicalNodeOrder;
  for (Operation &operation : fu.getBody().front().without_terminator())
    if (isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(operation))
      physicalNodeOrder.push_back(&operation);
  if (physicalNodeOrder.size() != canonicalNodeOrder.size())
    return invalid("canonical FU node domain has the wrong size");

  llvm::DenseMap<Operation *, std::uint64_t> canonicalOrdinal;
  for (auto [ordinal, operation] : llvm::enumerate(canonicalNodeOrder)) {
    if (!operation || !llvm::is_contained(physicalNodeOrder, operation))
      return invalid("canonical FU node domain contains a foreign node");
    if (!canonicalOrdinal.try_emplace(operation, ordinal).second)
      return invalid("canonical FU node domain repeats a node");
  }
  const llvm::ArrayRef<Operation *> sourceNodeOrder =
      sourceOrdinalSpace == FabricFuCapabilityOrdinalSpace::AuthoringPhysical
          ? llvm::ArrayRef<Operation *>(physicalNodeOrder)
          : canonicalNodeOrder;

  ::fabric::FuCapabilityTemplateSelection destination;
  destination.activeOperationNodeOrdinals.reserve(
      selection.activeOperationNodeOrdinals.size());
  for (std::uint64_t ordinal : selection.activeOperationNodeOrdinals) {
    if (ordinal >= sourceNodeOrder.size())
      return invalid("capability domain names an unknown operation node");
    Operation *operation = sourceNodeOrder[ordinal];
    if (!isa<::fabric::OpOp>(operation))
      return invalid("capability domain activates a non-operation node");
    destination.activeOperationNodeOrdinals.push_back(
        canonicalOrdinal.lookup(operation));
  }
  destination.routes.reserve(selection.routes.size());
  for (const ::fabric::FuCapabilityRouteSelection &route : selection.routes) {
    if (route.selectorNodeOrdinal >= sourceNodeOrder.size())
      return invalid("capability domain names an unknown selector node");
    Operation *selector = sourceNodeOrder[route.selectorNodeOrdinal];
    if (auto mux = dyn_cast<::fabric::MuxOp>(selector)) {
      if (route.selectedPort >= mux.getInputs().size())
        return invalid("capability domain selects an unknown mux input");
    } else if (auto demux = dyn_cast<::fabric::DemuxOp>(selector)) {
      if (route.selectedPort >= demux.getOutputs().size())
        return invalid("capability domain selects an unknown demux output");
    } else {
      return invalid("capability domain route names a non-selector node");
    }
    destination.routes.push_back(
        {canonicalOrdinal.lookup(selector), route.selectedPort});
  }

  auto normalized =
      ::fabric::FuCapabilityDomainRecord::create({std::move(destination)});
  if (!normalized)
    return normalized.takeError();
  return normalized->templates().front();
}

llvm::Expected<FabricFuCapabilityTemplateRecord>
deriveFabricFuCapabilityTemplate(
    ::fabric::FuOp fu, FabricFuTemplateRef owner,
    llvm::ArrayRef<Operation *> canonicalNodeOrder,
    const ::fabric::FuCapabilityTemplateSelection &selection) {
  return Deriver(fu, owner, canonicalNodeOrder).run(selection);
}

llvm::Expected<std::vector<FabricFuCapabilityTemplateRecord>>
deriveFabricFuCapabilityTemplates(
    ::fabric::FuOp fu, FabricFuTemplateRef owner,
    llvm::ArrayRef<Operation *> canonicalNodeOrder) {
  auto domain = canonicalizeFabricFuCapabilityDomain(
      fu, canonicalNodeOrder,
      FabricFuCapabilityOrdinalSpace::CanonicalDefinition);
  if (!domain)
    return domain.takeError();

  std::vector<FabricFuCapabilityTemplateRecord> records;
  records.reserve(domain->templates().size());
  for (const ::fabric::FuCapabilityTemplateSelection &selection :
       domain->templates()) {
    auto record = deriveFabricFuCapabilityTemplate(
        fu, owner, canonicalNodeOrder, selection);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
  }
  return normalizeFabricFuCapabilityTemplateInventory(records);
}

} // namespace loom::fabric::detail
