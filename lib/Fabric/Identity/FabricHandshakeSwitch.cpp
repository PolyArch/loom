#include "Fabric/Identity/FabricHandshake.h"

#include "FabricHandshakeInternal.h"

#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/SwitchResourceContract.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {
namespace {

using Arc = std::pair<std::uint32_t, std::uint32_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_handshake_invalid: " + message);
}

detail::HandshakeFragmentSelector
traversalSelector(FabricPhysicalTraversalRef traversal) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::AnyTraversal;
  selector.traversalWitnesses.push_back(std::move(traversal));
  return selector;
}

detail::HandshakeFragmentSelector
anyTraversalSelector(llvm::ArrayRef<FabricPhysicalTraversalRef> traversals) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::AnyTraversal;
  selector.traversalWitnesses.assign(traversals.begin(), traversals.end());
  return selector;
}

detail::HandshakeFragmentSelector
switchActivationSelector(FabricSwitchHandshakeActivationKey activation,
                         llvm::ArrayRef<FabricPhysicalTraversalRef> traversals,
                         bool any) {
  detail::HandshakeFragmentSelector selector;
  selector.kind =
      any ? detail::HandshakeFragmentSelectorKind::AnySwitchActivationTraversal
          : detail::HandshakeFragmentSelectorKind::
                ExactSwitchActivationTraversal;
  selector.traversalWitnesses.assign(traversals.begin(), traversals.end());
  selector.switchActivation = activation;
  return selector;
}

} // namespace

struct FabricSwitchSelectedContentionScratch::Storage final {
  struct ForestEdge final {
    std::uint32_t inputVertex = 0;
    std::uint32_t outputVertex = 0;
    FabricOrdinal input = 0;
    FabricOrdinal output = 0;
  };

  std::vector<FabricSwitchSelectedCrosspoint> normalized;
  std::vector<FabricOrdinal> inputs;
  std::vector<FabricOrdinal> outputs;
  std::vector<std::uint32_t> parent;
  std::vector<std::uint8_t> rank;
  std::vector<ForestEdge> forest;
  std::vector<FabricOrdinal> componentMinimumInput;
  std::vector<std::uint32_t> componentRoots;
  std::vector<std::uint32_t> componentByRoot;
  std::vector<std::uint32_t> degree;
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint32_t> next;
  std::vector<std::uint32_t> adjacency;
  std::vector<std::uint8_t> visited;
  std::vector<std::uint32_t> worklist;
};

FabricSwitchSelectedContentionScratch::FabricSwitchSelectedContentionScratch()
    : storage_(std::make_unique<Storage>()) {}

FabricSwitchSelectedContentionScratch::
    ~FabricSwitchSelectedContentionScratch() = default;

void FabricSwitchSelectedContentionScratch::prepare(
    std::size_t crosspointCapacity) {
  Storage &storage = *storage_;
  const std::size_t vertexCapacity = crosspointCapacity * 2;
  storage.normalized.reserve(crosspointCapacity);
  storage.inputs.reserve(crosspointCapacity);
  storage.outputs.reserve(crosspointCapacity);
  storage.parent.reserve(vertexCapacity);
  storage.rank.reserve(vertexCapacity);
  storage.forest.reserve(crosspointCapacity);
  storage.componentMinimumInput.reserve(vertexCapacity);
  storage.componentRoots.reserve(vertexCapacity);
  storage.componentByRoot.reserve(vertexCapacity);
  storage.degree.reserve(vertexCapacity);
  storage.offsets.reserve(vertexCapacity + 1);
  storage.next.reserve(vertexCapacity);
  storage.adjacency.reserve(crosspointCapacity * 2);
  storage.visited.reserve(vertexCapacity);
  storage.worklist.reserve(vertexCapacity);
}

std::size_t
FabricSwitchSelectedContentionScratch::retainedStorageBytes() const {
  const Storage &storage = *storage_;
  return storage.normalized.capacity() *
             sizeof(FabricSwitchSelectedCrosspoint) +
         storage.inputs.capacity() * sizeof(FabricOrdinal) +
         storage.outputs.capacity() * sizeof(FabricOrdinal) +
         storage.parent.capacity() * sizeof(std::uint32_t) +
         storage.rank.capacity() * sizeof(std::uint8_t) +
         storage.forest.capacity() * sizeof(Storage::ForestEdge) +
         storage.componentMinimumInput.capacity() * sizeof(FabricOrdinal) +
         storage.componentRoots.capacity() * sizeof(std::uint32_t) +
         storage.componentByRoot.capacity() * sizeof(std::uint32_t) +
         storage.degree.capacity() * sizeof(std::uint32_t) +
         storage.offsets.capacity() * sizeof(std::uint32_t) +
         storage.next.capacity() * sizeof(std::uint32_t) +
         storage.adjacency.capacity() * sizeof(std::uint32_t) +
         storage.visited.capacity() * sizeof(std::uint8_t) +
         storage.worklist.capacity() * sizeof(std::uint32_t);
}

FabricSwitchSelectedContention FabricSwitchSelectedContention::derive(
    FabricSwitchOccurrenceRef occurrence,
    llvm::ArrayRef<FabricSwitchSelectedCrosspoint> crosspoints) {
  FabricSwitchSelectedContention result;
  FabricSwitchSelectedContentionScratch scratch;
  result.rebuild(occurrence, crosspoints, scratch);
  return result;
}

void FabricSwitchSelectedContention::rebuild(
    FabricSwitchOccurrenceRef occurrence,
    llvm::ArrayRef<FabricSwitchSelectedCrosspoint> crosspoints,
    FabricSwitchSelectedContentionScratch &scratch) {
  FabricSwitchSelectedContentionScratch::Storage &storage = *scratch.storage_;
  const auto crosspointLess = [](const FabricSwitchSelectedCrosspoint &lhs,
                                 const FabricSwitchSelectedCrosspoint &rhs) {
    return std::tie(lhs.input, lhs.output) < std::tie(rhs.input, rhs.output);
  };

  storage.normalized.assign(crosspoints.begin(), crosspoints.end());
  llvm::sort(storage.normalized, crosspointLess);
  storage.normalized.erase(
      std::unique(storage.normalized.begin(), storage.normalized.end(),
                  [](const FabricSwitchSelectedCrosspoint &lhs,
                     const FabricSwitchSelectedCrosspoint &rhs) {
                    return lhs.input == rhs.input && lhs.output == rhs.output;
                  }),
      storage.normalized.end());

  storage.inputs.clear();
  storage.outputs.clear();
  storage.inputs.reserve(storage.normalized.size());
  storage.outputs.reserve(storage.normalized.size());
  for (const FabricSwitchSelectedCrosspoint &crosspoint : storage.normalized) {
    storage.inputs.push_back(crosspoint.input);
    storage.outputs.push_back(crosspoint.output);
  }
  llvm::sort(storage.inputs);
  storage.inputs.erase(
      std::unique(storage.inputs.begin(), storage.inputs.end()),
      storage.inputs.end());
  llvm::sort(storage.outputs);
  storage.outputs.erase(
      std::unique(storage.outputs.begin(), storage.outputs.end()),
      storage.outputs.end());

  const std::size_t inputCount = storage.inputs.size();
  const std::size_t vertexCount = inputCount + storage.outputs.size();
  storage.parent.resize(vertexCount);
  storage.rank.assign(vertexCount, 0);
  for (std::uint32_t vertex = 0; vertex != vertexCount; ++vertex)
    storage.parent[vertex] = vertex;
  const auto findRoot = [&](std::uint32_t vertex) {
    std::uint32_t root = vertex;
    while (storage.parent[root] != root)
      root = storage.parent[root];
    while (storage.parent[vertex] != vertex) {
      const std::uint32_t next = storage.parent[vertex];
      storage.parent[vertex] = root;
      vertex = next;
    }
    return root;
  };

  storage.forest.clear();
  storage.forest.reserve(storage.normalized.size());
  for (const FabricSwitchSelectedCrosspoint &crosspoint : storage.normalized) {
    const auto input = llvm::lower_bound(storage.inputs, crosspoint.input);
    const auto output = llvm::lower_bound(storage.outputs, crosspoint.output);
    const std::uint32_t inputVertex =
        static_cast<std::uint32_t>(input - storage.inputs.begin());
    const std::uint32_t outputVertex = static_cast<std::uint32_t>(
        inputCount + (output - storage.outputs.begin()));
    std::uint32_t inputRoot = findRoot(inputVertex);
    std::uint32_t outputRoot = findRoot(outputVertex);
    if (inputRoot == outputRoot)
      continue;
    storage.forest.push_back(
        {inputVertex, outputVertex, crosspoint.input, crosspoint.output});
    if (storage.rank[inputRoot] < storage.rank[outputRoot])
      std::swap(inputRoot, outputRoot);
    storage.parent[outputRoot] = inputRoot;
    if (storage.rank[inputRoot] == storage.rank[outputRoot])
      ++storage.rank[inputRoot];
  }

  const FabricOrdinal noInput = std::numeric_limits<FabricOrdinal>::max();
  const std::uint32_t noComponent = std::numeric_limits<std::uint32_t>::max();
  storage.componentMinimumInput.assign(vertexCount, noInput);
  for (std::uint32_t input = 0; input != inputCount; ++input) {
    const std::uint32_t root = findRoot(input);
    storage.componentMinimumInput[root] =
        std::min(storage.componentMinimumInput[root], storage.inputs[input]);
  }
  storage.componentRoots.clear();
  for (std::uint32_t vertex = 0; vertex != vertexCount; ++vertex)
    if (storage.componentMinimumInput[vertex] != noInput)
      storage.componentRoots.push_back(vertex);
  llvm::sort(storage.componentRoots, [&](std::uint32_t lhs, std::uint32_t rhs) {
    return storage.componentMinimumInput[lhs] <
           storage.componentMinimumInput[rhs];
  });
  storage.componentByRoot.assign(vertexCount, noComponent);

  occurrence_ = occurrence;
  components_.clear();
  components_.reserve(storage.componentRoots.size());
  for (auto [component, root] : llvm::enumerate(storage.componentRoots)) {
    storage.componentByRoot[root] = static_cast<std::uint32_t>(component);
    components_.push_back({storage.componentMinimumInput[root], 0});
  }
  componentOfInput_.clear();
  componentOfInput_.reserve(inputCount);
  for (std::uint32_t input = 0; input != inputCount; ++input) {
    const std::uint32_t component = storage.componentByRoot[findRoot(input)];
    ++components_[component].inputCount;
    componentOfInput_.emplace_back(storage.inputs[input], component);
  }
  componentOfOutput_.clear();
  componentOfOutput_.reserve(storage.outputs.size());
  for (std::uint32_t output = 0; output != storage.outputs.size(); ++output) {
    const std::uint32_t vertex =
        static_cast<std::uint32_t>(inputCount + output);
    componentOfOutput_.emplace_back(storage.outputs[output],
                                    storage.componentByRoot[findRoot(vertex)]);
  }

  storage.degree.assign(vertexCount, 0);
  for (const FabricSwitchSelectedContentionScratch::Storage::ForestEdge &edge :
       storage.forest) {
    ++storage.degree[edge.inputVertex];
    ++storage.degree[edge.outputVertex];
  }
  storage.offsets.resize(vertexCount + 1);
  storage.offsets[0] = 0;
  for (std::size_t vertex = 0; vertex != vertexCount; ++vertex)
    storage.offsets[vertex + 1] =
        storage.offsets[vertex] + storage.degree[vertex];
  storage.next.assign(storage.offsets.begin(), storage.offsets.end() - 1);
  storage.adjacency.resize(storage.forest.size() * 2);
  for (auto [edgeOrdinal, edge] : llvm::enumerate(storage.forest)) {
    storage.adjacency[storage.next[edge.inputVertex]++] =
        static_cast<std::uint32_t>(edgeOrdinal);
    storage.adjacency[storage.next[edge.outputVertex]++] =
        static_cast<std::uint32_t>(edgeOrdinal);
  }

  treeEdges_.clear();
  treeEdges_.reserve(storage.forest.size());
  storage.visited.assign(vertexCount, 0);
  for (const Component &component : components_) {
    const auto root = llvm::lower_bound(storage.inputs, component.root);
    const std::uint32_t rootVertex =
        static_cast<std::uint32_t>(root - storage.inputs.begin());
    storage.worklist.clear();
    storage.worklist.push_back(rootVertex);
    storage.visited[rootVertex] = 1;
    for (std::size_t position = 0; position != storage.worklist.size();
         ++position) {
      const std::uint32_t vertex = storage.worklist[position];
      for (std::uint32_t adjacency = storage.offsets[vertex];
           adjacency != storage.offsets[vertex + 1]; ++adjacency) {
        const FabricSwitchSelectedContentionScratch::Storage::ForestEdge &edge =
            storage.forest[storage.adjacency[adjacency]];
        const std::uint32_t next =
            vertex == edge.inputVertex ? edge.outputVertex : edge.inputVertex;
        if (storage.visited[next])
          continue;
        storage.visited[next] = 1;
        storage.worklist.push_back(next);
        treeEdges_.push_back(
            {edge.input, edge.output, vertex == edge.inputVertex});
      }
    }
  }
  llvm::sort(treeEdges_);

  selectedCrosspoints_.assign(storage.normalized.begin(),
                              storage.normalized.end());
  selectedInputsByOutput_.clear();
  selectedInputsByOutput_.reserve(storage.normalized.size());
  for (const FabricSwitchSelectedCrosspoint &crosspoint : storage.normalized)
    selectedInputsByOutput_.emplace_back(crosspoint.output, crosspoint.input);
  llvm::sort(selectedInputsByOutput_);
}

std::size_t FabricSwitchSelectedContention::retainedStorageBytes() const {
  std::size_t bytes =
      components_.capacity() * sizeof(Component) +
      componentOfInput_.capacity() *
          sizeof(std::pair<FabricOrdinal, std::size_t>) +
      componentOfOutput_.capacity() *
          sizeof(std::pair<FabricOrdinal, std::size_t>) +
      treeEdges_.capacity() * sizeof(TreeEdge) +
      selectedCrosspoints_.capacity() * sizeof(FabricSwitchSelectedCrosspoint) +
      selectedInputsByOutput_.capacity() *
          sizeof(std::pair<FabricOrdinal, FabricOrdinal>);
  return bytes;
}

void FabricSwitchSelectedContention::prepare(std::size_t crosspointCapacity) {
  components_.reserve(crosspointCapacity);
  componentOfInput_.reserve(crosspointCapacity);
  componentOfOutput_.reserve(crosspointCapacity);
  treeEdges_.reserve(crosspointCapacity);
  selectedCrosspoints_.reserve(crosspointCapacity);
  selectedInputsByOutput_.reserve(crosspointCapacity);
}

const FabricSwitchSelectedContention::Component *
FabricSwitchSelectedContention::componentOf(FabricOrdinal input) const {
  const auto found =
      llvm::lower_bound(componentOfInput_, input,
                        [](const std::pair<FabricOrdinal, std::size_t> &entry,
                           FabricOrdinal key) { return entry.first < key; });
  if (found == componentOfInput_.end() || found->first != input)
    return nullptr;
  return &components_[found->second];
}

const FabricSwitchSelectedContention::Component *
FabricSwitchSelectedContention::componentOfOutput(FabricOrdinal output) const {
  const auto found =
      llvm::lower_bound(componentOfOutput_, output,
                        [](const std::pair<FabricOrdinal, std::size_t> &entry,
                           FabricOrdinal key) { return entry.first < key; });
  if (found == componentOfOutput_.end() || found->first != output)
    return nullptr;
  return &components_[found->second];
}

bool FabricSwitchSelectedContention::contended(FabricOrdinal input) const {
  const Component *component = componentOf(input);
  return component && component->inputCount > 1;
}

bool FabricSwitchSelectedContention::outputContended(
    FabricOrdinal output) const {
  const Component *component = componentOfOutput(output);
  return component && component->inputCount > 1;
}

bool FabricSwitchSelectedContention::selected(FabricOrdinal input,
                                              FabricOrdinal output) const {
  return std::binary_search(selectedCrosspoints_.begin(),
                            selectedCrosspoints_.end(),
                            FabricSwitchSelectedCrosspoint{input, output},
                            [](const FabricSwitchSelectedCrosspoint &lhs,
                               const FabricSwitchSelectedCrosspoint &rhs) {
                              return std::tie(lhs.input, lhs.output) <
                                     std::tie(rhs.input, rhs.output);
                            });
}

bool FabricSwitchSelectedContention::directlyContended(
    FabricOrdinal output) const {
  const auto begin =
      llvm::lower_bound(selectedInputsByOutput_, output,
                        [](const std::pair<FabricOrdinal, FabricOrdinal> &entry,
                           FabricOrdinal key) { return entry.first < key; });
  if (begin == selectedInputsByOutput_.end() || begin->first != output)
    return false;
  return std::next(begin) != selectedInputsByOutput_.end() &&
         std::next(begin)->first == output;
}

bool FabricSwitchSelectedContention::activates(
    const FabricSwitchHandshakeContentionRelation &relation) const {
  if (relation.occurrence != occurrence_)
    return false;
  using Kind = FabricSwitchHandshakeContentionRelationKind;
  switch (relation.relation) {
  case Kind::ReadyInputValid:
  case Kind::InputReady:
  case Kind::FixedInputValid:
    return contended(relation.input);
  case Kind::ReadyTreeInputParent:
  case Kind::ReadyTreeOutputParent: {
    const TreeEdge key{relation.input, relation.output,
                       relation.relation == Kind::ReadyTreeInputParent};
    return std::binary_search(treeEdges_.begin(), treeEdges_.end(), key) &&
           contended(relation.input);
  }
  case Kind::ReadyRootBridge: {
    const Component *component = componentOf(relation.input);
    return component && component->inputCount > 1 &&
           component->root == relation.input;
  }
  case Kind::RoundRobinOutputValid:
    return outputContended(relation.output);
  case Kind::FixedSelectedCrosspoint:
    return selected(relation.input, relation.output) &&
           contended(relation.input);
  case Kind::FixedUnselectedCrosspoint:
    return !selected(relation.input, relation.output) &&
           directlyContended(relation.output);
  }
  llvm_unreachable("closed switch contention relation domain");
}

llvm::Expected<HandshakeOwnerModel> detail::compileSwitchHandshakeModel(
    const FabricArtifactView &view, FabricSwitchOccurrenceRef owner,
    llvm::ArrayRef<const FabricPhysicalTraversalView *> traversals) {
  struct Row final {
    FabricOrdinal output = 0;
    FabricPhysicalTraversalRef reference;
    FabricTransportEndpointRef source;
    FabricTransportEndpointRef destination;
  };
  std::map<FabricOrdinal, std::vector<Row>> byInput;
  for (const FabricPhysicalTraversalView *traversalPointer : traversals) {
    const FabricPhysicalTraversalView &traversal = *traversalPointer;
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::SwitchTraversal)
      continue;
    const auto &payload =
        std::get<FabricSwitchTraversalPayload>(traversal.reference.payload);
    if (payload.owner != owner)
      continue;
    if (traversal.sources.size() != 1 || traversal.destinations.size() != 1)
      return invalid("switch traversal has invalid endpoint cardinality");
    byInput[payload.input].push_back({payload.output, traversal.reference,
                                      traversal.sources.front(),
                                      traversal.destinations.front()});
  }

  const auto schedule = view.switchSchedule(owner);
  if (!schedule)
    return invalid("switch occurrence has no scheduling contract");
  const std::uint64_t residentRows = *schedule == ::fabric::Schedule::Temporal
                                         ? view.switchRouteTableSize(owner)
                                         : 1;
  if (residentRows == 0)
    return invalid("switch occurrence has no configurable route row");
  std::vector<HandshakeOwnerModel> rowShapes;
  rowShapes.reserve(byInput.size());
  for (auto &entry : byInput) {
    const FabricOrdinal input = entry.first;
    std::vector<Row> &rows = entry.second;
    llvm::sort(rows, [](const Row &lhs, const Row &rhs) {
      return lhs.output < rhs.output;
    });
    HandshakeOwnerModelBuilder builder(
        FabricHandshakeOwner::switchResource(owner));
    std::vector<FabricPhysicalTraversalRef> witnesses;
    witnesses.reserve(rows.size());
    for (const Row &row : rows)
      witnesses.push_back(row.reference);

    const FabricSwitchHandshakeActivationKey activation{owner, 0, input};
    std::vector<std::uint32_t> prefix(rows.size() + 1);
    std::vector<std::uint32_t> suffix(rows.size() + 1);
    for (std::size_t position = 0; position <= rows.size(); ++position) {
      prefix[position] =
          builder.junction(handshakeJunctionKey(0, input, position));
      suffix[position] =
          builder.junction(handshakeJunctionKey(1, input, position));
    }

    std::vector<Arc> base;
    base.reserve(rows.size() * 2 + 1);
    for (std::size_t position = 0; position < rows.size(); ++position) {
      base.emplace_back(prefix[position], prefix[position + 1]);
      base.emplace_back(suffix[position + 1], suffix[position]);
    }
    const std::uint32_t inputReady = builder.boundarySignal(
        {rows.front().source, HandshakeSignalKind::Ready});
    base.emplace_back(prefix.back(), inputReady);
    builder.addFragment(
        *schedule == ::fabric::Schedule::Temporal
            ? switchActivationSelector(activation, witnesses, true)
            : anyTraversalSelector(witnesses),
        std::move(base));

    for (std::size_t position = 0; position < rows.size(); ++position) {
      const Row &row = rows[position];
      const std::uint32_t inputValid =
          builder.boundarySignal({row.source, HandshakeSignalKind::Valid});
      const std::uint32_t outputValid =
          builder.boundarySignal({row.destination, HandshakeSignalKind::Valid});
      const std::uint32_t outputReady =
          builder.boundarySignal({row.destination, HandshakeSignalKind::Ready});
      const FabricPhysicalTraversalRef selected[] = {row.reference};
      builder.addFragment(
          *schedule == ::fabric::Schedule::Temporal
              ? switchActivationSelector(activation, selected, false)
              : traversalSelector(row.reference),
          {{outputReady, prefix[position + 1]},
           {outputReady, suffix[position]},
           {inputValid, outputValid},
           {prefix[position], outputValid},
           {suffix[position + 1], outputValid}});
    }
    auto shape = builder.finish();
    if (!shape)
      return shape.takeError();
    rowShapes.push_back(std::move(*shape));
  }

  std::optional<HandshakeOwnerModel> contentionShape;
  if (*schedule == ::fabric::Schedule::Temporal) {
    std::map<FabricOrdinal, std::vector<FabricOrdinal>> inputsByOutput;
    std::map<FabricOrdinal, FabricTransportEndpointRef> inputEndpoints;
    std::map<FabricOrdinal, FabricTransportEndpointRef> outputEndpoints;
    for (const auto &entry : byInput) {
      const FabricOrdinal input = entry.first;
      const std::vector<Row> &rows = entry.second;
      if (rows.empty())
        continue;
      inputEndpoints.emplace(input, rows.front().source);
      for (const Row &row : rows) {
        if (row.source != rows.front().source)
          return invalid("switch input has inconsistent endpoints");
        const auto insertion =
            outputEndpoints.try_emplace(row.output, row.destination);
        if (!insertion.second && insertion.first->second != row.destination)
          return invalid("switch output has inconsistent endpoints");
        inputsByOutput[row.output].push_back(input);
      }
    }
    if (inputEndpoints.empty() || outputEndpoints.empty() ||
        inputEndpoints.rbegin()->first >=
            std::numeric_limits<std::uint32_t>::max() ||
        outputEndpoints.rbegin()->first >=
            std::numeric_limits<std::uint32_t>::max())
      return invalid("switch endpoint domain is empty or exceeds u32");
    const std::uint32_t inputCount =
        static_cast<std::uint32_t>(inputEndpoints.rbegin()->first + 1);
    const std::uint32_t outputCount =
        static_cast<std::uint32_t>(outputEndpoints.rbegin()->first + 1);
    std::vector<std::vector<std::uint32_t>> sourcesByOutput(outputCount);
    for (auto &entry : inputsByOutput) {
      std::vector<FabricOrdinal> &inputs = entry.second;
      llvm::sort(inputs);
      inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
      for (FabricOrdinal input : inputs)
        sourcesByOutput[entry.first].push_back(
            static_cast<std::uint32_t>(input));
    }
    const ::fabric::ResourceContract *contract =
        view.resourceContract(FabricInventoryOwnerRef::of(owner));
    if (!contract)
      return invalid("Temporal switch has no ResourceContract");
    auto arbitrationComponents = ::fabric::deriveSwitchArbitrationComponents(
        *schedule, inputCount, outputCount, sourcesByOutput, *contract);
    if (!arbitrationComponents)
      return arbitrationComponents.takeError();
    const bool canContend =
        llvm::any_of(*arbitrationComponents, [](const auto &component) {
          return component.inputs.size() > 1;
        });
    if (canContend) {
      using Relation = FabricSwitchHandshakeContentionRelationKind;
      const bool roundRobin =
          llvm::any_of(*arbitrationComponents, [](const auto &component) {
            return component.roundRobinResetPosition.has_value();
          });
      std::map<FabricOrdinal, std::size_t> priorityRank;
      if (!roundRobin)
        for (const auto &component : *arbitrationComponents)
          for (auto [rank, input] : llvm::enumerate(component.requesterOrder))
            if (!priorityRank.emplace(input, rank).second)
              return invalid("FixedPriority repeats a switch requester");

      HandshakeOwnerModelBuilder builder(
          FabricHandshakeOwner::switchResource(owner));
      std::map<FabricOrdinal, std::uint32_t> upInputs;
      std::map<FabricOrdinal, std::uint32_t> downInputs;
      std::map<FabricOrdinal, std::uint32_t> upOutputs;
      std::map<FabricOrdinal, std::uint32_t> downOutputs;
      for (const auto &entry : inputEndpoints) {
        const FabricOrdinal input = entry.first;
        upInputs.emplace(input,
                         builder.junction(handshakeJunctionKey(0, input, 0)));
        downInputs.emplace(input,
                           builder.junction(handshakeJunctionKey(1, input, 0)));
      }
      for (const auto &entry : outputEndpoints) {
        const FabricOrdinal output = entry.first;
        upOutputs.emplace(output,
                          builder.junction(handshakeJunctionKey(2, output, 0)));
        downOutputs.emplace(
            output, builder.junction(handshakeJunctionKey(3, output, 0)));
      }

      for (const auto &entry : inputEndpoints) {
        const FabricOrdinal input = entry.first;
        const FabricTransportEndpointRef endpoint = entry.second;
        builder.addFragment(
            switchContentionSelector(
                {owner, Relation::ReadyInputValid, input, 0}),
            {{builder.boundarySignal({endpoint, HandshakeSignalKind::Valid}),
              upInputs.at(input)}});
        builder.addFragment(switchContentionSelector(
                                {owner, Relation::ReadyRootBridge, input, 0}),
                            {{upInputs.at(input), downInputs.at(input)}});
        builder.addFragment(
            switchContentionSelector({owner, Relation::InputReady, input, 0}),
            {{downInputs.at(input),
              builder.boundarySignal({endpoint, HandshakeSignalKind::Ready})}});
      }
      if (roundRobin)
        for (const auto &entry : outputEndpoints) {
          const FabricOrdinal output = entry.first;
          builder.addFragment(
              switchContentionSelector(
                  {owner, Relation::RoundRobinOutputValid, 0, output}),
              {{downOutputs.at(output),
                builder.boundarySignal(
                    {entry.second, HandshakeSignalKind::Valid})}});
        }
      for (const auto &entry : byInput) {
        const FabricOrdinal input = entry.first;
        for (const Row &row : entry.second) {
          builder.addFragment(
              switchContentionSelector(
                  {owner, Relation::ReadyTreeInputParent, input, row.output}),
              {{upOutputs.at(row.output), upInputs.at(input)},
               {downInputs.at(input), downOutputs.at(row.output)}});
          builder.addFragment(
              switchContentionSelector(
                  {owner, Relation::ReadyTreeOutputParent, input, row.output}),
              {{upInputs.at(input), upOutputs.at(row.output)},
               {downOutputs.at(row.output), downInputs.at(input)}});
        }
      }

      if (!roundRobin) {
        if (priorityRank.size() != inputEndpoints.size())
          return invalid("FixedPriority omits a switch requester");

        std::map<FabricOrdinal, std::uint32_t> selections;
        for (const auto &entry : inputEndpoints) {
          const FabricOrdinal input = entry.first;
          if (priorityRank.count(input) == 0)
            return invalid("FixedPriority names a foreign switch requester");
          const std::uint32_t selection =
              builder.junction(handshakeJunctionKey(4, input, 0));
          selections.emplace(input, selection);
          builder.addFragment(switchContentionSelector(
                                  {owner, Relation::FixedInputValid, input, 0}),
                              {{builder.boundarySignal(
                                    {entry.second, HandshakeSignalKind::Valid}),
                                selection}});
        }

        for (const auto &entry : inputsByOutput) {
          const FabricOrdinal output = entry.first;
          std::vector<FabricOrdinal> requesters = entry.second;
          llvm::sort(requesters, [&](FabricOrdinal lhs, FabricOrdinal rhs) {
            return priorityRank.at(lhs) < priorityRank.at(rhs);
          });
          std::vector<std::uint32_t> prefix(requesters.size() + 1);
          for (std::size_t position = 0; position != prefix.size(); ++position)
            prefix[position] =
                builder.junction(handshakeJunctionKey(5, output, position));
          const std::uint32_t outputValid = builder.boundarySignal(
              {outputEndpoints.at(output), HandshakeSignalKind::Valid});
          for (auto [position, input] : llvm::enumerate(requesters)) {
            builder.addFragment(
                switchContentionSelector(
                    {owner, Relation::FixedSelectedCrosspoint, input, output}),
                {{prefix[position], selections.at(input)},
                 {selections.at(input), prefix[position + 1]},
                 {selections.at(input), outputValid}});
            builder.addFragment(switchContentionSelector(
                                    {owner, Relation::FixedUnselectedCrosspoint,
                                     input, output}),
                                {{prefix[position], prefix[position + 1]}});
          }
        }
      }
      auto shape = builder.finish();
      if (!shape)
        return shape.takeError();
      contentionShape = std::move(*shape);
    }
  }
  return HandshakeOwnerModelFactory::instantiateSwitchRows(
      owner, rowShapes, residentRows, *schedule == ::fabric::Schedule::Temporal,
      contentionShape ? &*contentionShape : nullptr);
}

} // namespace loom::fabric
