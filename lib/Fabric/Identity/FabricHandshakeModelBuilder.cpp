#include "FabricHandshakeInternal.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_handshake_invalid: " + message);
}

template <typename T>
llvm::Expected<std::uint32_t> checkedSize(const T &container,
                                          llvm::StringRef description) {
  if (container.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid(description + " exceeds the owner-model index domain");
  return static_cast<std::uint32_t>(container.size());
}

std::uint64_t packedArc(std::uint32_t source, std::uint32_t destination) {
  return (static_cast<std::uint64_t>(source) << 32) | destination;
}

llvm::Expected<std::shared_ptr<const HandshakeOwnerModelStorage>>
buildStorage(FabricHandshakeOwner owner,
             std::vector<HandshakeOwnerModelInstance> instances) {
  auto storage = std::make_shared<HandshakeOwnerModelStorage>(std::move(owner));
  storage->instances = std::move(instances);
  storage->layouts.reserve(storage->instances.size());
  std::vector<const HandshakeOwnerModelInstance *> projectionShapes;
  for (const HandshakeOwnerModelInstance &instance : storage->instances) {
    if (!instance.structure || !instance.nodeBindings ||
        !instance.traversalWitnessBindings || !instance.selectors)
      return invalid("handshake structural instance has a null binding");
    if (instance.nodeBindings->size() != instance.structure->nodeKinds.size() ||
        instance.selectors->size() != instance.structure->fragments.size())
      return invalid("handshake structural instance has the wrong shape");
    for (auto [ordinal, node] : llvm::enumerate(*instance.nodeBindings))
      if (node.kind != instance.structure->nodeKinds[ordinal] ||
          (node.kind == HandshakeOwnerNodeKind::BoundarySignal) !=
              node.boundarySignal.has_value())
        return invalid("handshake structural node binding is inconsistent");

    if (instance.projectionShapeOrdinal > projectionShapes.size())
      return invalid("handshake projection shapes are not canonical");
    if (instance.projectionShapeOrdinal == projectionShapes.size()) {
      projectionShapes.push_back(&instance);
    } else {
      const HandshakeOwnerModelInstance &shape =
          *projectionShapes[instance.projectionShapeOrdinal];
      if (instance.structure != shape.structure ||
          instance.nodeBindings != shape.nodeBindings ||
          instance.traversalWitnessBindings != shape.traversalWitnessBindings ||
          instance.selectors != shape.selectors)
        return invalid("handshake projection shape binding is inconsistent");
    }

    storage->layouts.push_back(
        {storage->nodeCount, storage->arcCount, storage->fragmentCount,
         storage->contributionCount, storage->witnessCount});
    const auto add = [&](std::uint32_t &value, std::size_t count,
                         llvm::StringRef description) -> llvm::Error {
      if (count > std::numeric_limits<std::uint32_t>::max() - value)
        return invalid(description + " exceeds the owner-model index domain");
      value += static_cast<std::uint32_t>(count);
      return llvm::Error::success();
    };
    if (llvm::Error error =
            add(storage->nodeCount, instance.structure->nodeKinds.size(),
                "handshake node count"))
      return std::move(error);
    if (llvm::Error error =
            add(storage->arcCount, instance.structure->arcs.size(),
                "handshake arc count"))
      return std::move(error);
    if (llvm::Error error =
            add(storage->fragmentCount, instance.structure->fragments.size(),
                "handshake fragment count"))
      return std::move(error);
    if (llvm::Error error =
            add(storage->contributionCount,
                instance.structure->fragmentContributionOrdinals.size(),
                "handshake contribution count"))
      return std::move(error);
    if (llvm::Error error = add(storage->witnessCount,
                                instance.traversalWitnessBindings->size(),
                                "handshake witness count"))
      return std::move(error);
  }
  return std::shared_ptr<const HandshakeOwnerModelStorage>(std::move(storage));
}

template <typename Offset>
std::size_t locateInstance(const HandshakeOwnerModelStorage &storage,
                           std::uint32_t ordinal, Offset offset) {
  std::size_t lower = 0;
  std::size_t upper = storage.layouts.size();
  while (lower + 1 < upper) {
    const std::size_t middle = lower + (upper - lower) / 2;
    if (offset(storage.layouts[middle]) <= ordinal)
      lower = middle;
    else
      upper = middle;
  }
  return lower;
}

std::vector<HandshakeOwnerNode>
rebindNodes(const std::vector<HandshakeOwnerNode> &nodes,
            const FabricTransportEndpointOwnerRef &source,
            const FabricTransportEndpointOwnerRef &destination) {
  std::vector<HandshakeOwnerNode> rebound = nodes;
  for (HandshakeOwnerNode &node : rebound)
    if (node.boundarySignal && node.boundarySignal->endpoint.owner == source)
      node.boundarySignal->endpoint.owner = destination;
  return rebound;
}

} // namespace

std::vector<std::uint8_t> handshakeSignalKey(const HandshakeSignalRef &signal) {
  std::vector<std::uint8_t> key = canonicalFabricBytes(signal.endpoint);
  key.insert(key.begin(), static_cast<std::uint8_t>(signal.signal));
  key.insert(key.begin(), 0);
  return key;
}

HandshakeOwnerModelBuilder::HandshakeOwnerModelBuilder(
    FabricHandshakeOwner owner)
    : owner_(std::move(owner)) {
  boundaryNodes_.reserve(32);
  junctionNodes_.reserve(64);
}

std::uint32_t
HandshakeOwnerModelBuilder::boundarySignal(HandshakeSignalRef signal) {
  std::vector<std::uint8_t> key = canonicalFabricBytes(signal.endpoint);
  key.push_back(static_cast<std::uint8_t>(signal.signal));
  auto found = boundaryNodes_.find(key);
  if (found != boundaryNodes_.end())
    return found->second;
  const std::uint32_t ordinal = static_cast<std::uint32_t>(nodes_.size());
  nodes_.push_back({HandshakeOwnerNodeKind::BoundarySignal, std::move(signal)});
  boundaryNodes_.emplace(std::move(key), ordinal);
  return ordinal;
}

std::uint32_t
HandshakeOwnerModelBuilder::junction(std::vector<std::uint8_t> ownerLocalKey) {
  const auto found = junctionNodes_.find(ownerLocalKey);
  if (found != junctionNodes_.end())
    return found->second;
  const std::uint32_t ordinal = static_cast<std::uint32_t>(nodes_.size());
  nodes_.push_back({HandshakeOwnerNodeKind::OwnerLocalJunction, std::nullopt});
  junctionNodes_.emplace(std::move(ownerLocalKey), ordinal);
  return ordinal;
}

void HandshakeOwnerModelBuilder::addFragment(
    HandshakeFragmentSelector selector,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> arcs) {
  pending_.push_back({std::move(selector), std::move(arcs)});
}

llvm::Expected<HandshakeOwnerModel> HandshakeOwnerModelBuilder::finish() {
  if (auto count = checkedSize(nodes_, "handshake node count"); !count)
    return count.takeError();

  auto structure = std::make_shared<HandshakeStructuralTemplate>();
  structure->nodeKinds.reserve(nodes_.size());
  for (const HandshakeOwnerNode &node : nodes_)
    structure->nodeKinds.push_back(node.kind);

  std::vector<std::uint64_t> uniqueArcs;
  for (const PendingFragment &fragment : pending_)
    for (const auto &[source, destination] : fragment.arcs)
      uniqueArcs.push_back(packedArc(source, destination));
  llvm::sort(uniqueArcs);
  uniqueArcs.erase(std::unique(uniqueArcs.begin(), uniqueArcs.end()),
                   uniqueArcs.end());
  if (uniqueArcs.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("handshake arc count exceeds the owner-model index domain");

  std::unordered_map<std::uint64_t, std::uint32_t> arcOrdinals;
  arcOrdinals.reserve(uniqueArcs.size());
  structure->arcs.reserve(uniqueArcs.size());
  for (const std::uint64_t arc : uniqueArcs) {
    const std::uint32_t ordinal =
        static_cast<std::uint32_t>(structure->arcs.size());
    arcOrdinals.emplace(arc, ordinal);
    structure->arcs.push_back({static_cast<std::uint32_t>(arc >> 32),
                               static_cast<std::uint32_t>(arc)});
  }

  structure->fragments.reserve(pending_.size());
  std::vector<HandshakeFragmentSelector> selectors;
  selectors.reserve(pending_.size());
  std::vector<FabricPhysicalTraversalRef> traversalWitnesses;
  for (PendingFragment &pending : pending_) {
    std::vector<std::uint32_t> contributions;
    contributions.reserve(pending.arcs.size());
    for (const auto &[source, destination] : pending.arcs)
      contributions.push_back(arcOrdinals.at(packedArc(source, destination)));
    llvm::sort(contributions);
    contributions.erase(std::unique(contributions.begin(), contributions.end()),
                        contributions.end());
    auto offset = checkedSize(structure->fragmentContributionOrdinals,
                              "handshake contribution offset");
    auto count = checkedSize(contributions, "handshake contribution count");
    if (!offset)
      return offset.takeError();
    if (!count)
      return count.takeError();

    HandshakeActivationKind activationKind =
        HandshakeActivationKind::ExactOwnerSelection;
    switch (pending.selector.kind) {
    case HandshakeFragmentSelectorKind::Always:
      activationKind = HandshakeActivationKind::Always;
      break;
    case HandshakeFragmentSelectorKind::AnyTraversal:
      activationKind = HandshakeActivationKind::AnyTraversal;
      break;
    case HandshakeFragmentSelectorKind::AllTraversals:
      activationKind = HandshakeActivationKind::AllTraversals;
      break;
    case HandshakeFragmentSelectorKind::AnySwitchActivationTraversal:
      activationKind = HandshakeActivationKind::AnySwitchActivationTraversal;
      break;
    case HandshakeFragmentSelectorKind::ExactSwitchActivationTraversal:
      activationKind = HandshakeActivationKind::ExactSwitchActivationTraversal;
      break;
    case HandshakeFragmentSelectorKind::FuCapability:
    case HandshakeFragmentSelectorKind::FuOperationCase:
    case HandshakeFragmentSelectorKind::FuOperationInputActive:
    case HandshakeFragmentSelectorKind::FuOperationResultActive:
    case HandshakeFragmentSelectorKind::MemoryOperationPlan:
      break;
    }

    std::vector<FabricPhysicalTraversalRef> witnesses;
    if (activationKind == HandshakeActivationKind::AnyTraversal ||
        activationKind == HandshakeActivationKind::AllTraversals ||
        activationKind ==
            HandshakeActivationKind::AnySwitchActivationTraversal ||
        activationKind ==
            HandshakeActivationKind::ExactSwitchActivationTraversal) {
      struct KeyedTraversal final {
        std::vector<std::uint8_t> key;
        FabricPhysicalTraversalRef traversal;
      };
      std::vector<KeyedTraversal> keyed;
      keyed.reserve(pending.selector.traversalWitnesses.size());
      for (const FabricPhysicalTraversalRef &traversal :
           pending.selector.traversalWitnesses)
        keyed.push_back({canonicalFabricBytes(traversal), traversal});
      llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
        return lhs.key < rhs.key;
      });
      keyed.erase(std::unique(keyed.begin(), keyed.end(),
                              [](const auto &lhs, const auto &rhs) {
                                return lhs.traversal == rhs.traversal;
                              }),
                  keyed.end());
      witnesses.reserve(keyed.size());
      for (KeyedTraversal &entry : keyed)
        witnesses.push_back(std::move(entry.traversal));
      if (witnesses.empty())
        return invalid("traversal-selected fragment has no witness");
    }
    auto witnessOffset =
        checkedSize(traversalWitnesses, "handshake witness offset");
    auto witnessCount = checkedSize(witnesses, "handshake witness count");
    if (!witnessOffset)
      return witnessOffset.takeError();
    if (!witnessCount)
      return witnessCount.takeError();
    if ((activationKind ==
             HandshakeActivationKind::AnySwitchActivationTraversal ||
         activationKind ==
             HandshakeActivationKind::ExactSwitchActivationTraversal) !=
        pending.selector.switchActivation.has_value())
      return invalid("switch activation fragment has no exact key");
    structure->fragments.push_back(
        {*offset, *count, activationKind, *witnessOffset, *witnessCount});
    structure->fragmentContributionOrdinals.insert(
        structure->fragmentContributionOrdinals.end(), contributions.begin(),
        contributions.end());
    traversalWitnesses.insert(traversalWitnesses.end(), witnesses.begin(),
                              witnesses.end());
    selectors.push_back(std::move(pending.selector));
  }
  HandshakeOwnerModelInstance instance{
      std::shared_ptr<const HandshakeStructuralTemplate>(std::move(structure)),
      std::make_shared<const std::vector<HandshakeOwnerNode>>(
          std::move(nodes_)),
      std::make_shared<const std::vector<FabricPhysicalTraversalRef>>(
          std::move(traversalWitnesses)),
      std::make_shared<const std::vector<HandshakeFragmentSelector>>(
          std::move(selectors)),
      std::nullopt};
  auto storage = buildStorage(std::move(owner_), {std::move(instance)});
  if (!storage)
    return storage.takeError();
  return HandshakeOwnerModel(std::move(*storage));
}

} // namespace loom::fabric::detail

namespace loom::fabric {

const FabricHandshakeOwner &HandshakeOwnerModel::owner() const {
  assert(storage_);
  return storage_->owner;
}

std::uint32_t HandshakeOwnerModel::nodeCount() const {
  return storage_ ? storage_->nodeCount : 0;
}

HandshakeOwnerNode HandshakeOwnerModel::node(std::uint32_t ordinal) const {
  assert(storage_ && ordinal < storage_->nodeCount);
  const std::size_t instance = detail::locateInstance(
      *storage_, ordinal,
      [](const detail::HandshakeOwnerModelInstanceLayout &layout) {
        return layout.nodeOffset;
      });
  const auto &layout = storage_->layouts[instance];
  return (
      *storage_->instances[instance].nodeBindings)[ordinal - layout.nodeOffset];
}

std::uint32_t HandshakeOwnerModel::arcCount() const {
  return storage_ ? storage_->arcCount : 0;
}

HandshakeOwnerArc HandshakeOwnerModel::arc(std::uint32_t ordinal) const {
  assert(storage_ && ordinal < storage_->arcCount);
  const std::size_t instance = detail::locateInstance(
      *storage_, ordinal,
      [](const detail::HandshakeOwnerModelInstanceLayout &layout) {
        return layout.arcOffset;
      });
  const auto &layout = storage_->layouts[instance];
  HandshakeOwnerArc result =
      storage_->instances[instance].structure->arcs[ordinal - layout.arcOffset];
  result.source += layout.nodeOffset;
  result.destination += layout.nodeOffset;
  return result;
}

std::uint32_t HandshakeOwnerModel::fragmentCount() const {
  return storage_ ? storage_->fragmentCount : 0;
}

HandshakeActivationFragment
HandshakeOwnerModel::fragment(std::uint32_t ordinal) const {
  assert(storage_ && ordinal < storage_->fragmentCount);
  const std::size_t instance = detail::locateInstance(
      *storage_, ordinal,
      [](const detail::HandshakeOwnerModelInstanceLayout &layout) {
        return layout.fragmentOffset;
      });
  const auto &layout = storage_->layouts[instance];
  const auto &record =
      storage_->instances[instance]
          .structure->fragments[ordinal - layout.fragmentOffset];
  const auto &instanceStorage = storage_->instances[instance];
  const auto switchActivation =
      instanceStorage.switchActivationOverride
          ? instanceStorage.switchActivationOverride
          : (*instanceStorage.selectors)[ordinal - layout.fragmentOffset]
                .switchActivation;
  return {record.contributionOffset + layout.contributionOffset,
          record.contributionCount,
          record.activationKind,
          record.witnessOffset + layout.witnessOffset,
          record.witnessCount,
          switchActivation};
}

std::uint32_t HandshakeOwnerModel::fragmentContributionCount() const {
  return storage_ ? storage_->contributionCount : 0;
}

std::uint32_t
HandshakeOwnerModel::fragmentContributionOrdinal(std::uint32_t ordinal) const {
  assert(storage_ && ordinal < storage_->contributionCount);
  const std::size_t instance = detail::locateInstance(
      *storage_, ordinal,
      [](const detail::HandshakeOwnerModelInstanceLayout &layout) {
        return layout.contributionOffset;
      });
  const auto &layout = storage_->layouts[instance];
  return storage_->instances[instance].structure->fragmentContributionOrdinals
             [ordinal - layout.contributionOffset] +
         layout.arcOffset;
}

std::uint32_t HandshakeOwnerModel::traversalWitnessCount() const {
  return storage_ ? storage_->witnessCount : 0;
}

FabricPhysicalTraversalRef
HandshakeOwnerModel::traversalWitness(std::uint32_t ordinal) const {
  assert(storage_ && ordinal < storage_->witnessCount);
  const std::size_t instance = detail::locateInstance(
      *storage_, ordinal,
      [](const detail::HandshakeOwnerModelInstanceLayout &layout) {
        return layout.witnessOffset;
      });
  const auto &layout = storage_->layouts[instance];
  return (*storage_->instances[instance]
               .traversalWitnessBindings)[ordinal - layout.witnessOffset];
}

detail::HandshakeFragmentSelector
HandshakeOwnerModel::fragmentSelector(std::uint32_t ordinal) const {
  assert(storage_ && ordinal < storage_->fragmentCount);
  const std::size_t instance = detail::locateInstance(
      *storage_, ordinal,
      [](const detail::HandshakeOwnerModelInstanceLayout &layout) {
        return layout.fragmentOffset;
      });
  const auto &layout = storage_->layouts[instance];
  detail::HandshakeFragmentSelector selector =
      (*storage_->instances[instance]
            .selectors)[ordinal - layout.fragmentOffset];
  if (storage_->instances[instance].switchActivationOverride)
    selector.switchActivation =
        storage_->instances[instance].switchActivationOverride;
  return selector;
}

namespace detail {

llvm::Expected<HandshakeOwnerModel>
HandshakeOwnerModelFactory::rebindFuOccurrence(
    const FabricArtifactView &view, const HandshakeOwnerModel &definitionModel,
    FabricFuOccurrenceRef occurrence) {
  if (!definitionModel.storage_ ||
      definitionModel.owner().kind() != FabricHandshakeOwnerKind::FuOccurrence)
    return invalid("FU handshake definition model has the wrong owner");
  const FabricFuOccurrenceRef source =
      std::get<FabricFuOccurrenceRef>(definitionModel.owner().payload());
  const auto sourceDefinition = view.fuTemplateOf(source);
  const auto targetDefinition = view.fuTemplateOf(occurrence);
  if (!sourceDefinition || !targetDefinition ||
      *sourceDefinition != *targetDefinition)
    return invalid("FU handshake occurrences have different templates");

  std::vector<HandshakeOwnerModelInstance> instances;
  instances.reserve(definitionModel.storage_->instances.size());
  for (const HandshakeOwnerModelInstance &sourceInstance :
       definitionModel.storage_->instances) {
    if (!sourceInstance.traversalWitnessBindings->empty())
      return invalid("FU handshake template unexpectedly binds traversals");
    for (const HandshakeOwnerNode &node : *sourceInstance.nodeBindings)
      if (node.boundarySignal &&
          node.boundarySignal->endpoint.owner !=
              FabricTransportEndpointOwnerRef::of(source))
        return invalid("FU handshake template binds a foreign endpoint");
    auto nodes = rebindNodes(*sourceInstance.nodeBindings,
                             FabricTransportEndpointOwnerRef::of(source),
                             FabricTransportEndpointOwnerRef::of(occurrence));
    std::vector<HandshakeFragmentSelector> selectors =
        *sourceInstance.selectors;
    for (HandshakeFragmentSelector &selector : selectors) {
      if (selector.fuOccurrence) {
        if (*selector.fuOccurrence != source)
          return invalid("FU handshake selector binds a foreign occurrence");
        selector.fuOccurrence = occurrence;
      }
      if (selector.fuOperation) {
        if (selector.fuOperation->operation.fu != source)
          return invalid("FU handshake selector binds a foreign operation");
        selector.fuOperation->operation.fu = occurrence;
      }
    }
    instances.push_back(
        {sourceInstance.structure,
         std::make_shared<const std::vector<HandshakeOwnerNode>>(
             std::move(nodes)),
         sourceInstance.traversalWitnessBindings,
         std::make_shared<const std::vector<HandshakeFragmentSelector>>(
             std::move(selectors)),
         std::nullopt, sourceInstance.projectionShapeOrdinal});
  }
  auto storage =
      buildStorage(FabricHandshakeOwner::fu(occurrence), std::move(instances));
  if (!storage)
    return storage.takeError();
  return HandshakeOwnerModel(std::move(*storage));
}

llvm::Expected<HandshakeOwnerModel>
HandshakeOwnerModelFactory::rebindMemoryOccurrence(
    const FabricArtifactView &view, const HandshakeOwnerModel &definitionModel,
    FabricMemoryOccurrenceRef occurrence) {
  if (!definitionModel.storage_ ||
      definitionModel.owner().kind() !=
          FabricHandshakeOwnerKind::MemoryOccurrence)
    return invalid("Memory handshake definition model has the wrong owner");
  const FabricMemoryOccurrenceRef source =
      std::get<FabricMemoryOccurrenceRef>(definitionModel.owner().payload());
  const auto sourceDefinition = view.memoryEngineTemplateOf(source);
  const auto targetDefinition = view.memoryEngineTemplateOf(occurrence);
  if (!sourceDefinition || !targetDefinition ||
      *sourceDefinition != *targetDefinition)
    return invalid("Memory handshake occurrences have different templates");

  std::vector<HandshakeOwnerModelInstance> instances;
  instances.reserve(definitionModel.storage_->instances.size());
  for (const HandshakeOwnerModelInstance &sourceInstance :
       definitionModel.storage_->instances) {
    if (!sourceInstance.traversalWitnessBindings->empty())
      return invalid("Memory handshake template unexpectedly binds traversals");
    for (const HandshakeOwnerNode &node : *sourceInstance.nodeBindings)
      if (node.boundarySignal &&
          node.boundarySignal->endpoint.owner !=
              FabricTransportEndpointOwnerRef::of(source))
        return invalid("Memory handshake template binds a foreign endpoint");
    auto nodes = rebindNodes(*sourceInstance.nodeBindings,
                             FabricTransportEndpointOwnerRef::of(source),
                             FabricTransportEndpointOwnerRef::of(occurrence));
    std::vector<HandshakeFragmentSelector> selectors =
        *sourceInstance.selectors;
    for (HandshakeFragmentSelector &selector : selectors) {
      if (!selector.memoryCapability && !selector.memoryUsePattern)
        continue;
      if (!selector.memoryCapability || !selector.memoryUsePattern ||
          selector.memoryCapability->port.memory != source)
        return invalid("Memory handshake selector has a foreign plan");
      FabricMemoryOperationPortRef port{
          occurrence, selector.memoryCapability->port.ordinal};
      selector.memoryCapability = FabricMemoryCapabilityAlternativeRef{
          port, selector.memoryCapability->ordinal};
      selector.memoryUsePattern = FabricUsePatternRef{
          FabricUsePatternOwnerRef(FabricInventoryOwnerRef::of(port)),
          selector.memoryUsePattern->ordinal};
    }
    instances.push_back(
        {sourceInstance.structure,
         std::make_shared<const std::vector<HandshakeOwnerNode>>(
             std::move(nodes)),
         sourceInstance.traversalWitnessBindings,
         std::make_shared<const std::vector<HandshakeFragmentSelector>>(
             std::move(selectors)),
         std::nullopt, sourceInstance.projectionShapeOrdinal});
  }
  auto storage = buildStorage(FabricHandshakeOwner::memory(occurrence),
                              std::move(instances));
  if (!storage)
    return storage.takeError();
  return HandshakeOwnerModel(std::move(*storage));
}

llvm::Expected<HandshakeOwnerModel>
HandshakeOwnerModelFactory::instantiateSwitchRows(
    FabricSwitchOccurrenceRef occurrence,
    llvm::ArrayRef<HandshakeOwnerModel> rowShapes, std::uint64_t residentRows,
    bool temporal) {
  if (residentRows == 0 || rowShapes.empty())
    return invalid("switch handshake has no row shape");
  if (residentRows > std::numeric_limits<std::uint32_t>::max())
    return invalid("switch resident-row count exceeds the model domain");
  std::vector<HandshakeOwnerModelInstance> instances;
  if (rowShapes.size() > std::numeric_limits<std::size_t>::max() / residentRows)
    return invalid("switch handshake instance count overflows");
  instances.reserve(rowShapes.size() * static_cast<std::size_t>(residentRows));
  for (std::uint64_t row = 0; row != residentRows; ++row) {
    for (auto [shapeOrdinal, shape] : llvm::enumerate(rowShapes)) {
      if (!shape.storage_ || shape.storage_->instances.size() != 1 ||
          shape.owner() != FabricHandshakeOwner::switchResource(occurrence))
        return invalid("switch handshake row shape has the wrong owner");
      HandshakeOwnerModelInstance instance = shape.storage_->instances.front();
      if (temporal) {
        std::optional<FabricSwitchHandshakeActivationKey> activation;
        for (const HandshakeFragmentSelector &selector : *instance.selectors) {
          if (!selector.switchActivation)
            continue;
          if (selector.switchActivation->occurrence != occurrence ||
              (activation &&
               activation->input != selector.switchActivation->input))
            return invalid("switch row shape has inconsistent activation");
          activation = FabricSwitchHandshakeActivationKey{
              occurrence, row, selector.switchActivation->input};
        }
        if (!activation)
          return invalid("Temporal switch row shape has no activation");
        instance.switchActivationOverride = *activation;
      }
      instance.projectionShapeOrdinal =
          static_cast<std::uint32_t>(shapeOrdinal);
      instances.push_back(std::move(instance));
    }
  }
  auto storage = buildStorage(FabricHandshakeOwner::switchResource(occurrence),
                              std::move(instances));
  if (!storage)
    return storage.takeError();
  return HandshakeOwnerModel(std::move(*storage));
}

void HandshakeOwnerModelFactory::accumulateStatistics(
    llvm::ArrayRef<HandshakeOwnerModel> models,
    FabricHandshakeContextStatistics &statistics) {
  statistics.ownerCount = models.size();
  statistics.retainedBytes = models.size() * sizeof(HandshakeOwnerModel);
  std::set<const HandshakeOwnerModelStorage *> storages;
  std::set<const HandshakeStructuralTemplate *> structures;
  std::set<const std::vector<HandshakeOwnerNode> *> nodeBindings;
  std::set<const std::vector<FabricPhysicalTraversalRef> *> witnessBindings;
  std::set<const std::vector<HandshakeFragmentSelector> *> selectorBindings;
  for (const HandshakeOwnerModel &model : models) {
    if (!model.storage_ || !storages.insert(model.storage_.get()).second)
      continue;
    statistics.retainedBytes +=
        sizeof(HandshakeOwnerModelStorage) +
        model.storage_->instances.size() * sizeof(HandshakeOwnerModelInstance) +
        model.storage_->layouts.size() *
            sizeof(HandshakeOwnerModelInstanceLayout);
    statistics.bindingInstanceCount += model.storage_->instances.size();
    statistics.nodeCount += model.storage_->nodeCount;
    statistics.arcCount += model.storage_->arcCount;
    statistics.fragmentCount += model.storage_->fragmentCount;
    for (const HandshakeOwnerModelInstance &instance :
         model.storage_->instances) {
      if (structures.insert(instance.structure.get()).second) {
        ++statistics.structuralTemplateCount;
        statistics.structuralNodeCount += instance.structure->nodeKinds.size();
        statistics.structuralArcCount += instance.structure->arcs.size();
        statistics.structuralFragmentCount +=
            instance.structure->fragments.size();
        statistics.retainedBytes +=
            sizeof(HandshakeStructuralTemplate) +
            instance.structure->nodeKinds.size() *
                sizeof(HandshakeOwnerNodeKind) +
            instance.structure->arcs.size() * sizeof(HandshakeOwnerArc) +
            instance.structure->fragments.size() *
                sizeof(HandshakeStructuralFragment) +
            instance.structure->fragmentContributionOrdinals.size() *
                sizeof(std::uint32_t);
      }
      if (nodeBindings.insert(instance.nodeBindings.get()).second)
        statistics.retainedBytes +=
            sizeof(std::vector<HandshakeOwnerNode>) +
            instance.nodeBindings->size() * sizeof(HandshakeOwnerNode);
      if (witnessBindings.insert(instance.traversalWitnessBindings.get())
              .second)
        statistics.retainedBytes +=
            sizeof(std::vector<FabricPhysicalTraversalRef>) +
            instance.traversalWitnessBindings->size() *
                sizeof(FabricPhysicalTraversalRef);
      if (selectorBindings.insert(instance.selectors.get()).second) {
        statistics.retainedBytes +=
            sizeof(std::vector<HandshakeFragmentSelector>) +
            instance.selectors->size() * sizeof(HandshakeFragmentSelector);
        for (const HandshakeFragmentSelector &selector : *instance.selectors)
          statistics.retainedBytes +=
              selector.traversalWitnesses.size() *
                  sizeof(FabricPhysicalTraversalRef) +
              selector.requiredExternalMemoryInputRoles.size() *
                  sizeof(::dataflow::semantics::ServiceValueRole) +
              selector.requiredExternalMemoryOutputRoles.size() *
                  sizeof(::dataflow::semantics::ServiceValueRole);
      }
    }
  }
  statistics.deterministicWork =
      statistics.ownerCount + statistics.bindingInstanceCount +
      statistics.structuralTemplateCount + statistics.structuralNodeCount +
      statistics.structuralArcCount + statistics.structuralFragmentCount +
      statistics.nodeCount + statistics.arcCount + statistics.fragmentCount;
}

} // namespace detail
} // namespace loom::fabric
