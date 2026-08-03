#include "Fabric/Identity/FabricHandshake.h"

#include "FabricHandshakeInternal.h"

#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

namespace loom::fabric {
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

std::vector<std::uint8_t> signalKey(const HandshakeSignalRef &signal) {
  std::vector<std::uint8_t> key = canonicalFabricBytes(signal.endpoint);
  key.insert(key.begin(), static_cast<std::uint8_t>(signal.signal));
  key.insert(key.begin(), 0);
  return key;
}

std::vector<std::uint8_t> ownerKey(const FabricHandshakeOwner &owner) {
  std::vector<std::uint8_t> key{static_cast<std::uint8_t>(owner.kind())};
  std::visit(
      [&](const auto &payload) {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<Payload, FabricPointConnectionPayload>) {
          const auto source = canonicalFabricBytes(payload.source);
          const auto destination = canonicalFabricBytes(payload.destination);
          key.insert(key.end(), source.begin(), source.end());
          key.push_back(0xff);
          key.insert(key.end(), destination.begin(), destination.end());
        } else {
          const auto bytes = canonicalFabricBytes(payload);
          key.insert(key.end(), bytes.begin(), bytes.end());
        }
      },
      owner.payload());
  return key;
}

std::vector<std::uint8_t>
junctionKey(llvm::ArrayRef<std::uint8_t> ownerLocalKey) {
  std::vector<std::uint8_t> key;
  key.reserve(ownerLocalKey.size() + 1);
  key.push_back(1);
  key.insert(key.end(), ownerLocalKey.begin(), ownerLocalKey.end());
  return key;
}

std::vector<std::uint8_t> ordinalKey(std::uint8_t family, std::uint64_t first,
                                     std::uint64_t second) {
  std::vector<std::uint8_t> key;
  key.reserve(17);
  key.push_back(family);
  for (unsigned shift = 0; shift != 64; shift += 8)
    key.push_back(static_cast<std::uint8_t>(first >> (56 - shift)));
  for (unsigned shift = 0; shift != 64; shift += 8)
    key.push_back(static_cast<std::uint8_t>(second >> (56 - shift)));
  return key;
}

std::optional<FabricHandshakeOwner>
ownerOfTraversal(const FabricPhysicalTraversalRef &traversal) {
  switch (traversal.kind()) {
  case FabricPhysicalTraversalKind::PointConnection:
    return FabricHandshakeOwner::pointConnection(
        std::get<FabricPointConnectionPayload>(traversal.payload));
  case FabricPhysicalTraversalKind::PeSelectorTraversal:
    return FabricHandshakeOwner::pe(
        std::get<FabricPeSelectorPayload>(traversal.payload).owner);
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal:
    return FabricHandshakeOwner::pe(
        std::get<FabricPeRegisterFifoPayload>(traversal.payload).owner);
  case FabricPhysicalTraversalKind::SwitchTraversal:
    return FabricHandshakeOwner::switchResource(
        std::get<FabricSwitchTraversalPayload>(traversal.payload).owner);
  case FabricPhysicalTraversalKind::FifoTraversal:
    return FabricHandshakeOwner::fifo(
        std::get<FabricFifoTraversalPayload>(traversal.payload).owner);
  case FabricPhysicalTraversalKind::BoundaryTraversal:
    return FabricHandshakeOwner::boundary(
        std::get<FabricBoundaryTraversalPayload>(traversal.payload).owner);
  case FabricPhysicalTraversalKind::SystemTransferPatternLeg:
    return FabricHandshakeOwner::transferPattern(
        std::get<FabricTransferPatternLegPayload>(traversal.payload).owner);
  }
  return std::nullopt;
}

bool containsTraversal(llvm::ArrayRef<FabricPhysicalTraversalRef> values,
                       const FabricPhysicalTraversalRef &needle) {
  return llvm::is_contained(values, needle);
}

std::vector<FabricTransportEndpointRef>
directionalEndpoints(const FabricArtifactView &view,
                     const FabricTransportEndpointOwnerRef &owner,
                     FabricPortDirection direction) {
  std::vector<FabricTransportEndpointRef> endpoints;
  const std::uint64_t count = view.transportEndpointCount(owner);
  for (FabricOrdinal ordinal = 0; ordinal < count; ++ordinal) {
    FabricTransportEndpointRef endpoint{owner, ordinal};
    if (view.transportEndpointDirection(endpoint) == direction)
      endpoints.push_back(endpoint);
  }
  return endpoints;
}

FabricMemoryOperationPortRef
memoryPlacementPort(const FabricMemoryHandshakePlacement &placement) {
  if (const auto *port = std::get_if<FabricMemoryOperationPortRef>(&placement))
    return *port;
  return std::get<FabricMemoryOperationContextRef>(placement).port;
}

std::vector<std::uint8_t>
memoryPlacementKey(const FabricMemoryHandshakePlacement &placement) {
  if (const auto *port =
          std::get_if<FabricMemoryOperationPortRef>(&placement)) {
    std::vector<std::uint8_t> key = canonicalFabricBytes(*port);
    key.insert(key.begin(), 0);
    return key;
  }
  std::vector<std::uint8_t> key = canonicalFabricBytes(
      std::get<FabricMemoryOperationContextRef>(placement));
  key.insert(key.begin(), 1);
  return key;
}

std::vector<std::uint8_t>
memorySelectionKey(const FabricMemoryHandshakeSelection &selection) {
  std::vector<std::uint8_t> key = memoryPlacementKey(selection.placement());
  const std::vector<std::uint8_t> capability =
      canonicalFabricBytes(selection.capability());
  const std::vector<std::uint8_t> pattern =
      canonicalFabricBytes(selection.usePattern());
  key.push_back(2);
  key.insert(key.end(), capability.begin(), capability.end());
  key.push_back(3);
  key.insert(key.end(), pattern.begin(), pattern.end());
  key.push_back(static_cast<std::uint8_t>(selection.maskForm()));
  return key;
}

bool supportsMaskForm(const MemoryCapabilityAlternativeView &capability,
                      ::dataflow::semantics::MemoryMaskForm maskForm) {
  if (!capability.accessDomain)
    return maskForm == ::dataflow::semantics::MemoryMaskForm::Absent;
  for (const ::fabric::MemoryAccessClass &access :
       capability.accessDomain->accessClasses())
    for (const ::fabric::MaskInactivePair pair : access.maskInactivePairs())
      if (pair.mask == maskForm)
        return true;
  return false;
}

} // namespace

namespace detail {

HandshakeOwnerModelBuilder::HandshakeOwnerModelBuilder(
    FabricHandshakeOwner owner)
    : model_(std::move(owner)) {}

std::uint32_t
HandshakeOwnerModelBuilder::boundarySignal(HandshakeSignalRef signal) {
  const std::vector<std::uint8_t> key = signalKey(signal);
  auto found = nodes_.find(key);
  if (found != nodes_.end())
    return found->second;
  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(model_.nodes_.size());
  model_.nodes_.push_back(
      {HandshakeOwnerNodeKind::BoundarySignal, std::move(signal)});
  nodes_.emplace(key, ordinal);
  return ordinal;
}

std::uint32_t HandshakeOwnerModelBuilder::junction(
    llvm::ArrayRef<std::uint8_t> ownerLocalKey) {
  const std::vector<std::uint8_t> key = junctionKey(ownerLocalKey);
  auto found = nodes_.find(key);
  if (found != nodes_.end())
    return found->second;
  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(model_.nodes_.size());
  model_.nodes_.push_back(
      {HandshakeOwnerNodeKind::OwnerLocalJunction, std::nullopt});
  nodes_.emplace(key, ordinal);
  return ordinal;
}

void HandshakeOwnerModelBuilder::addFragment(
    HandshakeFragmentSelector selector,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> arcs) {
  pending_.push_back({std::move(selector), std::move(arcs)});
}

llvm::Expected<HandshakeOwnerModel> HandshakeOwnerModelBuilder::finish() {
  if (auto count = checkedSize(model_.nodes_, "handshake node count"); !count)
    return count.takeError();

  std::set<std::pair<std::uint32_t, std::uint32_t>> uniqueArcs;
  for (const PendingFragment &fragment : pending_)
    uniqueArcs.insert(fragment.arcs.begin(), fragment.arcs.end());
  if (uniqueArcs.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("handshake arc count exceeds the owner-model index domain");

  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> arcOrdinals;
  model_.arcs_.reserve(uniqueArcs.size());
  for (const auto &arc : uniqueArcs) {
    const std::uint32_t ordinal =
        static_cast<std::uint32_t>(model_.arcs_.size());
    arcOrdinals.emplace(arc, ordinal);
    model_.arcs_.push_back({arc.first, arc.second});
  }

  model_.fragments_.reserve(pending_.size());
  model_.fragmentSelectors_.reserve(pending_.size());
  for (PendingFragment &pending : pending_) {
    std::vector<std::uint32_t> contributions;
    contributions.reserve(pending.arcs.size());
    for (const auto &arc : pending.arcs)
      contributions.push_back(arcOrdinals.at(arc));
    llvm::sort(contributions);
    contributions.erase(std::unique(contributions.begin(), contributions.end()),
                        contributions.end());
    auto offset = checkedSize(model_.fragmentContributionOrdinals_,
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
    case HandshakeFragmentSelectorKind::FuCapability:
    case HandshakeFragmentSelectorKind::FuOperationCase:
    case HandshakeFragmentSelectorKind::FuOperationInputActive:
    case HandshakeFragmentSelectorKind::FuOperationResultActive:
    case HandshakeFragmentSelectorKind::MemoryOperationPlan:
      break;
    }

    std::vector<FabricPhysicalTraversalRef> witnesses;
    if (activationKind == HandshakeActivationKind::AnyTraversal ||
        activationKind == HandshakeActivationKind::AllTraversals) {
      witnesses = pending.selector.traversalWitnesses;
      llvm::sort(witnesses, [](const auto &lhs, const auto &rhs) {
        return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
      });
      witnesses.erase(std::unique(witnesses.begin(), witnesses.end()),
                      witnesses.end());
      if (witnesses.empty())
        return invalid("traversal-selected fragment has no witness");
    }
    auto witnessOffset =
        checkedSize(model_.traversalWitnesses_, "handshake witness offset");
    auto witnessCount = checkedSize(witnesses, "handshake witness count");
    if (!witnessOffset)
      return witnessOffset.takeError();
    if (!witnessCount)
      return witnessCount.takeError();
    model_.fragments_.push_back(
        {*offset, *count, activationKind, *witnessOffset, *witnessCount});
    model_.fragmentContributionOrdinals_.insert(
        model_.fragmentContributionOrdinals_.end(), contributions.begin(),
        contributions.end());
    model_.traversalWitnesses_.insert(model_.traversalWitnesses_.end(),
                                      witnesses.begin(), witnesses.end());
    model_.fragmentSelectors_.push_back(std::move(pending.selector));
  }
  return std::move(model_);
}

} // namespace detail

FabricHandshakeOwner
FabricHandshakeOwner::pointConnection(FabricPointConnectionPayload connection) {
  return FabricHandshakeOwner(Payload(
      std::in_place_type<FabricPointConnectionPayload>, std::move(connection)));
}

FabricHandshakeOwner FabricHandshakeOwner::pe(FabricPeOccurrenceRef owner) {
  return FabricHandshakeOwner(
      Payload(std::in_place_type<FabricPeOccurrenceRef>, owner));
}

FabricHandshakeOwner FabricHandshakeOwner::fu(FabricFuOccurrenceRef owner) {
  return FabricHandshakeOwner(
      Payload(std::in_place_type<FabricFuOccurrenceRef>, owner));
}

FabricHandshakeOwner
FabricHandshakeOwner::memory(FabricMemoryOccurrenceRef owner) {
  return FabricHandshakeOwner(
      Payload(std::in_place_type<FabricMemoryOccurrenceRef>, owner));
}

FabricHandshakeOwner
FabricHandshakeOwner::switchResource(FabricSwitchOccurrenceRef owner) {
  return FabricHandshakeOwner(
      Payload(std::in_place_type<FabricSwitchOccurrenceRef>, owner));
}

FabricHandshakeOwner FabricHandshakeOwner::fifo(FabricFifoOccurrenceRef owner) {
  return FabricHandshakeOwner(
      Payload(std::in_place_type<FabricFifoOccurrenceRef>, owner));
}

FabricHandshakeOwner
FabricHandshakeOwner::boundary(FabricBoundaryOccurrenceRef owner) {
  return FabricHandshakeOwner(
      Payload(std::in_place_type<FabricBoundaryOccurrenceRef>, owner));
}

FabricHandshakeOwner
FabricHandshakeOwner::transferPattern(FabricTransferPatternRef owner) {
  return FabricHandshakeOwner(
      Payload(std::in_place_type<FabricTransferPatternRef>, owner));
}

llvm::Expected<FabricMemoryHandshakeSelection>
makeMemoryHandshakeSelection(const FabricArtifactView &view,
                             FabricMemoryHandshakePlacement placement,
                             FabricMemoryCapabilityAlternativeRef capability,
                             FabricUsePatternRef usePattern,
                             ::dataflow::semantics::MemoryMaskForm maskForm) {
  const FabricMemoryOperationPortRef port = memoryPlacementPort(placement);
  if (capability.port != port)
    return invalid("memory capability and placement name different ports");
  const auto schedule = view.memorySchedule(port.memory);
  if (!schedule)
    return invalid("memory operation placement has no scheduling contract");
  if (const auto *spatial =
          std::get_if<FabricMemoryOperationPortRef>(&placement)) {
    if (*spatial != port || *schedule != ::fabric::Schedule::Spatial)
      return invalid("Spatial memory placement has the wrong schedule");
    if (llvm::Error error = validateFabricRef(view, *spatial))
      return std::move(error);
  } else {
    const auto &context = std::get<FabricMemoryOperationContextRef>(placement);
    if (*schedule != ::fabric::Schedule::Temporal)
      return invalid("Temporal memory context has the wrong schedule");
    if (llvm::Error error = validateFabricRef(view, context))
      return std::move(error);
  }

  const MemoryOperationPortView *portRecord = view.memoryOperationPort(port);
  const MemoryCapabilityAlternativeView *alternative =
      view.memoryCapabilityAlternative(capability);
  if (!portRecord || !alternative)
    return invalid("memory capability alternative does not resolve");
  const FabricUsePatternOwnerRef expectedOwner(
      FabricInventoryOwnerRef::of(port));
  if (usePattern.owner != expectedOwner)
    return invalid("memory use pattern has the wrong operation-port owner");
  if (usePattern.ordinal >= portRecord->resourceContract().usePatternCount())
    return invalid("memory operation plan names an unknown use pattern");
  const ::fabric::UsePatternKey pattern(
      static_cast<std::uint32_t>(usePattern.ordinal));
  if (!llvm::is_contained(alternative->admissibleUsePatterns, pattern))
    return invalid("memory operation plan selects an inadmissible use pattern");
  if (!supportsMaskForm(*alternative, maskForm))
    return invalid("memory operation plan selects an unsupported mask form");
  return FabricMemoryHandshakeSelection(std::move(placement), capability,
                                        usePattern, maskForm);
}

std::optional<std::uint32_t>
HandshakeOwnerModel::nodeForSignal(const HandshakeSignalRef &signal) const {
  for (auto [ordinal, node] : llvm::enumerate(nodes_))
    if (node.boundarySignal && *node.boundarySignal == signal)
      return static_cast<std::uint32_t>(ordinal);
  return std::nullopt;
}

namespace {

using Arc = std::pair<std::uint32_t, std::uint32_t>;

detail::HandshakeFragmentSelector
alwaysSelector(llvm::ArrayRef<FabricPhysicalTraversalRef> witnesses = {}) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::Always;
  selector.traversalWitnesses.assign(witnesses.begin(), witnesses.end());
  return selector;
}

detail::HandshakeFragmentSelector
traversalSelector(FabricPhysicalTraversalRef traversal,
                  std::optional<std::uint32_t> exclusiveGroup = std::nullopt) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::AnyTraversal;
  selector.traversalWitnesses.push_back(std::move(traversal));
  selector.exclusiveGroup = exclusiveGroup;
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
allTraversalsSelector(llvm::ArrayRef<FabricPhysicalTraversalRef> traversals) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::AllTraversals;
  selector.traversalWitnesses.assign(traversals.begin(), traversals.end());
  return selector;
}

detail::HandshakeFragmentSelector
memorySelector(FabricMemoryCapabilityAlternativeRef capability,
               FabricUsePatternRef usePattern,
               ::dataflow::semantics::MemoryMaskForm maskForm) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::MemoryOperationPlan;
  selector.memoryCapability = capability;
  selector.memoryUsePattern = usePattern;
  selector.memoryMaskForm = maskForm;
  return selector;
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> (56 - shift)));
}

std::vector<std::uint8_t> memoryJunctionKey(
    std::uint8_t family, FabricMemoryCapabilityAlternativeRef capability,
    FabricUsePatternRef usePattern,
    ::dataflow::semantics::MemoryMaskForm maskForm, std::uint64_t position) {
  std::vector<std::uint8_t> key;
  key.reserve(35);
  key.push_back(family);
  appendU64(key, capability.port.ordinal);
  appendU64(key, capability.ordinal);
  appendU64(key, usePattern.ordinal);
  key.push_back(static_cast<std::uint8_t>(maskForm));
  appendU64(key, position);
  return key;
}

void addAtomicPeerDependencies(
    detail::HandshakeOwnerModelBuilder &builder,
    llvm::ArrayRef<FabricTransportEndpointRef> endpoints,
    HandshakeSignalKind driverSignal, HandshakeSignalKind targetSignal,
    std::uint8_t family, FabricMemoryCapabilityAlternativeRef capability,
    FabricUsePatternRef usePattern,
    ::dataflow::semantics::MemoryMaskForm maskForm, std::vector<Arc> &arcs) {
  for (FabricTransportEndpointRef endpoint : endpoints) {
    builder.boundarySignal({endpoint, HandshakeSignalKind::Valid});
    builder.boundarySignal({endpoint, HandshakeSignalKind::Ready});
  }
  if (endpoints.size() < 2)
    return;

  std::vector<std::uint32_t> prefix(endpoints.size() + 1);
  std::vector<std::uint32_t> suffix(endpoints.size() + 1);
  for (std::size_t position = 0; position <= endpoints.size(); ++position) {
    prefix[position] = builder.junction(
        memoryJunctionKey(family, capability, usePattern, maskForm, position));
    suffix[position] = builder.junction(memoryJunctionKey(
        family + 1, capability, usePattern, maskForm, position));
  }
  for (std::size_t position = 0; position < endpoints.size(); ++position) {
    const std::uint32_t driver =
        builder.boundarySignal({endpoints[position], driverSignal});
    const std::uint32_t target =
        builder.boundarySignal({endpoints[position], targetSignal});
    arcs.emplace_back(prefix[position], prefix[position + 1]);
    arcs.emplace_back(driver, prefix[position + 1]);
    arcs.emplace_back(suffix[position + 1], suffix[position]);
    arcs.emplace_back(driver, suffix[position]);
    arcs.emplace_back(prefix[position], target);
    arcs.emplace_back(suffix[position + 1], target);
  }
}

void addDirectTraversal(detail::HandshakeOwnerModelBuilder &builder,
                        detail::HandshakeFragmentSelector selector,
                        FabricTransportEndpointRef source,
                        FabricTransportEndpointRef destination,
                        bool forwardValid, bool backwardReady) {
  std::vector<Arc> arcs;
  if (forwardValid)
    arcs.emplace_back(
        builder.boundarySignal({source, HandshakeSignalKind::Valid}),
        builder.boundarySignal({destination, HandshakeSignalKind::Valid}));
  if (backwardReady)
    arcs.emplace_back(
        builder.boundarySignal({destination, HandshakeSignalKind::Ready}),
        builder.boundarySignal({source, HandshakeSignalKind::Ready}));
  builder.addFragment(std::move(selector), std::move(arcs));
}

llvm::Expected<HandshakeOwnerModel>
compilePointModel(const FabricArtifactView &view,
                  const FabricPointConnectionPayload &connection) {
  detail::HandshakeOwnerModelBuilder builder(
      FabricHandshakeOwner::pointConnection(connection));
  const FabricPhysicalTraversalRef traversal =
      FabricPhysicalTraversalRef::pointConnection(connection.source,
                                                  connection.destination);
  addDirectTraversal(
      builder,
      alwaysSelector(llvm::ArrayRef<FabricPhysicalTraversalRef>(&traversal, 1)),
      connection.source, connection.destination,
      /*forwardValid=*/true, /*backwardReady=*/true);
  return builder.finish();
}

llvm::Expected<HandshakeOwnerModel>
compilePeModel(const FabricArtifactView &view, FabricPeOccurrenceRef owner) {
  detail::HandshakeOwnerModelBuilder builder(FabricHandshakeOwner::pe(owner));
  for (const FabricPhysicalTraversalView &traversal :
       view.physicalTraversals()) {
    auto traversalOwner = ownerOfTraversal(traversal.reference);
    if (!traversalOwner || *traversalOwner != FabricHandshakeOwner::pe(owner))
      continue;
    if (traversal.reference.kind() ==
        FabricPhysicalTraversalKind::PeRegisterFifoTraversal) {
      builder.addFragment(traversalSelector(traversal.reference), {});
      continue;
    }
    if (traversal.sources.size() != 1 || traversal.destinations.size() != 1)
      return invalid("PE selector traversal has invalid endpoint cardinality");
    addDirectTraversal(builder, traversalSelector(traversal.reference),
                       traversal.sources.front(),
                       traversal.destinations.front(),
                       /*forwardValid=*/true, /*backwardReady=*/true);
  }
  return builder.finish();
}

llvm::Expected<std::vector<FabricTransportEndpointRef>> activeMemoryEndpoints(
    const FabricArtifactView &view, FabricMemoryOccurrenceRef owner,
    const MemoryCapabilityAlternativeView &capability,
    ::dataflow::semantics::MemoryMaskForm maskForm, bool arguments) {
  auto kind = ::dataflow::semantics::getMemoryServiceKind(
      capability.actorContractDomain.actorSchema());
  if (!kind)
    return kind.takeError();
  const auto &schema = ::dataflow::semantics::getServiceRoleSchema(*kind);
  const auto roles = arguments ? schema.arguments : schema.results;
  std::vector<FabricTransportEndpointRef> endpoints;
  endpoints.reserve(roles.size());
  for (::dataflow::semantics::ServiceValueRole role : roles) {
    if (role == ::dataflow::semantics::ServiceValueRole::Mask &&
        maskForm == ::dataflow::semantics::MemoryMaskForm::Absent)
      continue;
    auto binding = llvm::find_if(
        capability.roleToEndpoint,
        [&](const ::fabric::MemoryRoleEndpointBindingRecord &candidate) {
          return candidate.role == role;
        });
    if (binding == capability.roleToEndpoint.end())
      return invalid("memory capability omits an active service role");
    FabricTransportEndpointRef endpoint{
        FabricTransportEndpointOwnerRef::of(owner), binding->endpointOrdinal};
    const auto direction = view.transportEndpointDirection(endpoint);
    const FabricPortDirection expected =
        arguments ? FabricPortDirection::Input : FabricPortDirection::Output;
    if (!direction || *direction != expected)
      return invalid("memory role resolves to the wrong endpoint direction");
    if (!llvm::is_contained(endpoints, endpoint))
      endpoints.push_back(endpoint);
  }
  return endpoints;
}

llvm::Expected<HandshakeOwnerModel>
compileMemoryModel(const FabricArtifactView &view,
                   FabricMemoryOccurrenceRef owner) {
  const auto schedule = view.memorySchedule(owner);
  if (!schedule)
    return invalid("memory occurrence has no scheduling contract");
  detail::HandshakeOwnerModelBuilder builder(
      FabricHandshakeOwner::memory(owner));
  for (FabricMemoryOperationPortRef port : view.memoryOperationPorts(owner)) {
    const MemoryOperationPortView *record = view.memoryOperationPort(port);
    if (!record)
      return invalid("memory operation port does not resolve");
    for (auto [alternativeOrdinal, capability] :
         llvm::enumerate(record->capabilityAlternatives())) {
      const FabricMemoryCapabilityAlternativeRef capabilityRef{
          port, static_cast<FabricOrdinal>(alternativeOrdinal)};
      std::vector<::dataflow::semantics::MemoryMaskForm> maskForms;
      if (!capability.accessDomain) {
        maskForms.push_back(::dataflow::semantics::MemoryMaskForm::Absent);
      } else {
        for (const ::fabric::MemoryAccessClass &access :
             capability.accessDomain->accessClasses())
          for (const ::fabric::MaskInactivePair pair :
               access.maskInactivePairs())
            if (!llvm::is_contained(maskForms, pair.mask))
              maskForms.push_back(pair.mask);
        llvm::sort(maskForms, [](auto lhs, auto rhs) {
          return static_cast<std::uint8_t>(lhs) <
                 static_cast<std::uint8_t>(rhs);
        });
      }
      for (::fabric::UsePatternKey pattern : capability.admissibleUsePatterns) {
        const FabricUsePatternRef usePattern{
            FabricUsePatternOwnerRef(FabricInventoryOwnerRef::of(port)),
            pattern.ordinal()};
        for (::dataflow::semantics::MemoryMaskForm maskForm : maskForms) {
          auto inputs =
              activeMemoryEndpoints(view, owner, capability, maskForm, true);
          if (!inputs)
            return inputs.takeError();
          auto outputs =
              activeMemoryEndpoints(view, owner, capability, maskForm, false);
          if (!outputs)
            return outputs.takeError();

          std::vector<Arc> arcs;
          if (*schedule == ::fabric::Schedule::Spatial) {
            addAtomicPeerDependencies(
                builder, *inputs, HandshakeSignalKind::Valid,
                HandshakeSignalKind::Ready, 16, capabilityRef, usePattern,
                maskForm, arcs);
          } else {
            for (FabricTransportEndpointRef endpoint : *inputs) {
              builder.boundarySignal({endpoint, HandshakeSignalKind::Valid});
              builder.boundarySignal({endpoint, HandshakeSignalKind::Ready});
            }
          }
          addAtomicPeerDependencies(builder, *outputs,
                                    HandshakeSignalKind::Ready,
                                    HandshakeSignalKind::Valid, 18,
                                    capabilityRef, usePattern, maskForm, arcs);
          builder.addFragment(
              memorySelector(capabilityRef, usePattern, maskForm),
              std::move(arcs));
        }
      }
    }
  }
  return builder.finish();
}

llvm::Expected<HandshakeOwnerModel>
compileSwitchModel(const FabricArtifactView &view,
                   FabricSwitchOccurrenceRef owner) {
  struct Row final {
    FabricOrdinal output = 0;
    FabricPhysicalTraversalRef reference;
    FabricTransportEndpointRef source;
    FabricTransportEndpointRef destination;
  };
  std::map<FabricOrdinal, std::vector<Row>> byInput;
  for (const FabricPhysicalTraversalView &traversal :
       view.physicalTraversals()) {
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

  detail::HandshakeOwnerModelBuilder builder(
      FabricHandshakeOwner::switchResource(owner));
  for (auto &[input, rows] : byInput) {
    llvm::sort(rows, [](const Row &lhs, const Row &rhs) {
      return lhs.output < rhs.output;
    });
    std::vector<FabricPhysicalTraversalRef> witnesses;
    witnesses.reserve(rows.size());
    for (const Row &row : rows)
      witnesses.push_back(row.reference);

    std::vector<std::uint32_t> prefix(rows.size() + 1);
    std::vector<std::uint32_t> suffix(rows.size() + 1);
    for (std::size_t position = 0; position <= rows.size(); ++position) {
      prefix[position] = builder.junction(ordinalKey(0, input, position));
      suffix[position] = builder.junction(ordinalKey(1, input, position));
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
    builder.addFragment(anyTraversalSelector(witnesses), std::move(base));

    for (std::size_t position = 0; position < rows.size(); ++position) {
      const Row &row = rows[position];
      const std::uint32_t inputValid =
          builder.boundarySignal({row.source, HandshakeSignalKind::Valid});
      const std::uint32_t outputValid =
          builder.boundarySignal({row.destination, HandshakeSignalKind::Valid});
      const std::uint32_t outputReady =
          builder.boundarySignal({row.destination, HandshakeSignalKind::Ready});
      builder.addFragment(traversalSelector(row.reference),
                          {{outputReady, prefix[position + 1]},
                           {outputReady, suffix[position]},
                           {inputValid, outputValid},
                           {prefix[position], outputValid},
                           {suffix[position + 1], outputValid}});
    }
  }
  return builder.finish();
}

llvm::Expected<HandshakeOwnerModel>
compileFifoModel(const FabricArtifactView &view,
                 FabricFifoOccurrenceRef owner) {
  detail::HandshakeOwnerModelBuilder builder(FabricHandshakeOwner::fifo(owner));
  for (const FabricPhysicalTraversalView &traversal :
       view.physicalTraversals()) {
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::FifoTraversal)
      continue;
    const auto &payload =
        std::get<FabricFifoTraversalPayload>(traversal.reference.payload);
    if (payload.owner != owner)
      continue;
    if (traversal.sources.size() != 1 || traversal.destinations.size() != 1)
      return invalid("FIFO traversal has invalid endpoint cardinality");
    addDirectTraversal(builder, traversalSelector(traversal.reference, 0),
                       traversal.sources.front(),
                       traversal.destinations.front(),
                       payload.mode == FabricFifoTraversalMode::Bypass,
                       /*backwardReady=*/true);
  }
  return builder.finish();
}

llvm::Expected<HandshakeOwnerModel>
compileBoundaryModel(const FabricArtifactView &view,
                     FabricBoundaryOccurrenceRef owner) {
  detail::HandshakeOwnerModelBuilder builder(
      FabricHandshakeOwner::boundary(owner));
  const auto endpointOwner = FabricTransportEndpointOwnerRef::of(owner);
  const auto inputs =
      directionalEndpoints(view, endpointOwner, FabricPortDirection::Input);
  const auto outputs =
      directionalEndpoints(view, endpointOwner, FabricPortDirection::Output);
  std::vector<FabricPhysicalTraversalRef> witnesses;
  for (const FabricPhysicalTraversalRef &traversal :
       view.admittedTraversals()) {
    if (traversal.kind() != FabricPhysicalTraversalKind::BoundaryTraversal)
      continue;
    if (std::get<FabricBoundaryTraversalPayload>(traversal.payload).owner ==
        owner)
      witnesses.push_back(traversal);
  }
  std::vector<Arc> arcs;
  if (inputs.size() == 2 && outputs.size() == 1) {
    const std::uint32_t dataValid =
        builder.boundarySignal({inputs[0], HandshakeSignalKind::Valid});
    const std::uint32_t tagValid =
        builder.boundarySignal({inputs[1], HandshakeSignalKind::Valid});
    const std::uint32_t dataReady =
        builder.boundarySignal({inputs[0], HandshakeSignalKind::Ready});
    const std::uint32_t tagReady =
        builder.boundarySignal({inputs[1], HandshakeSignalKind::Ready});
    const std::uint32_t outputValid =
        builder.boundarySignal({outputs[0], HandshakeSignalKind::Valid});
    const std::uint32_t outputReady =
        builder.boundarySignal({outputs[0], HandshakeSignalKind::Ready});
    arcs = {{dataValid, outputValid}, {tagValid, outputValid},
            {outputReady, dataReady}, {tagValid, dataReady},
            {outputReady, tagReady},  {dataValid, tagReady}};
  } else if (inputs.size() == 1 && outputs.size() == 2) {
    const std::uint32_t inputValid =
        builder.boundarySignal({inputs[0], HandshakeSignalKind::Valid});
    const std::uint32_t inputReady =
        builder.boundarySignal({inputs[0], HandshakeSignalKind::Ready});
    const std::uint32_t dataValid =
        builder.boundarySignal({outputs[0], HandshakeSignalKind::Valid});
    const std::uint32_t tagValid =
        builder.boundarySignal({outputs[1], HandshakeSignalKind::Valid});
    const std::uint32_t dataReady =
        builder.boundarySignal({outputs[0], HandshakeSignalKind::Ready});
    const std::uint32_t tagReady =
        builder.boundarySignal({outputs[1], HandshakeSignalKind::Ready});
    arcs = {{inputValid, dataValid}, {tagReady, dataValid},
            {inputValid, tagValid},  {dataReady, tagValid},
            {dataReady, inputReady}, {tagReady, inputReady}};
  } else if (inputs.size() == 1 && outputs.size() == 1) {
    const std::uint32_t inputValid =
        builder.boundarySignal({inputs[0], HandshakeSignalKind::Valid});
    const std::uint32_t inputReady =
        builder.boundarySignal({inputs[0], HandshakeSignalKind::Ready});
    const std::uint32_t outputValid =
        builder.boundarySignal({outputs[0], HandshakeSignalKind::Valid});
    const std::uint32_t outputReady =
        builder.boundarySignal({outputs[0], HandshakeSignalKind::Ready});
    arcs = {{inputValid, outputValid}, {outputReady, inputReady}};
  } else {
    return invalid("boundary occurrence has an unsupported endpoint shape");
  }
  builder.addFragment(anyTraversalSelector(witnesses), std::move(arcs));
  return builder.finish();
}

llvm::Expected<HandshakeOwnerModel>
compileTransferPatternModel(const FabricArtifactView &view,
                            FabricTransferPatternRef owner) {
  std::vector<const FabricPhysicalTraversalView *> legs;
  for (const FabricPhysicalTraversalView &traversal :
       view.physicalTraversals()) {
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::SystemTransferPatternLeg)
      continue;
    if (std::get<FabricTransferPatternLegPayload>(traversal.reference.payload)
            .owner == owner)
      legs.push_back(&traversal);
  }
  llvm::sort(legs, [](const auto *lhs, const auto *rhs) {
    return std::get<FabricTransferPatternLegPayload>(lhs->reference.payload)
               .egress <
           std::get<FabricTransferPatternLegPayload>(rhs->reference.payload)
               .egress;
  });
  if (legs.empty())
    return invalid("transfer pattern has no traversal legs");

  detail::HandshakeOwnerModelBuilder builder(
      FabricHandshakeOwner::transferPattern(owner));
  std::vector<FabricPhysicalTraversalRef> witnesses;
  witnesses.reserve(legs.size());
  for (const auto *leg : legs)
    witnesses.push_back(leg->reference);
  std::vector<std::uint32_t> prefix(legs.size() + 1);
  std::vector<std::uint32_t> suffix(legs.size() + 1);
  for (std::size_t position = 0; position <= legs.size(); ++position) {
    prefix[position] = builder.junction(ordinalKey(2, 0, position));
    suffix[position] = builder.junction(ordinalKey(3, 0, position));
  }
  std::vector<Arc> base;
  for (std::size_t position = 0; position < legs.size(); ++position) {
    base.emplace_back(prefix[position], prefix[position + 1]);
    base.emplace_back(suffix[position + 1], suffix[position]);
  }
  const FabricTransportEndpointRef ingress = legs.front()->sources.front();
  base.emplace_back(prefix.back(), builder.boundarySignal(
                                       {ingress, HandshakeSignalKind::Ready}));
  builder.addFragment(allTraversalsSelector(witnesses), std::move(base));
  for (std::size_t position = 0; position < legs.size(); ++position) {
    if (legs[position]->sources.size() != 1 ||
        legs[position]->destinations.size() != 1 ||
        legs[position]->sources.front() != ingress)
      return invalid("transfer-pattern legs do not share one ingress");
    const FabricTransportEndpointRef egress =
        legs[position]->destinations.front();
    const std::uint32_t ingressValid =
        builder.boundarySignal({ingress, HandshakeSignalKind::Valid});
    const std::uint32_t egressValid =
        builder.boundarySignal({egress, HandshakeSignalKind::Valid});
    const std::uint32_t egressReady =
        builder.boundarySignal({egress, HandshakeSignalKind::Ready});
    builder.addFragment(traversalSelector(legs[position]->reference),
                        {{egressReady, prefix[position + 1]},
                         {egressReady, suffix[position]},
                         {ingressValid, egressValid},
                         {prefix[position], egressValid},
                         {suffix[position + 1], egressValid}});
  }
  return builder.finish();
}

} // namespace

llvm::Expected<std::vector<HandshakeOwnerModel>>
compileHandshakeOwnerModels(const FabricArtifactView &view) {
  std::vector<HandshakeOwnerModel> models;
  models.reserve(view.pointConnections().size() + view.peOccurrences().size() +
                 view.fuOccurrences().size() + view.memoryOccurrences().size() +
                 view.switchOccurrences().size() +
                 view.fifoOccurrences().size() +
                 view.boundaryOccurrences().size());
  for (const FabricPointConnectionPayload &connection :
       view.pointConnections()) {
    auto model = compilePointModel(view, connection);
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  for (FabricPeOccurrenceRef owner : view.peOccurrences()) {
    auto model = compilePeModel(view, owner);
    if (!model)
      return model.takeError();
    if (!model->fragments().empty())
      models.push_back(std::move(*model));
  }
  for (FabricFuOccurrenceRef owner : view.fuOccurrences()) {
    auto model = detail::compileFuHandshakeModel(view, owner);
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  for (FabricMemoryOccurrenceRef owner : view.memoryOccurrences()) {
    if (view.memoryOperationPorts(owner).empty())
      continue;
    auto model = compileMemoryModel(view, owner);
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  for (FabricSwitchOccurrenceRef owner : view.switchOccurrences()) {
    auto model = compileSwitchModel(view, owner);
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  for (FabricFifoOccurrenceRef owner : view.fifoOccurrences()) {
    auto model = compileFifoModel(view, owner);
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  for (FabricBoundaryOccurrenceRef owner : view.boundaryOccurrences()) {
    auto model = compileBoundaryModel(view, owner);
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }

  std::set<std::vector<std::uint8_t>> transferPatterns;
  for (const FabricPhysicalTraversalRef &traversal :
       view.admittedTraversals()) {
    if (traversal.kind() !=
        FabricPhysicalTraversalKind::SystemTransferPatternLeg)
      continue;
    const FabricTransferPatternRef owner =
        std::get<FabricTransferPatternLegPayload>(traversal.payload).owner;
    if (!transferPatterns.insert(canonicalFabricBytes(owner)).second)
      continue;
    auto model = compileTransferPatternModel(view, owner);
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  return models;
}

namespace {

const FabricFuOperationHandshakeSelection *
findFuOperationSelection(const FabricFuHandshakeSelection &selection,
                         const detail::HandshakeFuOperationSelector &selector) {
  const auto found =
      llvm::find_if(selection.operations(),
                    [&](const FabricFuOperationHandshakeSelection &operation) {
                      return operation.operation == selector.operation &&
                             operation.schema == selector.schema;
                    });
  return found == selection.operations().end() ? nullptr : &*found;
}

llvm::Expected<bool>
isFuOperationFragmentActive(const detail::HandshakeFragmentSelector &fragment,
                            const FabricFuHandshakeSelection &selection) {
  if (!fragment.fuOperation)
    return invalid("FU operation fragment has no typed selector");
  const detail::HandshakeFuOperationSelector &selector = *fragment.fuOperation;
  const FabricFuOperationHandshakeSelection *operation =
      findFuOperationSelection(selection, selector);
  if (!operation)
    return false;
  auto cases = ::dataflow::semantics::projectActorHandshakeCases(
      operation->schema,
      static_cast<std::uint32_t>(operation->operandPorts.size()),
      static_cast<std::uint32_t>(operation->resultPorts.size()));
  if (!cases)
    return cases.takeError();
  const auto transition = llvm::find_if(
      *cases, [&](const ::dataflow::semantics::ActorHandshakeCase &candidate) {
        return candidate.ordinal == selector.caseOrdinal;
      });
  if (transition == cases->end())
    return false;

  switch (fragment.kind) {
  case detail::HandshakeFragmentSelectorKind::FuOperationCase:
    return true;
  case detail::HandshakeFragmentSelectorKind::FuOperationInputActive:
    return llvm::any_of(
        transition->consumedInputs, [&](std::uint32_t logicalOrdinal) {
          return logicalOrdinal < operation->operandPorts.size() &&
                 operation->operandPorts[logicalOrdinal] ==
                     selector.physicalPortOrdinal;
        });
  case detail::HandshakeFragmentSelectorKind::FuOperationResultActive:
    return llvm::any_of(
        transition->activeResults, [&](std::uint32_t logicalOrdinal) {
          return logicalOrdinal < operation->resultPorts.size() &&
                 operation->resultPorts[logicalOrdinal] ==
                     selector.physicalPortOrdinal;
        });
  default:
    return invalid("non-operation fragment reached the FU operation resolver");
  }
}

} // namespace

llvm::Expected<ResolvedHandshakeActivation>
resolveSelectedHandshake(const HandshakeOwnerModel &model,
                         const FabricHandshakeSelection &selection) {
  std::set<std::vector<std::uint8_t>> traversalKeys;
  for (const FabricPhysicalTraversalRef &traversal : selection.traversals)
    if (!traversalKeys.insert(canonicalFabricBytes(traversal)).second)
      return invalid("selected traversal relation contains a duplicate");
  for (auto [index, lhs] : llvm::enumerate(selection.fuCapabilities))
    for (std::size_t rhs = index + 1; rhs < selection.fuCapabilities.size();
         ++rhs)
      if (lhs.occurrence() == selection.fuCapabilities[rhs].occurrence())
        return invalid(
            "selected FU capability relation repeats one occurrence");

  std::set<std::vector<std::uint8_t>> memorySelections;
  std::set<std::vector<std::uint8_t>> memoryPlacements;
  for (const FabricMemoryHandshakeSelection &selected :
       selection.memoryOperations) {
    if (!memorySelections.insert(memorySelectionKey(selected)).second)
      return invalid("selected memory operation relation contains a duplicate");
    if (!memoryPlacements.insert(memoryPlacementKey(selected.placement()))
             .second)
      return invalid("selected memory operation placement is contradictory");
  }

  std::vector<bool> active(model.fragments_.size(), false);
  std::vector<bool> selectedTraversalConsumed(selection.traversals.size(),
                                              false);
  std::vector<bool> selectedMemoryConsumed(selection.memoryOperations.size(),
                                           false);
  const FabricFuHandshakeSelection *selectedFu = nullptr;
  if (model.owner_.kind() == FabricHandshakeOwnerKind::FuOccurrence) {
    const FabricFuOccurrenceRef owner =
        std::get<FabricFuOccurrenceRef>(model.owner_.payload());
    for (const FabricFuHandshakeSelection &candidate : selection.fuCapabilities)
      if (candidate.occurrence() == owner)
        selectedFu = &candidate;
  }
  std::map<std::uint32_t, std::uint32_t> selectedExclusiveGroups;
  for (std::size_t ordinal = 0; ordinal < model.fragmentSelectors_.size();
       ++ordinal) {
    const auto &selector = model.fragmentSelectors_[ordinal];
    bool selected = false;
    switch (selector.kind) {
    case detail::HandshakeFragmentSelectorKind::Always:
      selected = true;
      for (std::size_t selectedOrdinal = 0;
           selectedOrdinal < selection.traversals.size(); ++selectedOrdinal)
        if (containsTraversal(selector.traversalWitnesses,
                              selection.traversals[selectedOrdinal]))
          selectedTraversalConsumed[selectedOrdinal] = true;
      break;
    case detail::HandshakeFragmentSelectorKind::AnyTraversal:
      for (std::size_t selectedOrdinal = 0;
           selectedOrdinal < selection.traversals.size(); ++selectedOrdinal) {
        if (!containsTraversal(selector.traversalWitnesses,
                               selection.traversals[selectedOrdinal]))
          continue;
        selected = true;
        selectedTraversalConsumed[selectedOrdinal] = true;
      }
      break;
    case detail::HandshakeFragmentSelectorKind::AllTraversals: {
      std::size_t selectedWitnessCount = 0;
      for (const FabricPhysicalTraversalRef &witness :
           selector.traversalWitnesses) {
        auto found = llvm::find(selection.traversals, witness);
        if (found == selection.traversals.end())
          continue;
        ++selectedWitnessCount;
        selectedTraversalConsumed[std::distance(selection.traversals.begin(),
                                                found)] = true;
      }
      if (selectedWitnessCount != 0 &&
          selectedWitnessCount != selector.traversalWitnesses.size())
        return invalid("selected owner has an incomplete traversal set");
      selected = selectedWitnessCount != 0;
      break;
    }
    case detail::HandshakeFragmentSelectorKind::FuCapability:
      selected = selectedFu && selector.fuOccurrence && selector.fuCapability &&
                 *selector.fuOccurrence == selectedFu->occurrence() &&
                 *selector.fuCapability == selectedFu->capability();
      break;
    case detail::HandshakeFragmentSelectorKind::FuOperationCase:
    case detail::HandshakeFragmentSelectorKind::FuOperationInputActive:
    case detail::HandshakeFragmentSelectorKind::FuOperationResultActive:
      if (selectedFu && selector.fuOccurrence && selector.fuCapability &&
          *selector.fuOccurrence == selectedFu->occurrence() &&
          *selector.fuCapability == selectedFu->capability()) {
        auto active = isFuOperationFragmentActive(selector, *selectedFu);
        if (!active)
          return active.takeError();
        selected = *active;
      }
      break;
    case detail::HandshakeFragmentSelectorKind::MemoryOperationPlan:
      for (std::size_t selectedOrdinal = 0;
           selectedOrdinal < selection.memoryOperations.size();
           ++selectedOrdinal) {
        const FabricMemoryHandshakeSelection &candidate =
            selection.memoryOperations[selectedOrdinal];
        if (!selector.memoryCapability || !selector.memoryUsePattern ||
            !selector.memoryMaskForm ||
            candidate.capability() != *selector.memoryCapability ||
            candidate.usePattern() != *selector.memoryUsePattern ||
            candidate.maskForm() != *selector.memoryMaskForm)
          continue;
        selected = true;
        selectedMemoryConsumed[selectedOrdinal] = true;
      }
      break;
    }
    if (!selected)
      continue;
    active[ordinal] = true;
    if (selector.exclusiveGroup &&
        ++selectedExclusiveGroups[*selector.exclusiveGroup] != 1)
      return invalid("selected owner alternatives are contradictory");
  }

  for (std::size_t ordinal = 0; ordinal < selection.traversals.size();
       ++ordinal) {
    const auto owner = ownerOfTraversal(selection.traversals[ordinal]);
    if (owner && *owner == model.owner_ && !selectedTraversalConsumed[ordinal])
      return invalid("selected traversal is stale for its handshake owner");
  }
  if (model.owner_.kind() == FabricHandshakeOwnerKind::FuOccurrence) {
    const auto owner = std::get<FabricFuOccurrenceRef>(model.owner_.payload());
    bool hasSelection = false;
    for (const FabricFuHandshakeSelection &selected :
         selection.fuCapabilities) {
      if (selected.occurrence() != owner)
        continue;
      hasSelection = true;
      if (!llvm::any_of(model.fragmentSelectors_, [&](const auto &selector) {
            return selector.kind ==
                       detail::HandshakeFragmentSelectorKind::FuCapability &&
                   selector.fuOccurrence && selector.fuCapability &&
                   *selector.fuOccurrence == selected.occurrence() &&
                   *selector.fuCapability == selected.capability();
          }))
        return invalid("selected FU capability is stale for its occurrence");
      for (const FabricFuOperationHandshakeSelection &operation :
           selected.operations())
        if (!llvm::any_of(model.fragmentSelectors_, [&](const auto &selector) {
              return selector.fuOperation &&
                     selector.fuOperation->operation == operation.operation &&
                     selector.fuOperation->schema == operation.schema;
            }))
          return invalid("selected FU operation is stale for its occurrence");
    }
    if (!hasSelection && !model.fragments_.empty())
      return invalid("FU occurrence has no selected capability");
  }
  if (model.owner_.kind() == FabricHandshakeOwnerKind::MemoryOccurrence) {
    const auto owner =
        std::get<FabricMemoryOccurrenceRef>(model.owner_.payload());
    for (std::size_t ordinal = 0; ordinal < selection.memoryOperations.size();
         ++ordinal) {
      if (selection.memoryOperations[ordinal].capability().port.memory != owner)
        continue;
      if (!selectedMemoryConsumed[ordinal])
        return invalid("selected memory operation plan is stale for its owner");
    }
  }

  ResolvedHandshakeActivation result;
  std::vector<bool> activeArcs(model.arcs_.size(), false);
  for (std::size_t ordinal = 0; ordinal < active.size(); ++ordinal) {
    if (!active[ordinal])
      continue;
    result.fragmentOrdinals_.push_back(static_cast<std::uint32_t>(ordinal));
    const HandshakeActivationFragment &fragment = model.fragments_[ordinal];
    for (std::uint32_t index = 0; index < fragment.contributionCount; ++index) {
      const std::uint32_t arc =
          model.fragmentContributionOrdinals_[fragment.contributionOffset +
                                              index];
      activeArcs[arc] = true;
    }
  }
  for (std::uint32_t arc = 0; arc < activeArcs.size(); ++arc)
    if (activeArcs[arc])
      result.arcOrdinals_.push_back(arc);
  return result;
}

llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveUnconditionalHandshakeDependencyArcs(const FabricArtifactView &view) {
  auto models = compileHandshakeOwnerModels(view);
  if (!models)
    return models.takeError();
  std::vector<HandshakeDependencyArc> result;
  for (const HandshakeOwnerModel &model : *models) {
    std::vector<std::uint32_t> boundaryNodes;
    for (auto [ordinal, node] : llvm::enumerate(model.nodes()))
      if (node.boundarySignal)
        boundaryNodes.push_back(static_cast<std::uint32_t>(ordinal));
    if (boundaryNodes.size() < 2)
      continue;

    std::vector<std::uint32_t> alwaysFragments;
    std::map<std::uint32_t, std::vector<std::uint32_t>> exclusiveGroups;
    for (auto [ordinal, selector] : llvm::enumerate(model.fragmentSelectors_)) {
      const std::uint32_t fragment = static_cast<std::uint32_t>(ordinal);
      if (selector.kind == detail::HandshakeFragmentSelectorKind::Always)
        alwaysFragments.push_back(fragment);
      if (selector.exclusiveGroup)
        exclusiveGroups[*selector.exclusiveGroup].push_back(fragment);
    }
    if (exclusiveGroups.size() > 1)
      return invalid("handshake owner has multiple unnormalized mandatory "
                     "configuration domains");

    const auto reachability =
        [&](llvm::ArrayRef<std::uint32_t> alternativeFragments) {
          std::vector<bool> activeArcs(model.arcs().size(), false);
          const auto activate = [&](std::uint32_t fragment) {
            const HandshakeActivationFragment &record =
                model.fragments()[fragment];
            for (std::uint32_t index = 0; index < record.contributionCount;
                 ++index)
              activeArcs[model.fragmentContributionOrdinals()
                             [record.contributionOffset + index]] = true;
          };
          for (std::uint32_t fragment : alwaysFragments)
            activate(fragment);
          for (std::uint32_t fragment : alternativeFragments)
            activate(fragment);

          std::vector<std::vector<std::uint32_t>> adjacency(
              model.nodes().size());
          for (std::uint32_t arc = 0; arc < activeArcs.size(); ++arc)
            if (activeArcs[arc])
              adjacency[model.arcs()[arc].source].push_back(
                  model.arcs()[arc].destination);

          std::vector<bool> relation(
              boundaryNodes.size() * boundaryNodes.size(), false);
          std::vector<bool> visited(model.nodes().size(), false);
          std::vector<std::uint32_t> worklist;
          worklist.reserve(model.nodes().size());
          for (auto [sourceOrdinal, source] : llvm::enumerate(boundaryNodes)) {
            std::fill(visited.begin(), visited.end(), false);
            worklist.clear();
            worklist.push_back(source);
            visited[source] = true;
            while (!worklist.empty()) {
              const std::uint32_t current = worklist.back();
              worklist.pop_back();
              for (std::uint32_t next : adjacency[current]) {
                if (visited[next])
                  continue;
                visited[next] = true;
                worklist.push_back(next);
              }
            }
            for (auto [destinationOrdinal, destination] :
                 llvm::enumerate(boundaryNodes))
              if (source != destination && visited[destination])
                relation[sourceOrdinal * boundaryNodes.size() +
                         destinationOrdinal] = true;
          }
          return relation;
        };

    std::vector<bool> common;
    if (exclusiveGroups.empty()) {
      common = reachability({});
    } else {
      for (std::uint32_t alternative : exclusiveGroups.begin()->second) {
        const std::uint32_t selected[] = {alternative};
        std::vector<bool> current = reachability(selected);
        if (common.empty()) {
          common = std::move(current);
        } else {
          for (std::size_t edge = 0; edge < common.size(); ++edge)
            common[edge] = common[edge] && current[edge];
        }
      }
    }

    for (auto [sourceOrdinal, source] : llvm::enumerate(boundaryNodes))
      for (auto [destinationOrdinal, destination] :
           llvm::enumerate(boundaryNodes)) {
        if (!common[sourceOrdinal * boundaryNodes.size() + destinationOrdinal])
          continue;
        result.push_back({*model.nodes()[source].boundarySignal,
                          *model.nodes()[destination].boundarySignal});
      }
  }

  auto key = [](const HandshakeDependencyArc &arc) {
    std::vector<std::uint8_t> bytes = signalKey(arc.source);
    const std::vector<std::uint8_t> destination = signalKey(arc.destination);
    bytes.insert(bytes.end(), destination.begin(), destination.end());
    return bytes;
  };
  llvm::sort(result, [&](const auto &lhs, const auto &rhs) {
    return key(lhs) < key(rhs);
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

llvm::Error verifySelectedCombinationalHandshakeAcyclic(
    const FabricArtifactView &view, const FabricHandshakeSelection &selection) {
  auto models = compileHandshakeOwnerModels(view);
  if (!models)
    return models.takeError();

  using OwnerKey = std::vector<std::uint8_t>;
  std::map<OwnerKey, FabricHandshakeSelection> ownerSelections;
  std::set<std::vector<std::uint8_t>> traversalKeys;
  for (const FabricPhysicalTraversalRef &traversal : selection.traversals) {
    if (!traversalKeys.insert(canonicalFabricBytes(traversal)).second)
      return invalid("selected traversal relation contains a duplicate");
    const auto owner = ownerOfTraversal(traversal);
    if (!owner)
      return invalid("selected traversal has no handshake owner");
    ownerSelections[ownerKey(*owner)].traversals.push_back(traversal);
  }
  for (const FabricFuHandshakeSelection &selected : selection.fuCapabilities) {
    auto &local = ownerSelections[ownerKey(
        FabricHandshakeOwner::fu(selected.occurrence()))];
    if (!local.fuCapabilities.empty())
      return invalid("selected FU capability relation repeats one occurrence");
    local.fuCapabilities.push_back(selected);
  }
  std::set<std::vector<std::uint8_t>> memorySelections;
  std::set<std::vector<std::uint8_t>> memoryPlacements;
  for (const FabricMemoryHandshakeSelection &selected :
       selection.memoryOperations) {
    if (!memorySelections.insert(memorySelectionKey(selected)).second)
      return invalid("selected memory operation relation contains a duplicate");
    if (!memoryPlacements.insert(memoryPlacementKey(selected.placement()))
             .second)
      return invalid("selected memory operation placement is contradictory");
    ownerSelections[ownerKey(FabricHandshakeOwner::memory(
                        selected.capability().port.memory))]
        .memoryOperations.push_back(selected);
  }

  std::map<std::vector<std::uint8_t>, std::size_t> boundaryNodes;
  for (const HandshakeOwnerModel &model : *models)
    for (const HandshakeOwnerNode &node : model.nodes())
      if (node.boundarySignal)
        boundaryNodes.try_emplace(signalKey(*node.boundarySignal), 0);
  std::size_t nodeCount = 0;
  for (auto &[key, ordinal] : boundaryNodes) {
    (void)key;
    ordinal = nodeCount++;
  }

  std::vector<std::vector<std::size_t>> modelNodes(models->size());
  for (auto [modelOrdinal, model] : llvm::enumerate(*models)) {
    auto &nodes = modelNodes[modelOrdinal];
    nodes.reserve(model.nodes().size());
    for (const HandshakeOwnerNode &node : model.nodes()) {
      if (node.boundarySignal) {
        auto found = boundaryNodes.find(signalKey(*node.boundarySignal));
        if (found == boundaryNodes.end())
          return invalid("selected handshake boundary node is absent");
        nodes.push_back(found->second);
      } else {
        nodes.push_back(nodeCount++);
      }
    }
  }

  using Arc = std::pair<std::size_t, std::size_t>;
  std::vector<Arc> arcs;
  std::set<OwnerKey> consumedOwners;
  for (auto [modelOrdinal, model] : llvm::enumerate(*models)) {
    const OwnerKey key = ownerKey(model.owner());
    const auto selected = ownerSelections.find(key);
    if (selected != ownerSelections.end())
      consumedOwners.insert(key);

    std::vector<std::uint32_t> activeArcs;
    if (model.owner().kind() == FabricHandshakeOwnerKind::FuOccurrence &&
        (selected == ownerSelections.end() ||
         selected->second.fuCapabilities.empty())) {
      for (const HandshakeActivationFragment &fragment : model.fragments()) {
        if (fragment.activationKind != HandshakeActivationKind::Always)
          continue;
        for (std::uint32_t index = 0; index < fragment.contributionCount;
             ++index)
          activeArcs.push_back(
              model.fragmentContributionOrdinals()[fragment.contributionOffset +
                                                   index]);
      }
    } else {
      static const FabricHandshakeSelection empty;
      auto active = resolveSelectedHandshake(
          model, selected == ownerSelections.end() ? empty : selected->second);
      if (!active)
        return active.takeError();
      activeArcs.assign(active->arcOrdinals().begin(),
                        active->arcOrdinals().end());
    }
    for (std::uint32_t arcOrdinal : activeArcs) {
      if (arcOrdinal >= model.arcs().size())
        return invalid("selected handshake arc is out of range");
      const HandshakeOwnerArc &arc = model.arcs()[arcOrdinal];
      if (arc.source >= modelNodes[modelOrdinal].size() ||
          arc.destination >= modelNodes[modelOrdinal].size())
        return invalid("selected handshake arc endpoint is out of range");
      arcs.emplace_back(modelNodes[modelOrdinal][arc.source],
                        modelNodes[modelOrdinal][arc.destination]);
    }
  }
  if (consumedOwners.size() != ownerSelections.size())
    return invalid("selected handshake relation names a stale owner");
  llvm::sort(arcs);
  arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());

  std::vector<std::size_t> offsets(nodeCount + 1, 0);
  std::vector<std::size_t> indegree(nodeCount, 0);
  for (const Arc &arc : arcs) {
    ++offsets[arc.first + 1];
    ++indegree[arc.second];
  }
  for (std::size_t node = 1; node < offsets.size(); ++node)
    offsets[node] += offsets[node - 1];
  std::vector<std::size_t> destinations(arcs.size());
  std::vector<std::size_t> cursor(offsets.begin(), offsets.end() - 1);
  for (const Arc &arc : arcs)
    destinations[cursor[arc.first]++] = arc.second;

  std::vector<std::size_t> worklist;
  worklist.reserve(nodeCount);
  for (std::size_t node = 0; node < nodeCount; ++node)
    if (indegree[node] == 0)
      worklist.push_back(node);
  std::size_t visited = 0;
  while (!worklist.empty()) {
    const std::size_t node = worklist.back();
    worklist.pop_back();
    ++visited;
    for (std::size_t cursor = offsets[node]; cursor < offsets[node + 1];
         ++cursor) {
      const std::size_t destination = destinations[cursor];
      if (--indegree[destination] == 0)
        worklist.push_back(destination);
    }
  }
  if (visited != nodeCount)
    return invalid("SelectedCombinationalHandshakeCycle");
  return llvm::Error::success();
}

} // namespace loom::fabric
