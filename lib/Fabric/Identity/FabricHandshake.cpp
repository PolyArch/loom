#include "Fabric/Identity/FabricHandshake.h"

#include "FabricHandshakeInternal.h"

#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricMemoryInternalConnection.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace loom::fabric {
namespace {

constexpr llvm::StringLiteral handshakeContextAlgorithmIdentity =
    "loom.fabric.handshake_context.1";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_handshake_invalid: " + message);
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendBytes(std::vector<std::uint8_t> &target,
                 llvm::ArrayRef<std::uint8_t> bytes) {
  appendU32Be(target, static_cast<std::uint32_t>(bytes.size()));
  target.insert(target.end(), bytes.begin(), bytes.end());
}

void appendText(std::vector<std::uint8_t> &target, llvm::StringRef text) {
  appendBytes(target, llvm::ArrayRef<std::uint8_t>(
                          reinterpret_cast<const std::uint8_t *>(text.data()),
                          text.size()));
}

std::array<std::uint8_t, 32>
deriveHandshakeContextKey(const FabricArtifactView &view) {
  std::vector<std::uint8_t> preimage;
  appendText(preimage, handshakeContextAlgorithmIdentity);
  appendText(preimage, fabricArtifactSchema.identity);
  appendU32Be(preimage, fabricArtifactSchema.version.major);
  appendU32Be(preimage, fabricArtifactSchema.version.minor);
  appendU32Be(preimage, static_cast<std::uint32_t>(view.rootKind()));
  appendBytes(preimage, view.identity().bytes());
  return llvm::SHA256::hash(preimage);
}

std::uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point begin) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
}

struct KeyedHandshakeArc final {
  std::vector<std::uint8_t> sourceKey;
  std::vector<std::uint8_t> destinationKey;
  HandshakeDependencyArc arc;
};

std::vector<KeyedHandshakeArc>
keyHandshakeArcs(llvm::ArrayRef<HandshakeDependencyArc> arcs) {
  std::vector<KeyedHandshakeArc> keyed;
  keyed.reserve(arcs.size());
  for (const HandshakeDependencyArc &arc : arcs)
    keyed.push_back({detail::handshakeSignalKey(arc.source),
                     detail::handshakeSignalKey(arc.destination), arc});
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.sourceKey, lhs.destinationKey) <
           std::tie(rhs.sourceKey, rhs.destinationKey);
  });
  return keyed;
}

void sortHandshakeArcs(std::vector<HandshakeDependencyArc> &arcs,
                       bool deduplicate) {
  std::vector<KeyedHandshakeArc> keyed = keyHandshakeArcs(arcs);
  arcs.clear();
  arcs.reserve(keyed.size());
  for (KeyedHandshakeArc &entry : keyed)
    arcs.push_back(std::move(entry.arc));
  if (deduplicate)
    arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());
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

using HandshakeOwnerKey = std::vector<std::uint8_t>;

using MemoryRole = ::dataflow::semantics::ServiceValueRole;

constexpr std::size_t memoryRoleCount =
    static_cast<std::size_t>(MemoryRole::Completion) + 1;

std::optional<FabricHandshakeOwner>
ownerOfTraversal(const FabricPhysicalTraversalRef &traversal);

class HandshakeTraversalIndex final {
public:
  explicit HandshakeTraversalIndex(const FabricArtifactView &view) {
    for (const FabricPhysicalTraversalView &traversal :
         view.physicalTraversals()) {
      const auto owner = ownerOfTraversal(traversal.reference);
      if (owner)
        byOwner_[ownerKey(*owner)].push_back(&traversal);
    }
  }

  llvm::ArrayRef<const FabricPhysicalTraversalView *>
  forOwner(const FabricHandshakeOwner &owner) const {
    const auto found = byOwner_.find(ownerKey(owner));
    if (found == byOwner_.end())
      return {};
    return found->second;
  }

private:
  std::unordered_map<HandshakeOwnerKey,
                     std::vector<const FabricPhysicalTraversalView *>,
                     detail::CanonicalFabricByteKeyHash>
      byOwner_;
};

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
  const auto appendOrdinal = [&](FabricOrdinal value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
      key.push_back(static_cast<std::uint8_t>(value >> (56 - shift)));
  };
  key.push_back(4);
  appendOrdinal(selection.roleSources().size());
  for (const auto &source : selection.roleSources()) {
    if (!source) {
      key.push_back(0);
      continue;
    }
    if (const auto *external =
            std::get_if<FabricMemoryHandshakeExternalRoleSource>(&*source)) {
      key.push_back(1);
      appendOrdinal(external->endpoint);
    } else {
      key.push_back(2);
      appendOrdinal(std::get<FabricMemoryHandshakeInternalRoleSource>(*source)
                        .connection);
    }
  }
  key.push_back(5);
  appendOrdinal(selection.roleDestinations().size());
  for (const auto &destination : selection.roleDestinations()) {
    if (!destination) {
      key.push_back(0);
      continue;
    }
    key.push_back(1);
    key.push_back(destination->externalEndpoint ? 1 : 0);
    if (destination->externalEndpoint)
      appendOrdinal(*destination->externalEndpoint);
    appendOrdinal(destination->internalConnections.size());
    for (FabricOrdinal connection : destination->internalConnections)
      appendOrdinal(connection);
  }
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

struct ActiveMemoryRoles final {
  std::vector<bool> inputs;
  std::vector<bool> outputs;
};

llvm::Expected<ActiveMemoryRoles>
activeMemoryRoles(const MemoryCapabilityAlternativeView &capability,
                  ::dataflow::semantics::MemoryMaskForm maskForm) {
  auto kind = ::dataflow::semantics::getMemoryServiceKind(
      capability.actorContractDomain.actorSchema());
  if (!kind)
    return kind.takeError();
  const auto &schema = ::dataflow::semantics::getServiceRoleSchema(*kind);
  ActiveMemoryRoles result{std::vector<bool>(memoryRoleCount, false),
                           std::vector<bool>(memoryRoleCount, false)};
  const auto add = [&](::dataflow::semantics::ServiceValueRole role,
                       std::vector<bool> &roles) -> llvm::Error {
    if (role == MemoryRole::Mask &&
        maskForm == ::dataflow::semantics::MemoryMaskForm::Absent)
      return llvm::Error::success();
    const std::size_t ordinal = static_cast<std::size_t>(role);
    if (ordinal >= roles.size() || roles[ordinal])
      return invalid("memory service schema repeats an active role");
    roles[ordinal] = true;
    return llvm::Error::success();
  };
  for (MemoryRole role : schema.arguments)
    if (llvm::Error error = add(role, result.inputs))
      return std::move(error);
  for (MemoryRole role : schema.results)
    if (llvm::Error error = add(role, result.outputs))
      return std::move(error);
  return result;
}

const ::fabric::MemoryRoleEndpointBindingRecord *
bindingForRole(const MemoryCapabilityAlternativeView &capability,
               MemoryRole role) {
  const auto found = llvm::find_if(
      capability.roleToEndpoint,
      [&](const ::fabric::MemoryRoleEndpointBindingRecord &candidate) {
        return candidate.role == role;
      });
  return found == capability.roleToEndpoint.end() ? nullptr : &*found;
}

llvm::Error verifyMemoryHandshakeRoles(
    const FabricArtifactView &view, FabricMemoryOperationPortRef port,
    const MemoryCapabilityAlternativeView &capability,
    ::dataflow::semantics::MemoryMaskForm maskForm,
    llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleSource>> sources,
    llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleDestination>>
        destinations) {
  if (sources.size() != memoryRoleCount ||
      destinations.size() != memoryRoleCount)
    return invalid("memory handshake role vectors have the wrong shape");
  auto active = activeMemoryRoles(capability, maskForm);
  if (!active)
    return active.takeError();
  const auto *connectivity = view.memoryConnectivity(port.memory);
  if (!connectivity)
    return invalid("memory handshake occurrence has no connectivity");

  for (std::size_t ordinal = 0; ordinal != memoryRoleCount; ++ordinal) {
    if (sources[ordinal].has_value() != active->inputs[ordinal] ||
        destinations[ordinal].has_value() != active->outputs[ordinal])
      return invalid("memory handshake active roles disagree with capability");
    if (!sources[ordinal] && !destinations[ordinal])
      continue;
    const auto *binding =
        bindingForRole(capability, static_cast<MemoryRole>(ordinal));
    if (!binding)
      return invalid("memory handshake role has no physical endpoint");
    if (sources[ordinal]) {
      if (const auto *external =
              std::get_if<FabricMemoryHandshakeExternalRoleSource>(
                  &*sources[ordinal])) {
        if (external->endpoint != binding->endpointOrdinal)
          return invalid("memory external input selects the wrong endpoint");
      } else {
        const FabricOrdinal connection =
            std::get<FabricMemoryHandshakeInternalRoleSource>(*sources[ordinal])
                .connection;
        if (connection >= connectivity->internalConnections().size() ||
            connectivity->internalConnections()[connection]
                    .sinkEndpointOrdinal != binding->endpointOrdinal)
          return invalid("memory internal input selects an ineligible edge");
      }
    }
    if (!destinations[ordinal])
      continue;
    const auto &destination = *destinations[ordinal];
    if (!destination.externalEndpoint &&
        destination.internalConnections.empty())
      return invalid("memory output has no selected destination");
    if (destination.externalEndpoint &&
        *destination.externalEndpoint != binding->endpointOrdinal)
      return invalid("memory external output selects the wrong endpoint");
    FabricOrdinal previous = 0;
    bool hasPrevious = false;
    for (FabricOrdinal connection : destination.internalConnections) {
      if (connection >= connectivity->internalConnections().size() ||
          connectivity->internalConnections()[connection]
                  .sourceEndpointOrdinal != binding->endpointOrdinal ||
          (hasPrevious && connection <= previous))
        return invalid("memory internal output relation is noncanonical");
      previous = connection;
      hasPrevious = true;
    }
  }
  return llvm::Error::success();
}

} // namespace

std::vector<std::uint8_t>
detail::handshakeOwnerKey(const FabricHandshakeOwner &owner) {
  return ownerKey(owner);
}

std::optional<FabricHandshakeOwner>
detail::handshakeTraversalOwner(const FabricPhysicalTraversalRef &traversal) {
  return ownerOfTraversal(traversal);
}

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

llvm::Expected<FabricMemoryHandshakeSelection> makeMemoryHandshakeSelection(
    const FabricArtifactView &view, FabricMemoryHandshakePlacement placement,
    FabricMemoryCapabilityAlternativeRef capability,
    FabricUsePatternRef usePattern,
    ::dataflow::semantics::MemoryMaskForm maskForm,
    llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleSource>> roleSources,
    llvm::ArrayRef<std::optional<FabricMemoryHandshakeRoleDestination>>
        roleDestinations) {
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
  if (llvm::Error error = verifyMemoryHandshakeRoles(
          view, port, *alternative, maskForm, roleSources, roleDestinations))
    return std::move(error);
  return FabricMemoryHandshakeSelection(
      std::move(placement), capability, usePattern, maskForm,
      std::vector(roleSources.begin(), roleSources.end()),
      std::vector(roleDestinations.begin(), roleDestinations.end()));
}

std::optional<std::uint32_t>
HandshakeOwnerModel::nodeForSignal(const HandshakeSignalRef &signal) const {
  for (std::uint32_t ordinal = 0; ordinal != nodeCount(); ++ordinal) {
    const HandshakeOwnerNode node = this->node(ordinal);
    if (node.boundarySignal && *node.boundarySignal == signal)
      return ordinal;
  }
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

detail::HandshakeFragmentSelector
memorySelector(FabricMemoryCapabilityAlternativeRef capability,
               FabricUsePatternRef usePattern,
               ::dataflow::semantics::MemoryMaskForm maskForm,
               llvm::ArrayRef<MemoryRole> requiredExternalInputs = {},
               llvm::ArrayRef<MemoryRole> requiredExternalOutputs = {}) {
  detail::HandshakeFragmentSelector selector;
  selector.kind = detail::HandshakeFragmentSelectorKind::MemoryOperationPlan;
  selector.memoryCapability = capability;
  selector.memoryUsePattern = usePattern;
  selector.memoryMaskForm = maskForm;
  selector.requiredExternalMemoryInputRoles.assign(
      requiredExternalInputs.begin(), requiredExternalInputs.end());
  selector.requiredExternalMemoryOutputRoles.assign(
      requiredExternalOutputs.begin(), requiredExternalOutputs.end());
  return selector;
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
  (void)view;
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
compilePeModel(const FabricArtifactView &view, FabricPeOccurrenceRef owner,
               llvm::ArrayRef<const FabricPhysicalTraversalView *> traversals) {
  detail::HandshakeOwnerModelBuilder builder(FabricHandshakeOwner::pe(owner));
  const bool temporal = view.peSchedule(owner) == ::fabric::Schedule::Temporal;
  const FabricTransportEndpointOwnerRef peEndpointOwner =
      FabricTransportEndpointOwnerRef::of(owner);
  for (const FabricPhysicalTraversalView *traversalPointer : traversals) {
    const FabricPhysicalTraversalView &traversal = *traversalPointer;
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
    if (temporal && traversal.sources.front().owner == peEndpointOwner) {
      addDirectTraversal(builder, traversalSelector(traversal.reference),
                         traversal.sources.front(),
                         traversal.destinations.front(),
                         /*forwardValid=*/false, /*backwardReady=*/false);
      continue;
    }
    addDirectTraversal(builder, traversalSelector(traversal.reference),
                       traversal.sources.front(),
                       traversal.destinations.front(),
                       /*forwardValid=*/true, /*backwardReady=*/true);
  }
  return builder.finish();
}

struct ActiveMemoryEndpoint final {
  MemoryRole role = MemoryRole::Address;
  FabricTransportEndpointRef endpoint;
};

llvm::Expected<std::vector<ActiveMemoryEndpoint>> activeMemoryEndpoints(
    const FabricArtifactView &view, FabricMemoryOccurrenceRef owner,
    const MemoryCapabilityAlternativeView &capability,
    ::dataflow::semantics::MemoryMaskForm maskForm, bool arguments) {
  auto kind = ::dataflow::semantics::getMemoryServiceKind(
      capability.actorContractDomain.actorSchema());
  if (!kind)
    return kind.takeError();
  const auto &schema = ::dataflow::semantics::getServiceRoleSchema(*kind);
  const auto roles = arguments ? schema.arguments : schema.results;
  std::vector<ActiveMemoryEndpoint> endpoints;
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
    if (llvm::any_of(endpoints, [&](const ActiveMemoryEndpoint &candidate) {
          return candidate.role == role;
        }))
      return invalid("memory service schema repeats an active role");
    endpoints.push_back({role, endpoint});
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

          for (const ActiveMemoryEndpoint &endpoint : *inputs) {
            builder.boundarySignal(
                {endpoint.endpoint, HandshakeSignalKind::Valid});
            builder.boundarySignal(
                {endpoint.endpoint, HandshakeSignalKind::Ready});
          }
          for (const ActiveMemoryEndpoint &endpoint : *outputs) {
            builder.boundarySignal(
                {endpoint.endpoint, HandshakeSignalKind::Valid});
            builder.boundarySignal(
                {endpoint.endpoint, HandshakeSignalKind::Ready});
          }
          builder.addFragment(
              memorySelector(capabilityRef, usePattern, maskForm), {});

          if (*schedule == ::fabric::Schedule::Spatial)
            for (std::size_t target = 0; target != inputs->size(); ++target)
              for (std::size_t driver = 0; driver != inputs->size(); ++driver) {
                if (target == driver)
                  continue;
                const MemoryRole required[] = {(*inputs)[target].role,
                                               (*inputs)[driver].role};
                builder.addFragment(
                    memorySelector(capabilityRef, usePattern, maskForm,
                                   required, {}),
                    {{builder.boundarySignal({(*inputs)[driver].endpoint,
                                              HandshakeSignalKind::Valid}),
                      builder.boundarySignal({(*inputs)[target].endpoint,
                                              HandshakeSignalKind::Ready})}});
              }
          for (std::size_t target = 0; target != outputs->size(); ++target)
            for (std::size_t driver = 0; driver != outputs->size(); ++driver) {
              if (target == driver)
                continue;
              const MemoryRole required[] = {(*outputs)[target].role,
                                             (*outputs)[driver].role};
              builder.addFragment(
                  memorySelector(capabilityRef, usePattern, maskForm, {},
                                 required),
                  {{builder.boundarySignal({(*outputs)[driver].endpoint,
                                            HandshakeSignalKind::Ready}),
                    builder.boundarySignal({(*outputs)[target].endpoint,
                                            HandshakeSignalKind::Valid})}});
            }
        }
      }
    }
  }
  return builder.finish();
}

llvm::Expected<HandshakeOwnerModel> compileSwitchModel(
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
  for (auto &[input, rows] : byInput) {
    llvm::sort(rows, [](const Row &lhs, const Row &rhs) {
      return lhs.output < rhs.output;
    });
    detail::HandshakeOwnerModelBuilder builder(
        FabricHandshakeOwner::switchResource(owner));
    std::vector<FabricPhysicalTraversalRef> witnesses;
    witnesses.reserve(rows.size());
    for (const Row &row : rows)
      witnesses.push_back(row.reference);

    const FabricSwitchHandshakeActivationKey activation{owner, 0, input};
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
  return detail::HandshakeOwnerModelFactory::instantiateSwitchRows(
      owner, rowShapes, residentRows,
      *schedule == ::fabric::Schedule::Temporal);
}

llvm::Expected<HandshakeOwnerModel> compileFifoModel(
    FabricFifoOccurrenceRef owner,
    llvm::ArrayRef<const FabricPhysicalTraversalView *> traversals) {
  detail::HandshakeOwnerModelBuilder builder(FabricHandshakeOwner::fifo(owner));
  for (const FabricPhysicalTraversalView *traversalPointer : traversals) {
    const FabricPhysicalTraversalView &traversal = *traversalPointer;
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
                       payload.mode == FabricFifoTraversalMode::Bypass);
  }
  return builder.finish();
}

llvm::Expected<HandshakeOwnerModel> compileBoundaryModel(
    const FabricArtifactView &view, FabricBoundaryOccurrenceRef owner,
    llvm::ArrayRef<const FabricPhysicalTraversalView *> traversals) {
  detail::HandshakeOwnerModelBuilder builder(
      FabricHandshakeOwner::boundary(owner));
  const auto endpointOwner = FabricTransportEndpointOwnerRef::of(owner);
  const auto inputs =
      directionalEndpoints(view, endpointOwner, FabricPortDirection::Input);
  const auto outputs =
      directionalEndpoints(view, endpointOwner, FabricPortDirection::Output);
  std::vector<FabricPhysicalTraversalRef> witnesses;
  for (const FabricPhysicalTraversalView *traversalView : traversals) {
    const FabricPhysicalTraversalRef &traversal = traversalView->reference;
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

llvm::Expected<HandshakeOwnerModel> compileTransferPatternModel(
    const FabricArtifactView &view, FabricTransferPatternRef owner,
    llvm::ArrayRef<const FabricPhysicalTraversalView *> traversals) {
  auto system = requireSystemRoot(view);
  if (!system)
    return system.takeError();
  const SystemTransferPatternRecord *record = system->transferPattern(owner);
  if (!record)
    return invalid("transfer pattern does not resolve in the System root");
  const ::fabric::ResourceContract *contract =
      view.resourceContract(record->usePattern().owner.catalog());
  if (!contract || record->usePattern().ordinal >= contract->usePatternCount())
    return invalid("transfer pattern has no exact resource use contract");
  const ::fabric::UsePattern use = contract->usePattern(
      ::fabric::UsePatternKey(record->usePattern().ordinal));
  const auto eventOrder = contract->eventOrder(use.timingAndProgress);
  if (use.acquire.ordinal() >= eventOrder.size() ||
      use.release.ordinal() >= eventOrder.size())
    return invalid("transfer pattern has an invalid event order");
  const ClockCrossingContractRecord *crossing =
      system->clockCrossing(owner.resource);
  const bool registered =
      (crossing && crossing->transferPattern() == owner) ||
      eventOrder[use.acquire.ordinal()] != eventOrder[use.release.ordinal()];

  std::vector<const FabricPhysicalTraversalView *> legs;
  for (const FabricPhysicalTraversalView *traversal : traversals) {
    if (traversal->reference.kind() !=
        FabricPhysicalTraversalKind::SystemTransferPatternLeg)
      continue;
    if (std::get<FabricTransferPatternLegPayload>(traversal->reference.payload)
            .owner == owner)
      legs.push_back(traversal);
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
  if (registered) {
    builder.addFragment(allTraversalsSelector(witnesses), {});
    return builder.finish();
  }
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
  const HandshakeTraversalIndex traversalIndex(view);
  std::vector<HandshakeOwnerModel> models;
  std::map<FabricEntityId, HandshakeOwnerModel> fuDefinitions;
  std::map<FabricEntityId, HandshakeOwnerModel> memoryDefinitions;
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
    auto model = compilePeModel(
        view, owner, traversalIndex.forOwner(FabricHandshakeOwner::pe(owner)));
    if (!model)
      return model.takeError();
    if (model->fragmentCount() != 0)
      models.push_back(std::move(*model));
  }
  for (FabricFuOccurrenceRef owner : view.fuOccurrences()) {
    const auto definition = view.fuTemplateOf(owner);
    if (!definition)
      return invalid("FU occurrence has no handshake template relation");
    auto found = fuDefinitions.find(definition->id());
    llvm::Expected<HandshakeOwnerModel> model =
        found == fuDefinitions.end()
            ? detail::compileFuHandshakeModel(view, owner)
            : detail::HandshakeOwnerModelFactory::rebindFuOccurrence(
                  view, found->second, owner);
    if (!model)
      return model.takeError();
    if (found == fuDefinitions.end())
      fuDefinitions.emplace(definition->id(), *model);
    models.push_back(std::move(*model));
  }
  for (FabricMemoryOccurrenceRef owner : view.memoryOccurrences()) {
    if (view.memoryOperationPorts(owner).empty())
      continue;
    const auto definition = view.memoryEngineTemplateOf(owner);
    if (!definition)
      return invalid("Memory occurrence has no handshake template relation");
    auto found = memoryDefinitions.find(definition->id());
    llvm::Expected<HandshakeOwnerModel> model =
        found == memoryDefinitions.end()
            ? compileMemoryModel(view, owner)
            : detail::HandshakeOwnerModelFactory::rebindMemoryOccurrence(
                  view, found->second, owner);
    if (!model)
      return model.takeError();
    if (found == memoryDefinitions.end())
      memoryDefinitions.emplace(definition->id(), *model);
    models.push_back(std::move(*model));
  }
  for (FabricSwitchOccurrenceRef owner : view.switchOccurrences()) {
    auto model = compileSwitchModel(
        view, owner,
        traversalIndex.forOwner(FabricHandshakeOwner::switchResource(owner)));
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  for (FabricFifoOccurrenceRef owner : view.fifoOccurrences()) {
    auto model = compileFifoModel(
        owner, traversalIndex.forOwner(FabricHandshakeOwner::fifo(owner)));
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  for (FabricBoundaryOccurrenceRef owner : view.boundaryOccurrences()) {
    auto model = compileBoundaryModel(
        view, owner,
        traversalIndex.forOwner(FabricHandshakeOwner::boundary(owner)));
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
    auto model = compileTransferPatternModel(
        view, owner,
        traversalIndex.forOwner(FabricHandshakeOwner::transferPattern(owner)));
    if (!model)
      return model.takeError();
    models.push_back(std::move(*model));
  }
  return models;
}

llvm::Expected<FabricHandshakeContext>
buildFabricHandshakeContext(const FabricArtifactView &view) {
  const auto begin = std::chrono::steady_clock::now();
  auto models = compileHandshakeOwnerModels(view);
  if (!models)
    return models.takeError();
  auto unconditional =
      detail::HandshakeOwnerModelFactory::deriveUnconditionalDependencyArcs(
          *models);
  if (!unconditional)
    return unconditional.takeError();

  FabricHandshakeContextStatistics statistics;
  detail::HandshakeOwnerModelFactory::accumulateStatistics(*models, statistics);
  statistics.unconditionalArcCount = unconditional->size();
  statistics.retainedBytes +=
      unconditional->size() * sizeof(HandshakeDependencyArc);
  statistics.deterministicWork += statistics.unconditionalArcCount;
  statistics.constructionNanoseconds = elapsedNanoseconds(begin);
  auto owner = std::make_shared<const std::vector<HandshakeOwnerModel>>(
      std::move(*models));
  auto unconditionalOwner =
      std::make_shared<const std::vector<HandshakeDependencyArc>>(
          std::move(*unconditional));
  return FabricHandshakeContext(
      view.identity(), deriveHandshakeContextKey(view), std::move(owner),
      std::move(unconditionalOwner), statistics);
}

llvm::Error
revalidateFabricHandshakeContext(const FabricHandshakeContext &context,
                                 const FabricArtifactView &view) {
  if (context.fabricIdentity() != view.identity() ||
      context.key() != deriveHandshakeContextKey(view))
    return invalid("handshake context binds another Fabric or algorithm");
  FabricHandshakeContextStatistics expected;
  detail::HandshakeOwnerModelFactory::accumulateStatistics(
      context.ownerModels(), expected);
  expected.unconditionalArcCount = context.unconditionalDependencyArcs().size();
  expected.retainedBytes += context.unconditionalDependencyArcs().size() *
                            sizeof(HandshakeDependencyArc);
  expected.deterministicWork += expected.unconditionalArcCount;
  const FabricHandshakeContextStatistics &observed = context.statistics();
  if (observed.retainedBytes != expected.retainedBytes ||
      observed.deterministicWork != expected.deterministicWork ||
      observed.ownerCount != expected.ownerCount ||
      observed.structuralTemplateCount != expected.structuralTemplateCount ||
      observed.bindingInstanceCount != expected.bindingInstanceCount ||
      observed.structuralNodeCount != expected.structuralNodeCount ||
      observed.structuralArcCount != expected.structuralArcCount ||
      observed.structuralFragmentCount != expected.structuralFragmentCount ||
      observed.unconditionalArcCount != expected.unconditionalArcCount ||
      observed.nodeCount != expected.nodeCount ||
      observed.arcCount != expected.arcCount ||
      observed.fragmentCount != expected.fragmentCount)
    return invalid("handshake context statistics are inconsistent");
  return llvm::Error::success();
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

bool memoryRoleIsExternal(const FabricMemoryHandshakeSelection &selection,
                          MemoryRole role, bool input) {
  const std::size_t ordinal = static_cast<std::size_t>(role);
  if (input) {
    if (ordinal >= selection.roleSources().size() ||
        !selection.roleSources()[ordinal])
      return false;
    return std::holds_alternative<FabricMemoryHandshakeExternalRoleSource>(
        *selection.roleSources()[ordinal]);
  }
  return ordinal < selection.roleDestinations().size() &&
         selection.roleDestinations()[ordinal] &&
         selection.roleDestinations()[ordinal]->externalEndpoint.has_value();
}

bool memoryFragmentRolesMatch(const detail::HandshakeFragmentSelector &fragment,
                              const FabricMemoryHandshakeSelection &selection) {
  return llvm::all_of(fragment.requiredExternalMemoryInputRoles,
                      [&](MemoryRole role) {
                        return memoryRoleIsExternal(selection, role, true);
                      }) &&
         llvm::all_of(fragment.requiredExternalMemoryOutputRoles,
                      [&](MemoryRole role) {
                        return memoryRoleIsExternal(selection, role, false);
                      });
}

llvm::Error verifyMemoryInternalConnectionClosure(
    llvm::ArrayRef<FabricMemoryHandshakeSelection> selections) {
  std::vector<FabricMemoryInternalConnectionUse> uses;
  for (const FabricMemoryHandshakeSelection &selection : selections) {
    const auto occurrence = selection.capability().port.memory;
    for (const auto &source : selection.roleSources()) {
      if (!source)
        continue;
      const auto *internal =
          std::get_if<FabricMemoryHandshakeInternalRoleSource>(&*source);
      if (!internal)
        continue;
      uses.push_back({occurrence, internal->connection,
                      FabricMemoryInternalConnectionUseKind::Consumer});
    }
    for (const auto &destination : selection.roleDestinations()) {
      if (!destination)
        continue;
      for (FabricOrdinal connection : destination->internalConnections)
        uses.push_back({occurrence, connection,
                        FabricMemoryInternalConnectionUseKind::Producer});
    }
  }
  switch (deriveFabricMemoryInternalConnectionClosure(uses)) {
  case FabricMemoryInternalConnectionClosure::Closed:
    return llvm::Error::success();
  case FabricMemoryInternalConnectionClosure::Open:
    return invalid("memory internal connection is not closed");
  case FabricMemoryInternalConnectionClosure::MultipleProducers:
    return invalid("memory internal connection has multiple producers");
  }
  llvm_unreachable("closed memory connection closure domain");
}

} // namespace

void detail::sortHandshakeDependencyArcs(
    std::vector<HandshakeDependencyArc> &arcs, bool deduplicate) {
  sortHandshakeArcs(arcs, deduplicate);
}

llvm::Error detail::verifyMemoryInternalHandshakeClosure(
    llvm::ArrayRef<FabricMemoryHandshakeSelection> selections) {
  return verifyMemoryInternalConnectionClosure(selections);
}

llvm::Expected<ResolvedHandshakeActivation>
resolveSelectedHandshake(const HandshakeOwnerModel &model,
                         const FabricHandshakeSelection &selection) {
  std::set<std::vector<std::uint8_t>> traversalKeys;
  for (const FabricPhysicalTraversalRef &traversal : selection.traversals)
    if (!traversalKeys.insert(canonicalFabricBytes(traversal)).second)
      return invalid("selected traversal relation contains a duplicate");
  using SwitchActivationKey =
      std::tuple<std::vector<std::uint8_t>, FabricOrdinal, FabricOrdinal>;
  std::set<SwitchActivationKey> switchActivationKeys;
  std::vector<std::vector<bool>> selectedSwitchTraversalConsumed;
  selectedSwitchTraversalConsumed.reserve(selection.switchActivations.size());
  for (const FabricSwitchHandshakeActivationSelection &activation :
       selection.switchActivations) {
    if (activation.traversals.empty())
      return invalid("selected switch activation has no traversal");
    if (!switchActivationKeys
             .emplace(canonicalFabricBytes(activation.key.occurrence),
                      activation.key.row, activation.key.input)
             .second)
      return invalid("selected switch activation relation has a duplicate");
    std::set<std::vector<std::uint8_t>> members;
    for (const FabricPhysicalTraversalRef &traversal : activation.traversals)
      if (!members.insert(canonicalFabricBytes(traversal)).second)
        return invalid("selected switch activation repeats a traversal");
    selectedSwitchTraversalConsumed.emplace_back(activation.traversals.size(),
                                                 false);
  }
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

  std::vector<bool> active(model.fragmentCount(), false);
  std::vector<bool> selectedTraversalConsumed(selection.traversals.size(),
                                              false);
  std::vector<bool> selectedMemoryConsumed(selection.memoryOperations.size(),
                                           false);
  const FabricFuHandshakeSelection *selectedFu = nullptr;
  if (model.owner().kind() == FabricHandshakeOwnerKind::FuOccurrence) {
    const FabricFuOccurrenceRef owner =
        std::get<FabricFuOccurrenceRef>(model.owner().payload());
    for (const FabricFuHandshakeSelection &candidate : selection.fuCapabilities)
      if (candidate.occurrence() == owner)
        selectedFu = &candidate;
  }
  std::map<std::uint32_t, std::uint32_t> selectedExclusiveGroups;
  for (std::uint32_t ordinal = 0; ordinal != model.fragmentCount(); ++ordinal) {
    const detail::HandshakeFragmentSelector selector =
        model.fragmentSelector(ordinal);
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
    case detail::HandshakeFragmentSelectorKind::AnySwitchActivationTraversal:
    case detail::HandshakeFragmentSelectorKind::ExactSwitchActivationTraversal:
      if (!selector.switchActivation)
        return invalid("switch handshake fragment has no activation key");
      for (std::size_t activationOrdinal = 0;
           activationOrdinal < selection.switchActivations.size();
           ++activationOrdinal) {
        const auto &activation = selection.switchActivations[activationOrdinal];
        if (activation.key != *selector.switchActivation)
          continue;
        for (std::size_t traversalOrdinal = 0;
             traversalOrdinal < activation.traversals.size();
             ++traversalOrdinal) {
          if (!containsTraversal(selector.traversalWitnesses,
                                 activation.traversals[traversalOrdinal]))
            continue;
          selected = true;
          selectedSwitchTraversalConsumed[activationOrdinal][traversalOrdinal] =
              true;
        }
      }
      break;
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
            candidate.maskForm() != *selector.memoryMaskForm ||
            !memoryFragmentRolesMatch(selector, candidate))
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
    if (owner && *owner == model.owner() && !selectedTraversalConsumed[ordinal])
      return invalid("selected traversal is stale for its handshake owner");
  }
  if (model.owner().kind() == FabricHandshakeOwnerKind::SwitchOccurrence) {
    const FabricSwitchOccurrenceRef owner =
        std::get<FabricSwitchOccurrenceRef>(model.owner().payload());
    for (std::size_t activationOrdinal = 0;
         activationOrdinal < selection.switchActivations.size();
         ++activationOrdinal) {
      const auto &activation = selection.switchActivations[activationOrdinal];
      if (activation.key.occurrence != owner)
        continue;
      if (llvm::any_of(selectedSwitchTraversalConsumed[activationOrdinal],
                       [](bool consumed) { return !consumed; }))
        return invalid(
            "selected switch activation is stale for its handshake owner");
    }
  }
  if (model.owner().kind() == FabricHandshakeOwnerKind::FuOccurrence) {
    const auto owner = std::get<FabricFuOccurrenceRef>(model.owner().payload());
    bool hasSelection = false;
    for (const FabricFuHandshakeSelection &selected :
         selection.fuCapabilities) {
      if (selected.occurrence() != owner)
        continue;
      hasSelection = true;
      bool hasCapability = false;
      for (std::uint32_t fragment = 0; fragment != model.fragmentCount();
           ++fragment) {
        const detail::HandshakeFragmentSelector selector =
            model.fragmentSelector(fragment);
        hasCapability =
            selector.kind ==
                detail::HandshakeFragmentSelectorKind::FuCapability &&
            selector.fuOccurrence && selector.fuCapability &&
            *selector.fuOccurrence == selected.occurrence() &&
            *selector.fuCapability == selected.capability();
        if (hasCapability)
          break;
      }
      if (!hasCapability)
        return invalid("selected FU capability is stale for its occurrence");
      for (const FabricFuOperationHandshakeSelection &operation :
           selected.operations()) {
        bool hasOperation = false;
        for (std::uint32_t fragment = 0; fragment != model.fragmentCount();
             ++fragment) {
          const detail::HandshakeFragmentSelector selector =
              model.fragmentSelector(fragment);
          hasOperation =
              selector.fuOperation &&
              selector.fuOperation->operation == operation.operation &&
              selector.fuOperation->schema == operation.schema;
          if (hasOperation)
            break;
        }
        if (!hasOperation)
          return invalid("selected FU operation is stale for its occurrence");
      }
    }
    if (!hasSelection && model.fragmentCount() != 0)
      return invalid("FU occurrence has no selected capability");
  }
  if (model.owner().kind() == FabricHandshakeOwnerKind::MemoryOccurrence) {
    const auto owner =
        std::get<FabricMemoryOccurrenceRef>(model.owner().payload());
    for (std::size_t ordinal = 0; ordinal < selection.memoryOperations.size();
         ++ordinal) {
      if (selection.memoryOperations[ordinal].capability().port.memory != owner)
        continue;
      if (!selectedMemoryConsumed[ordinal])
        return invalid("selected memory operation plan is stale for its owner");
    }
  }

  ResolvedHandshakeActivation result;
  std::vector<bool> activeArcs(model.arcCount(), false);
  for (std::size_t ordinal = 0; ordinal < active.size(); ++ordinal) {
    if (!active[ordinal])
      continue;
    result.fragmentOrdinals_.push_back(static_cast<std::uint32_t>(ordinal));
    const HandshakeActivationFragment fragment =
        model.fragment(static_cast<std::uint32_t>(ordinal));
    for (std::uint32_t index = 0; index < fragment.contributionCount; ++index) {
      const std::uint32_t arc = model.fragmentContributionOrdinal(
          fragment.contributionOffset + index);
      activeArcs[arc] = true;
    }
  }
  for (std::uint32_t arc = 0; arc < activeArcs.size(); ++arc)
    if (activeArcs[arc])
      result.arcOrdinals_.push_back(arc);
  return result;
}

llvm::Expected<std::vector<HandshakeDependencyArc>>
detail::HandshakeOwnerModelFactory::deriveUnconditionalDependencyArcs(
    llvm::ArrayRef<HandshakeOwnerModel> models) {
  std::vector<HandshakeDependencyArc> result;
  for (const HandshakeOwnerModel &model : models) {
    struct ShapeReachability final {
      const detail::HandshakeOwnerModelInstance *definition = nullptr;
      std::vector<std::uint32_t> boundaryNodes;
      std::vector<bool> relation;
    };
    std::vector<std::optional<ShapeReachability>> shapes;
    for (const detail::HandshakeOwnerModelInstance &instance :
         model.storage_->instances) {
      if (instance.projectionShapeOrdinal >= shapes.size())
        shapes.resize(instance.projectionShapeOrdinal + 1);
      auto &shape = shapes[instance.projectionShapeOrdinal];
      if (!shape) {
        ShapeReachability derived;
        derived.definition = &instance;
        for (auto [ordinal, node] : llvm::enumerate(*instance.nodeBindings))
          if (node.boundarySignal)
            derived.boundaryNodes.push_back(
                static_cast<std::uint32_t>(ordinal));

        if (derived.boundaryNodes.size() >= 2) {
          std::vector<std::uint32_t> alwaysFragments;
          std::map<std::uint32_t, std::vector<std::uint32_t>> exclusiveGroups;
          for (auto [fragment, selector] :
               llvm::enumerate(*instance.selectors)) {
            if (selector.kind == detail::HandshakeFragmentSelectorKind::Always)
              alwaysFragments.push_back(static_cast<std::uint32_t>(fragment));
            if (selector.exclusiveGroup)
              exclusiveGroups[*selector.exclusiveGroup].push_back(
                  static_cast<std::uint32_t>(fragment));
          }
          if (exclusiveGroups.size() > 1)
            return invalid(
                "handshake owner has multiple unnormalized mandatory "
                "configuration domains");

          const auto reachability =
              [&](llvm::ArrayRef<std::uint32_t> alternatives) {
                std::vector<bool> activeArcs(instance.structure->arcs.size(),
                                             false);
                const auto activate = [&](std::uint32_t fragment) {
                  const detail::HandshakeStructuralFragment &record =
                      instance.structure->fragments[fragment];
                  for (std::uint32_t index = 0;
                       index != record.contributionCount; ++index)
                    activeArcs[instance.structure->fragmentContributionOrdinals
                                   [record.contributionOffset + index]] = true;
                };
                for (std::uint32_t fragment : alwaysFragments)
                  activate(fragment);
                for (std::uint32_t fragment : alternatives)
                  activate(fragment);

                std::vector<std::vector<std::uint32_t>> adjacency(
                    instance.structure->nodeKinds.size());
                for (auto [arcOrdinal, active] : llvm::enumerate(activeArcs))
                  if (active) {
                    const HandshakeOwnerArc arc =
                        instance.structure->arcs[arcOrdinal];
                    adjacency[arc.source].push_back(arc.destination);
                  }

                std::vector<bool> relation(derived.boundaryNodes.size() *
                                               derived.boundaryNodes.size(),
                                           false);
                std::vector<bool> visited(instance.structure->nodeKinds.size(),
                                          false);
                std::vector<std::uint32_t> worklist;
                worklist.reserve(instance.structure->nodeKinds.size());
                for (auto [sourceOrdinal, source] :
                     llvm::enumerate(derived.boundaryNodes)) {
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
                       llvm::enumerate(derived.boundaryNodes))
                    if (source != destination && visited[destination])
                      relation[sourceOrdinal * derived.boundaryNodes.size() +
                               destinationOrdinal] = true;
                }
                return relation;
              };

          if (exclusiveGroups.empty()) {
            derived.relation = reachability({});
          } else {
            for (std::uint32_t alternative : exclusiveGroups.begin()->second) {
              const std::uint32_t selected[] = {alternative};
              std::vector<bool> current = reachability(selected);
              if (derived.relation.empty()) {
                derived.relation = std::move(current);
              } else {
                for (std::size_t edge = 0; edge < derived.relation.size();
                     ++edge)
                  derived.relation[edge] =
                      derived.relation[edge] && current[edge];
              }
            }
          }
        }
        shape = std::move(derived);
      }

      if (shape->definition->structure != instance.structure ||
          shape->definition->nodeBindings != instance.nodeBindings ||
          shape->definition->selectors != instance.selectors)
        return invalid("handshake projection shape is inconsistent");
      for (auto [sourceOrdinal, source] : llvm::enumerate(shape->boundaryNodes))
        for (auto [destinationOrdinal, destination] :
             llvm::enumerate(shape->boundaryNodes)) {
          if (!shape->relation[sourceOrdinal * shape->boundaryNodes.size() +
                               destinationOrdinal])
            continue;
          result.push_back(
              {*(*instance.nodeBindings)[source].boundarySignal,
               *(*instance.nodeBindings)[destination].boundarySignal});
        }
    }
  }

  sortHandshakeArcs(result, true);
  return result;
}

llvm::Expected<std::vector<HandshakeDependencyArc>>
deriveUnconditionalHandshakeDependencyArcs(const FabricArtifactView &view) {
  auto context = buildFabricHandshakeContext(view);
  if (!context)
    return context.takeError();
  return std::vector<HandshakeDependencyArc>(
      context->unconditionalDependencyArcs().begin(),
      context->unconditionalDependencyArcs().end());
}

} // namespace loom::fabric
