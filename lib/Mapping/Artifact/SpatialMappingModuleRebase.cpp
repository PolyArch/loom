#include "SpatialMappingModuleRebase.h"

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeFabric(Attr attribute) {
  return ::loom::fabric::decodeFabricRef<Ref>(
      unsignedBytes(attribute.getRecord()));
}

/// Module-local correspondence for a child whose canonical labels may have
/// changed after a typed microarchitecture edit. Mapping decisions retain
/// owner-relative ordinals, while entity IDs are remapped by the canonical
/// occurrence inventories of the parent and child. This is intentionally
/// narrower than the System remapper: System entities and transfer patterns
/// are not legal Module-local route owners.
class ModuleReferenceRemapper final {
public:
  static llvm::Expected<ModuleReferenceRemapper>
  get(const ::loom::fabric::FabricArtifactView &parent,
      const ::loom::fabric::FabricArtifactView &child,
      llvm::ArrayRef<::loom::fabric::FabricModuleEntityCorrespondence>
          correspondence = {}) {
    ModuleReferenceRemapper result(parent, child);
    if (llvm::Error error =
            result.add(parent.peOccurrences(), child.peOccurrences(),
                       ::loom::fabric::FabricEntityKind::FabricPeOccurrence,
                       correspondence, result.pes_))
      return std::move(error);
    if (llvm::Error error =
            result.add(parent.fuOccurrences(), child.fuOccurrences(),
                       ::loom::fabric::FabricEntityKind::FabricFuOccurrence,
                       correspondence, result.fus_))
      return std::move(error);
    if (llvm::Error error =
            result.add(parent.memoryOccurrences(), child.memoryOccurrences(),
                       ::loom::fabric::FabricEntityKind::FabricMemoryOccurrence,
                       correspondence, result.memories_))
      return std::move(error);
    if (llvm::Error error =
            result.add(parent.switchOccurrences(), child.switchOccurrences(),
                       ::loom::fabric::FabricEntityKind::FabricSwitchOccurrence,
                       correspondence, result.switches_))
      return std::move(error);
    if (llvm::Error error =
            result.add(parent.fifoOccurrences(), child.fifoOccurrences(),
                       ::loom::fabric::FabricEntityKind::FabricFifoOccurrence,
                       correspondence, result.fifos_))
      return std::move(error);
    if (llvm::Error error = result.add(
            parent.boundaryOccurrences(), child.boundaryOccurrences(),
            ::loom::fabric::FabricEntityKind::FabricBoundaryOccurrence,
            correspondence, result.boundaries_))
      return std::move(error);
    return result;
  }

  llvm::Expected<::loom::fabric::FabricPeOccurrenceRef>
  remap(::loom::fabric::FabricPeOccurrenceRef ref) const {
    return lookup(pes_, ref, "PE occurrence");
  }
  llvm::Expected<::loom::fabric::FabricFuOccurrenceRef>
  remap(::loom::fabric::FabricFuOccurrenceRef ref) const {
    return lookup(fus_, ref, "FU occurrence");
  }
  llvm::Expected<::loom::fabric::FabricMemoryOccurrenceRef>
  remap(::loom::fabric::FabricMemoryOccurrenceRef ref) const {
    return lookup(memories_, ref, "Memory occurrence");
  }
  llvm::Expected<::loom::fabric::FabricSwitchOccurrenceRef>
  remap(::loom::fabric::FabricSwitchOccurrenceRef ref) const {
    return lookup(switches_, ref, "Switch occurrence");
  }
  llvm::Expected<::loom::fabric::FabricFifoOccurrenceRef>
  remap(::loom::fabric::FabricFifoOccurrenceRef ref) const {
    return lookup(fifos_, ref, "FIFO occurrence");
  }
  llvm::Expected<::loom::fabric::FabricBoundaryOccurrenceRef>
  remap(::loom::fabric::FabricBoundaryOccurrenceRef ref) const {
    return lookup(boundaries_, ref, "Boundary occurrence");
  }

  llvm::Expected<::loom::fabric::FabricTransportEndpointOwnerRef>
  remap(const ::loom::fabric::FabricTransportEndpointOwnerRef &owner) const {
    return std::visit(
        [&](const auto &value)
            -> llvm::Expected<::loom::fabric::FabricTransportEndpointOwnerRef> {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (
              std::is_same_v<Value, ::loom::fabric::SpatialCoreOccurrenceRef> ||
              std::is_same_v<Value, ::loom::fabric::SystemServiceEndpointRef> ||
              std::is_same_v<Value, ::loom::fabric::SystemTransportResourceRef>)
            return invalid("Module route names a non-local transport owner");
          else {
            auto mapped = remap(value);
            if (!mapped)
              return mapped.takeError();
            return ::loom::fabric::FabricTransportEndpointOwnerRef::of(*mapped);
          }
        },
        owner.payload);
  }

  llvm::Expected<::loom::fabric::FabricTransportEndpointRef>
  remap(const ::loom::fabric::FabricTransportEndpointRef &endpoint) const {
    auto owner = remap(endpoint.owner);
    if (!owner)
      return owner.takeError();
    auto result =
        ::loom::fabric::FabricTransportEndpointRef{*owner, endpoint.ordinal};
    if (!::loom::fabric::validateFabricRef(child_, result))
      return result;
    const auto direction = parent_.transportEndpointDirection(endpoint);
    const auto type = parent_.transportEndpointType(endpoint);
    std::vector<::loom::fabric::FabricTransportEndpointRef> matches;
    for (const auto &candidate : child_.transportEndpoints()) {
      if (!(candidate.owner == *owner))
        continue;
      if (direction &&
          child_.transportEndpointDirection(candidate) != direction)
        continue;
      if (!type.equals(child_.transportEndpointType(candidate)))
        continue;
      matches.push_back(candidate);
    }
    if (matches.size() == 1)
      return matches.front();
    return invalid(matches.empty()
                       ? "Module child lost a remapped transport endpoint"
                       : "Module child has ambiguous remapped transport "
                         "endpoint");
  }

  llvm::Expected<::loom::fabric::FabricPhysicalTraversalRef>
  remap(const ::loom::fabric::FabricPhysicalTraversalRef &traversal) const {
    return std::visit(
        [&](const auto &payload)
            -> llvm::Expected<::loom::fabric::FabricPhysicalTraversalRef> {
          using Payload = std::decay_t<decltype(payload)>;
          if constexpr (std::is_same_v<
                            Payload,
                            ::loom::fabric::FabricPointConnectionPayload>) {
            auto source = remap(payload.source);
            if (!source)
              return source.takeError();
            auto destination = remap(payload.destination);
            if (!destination)
              return destination.takeError();
            auto result =
                ::loom::fabric::FabricPhysicalTraversalRef::pointConnection(
                    *source, *destination);
            return remapAdmittedTraversal(result, *source, *destination);
          } else if constexpr (std::is_same_v<
                                   Payload,
                                   ::loom::fabric::FabricPeSelectorPayload>) {
            auto owner = remap(payload.owner);
            if (!owner)
              return owner.takeError();
            auto source = remap(payload.source);
            if (!source)
              return source.takeError();
            auto destination = remap(payload.destination);
            if (!destination)
              return destination.takeError();
            auto result =
                ::loom::fabric::FabricPhysicalTraversalRef::peSelector(
                    *owner, *source, *destination);
            return remapAdmittedTraversal(result, *source, *destination);
          } else if constexpr (std::is_same_v<
                                   Payload, ::loom::fabric::
                                                FabricPeRegisterFifoPayload>) {
            auto owner = remap(payload.owner);
            if (!owner)
              return owner.takeError();
            auto result =
                ::loom::fabric::FabricPhysicalTraversalRef::peRegisterFifo(
                    *owner, payload.registerFifo, payload.role);
            return remapAdmittedTraversal(result, std::nullopt, std::nullopt);
          } else if constexpr (std::is_same_v<
                                   Payload, ::loom::fabric::
                                                FabricSwitchTraversalPayload>) {
            auto owner = remap(payload.owner);
            if (!owner)
              return owner.takeError();
            auto result =
                ::loom::fabric::FabricPhysicalTraversalRef::switchTraversal(
                    *owner, payload.input, payload.output);
            return remapAdmittedTraversal(result, std::nullopt, std::nullopt);
          } else if constexpr (std::is_same_v<Payload,
                                              ::loom::fabric::
                                                  FabricFifoTraversalPayload>) {
            auto owner = remap(payload.owner);
            if (!owner)
              return owner.takeError();
            auto result =
                ::loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
                    *owner, payload.mode);
            return remapAdmittedTraversal(result, std::nullopt, std::nullopt);
          } else if constexpr (
              std::is_same_v<Payload,
                             ::loom::fabric::FabricBoundaryTraversalPayload>) {
            auto owner = remap(payload.owner);
            if (!owner)
              return owner.takeError();
            auto result =
                ::loom::fabric::FabricPhysicalTraversalRef::boundaryTraversal(
                    *owner, payload.output);
            return remapAdmittedTraversal(result, std::nullopt, std::nullopt);
          } else {
            return invalid("Module route names a System transfer pattern");
          }
        },
        traversal.payload);
  }

private:
  template <typename Ref> using Map = std::map<std::vector<std::uint8_t>, Ref>;

  ModuleReferenceRemapper(const ::loom::fabric::FabricArtifactView &parent,
                          const ::loom::fabric::FabricArtifactView &child)
      : parent_(parent), child_(child) {}

  llvm::Expected<::loom::fabric::FabricPhysicalTraversalRef>
  remapAdmittedTraversal(
      const ::loom::fabric::FabricPhysicalTraversalRef &candidate,
      std::optional<::loom::fabric::FabricTransportEndpointRef> source,
      std::optional<::loom::fabric::FabricTransportEndpointRef> destination)
      const {
    if (child_.physicalTraversal(candidate))
      return candidate;
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> matches;
    for (const auto &view : child_.physicalTraversals()) {
      if (view.reference.kind() != candidate.kind())
        continue;
      if (source && !llvm::is_contained(view.sources, *source))
        continue;
      if (destination && !llvm::is_contained(view.destinations, *destination))
        continue;
      matches.push_back(view.reference);
    }
    if (matches.size() == 1)
      return matches.front();

    // A local Module edit can renumber endpoint identities even when the
    // owner, direction, and physical type are unchanged. Preserve the
    // traversal only when that structural correspondence is unique; an
    // ambiguous or semantically changed traversal remains a typed cold
    // fallback instead of guessing by ordinal.
    if (source || destination) {
      const auto endpointCompatible = [&](const auto &actual,
                                          const auto &expected) {
        return actual.owner == expected.owner &&
               child_.transportEndpointDirection(actual) ==
                   parent_.transportEndpointDirection(expected) &&
               child_.transportEndpointType(actual) ==
                   parent_.transportEndpointType(expected);
      };
      std::vector<::loom::fabric::FabricPhysicalTraversalRef> structuralMatches;
      for (const auto &view : child_.physicalTraversals()) {
        if (view.reference.kind() != candidate.kind())
          continue;
        const bool sourceMatch =
            !source || llvm::any_of(view.sources, [&](const auto &actual) {
              return endpointCompatible(actual, *source);
            });
        const bool destinationMatch =
            !destination ||
            llvm::any_of(view.destinations, [&](const auto &actual) {
              return endpointCompatible(actual, *destination);
            });
        if (sourceMatch && destinationMatch)
          structuralMatches.push_back(view.reference);
      }
      if (structuralMatches.size() == 1)
        return structuralMatches.front();
    }
    std::size_t sourceOwnerMatches = 0;
    std::size_t destinationOwnerMatches = 0;
    if (source || destination) {
      for (const auto &view : child_.physicalTraversals()) {
        if (view.reference.kind() != candidate.kind())
          continue;
        if (source && llvm::any_of(view.sources, [&](const auto &actual) {
              return actual.owner == source->owner;
            }))
          ++sourceOwnerMatches;
        if (destination &&
            llvm::any_of(view.destinations, [&](const auto &actual) {
              return actual.owner == destination->owner;
            }))
          ++destinationOwnerMatches;
      }
    }
    return invalid(
        llvm::Twine(matches.empty() ? "Module child lost a remapped physical "
                                      "traversal"
                                    : "Module child has ambiguous remapped "
                                      "physical traversal") +
        "; kind=" + llvm::Twine(static_cast<std::uint32_t>(candidate.kind())) +
        "; child_traversal_count=" +
        llvm::Twine(child_.physicalTraversals().size()) +
        "; source_present=" + llvm::Twine(source.has_value()) +
        "; destination_present=" + llvm::Twine(destination.has_value()) +
        "; source_owner_matches=" + llvm::Twine(sourceOwnerMatches) +
        "; destination_owner_matches=" + llvm::Twine(destinationOwnerMatches));
  }

  template <typename Ref>
  llvm::Error
  add(llvm::ArrayRef<Ref> parent, llvm::ArrayRef<Ref> child,
      ::loom::fabric::FabricEntityKind kind,
      llvm::ArrayRef<::loom::fabric::FabricModuleEntityCorrespondence>
          correspondence,
      Map<Ref> &mapping) {
    if (!correspondence.empty())
      return addByCorrespondence(parent, child, kind, correspondence, mapping);
    if (parent.size() != child.size())
      return invalid("Module occurrence inventory changed during local rebase");
    for (std::size_t index = 0; index != parent.size(); ++index)
      if (!mapping
               .emplace(::loom::fabric::canonicalFabricBytes(parent[index]),
                        child[index])
               .second)
        return invalid("Module occurrence correspondence is not unique");
    return llvm::Error::success();
  }

  template <typename Ref>
  llvm::Error addByCorrespondence(
      llvm::ArrayRef<Ref> parent, llvm::ArrayRef<Ref> child,
      ::loom::fabric::FabricEntityKind kind,
      llvm::ArrayRef<::loom::fabric::FabricModuleEntityCorrespondence>
          correspondence,
      Map<Ref> &mapping) {
    using Entry = ::loom::fabric::FabricModuleEntityCorrespondence;
    std::map<std::uint64_t, const Entry *> bySourceOrdinal;
    std::set<std::uint64_t> targetOrdinals;
    for (const Entry &entry : correspondence) {
      if (entry.source.kind != kind)
        continue;
      if (!bySourceOrdinal.emplace(entry.source.occurrenceOrdinal, &entry)
               .second)
        return invalid("Module correspondence repeats a source ordinal");
      if (!targetOrdinals.insert(entry.target.occurrenceOrdinal).second)
        return invalid("Module correspondence repeats a target ordinal");
    }
    if (bySourceOrdinal.size() != parent.size() ||
        targetOrdinals.size() != child.size())
      return invalid("Module correspondence does not cover an occurrence "
                     "inventory");
    for (std::size_t index = 0; index != parent.size(); ++index) {
      auto source = bySourceOrdinal.find(index);
      if (source == bySourceOrdinal.end() ||
          source->second->source.id != parent[index].id())
        return invalid("Module correspondence source does not match the "
                       "parent occurrence inventory");
      const std::uint64_t targetOrdinal =
          source->second->target.occurrenceOrdinal;
      if (targetOrdinal >= child.size() ||
          child[targetOrdinal].id() != source->second->target.id)
        return invalid("Module correspondence target does not match the "
                       "child occurrence inventory");
      auto inserted =
          mapping.emplace(::loom::fabric::canonicalFabricBytes(parent[index]),
                          child[targetOrdinal]);
      if (!inserted.second && inserted.first->second != child[targetOrdinal])
        return invalid("Module correspondence maps one parent reference to "
                       "multiple child references");
    }
    return llvm::Error::success();
  }

  template <typename Ref>
  llvm::Expected<Ref> lookup(const Map<Ref> &mapping, Ref ref,
                             llvm::StringRef name) const {
    auto found = mapping.find(::loom::fabric::canonicalFabricBytes(ref));
    if (found == mapping.end())
      return invalid(llvm::Twine("Module child has no ") + name +
                     " correspondence");
    return found->second;
  }

  const ::loom::fabric::FabricArtifactView &parent_;
  const ::loom::fabric::FabricArtifactView &child_;
  Map<::loom::fabric::FabricPeOccurrenceRef> pes_;
  Map<::loom::fabric::FabricFuOccurrenceRef> fus_;
  Map<::loom::fabric::FabricMemoryOccurrenceRef> memories_;
  Map<::loom::fabric::FabricSwitchOccurrenceRef> switches_;
  Map<::loom::fabric::FabricFifoOccurrenceRef> fifos_;
  Map<::loom::fabric::FabricBoundaryOccurrenceRef> boundaries_;
};

llvm::Error
remapSpatialModuleReferences(::mapping::SpatialOp root,
                             const ModuleReferenceRemapper &remapper) {
  llvm::Error error = llvm::Error::success();
  root.walk([&](mlir::Operation *operation) {
    if (error)
      return mlir::WalkResult::interrupt();
    if (auto route = mlir::dyn_cast<::mapping::RouteTreeOp>(operation)) {
      auto endpoint = decodeFabric<::loom::fabric::FabricTransportEndpointRef>(
          route.getRootEndpoint());
      if (!endpoint)
        error = endpoint.takeError();
      else {
        auto mapped = remapper.remap(*endpoint);
        if (!mapped)
          error = mapped.takeError();
        else
          route->setAttr(
              "root_endpoint",
              ::mapping::FabricTransportEndpointRefAttr::get(
                  route.getContext(),
                  DenseI8ArrayAttr::get(
                      route.getContext(),
                      llvm::to_vector<32>(llvm::map_range(
                          ::loom::fabric::canonicalFabricBytes(*mapped),
                          [](std::uint8_t byte) {
                            return static_cast<std::int8_t>(byte);
                          })))));
      }
    } else if (auto node = mlir::dyn_cast<::mapping::RouteNodeOp>(operation)) {
      if (auto traversal = node.getIncomingTraversal()) {
        auto decoded = decodeFabric<::loom::fabric::FabricPhysicalTraversalRef>(
            *traversal);
        if (!decoded)
          error = decoded.takeError();
        else {
          auto mapped = remapper.remap(*decoded);
          if (!mapped)
            error = mapped.takeError();
          else
            node->setAttr(
                "incoming_traversal",
                ::mapping::FabricPhysicalTraversalRefAttr::get(
                    node.getContext(),
                    DenseI8ArrayAttr::get(
                        node.getContext(),
                        llvm::to_vector<32>(llvm::map_range(
                            ::loom::fabric::canonicalFabricBytes(*mapped),
                            [](std::uint8_t byte) {
                              return static_cast<std::int8_t>(byte);
                            })))));
        }
      }
    } else if (auto transfer =
                   mlir::dyn_cast<::mapping::RegisterFifoTransferOp>(
                       operation)) {
      for (const char *name : {"write_traversal", "read_traversal"}) {
        auto attribute =
            transfer->getAttrOfType<::mapping::FabricPhysicalTraversalRefAttr>(
                name);
        if (!attribute)
          continue;
        auto decoded =
            decodeFabric<::loom::fabric::FabricPhysicalTraversalRef>(attribute);
        if (!decoded) {
          error = decoded.takeError();
          break;
        }
        auto mapped = remapper.remap(*decoded);
        if (!mapped) {
          error = mapped.takeError();
          break;
        }
        auto bytes = ::loom::fabric::canonicalFabricBytes(*mapped);
        llvm::SmallVector<std::int8_t, 32> signedBytes;
        for (std::uint8_t byte : bytes)
          signedBytes.push_back(static_cast<std::int8_t>(byte));
        transfer->setAttr(name, ::mapping::FabricPhysicalTraversalRefAttr::get(
                                    transfer.getContext(),
                                    DenseI8ArrayAttr::get(transfer.getContext(),
                                                          signedBytes)));
      }
    }
    return error ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return error;
}

} // namespace

llvm::Error detail::remapSpatialMappingModuleReferences(
    ::mapping::SpatialOp root, const ::loom::fabric::FabricArtifactView &parent,
    const ::loom::fabric::FabricArtifactView &child,
    llvm::ArrayRef<::loom::fabric::FabricModuleEntityCorrespondence>
        correspondence) {
  auto remapper = ModuleReferenceRemapper::get(parent, child, correspondence);
  if (!remapper)
    return remapper.takeError();
  return remapSpatialModuleReferences(root, *remapper);
}

} // namespace loom::mapping
