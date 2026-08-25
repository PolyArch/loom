#include "FabricModuleCanonicalization.h"

#include "Fabric/IR/Elaboration.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ModuleDomain.h"
#include "FabricCanonicalLabeling.h"
#include "FabricFuCapabilityDerivation.h"
#include "FabricModuleCanonicalPayload.h"
#include "FabricModuleDomainMaterialization.h"
#include "FabricModuleDomainNormalization.h"
#include "FabricResourceContractFinalization.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::fabric {
namespace {

constexpr llvm::StringLiteral canonicalRootName("__loom_fabric_root");

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Error
reorderCanonicalGraphRegions(::fabric::ModuleOp root,
                             const detail::FabricCanonicalLabeling &labeling) {
  llvm::DenseMap<Operation *, std::uint64_t> rank;
  for (auto [index, operation] :
       llvm::enumerate(labeling.canonicalOperationOrder))
    rank[operation] = index;

  llvm::SmallVector<Block *> blocks;
  root->walk([&](Operation *operation) {
    if (!isa<::fabric::ModuleOp, ::fabric::PeOp, ::fabric::FuOp>(operation))
      return;
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        blocks.push_back(&block);
  });

  for (Block *block : blocks) {
    llvm::SmallVector<Operation *> ordered;
    Operation *terminator = nullptr;
    const bool fuDefinition = isa<::fabric::FuOp>(block->getParentOp());
    for (Operation &operation : *block) {
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        terminator = &operation;
        continue;
      }
      const bool known =
          fuDefinition
              ? labeling.definitionFuNodeOrdinalByOperation.count(&operation)
              : rank.count(&operation);
      if (!known)
        return invalid("canonical operation order omits a graph operation");
      ordered.push_back(&operation);
    }
    llvm::sort(ordered, [&](Operation *left, Operation *right) {
      if (fuDefinition)
        return labeling.definitionFuNodeOrdinalByOperation.lookup(left) <
               labeling.definitionFuNodeOrdinalByOperation.lookup(right);
      return rank.lookup(left) < rank.lookup(right);
    });
    for (Operation *operation : ordered) {
      if (terminator)
        operation->moveBefore(terminator);
      else
        operation->moveBefore(block, block->end());
    }
  }
  return llvm::Error::success();
}

std::optional<FabricEntityKind> moduleOccurrenceKind(Operation *operation) {
  if (isa<::fabric::ModuleOp>(operation))
    return FabricEntityKind::FabricModuleTemplate;
  if (isa<::fabric::PeOp>(operation))
    return FabricEntityKind::FabricPeOccurrence;
  if (isa<::fabric::FuOp>(operation))
    return FabricEntityKind::FabricFuOccurrence;
  if (isa<::fabric::MemOp>(operation))
    return FabricEntityKind::FabricMemoryOccurrence;
  if (isa<::fabric::SwitchOp>(operation))
    return FabricEntityKind::FabricSwitchOccurrence;
  if (isa<::fabric::FifoOp>(operation))
    return FabricEntityKind::FabricFifoOccurrence;
  if (isa<::fabric::BoundaryOp>(operation))
    return FabricEntityKind::FabricBoundaryOccurrence;
  return std::nullopt;
}

struct AuthoredFuCapabilityRow final {
  std::vector<Operation *> activeOperations;
  std::vector<std::pair<Operation *, FabricOrdinal>> routes;
};

using AuthoredFuCapabilityRows =
    std::map<Operation *, std::vector<AuthoredFuCapabilityRow>>;

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  return bytes;
}

llvm::Expected<AuthoredFuCapabilityRows> captureAuthoredFuCapabilityRows(
    ::fabric::ModuleOp root,
    const std::map<Operation *, FabricModuleEntityReference> &authored) {
  AuthoredFuCapabilityRows result;
  llvm::Error error = llvm::Error::success();
  root->walk([&](::fabric::FuOp fu) {
    if (error)
      return WalkResult::interrupt();
    if (!authored.count(fu.getOperation()))
      return WalkResult::advance();
    ::fabric::FuCapabilityDomainAttr attribute =
        fu.getCapabilityTemplatesAttr();
    if (!attribute)
      return WalkResult::advance();
    auto domain = ::fabric::decodeFuCapabilityDomainRecord(
        unsignedBytes(attribute.getRecord()));
    if (!domain) {
      error = domain.takeError();
      return WalkResult::interrupt();
    }

    llvm::SmallVector<Operation *, 16> nodes;
    for (Operation &operation : fu.getBody().front().without_terminator())
      if (isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(operation))
        nodes.push_back(&operation);
    std::vector<AuthoredFuCapabilityRow> rows;
    rows.reserve(domain->templates().size());
    for (const ::fabric::FuCapabilityTemplateSelection &selection :
         domain->templates()) {
      AuthoredFuCapabilityRow row;
      row.activeOperations.reserve(
          selection.activeOperationNodeOrdinals.size());
      for (FabricOrdinal ordinal : selection.activeOperationNodeOrdinals) {
        if (ordinal >= nodes.size() || !isa<::fabric::OpOp>(nodes[ordinal])) {
          error = invalid("authored FU capability names an invalid operation");
          return WalkResult::interrupt();
        }
        row.activeOperations.push_back(nodes[ordinal]);
      }
      row.routes.reserve(selection.routes.size());
      for (const ::fabric::FuCapabilityRouteSelection &route :
           selection.routes) {
        if (route.selectorNodeOrdinal >= nodes.size() ||
            !isa<::fabric::MuxOp, ::fabric::DemuxOp>(
                nodes[route.selectorNodeOrdinal])) {
          error = invalid("authored FU capability names an invalid selector");
          return WalkResult::interrupt();
        }
        row.routes.emplace_back(nodes[route.selectorNodeOrdinal],
                                route.selectedPort);
      }
      rows.push_back(std::move(row));
    }
    result.emplace(fu.getOperation(), std::move(rows));
    return WalkResult::advance();
  });
  if (error)
    return std::move(error);
  return result;
}

llvm::Expected<std::vector<FabricFuCapabilityTemplateCorrespondence>>
projectFuCapabilityTemplateCorrespondence(
    const std::map<Operation *, FabricModuleEntityReference> &authored,
    const AuthoredFuCapabilityRows &sourceRows,
    const detail::FabricCanonicalLabeling &labeling) {
  std::vector<FabricFuCapabilityTemplateCorrespondence> result;
  for (const auto &[operation, rows] : sourceRows) {
    auto fu = dyn_cast_or_null<::fabric::FuOp>(operation);
    auto sourceEntity = authored.find(operation);
    auto templateId = labeling.fuTemplateIdByOccurrence.find(operation);
    if (!fu || sourceEntity == authored.end() ||
        sourceEntity->second.kind != FabricEntityKind::FabricFuOccurrence ||
        templateId == labeling.fuTemplateIdByOccurrence.end())
      return invalid("authored FU capability lost its owner correspondence");
    const FabricFuTemplateRef owner(templateId->second);

    llvm::SmallVector<Operation *, 16> canonicalNodes;
    for (Operation &node : fu.getBody().front().without_terminator())
      if (isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(node))
        canonicalNodes.push_back(&node);
    llvm::sort(canonicalNodes, [&](Operation *left, Operation *right) {
      return labeling.definitionFuNodeOrdinalByOperation.lookup(left) <
             labeling.definitionFuNodeOrdinalByOperation.lookup(right);
    });
    for (auto [ordinal, node] : llvm::enumerate(canonicalNodes)) {
      auto found = labeling.definitionFuNodeOrdinalByOperation.find(node);
      if (found == labeling.definitionFuNodeOrdinalByOperation.end() ||
          found->second != ordinal)
        return invalid("authored FU capability lost its node correspondence");
    }

    auto finalRecords =
        detail::deriveFabricFuCapabilityTemplates(fu, owner, canonicalNodes);
    if (!finalRecords)
      return finalRecords.takeError();
    std::map<std::vector<std::uint8_t>, FabricOrdinal> finalOrdinals;
    for (auto [ordinal, record] : llvm::enumerate(*finalRecords)) {
      auto bytes = canonicalFabricFuCapabilityTemplateBytes(record);
      if (!bytes)
        return bytes.takeError();
      if (!finalOrdinals.emplace(std::move(*bytes), ordinal).second)
        return invalid("canonical FU capability inventory is not unique");
    }

    for (auto [sourceOrdinal, row] : llvm::enumerate(rows)) {
      ::fabric::FuCapabilityTemplateSelection selection;
      selection.activeOperationNodeOrdinals.reserve(
          row.activeOperations.size());
      for (Operation *active : row.activeOperations) {
        auto found = labeling.definitionFuNodeOrdinalByOperation.find(active);
        if (found == labeling.definitionFuNodeOrdinalByOperation.end())
          return invalid("authored FU capability operation was not relabeled");
        selection.activeOperationNodeOrdinals.push_back(found->second);
      }
      selection.routes.reserve(row.routes.size());
      for (const auto &[selector, selectedPort] : row.routes) {
        auto found = labeling.definitionFuNodeOrdinalByOperation.find(selector);
        if (found == labeling.definitionFuNodeOrdinalByOperation.end())
          return invalid("authored FU capability selector was not relabeled");
        selection.routes.push_back({found->second, selectedPort});
      }
      auto normalized =
          ::fabric::FuCapabilityDomainRecord::create({std::move(selection)});
      if (!normalized)
        return normalized.takeError();
      auto record = detail::deriveFabricFuCapabilityTemplate(
          fu, owner, canonicalNodes, normalized->templates().front());
      if (!record)
        return record.takeError();
      auto bytes = canonicalFabricFuCapabilityTemplateBytes(*record);
      if (!bytes)
        return bytes.takeError();
      auto target = finalOrdinals.find(*bytes);
      if (target == finalOrdinals.end())
        return invalid("authored FU capability has no canonical record");
      result.push_back(
          {{sourceEntity->second, sourceOrdinal}, {owner, target->second}});
    }
  }
  llvm::sort(result, [](const auto &left, const auto &right) {
    return std::tie(left.source.fu.kind, left.source.fu.id,
                    left.source.fu.occurrenceOrdinal, left.source.ordinal) <
           std::tie(right.source.fu.kind, right.source.fu.id,
                    right.source.fu.occurrenceOrdinal, right.source.ordinal);
  });
  return result;
}

llvm::Expected<std::vector<FabricModuleEntityCorrespondence>>
projectModuleEntityCorrespondence(
    const std::map<Operation *, FabricModuleEntityReference> &authored,
    const detail::FabricCanonicalLabeling &labeling) {
  std::vector<FabricModuleEntityCorrespondence> result;
  std::set<std::pair<FabricEntityKind, FabricEntityId>> sourceKeys;
  std::set<std::pair<FabricEntityKind, FabricEntityId>> targetKeys;
  std::map<Operation *, FabricEntityId> targetByOperation;
  std::map<FabricEntityKind, std::uint64_t> targetOrdinals;
  for (const detail::FabricEntityCarrier &carrier : labeling.carriers) {
    if (!carrier.op)
      continue;
    const auto priorTarget = targetByOperation.find(carrier.op);
    if (priorTarget != targetByOperation.end()) {
      if (priorTarget->second != carrier.id)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "fabric_module_correspondence_invalid: one operation has "
            "multiple canonical entity IDs");
      continue;
    }
    targetByOperation.emplace(carrier.op, carrier.id);
    const auto source = authored.find(carrier.op);
    if (source == authored.end() || source->second.kind != carrier.kind)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "fabric_module_correspondence_invalid: Module canonical entity has "
          "no authored correspondence kind=" +
              llvm::Twine(static_cast<std::uint32_t>(carrier.kind)) +
              " authored_count=" + llvm::Twine(authored.size()));
    const auto sourceKey =
        std::make_pair(source->second.kind, source->second.occurrenceOrdinal);
    const auto targetOrdinal = targetOrdinals[carrier.kind]++;
    const auto targetKey = std::make_pair(carrier.kind, targetOrdinal);
    const bool sourceInserted = sourceKeys.insert(sourceKey).second;
    const bool targetInserted = targetKeys.insert(targetKey).second;
    if (!sourceInserted || !targetInserted)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "fabric_module_correspondence_invalid: Module entity "
          "correspondence is not one-to-one");
    result.push_back(
        {source->second, {carrier.kind, carrier.id, targetOrdinal}});
  }
  if (result.size() != authored.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "fabric_module_correspondence_invalid: Module entity "
        "correspondence is incomplete");
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.source.kind, lhs.source.occurrenceOrdinal) <
           std::tie(rhs.source.kind, rhs.source.occurrenceOrdinal);
  });
  return result;
}

} // namespace

llvm::Expected<detail::CanonicalFabricModuleCandidate>
detail::buildCanonicalFabricModuleCandidate(
    ::fabric::ModuleOp source,
    const ::fabric::ModuleDomainAuthoringRelation *domainRelation,
    bool captureEntityCorrespondence, bool captureCapabilityCorrespondence) {
  auto sourceModule = source->getParentOfType<ModuleOp>();
  if (!sourceModule || source->getParentOp() != sourceModule.getOperation())
    return invalid("the selected Fabric Module must be top-level");
  if (failed(verify(sourceModule)))
    return invalid("the authoring module does not verify");

  IRMapping cloneMapping;
  OwningOpRef<ModuleOp> scratch(
      cast<ModuleOp>(sourceModule->clone(cloneMapping)));
  Operation *clonedOperation =
      SymbolTable::lookupSymbolIn(*scratch, source.getSymNameAttr());
  auto clonedRoot = dyn_cast_or_null<::fabric::ModuleOp>(clonedOperation);
  if (!clonedRoot)
    return invalid("the selected Fabric root was not cloned");

  std::map<Operation *, FabricModuleEntityReference> authoredEntities;
  if (captureEntityCorrespondence || captureCapabilityCorrespondence) {
    std::map<FabricEntityKind, std::uint64_t> authoredOrdinals;
    clonedRoot->walk([&](Operation *operation) {
      const auto kind = moduleOccurrenceKind(operation);
      if (!kind)
        return;
      auto id = operation->getAttrOfType<::fabric::EntityIdAttr>(
          ::fabric::kEntityIdAttrName);
      if (id)
        authoredEntities.emplace(
            operation, FabricModuleEntityReference{*kind, id.getId(),
                                                   authoredOrdinals[*kind]++});
    });
    if (authoredEntities.empty())
      return invalid(
          "Module correspondence capture requires canonical parent IDs");
  }

  std::optional<::fabric::ModuleDomainAuthoringRelation> remappedDomain;
  if (domainRelation && domainRelation->hasDomainAuthoring()) {
    auto remapped = domainRelation->remap(cloneMapping);
    if (!remapped)
      return remapped.takeError();
    remappedDomain = std::move(*remapped);
  }

  if (llvm::Error error = detail::stripFabricModuleAuthoringState(clonedRoot))
    return std::move(error);
  bool hasInstances = false;
  clonedRoot->walk([&](::fabric::InstantiateOp) { hasInstances = true; });
  if (hasInstances) {
    LogicalResult elaborated =
        remappedDomain
            ? ::fabric::elaborateInstances(clonedRoot, *remappedDomain)
            : ::fabric::elaborateInstances(clonedRoot);
    if (failed(elaborated))
      return invalid("fabric.instantiate elaboration failed");
  }
  if (llvm::Error error =
          detail::eraseElaboratedFabricModuleDeclarations(clonedRoot))
    return std::move(error);

  AuthoredFuCapabilityRows authoredCapabilities;
  if (captureCapabilityCorrespondence) {
    auto captured =
        captureAuthoredFuCapabilityRows(clonedRoot, authoredEntities);
    if (!captured)
      return captured.takeError();
    authoredCapabilities = std::move(*captured);
  }

  for (Operation &operation :
       llvm::make_early_inc_range(scratch->getBody()->getOperations()))
    if (&operation != clonedRoot.getOperation())
      operation.erase();
  clonedRoot.setSymName(canonicalRootName);
  if (failed(verify(*scratch)))
    return invalid("the root-complete Fabric candidate does not verify");

  auto normalizedDomain =
      remappedDomain
          ? detail::normalizeFabricModuleDomain(clonedRoot, *remappedDomain)
          : detail::buildDefaultFabricModuleDomain(clonedRoot);
  if (!normalizedDomain)
    return normalizedDomain.takeError();

  auto preliminary = detail::computeFabricModuleCanonicalLabeling(
      clonedRoot, *normalizedDomain);
  if (!preliminary)
    return preliminary.takeError();
  if (llvm::Error error =
          detail::materializeFabricResourceContracts(clonedRoot, *preliminary))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("the complete Fabric resource contracts do not verify");

  auto labeling = detail::computeFabricModuleCanonicalLabeling(
      clonedRoot, *normalizedDomain);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error = reorderCanonicalGraphRegions(clonedRoot, *labeling))
    return std::move(error);
  if (llvm::Error error =
          detail::materializeFabricCanonicalFuCapabilityDomains(*labeling))
    return std::move(error);
  auto reordered = detail::computeCanonicalFabricModulePayloadLabeling(
      clonedRoot, *normalizedDomain);
  if (!reordered)
    return reordered.takeError();
  std::vector<FabricModuleEntityCorrespondence> entities;
  std::vector<FabricFuCapabilityTemplateCorrespondence> capabilities;
  if (captureEntityCorrespondence) {
    auto projected =
        projectModuleEntityCorrespondence(authoredEntities, *reordered);
    if (!projected)
      return projected.takeError();
    entities = std::move(*projected);
  }
  if (captureCapabilityCorrespondence) {
    auto projectedCapabilities = projectFuCapabilityTemplateCorrespondence(
        authoredEntities, authoredCapabilities, *reordered);
    if (!projectedCapabilities)
      return projectedCapabilities.takeError();
    capabilities = std::move(*projectedCapabilities);
  }
  if (llvm::Error error = detail::materializeFabricCanonicalIds(*reordered))
    return std::move(error);
  if (llvm::Error error = detail::materializeFabricModuleDomainRelation(
          clonedRoot, *normalizedDomain, *reordered))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("canonical Fabric IDs produced invalid IR");
  return detail::CanonicalFabricModuleCandidate{
      std::move(scratch), std::move(entities), std::move(capabilities)};
}

} // namespace loom::fabric
