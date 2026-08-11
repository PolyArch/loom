#include "Fabric/Artifact/InterconnectImplementation.h"

#include "../Identity/FabricArtifactViewInternal.h"
#include "Common/ArtifactFinalizer.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "FabricArtifactBytecodeInternal.h"
#include "FabricArtifactDependencyClosureInternal.h"
#include "FabricInterconnectImplementationInternal.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
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

template <typename Ref>
void canonicalizeRefs(std::vector<Ref> &references) {
  llvm::sort(references, [](const Ref &left, const Ref &right) {
    return canonicalFabricBytes(left) < canonicalFabricBytes(right);
  });
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

struct Gem5EventTransfer final {
  std::int64_t ingress = 0;
  std::vector<std::int64_t> egresses;
  std::vector<std::int64_t> resources;
};

struct InterconnectCatalog final {
  std::vector<FabricTransportEndpointRef> endpoints;
  std::vector<FabricResourceStateRef> resourceStates;
  std::vector<FabricTransferPatternRef> transferPatterns;
  std::vector<FabricSemanticConfigFieldRef> configurationFields;
  std::vector<Gem5EventTransfer> transfers;
};

void appendConfigurationFields(const FabricArtifactView &artifact,
                               const FabricInventoryOwnerRef &owner,
                               InterconnectCatalog &catalog) {
  const std::uint64_t count = artifact.inventorySize(
      owner, FabricInventoryKind::SemanticConfigField);
  const FabricConfigurationOwnerRef projected(owner);
  for (FabricOrdinal ordinal = 0; ordinal < count; ++ordinal)
    catalog.configurationFields.push_back({projected, ordinal});
}

llvm::Expected<InterconnectCatalog>
buildCatalog(const FabricSystemRootView &system) {
  InterconnectCatalog catalog;
  const FabricArtifactView &artifact = system.artifact();
  catalog.endpoints.assign(artifact.transportEndpoints().begin(),
                           artifact.transportEndpoints().end());
  canonicalizeRefs(catalog.endpoints);

  for (SystemTransportResourceRef resource : system.transportResources()) {
    const FabricInventoryOwnerRef owner =
        FabricInventoryOwnerRef::of(resource);
    const FabricResourceStateOwnerRef stateOwner(owner);
    const std::uint64_t stateCount =
        artifact.inventorySize(owner, FabricInventoryKind::ResourceState);
    for (FabricOrdinal ordinal = 0; ordinal < stateCount; ++ordinal)
      catalog.resourceStates.push_back({stateOwner, ordinal});
    appendConfigurationFields(artifact, owner, catalog);

    for (FabricTransferPatternRef pattern :
         system.transferPatterns(resource)) {
      catalog.transferPatterns.push_back(pattern);
      appendConfigurationFields(
          artifact, FabricInventoryOwnerRef::of(pattern), catalog);
    }
  }
  canonicalizeRefs(catalog.resourceStates);
  canonicalizeRefs(catalog.transferPatterns);
  canonicalizeRefs(catalog.configurationFields);

  std::map<std::vector<std::uint8_t>, std::int64_t> endpointOrdinals;
  for (auto [ordinal, endpoint] : llvm::enumerate(catalog.endpoints))
    endpointOrdinals.emplace(canonicalFabricBytes(endpoint), ordinal);
  std::map<std::vector<std::uint8_t>, std::int64_t> resourceOrdinals;
  for (auto [ordinal, state] : llvm::enumerate(catalog.resourceStates))
    resourceOrdinals.emplace(canonicalFabricBytes(state), ordinal);

  catalog.transfers.reserve(catalog.transferPatterns.size());
  for (FabricTransferPatternRef pattern : catalog.transferPatterns) {
    const SystemTransferPatternRecord *record = system.transferPattern(pattern);
    if (!record)
      return invalid("interconnect catalog selected an unknown transfer "
                     "pattern");
    const auto ingress =
        endpointOrdinals.find(canonicalFabricBytes(record->ingress()));
    if (ingress == endpointOrdinals.end())
      return invalid("transfer pattern ingress is absent from the endpoint "
                     "catalog");

    Gem5EventTransfer transfer;
    transfer.ingress = ingress->second;
    for (const FabricTransportEndpointRef &egress : record->egresses()) {
      const auto found =
          endpointOrdinals.find(canonicalFabricBytes(egress));
      if (found == endpointOrdinals.end())
        return invalid("transfer pattern egress is absent from the endpoint "
                       "catalog");
      transfer.egresses.push_back(found->second);
    }
    llvm::sort(transfer.egresses);
    transfer.egresses.erase(
        std::unique(transfer.egresses.begin(), transfer.egresses.end()),
        transfer.egresses.end());

    const FabricInventoryOwnerRef &useOwner =
        record->usePattern().owner.catalog();
    const ::fabric::ResourceContract *contract =
        artifact.resourceContract(useOwner);
    if (!contract ||
        record->usePattern().ordinal >= contract->usePatternCount())
      return invalid("transfer pattern selects an unknown use pattern");
    const ::fabric::UsePattern use = contract->usePattern(
        ::fabric::UsePatternKey(record->usePattern().ordinal));
    for (const ::fabric::Claim &claim : use.claims) {
      const FabricResourceStateRef state{
          FabricResourceStateOwnerRef(useOwner), claim.state.ordinal()};
      const auto found =
          resourceOrdinals.find(canonicalFabricBytes(state));
      if (found == resourceOrdinals.end())
        return invalid("transfer use references a resource state outside the "
                       "interconnect catalog");
      transfer.resources.push_back(found->second);
    }
    llvm::sort(transfer.resources);
    transfer.resources.erase(
        std::unique(transfer.resources.begin(), transfer.resources.end()),
        transfer.resources.end());
    catalog.transfers.push_back(std::move(transfer));
  }
  return catalog;
}

std::vector<std::int8_t>
signedBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> result;
  result.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    result.push_back(static_cast<std::int8_t>(byte));
  return result;
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

void appendRefinement(OpBuilder &builder, Location location,
                      ::fabric::InterconnectRefinementKind kind,
                      llvm::ArrayRef<std::uint8_t> architecture,
                      llvm::ArrayRef<std::int64_t> protocol) {
  const std::vector<std::int8_t> bytes = signedBytes(architecture);
  ::fabric::InterconnectRefinementOp::create(builder, location, kind, bytes,
                                              protocol);
}

llvm::Expected<std::vector<std::uint8_t>>
buildCanonicalBytecode(const FabricArtifactView &artifact,
                       const InterconnectCatalog &catalog) {
  DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  const Location location = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(location);
  OpBuilder rootBuilder(module->getBodyRegion());
  rootBuilder.setInsertionPointToStart(module->getBody());
  auto root = ::fabric::InterconnectImplementationOp::create(
      rootBuilder, location, canonicalRootName,
      ::fabric::InterconnectProtocolSchema::Gem5EventTransportV1);
  root.getImplementation().push_back(new Block());
  root.getRefinements().push_back(new Block());

  OpBuilder bodyBuilder =
      OpBuilder::atBlockEnd(&root.getImplementation().front());
  for (auto [ordinal, endpoint] : llvm::enumerate(catalog.endpoints)) {
    const std::optional<FabricPortDirection> direction =
        artifact.transportEndpointDirection(endpoint);
    if (!direction)
      return invalid("interconnect endpoint has no architecture direction");
    ::fabric::InterconnectGem5EventEndpointOp::create(
        bodyBuilder,
        location, static_cast<std::int64_t>(ordinal),
        *direction == FabricPortDirection::Input
            ? ::fabric::InterconnectEndpointDirection::Ingress
            : ::fabric::InterconnectEndpointDirection::Egress);
  }
  for (std::size_t ordinal = 0; ordinal < catalog.resourceStates.size();
       ++ordinal)
    ::fabric::InterconnectGem5EventResourceOp::create(
        bodyBuilder, location, static_cast<std::int64_t>(ordinal));
  for (auto [ordinal, transfer] : llvm::enumerate(catalog.transfers))
    ::fabric::InterconnectGem5EventTransferOp::create(
        bodyBuilder, location, static_cast<std::int64_t>(ordinal),
        transfer.ingress, transfer.egresses, transfer.resources);
  for (std::size_t ordinal = 0;
       ordinal < catalog.configurationFields.size(); ++ordinal)
    ::fabric::InterconnectGem5EventConfigurationFieldOp::create(
        bodyBuilder, location, static_cast<std::int64_t>(ordinal));

  OpBuilder refinementBuilder =
      OpBuilder::atBlockEnd(&root.getRefinements().front());
  const auto oneToOne = [&](auto kind, const auto &references) {
    for (auto [ordinal, reference] : llvm::enumerate(references)) {
      const std::array<std::int64_t, 1> protocol = {
          static_cast<std::int64_t>(ordinal)};
      appendRefinement(refinementBuilder, location, kind,
                       canonicalFabricBytes(reference), protocol);
    }
  };
  oneToOne(::fabric::InterconnectRefinementKind::Endpoint,
           catalog.endpoints);
  oneToOne(::fabric::InterconnectRefinementKind::ResourceState,
           catalog.resourceStates);
  oneToOne(::fabric::InterconnectRefinementKind::TransferPattern,
           catalog.transferPatterns);
  oneToOne(::fabric::InterconnectRefinementKind::Configuration,
           catalog.configurationFields);

  if (failed(verify(module.get())))
    return invalid("canonical interconnect implementation failed verification");
  return detail::writeCanonicalFabricBytecode(module.get());
}

template <typename Ref>
llvm::Error validateRefinement(
    ::fabric::InterconnectRefinementOp operation,
    ::fabric::InterconnectRefinementKind expectedKind, const Ref &architecture,
    std::int64_t protocolOrdinal) {
  if (operation.getKind() != expectedKind)
    return invalid("interconnect refinement is not in canonical kind order");
  if (unsignedBytes(operation.getArchitectureRefAttr()) !=
      canonicalFabricBytes(architecture))
    return invalid("interconnect refinement selects the wrong architecture "
                   "reference");
  const ArrayRef<std::int64_t> protocol =
      operation.getProtocolRefsAttr().asArrayRef();
  if (protocol.size() != 1 || protocol.front() != protocolOrdinal)
    return invalid("gem5 event transport requires one-to-one canonical "
                   "refinement");
  return llvm::Error::success();
}

llvm::Error validateCanonicalRoot(
    ::fabric::InterconnectImplementationOp root,
    const InterconnectCatalog &catalog, const FabricArtifactView &artifact) {
  if (root.getProtocolSchema() !=
      ::fabric::InterconnectProtocolSchema::Gem5EventTransportV1)
    return invalid("unsupported interconnect protocol schema");

  auto operation = root.getImplementation().front().begin();
  for (auto [ordinal, endpoint] : llvm::enumerate(catalog.endpoints)) {
    if (operation == root.getImplementation().front().end())
      return invalid("interconnect implementation omits a protocol endpoint");
    auto concrete =
        dyn_cast<::fabric::InterconnectGem5EventEndpointOp>(&*operation++);
    const std::optional<FabricPortDirection> direction =
        artifact.transportEndpointDirection(endpoint);
    if (!concrete || !direction || concrete.getOrdinal() != ordinal ||
        concrete.getDirection() !=
            (*direction == FabricPortDirection::Input
                 ? ::fabric::InterconnectEndpointDirection::Ingress
                 : ::fabric::InterconnectEndpointDirection::Egress))
      return invalid("protocol endpoint catalog does not match the refined "
                     "System");
  }
  for (std::size_t ordinal = 0; ordinal < catalog.resourceStates.size();
       ++ordinal) {
    if (operation == root.getImplementation().front().end())
      return invalid("interconnect implementation omits a protocol resource");
    auto concrete =
        dyn_cast<::fabric::InterconnectGem5EventResourceOp>(&*operation++);
    if (!concrete || concrete.getOrdinal() != ordinal)
      return invalid("protocol resource catalog is not canonical");
  }
  for (auto [ordinal, transfer] : llvm::enumerate(catalog.transfers)) {
    if (operation == root.getImplementation().front().end())
      return invalid("interconnect implementation omits a protocol transfer");
    auto concrete =
        dyn_cast<::fabric::InterconnectGem5EventTransferOp>(&*operation++);
    if (!concrete || concrete.getOrdinal() != ordinal ||
        concrete.getIngress() !=
            static_cast<std::uint64_t>(transfer.ingress) ||
        concrete.getEgressesAttr().asArrayRef() !=
            ArrayRef<std::int64_t>(transfer.egresses) ||
        concrete.getResourcesAttr().asArrayRef() !=
            ArrayRef<std::int64_t>(transfer.resources))
      return invalid("protocol transfer catalog does not match the refined "
                     "System");
  }
  for (std::size_t ordinal = 0;
       ordinal < catalog.configurationFields.size(); ++ordinal) {
    if (operation == root.getImplementation().front().end())
      return invalid(
          "interconnect implementation omits a configuration field");
    auto concrete =
        dyn_cast<::fabric::InterconnectGem5EventConfigurationFieldOp>(
            &*operation++);
    if (!concrete || concrete.getOrdinal() != ordinal)
      return invalid("protocol configuration catalog is not canonical");
  }
  if (operation != root.getImplementation().front().end())
    return invalid("interconnect implementation has an extra protocol object");

  auto refinement = root.getRefinements().front().begin();
  const auto validateRange = [&](auto kind,
                                 const auto &references) -> llvm::Error {
    for (auto [ordinal, reference] : llvm::enumerate(references)) {
      if (refinement == root.getRefinements().front().end())
        return invalid("interconnect implementation omits a refinement");
      auto record = dyn_cast<::fabric::InterconnectRefinementOp>(
          &*refinement++);
      if (!record)
        return invalid("interconnect refinement region has a foreign record");
      if (llvm::Error error = validateRefinement(
              record, kind, reference, static_cast<std::int64_t>(ordinal)))
        return error;
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = validateRange(
          ::fabric::InterconnectRefinementKind::Endpoint, catalog.endpoints))
    return error;
  if (llvm::Error error =
          validateRange(::fabric::InterconnectRefinementKind::ResourceState,
                        catalog.resourceStates))
    return error;
  if (llvm::Error error =
          validateRange(::fabric::InterconnectRefinementKind::TransferPattern,
                        catalog.transferPatterns))
    return error;
  if (llvm::Error error = validateRange(
          ::fabric::InterconnectRefinementKind::Configuration,
          catalog.configurationFields))
    return error;
  if (refinement != root.getRefinements().front().end())
    return invalid("interconnect implementation has an extra refinement");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FabricArtifactView>
detail::strictImportInterconnectImplementation(
    const ArtifactRootReference &reference,
    const DecodedFabricArtifact &decoded, const ArtifactStore &store) {
  if (decoded.rootKind != FabricRootKind::InterconnectImplementation)
    return invalid("interconnect importer received the wrong root kind");
  if (decoded.dependencies.size() != 1 ||
      decoded.dependencies.front().role != FabricDependencyRole::RefinedSystem)
    return invalid("interconnect implementation requires one exact RefinedSystem "
                   "dependency");

  auto refined =
      importEntireFabricRoot(decoded.dependencies.front().root, store);
  if (!refined)
    return refined.takeError();
  auto system = requireSystemRoot(refined->view());
  if (!system)
    return system.takeError();
  auto catalog = buildCatalog(*system);
  if (!catalog)
    return catalog.takeError();

  auto parsed = parseFabricBytecodeModule(decoded.canonicalMlirBytecode);
  if (!parsed)
    return parsed.takeError();
  ModuleOp module = parsed->module.get();
  if (!llvm::hasSingleElement(module.getBody()->getOperations()))
    return invalid("canonical interconnect payload does not contain one root");
  auto root = dyn_cast<::fabric::InterconnectImplementationOp>(
      &module.getBody()->front());
  if (!root || root.getSymName() != canonicalRootName)
    return invalid("canonical payload has no canonical interconnect root");
  if (llvm::Error error =
          validateCanonicalRoot(root, *catalog, system->artifact()))
    return std::move(error);

  detail::FabricArtifactViewData data(
      reference.artifact, FabricRootKind::InterconnectImplementation);
  data.contextOwner = parsed->context;
  data.canonicalModule =
      std::make_shared<OwningOpRef<ModuleOp>>(std::move(parsed->module));
  return detail::buildFabricArtifactView(std::move(data));
}

llvm::Expected<FinalizedFabricRoot>
finalizeGem5EventInterconnectImplementation(
    const ArtifactRootReference &refinedSystem, const ArtifactStore &store) {
  auto refined = importEntireFabricRoot(refinedSystem, store);
  if (!refined)
    return refined.takeError();
  auto system = requireSystemRoot(refined->view());
  if (!system)
    return system.takeError();
  auto catalog = buildCatalog(*system);
  if (!catalog)
    return catalog.takeError();
  auto bytecode = buildCanonicalBytecode(system->artifact(), *catalog);
  if (!bytecode)
    return bytecode.takeError();

  const FabricDirectDependency dependency{FabricDependencyRole::RefinedSystem,
                                          refinedSystem};
  auto canonical = encodeFabricArtifactEnvelope(
      FabricRootKind::InterconnectImplementation, {dependency}, *bytecode);
  if (!canonical)
    return canonical.takeError();
  ArtifactRootReference reference{
      fabricArtifactSchema.identity.str(), fabricArtifactSchema.version,
      finalizeArtifactIdentity(fabricArtifactSchema, *canonical)};
  auto decoded = decodeFabricArtifactEnvelope(canonical->bytes());
  if (!decoded)
    return decoded.takeError();
  auto imported = detail::strictImportInterconnectImplementation(
      reference, *decoded, store);
  if (!imported)
    return imported.takeError();
  if (llvm::Error error =
          detail::validateFabricArtifactDependencyFramingClosure(store,
                                                                 *canonical))
    return std::move(error);
  auto stored = store.put(fabricArtifactSchema, *canonical);
  if (!stored)
    return stored.takeError();
  if (*stored != reference.artifact)
    return invalid("ArtifactStore returned a different Fabric identity");
  return importEntireFabricRoot(reference, store);
}

llvm::Expected<::fabric::InterconnectProtocolSchema>
interconnectProtocolSchema(const FinalizedFabricRoot &implementation) {
  if (implementation.view().rootKind() !=
      FabricRootKind::InterconnectImplementation)
    return invalid("Fabric root is not an InterconnectImplementation");
  const Operation *canonical = implementation.view().canonicalOperation();
  auto module = dyn_cast_if_present<ModuleOp>(canonical);
  if (!module || !llvm::hasSingleElement(module.getBody()->getOperations()))
    return invalid("interconnect implementation has no canonical operation");
  auto root = dyn_cast<::fabric::InterconnectImplementationOp>(
      &module.getBody()->front());
  if (!root)
    return invalid("interconnect implementation has the wrong canonical root");
  return root.getProtocolSchema();
}

} // namespace loom::fabric
