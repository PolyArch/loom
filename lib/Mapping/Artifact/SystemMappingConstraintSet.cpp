#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "MappingConstraintCanonicalization.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingAttrs.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_constraint_set_invalid: " +
                                     message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

llvm::Expected<ArtifactIdentity>
decodeIdentity(::mapping::ArtifactIdentityAttr attribute) {
  return ArtifactIdentity::fromBytes(unsignedBytes(attribute.getRecord()));
}

llvm::Expected<ArtifactRootReference>
decodeRootReference(::mapping::ArtifactRootReferenceAttr attribute) {
  std::vector<std::uint8_t> bytes = unsignedBytes(attribute.getRecord());
  auto decoded = decodeArtifactRootReferencePrefix(bytes);
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount != bytes.size() ||
      encodeArtifactRootReference(decoded->reference) != bytes)
    return invalid("root-reference table contains noncanonical framing");
  return std::move(decoded->reference);
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeDataflow(Attr attribute,
                                   const ArtifactIdentity &owner) {
  return ::dataflow::decodeDataflowReference<Ref>(
      unsignedBytes(attribute.getRecord()), owner);
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeFabric(Attr attribute) {
  return ::loom::fabric::decodeFabricRef<Ref>(
      unsignedBytes(attribute.getRecord()));
}

template <typename T>
llvm::Expected<T> contextual(llvm::Expected<T> value,
                             const llvm::Twine &context) {
  if (!value)
    return llvm::joinErrors(invalid(context), value.takeError());
  return std::move(*value);
}

llvm::Error contextual(llvm::Error error, const llvm::Twine &context) {
  if (!error)
    return llvm::Error::success();
  return llvm::joinErrors(invalid(context), std::move(error));
}

std::string recordKey(Attribute attribute) {
  DenseI8ArrayAttr record;
  if (auto reference = dyn_cast<::mapping::RootThreadLaunchRefAttr>(attribute))
    record = reference.getRecord();
  else
    record = cast<::mapping::ArtifactRootReferenceAttr>(attribute).getRecord();
  std::vector<std::uint8_t> bytes = unsignedBytes(record);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

ArrayAttr normalizeReferenceArray(ArrayAttr values) {
  std::vector<Attribute> normalized(values.begin(), values.end());
  llvm::sort(normalized, [](Attribute lhs, Attribute rhs) {
    return recordKey(lhs) < recordKey(rhs);
  });
  normalized.erase(std::unique(normalized.begin(), normalized.end(),
                               [](Attribute lhs, Attribute rhs) {
                                 return recordKey(lhs) == recordKey(rhs);
                               }),
                   normalized.end());
  return ArrayAttr::get(values.getContext(), normalized);
}

std::vector<Attribute> normalizeSystemDomain(MLIRContext *context,
                                             Attribute projection,
                                             ArrayRef<Attribute> values) {
  const auto kind = static_cast<::mapping::SystemConstraintProjection>(
      cast<::mapping::SystemConstraintProjectionKeyAttr>(projection)
          .getValue());
  if (kind == ::mapping::SystemConstraintProjection::TransferAssignedTagValues)
    return detail::normalizeUnsignedIntervalConstraintDomain(context, values);
  return detail::normalizeExactConstraintDomain(values);
}

std::vector<Attribute> intersectSystemDomains(MLIRContext *context,
                                              Attribute projection,
                                              ArrayRef<Attribute> lhs,
                                              ArrayRef<Attribute> rhs) {
  const auto kind = static_cast<::mapping::SystemConstraintProjection>(
      cast<::mapping::SystemConstraintProjectionKeyAttr>(projection)
          .getValue());
  if (kind == ::mapping::SystemConstraintProjection::TransferAssignedTagValues)
    return detail::intersectUnsignedIntervalConstraintDomains(context, lhs,
                                                              rhs);
  return detail::intersectExactConstraintDomains(lhs, rhs);
}

bool isGraphMappingProjection(Operation &operation) {
  auto projection =
      dyn_cast_or_null<::mapping::SystemConstraintProjectionKeyAttr>(
          operation.getAttr("projection"));
  return projection &&
         static_cast<::mapping::SystemConstraintProjection>(
             projection.getValue()) ==
             ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping;
}

struct ProvisionalSpatialMappingTable final {
  std::vector<ArtifactRootReference> references;
  std::vector<::mapping::ArtifactRootReferenceAttr> attributes;
};

llvm::Expected<ProvisionalSpatialMappingTable>
remapSpatialMappingReferenceOrdinals(::mapping::ConstraintsSystemOp root) {
  struct TableEntry final {
    ArtifactRootReference reference;
    ::mapping::ArtifactRootReferenceAttr attribute;
  };
  std::vector<TableEntry> authored;
  authored.reserve(root.getSpatialMappingReferenceTable().size());
  for (Attribute attribute : root.getSpatialMappingReferenceTable()) {
    auto typed = cast<::mapping::ArtifactRootReferenceAttr>(attribute);
    auto reference = decodeRootReference(typed);
    if (!reference)
      return reference.takeError();
    if (reference->schemaIdentity != mappingArtifactSchema.identity ||
        reference->schemaVersion != mappingArtifactSchema.version)
      return invalid(
          "spatial mapping reference table contains the wrong schema");
    authored.push_back({std::move(*reference), typed});
  }

  std::vector<TableEntry> canonical = authored;
  llvm::sort(canonical, [](const TableEntry &lhs, const TableEntry &rhs) {
    return artifactRootReferenceLess(lhs.reference, rhs.reference);
  });
  canonical.erase(std::unique(canonical.begin(), canonical.end(),
                              [](const TableEntry &lhs, const TableEntry &rhs) {
                                return lhs.reference == rhs.reference;
                              }),
                  canonical.end());

  std::vector<std::uint64_t> oldToProvisional;
  oldToProvisional.reserve(authored.size());
  for (const TableEntry &entry : authored) {
    const auto found =
        llvm::find_if(canonical, [&](const TableEntry &candidate) {
          return candidate.reference == entry.reference;
        });
    oldToProvisional.push_back(
        static_cast<std::uint64_t>(std::distance(canonical.begin(), found)));
  }

  for (Operation &operation : root.getBody().front()) {
    auto restriction =
        dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation);
    if (!restriction || !isGraphMappingProjection(operation))
      continue;
    SmallVector<Attribute> remapped;
    remapped.reserve(restriction.getAdmissibleDomain().size());
    for (Attribute value : restriction.getAdmissibleDomain()) {
      const std::uint64_t ordinal =
          cast<::mapping::ConstraintSpatialMappingReferenceAttr>(value)
              .getOrdinal();
      if (ordinal >= oldToProvisional.size())
        return invalid("SpatialMapping table ordinal is out of range");
      remapped.push_back(::mapping::ConstraintSpatialMappingReferenceAttr::get(
          root.getContext(), oldToProvisional[ordinal]));
    }
    restriction->setAttr("admissible_domain",
                         ArrayAttr::get(root.getContext(), remapped));
  }

  ProvisionalSpatialMappingTable result;
  result.references.reserve(canonical.size());
  result.attributes.reserve(canonical.size());
  for (TableEntry &entry : canonical) {
    result.references.push_back(std::move(entry.reference));
    result.attributes.push_back(entry.attribute);
  }
  return result;
}

llvm::Error deriveSpatialMappingReferenceTable(
    ::mapping::ConstraintsSystemOp root,
    const ProvisionalSpatialMappingTable &provisional) {
  std::set<std::uint64_t> used;
  for (Operation &operation : root.getBody().front()) {
    auto restriction =
        dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation);
    if (!restriction || !isGraphMappingProjection(operation))
      continue;
    for (Attribute value : restriction.getAdmissibleDomain()) {
      const std::uint64_t ordinal =
          cast<::mapping::ConstraintSpatialMappingReferenceAttr>(value)
              .getOrdinal();
      if (ordinal >= provisional.references.size())
        return invalid(
            "canonical SpatialMapping table ordinal is out of range");
      used.insert(ordinal);
    }
  }

  std::vector<std::uint64_t> provisionalToFinal(provisional.references.size());
  std::vector<Attribute> table;
  table.reserve(used.size());
  for (const std::uint64_t ordinal : used) {
    provisionalToFinal[ordinal] = table.size();
    table.push_back(provisional.attributes[ordinal]);
  }
  for (Operation &operation : root.getBody().front()) {
    auto restriction =
        dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation);
    if (!restriction || !isGraphMappingProjection(operation))
      continue;
    SmallVector<Attribute> remapped;
    remapped.reserve(restriction.getAdmissibleDomain().size());
    for (Attribute value : restriction.getAdmissibleDomain()) {
      const std::uint64_t ordinal =
          cast<::mapping::ConstraintSpatialMappingReferenceAttr>(value)
              .getOrdinal();
      remapped.push_back(::mapping::ConstraintSpatialMappingReferenceAttr::get(
          root.getContext(), provisionalToFinal[ordinal]));
    }
    restriction->setAttr("admissible_domain",
                         ArrayAttr::get(root.getContext(), remapped));
  }
  root->setAttr("spatial_mapping_reference_table",
                ArrayAttr::get(root.getContext(), table));
  return llvm::Error::success();
}

struct SystemConstraintScope final {
  std::vector<OperationServiceObligationFamilyKey> operations;
  std::vector<CanonicalServiceLegKey> legs;
  std::vector<SystemTransferTerminalKey> terminals;
};

llvm::Expected<SystemConstraintScope>
buildConstraintScope(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  auto obligations = projectSystemServiceObligations(dataflow, roots);
  if (!obligations)
    return obligations.takeError();
  SystemConstraintScope result;
  for (const SystemServiceObligationProjection &obligation : *obligations) {
    if (const auto *operation =
            std::get_if<OperationServiceObligationFamilyKey>(&obligation.key))
      result.operations.push_back(*operation);
    for (const CanonicalServiceLegKey &leg : obligation.legs) {
      result.legs.push_back(leg);
      result.terminals.push_back(SystemTransferSourceTerminalKey{leg});
      const std::size_t sinkCount =
          std::holds_alternative<TransferObligationFamilyKey>(obligation.key)
              ? obligation.sinks.size()
              : 1;
      for (std::size_t ordinal = 0; ordinal < sinkCount; ++ordinal)
        result.terminals.push_back(SystemTransferSinkTerminalKey{
            leg, static_cast<::dataflow::StructuralOrdinal>(ordinal)});
    }
  }
  return result;
}

llvm::Expected<SystemConstraintSubject>
decodeSystemSubject(::mapping::SystemConstraintProjection projection,
                    Attribute attribute,
                    const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    ArrayRef<::dataflow::RootThreadLaunchRef> roots,
                    const SystemConstraintScope &scope) {
  using Projection = ::mapping::SystemConstraintProjection;
  switch (projection) {
  case Projection::ThreadTargetAccCore: {
    auto subject = decodeDataflow<::dataflow::RootThreadLaunchRef>(
        cast<::mapping::RootThreadLaunchRefAttr>(attribute),
        dataflow.identity());
    if (!subject)
      return contextual(subject.takeError(),
                        "constraint thread subject is malformed");
    if (!llvm::is_contained(roots, *subject))
      return invalid("constraint thread subject is outside the root scope");
    return SystemConstraintSubject(std::move(*subject));
  }
  case Projection::GraphSelectedSpatialMapping:
  case Projection::GraphTargetSpatialCore: {
    auto subject = decodeDataflow<::dataflow::RootedGraphLaunchRef>(
        cast<::mapping::RootedGraphLaunchRefAttr>(attribute),
        dataflow.identity());
    if (!subject)
      return contextual(subject.takeError(),
                        "constraint graph subject is malformed");
    auto resolved = dataflow.resolve(*subject);
    if (!resolved)
      return contextual(resolved.takeError(),
                        "constraint graph subject does not resolve");
    if (!llvm::is_contained(roots, subject->rootThreadLaunch))
      return invalid("constraint graph subject is outside the root scope");
    return SystemConstraintSubject(std::move(*subject));
  }
  case Projection::ServiceTargetRegion: {
    auto subject = decodeSystemServiceObligationKey(
        unsignedBytes(cast<::mapping::SystemServiceObligationKeyAttr>(attribute)
                          .getRecord()),
        dataflow.identity());
    if (!subject)
      return contextual(subject.takeError(),
                        "constraint service subject is malformed");
    const auto *operation =
        std::get_if<OperationServiceObligationFamilyKey>(&*subject);
    if (!operation || !llvm::is_contained(scope.operations, *operation))
      return invalid(
          "constraint service subject is outside the operation-service scope");
    return SystemConstraintSubject(*operation);
  }
  case Projection::TransferTerminalAttachment: {
    auto subject = decodeSystemTransferTerminalKey(
        unsignedBytes(cast<::mapping::SystemTransferTerminalKeyAttr>(attribute)
                          .getRecord()),
        dataflow.identity());
    if (!subject)
      return contextual(subject.takeError(),
                        "constraint transfer terminal is malformed");
    if (!llvm::is_contained(scope.terminals, *subject))
      return invalid("constraint transfer terminal is outside the root scope");
    return SystemConstraintSubject(std::move(*subject));
  }
  case Projection::TransferSelectedTraversals:
  case Projection::TransferResourceStates:
  case Projection::TransferAssignedTagValues: {
    auto subject = decodeCanonicalServiceLegKey(
        unsignedBytes(
            cast<::mapping::CanonicalServiceLegKeyAttr>(attribute).getRecord()),
        dataflow.identity());
    if (!subject)
      return contextual(subject.takeError(),
                        "constraint service leg is malformed");
    if (!llvm::is_contained(scope.legs, *subject))
      return invalid("constraint service leg is outside the root scope");
    return SystemConstraintSubject(std::move(*subject));
  }
  }
  llvm_unreachable("unknown System constraint projection");
}

template <typename Ref, typename Attr>
llvm::Expected<Ref>
decodeValidatedFabric(Attr attribute,
                      const ::loom::fabric::FabricArtifactView &fabric,
                      const llvm::Twine &description) {
  auto reference = decodeFabric<Ref>(attribute);
  if (!reference)
    return contextual(reference.takeError(), description + " is malformed");
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *reference))
    return contextual(std::move(error), description + " does not resolve");
  return std::move(*reference);
}

llvm::Expected<SystemConstraintDomainValue>
decodeSystemDomainValue(::mapping::SystemConstraintProjection projection,
                        Attribute attribute,
                        const ::loom::fabric::FabricArtifactView &fabric,
                        ArrayRef<ArtifactRootReference> spatialMappings) {
  using Projection = ::mapping::SystemConstraintProjection;
  switch (projection) {
  case Projection::ThreadTargetAccCore: {
    auto value = decodeValidatedFabric<::loom::fabric::AccCoreOccurrenceRef>(
        cast<::mapping::FabricAccCoreOccurrenceRefAttr>(attribute), fabric,
        "constraint AccCore occurrence");
    if (!value)
      return value.takeError();
    return SystemConstraintDomainValue(std::move(*value));
  }
  case Projection::GraphSelectedSpatialMapping: {
    const std::uint64_t ordinal =
        cast<::mapping::ConstraintSpatialMappingReferenceAttr>(attribute)
            .getOrdinal();
    if (ordinal >= spatialMappings.size())
      return invalid("constraint SpatialMapping table ordinal is out of range");
    return SystemConstraintDomainValue(spatialMappings[ordinal]);
  }
  case Projection::GraphTargetSpatialCore: {
    auto value =
        decodeValidatedFabric<::loom::fabric::SpatialCoreOccurrenceRef>(
            cast<::mapping::FabricSpatialCoreOccurrenceRefAttr>(attribute),
            fabric, "constraint SpatialCore occurrence");
    if (!value)
      return value.takeError();
    return SystemConstraintDomainValue(std::move(*value));
  }
  case Projection::ServiceTargetRegion: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricMemoryServiceRegionRef>(
            cast<::mapping::FabricMemoryServiceRegionRefAttr>(attribute),
            fabric, "constraint memory service region");
    if (!value)
      return value.takeError();
    return SystemConstraintDomainValue(std::move(*value));
  }
  case Projection::TransferTerminalAttachment: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricTransportEndpointRef>(
            cast<::mapping::FabricTransportEndpointRefAttr>(attribute), fabric,
            "constraint transport endpoint");
    if (!value)
      return value.takeError();
    return SystemConstraintDomainValue(std::move(*value));
  }
  case Projection::TransferSelectedTraversals: {
    auto value =
        decodeValidatedFabric<::loom::fabric::FabricPhysicalTraversalRef>(
            cast<::mapping::FabricPhysicalTraversalRefAttr>(attribute), fabric,
            "constraint physical traversal");
    if (!value)
      return value.takeError();
    return SystemConstraintDomainValue(std::move(*value));
  }
  case Projection::TransferResourceStates: {
    auto value = decodeValidatedFabric<::loom::fabric::FabricResourceStateRef>(
        cast<::mapping::FabricResourceStateRefAttr>(attribute), fabric,
        "constraint resource state");
    if (!value)
      return value.takeError();
    return SystemConstraintDomainValue(std::move(*value));
  }
  case Projection::TransferAssignedTagValues: {
    auto interval = cast<::mapping::ConstraintUnsignedIntervalAttr>(attribute);
    return SystemConstraintDomainValue(SpatialConstraintUnsignedInterval{
        interval.getLower().getValue(), interval.getUpper().getValue()});
  }
  }
  llvm_unreachable("unknown System constraint projection");
}

llvm::Expected<std::vector<SystemConstraintSubject>>
decodeSystemSubjects(::mapping::SystemConstraintProjection projection,
                     ArrayAttr attributes,
                     const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     ArrayRef<::dataflow::RootThreadLaunchRef> roots,
                     const SystemConstraintScope &scope) {
  std::vector<SystemConstraintSubject> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto subject =
        decodeSystemSubject(projection, attribute, dataflow, roots, scope);
    if (!subject)
      return subject.takeError();
    result.push_back(std::move(*subject));
  }
  return result;
}

llvm::Expected<std::vector<SystemConstraintDomainValue>>
decodeSystemDomain(::mapping::SystemConstraintProjection projection,
                   ArrayAttr attributes,
                   const ::loom::fabric::FabricArtifactView &fabric,
                   ArrayRef<ArtifactRootReference> spatialMappings) {
  std::vector<SystemConstraintDomainValue> result;
  result.reserve(attributes.size());
  for (Attribute attribute : attributes) {
    auto value =
        decodeSystemDomainValue(projection, attribute, fabric, spatialMappings);
    if (!value)
      return value.takeError();
    result.push_back(std::move(*value));
  }
  return result;
}

struct ParsedSystemConstraintRoot final {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::ConstraintsSystemOp root;
};

llvm::Expected<ParsedSystemConstraintRoot>
parseSystemConstraintRoot(const CanonicalSemanticBytes &canonicalBytes) {
  std::string wrapped = "module {\n";
  wrapped.append(reinterpret_cast<const char *>(canonicalBytes.bytes().data()),
                 canonicalBytes.bytes().size());
  wrapped += "}\n";

  DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  auto context =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  auto module = parseSourceString<ModuleOp>(wrapped, context.get());
  if (!module)
    return invalid("canonical System constraint payload cannot be parsed");

  ::mapping::ConstraintsSystemOp root;
  unsigned rootCount = 0;
  for (Operation &operation : module->getBody()->without_terminator()) {
    auto candidate = dyn_cast<::mapping::ConstraintsSystemOp>(operation);
    if (!candidate)
      return invalid("constraint artifact contains a non-System root");
    root = candidate;
    ++rootCount;
  }
  if (rootCount != 1)
    return invalid("constraint artifact must contain exactly one System root");
  if (failed(verify(root)))
    return invalid("System constraint root is structurally invalid");
  return ParsedSystemConstraintRoot{std::move(context), std::move(module),
                                    root};
}

struct PreparedSystemConstraintSet final {
  ArtifactRootReference reference;
  CanonicalSemanticBytes canonicalBytes;
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  ::mapping::ConstraintsSystemOp root;
};

llvm::Expected<PreparedSystemConstraintSet>
prepareSystemConstraintSet(::mapping::ConstraintsSystemOp source) {
  auto canonicalBytes = writeCanonicalSystemConstraintAssembly(source);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  auto parsed = parseSystemConstraintRoot(*canonicalBytes);
  if (!parsed)
    return parsed.takeError();
  ArtifactRootReference reference{
      mappingConstraintSetSchema.identity.str(),
      mappingConstraintSetSchema.version,
      finalizeArtifactIdentity(mappingConstraintSetSchema, *canonicalBytes)};
  return PreparedSystemConstraintSet{
      std::move(reference), std::move(*canonicalBytes),
      std::move(parsed->context), std::move(parsed->module), parsed->root};
}

llvm::Error requirePublishedUpstreams(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store) {
  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, dataflow.identity()};
  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      fabric.artifact().identity()};
  auto dataflowBytes = store.get(dataflowReference);
  if (!dataflowBytes)
    return dataflowBytes.takeError();
  auto fabricBytes = store.get(fabricReference);
  if (!fabricBytes)
    return fabricBytes.takeError();
  return llvm::Error::success();
}

llvm::Expected<SystemMappingConstraintSetView>
strictImport(const ArtifactIdentity &identity,
             const CanonicalSemanticBytes &canonicalBytes,
             const ArtifactStore &store) {
  if (finalizeArtifactIdentity(mappingConstraintSetSchema, canonicalBytes) !=
      identity)
    return invalid("constraint identity does not match canonical bytes");
  auto parsed = parseSystemConstraintRoot(canonicalBytes);
  if (!parsed)
    return parsed.takeError();

  auto dataflowIdentity = decodeIdentity(parsed->root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  auto fabricIdentity = decodeIdentity(parsed->root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();

  ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();

  ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version, *fabricIdentity};
  auto fabric = ::loom::fabric::importEntireFabricRoot(fabricReference, store);
  if (!fabric)
    return fabric.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabric->view());
  if (!system)
    return system.takeError();

  auto view = SystemMappingConstraintSetView::import(
      identity, parsed->root, *dataflowView, *system, store);
  if (!view)
    return view.takeError();
  auto rewritten = writeCanonicalSystemConstraintAssembly(parsed->root);
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(canonicalBytes.bytes()))
    return invalid("stored System constraint payload is not canonical");
  return view;
}

llvm::Error
publishPreparedSystemConstraintSet(const PreparedSystemConstraintSet &prepared,
                                   const ArtifactStore &store) {
  auto stored = store.put(mappingConstraintSetSchema, prepared.canonicalBytes);
  if (!stored)
    return stored.takeError();
  if (*stored != prepared.reference.artifact)
    return invalid("ArtifactStore returned a different constraint identity");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSystemConstraintAssembly(::mapping::ConstraintsSystemOp root) {
  OwningOpRef<Operation *> clone(root->clone());
  auto canonical = cast<::mapping::ConstraintsSystemOp>(clone.get());
  if (failed(verify(canonical)))
    return invalid("System MappingConstraintSet is structurally invalid");

  canonical->setAttr(
      "root_thread_launches",
      normalizeReferenceArray(canonical.getRootThreadLaunches()));
  auto provisional = remapSpatialMappingReferenceOrdinals(canonical);
  if (!provisional)
    return provisional.takeError();
  detail::canonicalizeConstraintClauses(
      canonical.getBody().front(), canonical.getLoc(), normalizeSystemDomain,
      intersectSystemDomains);
  if (llvm::Error error =
          deriveSpatialMappingReferenceTable(canonical, *provisional))
    return std::move(error);
  if (failed(verify(canonical)))
    return invalid(
        "canonical System MappingConstraintSet is structurally invalid");

  std::string text;
  llvm::raw_string_ostream stream(text);
  canonical.print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
}

llvm::Expected<SystemMappingConstraintSetView>
SystemMappingConstraintSetView::import(
    const ArtifactIdentity &identity, ::mapping::ConstraintsSystemOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store) {
  auto dataflowIdentity = decodeIdentity(root.getDataflow());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  auto fabricIdentity = decodeIdentity(root.getFabric());
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  if (*dataflowIdentity != dataflow.identity())
    return invalid(
        "System constraint dataflow binding does not match importer");
  if (*fabricIdentity != fabric.artifact().identity())
    return invalid("System constraint Fabric binding does not match importer");

  std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches;
  rootThreadLaunches.reserve(root.getRootThreadLaunches().size());
  for (Attribute attribute : root.getRootThreadLaunches()) {
    auto decoded =
        ::dataflow::decodeDataflowReference<::dataflow::RootThreadLaunchRef>(
            unsignedBytes(cast<::mapping::RootThreadLaunchRefAttr>(attribute)
                              .getRecord()),
            dataflow.identity());
    if (!decoded)
      return decoded.takeError();
    auto resolved = dataflow.resolve(*decoded);
    if (!resolved)
      return llvm::joinErrors(
          invalid("root thread launch does not resolve in exact dataflow"),
          resolved.takeError());
    rootThreadLaunches.push_back(std::move(*decoded));
  }
  if (rootThreadLaunches.empty())
    return invalid("System constraint root launch set is empty");

  std::vector<ArtifactRootReference> spatialMappingReferences;
  spatialMappingReferences.reserve(
      root.getSpatialMappingReferenceTable().size());
  for (Attribute attribute : root.getSpatialMappingReferenceTable()) {
    auto decoded = decodeRootReference(
        cast<::mapping::ArtifactRootReferenceAttr>(attribute));
    if (!decoded)
      return decoded.takeError();
    if (decoded->schemaIdentity != mappingArtifactSchema.identity ||
        decoded->schemaVersion != mappingArtifactSchema.version)
      return invalid(
          "spatial mapping reference table contains the wrong schema");
    auto mapping = importSpatialMapping(*decoded, store);
    if (!mapping)
      return contextual(mapping.takeError(),
                        "spatial mapping reference cannot be imported");
    if (mapping->view().dataflowIdentity() != dataflow.identity())
      return invalid("spatial mapping reference has a foreign Dataflow owner");
    const bool attachedModule = llvm::any_of(
        fabric.artifact().accCoreOccurrences(), [&](const auto core) {
          const auto target = fabric.spatialCoreTarget(core);
          return target &&
                 target->dependencyOrdinal <
                     fabric.artifact().importedModules().size() &&
                 fabric.artifact()
                         .importedModules()[target->dependencyOrdinal]
                         .identity() == mapping->view().fabricIdentity();
        });
    if (!attachedModule)
      return invalid(
          "spatial mapping reference Fabric is not an attached Module");
    spatialMappingReferences.push_back(std::move(*decoded));
  }

  auto scope = buildConstraintScope(dataflow, rootThreadLaunches);
  if (!scope)
    return contextual(scope.takeError(),
                      "cannot derive the System constraint subject scope");

  std::vector<SystemConstraintClauseView> clauses;
  clauses.reserve(std::distance(root.getBody().front().begin(),
                                root.getBody().front().end()));
  for (Operation &operation : root.getBody().front()) {
    auto projectionAttribute =
        cast<::mapping::SystemConstraintProjectionKeyAttr>(
            operation.getAttr("projection"));
    const auto projection = static_cast<::mapping::SystemConstraintProjection>(
        projectionAttribute.getValue());
    if (auto restriction =
            dyn_cast<::mapping::ConstraintDomainRestrictionOp>(operation)) {
      auto subject = decodeSystemSubject(projection, restriction.getSubject(),
                                         dataflow, rootThreadLaunches, *scope);
      if (!subject)
        return subject.takeError();
      auto domain =
          decodeSystemDomain(projection, restriction.getAdmissibleDomain(),
                             fabric.artifact(), spatialMappingReferences);
      if (!domain)
        return domain.takeError();
      clauses.emplace_back(SystemDomainRestrictionView{
          projection, std::move(*subject), std::move(*domain)});
      continue;
    }
    if (auto equal = dyn_cast<::mapping::ConstraintEqualOp>(operation)) {
      auto subjects =
          decodeSystemSubjects(projection, equal.getSubjects(), dataflow,
                               rootThreadLaunches, *scope);
      if (!subjects)
        return subjects.takeError();
      clauses.emplace_back(SystemEqualView{projection, std::move(*subjects)});
      continue;
    }
    auto disjoint = cast<::mapping::ConstraintDisjointOp>(operation);
    auto subjects = decodeSystemSubjects(projection, disjoint.getSubjects(),
                                         dataflow, rootThreadLaunches, *scope);
    if (!subjects)
      return subjects.takeError();
    clauses.emplace_back(SystemDisjointView{projection, std::move(*subjects)});
  }

  return SystemMappingConstraintSetView(
      identity, std::move(*dataflowIdentity), std::move(*fabricIdentity),
      std::move(rootThreadLaunches), std::move(spatialMappingReferences),
      std::move(clauses));
}

llvm::Expected<FinalizedSystemMappingConstraintSet>
finalizeSystemMappingConstraintSet(::mapping::ConstraintsSystemOp source,
                                   const ArtifactStore &store) {
  auto prepared = prepareSystemConstraintSet(source);
  if (!prepared)
    return prepared.takeError();
  auto view = strictImport(prepared->reference.artifact,
                           prepared->canonicalBytes, store);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedSystemConstraintSet(*prepared, store))
    return std::move(error);
  return FinalizedSystemMappingConstraintSet(
      std::move(prepared->reference), std::move(prepared->canonicalBytes),
      std::move(*view));
}

llvm::Expected<FinalizedSystemMappingConstraintSet>
finalizeSystemMappingConstraintSet(
    ::mapping::ConstraintsSystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store) {
  if (llvm::Error error = requirePublishedUpstreams(dataflow, fabric, store))
    return std::move(error);
  auto prepared = prepareSystemConstraintSet(source);
  if (!prepared)
    return prepared.takeError();
  auto view = SystemMappingConstraintSetView::import(
      prepared->reference.artifact, prepared->root, dataflow, fabric, store);
  if (!view)
    return view.takeError();
  if (llvm::Error error = publishPreparedSystemConstraintSet(*prepared, store))
    return std::move(error);
  return FinalizedSystemMappingConstraintSet(
      std::move(prepared->reference), std::move(prepared->canonicalBytes),
      std::move(*view));
}

llvm::Expected<FinalizedSystemMappingConstraintSet>
importSystemMappingConstraintSet(const ArtifactRootReference &reference,
                                 const ArtifactStore &store) {
  if (reference.schemaIdentity != mappingConstraintSetSchema.identity ||
      reference.schemaVersion != mappingConstraintSetSchema.version)
    return invalid("root reference has the wrong constraint schema");
  auto canonicalBytes = store.get(reference);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  auto view = strictImport(reference.artifact, *canonicalBytes, store);
  if (!view)
    return view.takeError();
  return FinalizedSystemMappingConstraintSet(
      reference, std::move(*canonicalBytes), std::move(*view));
}

} // namespace loom::mapping
