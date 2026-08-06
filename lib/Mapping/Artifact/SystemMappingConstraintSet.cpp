#include "Mapping/Artifact/SystemMappingConstraintSet.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
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
  canonical->setAttr(
      "spatial_mapping_reference_table",
      normalizeReferenceArray(canonical.getSpatialMappingReferenceTable()));
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
    spatialMappingReferences.push_back(std::move(*decoded));
  }

  const std::uint64_t clauseCount =
      root.getBody().front().getOperations().size();
  if (!spatialMappingReferences.empty() && clauseCount == 0)
    return invalid("spatial mapping reference table contains unused rows");
  (void)store;

  return SystemMappingConstraintSetView(
      identity, std::move(*dataflowIdentity), std::move(*fabricIdentity),
      std::move(rootThreadLaunches), std::move(spatialMappingReferences),
      clauseCount);
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
