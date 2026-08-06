#include "SystemPnrSearchDomainInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {

namespace detail {
struct SystemPnrSearchDomainViewBuilder final {
  static SystemPnrSearchDomainView
  create(ArtifactRootReference dataflowReference,
         ArtifactRootReference fabricReference,
         ArtifactRootReference constraintReference,
         std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
         std::vector<SystemSearchBindingDomain> bindings,
         std::vector<std::uint8_t> canonicalViewBytes,
         SystemPnrSearchDomainDigest digest) {
    return SystemPnrSearchDomainView(
        std::move(dataflowReference), std::move(fabricReference),
        std::move(constraintReference), std::move(rootThreadLaunches),
        std::move(bindings), std::move(canonicalViewBytes), std::move(digest));
  }
};
} // namespace detail

namespace {

constexpr char kSchemaDescriptor[] = "loom.system_pnr_search_domain.1.0";
constexpr char kDigestDomain[] = "loom.system.pnr.search.domain.digest.v1\0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_pnr_search_domain_invalid: " +
                                     message);
}

class WireWriter final {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }
  void bytes(llvm::ArrayRef<std::uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void sizedBytes(llvm::ArrayRef<std::uint8_t> value) {
    u64(value.size());
    bytes(value);
  }
  void rootReference(const ArtifactRootReference &reference) {
    bytes(encodeArtifactRootReference(reference));
  }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class WireReader final {
public:
  explicit WireReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    if (bytes_.size() < 4)
      return invalid("truncated u32 field");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    if (bytes_.size() < 8)
      return invalid("truncated u64 field");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[index];
    bytes_ = bytes_.drop_front(8);
    return value;
  }

  llvm::Expected<std::int64_t> i64() {
    auto value = u64();
    if (!value)
      return value.takeError();
    return static_cast<std::int64_t>(*value);
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> take(std::uint64_t size) {
    if (size > bytes_.size())
      return invalid("truncated byte field");
    llvm::ArrayRef<std::uint8_t> value = bytes_.take_front(size);
    bytes_ = bytes_.drop_front(size);
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> sizedBytes() {
    auto size = u64();
    if (!size)
      return size.takeError();
    return take(*size);
  }

  llvm::Expected<ArtifactRootReference> rootReference() {
    auto decoded = decodeArtifactRootReferencePrefix(bytes_);
    if (!decoded)
      return decoded.takeError();
    bytes_ = bytes_.drop_front(decoded->byteCount);
    return std::move(decoded->reference);
  }

  llvm::Expected<std::size_t> count(std::size_t minimumElementBytes,
                                    const llvm::Twine &what) {
    auto value = u64();
    if (!value)
      return value.takeError();
    if (*value > std::numeric_limits<std::size_t>::max() ||
        (minimumElementBytes != 0 &&
         *value > bytes_.size() / minimumElementBytes))
      return invalid(what + " count exceeds remaining bytes");
    return static_cast<std::size_t>(*value);
  }

  bool empty() const { return bytes_.empty(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

ArtifactRootReference dataflowRootReference(
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  return {::dataflow::canonicalDataflowSchema.identity.str(),
          ::dataflow::canonicalDataflowSchema.version, dataflow.identity()};
}

ArtifactRootReference
fabricRootReference(const ::loom::fabric::FabricSystemRootView &fabric) {
  return {::loom::fabric::fabricArtifactSchema.identity.str(),
          ::loom::fabric::fabricArtifactSchema.version,
          fabric.artifact().identity()};
}

void encodeCell(WireWriter &writer, const SystemPresburgerCell &cell) {
  writer.u32(cell.dimensionCount);
  writer.u32(cell.symbolCount);
  writer.u64(cell.equalities.size());
  for (const std::vector<std::int64_t> &row : cell.equalities)
    for (std::int64_t value : row)
      writer.i64(value);
  writer.u64(cell.inequalities.size());
  for (const std::vector<std::int64_t> &row : cell.inequalities)
    for (std::int64_t value : row)
      writer.i64(value);
}

llvm::Expected<SystemPresburgerCell> decodeCell(WireReader &reader) {
  auto dimensions = reader.u32();
  if (!dimensions)
    return dimensions.takeError();
  auto symbols = reader.u32();
  if (!symbols)
    return symbols.takeError();
  const std::uint64_t rowWidth =
      static_cast<std::uint64_t>(*dimensions) + *symbols + 1;
  if (rowWidth > std::numeric_limits<std::size_t>::max() / 8)
    return invalid("Presburger row width exceeds native range");
  const std::size_t minimumRowBytes = static_cast<std::size_t>(rowWidth) * 8;

  SystemPresburgerCell cell;
  cell.dimensionCount = *dimensions;
  cell.symbolCount = *symbols;
  auto equalities = reader.count(minimumRowBytes, "equality row");
  if (!equalities)
    return equalities.takeError();
  cell.equalities.resize(*equalities);
  for (std::vector<std::int64_t> &row : cell.equalities) {
    row.reserve(rowWidth);
    for (std::uint64_t column = 0; column < rowWidth; ++column) {
      auto value = reader.i64();
      if (!value)
        return value.takeError();
      row.push_back(*value);
    }
  }
  auto inequalities = reader.count(minimumRowBytes, "inequality row");
  if (!inequalities)
    return inequalities.takeError();
  cell.inequalities.resize(*inequalities);
  for (std::vector<std::int64_t> &row : cell.inequalities) {
    row.reserve(rowWidth);
    for (std::uint64_t column = 0; column < rowWidth; ++column) {
      auto value = reader.i64();
      if (!value)
        return value.takeError();
      row.push_back(*value);
    }
  }
  return cell;
}

template <typename Ref>
llvm::Error encodeFabricDomain(WireWriter &writer, llvm::ArrayRef<Ref> values) {
  writer.u64(values.size());
  for (const Ref &value : values)
    writer.sizedBytes(::loom::fabric::canonicalFabricBytes(value));
  return llvm::Error::success();
}

void encodeRootDomain(WireWriter &writer,
                      llvm::ArrayRef<ArtifactRootReference> values) {
  writer.u64(values.size());
  for (const ArtifactRootReference &value : values)
    writer.rootReference(value);
}

llvm::Error encodeDomains(WireWriter &writer,
                          const SystemSearchAtomDomains &domains) {
  std::uint32_t presence = 0;
  if (domains.compatibleAccCores)
    presence |= 1u << 0;
  if (domains.compatibleSpatialMappings)
    presence |= 1u << 1;
  if (domains.compatibleServiceRegions)
    presence |= 1u << 2;
  if (domains.compatibleTransportEndpoints)
    presence |= 1u << 3;
  writer.u32(presence);
  if (domains.compatibleAccCores)
    if (llvm::Error error = encodeFabricDomain(
            writer, llvm::ArrayRef(*domains.compatibleAccCores)))
      return error;
  if (domains.compatibleSpatialMappings)
    encodeRootDomain(writer, *domains.compatibleSpatialMappings);
  if (domains.compatibleServiceRegions)
    if (llvm::Error error = encodeFabricDomain(
            writer, llvm::ArrayRef(*domains.compatibleServiceRegions)))
      return error;
  if (domains.compatibleTransportEndpoints)
    if (llvm::Error error = encodeFabricDomain(
            writer, llvm::ArrayRef(*domains.compatibleTransportEndpoints)))
      return error;
  return llvm::Error::success();
}

template <typename Ref>
llvm::Expected<std::vector<Ref>>
decodeFabricDomain(WireReader &reader,
                   const ::loom::fabric::FabricArtifactView &fabric) {
  auto count = reader.count(/*minimumElementBytes=*/8, "Fabric target");
  if (!count)
    return count.takeError();
  std::vector<Ref> values;
  values.reserve(*count);
  std::vector<std::uint8_t> previous;
  for (std::size_t index = 0; index < *count; ++index) {
    auto bytes = reader.sizedBytes();
    if (!bytes)
      return bytes.takeError();
    auto decoded = ::loom::fabric::decodeFabricRef<Ref>(*bytes);
    if (!decoded)
      return decoded.takeError();
    const std::vector<std::uint8_t> canonical =
        ::loom::fabric::canonicalFabricBytes(*decoded);
    if (llvm::ArrayRef(canonical) != *bytes)
      return invalid("noncanonical Fabric target reference");
    if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, *decoded))
      return std::move(error);
    std::vector<std::uint8_t> key(bytes->begin(), bytes->end());
    if (!previous.empty() && !(previous < key))
      return invalid("Fabric target domain is not strictly ordered");
    previous = std::move(key);
    values.push_back(std::move(*decoded));
  }
  return values;
}

llvm::Expected<std::vector<ArtifactRootReference>>
decodeRootDomain(WireReader &reader) {
  auto count = reader.count(/*minimumElementBytes=*/44, "Artifact target");
  if (!count)
    return count.takeError();
  std::vector<ArtifactRootReference> values;
  values.reserve(*count);
  for (std::size_t index = 0; index < *count; ++index) {
    auto reference = reader.rootReference();
    if (!reference)
      return reference.takeError();
    if (!values.empty() &&
        !artifactRootReferenceLess(values.back(), *reference))
      return invalid("Artifact target domain is not strictly ordered");
    values.push_back(std::move(*reference));
  }
  return values;
}

llvm::Expected<SystemSearchAtomDomains>
decodeDomains(WireReader &reader,
              const ::loom::fabric::FabricArtifactView &fabric) {
  auto presence = reader.u32();
  if (!presence)
    return presence.takeError();
  if ((*presence & ~0xfu) != 0)
    return invalid("target-domain presence mask has unknown fields");
  SystemSearchAtomDomains domains;
  if ((*presence & (1u << 0)) != 0) {
    auto values = decodeFabricDomain<::loom::fabric::AccCoreOccurrenceRef>(
        reader, fabric);
    if (!values)
      return values.takeError();
    domains.compatibleAccCores = std::move(*values);
  }
  if ((*presence & (1u << 1)) != 0) {
    auto values = decodeRootDomain(reader);
    if (!values)
      return values.takeError();
    domains.compatibleSpatialMappings = std::move(*values);
  }
  if ((*presence & (1u << 2)) != 0) {
    auto values =
        decodeFabricDomain<::loom::fabric::FabricMemoryServiceRegionRef>(
            reader, fabric);
    if (!values)
      return values.takeError();
    domains.compatibleServiceRegions = std::move(*values);
  }
  if ((*presence & (1u << 3)) != 0) {
    auto values =
        decodeFabricDomain<::loom::fabric::FabricTransportEndpointRef>(reader,
                                                                       fabric);
    if (!values)
      return values.takeError();
    domains.compatibleTransportEndpoints = std::move(*values);
  }
  return domains;
}

template <typename Ref>
llvm::Expected<Ref>
decodeDataflowKey(WireReader &reader,
                  const ArtifactIdentity &dataflowIdentity) {
  auto bytes = reader.sizedBytes();
  if (!bytes)
    return bytes.takeError();
  auto decoded =
      ::dataflow::decodeDataflowReference<Ref>(*bytes, dataflowIdentity);
  if (!decoded)
    return decoded.takeError();
  auto canonical =
      ::dataflow::encodeDataflowReference(dataflowIdentity, *decoded);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef(*canonical) != *bytes)
    return invalid("noncanonical Dataflow binding reference");
  return *decoded;
}

llvm::Expected<SystemSearchBindingKey>
decodeBindingKey(WireReader &reader, const ArtifactIdentity &dataflowIdentity) {
  auto kind = reader.u32();
  if (!kind)
    return kind.takeError();
  if (*kind == 0) {
    auto ref = decodeDataflowKey<::dataflow::RootThreadLaunchRef>(
        reader, dataflowIdentity);
    if (!ref)
      return ref.takeError();
    return SystemSearchBindingKey(*ref);
  }
  if (*kind == 1) {
    auto ref = decodeDataflowKey<::dataflow::RootedGraphLaunchRef>(
        reader, dataflowIdentity);
    if (!ref)
      return ref.takeError();
    return SystemSearchBindingKey(*ref);
  }
  return invalid("unknown binding-key kind");
}

llvm::Error encodeBindingKey(WireWriter &writer,
                             const SystemSearchBindingKey &key,
                             const ArtifactIdentity &dataflowIdentity) {
  if (const auto *thread = std::get_if<::dataflow::RootThreadLaunchRef>(&key)) {
    writer.u32(0);
    auto bytes = ::dataflow::encodeDataflowReference(dataflowIdentity, *thread);
    if (!bytes)
      return bytes.takeError();
    writer.sizedBytes(*bytes);
    return llvm::Error::success();
  }
  writer.u32(1);
  auto bytes = ::dataflow::encodeDataflowReference(
      dataflowIdentity, std::get<::dataflow::RootedGraphLaunchRef>(key));
  if (!bytes)
    return bytes.takeError();
  writer.sizedBytes(*bytes);
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
encodeView(const ArtifactRootReference &dataflowReference,
           const ArtifactRootReference &fabricReference,
           const ArtifactRootReference &constraintReference,
           llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
           llvm::ArrayRef<SystemSearchBindingDomain> bindings) {
  WireWriter writer;
  writer.rootReference(dataflowReference);
  writer.rootReference(fabricReference);
  writer.rootReference(constraintReference);
  writer.u64(roots.size());
  for (::dataflow::RootThreadLaunchRef root : roots) {
    auto bytes =
        ::dataflow::encodeDataflowReference(dataflowReference.artifact, root);
    if (!bytes)
      return bytes.takeError();
    writer.sizedBytes(*bytes);
  }
  writer.u64(bindings.size());
  for (const SystemSearchBindingDomain &binding : bindings) {
    if (llvm::Error error =
            encodeBindingKey(writer, binding.key, dataflowReference.artifact))
      return std::move(error);
    writer.u64(binding.atoms.size());
    for (const SystemSearchAtom &atom : binding.atoms) {
      encodeCell(writer, atom.cell);
      if (llvm::Error error = encodeDomains(writer, atom.domains))
        return std::move(error);
    }
  }
  return writer.take();
}

struct DecodedView final {
  ArtifactRootReference dataflowReference;
  ArtifactRootReference fabricReference;
  ArtifactRootReference constraintReference;
  std::vector<::dataflow::RootThreadLaunchRef> roots;
  std::vector<SystemSearchBindingDomain> bindings;
};

llvm::Expected<DecodedView>
decodeView(llvm::ArrayRef<std::uint8_t> bytes,
           const ::loom::fabric::FabricArtifactView &fabric) {
  WireReader reader(bytes);
  auto dataflowReference = reader.rootReference();
  if (!dataflowReference)
    return dataflowReference.takeError();
  auto fabricReference = reader.rootReference();
  if (!fabricReference)
    return fabricReference.takeError();
  auto constraintReference = reader.rootReference();
  if (!constraintReference)
    return constraintReference.takeError();

  auto rootCount = reader.count(/*minimumElementBytes=*/16, "root launch");
  if (!rootCount)
    return rootCount.takeError();
  std::vector<::dataflow::RootThreadLaunchRef> roots;
  roots.reserve(*rootCount);
  for (std::size_t index = 0; index < *rootCount; ++index) {
    auto root = decodeDataflowKey<::dataflow::RootThreadLaunchRef>(
        reader, dataflowReference->artifact);
    if (!root)
      return root.takeError();
    roots.push_back(*root);
  }

  auto bindingCount = reader.count(/*minimumElementBytes=*/20, "binding");
  if (!bindingCount)
    return bindingCount.takeError();
  std::vector<SystemSearchBindingDomain> bindings;
  bindings.reserve(*bindingCount);
  for (std::size_t index = 0; index < *bindingCount; ++index) {
    auto key = decodeBindingKey(reader, dataflowReference->artifact);
    if (!key)
      return key.takeError();
    auto atomCount = reader.count(/*minimumElementBytes=*/20, "binding atom");
    if (!atomCount)
      return atomCount.takeError();
    SystemSearchBindingDomain binding{std::move(*key), {}};
    binding.atoms.reserve(*atomCount);
    for (std::size_t atomIndex = 0; atomIndex < *atomCount; ++atomIndex) {
      auto cell = decodeCell(reader);
      if (!cell)
        return cell.takeError();
      auto domains = decodeDomains(reader, fabric);
      if (!domains)
        return domains.takeError();
      binding.atoms.push_back({std::move(*cell), std::move(*domains)});
    }
    bindings.push_back(std::move(binding));
  }
  if (!reader.empty())
    return invalid("trailing canonical view bytes");
  return DecodedView{std::move(*dataflowReference), std::move(*fabricReference),
                     std::move(*constraintReference), std::move(roots),
                     std::move(bindings)};
}

struct SpatialCatalogEntry final {
  ArtifactRootReference reference;
  ArtifactIdentity fabricIdentity;
  std::vector<::dataflow::GraphRef> covers;
};

llvm::Expected<std::vector<SpatialCatalogEntry>>
importSpatialCatalog(llvm::ArrayRef<ArtifactRootReference> references,
                     const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     const ::loom::fabric::FabricSystemRootView &system,
                     const ArtifactStore &store) {
  std::vector<ArtifactRootReference> canonical(references.begin(),
                                               references.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  canonical.erase(std::unique(canonical.begin(), canonical.end()),
                  canonical.end());

  std::vector<ArtifactIdentity> attachedModules;
  for (::loom::fabric::AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    std::optional<::loom::fabric::FabricImportedModuleTargetRef> target =
        system.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= system.artifact().importedModules().size())
      return invalid("AccCore SpatialCore target does not resolve");
    attachedModules.push_back(system.artifact()
                                  .importedModules()[target->dependencyOrdinal]
                                  .identity());
  }

  std::vector<SpatialCatalogEntry> result;
  result.reserve(canonical.size());
  for (const ArtifactRootReference &reference : canonical) {
    auto spatial = ::loom::mapping::importSpatialMapping(reference, store);
    if (!spatial)
      return spatial.takeError();
    if (spatial->view().dataflowIdentity() != dataflow.identity())
      return invalid(
          "SpatialMapping catalog contains a foreign Dataflow owner");
    if (!llvm::is_contained(attachedModules, spatial->view().fabricIdentity()))
      return invalid(
          "SpatialMapping Fabric is not attached to a System AccCore");

    ArtifactRootReference techReference{
        ::loom::mapping::mappingArtifactSchema.identity.str(),
        ::loom::mapping::mappingArtifactSchema.version,
        spatial->view().techMappingIdentity()};
    auto tech = ::loom::mapping::importTechMapping(techReference, store);
    if (!tech)
      return tech.takeError();
    if (tech->view().dataflowIdentity() != dataflow.identity() ||
        tech->view().fabricIdentity() != spatial->view().fabricIdentity())
      return invalid(
          "SpatialMapping catalog has inconsistent TechMapping lineage");
    result.push_back(
        {reference, spatial->view().fabricIdentity(),
         std::vector<::dataflow::GraphRef>(tech->view().covers().begin(),
                                           tech->view().covers().end())});
  }
  return result;
}

std::vector<::loom::fabric::AccCoreOccurrenceRef>
canonicalAccCores(const ::loom::fabric::FabricSystemRootView &system) {
  std::vector<::loom::fabric::AccCoreOccurrenceRef> cores(
      system.artifact().accCoreOccurrences().begin(),
      system.artifact().accCoreOccurrences().end());
  llvm::sort(cores, [](auto lhs, auto rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  return cores;
}

llvm::Error validateConstraintInputs(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> canonicalRoots,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings) {
  if (constraints.view().dataflowIdentity() != dataflow.identity() ||
      constraints.view().fabricIdentity() != fabric.artifact().identity())
    return invalid("System MappingConstraintSet has foreign D/F owners");
  auto constraintRoots = detail::canonicalRootThreadLaunchSet(
      dataflow, constraints.view().rootThreadLaunches());
  if (!constraintRoots)
    return constraintRoots.takeError();
  if (llvm::ArrayRef(*constraintRoots) != canonicalRoots)
    return invalid("System MappingConstraintSet root closure differs from H");
  for (const ArtifactRootReference &required :
       constraints.view().spatialMappingReferences())
    if (!llvm::is_contained(spatialMappings, required))
      return invalid(
          "System MappingConstraintSet references a mapping outside H");
  return llvm::Error::success();
}

llvm::Expected<SystemPnrSearchDomainView> buildView(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const SystemBindingPartitionPlan &partitionPlan,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store) {
  auto roots = detail::canonicalRootThreadLaunchSet(
      dataflow, constraints.view().rootThreadLaunches());
  if (!roots)
    return roots.takeError();
  if (llvm::Error error = validateConstraintInputs(
          dataflow, fabric, constraints, *roots, spatialMappings))
    return std::move(error);
  auto partitions = detail::canonicalizeAndValidateSystemPartition(
      dataflow, *roots, partitionPlan);
  if (!partitions)
    return partitions.takeError();
  auto catalog = importSpatialCatalog(spatialMappings, dataflow, fabric, store);
  if (!catalog)
    return catalog.takeError();
  std::vector<::loom::fabric::AccCoreOccurrenceRef> cores =
      canonicalAccCores(fabric);

  std::vector<SystemSearchBindingDomain> bindings;
  bindings.reserve(partitions->size());
  for (detail::CanonicalSystemPartitionBinding &partition : *partitions) {
    SystemSearchAtomDomains domains;
    if (std::holds_alternative<::dataflow::RootThreadLaunchRef>(
            partition.key)) {
      domains.compatibleAccCores = cores;
    } else {
      const auto graphLaunch =
          std::get<::dataflow::RootedGraphLaunchRef>(partition.key);
      auto graph = dataflow.resolve(graphLaunch);
      if (!graph)
        return graph.takeError();
      std::vector<ArtifactRootReference> compatible;
      for (const SpatialCatalogEntry &entry : *catalog)
        if (llvm::is_contained(entry.covers, *graph))
          compatible.push_back(entry.reference);
      domains.compatibleSpatialMappings = std::move(compatible);
    }
    SystemSearchBindingDomain binding{std::move(partition.key), {}};
    binding.atoms.reserve(partition.cells.size());
    for (SystemPresburgerCell &cell : partition.cells)
      binding.atoms.push_back({std::move(cell), domains});
    bindings.push_back(std::move(binding));
  }

  ArtifactRootReference dataflowReference = dataflowRootReference(dataflow);
  ArtifactRootReference fabricReference = fabricRootReference(fabric);
  auto bytes = encodeView(dataflowReference, fabricReference,
                          constraints.reference(), *roots, bindings);
  if (!bytes)
    return bytes.takeError();
  auto digest = computeSystemPnrSearchDomainDigest(
      systemPnrSearchDomainSchemaDescriptorBytes(), *bytes);
  if (!digest)
    return digest.takeError();
  return detail::SystemPnrSearchDomainViewBuilder::create(
      std::move(dataflowReference), std::move(fabricReference),
      constraints.reference(), std::move(*roots), std::move(bindings),
      std::move(*bytes), std::move(*digest));
}

} // namespace

char UnsupportedSystemPnrSearchDomain::ID = 0;

void UnsupportedSystemPnrSearchDomain::log(llvm::raw_ostream &stream) const {
  stream << "system_pnr_search_domain_unsupported: " << message_;
}

std::error_code UnsupportedSystemPnrSearchDomain::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<SystemPnrSearchDomainDigest>
SystemPnrSearchDomainDigest::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return invalid("digest must contain exactly 32 bytes");
  Storage storage{};
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return SystemPnrSearchDomainDigest(storage);
}

llvm::ArrayRef<std::uint8_t> systemPnrSearchDomainSchemaDescriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(kSchemaDescriptor),
          sizeof(kSchemaDescriptor) - 1};
}

llvm::Expected<SystemPnrSearchDomainDigest> computeSystemPnrSearchDomainDigest(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes) {
  if (schemaDescriptorBytes.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("schema descriptor exceeds u32 framing");
  WireWriter writer;
  writer.bytes({reinterpret_cast<const std::uint8_t *>(kDigestDomain),
                sizeof(kDigestDomain) - 1});
  writer.u32(static_cast<std::uint32_t>(schemaDescriptorBytes.size()));
  writer.bytes(schemaDescriptorBytes);
  writer.u64(canonicalViewBytes.size());
  writer.bytes(canonicalViewBytes);
  const auto digest = llvm::SHA256::hash(writer.take());
  SystemPnrSearchDomainDigest::Storage storage{};
  std::copy(digest.begin(), digest.end(), storage.begin());
  return SystemPnrSearchDomainDigest(storage);
}

llvm::Error validateSystemPnrSearchDomainDigest(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const SystemPnrSearchDomainDigest &digest) {
  auto expected = computeSystemPnrSearchDomainDigest(schemaDescriptorBytes,
                                                     canonicalViewBytes);
  if (!expected)
    return expected.takeError();
  if (*expected != digest)
    return invalid("digest does not match canonical view bytes");
  return llvm::Error::success();
}

llvm::Expected<SystemPnrSearchDomainView> projectSystemPnrSearchDomain(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const SystemBindingPartitionPlan &partitionPlan,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store) {
  return buildView(dataflow, fabric, constraints, partitionPlan,
                   spatialMappings, store);
}

llvm::Expected<SystemPnrSearchDomainView>
adoptSystemPnrSearchDomain(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                           llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
                           const SystemPnrSearchDomainDigest &digest,
                           const ArtifactStore &store) {
  if (schemaDescriptorBytes != systemPnrSearchDomainSchemaDescriptorBytes())
    return invalid("schema descriptor is not exact version 1.0");
  if (llvm::Error error = validateSystemPnrSearchDomainDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);

  WireReader prefix(canonicalViewBytes);
  auto dataflowReference = prefix.rootReference();
  if (!dataflowReference)
    return dataflowReference.takeError();
  auto fabricReference = prefix.rootReference();
  if (!fabricReference)
    return fabricReference.takeError();
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(*dataflowReference, store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto fabricRoot =
      ::loom::fabric::importEntireFabricRoot(*fabricReference, store);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricRoot->view());
  if (!system)
    return system.takeError();

  auto decoded = decodeView(canonicalViewBytes, system->artifact());
  if (!decoded)
    return decoded.takeError();
  if (decoded->dataflowReference != *dataflowReference ||
      decoded->fabricReference != *fabricReference)
    return invalid("canonical prefix owners changed during decode");
  auto constraints = ::loom::mapping::importSystemMappingConstraintSet(
      decoded->constraintReference, store);
  if (!constraints)
    return constraints.takeError();

  SystemBindingPartitionPlan plan;
  std::vector<ArtifactRootReference> spatialMappings;
  for (const SystemSearchBindingDomain &binding : decoded->bindings) {
    SystemPresburgerBindingPartition partition{binding.key, {}};
    for (const SystemSearchAtom &atom : binding.atoms) {
      partition.cells.push_back(atom.cell);
      if (atom.domains.compatibleSpatialMappings)
        spatialMappings.insert(spatialMappings.end(),
                               atom.domains.compatibleSpatialMappings->begin(),
                               atom.domains.compatibleSpatialMappings->end());
    }
    plan.bindings.push_back(std::move(partition));
  }
  llvm::sort(spatialMappings, artifactRootReferenceLess);
  spatialMappings.erase(
      std::unique(spatialMappings.begin(), spatialMappings.end()),
      spatialMappings.end());
  auto projected =
      buildView(*dataflow, *system, *constraints, plan, spatialMappings, store);
  if (!projected)
    return projected.takeError();
  if (projected->rootThreadLaunches() !=
          llvm::ArrayRef<::dataflow::RootThreadLaunchRef>(decoded->roots) ||
      projected->canonicalViewBytes() != canonicalViewBytes ||
      projected->digest() != digest)
    return invalid("adopted view does not re-encode byte-exactly");
  return projected;
}

} // namespace loom::pnr
