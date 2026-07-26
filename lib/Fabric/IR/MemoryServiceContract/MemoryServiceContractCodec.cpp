#include "Fabric/IR/MemoryServiceContract.h"

#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "MemoryServiceContractInternal.h"

#include <limits>
#include <system_error>

namespace fabric {
namespace {

enum class RegionWireTag : std::uint32_t { Storage = 0, Mmio = 1 };
enum class AccessWireTag : std::uint32_t { NoAccess = 0, Access = 1 };
enum class ConsistencyWireTag : std::uint32_t {
  None = 0,
  LocalProvider = 1,
  SystemDomain = 2,
};
enum class ReleaseWireTag : std::uint32_t {
  AtLinearization = 0,
  ByRetirement = 1,
};
enum class LocalProgressWireTag : std::uint32_t {
  BoundedCompletionCycles = 0,
  FairEventual = 1,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument,
                                 "invalid MemoryServiceContractRecord: %s",
                                 message.str().c_str());
}

class Writer {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  llvm::Error count(std::size_t value, llvm::StringRef field) {
    if (value > std::numeric_limits<std::uint64_t>::max())
      return invalid(field + " count exceeds u64");
    u64(value);
    return llvm::Error::success();
  }
  llvm::Error frame(llvm::ArrayRef<std::uint8_t> bytes, llvm::StringRef field) {
    if (llvm::Error error = count(bytes.size(), field))
      return error;
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    return llvm::Error::success();
  }
  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef field) {
    if (remaining() < 4)
      return invalid("truncated " + field);
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }
  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    if (remaining() < 8)
      return invalid("truncated " + field);
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }
  llvm::Expected<llvm::ArrayRef<std::uint8_t>> frame(llvm::StringRef field) {
    auto size = u64((field + " length").str());
    if (!size)
      return size.takeError();
    if (*size > remaining())
      return invalid("truncated " + field);
    llvm::ArrayRef<std::uint8_t> value = bytes_.slice(offset_, *size);
    offset_ += *size;
    return value;
  }
  llvm::Error finish(llvm::StringRef record) const {
    if (remaining() != 0)
      return invalid(record + " has trailing bytes");
    return llvm::Error::success();
  }
  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<std::uint32_t> releaseTag(ReleaseVisibilityPoint point) {
  switch (point) {
  case ReleaseVisibilityPoint::AtLinearization:
    return static_cast<std::uint32_t>(ReleaseWireTag::AtLinearization);
  case ReleaseVisibilityPoint::ByRetirement:
    return static_cast<std::uint32_t>(ReleaseWireTag::ByRetirement);
  }
  return invalid("unknown release visibility point");
}

llvm::Expected<ReleaseVisibilityPoint> decodeRelease(std::uint32_t tag) {
  switch (tag) {
  case static_cast<std::uint32_t>(ReleaseWireTag::AtLinearization):
    return ReleaseVisibilityPoint::AtLinearization;
  case static_cast<std::uint32_t>(ReleaseWireTag::ByRetirement):
    return ReleaseVisibilityPoint::ByRetirement;
  default:
    return invalid("unknown release visibility tag");
  }
}

llvm::Error appendRegion(Writer &writer,
                         const MemoryServiceRegionDeclaration &region) {
  writer.u64(region.addressBaseBytes);
  writer.u64(region.sizeBytes);
  switch (region.behavior) {
  case MemoryServiceRegionBehavior::Storage:
    writer.u32(static_cast<std::uint32_t>(RegionWireTag::Storage));
    if (region.mmioAcceptedAccessDomain)
      return invalid("Storage region carries an MMIO access domain");
    return llvm::Error::success();
  case MemoryServiceRegionBehavior::Mmio: {
    writer.u32(static_cast<std::uint32_t>(RegionWireTag::Mmio));
    if (!region.mmioAcceptedAccessDomain)
      return invalid("MMIO region has no accepted access domain");
    auto access =
        encodeParameterizedMemoryAccessDomain(*region.mmioAcceptedAccessDomain);
    if (!access)
      return access.takeError();
    return writer.frame(*access, "MMIO access domain");
  }
  }
  return invalid("unknown memory service region behavior");
}

llvm::Expected<MemoryServiceRegionDeclaration> decodeRegion(Reader &reader) {
  auto base = reader.u64("region address base");
  auto size = reader.u64("region size");
  auto tag = reader.u32("region behavior");
  if (!base)
    return base.takeError();
  if (!size)
    return size.takeError();
  if (!tag)
    return tag.takeError();
  switch (*tag) {
  case static_cast<std::uint32_t>(RegionWireTag::Storage):
    return MemoryServiceRegionDeclaration{
        *base, *size, MemoryServiceRegionBehavior::Storage, std::nullopt};
  case static_cast<std::uint32_t>(RegionWireTag::Mmio): {
    auto accessBytes = reader.frame("MMIO access domain");
    if (!accessBytes)
      return accessBytes.takeError();
    auto access = decodeParameterizedMemoryAccessDomain(*accessBytes);
    if (!access)
      return access.takeError();
    return MemoryServiceRegionDeclaration{
        *base, *size, MemoryServiceRegionBehavior::Mmio, std::move(*access)};
  }
  default:
    return invalid("unknown memory service region behavior tag");
  }
}

llvm::Error appendRegionsAndBeat(
    Writer &writer, const detail::MemoryServiceCapabilityPhysicalFacts &facts) {
  if (llvm::Error error =
          writer.count(facts.serviceRegionOrdinals.size(), "service regions"))
    return error;
  for (std::uint64_t ordinal : facts.serviceRegionOrdinals)
    writer.u64(ordinal);
  writer.u64(facts.serviceBeatWidthBits);
  return llvm::Error::success();
}

llvm::Expected<std::pair<std::vector<std::uint64_t>, std::uint64_t>>
decodeRegionsAndBeat(Reader &reader) {
  auto count = reader.u64("service region count");
  if (!count)
    return count.takeError();
  if (*count > reader.remaining() / 8)
    return invalid("service region count exceeds its framing");
  std::vector<std::uint64_t> regions;
  regions.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto ordinal = reader.u64("service region ordinal");
    if (!ordinal)
      return ordinal.takeError();
    regions.push_back(*ordinal);
  }
  auto beat = reader.u64("service beat width");
  if (!beat)
    return beat.takeError();
  return std::make_pair(std::move(regions), *beat);
}

llvm::Error appendConsistency(Writer &writer,
                              const MemoryServiceConsistencyBinding &binding) {
  if (std::holds_alternative<NoMemoryServiceConsistency>(binding)) {
    writer.u32(static_cast<std::uint32_t>(ConsistencyWireTag::None));
    return llvm::Error::success();
  }
  if (const auto *local = std::get_if<LocalProviderConsistency>(&binding)) {
    writer.u32(static_cast<std::uint32_t>(ConsistencyWireTag::LocalProvider));
    auto release = releaseTag(local->releaseVisibilityPoint);
    if (!release)
      return release.takeError();
    writer.u32(*release);
    if (const auto *bounded =
            std::get_if<LocalBoundedCompletionCycles>(&local->progress)) {
      writer.u32(static_cast<std::uint32_t>(
          LocalProgressWireTag::BoundedCompletionCycles));
      writer.u64(bounded->maxIssueToRetireCycles);
    } else {
      writer.u32(
          static_cast<std::uint32_t>(LocalProgressWireTag::FairEventual));
    }
    return llvm::Error::success();
  }
  writer.u32(static_cast<std::uint32_t>(ConsistencyWireTag::SystemDomain));
  std::vector<std::uint8_t> reference = loom::fabric::canonicalFabricBytes(
      std::get<loom::fabric::MemoryConsistencyDomainRef>(binding));
  return writer.frame(reference, "memory consistency domain");
}

llvm::Expected<MemoryServiceConsistencyBinding>
decodeConsistency(Reader &reader) {
  auto tag = reader.u32("consistency binding");
  if (!tag)
    return tag.takeError();
  switch (*tag) {
  case static_cast<std::uint32_t>(ConsistencyWireTag::None):
    return MemoryServiceConsistencyBinding(
        std::in_place_type<NoMemoryServiceConsistency>);
  case static_cast<std::uint32_t>(ConsistencyWireTag::LocalProvider): {
    auto releaseTag = reader.u32("release visibility");
    auto progressTag = reader.u32("local progress");
    if (!releaseTag)
      return releaseTag.takeError();
    if (!progressTag)
      return progressTag.takeError();
    auto release = decodeRelease(*releaseTag);
    if (!release)
      return release.takeError();
    LocalProviderProgress progress;
    switch (*progressTag) {
    case static_cast<std::uint32_t>(
        LocalProgressWireTag::BoundedCompletionCycles): {
      auto cycles = reader.u64("max issue-to-retire cycles");
      if (!cycles)
        return cycles.takeError();
      progress.emplace<LocalBoundedCompletionCycles>(
          LocalBoundedCompletionCycles{*cycles});
      break;
    }
    case static_cast<std::uint32_t>(LocalProgressWireTag::FairEventual):
      progress.emplace<FairEventual>();
      break;
    default:
      return invalid("unknown local progress tag");
    }
    return MemoryServiceConsistencyBinding(
        std::in_place_type<LocalProviderConsistency>,
        LocalProviderConsistency{*release, std::move(progress)});
  }
  case static_cast<std::uint32_t>(ConsistencyWireTag::SystemDomain): {
    auto referenceBytes = reader.frame("memory consistency domain");
    if (!referenceBytes)
      return referenceBytes.takeError();
    auto reference =
        loom::fabric::decodeFabricRef<loom::fabric::MemoryConsistencyDomainRef>(
            *referenceBytes);
    if (!reference)
      return reference.takeError();
    return MemoryServiceConsistencyBinding(
        std::in_place_type<loom::fabric::MemoryConsistencyDomainRef>,
        std::move(*reference));
  }
  default:
    return invalid("unknown memory service consistency tag");
  }
}

llvm::Error appendPatterns(Writer &writer,
                           llvm::ArrayRef<UsePatternKey> patterns) {
  if (llvm::Error error = writer.count(patterns.size(), "use patterns"))
    return error;
  for (UsePatternKey pattern : patterns)
    writer.u32(pattern.ordinal());
  return llvm::Error::success();
}

llvm::Expected<std::vector<UsePatternKey>> decodePatterns(Reader &reader) {
  auto count = reader.u64("use pattern count");
  if (!count)
    return count.takeError();
  if (*count > reader.remaining() / 4)
    return invalid("use pattern count exceeds its framing");
  std::vector<UsePatternKey> patterns;
  patterns.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto ordinal = reader.u32("use pattern ordinal");
    if (!ordinal)
      return ordinal.takeError();
    patterns.emplace_back(*ordinal);
  }
  return patterns;
}

llvm::Error
appendCapability(Writer &writer,
                 const MemoryServiceCapabilityDeclaration &capability) {
  auto actor = encodeMemoryActorContractDomain(capability.actorContractDomain);
  if (!actor)
    return actor.takeError();
  if (llvm::Error error = writer.frame(*actor, "actor contract domain"))
    return error;
  if (!capability.accessDomain) {
    writer.u32(static_cast<std::uint32_t>(AccessWireTag::NoAccess));
  } else {
    writer.u32(static_cast<std::uint32_t>(AccessWireTag::Access));
    auto access =
        encodeParameterizedMemoryAccessDomain(*capability.accessDomain);
    if (!access)
      return access.takeError();
    if (llvm::Error error = writer.frame(*access, "memory access domain"))
      return error;
  }
  detail::MemoryServiceCapabilityPhysicalFacts facts{
      capability.serviceRegionOrdinals, capability.serviceBeatWidthBits,
      capability.consistencyBinding};
  if (llvm::Error error = appendRegionsAndBeat(writer, facts))
    return error;
  if (llvm::Error error =
          appendPatterns(writer, capability.admissibleUsePatterns))
    return error;
  return appendConsistency(writer, capability.consistencyBinding);
}

llvm::Expected<MemoryServiceCapabilityDeclaration>
decodeCapability(Reader &reader, mlir::MLIRContext *context) {
  auto actorBytes = reader.frame("actor contract domain");
  if (!actorBytes)
    return actorBytes.takeError();
  auto actors = decodeMemoryActorContractDomain(*actorBytes, context);
  if (!actors)
    return actors.takeError();
  auto accessTag = reader.u32("memory access domain tag");
  if (!accessTag)
    return accessTag.takeError();
  std::optional<ParameterizedMemoryAccessDomain> access;
  switch (*accessTag) {
  case static_cast<std::uint32_t>(AccessWireTag::NoAccess):
    break;
  case static_cast<std::uint32_t>(AccessWireTag::Access): {
    auto accessBytes = reader.frame("memory access domain");
    if (!accessBytes)
      return accessBytes.takeError();
    auto decoded = decodeParameterizedMemoryAccessDomain(*accessBytes);
    if (!decoded)
      return decoded.takeError();
    access = std::move(*decoded);
    break;
  }
  default:
    return invalid("unknown memory access domain tag");
  }
  auto regionsAndBeat = decodeRegionsAndBeat(reader);
  if (!regionsAndBeat)
    return regionsAndBeat.takeError();
  auto patterns = decodePatterns(reader);
  if (!patterns)
    return patterns.takeError();
  auto consistency = decodeConsistency(reader);
  if (!consistency)
    return consistency.takeError();
  return MemoryServiceCapabilityDeclaration{std::move(*actors),
                                            std::move(access),
                                            std::move(regionsAndBeat->first),
                                            regionsAndBeat->second,
                                            std::move(*patterns),
                                            std::move(*consistency)};
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
detail::encodeMemoryServiceCapabilityPhysicalFacts(
    const MemoryServiceCapabilityPhysicalFacts &facts) {
  Writer writer;
  if (llvm::Error error = appendRegionsAndBeat(writer, facts))
    return std::move(error);
  if (llvm::Error error = appendConsistency(writer, facts.consistencyBinding))
    return std::move(error);
  return writer.take();
}

llvm::Expected<detail::MemoryServiceCapabilityPhysicalFacts>
detail::decodeMemoryServiceCapabilityPhysicalFacts(
    llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto regionsAndBeat = decodeRegionsAndBeat(reader);
  if (!regionsAndBeat)
    return regionsAndBeat.takeError();
  auto consistency = decodeConsistency(reader);
  if (!consistency)
    return consistency.takeError();
  if (llvm::Error error = reader.finish("capability physical facts"))
    return std::move(error);
  return MemoryServiceCapabilityPhysicalFacts{std::move(regionsAndBeat->first),
                                              regionsAndBeat->second,
                                              std::move(*consistency)};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryServiceContractRecord(const MemoryServiceContractRecord &record) {
  Writer writer;
  if (llvm::Error error = writer.count(record.regions().size(), "regions"))
    return std::move(error);
  for (const MemoryServiceRegionDeclaration &region : record.regions()) {
    Writer nested;
    if (llvm::Error error = appendRegion(nested, region))
      return std::move(error);
    if (llvm::Error error = writer.frame(nested.take(), "region"))
      return std::move(error);
  }
  auto resource = encodeResourceContractRecord(record.resourceContract());
  if (!resource)
    return resource.takeError();
  if (llvm::Error error = writer.frame(*resource, "resource contract"))
    return std::move(error);
  if (llvm::Error error =
          writer.count(record.capabilities().size(), "capabilities"))
    return std::move(error);
  for (const MemoryServiceCapabilityDeclaration &capability :
       record.capabilities()) {
    Writer nested;
    if (llvm::Error error = appendCapability(nested, capability))
      return std::move(error);
    if (llvm::Error error = writer.frame(nested.take(), "capability"))
      return std::move(error);
  }
  return writer.take();
}

llvm::Expected<MemoryServiceContractRecord>
decodeMemoryServiceContractRecord(llvm::ArrayRef<std::uint8_t> bytes,
                                  mlir::MLIRContext *context,
                                  MemoryServiceOwnerKind owner) {
  Reader reader(bytes);
  auto regionCount = reader.u64("region count");
  if (!regionCount)
    return regionCount.takeError();
  if (*regionCount > reader.remaining() / 8)
    return invalid("region count exceeds its framing");
  std::vector<MemoryServiceRegionDeclaration> regions;
  regions.reserve(*regionCount);
  for (std::uint64_t index = 0; index < *regionCount; ++index) {
    auto regionBytes = reader.frame("region");
    if (!regionBytes)
      return regionBytes.takeError();
    Reader nested(*regionBytes);
    auto region = decodeRegion(nested);
    if (!region)
      return region.takeError();
    if (llvm::Error error = nested.finish("region"))
      return std::move(error);
    regions.push_back(std::move(*region));
  }
  auto resourceBytes = reader.frame("resource contract");
  if (!resourceBytes)
    return resourceBytes.takeError();
  auto resource = decodeResourceContractRecord(*resourceBytes);
  if (!resource)
    return resource.takeError();
  auto capabilityCount = reader.u64("capability count");
  if (!capabilityCount)
    return capabilityCount.takeError();
  if (*capabilityCount > reader.remaining() / 8)
    return invalid("capability count exceeds its framing");
  std::vector<MemoryServiceCapabilityDeclaration> capabilities;
  capabilities.reserve(*capabilityCount);
  for (std::uint64_t index = 0; index < *capabilityCount; ++index) {
    auto capabilityBytes = reader.frame("capability");
    if (!capabilityBytes)
      return capabilityBytes.takeError();
    Reader nested(*capabilityBytes);
    auto capability = decodeCapability(nested, context);
    if (!capability)
      return capability.takeError();
    if (llvm::Error error = nested.finish("capability"))
      return std::move(error);
    capabilities.push_back(std::move(*capability));
  }
  if (llvm::Error error = reader.finish("memory service contract"))
    return std::move(error);
  return MemoryServiceContractRecord::fromCanonical(
      context, owner,
      MemoryServiceContractDeclaration{std::move(regions), std::move(*resource),
                                       std::move(capabilities)});
}

} // namespace fabric
