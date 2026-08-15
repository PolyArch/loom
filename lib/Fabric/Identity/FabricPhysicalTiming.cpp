#include "Fabric/Identity/FabricPhysicalTiming.h"

#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

namespace {

constexpr llvm::StringLiteral descriptor =
    "loom.fabric.physical_timing_profile.1.0";
constexpr llvm::StringLiteral normalizedProviderIdentity =
    "loom.fabric.physical_timing.normalized_topology.1.0";
constexpr llvm::StringLiteral normalizedTechnologyIdentity = "target-neutral";
constexpr llvm::StringLiteral normalizedCharacterizationIdentity =
    "normalized-topology-1.0";
constexpr std::uint64_t normalizedClockBudgetQuanta = 8;

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "fabric_physical_timing_invalid: " + message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendString(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

bool invalidIdentity(llvm::StringRef value) {
  return value.empty() || value.contains('\0');
}

class ProfileReader final {
public:
  explicit ProfileReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    auto field = take(4, "u32");
    if (!field)
      return field.takeError();
    std::uint32_t value = 0;
    for (std::uint8_t byte : *field)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    auto field = take(8, "u64");
    if (!field)
      return field.takeError();
    std::uint64_t value = 0;
    for (std::uint8_t byte : *field)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>>
  take(std::uint64_t size, llvm::StringRef role) {
    if (size > bytes_.size())
      return invalid(llvm::Twine("truncated profile ") + role);
    llvm::ArrayRef<std::uint8_t> result = bytes_.take_front(size);
    bytes_ = bytes_.drop_front(size);
    return result;
  }

  llvm::Expected<std::string> string(llvm::StringRef role) {
    auto size = u64();
    if (!size)
      return size.takeError();
    auto value = take(*size, role);
    if (!value)
      return value.takeError();
    return std::string(reinterpret_cast<const char *>(value->data()),
                       value->size());
  }

  bool empty() const { return bytes_.empty(); }
  std::size_t remaining() const { return bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
};

llvm::Expected<FabricPhysicalTimingProfileView>
decodeProfile(llvm::ArrayRef<std::uint8_t> bytes,
              const FabricArtifactView &fabric) {
  ProfileReader reader(bytes);
  auto identityBytes = reader.take(ArtifactIdentity::byteSize,
                                   "Fabric identity");
  if (!identityBytes)
    return identityBytes.takeError();
  auto fabricIdentity = ArtifactIdentity::fromBytes(*identityBytes);
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  if (*fabricIdentity != fabric.identity())
    return invalid("profile is bound to another Fabric artifact");

  auto rawKind = reader.u32();
  if (!rawKind)
    return rawKind.takeError();
  FabricPhysicalTimingProfileKind kind;
  switch (*rawKind) {
  case static_cast<std::uint32_t>(
      FabricPhysicalTimingProfileKind::NormalizedHeuristic):
    kind = FabricPhysicalTimingProfileKind::NormalizedHeuristic;
    break;
  case static_cast<std::uint32_t>(
      FabricPhysicalTimingProfileKind::TargetCharacterization):
    kind = FabricPhysicalTimingProfileKind::TargetCharacterization;
    break;
  default:
    return invalid("profile kind is outside the closed domain");
  }

  auto provider = reader.string("provider identity");
  if (!provider)
    return provider.takeError();
  auto technology = reader.string("technology identity");
  if (!technology)
    return technology.takeError();
  auto characterization = reader.string("characterization identity");
  if (!characterization)
    return characterization.takeError();
  auto requiredDelay = reader.u64();
  if (!requiredDelay)
    return requiredDelay.takeError();
  auto traversalCount = reader.u64();
  if (!traversalCount)
    return traversalCount.takeError();
  constexpr std::uint64_t minimumTraversalBytes = 8 + 1 + 8 + 4;
  if (*traversalCount >
      static_cast<std::uint64_t>(reader.remaining()) /
          minimumTraversalBytes)
    return invalid("profile traversal count exceeds its payload");
  if (*traversalCount > std::numeric_limits<std::size_t>::max())
    return invalid("profile traversal count exceeds host size");

  std::vector<FabricTraversalPhysicalTiming> traversals;
  traversals.reserve(static_cast<std::size_t>(*traversalCount));
  for (std::uint64_t ordinal = 0; ordinal != *traversalCount; ++ordinal) {
    auto referenceSize = reader.u64();
    if (!referenceSize)
      return referenceSize.takeError();
    auto referenceBytes = reader.take(*referenceSize, "traversal reference");
    if (!referenceBytes)
      return referenceBytes.takeError();
    auto reference =
        decodeFabricRef<FabricPhysicalTraversalRef>(*referenceBytes);
    if (!reference)
      return llvm::joinErrors(invalid("profile traversal is malformed"),
                              reference.takeError());
    auto delay = reader.u64();
    if (!delay)
      return delay.takeError();
    auto rawBoundary = reader.u32();
    if (!rawBoundary)
      return rawBoundary.takeError();
    FabricPhysicalTimingBoundaryKind boundary;
    switch (*rawBoundary) {
    case static_cast<std::uint32_t>(
        FabricPhysicalTimingBoundaryKind::Combinational):
      boundary = FabricPhysicalTimingBoundaryKind::Combinational;
      break;
    case static_cast<std::uint32_t>(
        FabricPhysicalTimingBoundaryKind::RegisteredDestination):
      boundary = FabricPhysicalTimingBoundaryKind::RegisteredDestination;
      break;
    default:
      return invalid("profile boundary kind is outside the closed domain");
    }
    traversals.push_back({*reference, *delay, boundary});
  }
  if (!reader.empty())
    return invalid("profile has trailing canonical bytes");

  auto profile = createFabricPhysicalTimingProfile(
      fabric, kind, *provider, *technology, *characterization, *requiredDelay,
      traversals);
  if (!profile)
    return profile.takeError();
  if (profile->canonicalViewBytes() != bytes)
    return invalid("profile bytes are not in canonical form");
  return profile;
}

FabricTraversalPhysicalTiming
normalizedTiming(const FabricPhysicalTraversalRef &traversal) {
  FabricTraversalPhysicalTiming timing;
  timing.traversal = traversal;
  switch (traversal.kind()) {
  case FabricPhysicalTraversalKind::PointConnection:
    timing.delayQuanta = 1;
    break;
  case FabricPhysicalTraversalKind::PeSelectorTraversal:
    timing.delayQuanta = 2;
    break;
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal: {
    timing.delayQuanta = 1;
    const auto &payload =
        std::get<FabricPeRegisterFifoPayload>(traversal.payload);
    if (payload.role == FabricRegisterFifoPathRole::Write)
      timing.boundary = FabricPhysicalTimingBoundaryKind::RegisteredDestination;
    break;
  }
  case FabricPhysicalTraversalKind::SwitchTraversal:
    timing.delayQuanta = 3;
    break;
  case FabricPhysicalTraversalKind::FifoTraversal: {
    const auto &payload =
        std::get<FabricFifoTraversalPayload>(traversal.payload);
    if (payload.mode == FabricFifoTraversalMode::Buffered) {
      timing.delayQuanta = 1;
      timing.boundary = FabricPhysicalTimingBoundaryKind::RegisteredDestination;
    } else {
      timing.delayQuanta = 2;
    }
    break;
  }
  case FabricPhysicalTraversalKind::BoundaryTraversal:
    timing.delayQuanta = 1;
    break;
  case FabricPhysicalTraversalKind::SystemTransferPatternLeg:
    timing.delayQuanta = 1;
    break;
  }
  return timing;
}

std::vector<std::uint8_t> encodeProfile(
    const ArtifactIdentity &fabricIdentity,
    FabricPhysicalTimingProfileKind kind, llvm::StringRef providerIdentity,
    llvm::StringRef technologyIdentity,
    llvm::StringRef characterizationIdentity, std::uint64_t requiredDelay,
    llvm::ArrayRef<FabricTraversalPhysicalTiming> traversals) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(ArtifactIdentity::byteSize + providerIdentity.size() +
                technologyIdentity.size() + characterizationIdentity.size() +
                44 + traversals.size() * 32);
  bytes.insert(bytes.end(), fabricIdentity.bytes().begin(),
               fabricIdentity.bytes().end());
  appendU32(bytes, static_cast<std::uint32_t>(kind));
  appendString(bytes, providerIdentity);
  appendString(bytes, technologyIdentity);
  appendString(bytes, characterizationIdentity);
  appendU64(bytes, requiredDelay);
  appendU64(bytes, traversals.size());
  for (const FabricTraversalPhysicalTiming &timing : traversals) {
    const std::vector<std::uint8_t> reference =
        canonicalFabricBytes(timing.traversal);
    appendU64(bytes, reference.size());
    bytes.insert(bytes.end(), reference.begin(), reference.end());
    appendU64(bytes, timing.delayQuanta);
    appendU32(bytes, static_cast<std::uint32_t>(timing.boundary));
  }
  return bytes;
}

} // namespace

llvm::ArrayRef<std::uint8_t>
loom::fabric::fabricPhysicalTimingProfileSchemaDescriptorBytes() {
  return descriptorBytes();
}

llvm::ArrayRef<std::uint8_t>
FabricPhysicalTimingProfileView::schemaDescriptorBytes() const {
  return descriptorBytes();
}

llvm::Expected<FabricPhysicalTimingProfileView>
loom::fabric::createFabricPhysicalTimingProfile(
    const FabricArtifactView &fabric, FabricPhysicalTimingProfileKind kind,
    llvm::StringRef providerIdentity, llvm::StringRef technologyIdentity,
    llvm::StringRef characterizationIdentity,
    std::uint64_t requiredCombinationalDelayQuanta,
    llvm::ArrayRef<FabricTraversalPhysicalTiming> traversals) {
  switch (kind) {
  case FabricPhysicalTimingProfileKind::NormalizedHeuristic:
  case FabricPhysicalTimingProfileKind::TargetCharacterization:
    break;
  }
  if (invalidIdentity(providerIdentity))
    return invalid("provider identity is empty or contains NUL");
  if (invalidIdentity(technologyIdentity))
    return invalid("technology identity is empty or contains NUL");
  if (invalidIdentity(characterizationIdentity))
    return invalid("characterization identity is empty or contains NUL");

  std::vector<FabricTraversalPhysicalTiming> canonicalTraversals(
      traversals.begin(), traversals.end());
  llvm::sort(canonicalTraversals, [](const FabricTraversalPhysicalTiming &lhs,
                                     const FabricTraversalPhysicalTiming &rhs) {
    return canonicalFabricBytes(lhs.traversal) <
           canonicalFabricBytes(rhs.traversal);
  });
  std::vector<std::uint8_t> canonical =
      encodeProfile(fabric.identity(), kind, providerIdentity,
                    technologyIdentity, characterizationIdentity,
                    requiredCombinationalDelayQuanta, canonicalTraversals);
  auto digest = computeComponentViewDigest(descriptorBytes(), canonical);
  if (!digest)
    return digest.takeError();
  FabricPhysicalTimingProfileView profile(
      fabric.identity(), kind, providerIdentity.str(), technologyIdentity.str(),
      characterizationIdentity.str(), requiredCombinationalDelayQuanta,
      std::move(canonicalTraversals), std::move(canonical), *digest);
  if (llvm::Error error = validateFabricPhysicalTimingProfile(fabric, profile))
    return std::move(error);
  return profile;
}

llvm::Expected<FabricPhysicalTimingProfileView>
loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
    const FabricArtifactView &fabric) {
  std::vector<FabricTraversalPhysicalTiming> traversals;
  traversals.reserve(fabric.physicalTraversals().size());
  for (const FabricPhysicalTraversalView &traversal :
       fabric.physicalTraversals())
    traversals.push_back(normalizedTiming(traversal.reference));
  return createFabricPhysicalTimingProfile(
      fabric, FabricPhysicalTimingProfileKind::NormalizedHeuristic,
      normalizedProviderIdentity, normalizedTechnologyIdentity,
      normalizedCharacterizationIdentity, normalizedClockBudgetQuanta,
      traversals);
}

llvm::Expected<std::vector<FabricPhysicalTimingProfileView>>
loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(
    const FabricSystemRootView &system) {
  std::map<ArtifactIdentity::Storage, const FabricArtifactView *> modules;
  for (const AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    const auto target = system.spatialCoreTarget(core);
    if (!target || target->dependencyOrdinal >=
                       system.artifact().importedModules().size())
      return invalid("System AccCore SpatialCore target does not resolve");
    const FabricArtifactView &module =
        system.artifact().importedModules()[target->dependencyOrdinal];
    modules.emplace(module.identity().bytes(), &module);
  }
  std::vector<FabricPhysicalTimingProfileView> profiles;
  profiles.reserve(modules.size());
  for (const auto &[identity, module] : modules) {
    (void)identity;
    auto profile = projectNormalizedFabricPhysicalTimingProfile(*module);
    if (!profile)
      return profile.takeError();
    profiles.push_back(std::move(*profile));
  }
  return profiles;
}

llvm::Expected<ArtifactRootReference>
loom::fabric::publishFabricPhysicalTimingProfile(
    const FabricPhysicalTimingProfileView &profile,
    const ArtifactStore &store) {
  auto identity = store.put(
      fabricPhysicalTimingProfileArtifactSchema,
      CanonicalSemanticBytes(profile.canonicalViewBytes().vec()));
  if (!identity)
    return identity.takeError();
  return ArtifactRootReference{
      fabricPhysicalTimingProfileArtifactSchema.identity.str(),
      fabricPhysicalTimingProfileArtifactSchema.version, *identity};
}

llvm::Expected<FabricPhysicalTimingProfileView>
loom::fabric::importFabricPhysicalTimingProfile(
    const ArtifactRootReference &reference, const FabricArtifactView &fabric,
    const ArtifactStore &store) {
  if (reference.schemaIdentity !=
          fabricPhysicalTimingProfileArtifactSchema.identity ||
      reference.schemaVersion !=
          fabricPhysicalTimingProfileArtifactSchema.version)
    return invalid("root reference has the wrong profile schema");
  auto bytes = store.get(fabricPhysicalTimingProfileArtifactSchema,
                         reference.artifact);
  if (!bytes)
    return bytes.takeError();
  return decodeProfile(bytes->bytes(), fabric);
}

llvm::Expected<ArtifactIdentity>
loom::fabric::resolveFabricPhysicalTimingProfileOwner(
    const ArtifactRootReference &reference, const ArtifactStore &store) {
  if (reference.schemaIdentity !=
          fabricPhysicalTimingProfileArtifactSchema.identity ||
      reference.schemaVersion !=
          fabricPhysicalTimingProfileArtifactSchema.version)
    return invalid("root reference has the wrong profile schema");
  auto bytes = store.get(fabricPhysicalTimingProfileArtifactSchema,
                         reference.artifact);
  if (!bytes)
    return bytes.takeError();
  if (bytes->bytes().size() < ArtifactIdentity::byteSize)
    return invalid("truncated profile Fabric identity");
  return ArtifactIdentity::fromBytes(
      bytes->bytes().take_front(ArtifactIdentity::byteSize));
}

llvm::Error loom::fabric::validateFabricPhysicalTimingProfile(
    const FabricArtifactView &fabric,
    const FabricPhysicalTimingProfileView &profile) {
  if (profile.fabricIdentity() != fabric.identity())
    return invalid("profile is bound to another Fabric artifact");
  switch (profile.kind()) {
  case FabricPhysicalTimingProfileKind::NormalizedHeuristic:
  case FabricPhysicalTimingProfileKind::TargetCharacterization:
    break;
  }
  if (invalidIdentity(profile.providerIdentity()))
    return invalid("profile provider identity is invalid");
  if (invalidIdentity(profile.technologyIdentity()))
    return invalid("profile technology identity is invalid");
  if (invalidIdentity(profile.characterizationIdentity()))
    return invalid("profile characterization identity is invalid");
  if (profile.requiredCombinationalDelayQuanta() == 0)
    return invalid("required combinational delay is zero");
  if (llvm::Error error = validateComponentViewDigest(
          profile.schemaDescriptorBytes(), profile.canonicalViewBytes(),
          profile.digest()))
    return llvm::joinErrors(invalid("profile digest is invalid"),
                            std::move(error));
  if (profile.traversals().size() != fabric.physicalTraversals().size())
    return invalid("profile traversal inventory is incomplete");

  std::set<std::vector<std::uint8_t>> expected;
  for (const FabricPhysicalTraversalView &traversal :
       fabric.physicalTraversals())
    expected.insert(canonicalFabricBytes(traversal.reference));
  std::vector<std::uint8_t> previous;
  bool first = true;
  for (const FabricTraversalPhysicalTiming &timing : profile.traversals()) {
    const std::vector<std::uint8_t> key =
        canonicalFabricBytes(timing.traversal);
    if (!first && previous >= key)
      return invalid("profile traversal inventory is not canonical");
    first = false;
    previous = key;
    if (!expected.erase(key))
      return invalid("profile contains a foreign or repeated traversal");
    if (timing.delayQuanta == 0)
      return invalid("profile traversal delay is zero");
    switch (timing.boundary) {
    case FabricPhysicalTimingBoundaryKind::Combinational:
    case FabricPhysicalTimingBoundaryKind::RegisteredDestination:
      break;
    }
  }
  if (!expected.empty())
    return invalid("profile omits a Fabric traversal");
  const std::vector<std::uint8_t> expectedCanonical = encodeProfile(
      profile.fabricIdentity(), profile.kind(), profile.providerIdentity(),
      profile.technologyIdentity(), profile.characterizationIdentity(),
      profile.requiredCombinationalDelayQuanta(), profile.traversals());
  if (llvm::ArrayRef(expectedCanonical) != profile.canonicalViewBytes())
    return invalid("profile canonical bytes disagree with its typed view");
  return llvm::Error::success();
}
