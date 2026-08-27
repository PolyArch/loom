#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricArtifactLocalReference.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

static_assert(fabricArtifactLocalReferenceKindOrdinal(
                  FabricArtifactLocalReferenceKind::FabricModuleTemplateRef) ==
              0);
static_assert(
    fabricArtifactLocalReferenceKindOrdinal(
        FabricArtifactLocalReferenceKind::FabricFuCapabilityTemplateRef) == 24);
static_assert(fabricArtifactLocalReferenceKindOrdinal(
                  FabricArtifactLocalReferenceKind::ResetDomainRef) == 44);
static_assert(fabricArtifactLocalReferenceKindOrdinal(
                  FabricArtifactLocalReferenceKind::
                      FabricMemoryEngineTemplateInternalConnectionRef) == 49);
static_assert(fabricArtifactSchema.version.major == 7);
static_assert(fabricArtifactSchema.version.minor == 0);

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << message << "\n";
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

template <typename T> T takeExpected(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

ArtifactIdentity identity(std::uint8_t seed) {
  return takeExpected(ArtifactIdentity::fromBytes(
      std::vector<std::uint8_t>(ArtifactIdentity::byteSize, seed)));
}

template <typename Ref>
void requireTypedRoundTrip(const ArtifactIdentity &artifact,
                           const Ref &entity) {
  const ArtifactReference<Ref> typed{artifact, entity};
  const EncodedArtifactLocalReference encoded =
      encodeFabricArtifactLocalReference(typed);
  const std::uint32_t expected = fabricArtifactLocalReferenceKindOrdinal(
      FabricArtifactLocalReferenceKindTraits<Ref>::kind);
  require(encoded.ownerLocalKind == expected, "encoded wrong owner-local kind");

  const ArtifactReference<Ref> decoded =
      takeExpected(decodeFabricArtifactLocalReference<Ref>(encoded));
  require(decoded == typed, "typed owner-local reference did not round-trip");
}

void testGeneratedCatalog() {
  const auto catalog = fabricArtifactLocalReferenceKindCatalog();
  require(catalog.size() == 50,
          "current loom.fabric must register 50 local kinds");

  std::set<std::string> targets;
  for (std::size_t index = 0; index < catalog.size(); ++index) {
    const auto &entry = catalog[index];
    require(fabricArtifactLocalReferenceKindOrdinal(entry.kind) == index,
            "generated catalog order does not match stable kind ordinals");
    require(targets.insert(entry.typedTarget.str()).second,
            "generated catalog contains a duplicate typed target");
  }
}

void testRepresentativeRoundTrips() {
  const ArtifactIdentity artifact = identity(0x31);
  requireTypedRoundTrip(artifact, FabricFuTemplateRef(7));

  const FabricResourceStateOwnerRef owner(
      FabricInventoryOwnerRef::of(FabricSwitchOccurrenceRef(11)));
  requireTypedRoundTrip(artifact, FabricResourceStateRef{owner, 3});

  const ClockDomainRef clock(HardwareDomainRef(17));
  requireTypedRoundTrip(artifact, clock);

  const FabricMemoryEngineTemplateRef engine(19);
  const FabricMemoryEngineTemplateEndpointRef source{engine, 1};
  const FabricMemoryEngineTemplateEndpointRef sink{engine, 2};
  requireTypedRoundTrip(
      artifact,
      FabricMemoryEngineTemplateInternalConnectionRef{engine, source, sink});

  const auto base = encodeFabricArtifactLocalReference(
      ArtifactReference<HardwareDomainRef>{artifact, clock.underlying()});
  const auto refined = encodeFabricArtifactLocalReference(
      ArtifactReference<ClockDomainRef>{artifact, clock});
  require(base.payload == refined.payload,
          "a role refinement changed canonical target bytes");
  require(base.ownerLocalKind != refined.ownerLocalKind,
          "a role refinement lost its stricter local kind");
}

void testStrictTypedDecodeRejections() {
  const ArtifactIdentity artifact = identity(0x42);
  EncodedArtifactLocalReference encoded =
      encodeFabricArtifactLocalReference(ArtifactReference<FabricFuTemplateRef>{
          artifact, FabricFuTemplateRef(23)});
  encoded.ownerLocalKind = fabricArtifactLocalReferenceKindOrdinal(
      FabricArtifactLocalReferenceKind::FabricPeOccurrenceRef);
  auto wrongKind =
      decodeFabricArtifactLocalReference<FabricFuTemplateRef>(encoded);
  require(!wrongKind, "wrong owner-local kind was accepted");
  llvm::consumeError(wrongKind.takeError());

  encoded =
      encodeFabricArtifactLocalReference(ArtifactReference<FabricFuTemplateRef>{
          artifact, FabricFuTemplateRef(23)});
  encoded.ownerLocalKind = fabricArtifactLocalReferenceKindCount();
  auto unknownKind =
      decodeFabricArtifactLocalReference<FabricFuTemplateRef>(encoded);
  require(!unknownKind, "unknown owner-local kind was accepted");
  llvm::consumeError(unknownKind.takeError());

  encoded =
      encodeFabricArtifactLocalReference(ArtifactReference<FabricFuTemplateRef>{
          artifact, FabricFuTemplateRef(23)});
  encoded.artifact.schemaIdentity = "foreign.fabric";
  auto foreign =
      decodeFabricArtifactLocalReference<FabricFuTemplateRef>(encoded);
  require(!foreign, "foreign artifact reference was accepted");
  llvm::consumeError(foreign.takeError());

  encoded =
      encodeFabricArtifactLocalReference(ArtifactReference<FabricFuTemplateRef>{
          artifact, FabricFuTemplateRef(23)});
  encoded.artifact.schemaVersion = SchemaVersion{3, 0};
  auto priorMajor =
      decodeFabricArtifactLocalReference<FabricFuTemplateRef>(encoded);
  require(!priorMajor, "loom.fabric 3.0 local reference was accepted");
  llvm::consumeError(priorMajor.takeError());
}

static_assert(!isFabricArtifactLocalReference<ArtifactRootReference>,
              "an Artifact root must not consume a local-kind sentinel");

} // namespace

int main() {
  testGeneratedCatalog();
  testRepresentativeRoundTrips();
  testStrictTypedDecodeRejections();
  return 0;
}
