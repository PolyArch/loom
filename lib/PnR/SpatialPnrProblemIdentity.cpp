#include "SpatialPnrProblemIdentity.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr char cacheKeyDomain[] = "loom.spatial_pnr.frozen_model.key.v2.23\0";
constexpr std::size_t cacheKeyDomainSize = sizeof(cacheKeyDomain) - 1;
constexpr std::uint32_t cacheSchemaMajor = 2;
// Bumped with loom.mapping_constraints 1.1: freeze now indexes cross-projection
// runtime-counterexample no-goods, so a 2.22 frozen model is not reusable.
constexpr std::uint32_t cacheSchemaMinor = 23;
constexpr llvm::StringLiteral freezeSemanticIdentity =
    "loom.spatial_pnr.freeze.2.23";
constexpr llvm::StringLiteral importerSemanticIdentity =
    "loom.spatial_pnr.importers.2.1";
constexpr llvm::StringLiteral nativeLayoutAbi =
    "loom.spatial_pnr.native_layout.2.11";

enum class CacheField : std::uint32_t {
  DataflowIdentity = 1,
  TechMappingIdentity = 2,
  FabricIdentity = 3,
  ConstraintSetIdentity = 4,
  ConfigViewDescriptor = 5,
  ConfigViewDigest = 6,
  FreezeSemanticIdentity = 7,
  ImporterSemanticIdentity = 8,
  NativeLayoutAbi = 9,
  PnrIndexWidth = 10,
  PhysicalTimingProfileDigest = 11,
};

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendField(std::vector<std::uint8_t> &bytes, CacheField field,
                 llvm::ArrayRef<std::uint8_t> value) {
  appendU32Be(bytes, static_cast<std::uint32_t>(field));
  appendU64Be(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendField(std::vector<std::uint8_t> &bytes, CacheField field,
                 llvm::StringRef value) {
  appendField(
      bytes, field,
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(value.data()), value.size()));
}

void appendU32Field(std::vector<std::uint8_t> &bytes, CacheField field,
                    std::uint32_t value) {
  std::vector<std::uint8_t> encoded;
  encoded.reserve(4);
  appendU32Be(encoded, value);
  appendField(bytes, field, encoded);
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

} // namespace

llvm::Error loom::pnr::detail::SpatialPnrProblemIdentity::validateInputs(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraintSet) {
  if (config.domain() != PnrConfigDomain::Spatial)
    return invalid("Spatial PnR requires the Spatial config projection");
  if (techMapping.dataflowIdentity() != dataflow.identity())
    return invalid("TechMapping is bound to a different Dataflow artifact");
  if (techMapping.fabricIdentity() != fabric.identity())
    return invalid("TechMapping is bound to a different Fabric artifact");
  if (constraintSet.dataflowIdentity() != dataflow.identity() ||
      constraintSet.techMappingIdentity() != techMapping.identity() ||
      constraintSet.fabricIdentity() != fabric.identity())
    return invalid("MappingConstraintSet is bound to a different D/T/F tuple");
  if (fabric.rootKind() != FabricRootKind::Module)
    return invalid("Spatial PnR requires one fully elaborated Module root");
  if (llvm::Error error = validateComponentViewDigest(
          config.schemaDescriptorBytes(), config.canonicalViewBytes(),
          config.digest()))
    return llvm::joinErrors(
        invalid("PnR config component-view digest is invalid"),
        std::move(error));
  return llvm::Error::success();
}

FrozenSpatialPnrCacheKey
loom::pnr::detail::SpatialPnrProblemIdentity::deriveCacheKey(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraintSet,
    const ComponentViewDigest &physicalTimingDigest) {
  std::vector<std::uint8_t> preimage;
  preimage.reserve(cacheKeyDomainSize + 2 * sizeof(std::uint32_t) + 512);
  preimage.insert(preimage.end(), cacheKeyDomain,
                  cacheKeyDomain + cacheKeyDomainSize);
  appendU32Be(preimage, cacheSchemaMajor);
  appendU32Be(preimage, cacheSchemaMinor);
  appendField(preimage, CacheField::DataflowIdentity,
              dataflow.identity().bytes());
  appendField(preimage, CacheField::TechMappingIdentity,
              techMapping.identity().bytes());
  appendField(preimage, CacheField::FabricIdentity, fabric.identity().bytes());
  appendField(preimage, CacheField::ConstraintSetIdentity,
              constraintSet.identity().bytes());
  appendField(preimage, CacheField::ConfigViewDescriptor,
              config.schemaDescriptorBytes());
  appendField(preimage, CacheField::ConfigViewDigest, config.digest().bytes());
  appendField(preimage, CacheField::FreezeSemanticIdentity,
              freezeSemanticIdentity);
  appendField(preimage, CacheField::ImporterSemanticIdentity,
              importerSemanticIdentity);
  appendField(preimage, CacheField::NativeLayoutAbi, nativeLayoutAbi);
  appendU32Field(preimage, CacheField::PnrIndexWidth, getPnrIndexBits());
  appendField(preimage, CacheField::PhysicalTimingProfileDigest,
              physicalTimingDigest.bytes());
  return FrozenSpatialPnrCacheKey(llvm::SHA256::hash(preimage));
}

llvm::Error loom::pnr::detail::SpatialPnrProblemIdentity::revalidateCacheHit(
    const FrozenSpatialPnrProblem &problem,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const ResolvedPnrConfigView &config,
    const SpatialMappingConstraintSetView &constraintSet) {
  if (llvm::Error error =
          validateInputs(dataflow, techMapping, fabric, config, constraintSet))
    return error;
  if (problem.dataflowIdentity() != dataflow.identity() ||
      problem.techMappingIdentity() != techMapping.identity() ||
      problem.fabricIdentity() != fabric.identity() ||
      problem.constraintSetIdentity() != constraintSet.identity())
    return invalid("cache hit does not bind the exact artifact inputs");
  if (problem.config().schemaDescriptorBytes() !=
          config.schemaDescriptorBytes() ||
      problem.config().canonicalViewBytes() != config.canonicalViewBytes() ||
      problem.config().digest() != config.digest())
    return invalid("cache hit does not bind the exact PnR config view");
  if (llvm::Error error = ::loom::fabric::validateFabricPhysicalTimingProfile(
          fabric, physicalTiming))
    return error;
  if (problem.cacheKey() !=
      deriveCacheKey(dataflow, techMapping, fabric, config, constraintSet,
                     physicalTiming.digest()))
    return invalid("cache key does not match its dependency closure");
  return llvm::Error::success();
}
