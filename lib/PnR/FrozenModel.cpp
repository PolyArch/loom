#include "PnR/FrozenModel.h"

#include "FrozenModelInternal.h"

#include "Common/ComponentViewDigest.h"
#include "Mapping/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

namespace {

constexpr char cacheKeyDomain[] = "loom.spatial_pnr.frozen_model.key.v1\0";
constexpr std::size_t cacheKeyDomainSize = sizeof(cacheKeyDomain) - 1;
constexpr std::uint32_t cacheSchemaMajor = 1;
constexpr std::uint32_t cacheSchemaMinor = 0;
constexpr llvm::StringLiteral freezeSemanticIdentity =
    "loom.spatial_pnr.freeze.1.0";
constexpr llvm::StringLiteral importerSemanticIdentity =
    "loom.spatial_pnr.importers.1.0";
constexpr llvm::StringLiteral nativeLayoutAbi =
    "loom.spatial_pnr.native_layout.1.0";

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

llvm::Error invalidModel(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("frozen_model_invalid: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

llvm::Error verifyAggregate(const FrozenRealizationGraph &realizations,
                            const FrozenRoutingGraph &routing) {
  const auto computeEndpoints = realizations.physicalEndpoints();
  const auto computeVertices = routing.computeEndpointVertices();
  const std::size_t routableComputeEndpointCount =
      static_cast<std::size_t>(llvm::count_if(
          computeEndpoints, [](const FrozenPhysicalEndpoint &endpoint) {
            return endpoint.kind != mapping::PortKind::Memory;
          }));
  if (routableComputeEndpointCount != computeVertices.size())
    return invalidModel("compute endpoint projections have different sizes");
  std::size_t vertexIndex = 0;
  for (const FrozenPhysicalEndpoint &endpoint : computeEndpoints) {
    if (endpoint.kind == mapping::PortKind::Memory)
      continue;
    if (computeVertices[vertexIndex] >= routing.routingEndpoints().size())
      return invalidModel("compute endpoint projection is out of range");
    if (routing.routingEndpoints()[computeVertices[vertexIndex]].id !=
        endpoint.id)
      return invalidModel("compute endpoint projections disagree");
    ++vertexIndex;
  }

  const auto memoryEndpoints = realizations.memoryPhysicalEndpoints();
  const auto memoryVertices = routing.memoryEndpointVertices();
  if (memoryEndpoints.size() != memoryVertices.size())
    return invalidModel("memory endpoint projections have different sizes");
  for (std::size_t index = 0; index < memoryEndpoints.size(); ++index) {
    if (memoryVertices[index] >= routing.routingEndpoints().size())
      return invalidModel("memory endpoint projection is out of range");
    if (routing.routingEndpoints()[memoryVertices[index]].id !=
        memoryEndpoints[index].id)
      return invalidModel("memory endpoint projections disagree");
  }
  return llvm::Error::success();
}

} // namespace

FrozenModelCacheKey detail::FrozenModelBuilder::deriveValidatedCacheKey(
    const PnrProblemInputs &inputs) {
  std::vector<std::uint8_t> preimage;
  preimage.reserve(cacheKeyDomainSize + 2 * sizeof(std::uint32_t) + 512);
  preimage.insert(preimage.end(), cacheKeyDomain,
                  cacheKeyDomain + cacheKeyDomainSize);
  appendU32Be(preimage, cacheSchemaMajor);
  appendU32Be(preimage, cacheSchemaMinor);
  appendField(preimage, CacheField::DataflowIdentity,
              inputs.dataflow.identity.bytes());
  appendField(preimage, CacheField::TechMappingIdentity,
              inputs.techMapping.identity().bytes());
  appendField(preimage, CacheField::FabricIdentity,
              inputs.fabric.identity.bytes());
  appendField(preimage, CacheField::ConstraintSetIdentity,
              inputs.constraints.identity.bytes());
  appendField(preimage, CacheField::ConfigViewDescriptor,
              inputs.config.schemaDescriptorBytes());
  appendField(preimage, CacheField::ConfigViewDigest,
              inputs.config.digest().bytes());
  appendField(preimage, CacheField::FreezeSemanticIdentity,
              freezeSemanticIdentity);
  appendField(preimage, CacheField::ImporterSemanticIdentity,
              importerSemanticIdentity);
  appendField(preimage, CacheField::NativeLayoutAbi, nativeLayoutAbi);
  appendU32Field(preimage, CacheField::PnrIndexWidth, getPnrIndexBits());
  return FrozenModelCacheKey(llvm::SHA256::hash(preimage));
}

llvm::Expected<FrozenModelCacheKey>
detail::FrozenModelBuilder::deriveCacheKey(const PnrProblemInputs &inputs) {
  if (llvm::Error error = validatePnrProblemInputs(inputs))
    return std::move(error);
  if (llvm::Error error = validateComponentViewDigest(
          inputs.config.schemaDescriptorBytes(),
          inputs.config.canonicalViewBytes(), inputs.config.digest()))
    return std::move(error);
  return deriveValidatedCacheKey(inputs);
}

llvm::Expected<FrozenModelHandle>
detail::FrozenModelBuilder::build(const PnrProblemInputs &inputs) {
  if (llvm::Error error = validatePnrProblemInputs(inputs))
    return std::move(error);
  if (llvm::Error error = validateComponentViewDigest(
          inputs.config.schemaDescriptorBytes(),
          inputs.config.canonicalViewBytes(), inputs.config.digest()))
    return std::move(error);

  auto realizations = buildRealizations(inputs);
  if (!realizations)
    return realizations.takeError();
  auto routing = buildRouting(inputs);
  if (!routing)
    return routing.takeError();
  if (llvm::Error error = verifyAggregate(*realizations, *routing))
    return std::move(error);

  FrozenModelCacheKey cacheKey = deriveValidatedCacheKey(inputs);
  std::vector<DeterministicWorkBudgetEntry> workBudget =
      deriveDeterministicWorkBudgetView(inputs.config);
  return FrozenModelHandle(
      new FrozenModel(inputs.dataflow.identity, inputs.techMapping.identity(),
                      inputs.fabric.identity, inputs.constraints.identity,
                      inputs.config, std::move(workBudget),
                      std::move(*realizations), std::move(*routing), cacheKey));
}

llvm::Error
detail::FrozenModelBuilder::revalidateCacheHit(const FrozenModel &model,
                                               const PnrProblemInputs &inputs) {
  if (llvm::Error error = validatePnrProblemInputs(inputs))
    return error;
  if (llvm::Error error = validateComponentViewDigest(
          inputs.config.schemaDescriptorBytes(),
          inputs.config.canonicalViewBytes(), inputs.config.digest()))
    return error;
  if (model.dataflowIdentity() != inputs.dataflow.identity ||
      model.techMappingIdentity() != inputs.techMapping.identity() ||
      model.fabricIdentity() != inputs.fabric.identity ||
      model.constraintSetIdentity() != inputs.constraints.identity)
    return invalidModel("cache hit does not bind the exact artifact inputs");
  if (model.config().schemaDescriptorBytes() !=
          inputs.config.schemaDescriptorBytes() ||
      model.config().canonicalViewBytes() !=
          inputs.config.canonicalViewBytes() ||
      model.config().digest() != inputs.config.digest())
    return invalidModel("cache hit does not bind the exact PnR config view");
  if (model.cacheKey() != deriveValidatedCacheKey(inputs))
    return invalidModel("cache key does not match its dependency closure");
  return llvm::Error::success();
}

llvm::Expected<FrozenModelCacheKey>
loom::pnr::deriveFrozenModelCacheKey(const PnrProblemInputs &inputs) {
  return detail::FrozenModelBuilder::deriveCacheKey(inputs);
}

llvm::Expected<FrozenModelHandle>
loom::pnr::freezeSpatialPnrModel(const PnrProblemInputs &inputs) {
  return detail::FrozenModelBuilder::build(inputs);
}

llvm::Error
loom::pnr::revalidateFrozenModelCacheHit(const FrozenModel &model,
                                         const PnrProblemInputs &inputs) {
  return detail::FrozenModelBuilder::revalidateCacheHit(model, inputs);
}

std::string
loom::pnr::formatFrozenModelCacheKeyHex(const FrozenModelCacheKey &key) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(FrozenModelCacheKey::byteSize * 2);
  for (std::uint8_t byte : key.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}
