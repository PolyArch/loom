#include "Mapping/Artifact/SpatialMappingHardwareDemand.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace {

constexpr llvm::StringLiteral feedbackSchema =
    "loom.mapping.spatial_graph_boundary_endpoint_hall_feedback.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_mapping_hardware_demand_invalid: " +
                                     message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    return invalid("payload is truncated");
  std::uint64_t value = 0;
  for (unsigned index = 0; index != 8; ++index)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<ArtifactRootReference>
readRootReference(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  if (offset > bytes.size())
    return invalid("root-reference offset is outside the payload");
  auto decoded = decodeArtifactRootReferencePrefix(bytes.drop_front(offset));
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount > bytes.size() - offset)
    return invalid("root reference is truncated");
  offset += decoded->byteCount;
  return std::move(decoded->reference);
}

} // namespace

llvm::Expected<SpatialGraphBoundaryEndpointHallDeficit>
SpatialGraphBoundaryEndpointHallDeficit::get(
    ArtifactRootReference module, ArtifactRootReference techMapping,
    std::uint64_t inputDemandCount, std::uint64_t inputEndpointCount,
    std::uint64_t outputDemandCount, std::uint64_t outputEndpointCount) {
  if (module.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      module.schemaVersion != fabric::fabricArtifactSchema.version)
    return invalid("target is not an exact Fabric Artifact root");
  if (techMapping.schemaIdentity != mappingArtifactSchema.identity ||
      techMapping.schemaVersion != mappingArtifactSchema.version)
    return invalid("source is not an exact Mapping Artifact root");
  if (inputDemandCount >
      std::numeric_limits<std::uint64_t>::max() - outputDemandCount)
    return invalid("directional demand count overflows u64");
  if (inputEndpointCount >
      std::numeric_limits<std::uint64_t>::max() - outputEndpointCount)
    return invalid("directional endpoint count overflows u64");
  const std::uint64_t demandCount = inputDemandCount + outputDemandCount;
  const std::uint64_t endpointCount = inputEndpointCount + outputEndpointCount;
  if (demandCount == 0 || endpointCount >= demandCount)
    return invalid("Hall cardinalities are inconsistent or not deficient");
  return SpatialGraphBoundaryEndpointHallDeficit(
      std::move(module), std::move(techMapping), inputDemandCount,
      inputEndpointCount, outputDemandCount, outputEndpointCount);
}

std::uint64_t
SpatialGraphBoundaryEndpointHallDeficit::requiredBoundaryPairs() const {
  const std::uint64_t inputDeficit =
      inputDemandCount_ > inputEndpointCount_
          ? inputDemandCount_ - inputEndpointCount_
          : 0;
  const std::uint64_t outputDeficit =
      outputDemandCount_ > outputEndpointCount_
          ? outputDemandCount_ - outputEndpointCount_
          : 0;
  return std::max(inputDeficit, outputDeficit);
}

llvm::ArrayRef<std::uint8_t>
spatialGraphBoundaryEndpointHallFeedbackSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(feedbackSchema.data()),
          feedbackSchema.size()};
}

std::vector<std::uint8_t> encodeSpatialGraphBoundaryEndpointHallFeedback(
    const SpatialGraphBoundaryEndpointHallDeficit &feedback) {
  std::vector<std::uint8_t> bytes =
      encodeArtifactRootReference(feedback.module());
  const std::vector<std::uint8_t> tech =
      encodeArtifactRootReference(feedback.techMapping());
  bytes.insert(bytes.end(), tech.begin(), tech.end());
  appendU64(bytes, feedback.inputDemandCount());
  appendU64(bytes, feedback.inputEndpointCount());
  appendU64(bytes, feedback.outputDemandCount());
  appendU64(bytes, feedback.outputEndpointCount());
  return bytes;
}

llvm::Expected<SpatialGraphBoundaryEndpointHallDeficit>
adoptSpatialGraphBoundaryEndpointHallFeedback(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &module,
    llvm::ArrayRef<ArtifactRootReference> techMappings,
    const ArtifactStore &store) {
  std::size_t offset = 0;
  auto encodedModule = readRootReference(bytes, offset);
  if (!encodedModule)
    return encodedModule.takeError();
  auto techMapping = readRootReference(bytes, offset);
  if (!techMapping)
    return techMapping.takeError();
  auto inputDemandCount = readU64(bytes, offset);
  if (!inputDemandCount)
    return inputDemandCount.takeError();
  auto inputEndpointCount = readU64(bytes, offset);
  if (!inputEndpointCount)
    return inputEndpointCount.takeError();
  auto outputDemandCount = readU64(bytes, offset);
  if (!outputDemandCount)
    return outputDemandCount.takeError();
  auto outputEndpointCount = readU64(bytes, offset);
  if (!outputEndpointCount)
    return outputEndpointCount.takeError();
  if (offset != bytes.size())
    return invalid("payload has trailing bytes");
  auto feedback = SpatialGraphBoundaryEndpointHallDeficit::get(
      std::move(*encodedModule), std::move(*techMapping), *inputDemandCount,
      *inputEndpointCount, *outputDemandCount, *outputEndpointCount);
  if (!feedback)
    return feedback.takeError();
  if (feedback->module() != module)
    return invalid("payload names a different Module input");
  if (!llvm::is_contained(techMappings, feedback->techMapping()))
    return invalid("payload names a TechMapping outside the input frontier");
  auto imported = importTechMapping(feedback->techMapping(), store);
  if (!imported)
    return imported.takeError();
  if (imported->view().fabricIdentity() != module.artifact)
    return invalid("payload TechMapping targets a different Module");
  const std::vector<std::uint8_t> canonical =
      encodeSpatialGraphBoundaryEndpointHallFeedback(*feedback);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("payload is not canonical");
  return feedback;
}

void retainSpatialGraphBoundaryEndpointHallFeedback(
    std::optional<SpatialGraphBoundaryEndpointHallDeficit> &retained,
    SpatialGraphBoundaryEndpointHallDeficit candidate) {
  if (!retained ||
      candidate.requiredBoundaryPairs() > retained->requiredBoundaryPairs() ||
      (candidate.requiredBoundaryPairs() == retained->requiredBoundaryPairs() &&
       candidate.demandCount() > retained->demandCount()) ||
      (candidate.requiredBoundaryPairs() == retained->requiredBoundaryPairs() &&
       candidate.demandCount() == retained->demandCount() &&
       encodeSpatialGraphBoundaryEndpointHallFeedback(candidate) <
           encodeSpatialGraphBoundaryEndpointHallFeedback(*retained)))
    retained = std::move(candidate);
}

} // namespace loom::mapping
