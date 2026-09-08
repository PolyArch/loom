#include "Mapping/Artifact/SpatialMappingHardwareDemand.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace {

constexpr llvm::StringLiteral feedbackSchema =
    "loom.mapping.spatial_hardware_feedback.2.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_mapping_hardware_demand_invalid: " +
                                     message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendBytes(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
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

llvm::Expected<llvm::ArrayRef<std::uint8_t>>
readBytes(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  auto size = readU64(bytes, offset);
  if (!size)
    return size.takeError();
  if (*size > bytes.size() - offset)
    return invalid("reference bytes are truncated");
  auto result = bytes.slice(offset, *size);
  offset += *size;
  return result;
}

llvm::Error validateRoots(const ArtifactRootReference &module,
                          const ArtifactRootReference &techMapping) {
  if (module.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      module.schemaVersion != fabric::fabricArtifactSchema.version)
    return invalid("target is not an exact Fabric Artifact root");
  if (techMapping.schemaIdentity != mappingArtifactSchema.identity ||
      techMapping.schemaVersion != mappingArtifactSchema.version)
    return invalid("source is not an exact Mapping Artifact root");
  return llvm::Error::success();
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

llvm::Expected<SpatialGraphBoundaryCapacitySuggestion>
SpatialGraphBoundaryCapacitySuggestion::get(
    ArtifactRootReference module, ArtifactRootReference techMapping,
    std::uint64_t inputDemandCount, std::uint64_t inputEndpointCount,
    std::uint64_t outputDemandCount, std::uint64_t outputEndpointCount) {
  if (auto error = validateRoots(module, techMapping))
    return std::move(error);
  if (inputDemandCount >
      std::numeric_limits<std::uint64_t>::max() - outputDemandCount)
    return invalid("directional demand count overflows u64");
  if (inputEndpointCount >
      std::numeric_limits<std::uint64_t>::max() - outputEndpointCount)
    return invalid("directional endpoint count overflows u64");
  if (inputDemandCount <= inputEndpointCount &&
      outputDemandCount <= outputEndpointCount)
    return invalid("boundary proposal has no directional capacity increase");
  return SpatialGraphBoundaryCapacitySuggestion(
      std::move(module), std::move(techMapping), inputDemandCount,
      inputEndpointCount, outputDemandCount, outputEndpointCount);
}

std::uint64_t
SpatialGraphBoundaryCapacitySuggestion::proposedAdditionalBoundaryPairs() const {
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

llvm::Expected<std::optional<SpatialGraphBoundaryCapacitySuggestion>>
deriveSpatialGraphBoundaryCapacitySuggestion(
    const ArtifactRootReference &module,
    const ArtifactRootReference &techMapping, const TechMappingView &tech,
    const fabric::FabricArtifactView &fabric) {
  if (tech.fabricIdentity() != module.artifact ||
      fabric.identity() != module.artifact)
    return invalid("boundary proposal has inconsistent Module owners");
  std::uint64_t inputDemand = 0;
  std::uint64_t outputDemand = 0;
  for (const auto &net : tech.residualLogicalNets()) {
    inputDemand += std::holds_alternative<dataflow::GraphIngressTokenRef>(
        net.producer);
    // One producer can multicast to multiple graph egresses on one endpoint.
    outputDemand += llvm::any_of(net.sinks, [](const auto &sink) {
      return std::holds_alternative<dataflow::GraphEgressTokenRef>(sink);
    });
  }
  std::set<std::vector<std::uint8_t>> inputs;
  std::set<std::vector<std::uint8_t>> outputs;
  for (const auto &attachment : fabric.moduleBoundaryTransportAttachments()) {
    auto &endpoints = attachment.boundary.direction ==
                              fabric::FabricPortDirection::Input
                          ? inputs
                          : outputs;
    endpoints.insert(fabric::canonicalFabricBytes(attachment.endpoint));
  }
  if (inputDemand <= inputs.size() && outputDemand <= outputs.size())
    return std::optional<SpatialGraphBoundaryCapacitySuggestion>();
  auto suggestion = SpatialGraphBoundaryCapacitySuggestion::get(
      module, techMapping, inputDemand, inputs.size(), outputDemand,
      outputs.size());
  if (!suggestion)
    return suggestion.takeError();
  return std::optional<SpatialGraphBoundaryCapacitySuggestion>(
      std::move(*suggestion));
}

llvm::Expected<SpatialFifoChannelCapacitySuggestion>
SpatialFifoChannelCapacitySuggestion::get(
    ArtifactRootReference module, ArtifactRootReference techMapping,
    fabric::FabricFifoOccurrenceRef owner, std::uint64_t selectedChannels,
    std::uint64_t proposedChannels,
    std::vector<dataflow::CanonicalGraphProducerEndpointRef> logicalNets,
    std::vector<fabric::FabricPhysicalTraversalRef> routeAnchors) {
  if (auto error = validateRoots(module, techMapping))
    return std::move(error);
  if (selectedChannels == 0 || proposedChannels <= selectedChannels ||
      logicalNets.empty() || routeAnchors.empty())
    return invalid("FIFO channel proposal has no positive reservation growth "
                   "or retained route witness");
  for (const auto &net : logicalNets) {
    auto encoded = dataflow::encodeDataflowReference(net);
    if (!encoded)
      return encoded.takeError();
  }
  llvm::sort(logicalNets, [](const auto &lhs, const auto &rhs) {
    return llvm::cantFail(dataflow::encodeDataflowReference(lhs)) <
           llvm::cantFail(dataflow::encodeDataflowReference(rhs));
  });
  logicalNets.erase(std::unique(logicalNets.begin(), logicalNets.end()),
                    logicalNets.end());
  llvm::sort(routeAnchors, [](const auto &lhs, const auto &rhs) {
    return fabric::canonicalFabricBytes(lhs) <
           fabric::canonicalFabricBytes(rhs);
  });
  routeAnchors.erase(std::unique(routeAnchors.begin(), routeAnchors.end()),
                     routeAnchors.end());
  return SpatialFifoChannelCapacitySuggestion(
      std::move(module), std::move(techMapping), owner, selectedChannels,
      proposedChannels, std::move(logicalNets), std::move(routeAnchors));
}

llvm::ArrayRef<std::uint8_t> spatialMappingHardwareFeedbackSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(feedbackSchema.data()),
          feedbackSchema.size()};
}

std::vector<std::uint8_t> encodeSpatialMappingHardwareFeedback(
    const SpatialMappingHardwareFeedback &feedback) {
  std::vector<std::uint8_t> bytes;
  std::visit(
      [&](const auto &value) {
        bytes = encodeArtifactRootReference(value.module());
        const auto tech = encodeArtifactRootReference(value.techMapping());
        bytes.insert(bytes.end(), tech.begin(), tech.end());
        appendU64(bytes, feedback.index());
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value,
                                     SpatialGraphBoundaryCapacitySuggestion>) {
          appendU64(bytes, value.inputDemandCount());
          appendU64(bytes, value.inputEndpointCount());
          appendU64(bytes, value.outputDemandCount());
          appendU64(bytes, value.outputEndpointCount());
        } else {
          appendBytes(bytes, fabric::canonicalFabricBytes(value.owner()));
          appendU64(bytes, value.selectedChannels());
          appendU64(bytes, value.proposedChannels());
          appendU64(bytes, value.logicalNets().size());
          for (const auto &net : value.logicalNets())
            appendBytes(bytes,
                        llvm::cantFail(dataflow::encodeDataflowReference(net)));
          appendU64(bytes, value.routeAnchors().size());
          for (const auto &anchor : value.routeAnchors())
            appendBytes(bytes, fabric::canonicalFabricBytes(anchor));
        }
      },
      feedback);
  return bytes;
}

llvm::Expected<SpatialMappingHardwareFeedback>
adoptSpatialMappingHardwareFeedback(
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
  if (*encodedModule != module)
    return invalid("payload names a different Module input");
  if (!llvm::is_contained(techMappings, *techMapping))
    return invalid("payload names a TechMapping outside the input frontier");
  auto imported = importTechMapping(*techMapping, store);
  if (!imported)
    return imported.takeError();
  if (imported->view().fabricIdentity() != module.artifact)
    return invalid("payload TechMapping targets a different Module");
  auto kind = readU64(bytes, offset);
  if (!kind)
    return kind.takeError();
  std::optional<SpatialMappingHardwareFeedback> feedback;
  if (*kind == 0) {
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
    auto fabric = fabric::importEntireFabricRoot(module, store);
    if (!fabric)
      return fabric.takeError();
    auto suggestion = deriveSpatialGraphBoundaryCapacitySuggestion(
        module, *techMapping, imported->view(), fabric->view());
    if (!suggestion)
      return suggestion.takeError();
    if (!*suggestion ||
        (**suggestion).inputDemandCount() != *inputDemandCount ||
        (**suggestion).inputEndpointCount() != *inputEndpointCount ||
        (**suggestion).outputDemandCount() != *outputDemandCount ||
        (**suggestion).outputEndpointCount() != *outputEndpointCount)
      return invalid("boundary proposal disagrees with its exact inputs");
    feedback.emplace(std::move(**suggestion));
  } else if (*kind == 1) {
    auto ownerBytes = readBytes(bytes, offset);
    if (!ownerBytes)
      return ownerBytes.takeError();
    auto owner =
        fabric::decodeFabricRef<fabric::FabricFifoOccurrenceRef>(*ownerBytes);
    if (!owner)
      return owner.takeError();
    auto selected = readU64(bytes, offset);
    if (!selected)
      return selected.takeError();
    auto proposed = readU64(bytes, offset);
    if (!proposed)
      return proposed.takeError();
    auto netCount = readU64(bytes, offset);
    if (!netCount)
      return netCount.takeError();
    if (*netCount > (bytes.size() - offset) / 8)
      return invalid("logical net table is truncated");
    std::vector<dataflow::CanonicalGraphProducerEndpointRef> nets;
    for (std::uint64_t ordinal = 0; ordinal != *netCount; ++ordinal) {
      auto localBytes = readBytes(bytes, offset);
      if (!localBytes)
        return localBytes.takeError();
      auto net = dataflow::decodeDataflowReference<
          dataflow::CanonicalGraphProducerEndpointRef>(
          *localBytes, imported->view().dataflowIdentity());
      if (!net)
        return net.takeError();
      if (!imported->view().residualLogicalNet(*net))
        return invalid("FIFO proposal names no residual TechMapping net");
      nets.push_back(std::move(*net));
    }
    auto anchorCount = readU64(bytes, offset);
    if (!anchorCount)
      return anchorCount.takeError();
    if (*anchorCount > (bytes.size() - offset) / 8)
      return invalid("route anchor table is truncated");
    auto hardware = fabric::importEntireFabricRoot(module, store);
    if (!hardware)
      return hardware.takeError();
    if (auto error = fabric::validateFabricRef(hardware->view(), *owner))
      return std::move(error);
    if (hardware->view().fifoReservedChannels(*owner) != *selected)
      return invalid("FIFO proposal differs from the selected reservation");
    std::vector<fabric::FabricPhysicalTraversalRef> anchors;
    for (std::uint64_t ordinal = 0; ordinal != *anchorCount; ++ordinal) {
      auto localBytes = readBytes(bytes, offset);
      if (!localBytes)
        return localBytes.takeError();
      auto anchor = fabric::decodeFabricRef<fabric::FabricPhysicalTraversalRef>(
          *localBytes);
      if (!anchor)
        return anchor.takeError();
      if (auto error = fabric::validateFabricRef(hardware->view(), *anchor))
        return std::move(error);
      anchors.push_back(std::move(*anchor));
    }
    auto fifo = SpatialFifoChannelCapacitySuggestion::get(
        std::move(*encodedModule), std::move(*techMapping), *owner, *selected,
        *proposed, std::move(nets), std::move(anchors));
    if (!fifo)
      return fifo.takeError();
    feedback.emplace(std::move(*fifo));
  } else {
    return invalid("hardware feedback kind is outside its closed domain");
  }
  if (offset != bytes.size())
    return invalid("payload has trailing bytes");
  const std::vector<std::uint8_t> canonical =
      encodeSpatialMappingHardwareFeedback(*feedback);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("payload is not canonical");
  return std::move(*feedback);
}

void retainSpatialMappingHardwareFeedback(
    std::optional<SpatialMappingHardwareFeedback> &retained,
    SpatialMappingHardwareFeedback candidate) {
  const auto rank = [](const SpatialMappingHardwareFeedback &feedback) {
    return std::visit(
        [&](const auto &value) {
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value,
                                       SpatialGraphBoundaryCapacitySuggestion>)
            return std::make_tuple(feedback.index(),
                                   value.proposedAdditionalBoundaryPairs(),
                                   value.demandCount());
          else
            return std::make_tuple(
                feedback.index(), value.proposedChannels(),
                static_cast<std::uint64_t>(value.logicalNets().size()));
        },
        feedback);
  };
  if (!retained || rank(candidate) > rank(*retained) ||
      (rank(candidate) == rank(*retained) &&
       encodeSpatialMappingHardwareFeedback(candidate) <
           encodeSpatialMappingHardwareFeedback(*retained)))
    retained = std::move(candidate);
}

} // namespace loom::mapping
