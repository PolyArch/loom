#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "Fabric/Identity/FabricMemoryConfiguration.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_semantic_field_relation_rejected: " +
                                     message);
}

CanonicalSemanticBytes tagged(std::uint32_t tag) {
  FabricByteWriter writer;
  writer.tag(tag);
  return CanonicalSemanticBytes(writer.take());
}

template <typename Ref>
CanonicalSemanticBytes tagged(std::uint32_t tag, const Ref &reference) {
  FabricByteWriter writer;
  writer.tag(tag);
  encodeFabricRef(writer, reference);
  return CanonicalSemanticBytes(writer.take());
}

std::vector<std::uint8_t> zeroBits(std::uint64_t bitCount) {
  return std::vector<std::uint8_t>(
      static_cast<std::size_t>(bitCount / 8 + (bitCount % 8 != 0)), 0);
}

bool bit(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t index) {
  return ((bytes[static_cast<std::size_t>(index / 8)] >> (index % 8)) & 1U) !=
         0;
}

void setBit(std::vector<std::uint8_t> &bytes, std::uint64_t index) {
  bytes[static_cast<std::size_t>(index / 8)] |=
      static_cast<std::uint8_t>(1U << (index % 8));
}

llvm::Error validateBitCarrier(llvm::ArrayRef<std::uint8_t> bytes,
                               std::uint64_t bitCount) {
  const std::uint64_t byteCount = bitCount / 8 + (bitCount % 8 != 0);
  if (bytes.size() != byteCount)
    return rejected("direct value has the wrong byte count");
  const unsigned usedBits = static_cast<unsigned>(bitCount % 8);
  if (usedBits != 0 && !bytes.empty() &&
      (bytes.back() & static_cast<std::uint8_t>(0xffU << usedBits)) != 0)
    return rejected("direct value has nonzero padding bits");
  return llvm::Error::success();
}

struct SwitchCrosspoint final {
  FabricOrdinal input = 0;
  FabricOrdinal output = 0;
};

std::vector<SwitchCrosspoint>
switchCrosspoints(const FabricArtifactView &fabric,
                  FabricSwitchOccurrenceRef sw) {
  std::vector<SwitchCrosspoint> result;
  for (const FabricPhysicalTraversalRef &traversal :
       fabric.admittedTraversals()) {
    const auto *payload =
        std::get_if<FabricSwitchTraversalPayload>(&traversal.payload);
    if (payload && payload->owner == sw)
      result.push_back({payload->input, payload->output});
  }
  llvm::sort(result, [](const SwitchCrosspoint &lhs,
                        const SwitchCrosspoint &rhs) {
    return std::tie(lhs.output, lhs.input) < std::tie(rhs.output, rhs.input);
  });
  return result;
}

} // namespace

llvm::Error FabricSemanticFieldRelation::validateSemanticValue(
    llvm::ArrayRef<std::uint8_t> value) const {
  switch (kind_) {
  case FabricSemanticFieldRelationKind::None:
    return rejected("field has no semantic configuration domain");
  case FabricSemanticFieldRelationKind::Finite:
    if (validator_)
      return validator_(value);
    for (const CanonicalSemanticBytes &candidate : finiteDomain_)
      if (candidate.bytes().equals(value))
        return llvm::Error::success();
    return rejected("value is outside the finite semantic domain");
  case FabricSemanticFieldRelationKind::Direct:
    if (!validator_)
      return rejected("direct semantic domain has no validator");
    return validator_(value);
  }
  llvm_unreachable("unknown Fabric semantic field relation kind");
}

llvm::Expected<FabricSemanticFieldRelation>
FabricArtifactView::semanticFieldRelation(
    const FabricSemanticConfigFieldRef &field,
    ::mlir::MLIRContext &context) const {
  if (llvm::Error error = validateFabricRef(*this, field))
    return error;

  const FabricInventoryOwnerRef &owner = field.owner.catalog();
  if (owner.kind() == FabricInventoryOwnerKind::PeOccurrence) {
    const FabricPeOccurrenceRef pe =
        std::get<FabricPeOccurrenceRef>(owner.payload);
    if (peSchedule(pe) == ::fabric::Schedule::Temporal) {
      if (field.ordinal != 0)
        return rejected("Temporal PE field ordinal is not zero");
      auto schema = temporalPeConfigurationSchema(pe);
      if (!schema)
        return schema.takeError();
      const std::uint64_t width = schema->layout().carrierBitCount;
      auto shared = std::make_shared<FabricTemporalPeConfigurationSchemaView>(
          std::move(*schema));
      return FabricSemanticFieldRelation(
          FabricSemanticFieldRelationKind::Direct, {}, width,
          [shared](llvm::ArrayRef<std::uint8_t> value) -> llvm::Error {
            auto decoded = shared->decode(value);
            if (!decoded)
              return decoded.takeError();
            return llvm::Error::success();
          });
    }
    auto schema = spatialPeConfigurationSchema(pe);
    if (!schema)
      return schema.takeError();
    auto values = schema->finiteDomain(field);
    if (!values)
      return values.takeError();
    std::vector<CanonicalSemanticBytes> domain;
    domain.reserve(values->size());
    for (const FabricPeConfigurationValue &value : *values) {
      auto encoded = schema->encode(field, value);
      if (!encoded)
        return encoded.takeError();
      domain.push_back(std::move(*encoded));
    }
    if (domain.empty())
      return rejected("Spatial PE field has an empty semantic domain");
    return FabricSemanticFieldRelation(FabricSemanticFieldRelationKind::Finite,
                                       std::move(domain), 0);
  }

  if (owner.kind() == FabricInventoryOwnerKind::FuOccurrenceNode) {
    const auto occurrence = std::get<FabricFuOccurrenceNodeRef>(owner.payload);
    const auto *capability = resolvedFabricOpCapability(occurrence);
    if (!capability)
      return rejected("operation field has no concrete capability");
    auto operationRelation = capability->resolveSemanticFieldRelation(context);
    if (!operationRelation)
      return operationRelation.takeError();
    if (operationRelation->kind() ==
        ::fabric::FabricOpSemanticFieldRelationKind::None)
      return rejected("fixed operation capability owns no field");

    if (operationRelation->kind() ==
        ::fabric::FabricOpSemanticFieldRelationKind::Finite) {
      auto shared = std::make_shared<::fabric::FabricOpSemanticFieldRelation>(
          std::move(*operationRelation));
      std::vector<CanonicalSemanticBytes> domain;
      for (const auto &point : shared->finiteBehaviorDomain()) {
        if (!point.semanticConfiguration)
          return rejected("finite operation behavior has no semantic value");
        domain.push_back(*point.semanticConfiguration);
      }
      if (domain.empty())
        return rejected("operation field has an empty finite domain");
      return FabricSemanticFieldRelation(
          FabricSemanticFieldRelationKind::Finite, std::move(domain), 0,
          [shared](llvm::ArrayRef<std::uint8_t> value) {
            return shared->validateSemanticValue(value);
          });
    }

    const std::uint64_t width = *operationRelation->directEncodedBitCount();
    auto shared = std::make_shared<::fabric::FabricOpSemanticFieldRelation>(
        std::move(*operationRelation));
    return FabricSemanticFieldRelation(
        FabricSemanticFieldRelationKind::Direct, {}, width,
        [shared](llvm::ArrayRef<std::uint8_t> value) {
          return shared->validateSemanticValue(value);
        });
  }

  if (owner.kind() == FabricInventoryOwnerKind::FuOccurrence) {
    if (field.ordinal != 0)
      return rejected("FU topology field ordinal is not zero");
    const auto fu = std::get<FabricFuOccurrenceRef>(owner.payload);
    const auto definition = fuTemplateOf(fu);
    if (!definition)
      return rejected("FU occurrence has no exact template");
    std::vector<CanonicalSemanticBytes> domain;
    domain.push_back(tagged(0));
    const auto templates = fuCapabilityTemplates(*definition);
    domain.reserve(1 + templates.size());
    for (FabricOrdinal ordinal = 0; ordinal < templates.size(); ++ordinal)
      domain.push_back(
          tagged(1, FabricFuCapabilityTemplateRef{*definition, ordinal}));
    if (domain.size() == 1)
      return rejected("FU occurrence has no capability template");
    return FabricSemanticFieldRelation(FabricSemanticFieldRelationKind::Finite,
                                       std::move(domain), 0);
  }

  if (owner.kind() == FabricInventoryOwnerKind::MemoryOccurrence) {
    if (field.ordinal != 0)
      return rejected("memory configuration field ordinal is not zero");
    const auto memory = std::get<FabricMemoryOccurrenceRef>(owner.payload);
    auto schema = memoryConfigurationSchema(memory);
    if (!schema)
      return schema.takeError();
    const std::uint64_t width = schema->layout().carrierBitCount;
    auto shared = std::make_shared<FabricMemoryConfigurationSchemaView>(
        std::move(*schema));
    return FabricSemanticFieldRelation(
        FabricSemanticFieldRelationKind::Direct, {}, width,
        [shared](llvm::ArrayRef<std::uint8_t> value) -> llvm::Error {
          auto decoded = shared->decode(value);
          if (!decoded)
            return decoded.takeError();
          return llvm::Error::success();
        });
  }

  if (owner.kind() == FabricInventoryOwnerKind::FifoOccurrence) {
    if (field.ordinal != 0)
      return rejected("FIFO mode field ordinal is not zero");
    const auto fifo = std::get<FabricFifoOccurrenceRef>(owner.payload);
    std::vector<CanonicalSemanticBytes> domain;
    domain.push_back(tagged(0));
    domain.push_back(tagged(1));
    if (admitsTraversal(FabricPhysicalTraversalRef::fifoTraversal(
            fifo, FabricFifoTraversalMode::Bypass)))
      domain.push_back(tagged(2));
    return FabricSemanticFieldRelation(FabricSemanticFieldRelationKind::Finite,
                                       std::move(domain), 0);
  }

  if (owner.kind() == FabricInventoryOwnerKind::BoundaryOccurrence) {
    if (field.ordinal != 0)
      return rejected("boundary configuration field ordinal is not zero");
    const auto boundary = std::get<FabricBoundaryOccurrenceRef>(owner.payload);
    const auto point = boundaryTagContinuityPoint(boundary);
    if (!point)
      return rejected("boundary has no exact continuity shape");
    if (point->kind == FabricBoundaryTagContinuityKind::TokenWriter ||
        point->kind == FabricBoundaryTagContinuityKind::Remover)
      return FabricSemanticFieldRelation(
          FabricSemanticFieldRelationKind::Finite, {tagged(0), tagged(1)}, 0);

    if (point->kind == FabricBoundaryTagContinuityKind::ConfigurableWriter) {
      const std::uint64_t width = 1 + point->outputTagWidthBits;
      auto validator =
          [width](llvm::ArrayRef<std::uint8_t> value) -> llvm::Error {
        if (llvm::Error error = validateBitCarrier(value, width))
          return error;
        if (!bit(value, 0))
          for (std::uint64_t index = 1; index < width; ++index)
            if (bit(value, index))
              return rejected(
                  "disabled boundary tag carrier has nonzero payload");
        return llvm::Error::success();
      };
      return FabricSemanticFieldRelation(
          FabricSemanticFieldRelationKind::Direct, {}, width,
          std::move(validator));
    }

    const std::uint64_t rowCount = boundaryLookupTableSize(boundary);
    const std::uint64_t rowWidth =
        1 + point->inputTagWidthBits + point->outputTagWidthBits;
    if (rowCount == 0 || rowWidth == 0 || rowCount > UINT64_MAX / rowWidth)
      return rejected("boundary lookup carrier is too large");
    const std::uint64_t width = rowCount * rowWidth;
    auto validator = [rowCount, rowWidth,
                      inputWidth = point->inputTagWidthBits](
                         llvm::ArrayRef<std::uint8_t> value) -> llvm::Error {
      const std::uint64_t width = rowCount * rowWidth;
      if (llvm::Error error = validateBitCarrier(value, width))
        return error;
      std::set<std::vector<std::uint8_t>> inputTags;
      for (std::uint64_t row = 0; row < rowCount; ++row) {
        const std::uint64_t base = row * rowWidth;
        if (!bit(value, base)) {
          for (std::uint64_t offset = 1; offset < rowWidth; ++offset)
            if (bit(value, base + offset))
              return rejected(
                  "inactive boundary lookup row has nonzero payload");
          continue;
        }
        std::vector<std::uint8_t> inputTag(
            static_cast<std::size_t>((inputWidth + 7) / 8), 0);
        for (std::uint64_t tagBit = 0; tagBit < inputWidth; ++tagBit)
          if (bit(value, base + 1 + tagBit))
            inputTag[static_cast<std::size_t>(tagBit / 8)] |=
                static_cast<std::uint8_t>(1U << (tagBit % 8));
        if (!inputTags.insert(std::move(inputTag)).second)
          return rejected("boundary lookup rows repeat an input tag");
      }
      return llvm::Error::success();
    };
    return FabricSemanticFieldRelation(FabricSemanticFieldRelationKind::Direct,
                                       {}, width, std::move(validator));
  }

  if (owner.kind() == FabricInventoryOwnerKind::SwitchOccurrence) {
    if (field.ordinal != 0)
      return rejected("switch route field ordinal is not zero");
    const auto sw = std::get<FabricSwitchOccurrenceRef>(owner.payload);
    std::vector<SwitchCrosspoint> crosspoints = switchCrosspoints(*this, sw);
    if (crosspoints.empty())
      return rejected("switch has no admitted crosspoint");

    const auto schedule = switchSchedule(sw);
    if (!schedule)
      return rejected("switch has no schedule");
    std::uint64_t entryCount = 1;
    std::uint64_t tagWidth = 0;
    if (*schedule == ::fabric::Schedule::Temporal) {
      entryCount = switchRouteTableSize(sw);
      const FabricTransportEndpointRef first{
          FabricTransportEndpointOwnerRef::of(sw), 0};
      const auto path = transportEndpointDataPath(first);
      if (entryCount == 0 || !path || path->tagWidthBits == 0)
        return rejected("temporal switch has an incomplete route shape");
      tagWidth = path->tagWidthBits;
    } else if (*schedule != ::fabric::Schedule::Spatial) {
      return rejected("switch has an unknown schedule");
    }

    const std::uint64_t entryWidth =
        (*schedule == ::fabric::Schedule::Temporal ? 1 + tagWidth : 0) +
        crosspoints.size();
    if (entryCount > UINT64_MAX / entryWidth)
      return rejected("switch route carrier is too large");
    const std::uint64_t width = entryCount * entryWidth;
    auto validator =
        [crosspoints, schedule = *schedule, entryCount, tagWidth,
         entryWidth](llvm::ArrayRef<std::uint8_t> value) -> llvm::Error {
      const std::uint64_t width = entryCount * entryWidth;
      if (llvm::Error error = validateBitCarrier(value, width))
        return error;
      std::set<std::uint64_t> tags;
      for (std::uint64_t entry = 0; entry < entryCount; ++entry) {
        const std::uint64_t base = entry * entryWidth;
        const bool valid =
            schedule == ::fabric::Schedule::Spatial || bit(value, base);
        const std::uint64_t routeBase =
            base +
            (schedule == ::fabric::Schedule::Temporal ? 1 + tagWidth : 0);
        if (!valid) {
          for (std::uint64_t offset = 1; offset < entryWidth; ++offset)
            if (bit(value, base + offset))
              return rejected("unused switch row has nonzero payload");
          continue;
        }
        std::set<FabricOrdinal> selectedOutputs;
        bool selected = false;
        for (const auto &[ordinal, crosspoint] : llvm::enumerate(crosspoints)) {
          if (!bit(value, routeBase + ordinal))
            continue;
          selected = true;
          if (!selectedOutputs.insert(crosspoint.output).second)
            return rejected("switch row selects fan-in for one output");
        }
        if (schedule == ::fabric::Schedule::Temporal) {
          if (!selected)
            return rejected("active temporal switch row selects no route");
          std::uint64_t tag = 0;
          for (std::uint64_t tagBit = 0; tagBit < tagWidth; ++tagBit)
            tag |= static_cast<std::uint64_t>(bit(value, base + 1 + tagBit))
                   << tagBit;
          if (!tags.insert(tag).second)
            return rejected("temporal switch rows repeat a tag");
        }
      }
      return llvm::Error::success();
    };
    return FabricSemanticFieldRelation(FabricSemanticFieldRelationKind::Direct,
                                       {}, width, std::move(validator));
  }

  return rejected("field owner has no shared semantic relation");
}

llvm::Expected<CanonicalSemanticBytes> encodeFabricFuConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricFuCapabilityTemplateRef> activeTemplate) {
  const FabricInventoryOwnerRef &owner = field.owner.catalog();
  if (owner.kind() != FabricInventoryOwnerKind::FuOccurrence ||
      field.ordinal != 0)
    return rejected("FU codec received a non-FU topology field");
  const auto fu = std::get<FabricFuOccurrenceRef>(owner.payload);
  const auto definition = fabric.fuTemplateOf(fu);
  if (!definition)
    return rejected("FU codec cannot resolve the occurrence template");
  CanonicalSemanticBytes encoded =
      activeTemplate ? tagged(1, *activeTemplate) : tagged(0);
  if (activeTemplate && activeTemplate->fu != *definition)
    return rejected("FU codec selected a template from another definition");
  mlir::MLIRContext context;
  auto relation = fabric.semanticFieldRelation(field, context);
  if (!relation)
    return relation.takeError();
  if (llvm::Error error = relation->validateSemanticValue(encoded.bytes()))
    return std::move(error);
  return encoded;
}

llvm::Expected<CanonicalSemanticBytes> encodeFabricFifoConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricFifoTraversalMode> activeMode) {
  const FabricInventoryOwnerRef &owner = field.owner.catalog();
  if (owner.kind() != FabricInventoryOwnerKind::FifoOccurrence ||
      field.ordinal != 0)
    return rejected("FIFO codec received a non-FIFO mode field");
  std::uint32_t tag = 0;
  if (activeMode)
    tag = *activeMode == FabricFifoTraversalMode::Buffered ? 1 : 2;
  CanonicalSemanticBytes encoded = tagged(tag);
  mlir::MLIRContext context;
  auto relation = fabric.semanticFieldRelation(field, context);
  if (!relation)
    return relation.takeError();
  if (llvm::Error error = relation->validateSemanticValue(encoded.bytes()))
    return std::move(error);
  return encoded;
}

llvm::Expected<CanonicalSemanticBytes> encodeSpatialSwitchConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    llvm::ArrayRef<FabricPhysicalTraversalRef> selectedTraversals) {
  const FabricInventoryOwnerRef &owner = field.owner.catalog();
  if (owner.kind() != FabricInventoryOwnerKind::SwitchOccurrence ||
      field.ordinal != 0)
    return rejected("switch codec received a non-switch route field");
  const auto sw = std::get<FabricSwitchOccurrenceRef>(owner.payload);
  if (fabric.switchSchedule(sw) != ::fabric::Schedule::Spatial)
    return rejected("Spatial switch codec received a Temporal switch");
  const std::vector<SwitchCrosspoint> crosspoints =
      switchCrosspoints(fabric, sw);
  std::vector<std::uint8_t> carrier = zeroBits(crosspoints.size());
  for (const FabricPhysicalTraversalRef &traversal : selectedTraversals) {
    const auto *payload =
        std::get_if<FabricSwitchTraversalPayload>(&traversal.payload);
    if (!payload || payload->owner != sw)
      return rejected("switch codec received a foreign traversal");
    const auto found = llvm::find_if(crosspoints, [&](const auto &candidate) {
      return candidate.input == payload->input &&
             candidate.output == payload->output;
    });
    if (found == crosspoints.end())
      return rejected("switch codec received an unadmitted traversal");
    setBit(carrier, static_cast<std::uint64_t>(found - crosspoints.begin()));
  }
  CanonicalSemanticBytes encoded(std::move(carrier));
  mlir::MLIRContext context;
  auto relation = fabric.semanticFieldRelation(field, context);
  if (!relation)
    return relation.takeError();
  if (llvm::Error error = relation->validateSemanticValue(encoded.bytes()))
    return std::move(error);
  return encoded;
}

llvm::Expected<CanonicalSemanticBytes> encodeFabricBoundaryConfiguration(
    const FabricArtifactView &fabric, const FabricSemanticConfigFieldRef &field,
    std::optional<FabricBoundaryConfiguration> activeConfiguration) {
  const FabricInventoryOwnerRef &owner = field.owner.catalog();
  if (owner.kind() != FabricInventoryOwnerKind::BoundaryOccurrence ||
      field.ordinal != 0)
    return rejected("boundary codec received a non-boundary field");
  const auto boundary = std::get<FabricBoundaryOccurrenceRef>(owner.payload);
  const auto point = fabric.boundaryTagContinuityPoint(boundary);
  if (!point)
    return rejected("boundary codec cannot resolve the continuity shape");

  mlir::MLIRContext context;
  auto relation = fabric.semanticFieldRelation(field, context);
  if (!relation)
    return relation.takeError();

  CanonicalSemanticBytes encoded = tagged(0);
  using Kind = FabricBoundaryTagContinuityKind;
  switch (point->kind) {
  case Kind::TokenWriter:
  case Kind::Remover:
    if (activeConfiguration &&
        (activeConfiguration->configuredTag ||
         !activeConfiguration->tagRewrites.empty()))
      return rejected("payload-free boundary received configuration payload");
    encoded = tagged(activeConfiguration ? 1 : 0);
    break;
  case Kind::ConfigurableWriter: {
    const std::uint64_t width = 1 + point->outputTagWidthBits;
    std::vector<std::uint8_t> carrier = zeroBits(width);
    if (activeConfiguration) {
      if (!activeConfiguration->configuredTag ||
          !activeConfiguration->tagRewrites.empty())
        return rejected("configurable tag writer requires exactly one tag");
      const llvm::APInt &tag = *activeConfiguration->configuredTag;
      if (tag.getBitWidth() != point->outputTagWidthBits)
        return rejected("configured boundary tag has the wrong width");
      setBit(carrier, 0);
      for (std::uint64_t bit = 0; bit < point->outputTagWidthBits; ++bit)
        if (tag[bit])
          setBit(carrier, 1 + bit);
    }
    encoded = CanonicalSemanticBytes(std::move(carrier));
    break;
  }
  case Kind::Rewriter: {
    if (activeConfiguration && activeConfiguration->configuredTag)
      return rejected("tag rewriter cannot carry a configured writer tag");
    std::vector<FabricBoundaryTagRewrite> rewrites =
        activeConfiguration ? std::move(activeConfiguration->tagRewrites)
                            : std::vector<FabricBoundaryTagRewrite>();
    if (rewrites.size() > fabric.boundaryLookupTableSize(boundary))
      return rejected("boundary tag rewrite set exceeds its lookup table");
    for (const FabricBoundaryTagRewrite &rewrite : rewrites)
      if (rewrite.inputTag.getBitWidth() != point->inputTagWidthBits ||
          rewrite.outputTag.getBitWidth() != point->outputTagWidthBits)
        return rejected("boundary tag rewrite has the wrong width");
    llvm::sort(rewrites, [](const auto &lhs, const auto &rhs) {
      if (lhs.inputTag != rhs.inputTag)
        return lhs.inputTag.ult(rhs.inputTag);
      return lhs.outputTag.ult(rhs.outputTag);
    });
    std::vector<FabricBoundaryTagRewrite> canonicalRewrites;
    canonicalRewrites.reserve(rewrites.size());
    for (FabricBoundaryTagRewrite &rewrite : rewrites) {
      if (canonicalRewrites.empty() ||
          canonicalRewrites.back().inputTag != rewrite.inputTag) {
        canonicalRewrites.push_back(std::move(rewrite));
        continue;
      }
      if (canonicalRewrites.back().outputTag != rewrite.outputTag)
        return rejected("boundary tag rewrites repeat an input tag");
    }
    rewrites = std::move(canonicalRewrites);

    const std::uint64_t rowWidth =
        1 + point->inputTagWidthBits + point->outputTagWidthBits;
    const std::uint64_t width =
        fabric.boundaryLookupTableSize(boundary) * rowWidth;
    std::vector<std::uint8_t> carrier = zeroBits(width);
    for (const auto &[row, rewrite] : llvm::enumerate(rewrites)) {
      const std::uint64_t base = row * rowWidth;
      setBit(carrier, base);
      for (std::uint64_t bit = 0; bit < point->inputTagWidthBits; ++bit)
        if (rewrite.inputTag[bit])
          setBit(carrier, base + 1 + bit);
      for (std::uint64_t bit = 0; bit < point->outputTagWidthBits; ++bit)
        if (rewrite.outputTag[bit])
          setBit(carrier, base + 1 + point->inputTagWidthBits + bit);
    }
    encoded = CanonicalSemanticBytes(std::move(carrier));
    break;
  }
  }

  if (llvm::Error error = relation->validateSemanticValue(encoded.bytes()))
    return std::move(error);
  return encoded;
}

} // namespace loom::fabric
