#include "DSE/ResolvedConfigView.h"

#include "Common/ArtifactLocalReference.h"
#include "Evaluation/Metric.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

constexpr char schemaDescriptor[] = "loom.dse.config.1.3";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "resolved_dse_config_invalid: " + message);
}

std::uint64_t signedBits(std::int64_t value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::int64_t signedValue(std::uint64_t bits) {
  std::int64_t value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

class Encoder final {
public:
  void u32(std::uint32_t value) {
    bytes_.push_back(static_cast<std::uint8_t>(value >> 24));
    bytes_.push_back(static_cast<std::uint8_t>(value >> 16));
    bytes_.push_back(static_cast<std::uint8_t>(value >> 8));
    bytes_.push_back(static_cast<std::uint8_t>(value));
  }

  void u64(std::uint64_t value) {
    for (unsigned shift = 56; shift != 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
    bytes_.push_back(static_cast<std::uint8_t>(value));
  }

  void i64(std::int64_t value) { u64(signedBits(value)); }

  void bytes(llvm::ArrayRef<std::uint8_t> value) {
    u64(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  void text(llvm::StringRef value) {
    bytes(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(value.data()), value.size()));
  }

  void digest(const ComponentViewDigest &value) {
    bytes_.insert(bytes_.end(), value.bytes().begin(), value.bytes().end());
  }

  void root(const ArtifactRootReference &value) {
    text(value.schemaIdentity);
    u32(value.schemaVersion.major);
    u32(value.schemaVersion.minor);
    bytes_.insert(bytes_.end(), value.artifact.bytes().begin(),
                  value.artifact.bytes().end());
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    if (remaining() < 4)
      return invalid("truncated u32 field");
    std::uint32_t value = 0;
    for (unsigned index = 0; index != 4; ++index)
      value = (value << 8) | bytes_[cursor_++];
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    if (remaining() < 8)
      return invalid("truncated u64 field");
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
      value = (value << 8) | bytes_[cursor_++];
    return value;
  }

  llvm::Expected<std::int64_t> i64() {
    auto bits = u64();
    if (!bits)
      return bits.takeError();
    return signedValue(*bits);
  }

  llvm::Expected<std::size_t> count(std::size_t minimumRecordBytes) {
    auto value = u64();
    if (!value)
      return value.takeError();
    if (*value > std::numeric_limits<std::size_t>::max())
      return invalid("count is not representable on this host");
    if (minimumRecordBytes != 0 && *value > remaining() / minimumRecordBytes)
      return invalid("count exceeds the remaining canonical bytes");
    return static_cast<std::size_t>(*value);
  }

  llvm::Expected<std::vector<std::uint8_t>> bytes() {
    auto size = count(1);
    if (!size)
      return size.takeError();
    std::vector<std::uint8_t> value(bytes_.begin() + cursor_,
                                    bytes_.begin() + cursor_ + *size);
    cursor_ += *size;
    return value;
  }

  llvm::Expected<std::string> text() {
    auto value = bytes();
    if (!value)
      return value.takeError();
    return std::string(reinterpret_cast<const char *>(value->data()),
                       value->size());
  }

  llvm::Expected<ComponentViewDigest> digest() {
    if (remaining() < ComponentViewDigest::byteSize)
      return invalid("truncated component view digest");
    auto value = ComponentViewDigest::fromBytes(
        bytes_.slice(cursor_, ComponentViewDigest::byteSize));
    cursor_ += ComponentViewDigest::byteSize;
    return value;
  }

  llvm::Expected<ArtifactRootReference> root() {
    auto identity = text();
    auto major = u32();
    auto minor = u32();
    if (!identity)
      return identity.takeError();
    if (!major)
      return major.takeError();
    if (!minor)
      return minor.takeError();
    if (remaining() < ArtifactIdentity::byteSize)
      return invalid("truncated Artifact identity");
    auto artifact = ArtifactIdentity::fromBytes(
        bytes_.slice(cursor_, ArtifactIdentity::byteSize));
    cursor_ += ArtifactIdentity::byteSize;
    if (!artifact)
      return artifact.takeError();
    return ArtifactRootReference{
        std::move(*identity), {*major, *minor}, std::move(*artifact)};
  }

  std::size_t remaining() const { return bytes_.size() - cursor_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t cursor_ = 0;
};

bool authorizationLess(const ModelAuthorization &lhs,
                       const ModelAuthorization &rhs) {
  const SchemaVersion left = lhs.descriptor.schemaVersion();
  const SchemaVersion right = rhs.descriptor.schemaVersion();
  if (left.major != right.major)
    return left.major < right.major;
  if (left.minor != right.minor)
    return left.minor < right.minor;
  return lhs.descriptor.modelKind() < rhs.descriptor.modelKind();
}

bool authorizationEqual(const ModelAuthorization &lhs,
                        const ModelAuthorization &rhs) {
  return lhs.descriptor == rhs.descriptor;
}

void encodeScalar(Encoder &encoder, const ResolvedObjectiveScalar &scalar) {
  if (const auto *integer = std::get_if<ResolvedObjectiveInteger>(&scalar)) {
    encoder.u32(0);
    encoder.u32(integer->negative ? 1 : 0);
    encoder.u64(integer->magnitude);
    return;
  }
  const auto &decimal = std::get<ResolvedObjectiveDecimal>(scalar);
  encoder.u32(1);
  encoder.i64(decimal.coefficient);
  encoder.i64(decimal.base10Exponent);
}

llvm::Expected<ResolvedObjectiveScalar> decodeScalar(Decoder &decoder) {
  auto tag = decoder.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0) {
    auto negative = decoder.u32();
    auto magnitude = decoder.u64();
    if (!negative)
      return negative.takeError();
    if (!magnitude)
      return magnitude.takeError();
    if (*negative > 1)
      return invalid("objective integer has an invalid sign tag");
    return resolvedObjectiveInteger(*magnitude, *negative == 1);
  }
  if (*tag == 1) {
    auto coefficient = decoder.i64();
    auto exponent = decoder.i64();
    if (!coefficient)
      return coefficient.takeError();
    if (!exponent)
      return exponent.takeError();
    return resolvedObjectiveDecimal(*coefficient, *exponent);
  }
  return invalid("objective scalar has an unknown tag");
}

void encodeMetricValue(Encoder &encoder, const evaluation::MetricValue &value) {
  if (const auto *integer = std::get_if<evaluation::IntegerValue>(&value)) {
    encoder.u32(0);
    encoder.i64(integer->value());
    return;
  }
  const auto decimal = std::get<evaluation::DecimalValue>(value);
  encoder.u32(1);
  encoder.i64(decimal.coefficient());
  encoder.i64(decimal.base10Exponent());
}

llvm::Expected<evaluation::MetricValue> decodeMetricValue(Decoder &decoder) {
  auto tag = decoder.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0) {
    auto value = decoder.i64();
    if (!value)
      return value.takeError();
    return evaluation::MetricValue{evaluation::IntegerValue(*value)};
  }
  if (*tag == 1) {
    auto coefficient = decoder.i64();
    auto exponent = decoder.i64();
    if (!coefficient)
      return coefficient.takeError();
    if (!exponent)
      return exponent.takeError();
    auto value = evaluation::DecimalValue::get(*coefficient, *exponent);
    if (!value)
      return value.takeError();
    return evaluation::MetricValue{*value};
  }
  return invalid("metric threshold has an unknown value tag");
}

void encodeObjectiveCatalogs(Encoder &encoder,
                             const ResolvedObjectiveCatalogs &catalogs) {
  encoder.u64(catalogs.dimensions.size());
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    if (const auto *violation =
            std::get_if<ResolvedMappingViolationObjectiveSource>(
                &dimension.source)) {
      encoder.u32(0);
      encoder.u32(static_cast<std::uint32_t>(violation->kind));
    } else if (const auto *measure =
                   std::get_if<ResolvedMappingMeasureObjectiveSource>(
                       &dimension.source)) {
      encoder.u32(1);
      encoder.u32(measure->ordinal);
    } else {
      const auto &metric =
          std::get<ResolvedEvaluationMetricObjectiveSource>(dimension.source);
      encoder.u32(2);
      encoder.u32(metric.evidenceObligationTemplate);
      encoder.u64(metric.metricRequestOrdinal);
    }
    encoder.u32(static_cast<std::uint32_t>(dimension.direction));
    encodeScalar(encoder, dimension.origin);
    encodeScalar(encoder, dimension.quantum);
    encoder.u64(dimension.lowerIndex);
    encoder.u64(dimension.upperIndex);
  }

  encoder.u64(catalogs.weightedLevels.size());
  for (const ResolvedWeightedObjectiveLevel &level : catalogs.weightedLevels) {
    encoder.u64(level.terms.size());
    for (const ResolvedWeightedObjectiveTerm &term : level.terms) {
      encoder.u32(term.dimension);
      encoder.u64(term.weight);
    }
  }

  encoder.u64(catalogs.totalOrderings.size());
  for (const ResolvedTotalOrdering &ordering : catalogs.totalOrderings) {
    encoder.u64(ordering.weightedLevels.size());
    for (std::uint32_t level : ordering.weightedLevels)
      encoder.u32(level);
  }
}

llvm::Expected<ResolvedObjectiveCatalogs>
decodeObjectiveCatalogs(Decoder &decoder) {
  ResolvedObjectiveCatalogs catalogs;
  auto dimensionCount = decoder.count(28);
  if (!dimensionCount)
    return dimensionCount.takeError();
  catalogs.dimensions.reserve(*dimensionCount);
  for (std::size_t index = 0; index != *dimensionCount; ++index) {
    auto sourceTag = decoder.u32();
    auto sourceValue = decoder.u32();
    if (!sourceTag)
      return sourceTag.takeError();
    if (!sourceValue)
      return sourceValue.takeError();
    ResolvedObjectiveScalarSource source;
    if (*sourceTag == 0) {
      source = ResolvedMappingViolationObjectiveSource{
          static_cast<ResolvedPnrViolationKind>(*sourceValue)};
    } else if (*sourceTag == 1) {
      source = ResolvedMappingMeasureObjectiveSource{*sourceValue};
    } else if (*sourceTag == 2) {
      auto request = decoder.u64();
      if (!request)
        return request.takeError();
      source = ResolvedEvaluationMetricObjectiveSource{*sourceValue, *request};
    } else {
      return invalid("objective source has an unknown tag");
    }
    auto direction = decoder.u32();
    auto origin = decodeScalar(decoder);
    auto quantum = decodeScalar(decoder);
    auto lower = decoder.u64();
    auto upper = decoder.u64();
    if (!direction)
      return direction.takeError();
    if (!origin)
      return origin.takeError();
    if (!quantum)
      return quantum.takeError();
    if (!lower)
      return lower.takeError();
    if (!upper)
      return upper.takeError();
    catalogs.dimensions.push_back(
        {std::move(source), static_cast<ResolvedObjectiveDirection>(*direction),
         std::move(*origin), std::move(*quantum), *lower, *upper});
  }

  auto levelCount = decoder.count(8);
  if (!levelCount)
    return levelCount.takeError();
  catalogs.weightedLevels.reserve(*levelCount);
  for (std::size_t index = 0; index != *levelCount; ++index) {
    auto termCount = decoder.count(12);
    if (!termCount)
      return termCount.takeError();
    ResolvedWeightedObjectiveLevel level;
    level.terms.reserve(*termCount);
    for (std::size_t term = 0; term != *termCount; ++term) {
      auto dimension = decoder.u32();
      auto weight = decoder.u64();
      if (!dimension)
        return dimension.takeError();
      if (!weight)
        return weight.takeError();
      level.terms.push_back({*dimension, *weight});
    }
    catalogs.weightedLevels.push_back(std::move(level));
  }

  auto orderingCount = decoder.count(8);
  if (!orderingCount)
    return orderingCount.takeError();
  catalogs.totalOrderings.reserve(*orderingCount);
  for (std::size_t index = 0; index != *orderingCount; ++index) {
    auto levelCountForOrdering = decoder.count(4);
    if (!levelCountForOrdering)
      return levelCountForOrdering.takeError();
    ResolvedTotalOrdering ordering;
    ordering.weightedLevels.reserve(*levelCountForOrdering);
    for (std::size_t level = 0; level != *levelCountForOrdering; ++level) {
      auto reference = decoder.u32();
      if (!reference)
        return reference.takeError();
      ordering.weightedLevels.push_back(*reference);
    }
    catalogs.totalOrderings.push_back(std::move(ordering));
  }
  return catalogs;
}

void encodeQualityGate(Encoder &encoder, const QualityGatePolicy &gate) {
  encoder.u64(gate.clauses().size());
  for (const QualityGateClause &clause : gate.clauses()) {
    encoder.u64(clause.atoms.size());
    for (const QualityGateAtom &atom : clause.atoms) {
      if (const auto *metric = std::get_if<MetricGate>(&atom)) {
        encoder.u32(0);
        encoder.u32(metric->evidenceObligationTemplate);
        encoder.u64(metric->metricRequest.ordinal());
        encoder.u32(static_cast<std::uint32_t>(metric->comparator));
        encodeMetricValue(encoder, metric->threshold);
      } else {
        const auto &finding = std::get<FindingGate>(atom);
        encoder.u32(1);
        encoder.u32(finding.evidenceObligationTemplate);
        encoder.u64(finding.findingRequest.ordinal());
        encoder.u32(static_cast<std::uint32_t>(finding.requiredState));
      }
    }
  }
}

llvm::Expected<QualityGatePolicy> decodeQualityGate(Decoder &decoder) {
  auto clauseCount = decoder.count(8);
  if (!clauseCount)
    return clauseCount.takeError();
  std::vector<QualityGateClause> clauses;
  clauses.reserve(*clauseCount);
  for (std::size_t clauseIndex = 0; clauseIndex != *clauseCount;
       ++clauseIndex) {
    auto atomCount = decoder.count(16);
    if (!atomCount)
      return atomCount.takeError();
    QualityGateClause clause;
    clause.atoms.reserve(*atomCount);
    for (std::size_t atomIndex = 0; atomIndex != *atomCount; ++atomIndex) {
      auto tag = decoder.u32();
      auto obligation = decoder.u32();
      auto request = decoder.u64();
      if (!tag)
        return tag.takeError();
      if (!obligation)
        return obligation.takeError();
      if (!request)
        return request.takeError();
      if (*tag == 0) {
        auto comparator = decoder.u32();
        if (!comparator)
          return comparator.takeError();
        auto threshold = decodeMetricValue(decoder);
        if (!threshold)
          return threshold.takeError();
        clause.atoms.push_back(
            MetricGate{*obligation, evaluation::MetricRequestOrdinal(*request),
                       static_cast<MetricGateComparator>(*comparator),
                       std::move(*threshold)});
      } else if (*tag == 1) {
        auto state = decoder.u32();
        if (!state)
          return state.takeError();
        clause.atoms.push_back(FindingGate{
            *obligation, evaluation::FindingRequestOrdinal(*request),
            static_cast<RequiredFindingState>(*state)});
      } else {
        return invalid("quality gate atom has an unknown tag");
      }
    }
    clauses.push_back(std::move(clause));
  }
  return QualityGatePolicy::get(std::move(clauses));
}

void encodePlanInput(Encoder &encoder, const PlanInputBinding &input) {
  if (const auto *exact = std::get_if<ExactPlanArtifacts>(&input)) {
    encoder.u32(0);
    encoder.u64(exact->artifacts.size());
    for (const ArtifactRootReference &artifact : exact->artifacts)
      encoder.root(artifact);
    return;
  }
  if (const auto *output = std::get_if<PlanOutputRef>(&input)) {
    encoder.u32(1);
    encoder.u64(output->producerNodeOrdinal);
    encoder.u32(output->outputSlotOrdinal);
    return;
  }
  const auto &join = std::get<BoundedPlanOutputJoin>(input);
  const bool hasDistinctProducerBound =
      join.maximumProducerArtifacts != 0 &&
      join.maximumProducerArtifacts != join.maximumArtifacts;
  const bool hasExactArtifacts = !join.exactArtifacts.empty();
  encoder.u32(hasExactArtifacts ? (hasDistinctProducerBound ? 5 : 4)
                                : (hasDistinctProducerBound ? 3 : 2));
  encoder.u64(join.outputs.size());
  for (PlanOutputRef output : join.outputs) {
    encoder.u64(output.producerNodeOrdinal);
    encoder.u32(output.outputSlotOrdinal);
  }
  encoder.u64(join.maximumArtifacts);
  if (hasDistinctProducerBound)
    encoder.u64(join.maximumProducerArtifacts);
  if (hasExactArtifacts) {
    encoder.u64(join.exactArtifacts.size());
    for (const ArtifactRootReference &artifact : join.exactArtifacts)
      encoder.root(artifact);
  }
}

llvm::Expected<PlanInputBinding> decodePlanInput(Decoder &decoder) {
  auto tag = decoder.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0) {
    auto count = decoder.count(48);
    if (!count)
      return count.takeError();
    ExactPlanArtifacts exact;
    exact.artifacts.reserve(*count);
    for (std::size_t index = 0; index != *count; ++index) {
      auto artifact = decoder.root();
      if (!artifact)
        return artifact.takeError();
      exact.artifacts.push_back(std::move(*artifact));
    }
    return PlanInputBinding{std::move(exact)};
  }
  if (*tag == 1) {
    auto producer = decoder.u64();
    auto slot = decoder.u32();
    if (!producer)
      return producer.takeError();
    if (!slot)
      return slot.takeError();
    return PlanInputBinding{PlanOutputRef{*producer, *slot}};
  }
  if (*tag >= 2 && *tag <= 5) {
    auto count = decoder.count(12);
    if (!count)
      return count.takeError();
    BoundedPlanOutputJoin join;
    join.outputs.reserve(*count);
    for (std::size_t index = 0; index != *count; ++index) {
      auto producer = decoder.u64();
      auto slot = decoder.u32();
      if (!producer)
        return producer.takeError();
      if (!slot)
        return slot.takeError();
      join.outputs.push_back(PlanOutputRef{*producer, *slot});
    }
    auto maximum = decoder.u64();
    if (!maximum)
      return maximum.takeError();
    join.maximumArtifacts = *maximum;
    if (*tag == 3 || *tag == 5) {
      auto producerMaximum = decoder.u64();
      if (!producerMaximum)
        return producerMaximum.takeError();
      join.maximumProducerArtifacts = *producerMaximum;
    }
    if (*tag == 4 || *tag == 5) {
      auto exactCount = decoder.count(48);
      if (!exactCount)
        return exactCount.takeError();
      join.exactArtifacts.reserve(*exactCount);
      for (std::size_t index = 0; index != *exactCount; ++index) {
        auto artifact = decoder.root();
        if (!artifact)
          return artifact.takeError();
        join.exactArtifacts.push_back(std::move(*artifact));
      }
    }
    return PlanInputBinding{std::move(join)};
  }
  return invalid("plan input has an unknown tag");
}

void encodeSelection(Encoder &encoder,
                     const CandidateSelectionPolicy &selection) {
  if (std::holds_alternative<AllPassingSelection>(selection)) {
    encoder.u32(0);
    return;
  }
  if (const auto *topK = std::get_if<TopKSelection>(&selection)) {
    encoder.u32(1);
    encoder.u32(topK->totalOrdering);
    encoder.u64(topK->k);
    return;
  }
  encoder.u32(2);
  const auto &dimensions =
      std::get<ParetoSelection>(selection).objectiveDimensions;
  encoder.u64(dimensions.size());
  for (std::uint32_t dimension : dimensions)
    encoder.u32(dimension);
}

llvm::Expected<CandidateSelectionPolicy> decodeSelection(Decoder &decoder) {
  auto tag = decoder.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0)
    return CandidateSelectionPolicy{AllPassingSelection{}};
  if (*tag == 1) {
    auto ordering = decoder.u32();
    auto k = decoder.u64();
    if (!ordering)
      return ordering.takeError();
    if (!k)
      return k.takeError();
    return CandidateSelectionPolicy{TopKSelection{*ordering, *k}};
  }
  if (*tag == 2) {
    auto count = decoder.count(4);
    if (!count)
      return count.takeError();
    std::vector<std::uint32_t> dimensions;
    dimensions.reserve(*count);
    for (std::size_t index = 0; index != *count; ++index) {
      auto dimension = decoder.u32();
      if (!dimension)
        return dimension.takeError();
      dimensions.push_back(*dimension);
    }
    return CandidateSelectionPolicy{ParetoSelection{std::move(dimensions)}};
  }
  return invalid("candidate selection has an unknown tag");
}

void encodePlanNodes(Encoder &encoder,
                     llvm::ArrayRef<DsePlanNodeDefinition> nodes) {
  encoder.u64(nodes.size());
  for (const DsePlanNodeDefinition &node : nodes) {
    if (const auto *generate = std::get_if<GeneratePlanNodeDefinition>(&node)) {
      encoder.u32(0);
      encoder.u32(generate->descriptor.kind().ordinal());
      encoder.u64(generate->inputBindings.size());
      for (const PlanInputBinding &input : generate->inputBindings)
        encodePlanInput(encoder, input);
      encoder.bytes(generate->canonicalConfigBytes);
      encoder.digest(generate->configDigest);
      continue;
    }
    const auto &promote = std::get<PromotePlanNodeDefinition>(node);
    encoder.u32(1);
    encoder.u32(promote.acquisition.kind().ordinal());
    encoder.u64(promote.inputBindings.size());
    for (const PlanInputBinding &input : promote.inputBindings)
      encodePlanInput(encoder, input);
    encoder.bytes(promote.canonicalConfigBytes);
    encoder.digest(promote.configDigest);
    encoder.u32(promote.qualityGate.ordinal());
    encodeSelection(encoder, promote.selection);
    encoder.u32(static_cast<std::uint32_t>(promote.purpose));
  }
}

llvm::Expected<std::vector<DsePlanNodeDefinition>>
decodePlanNodes(Decoder &decoder) {
  auto nodeCount = decoder.count(48);
  if (!nodeCount)
    return nodeCount.takeError();
  std::vector<DsePlanNodeDefinition> nodes;
  nodes.reserve(*nodeCount);
  for (std::size_t nodeIndex = 0; nodeIndex != *nodeCount; ++nodeIndex) {
    auto tag = decoder.u32();
    auto kind = decoder.u32();
    auto inputCount = decoder.count(4);
    if (!tag)
      return tag.takeError();
    if (!kind)
      return kind.takeError();
    if (!inputCount)
      return inputCount.takeError();
    std::vector<PlanInputBinding> inputs;
    inputs.reserve(*inputCount);
    for (std::size_t index = 0; index != *inputCount; ++index) {
      auto input = decodePlanInput(decoder);
      if (!input)
        return input.takeError();
      inputs.push_back(std::move(*input));
    }
    auto config = decoder.bytes();
    auto digest = decoder.digest();
    if (!config)
      return config.takeError();
    if (!digest)
      return digest.takeError();
    if (*tag == 0) {
      auto descriptor = CandidateGeneratorDescriptorRef::get(
          candidateGeneratorDescriptorSchema, CandidateGeneratorKind(*kind));
      if (!descriptor)
        return descriptor.takeError();
      nodes.push_back(GeneratePlanNodeDefinition{*descriptor, std::move(inputs),
                                                 std::move(*config), *digest});
      continue;
    }
    if (*tag != 1)
      return invalid("plan node has an unknown tag");
    auto acquisition = PromotionAcquisitionDescriptorRef::get(
        PromotionAcquisitionDescriptor::schema,
        PromotionAcquisitionKind(*kind));
    if (!acquisition)
      return acquisition.takeError();
    auto gate = decoder.u32();
    if (!gate)
      return gate.takeError();
    auto selection = decodeSelection(decoder);
    if (!selection)
      return selection.takeError();
    auto purpose = decoder.u32();
    if (!purpose)
      return purpose.takeError();
    nodes.push_back(PromotePlanNodeDefinition{
        *acquisition, std::move(inputs), std::move(*config), *digest,
        QualityGatePolicyRef(*gate), std::move(*selection),
        static_cast<PromotePurpose>(*purpose)});
  }
  return nodes;
}

struct ViewParts final {
  std::vector<ModelAuthorization> modelAuthorizations;
  std::vector<EvidenceObligationTemplate> templates;
  ResolvedObjectiveCatalogs objectives;
  std::vector<QualityGatePolicy> gates;
  std::vector<DsePlanNodeDefinition> planNodes;
};

llvm::Expected<std::vector<std::uint8_t>> encodeParts(const ViewParts &parts) {
  Encoder encoder;
  encoder.u64(parts.modelAuthorizations.size());
  for (const ModelAuthorization &authorization : parts.modelAuthorizations) {
    encoder.u32(authorization.descriptor.schemaVersion().major);
    encoder.u32(authorization.descriptor.schemaVersion().minor);
    encoder.u32(authorization.descriptor.modelKind().ordinal());
  }
  encoder.u64(parts.templates.size());
  for (const EvidenceObligationTemplate &obligation : parts.templates)
    encoder.bytes(obligation.canonicalBytes());
  encodeObjectiveCatalogs(encoder, parts.objectives);
  encoder.u64(parts.gates.size());
  for (const QualityGatePolicy &gate : parts.gates)
    encodeQualityGate(encoder, gate);
  encodePlanNodes(encoder, parts.planNodes);
  return encoder.take();
}

llvm::Expected<ViewParts> decodeParts(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  ViewParts parts;
  auto authorizationCount = decoder.count(12);
  if (!authorizationCount)
    return authorizationCount.takeError();
  parts.modelAuthorizations.reserve(*authorizationCount);
  for (std::size_t index = 0; index != *authorizationCount; ++index) {
    auto major = decoder.u32();
    auto minor = decoder.u32();
    auto kind = decoder.u32();
    if (!major)
      return major.takeError();
    if (!minor)
      return minor.takeError();
    if (!kind)
      return kind.takeError();
    auto descriptor = evaluation::EvaluationModelDescriptorRef::get(
        {*major, *minor}, evaluation::EvaluationModelKind(*kind));
    if (!descriptor)
      return descriptor.takeError();
    parts.modelAuthorizations.push_back({*descriptor});
  }
  auto templateCount = decoder.count(8);
  if (!templateCount)
    return templateCount.takeError();
  parts.templates.reserve(*templateCount);
  for (std::size_t index = 0; index != *templateCount; ++index) {
    auto templateBytes = decoder.bytes();
    if (!templateBytes)
      return templateBytes.takeError();
    auto obligation = adoptEvidenceObligationTemplate(*templateBytes);
    if (!obligation)
      return obligation.takeError();
    parts.templates.push_back(std::move(*obligation));
  }
  auto objectives = decodeObjectiveCatalogs(decoder);
  if (!objectives)
    return objectives.takeError();
  parts.objectives = std::move(*objectives);
  auto gateCount = decoder.count(8);
  if (!gateCount)
    return gateCount.takeError();
  parts.gates.reserve(*gateCount);
  for (std::size_t index = 0; index != *gateCount; ++index) {
    auto gate = decodeQualityGate(decoder);
    if (!gate)
      return gate.takeError();
    parts.gates.push_back(std::move(*gate));
  }
  auto plan = decodePlanNodes(decoder);
  if (!plan)
    return plan.takeError();
  parts.planNodes = std::move(*plan);
  if (decoder.remaining() != 0)
    return invalid("component view has trailing bytes");
  return parts;
}

bool isAuthorized(llvm::ArrayRef<ModelAuthorization> authorizations,
                  evaluation::EvaluationModelDescriptorRef descriptor) {
  return std::binary_search(authorizations.begin(), authorizations.end(),
                            ModelAuthorization{descriptor}, authorizationLess);
}

llvm::Error validateCanonicalInputs(const ViewParts &parts) {
  if (!llvm::is_sorted(parts.modelAuthorizations, authorizationLess) ||
      std::adjacent_find(parts.modelAuthorizations.begin(),
                         parts.modelAuthorizations.end(),
                         authorizationEqual) != parts.modelAuthorizations.end())
    return invalid("model authorizations are not canonical and unique");
  for (const ModelAuthorization &authorization : parts.modelAuthorizations)
    if (!authorization.descriptor.descriptor())
      return invalid("model authorization references an unregistered model");

  for (std::size_t index = 0; index < parts.templates.size(); ++index) {
    const EvidenceObligationTemplate &obligation = parts.templates[index];
    if (index != 0 && !std::lexicographical_compare(
                          parts.templates[index - 1].canonicalBytes().begin(),
                          parts.templates[index - 1].canonicalBytes().end(),
                          obligation.canonicalBytes().begin(),
                          obligation.canonicalBytes().end()))
      return invalid("evidence obligation templates are not canonical and "
                     "unique");
    if (!isAuthorized(parts.modelAuthorizations,
                      obligation.modelBinding().descriptorRef()))
      return invalid("evidence obligation uses an unauthorized model");
  }

  if (llvm::Error error = validateResolvedObjectiveCatalogs(parts.objectives))
    return error;
  for (const ResolvedObjectiveDimension &dimension :
       parts.objectives.dimensions) {
    const auto *metric =
        std::get_if<ResolvedEvaluationMetricObjectiveSource>(&dimension.source);
    if (!metric)
      continue;
    if (metric->evidenceObligationTemplate >= parts.templates.size())
      return invalid("objective references a foreign evidence obligation");
    const EvidenceObligationTemplate &obligation =
        parts.templates[metric->evidenceObligationTemplate];
    if (obligation.calibrationPartitionRole() ==
        CalibrationPartitionRole::HeldOut)
      return invalid("held-out obligation cannot feed an objective");
    if (metric->metricRequestOrdinal >= obligation.metricRequests().size())
      return invalid("objective metric request ordinal is out of range");
  }

  std::vector<std::vector<std::uint8_t>> gateKeys;
  gateKeys.reserve(parts.gates.size());
  for (const QualityGatePolicy &gate : parts.gates) {
    Encoder encoder;
    encodeQualityGate(encoder, gate);
    gateKeys.push_back(encoder.take());
    for (const QualityGateClause &clause : gate.clauses()) {
      for (const QualityGateAtom &atom : clause.atoms) {
        if (const auto *metric = std::get_if<MetricGate>(&atom)) {
          if (metric->evidenceObligationTemplate >= parts.templates.size())
            return invalid("quality gate references a foreign obligation");
          const EvidenceObligationTemplate &obligation =
              parts.templates[metric->evidenceObligationTemplate];
          if (metric->metricRequest.ordinal() >=
              obligation.metricRequests().size())
            return invalid("quality gate metric request is out of range");
          const evaluation::MetricKind kind =
              obligation.metricRequests()[metric->metricRequest.ordinal()]
                  .query.metric;
          if (llvm::Error error = evaluation::validateMetricObservationValue(
                  kind, evaluation::UncertaintyKind::ExactWithinModel,
                  evaluation::PointObservation{metric->threshold}))
            return error;
        } else {
          const auto &finding = std::get<FindingGate>(atom);
          if (finding.evidenceObligationTemplate >= parts.templates.size())
            return invalid("quality gate references a foreign obligation");
          const EvidenceObligationTemplate &obligation =
              parts.templates[finding.evidenceObligationTemplate];
          if (finding.findingRequest.ordinal() >=
              obligation.findingRequests().size())
            return invalid("quality gate finding request is out of range");
        }
      }
    }
  }
  if (!llvm::is_sorted(gateKeys) ||
      std::adjacent_find(gateKeys.begin(), gateKeys.end()) != gateKeys.end())
    return invalid("quality gate policies are not canonical and unique");

  for (const DsePlanNodeDefinition &node : parts.planNodes) {
    llvm::ArrayRef<PlanInputBinding> inputs;
    if (const auto *generate = std::get_if<GeneratePlanNodeDefinition>(&node)) {
      inputs = generate->inputBindings;
    } else {
      const auto &promote = std::get<PromotePlanNodeDefinition>(node);
      inputs = promote.inputBindings;
      if (promote.qualityGate.ordinal() >= parts.gates.size())
        return invalid("Promote quality gate reference is out of range");
    }
    for (const PlanInputBinding &input : inputs) {
      if (const auto *exact = std::get_if<ExactPlanArtifacts>(&input)) {
        if (!llvm::is_sorted(exact->artifacts, artifactRootReferenceLess) ||
            std::adjacent_find(exact->artifacts.begin(),
                               exact->artifacts.end()) !=
                exact->artifacts.end())
          return invalid("exact plan artifacts are not canonical and unique");
        continue;
      }
      const auto *join = std::get_if<BoundedPlanOutputJoin>(&input);
      if (!join)
        continue;
      if (join->maximumArtifacts == 0 ||
          (join->outputs.empty() && join->exactArtifacts.empty()) ||
          join->producerArtifactLimit() < join->maximumArtifacts ||
          !llvm::is_sorted(join->outputs) ||
          std::adjacent_find(join->outputs.begin(), join->outputs.end()) !=
              join->outputs.end() ||
          !llvm::is_sorted(join->exactArtifacts, artifactRootReferenceLess) ||
          std::adjacent_find(join->exactArtifacts.begin(),
                             join->exactArtifacts.end()) !=
              join->exactArtifacts.end())
        return invalid("bounded output join is not canonical and bounded");
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ResolvedDseConfigView> ResolvedDseConfigView::get(
    std::vector<ModelAuthorization> modelAuthorizations,
    std::vector<EvidenceObligationTemplate> evidenceObligationTemplates,
    ResolvedObjectiveCatalogs objectiveCatalogs,
    std::vector<QualityGatePolicy> qualityGatePolicies,
    std::vector<DsePlanNodeDefinition> planNodes) {
  ViewParts parts{std::move(modelAuthorizations),
                  std::move(evidenceObligationTemplates),
                  std::move(objectiveCatalogs), std::move(qualityGatePolicies),
                  std::move(planNodes)};
  if (llvm::Error error = validateCanonicalInputs(parts))
    return std::move(error);
  auto bytes = encodeParts(parts);
  if (!bytes)
    return bytes.takeError();
  auto plan = ResolvedDsePlan::get(parts.planNodes, parts.templates,
                                   parts.objectives, parts.gates);
  if (!plan)
    return plan.takeError();
  const llvm::ArrayRef<std::uint8_t> descriptor(
      reinterpret_cast<const std::uint8_t *>(schemaDescriptor),
      sizeof(schemaDescriptor) - 1);
  auto digest = computeComponentViewDigest(descriptor, *bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedDseConfigView(std::move(parts.modelAuthorizations),
                               std::move(parts.templates),
                               std::move(parts.objectives), std::move(*plan),
                               std::move(*bytes), *digest);
}

llvm::ArrayRef<std::uint8_t>
ResolvedDseConfigView::schemaDescriptorBytes() const {
  return llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(schemaDescriptor),
      sizeof(schemaDescriptor) - 1);
}

std::vector<std::uint8_t>
canonicalQualityGatePolicyBytes(const QualityGatePolicy &policy) {
  Encoder encoder;
  encodeQualityGate(encoder, policy);
  return encoder.take();
}

llvm::Expected<QualityGatePolicy>
adoptQualityGatePolicy(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto policy = decodeQualityGate(decoder);
  if (!policy)
    return policy.takeError();
  if (decoder.remaining() != 0)
    return invalid("quality gate policy has trailing bytes");
  const std::vector<std::uint8_t> canonical =
      canonicalQualityGatePolicyBytes(*policy);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("quality gate policy bytes are not canonical");
  return policy;
}

std::vector<std::uint8_t>
canonicalDsePlanNodeBytes(const DsePlanNodeDefinition &node) {
  Encoder encoder;
  encodePlanNodes(encoder, llvm::ArrayRef<DsePlanNodeDefinition>(&node, 1));
  return encoder.take();
}

llvm::Expected<DsePlanNodeDefinition>
adoptDsePlanNode(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto nodes = decodePlanNodes(decoder);
  if (!nodes)
    return nodes.takeError();
  if (nodes->size() != 1)
    return invalid("plan-node payload must contain exactly one node");
  if (decoder.remaining() != 0)
    return invalid("plan-node payload has trailing bytes");
  const std::vector<std::uint8_t> canonical =
      canonicalDsePlanNodeBytes(nodes->front());
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("plan-node bytes are not canonical");
  return std::move(nodes->front());
}

llvm::Expected<ResolvedDseConfigView>
adoptResolvedDseConfigView(llvm::ArrayRef<std::uint8_t> suppliedDescriptor,
                           llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
                           const ComponentViewDigest &digest) {
  const llvm::ArrayRef<std::uint8_t> expectedDescriptor(
      reinterpret_cast<const std::uint8_t *>(schemaDescriptor),
      sizeof(schemaDescriptor) - 1);
  if (suppliedDescriptor != expectedDescriptor)
    return invalid("component view schema descriptor mismatch");
  if (llvm::Error error = validateComponentViewDigest(
          suppliedDescriptor, canonicalViewBytes, digest))
    return std::move(error);
  auto parts = decodeParts(canonicalViewBytes);
  if (!parts)
    return parts.takeError();
  auto adopted = ResolvedDseConfigView::get(
      std::move(parts->modelAuthorizations), std::move(parts->templates),
      std::move(parts->objectives), std::move(parts->gates),
      std::move(parts->planNodes));
  if (!adopted)
    return adopted.takeError();
  if (adopted->canonicalViewBytes() != canonicalViewBytes)
    return invalid("component view bytes are not canonical");
  return adopted;
}

} // namespace loom::dse
