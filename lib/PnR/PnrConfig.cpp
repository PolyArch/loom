#include "PnR/PnrConfig.h"

#include "DSE/Objective.h"

#include "Config/ResolvedConfig.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace loom::pnr {

struct ResolvedPnrConfigViewAccess final {
  static ResolvedPnrConfigView
  create(PnrConfigDomain domain, ResolvedPnrPolicyConfig policy,
         ResolvedObjectiveCatalogs selectedObjectiveCatalogs,
         std::vector<SystemBindingPartitionIntent> systemBindingPartitions,
         std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest) {
    return ResolvedPnrConfigView(
        domain, std::move(policy), std::move(selectedObjectiveCatalogs),
        std::move(systemBindingPartitions), std::move(canonicalBytes), digest);
  }
};

namespace {

constexpr llvm::StringLiteral spatialDescriptor =
    "loom.spatial_pnr.config.15.2";
constexpr llvm::StringLiteral systemDescriptor = "loom.system_pnr.config.8.1";

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "pnr_config_bytes_invalid: " + detail);
}

llvm::Error unsupported(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "pnr_config_unsupported: " + detail);
}

llvm::Error
validateObjectiveArithmetic(const ResolvedObjectiveCatalogs &catalogs) {
  auto program = dse::ObjectiveProgram::get(catalogs);
  if (!program)
    return program.takeError();
  return llvm::Error::success();
}

llvm::Error
validateDomainCapabilities(PnrConfigDomain domain,
                           const ResolvedPnrPolicyConfig &policy,
                           const ResolvedObjectiveCatalogs &catalogs) {
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions)
    if (std::holds_alternative<ResolvedEvaluationMetricObjectiveSource>(
            dimension.source))
      return unsupported(
          "PnR objective selection requires an unavailable Evaluation owner");
  if (domain == PnrConfigDomain::Spatial &&
      !std::holds_alternative<ResolvedPathFinderPolicy>(
          policy.search.routing.negotiation))
    return unsupported("Spatial PnR supports only PathFinder negotiation");
  if (domain == PnrConfigDomain::System &&
      policy.search.exactRepair.kind != ResolvedPnrExactRepairKind::Disabled)
    return unsupported("System PnR has no exact-repair provider");
  return llvm::Error::success();
}

llvm::ArrayRef<std::uint8_t> descriptorBytes(PnrConfigDomain domain) {
  const llvm::StringRef descriptor =
      domain == PnrConfigDomain::Spatial ? spatialDescriptor : systemDescriptor;
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

bool partitionIntentLess(const SystemBindingPartitionIntent &lhs,
                         const SystemBindingPartitionIntent &rhs) {
  if (lhs.root.artifact != rhs.root.artifact)
    return lhs.root.artifact.bytes() < rhs.root.artifact.bytes();
  return lhs.root.entity.value() < rhs.root.entity.value();
}

llvm::Expected<std::vector<SystemBindingPartitionIntent>>
canonicalPartitionIntent(
    PnrConfigDomain domain,
    llvm::ArrayRef<SystemBindingPartitionIntent> partitions) {
  if (domain == PnrConfigDomain::Spatial && !partitions.empty())
    return invalid("Spatial PnR cannot carry a System partition intent");
  std::vector<SystemBindingPartitionIntent> canonical(partitions.begin(),
                                                      partitions.end());
  llvm::sort(canonical, partitionIntentLess);
  for (const auto &partition : canonical)
    if (partition.partitionCount == 0)
      return invalid("System partition count is zero");
  for (std::size_t index = 1; index < canonical.size(); ++index)
    if (canonical[index - 1].root == canonical[index].root)
      return invalid("System partition intent repeats a Dataflow root");
  return canonical;
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

  void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }

  void bytes(llvm::ArrayRef<std::uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  void ratio(const ResolvedExactRatio &value) {
    u64(value.numerator);
    u64(value.denominator);
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
    for (unsigned ordinal = 0; ordinal != 4; ++ordinal)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    if (remaining() < 8)
      return invalid("truncated u64 field");
    std::uint64_t value = 0;
    for (unsigned ordinal = 0; ordinal != 8; ++ordinal)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::int64_t> i64() {
    auto value = u64();
    if (!value)
      return value.takeError();
    return static_cast<std::int64_t>(*value);
  }

  llvm::Expected<std::size_t> count() {
    auto countOrErr = u64();
    if (!countOrErr)
      return countOrErr.takeError();
    if (*countOrErr > remaining())
      return invalid("sequence count exceeds remaining bytes");
    return static_cast<std::size_t>(*countOrErr);
  }

  llvm::Expected<std::vector<std::uint8_t>> bytes(std::size_t count) {
    if (count > remaining())
      return invalid("truncated byte sequence");
    std::vector<std::uint8_t> result(bytes_.begin() + offset_,
                                     bytes_.begin() + offset_ + count);
    offset_ += count;
    return result;
  }

  llvm::Expected<ResolvedExactRatio> ratio() {
    auto numerator = u64();
    auto denominator = u64();
    if (!numerator)
      return numerator.takeError();
    if (!denominator)
      return denominator.takeError();
    return ResolvedExactRatio{*numerator, *denominator};
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

void encodeNegotiation(Encoder &encoder,
                       const ResolvedRoutingNegotiationPolicy &policy) {
  if (const auto *pathFinder = std::get_if<ResolvedPathFinderPolicy>(&policy)) {
    encoder.u32(0);
    encoder.u32(static_cast<std::uint32_t>(pathFinder->priceKernel));
    encoder.u64(pathFinder->presentPressureInitial);
    encoder.ratio(pathFinder->presentPressureGrowth);
    encoder.u64(pathFinder->historyPressureIncrement);
    return;
  }
  const auto &dual = std::get<ResolvedDualSubgradientPolicy>(policy);
  encoder.u32(1);
  encoder.u32(static_cast<std::uint32_t>(dual.directionKernel));
  if (dual.directionKernel == ResolvedDualDirectionKernel::MomentumDeflected)
    encoder.ratio(*dual.momentum);
  encoder.u32(static_cast<std::uint32_t>(dual.stepSchedule.kind));
  switch (dual.stepSchedule.kind) {
  case ResolvedDualStepScheduleKind::Constant:
    encoder.u64(dual.stepSchedule.first);
    break;
  case ResolvedDualStepScheduleKind::GeometricDecay:
    encoder.u64(dual.stepSchedule.first);
    encoder.u64(dual.stepSchedule.second);
    encoder.ratio({dual.stepSchedule.third, dual.stepSchedule.fourth});
    break;
  case ResolvedDualStepScheduleKind::HarmonicDecay:
    encoder.u64(dual.stepSchedule.first);
    encoder.u64(dual.stepSchedule.second);
    encoder.u64(dual.stepSchedule.third);
    break;
  }
}

void encodePolicy(Encoder &encoder, const ResolvedPnrPolicyConfig &policy) {
  const ResolvedPnrSearchPolicy &search = policy.search;
  encoder.u32(search.initializer.seedAttemptCount);
  encoder.u64(search.initializer.assignmentAttemptLimitPerSeed);
  encoder.u64(search.actionProposal.realizationBindingWeight);
  encoder.u64(search.actionProposal.transportRoutingWeight);
  encoder.u64(search.actionProposal.resourceAllocationWeight);
  encoder.u64(search.routing.endpointExpansionLimit);
  encoder.u64(search.routing.negotiationIterationLimit);
  encoder.u64(search.routing.noProgressIterationLimit);
  encoder.u64(search.routing.noProgressTrendWindow);
  encodeNegotiation(encoder, search.routing.negotiation);
  encoder.u64(search.annealing.calibrationProposalCount);
  encoder.ratio(search.annealing.positiveDeltaQuantile);
  encoder.ratio(search.annealing.targetInitialAcceptance);
  encoder.u64(search.annealing.fallbackTemperature);
  encoder.u64(search.annealing.minimumTemperature);
  encoder.ratio(search.annealing.coolingRatio);
  encoder.u64(search.annealing.temperatureLevelLimit);
  encoder.u64(search.annealing.proposalsPerLevelBase);
  encoder.u64(search.annealing.proposalsPerMovableDecision);
  encoder.u32(static_cast<std::uint32_t>(search.exactRepair.kind));
  if (search.exactRepair.kind == ResolvedPnrExactRepairKind::CpSat) {
    encoder.u64(search.exactRepair.maxRegionDecisions);
    encoder.u64(search.exactRepair.maxSolverCalls);
  }
  encoder.u32(static_cast<std::uint32_t>(search.completionGoal));
  encoder.u64(policy.determinism.masterSeed);
  encoder.u32(static_cast<std::uint32_t>(policy.determinism.prngProtocol));
  encoder.u32(
      static_cast<std::uint32_t>(policy.determinism.acceptanceProtocol));
  encoder.u64(policy.temporaryViolations.admitted.size());
  for (ResolvedPnrViolationKind violation : policy.temporaryViolations.admitted)
    encoder.u32(static_cast<std::uint32_t>(violation));
}

void encodeClosure(Encoder &encoder, const ResolvedPnrPolicyConfig &policy,
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
    } else
      llvm_unreachable("PnR config cannot encode an Evaluation metric");
    encoder.u32(static_cast<std::uint32_t>(dimension.direction));
    const auto encodeScalar = [&](const ResolvedObjectiveScalar &value) {
      if (const auto *integer = std::get_if<ResolvedObjectiveInteger>(&value)) {
        encoder.u32(0);
        encoder.u32(integer->negative ? 1 : 0);
        encoder.u64(integer->magnitude);
      } else {
        const auto &decimal = std::get<ResolvedObjectiveDecimal>(value);
        encoder.u32(1);
        encoder.i64(decimal.coefficient);
        encoder.i64(decimal.base10Exponent);
      }
    };
    encodeScalar(dimension.origin);
    encodeScalar(dimension.quantum);
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
  encoder.u32(policy.objectiveSelection.selectedTotalOrdering);
  encoder.u32(policy.objectiveSelection.selectedSearchEnergy);
}

std::vector<std::uint8_t>
encodeView(PnrConfigDomain domain, const ResolvedPnrPolicyConfig &policy,
           const ResolvedObjectiveCatalogs &catalogs,
           llvm::ArrayRef<SystemBindingPartitionIntent> partitions) {
  Encoder encoder;
  encodePolicy(encoder, policy);
  encodeClosure(encoder, policy, catalogs);
  if (domain == PnrConfigDomain::System) {
    encoder.u64(partitions.size());
    for (const SystemBindingPartitionIntent &partition : partitions) {
      encoder.bytes(partition.root.artifact.bytes());
      encoder.u64(partition.root.entity.value());
      encoder.u64(partition.partitionCount);
    }
  }
  return encoder.take();
}

llvm::Expected<ResolvedRoutingNegotiationPolicy>
decodeNegotiation(Decoder &decoder) {
  auto kind = decoder.u32();
  if (!kind)
    return kind.takeError();
  if (*kind == 0) {
    auto kernel = decoder.u32();
    auto initial = decoder.u64();
    auto growth = decoder.ratio();
    auto history = decoder.u64();
    if (!kernel)
      return kernel.takeError();
    if (*kernel >
        static_cast<std::uint32_t>(ResolvedPathFinderPriceKernel::Additive))
      return invalid("unknown PathFinder price kernel");
    if (!initial)
      return initial.takeError();
    if (!growth)
      return growth.takeError();
    if (!history)
      return history.takeError();
    return ResolvedRoutingNegotiationPolicy{ResolvedPathFinderPolicy{
        static_cast<ResolvedPathFinderPriceKernel>(*kernel), *initial, *growth,
        *history}};
  }
  if (*kind != 1)
    return invalid("unknown routing negotiation union tag");

  auto directionTag = decoder.u32();
  if (!directionTag)
    return directionTag.takeError();
  if (*directionTag > static_cast<std::uint32_t>(
                          ResolvedDualDirectionKernel::MomentumDeflected))
    return invalid("unknown dual direction kernel");
  const auto direction =
      static_cast<ResolvedDualDirectionKernel>(*directionTag);
  std::optional<ResolvedExactRatio> momentum;
  if (direction == ResolvedDualDirectionKernel::MomentumDeflected) {
    auto momentumOrErr = decoder.ratio();
    if (!momentumOrErr)
      return momentumOrErr.takeError();
    momentum = *momentumOrErr;
  }

  auto scheduleTag = decoder.u32();
  if (!scheduleTag)
    return scheduleTag.takeError();
  if (*scheduleTag >
      static_cast<std::uint32_t>(ResolvedDualStepScheduleKind::HarmonicDecay))
    return invalid("unknown dual step schedule");
  ResolvedDualStepSchedule schedule{};
  schedule.kind = static_cast<ResolvedDualStepScheduleKind>(*scheduleTag);
  auto first = decoder.u64();
  if (!first)
    return first.takeError();
  schedule.first = *first;
  if (schedule.kind == ResolvedDualStepScheduleKind::GeometricDecay) {
    auto second = decoder.u64();
    auto ratio = decoder.ratio();
    if (!second)
      return second.takeError();
    if (!ratio)
      return ratio.takeError();
    schedule.second = *second;
    schedule.third = ratio->numerator;
    schedule.fourth = ratio->denominator;
  } else if (schedule.kind == ResolvedDualStepScheduleKind::HarmonicDecay) {
    auto second = decoder.u64();
    auto third = decoder.u64();
    if (!second)
      return second.takeError();
    if (!third)
      return third.takeError();
    schedule.second = *second;
    schedule.third = *third;
  }
  return ResolvedRoutingNegotiationPolicy{
      ResolvedDualSubgradientPolicy{direction, momentum, schedule}};
}

llvm::Expected<ResolvedPnrPolicyConfig> decodePolicy(Decoder &decoder) {
  auto seeds = decoder.u32();
  auto assignments = decoder.u64();
  auto realization = decoder.u64();
  auto transport = decoder.u64();
  auto resource = decoder.u64();
  auto endpointLimit = decoder.u64();
  auto negotiationLimit = decoder.u64();
  auto noProgressLimit = decoder.u64();
  auto noProgressTrendWindow = decoder.u64();
  if (!seeds)
    return seeds.takeError();
  if (!assignments)
    return assignments.takeError();
  if (!realization)
    return realization.takeError();
  if (!transport)
    return transport.takeError();
  if (!resource)
    return resource.takeError();
  if (!endpointLimit)
    return endpointLimit.takeError();
  if (!negotiationLimit)
    return negotiationLimit.takeError();
  if (!noProgressLimit)
    return noProgressLimit.takeError();
  if (!noProgressTrendWindow)
    return noProgressTrendWindow.takeError();
  auto negotiation = decodeNegotiation(decoder);
  if (!negotiation)
    return negotiation.takeError();

  auto calibration = decoder.u64();
  auto quantile = decoder.ratio();
  auto acceptance = decoder.ratio();
  auto fallback = decoder.u64();
  auto minimum = decoder.u64();
  auto cooling = decoder.ratio();
  auto temperatureLevelLimit = decoder.u64();
  auto levelBase = decoder.u64();
  auto perMovable = decoder.u64();
  if (!calibration)
    return calibration.takeError();
  if (!quantile)
    return quantile.takeError();
  if (!acceptance)
    return acceptance.takeError();
  if (!fallback)
    return fallback.takeError();
  if (!minimum)
    return minimum.takeError();
  if (!cooling)
    return cooling.takeError();
  if (!temperatureLevelLimit)
    return temperatureLevelLimit.takeError();
  if (!levelBase)
    return levelBase.takeError();
  if (!perMovable)
    return perMovable.takeError();

  auto repairTag = decoder.u32();
  if (!repairTag)
    return repairTag.takeError();
  if (*repairTag >
      static_cast<std::uint32_t>(ResolvedPnrExactRepairKind::CpSat))
    return invalid("unknown exact-repair union tag");
  ResolvedPnrExactRepairPolicy repair{
      static_cast<ResolvedPnrExactRepairKind>(*repairTag), 0, 0};
  if (repair.kind == ResolvedPnrExactRepairKind::CpSat) {
    auto decisions = decoder.u64();
    auto calls = decoder.u64();
    if (!decisions)
      return decisions.takeError();
    if (!calls)
      return calls.takeError();
    repair.maxRegionDecisions = *decisions;
    repair.maxSolverCalls = *calls;
  }

  auto completionTag = decoder.u32();
  if (!completionTag)
    return completionTag.takeError();
  if (*completionTag > static_cast<std::uint32_t>(
                           ResolvedPnrCompletionGoal::FirstVerifiedCandidate))
    return invalid("unknown search completion goal");
  const auto completionGoal =
      static_cast<ResolvedPnrCompletionGoal>(*completionTag);

  auto masterSeed = decoder.u64();
  auto prng = decoder.u32();
  auto acceptanceProtocol = decoder.u32();
  if (!masterSeed)
    return masterSeed.takeError();
  if (!prng)
    return prng.takeError();
  if (*prng != static_cast<std::uint32_t>(
                   ResolvedPnrPrngProtocol::Sha256SeededXoshiro256StarStar_1_0))
    return invalid("unknown PRNG protocol");
  if (!acceptanceProtocol)
    return acceptanceProtocol.takeError();
  if (*acceptanceProtocol !=
      static_cast<std::uint32_t>(
          ResolvedPnrAcceptanceProtocol::ExpNegativeQ64Table_1_0))
    return invalid("unknown acceptance protocol");

  auto violationCount = decoder.count();
  if (!violationCount)
    return violationCount.takeError();
  ResolvedPnrTemporaryViolationPolicy violations;
  for (std::size_t ordinal = 0; ordinal != *violationCount; ++ordinal) {
    auto violation = decoder.u32();
    if (!violation)
      return violation.takeError();
    if (*violation >= resolvedPnrViolationKindCount)
      return invalid("unknown temporary violation kind");
    violations.admitted.push_back(
        static_cast<ResolvedPnrViolationKind>(*violation));
  }

  return ResolvedPnrPolicyConfig{
      {ResolvedPnrInitializerPolicy{*seeds, *assignments},
       ResolvedPnrActionProposalPolicy{*realization, *transport, *resource},
       ResolvedPnrRoutingPolicy{*endpointLimit, *negotiationLimit,
                                *noProgressLimit, *noProgressTrendWindow,
                                std::move(*negotiation)},
       ResolvedPnrAnnealingPolicy{
           *calibration, *quantile, *acceptance, *fallback, *minimum, *cooling,
           *temperatureLevelLimit, *levelBase, *perMovable},
       repair, completionGoal},
      ResolvedPnrDeterminismPolicy{
          *masterSeed,
          ResolvedPnrPrngProtocol::Sha256SeededXoshiro256StarStar_1_0,
          ResolvedPnrAcceptanceProtocol::ExpNegativeQ64Table_1_0},
      std::move(violations),
      ResolvedPnrObjectiveSelection{}};
}

llvm::Expected<
    std::pair<ResolvedObjectiveCatalogs, ResolvedPnrObjectiveSelection>>
decodeClosure(Decoder &decoder) {
  ResolvedObjectiveCatalogs catalogs;
  auto dimensionCount = decoder.count();
  if (!dimensionCount)
    return dimensionCount.takeError();
  catalogs.dimensions.reserve(*dimensionCount);
  for (std::size_t ordinal = 0; ordinal != *dimensionCount; ++ordinal) {
    auto sourceTag = decoder.u32();
    if (!sourceTag)
      return sourceTag.takeError();
    ResolvedObjectiveScalarSource source =
        ResolvedMappingMeasureObjectiveSource{0};
    if (*sourceTag == 0) {
      auto kind = decoder.u32();
      if (!kind)
        return kind.takeError();
      source = ResolvedMappingViolationObjectiveSource{
          static_cast<ResolvedPnrViolationKind>(*kind)};
    } else if (*sourceTag == 1) {
      auto kind = decoder.u32();
      if (!kind)
        return kind.takeError();
      source = ResolvedMappingMeasureObjectiveSource{*kind};
    } else {
      return invalid("PnR objective source is not Mapping-owned");
    }
    auto direction = decoder.u32();
    const auto decodeScalar = [&]() -> llvm::Expected<ResolvedObjectiveScalar> {
      auto tag = decoder.u32();
      if (!tag)
        return tag.takeError();
      if (*tag == 0) {
        auto negative = decoder.u32();
        auto magnitude = decoder.u64();
        if (!negative)
          return negative.takeError();
        if (*negative > 1)
          return invalid("invalid objective integer sign");
        if (!magnitude)
          return magnitude.takeError();
        return resolvedObjectiveInteger(*magnitude, *negative != 0);
      }
      if (*tag != 1)
        return invalid("unknown objective scalar kind");
      auto coefficient = decoder.i64();
      auto exponent = decoder.i64();
      if (!coefficient)
        return coefficient.takeError();
      if (!exponent)
        return exponent.takeError();
      return resolvedObjectiveDecimal(*coefficient, *exponent);
    };
    auto origin = decodeScalar();
    auto quantum = decodeScalar();
    auto lower = decoder.u64();
    auto upper = decoder.u64();
    if (!direction)
      return direction.takeError();
    if (*direction >
        static_cast<std::uint32_t>(ResolvedObjectiveDirection::Maximize))
      return invalid("unknown objective direction");
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

  auto levelCount = decoder.count();
  if (!levelCount)
    return levelCount.takeError();
  catalogs.weightedLevels.reserve(*levelCount);
  for (std::size_t levelOrdinal = 0; levelOrdinal != *levelCount;
       ++levelOrdinal) {
    auto termCount = decoder.count();
    if (!termCount)
      return termCount.takeError();
    ResolvedWeightedObjectiveLevel level;
    level.terms.reserve(*termCount);
    for (std::size_t termOrdinal = 0; termOrdinal != *termCount;
         ++termOrdinal) {
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

  auto orderingCount = decoder.count();
  if (!orderingCount)
    return orderingCount.takeError();
  catalogs.totalOrderings.reserve(*orderingCount);
  for (std::size_t orderingOrdinal = 0; orderingOrdinal != *orderingCount;
       ++orderingOrdinal) {
    auto referenceCount = decoder.count();
    if (!referenceCount)
      return referenceCount.takeError();
    ResolvedTotalOrdering ordering;
    ordering.weightedLevels.reserve(*referenceCount);
    for (std::size_t ordinal = 0; ordinal != *referenceCount; ++ordinal) {
      auto level = decoder.u32();
      if (!level)
        return level.takeError();
      ordering.weightedLevels.push_back(*level);
    }
    catalogs.totalOrderings.push_back(std::move(ordering));
  }

  auto selectedOrdering = decoder.u32();
  auto selectedEnergy = decoder.u32();
  if (!selectedOrdering)
    return selectedOrdering.takeError();
  if (!selectedEnergy)
    return selectedEnergy.takeError();
  ResolvedPnrObjectiveSelection selection{*selectedOrdering, *selectedEnergy};
  return std::make_pair(std::move(catalogs), std::move(selection));
}

struct DecodedPnrConfigView final {
  ResolvedPnrPolicyConfig policy;
  ResolvedObjectiveCatalogs catalogs;
  std::vector<SystemBindingPartitionIntent> partitions;
};

llvm::Expected<DecodedPnrConfigView>
decodeView(PnrConfigDomain domain, llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto policy = decodePolicy(decoder);
  if (!policy)
    return policy.takeError();
  auto closure = decodeClosure(decoder);
  if (!closure)
    return closure.takeError();
  policy->objectiveSelection = std::move(closure->second);
  std::vector<SystemBindingPartitionIntent> partitions;
  if (domain == PnrConfigDomain::System) {
    auto count = decoder.count();
    if (!count)
      return count.takeError();
    partitions.reserve(*count);
    for (std::size_t ordinal = 0; ordinal != *count; ++ordinal) {
      auto identityBytes = decoder.bytes(ArtifactIdentity::byteSize);
      auto entity = decoder.u64();
      auto partitionCount = decoder.u64();
      if (!identityBytes)
        return identityBytes.takeError();
      auto identity = ArtifactIdentity::fromBytes(*identityBytes);
      if (!identity)
        return identity.takeError();
      if (!entity)
        return entity.takeError();
      if (!partitionCount)
        return partitionCount.takeError();
      partitions.push_back(
          {{std::move(*identity), ::dataflow::RootThreadLaunchId(*entity)},
           *partitionCount});
    }
  }
  if (decoder.remaining() != 0)
    return invalid("trailing bytes");
  if (llvm::Error error =
          validateResolvedPnrPolicyConfig(*policy, closure->first))
    return std::move(error);
  if (llvm::Error error = validateObjectiveArithmetic(closure->first))
    return std::move(error);
  auto canonical = canonicalPartitionIntent(domain, partitions);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != partitions)
    return invalid("System partition intent is not canonical");
  return DecodedPnrConfigView{std::move(*policy), std::move(closure->first),
                              std::move(partitions)};
}

llvm::Expected<std::pair<ResolvedPnrPolicyConfig, ResolvedObjectiveCatalogs>>
projectSelectedClosure(const ResolvedPnrPolicyConfig &sourcePolicy,
                       const ResolvedObjectiveCatalogs &sourceCatalogs) {
  if (llvm::Error error =
          validateResolvedPnrPolicyConfig(sourcePolicy, sourceCatalogs))
    return std::move(error);

  std::set<std::uint32_t> selectedLevels;
  const ResolvedTotalOrdering &sourceOrdering =
      sourceCatalogs.totalOrderings[sourcePolicy.objectiveSelection
                                        .selectedTotalOrdering];
  selectedLevels.insert(sourceOrdering.weightedLevels.begin(),
                        sourceOrdering.weightedLevels.end());
  selectedLevels.insert(sourcePolicy.objectiveSelection.selectedSearchEnergy);

  std::set<std::uint32_t> selectedDimensions;
  for (std::uint32_t level : selectedLevels)
    for (const ResolvedWeightedObjectiveTerm &term :
         sourceCatalogs.weightedLevels[level].terms)
      selectedDimensions.insert(term.dimension);

  std::vector<std::uint32_t> dimensionMap(sourceCatalogs.dimensions.size(),
                                          UINT32_MAX);
  ResolvedObjectiveCatalogs selectedCatalogs;
  for (std::uint32_t oldOrdinal = 0;
       oldOrdinal != sourceCatalogs.dimensions.size(); ++oldOrdinal) {
    if (!selectedDimensions.count(oldOrdinal))
      continue;
    dimensionMap[oldOrdinal] = selectedCatalogs.dimensions.size();
    selectedCatalogs.dimensions.push_back(
        sourceCatalogs.dimensions[oldOrdinal]);
  }

  std::vector<std::uint32_t> levelMap(sourceCatalogs.weightedLevels.size(),
                                      UINT32_MAX);
  for (std::uint32_t oldOrdinal = 0;
       oldOrdinal != sourceCatalogs.weightedLevels.size(); ++oldOrdinal) {
    if (!selectedLevels.count(oldOrdinal))
      continue;
    ResolvedWeightedObjectiveLevel selected =
        sourceCatalogs.weightedLevels[oldOrdinal];
    for (ResolvedWeightedObjectiveTerm &term : selected.terms)
      term.dimension = dimensionMap[term.dimension];
    levelMap[oldOrdinal] = selectedCatalogs.weightedLevels.size();
    selectedCatalogs.weightedLevels.push_back(std::move(selected));
  }

  ResolvedTotalOrdering selectedOrdering = sourceOrdering;
  for (std::uint32_t &level : selectedOrdering.weightedLevels)
    level = levelMap[level];
  selectedCatalogs.totalOrderings.push_back(std::move(selectedOrdering));

  ResolvedPnrPolicyConfig selectedPolicy = sourcePolicy;
  selectedPolicy.objectiveSelection.selectedTotalOrdering = 0;
  selectedPolicy.objectiveSelection.selectedSearchEnergy =
      levelMap[sourcePolicy.objectiveSelection.selectedSearchEnergy];
  if (llvm::Error error =
          validateResolvedPnrPolicyConfig(selectedPolicy, selectedCatalogs))
    return std::move(error);
  if (llvm::Error error = validateObjectiveArithmetic(selectedCatalogs))
    return std::move(error);
  return std::make_pair(std::move(selectedPolicy), std::move(selectedCatalogs));
}

llvm::Expected<ResolvedPnrConfigView>
makeProjectedView(PnrConfigDomain domain, const ResolvedPnrPolicyConfig &policy,
                  const ResolvedObjectiveCatalogs &catalogs) {
  auto selected = projectSelectedClosure(policy, catalogs);
  if (!selected)
    return selected.takeError();
  if (llvm::Error error =
          validateDomainCapabilities(domain, selected->first, selected->second))
    return std::move(error);
  std::vector<SystemBindingPartitionIntent> partitions;
  std::vector<std::uint8_t> bytes =
      encodeView(domain, selected->first, selected->second, partitions);
  auto digest = computeComponentViewDigest(descriptorBytes(domain), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedPnrConfigViewAccess::create(
      domain, std::move(selected->first), std::move(selected->second),
      std::move(partitions), std::move(bytes), *digest);
}

llvm::Expected<ResolvedPnrConfigView>
adoptView(PnrConfigDomain domain,
          llvm::ArrayRef<std::uint8_t> suppliedDescriptor,
          llvm::ArrayRef<std::uint8_t> suppliedBytes,
          const ComponentViewDigest &suppliedDigest) {
  if (suppliedDescriptor != descriptorBytes(domain))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "pnr_config_descriptor_mismatch: expected exact domain descriptor");
  if (llvm::Error error = validateComponentViewDigest(
          suppliedDescriptor, suppliedBytes, suppliedDigest))
    return std::move(error);
  auto decoded = decodeView(domain, suppliedBytes);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = validateDomainCapabilities(domain, decoded->policy,
                                                     decoded->catalogs))
    return std::move(error);
  std::vector<std::uint8_t> canonical = encodeView(
      domain, decoded->policy, decoded->catalogs, decoded->partitions);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != suppliedBytes)
    return invalid("decoded value does not re-encode to exact source bytes");
  return ResolvedPnrConfigViewAccess::create(
      domain, std::move(decoded->policy), std::move(decoded->catalogs),
      std::move(decoded->partitions), std::move(canonical), suppliedDigest);
}

} // namespace

llvm::ArrayRef<std::uint8_t>
ResolvedPnrConfigView::schemaDescriptorBytes() const {
  return descriptorBytes(domain_);
}

llvm::ArrayRef<std::uint8_t> resolvedSpatialPnrConfigSchemaDescriptorBytes() {
  return descriptorBytes(PnrConfigDomain::Spatial);
}

llvm::ArrayRef<std::uint8_t> resolvedSystemPnrConfigSchemaDescriptorBytes() {
  return descriptorBytes(PnrConfigDomain::System);
}

llvm::Expected<ResolvedPnrConfigView>
projectResolvedSpatialPnrConfigView(const ResolvedConfig &config) {
  return makeProjectedView(PnrConfigDomain::Spatial, config.dse.spatialPnr,
                           config.dse.objectiveCatalogs);
}

llvm::Expected<ResolvedPnrConfigView>
projectResolvedSystemPnrConfigView(const ResolvedConfig &config) {
  return makeProjectedView(PnrConfigDomain::System, config.dse.systemPnr,
                           config.dse.objectiveCatalogs);
}

llvm::Expected<ResolvedPnrConfigView> adoptResolvedSpatialPnrConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  return adoptView(PnrConfigDomain::Spatial, schemaDescriptorBytes,
                   canonicalViewBytes, digest);
}

llvm::Expected<ResolvedPnrConfigView> adoptResolvedSystemPnrConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  return adoptView(PnrConfigDomain::System, schemaDescriptorBytes,
                   canonicalViewBytes, digest);
}

llvm::Expected<ResolvedPnrConfigView> specializeResolvedSystemPnrConfigView(
    const ResolvedPnrConfigView &base,
    llvm::ArrayRef<SystemBindingPartitionIntent> partitions) {
  if (base.domain() != PnrConfigDomain::System)
    return invalid("System partition intent requires a System PnR view");
  auto canonical =
      canonicalPartitionIntent(PnrConfigDomain::System, partitions);
  if (!canonical)
    return canonical.takeError();
  ResolvedPnrPolicyConfig policy = base.policy();
  ResolvedObjectiveCatalogs catalogs = base.selectedObjectiveCatalogs();
  std::vector<std::uint8_t> bytes =
      encodeView(PnrConfigDomain::System, policy, catalogs, *canonical);
  auto digest = computeComponentViewDigest(
      resolvedSystemPnrConfigSchemaDescriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedPnrConfigViewAccess::create(
      PnrConfigDomain::System, std::move(policy), std::move(catalogs),
      std::move(*canonical), std::move(bytes), *digest);
}

std::vector<DeterministicWorkBudgetEntry>
deriveDeterministicWorkBudgetView(const ResolvedPnrConfigView &view) {
  const ResolvedPnrSearchPolicy &search = view.policy().search;
  std::vector<DeterministicWorkBudgetEntry> result = {
      {PnrWorkUnit::SeedAttempt, search.initializer.seedAttemptCount},
      {PnrWorkUnit::AssignmentAttemptPerSeed,
       search.initializer.assignmentAttemptLimitPerSeed},
      {PnrWorkUnit::EndpointExpansion, search.routing.endpointExpansionLimit},
      {PnrWorkUnit::NegotiationIteration,
       search.routing.negotiationIterationLimit},
      {PnrWorkUnit::ConsecutiveNoProgressIteration,
       search.routing.noProgressIterationLimit},
      {PnrWorkUnit::NoProgressTrendTransition,
       search.routing.noProgressTrendWindow},
      {PnrWorkUnit::CalibrationProposal,
       search.annealing.calibrationProposalCount},
      {PnrWorkUnit::TemperatureLevel, search.annealing.temperatureLevelLimit},
      {PnrWorkUnit::ProposalPerLevelBase,
       search.annealing.proposalsPerLevelBase},
      {PnrWorkUnit::ProposalPerMovableDecision,
       search.annealing.proposalsPerMovableDecision},
  };
  if (search.exactRepair.kind == ResolvedPnrExactRepairKind::CpSat) {
    result.push_back({PnrWorkUnit::ExactRepairRegionDecision,
                      search.exactRepair.maxRegionDecisions});
    result.push_back({PnrWorkUnit::ExactRepairSolverCall,
                      search.exactRepair.maxSolverCalls});
  }
  return result;
}

} // namespace loom::pnr
