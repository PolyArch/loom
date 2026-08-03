#include "PnR/PnrConfig.h"

#include "Common/ResolvedConfig.h"

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
         std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest) {
    return ResolvedPnrConfigView(domain, std::move(policy),
                                 std::move(selectedObjectiveCatalogs),
                                 std::move(canonicalBytes), digest);
  }
};

namespace {

constexpr llvm::StringLiteral spatialDescriptor = "loom.spatial_pnr.config.1.0";
constexpr llvm::StringLiteral systemDescriptor = "loom.system_pnr.config.1.0";

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "pnr_config_bytes_invalid: " + detail);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes(PnrConfigDomain domain) {
  const llvm::StringRef descriptor =
      domain == PnrConfigDomain::Spatial ? spatialDescriptor : systemDescriptor;
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
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

  llvm::Expected<std::size_t> count() {
    auto countOrErr = u64();
    if (!countOrErr)
      return countOrErr.takeError();
    if (*countOrErr > remaining())
      return invalid("sequence count exceeds remaining bytes");
    return static_cast<std::size_t>(*countOrErr);
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
  encodeNegotiation(encoder, search.routing.negotiation);
  encoder.u32(search.routing.routeGuidanceBinding ? 1 : 0);
  if (search.routing.routeGuidanceBinding)
    encoder.u32(*search.routing.routeGuidanceBinding);
  encoder.u64(search.annealing.calibrationProposalCount);
  encoder.ratio(search.annealing.positiveDeltaQuantile);
  encoder.ratio(search.annealing.targetInitialAcceptance);
  encoder.u64(search.annealing.fallbackTemperature);
  encoder.u64(search.annealing.minimumTemperature);
  encoder.ratio(search.annealing.coolingRatio);
  encoder.u64(search.annealing.proposalsPerLevelBase);
  encoder.u64(search.annealing.proposalsPerMovableDecision);
  encoder.u64(search.focusedClosureProposalLimit);
  encoder.u32(static_cast<std::uint32_t>(search.exactRepair.kind));
  if (search.exactRepair.kind == ResolvedPnrExactRepairKind::CpSat) {
    encoder.u64(search.exactRepair.maxRegionDecisions);
    encoder.u64(search.exactRepair.maxSolverCalls);
  }
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
  encoder.u64(0);
  encoder.u64(catalogs.dimensions.size());
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    encoder.u32(static_cast<std::uint32_t>(dimension.sourceKind));
    encoder.u32(dimension.sourceOrdinal);
    encoder.u32(static_cast<std::uint32_t>(dimension.direction));
    encoder.u64(dimension.origin);
    encoder.u64(dimension.quantum);
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
  encoder.u64(policy.objectiveSelection.focusedClosureDimensions.size());
  for (std::uint32_t dimension :
       policy.objectiveSelection.focusedClosureDimensions)
    encoder.u32(dimension);
  encoder.u64(policy.evaluationBindings.size());
  for (const ResolvedPnrEvaluationBindingSelection &binding :
       policy.evaluationBindings) {
    encoder.u32(binding.obligationTemplate);
    encoder.u32(binding.interactionDomain);
  }
}

std::vector<std::uint8_t>
encodeView(const ResolvedPnrPolicyConfig &policy,
           const ResolvedObjectiveCatalogs &catalogs) {
  Encoder encoder;
  encodePolicy(encoder, policy);
  encodeClosure(encoder, policy, catalogs);
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
  auto negotiation = decodeNegotiation(decoder);
  if (!negotiation)
    return negotiation.takeError();

  auto guidancePresence = decoder.u32();
  if (!guidancePresence)
    return guidancePresence.takeError();
  if (*guidancePresence > 1)
    return invalid("invalid route-guidance optional tag");
  std::optional<std::uint32_t> guidance;
  if (*guidancePresence == 1) {
    auto guidanceRef = decoder.u32();
    if (!guidanceRef)
      return guidanceRef.takeError();
    guidance = *guidanceRef;
  }

  auto calibration = decoder.u64();
  auto quantile = decoder.ratio();
  auto acceptance = decoder.ratio();
  auto fallback = decoder.u64();
  auto minimum = decoder.u64();
  auto cooling = decoder.ratio();
  auto levelBase = decoder.u64();
  auto perMovable = decoder.u64();
  auto focusedLimit = decoder.u64();
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
  if (!levelBase)
    return levelBase.takeError();
  if (!perMovable)
    return perMovable.takeError();
  if (!focusedLimit)
    return focusedLimit.takeError();

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
                                std::move(*negotiation), guidance},
       ResolvedPnrAnnealingPolicy{*calibration, *quantile, *acceptance,
                                  *fallback, *minimum, *cooling, *levelBase,
                                  *perMovable},
       *focusedLimit, repair},
      ResolvedPnrDeterminismPolicy{
          *masterSeed,
          ResolvedPnrPrngProtocol::Sha256SeededXoshiro256StarStar_1_0,
          ResolvedPnrAcceptanceProtocol::ExpNegativeQ64Table_1_0},
      std::move(violations),
      ResolvedPnrObjectiveSelection{},
      {}};
}

llvm::Expected<
    std::pair<ResolvedObjectiveCatalogs, ResolvedPnrObjectiveSelection>>
decodeClosure(Decoder &decoder,
              std::vector<ResolvedPnrEvaluationBindingSelection> &bindings) {
  auto templateCount = decoder.count();
  if (!templateCount)
    return templateCount.takeError();
  if (*templateCount != 0)
    return invalid("Evaluation obligation owner is unavailable");

  ResolvedObjectiveCatalogs catalogs;
  auto dimensionCount = decoder.count();
  if (!dimensionCount)
    return dimensionCount.takeError();
  catalogs.dimensions.reserve(*dimensionCount);
  for (std::size_t ordinal = 0; ordinal != *dimensionCount; ++ordinal) {
    auto source = decoder.u32();
    auto sourceOrdinal = decoder.u32();
    auto direction = decoder.u32();
    auto origin = decoder.u64();
    auto quantum = decoder.u64();
    auto lower = decoder.u64();
    auto upper = decoder.u64();
    if (!source)
      return source.takeError();
    if (*source >
        static_cast<std::uint32_t>(ResolvedObjectiveSourceKind::MappingMeasure))
      return invalid("unknown objective source kind");
    if (!sourceOrdinal)
      return sourceOrdinal.takeError();
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
        {static_cast<ResolvedObjectiveSourceKind>(*source), *sourceOrdinal,
         static_cast<ResolvedObjectiveDirection>(*direction), *origin, *quantum,
         *lower, *upper});
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
  auto focusedCount = decoder.count();
  if (!selectedOrdering)
    return selectedOrdering.takeError();
  if (!selectedEnergy)
    return selectedEnergy.takeError();
  if (!focusedCount)
    return focusedCount.takeError();
  ResolvedPnrObjectiveSelection selection{
      *selectedOrdering, *selectedEnergy, {}};
  for (std::size_t ordinal = 0; ordinal != *focusedCount; ++ordinal) {
    auto dimension = decoder.u32();
    if (!dimension)
      return dimension.takeError();
    selection.focusedClosureDimensions.push_back(*dimension);
  }

  auto bindingCount = decoder.count();
  if (!bindingCount)
    return bindingCount.takeError();
  bindings.reserve(*bindingCount);
  for (std::size_t ordinal = 0; ordinal != *bindingCount; ++ordinal) {
    auto obligation = decoder.u32();
    auto domain = decoder.u32();
    if (!obligation)
      return obligation.takeError();
    if (!domain)
      return domain.takeError();
    bindings.push_back({*obligation, *domain});
  }
  return std::make_pair(std::move(catalogs), std::move(selection));
}

llvm::Expected<std::pair<ResolvedPnrPolicyConfig, ResolvedObjectiveCatalogs>>
decodeView(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto policy = decodePolicy(decoder);
  if (!policy)
    return policy.takeError();
  auto closure = decodeClosure(decoder, policy->evaluationBindings);
  if (!closure)
    return closure.takeError();
  policy->objectiveSelection = std::move(closure->second);
  if (decoder.remaining() != 0)
    return invalid("trailing bytes");
  if (llvm::Error error =
          validateResolvedPnrPolicyConfig(*policy, closure->first))
    return std::move(error);
  return std::make_pair(std::move(*policy), std::move(closure->first));
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

  std::set<std::uint32_t> selectedDimensions(
      sourcePolicy.objectiveSelection.focusedClosureDimensions.begin(),
      sourcePolicy.objectiveSelection.focusedClosureDimensions.end());
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
  for (std::uint32_t &dimension :
       selectedPolicy.objectiveSelection.focusedClosureDimensions)
    dimension = dimensionMap[dimension];

  if (llvm::Error error =
          validateResolvedPnrPolicyConfig(selectedPolicy, selectedCatalogs))
    return std::move(error);
  return std::make_pair(std::move(selectedPolicy), std::move(selectedCatalogs));
}

llvm::Expected<ResolvedPnrConfigView>
makeProjectedView(PnrConfigDomain domain, const ResolvedPnrPolicyConfig &policy,
                  const ResolvedObjectiveCatalogs &catalogs) {
  auto selected = projectSelectedClosure(policy, catalogs);
  if (!selected)
    return selected.takeError();
  std::vector<std::uint8_t> bytes =
      encodeView(selected->first, selected->second);
  auto digest = computeComponentViewDigest(descriptorBytes(domain), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedPnrConfigViewAccess::create(domain, std::move(selected->first),
                                             std::move(selected->second),
                                             std::move(bytes), *digest);
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
  auto decoded = decodeView(suppliedBytes);
  if (!decoded)
    return decoded.takeError();
  std::vector<std::uint8_t> canonical =
      encodeView(decoded->first, decoded->second);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != suppliedBytes)
    return invalid("decoded value does not re-encode to exact source bytes");
  return ResolvedPnrConfigViewAccess::create(
      domain, std::move(decoded->first), std::move(decoded->second),
      std::move(canonical), suppliedDigest);
}

} // namespace

llvm::ArrayRef<std::uint8_t>
ResolvedPnrConfigView::schemaDescriptorBytes() const {
  return descriptorBytes(domain_);
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
      {PnrWorkUnit::CalibrationProposal,
       search.annealing.calibrationProposalCount},
      {PnrWorkUnit::ProposalPerLevelBase,
       search.annealing.proposalsPerLevelBase},
      {PnrWorkUnit::ProposalPerMovableDecision,
       search.annealing.proposalsPerMovableDecision},
      {PnrWorkUnit::FocusedClosureProposal, search.focusedClosureProposalLimit},
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
