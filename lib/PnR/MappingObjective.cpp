#include "PnR/MappingObjective.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/System/SystemCandidateState.h"
#include "PnR/System/SystemServiceRouter.h"

#include "SpatialProgressAnalysis.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

namespace {

constexpr MappingObjectiveRegistryDescriptor registry{
    "loom.mapping.pnr.objective", 2, 0};

constexpr std::array<MappingViolationDescriptor, resolvedPnrViolationKindCount>
    violations{{
#define LOOM_MAPPING_VIOLATION(Name, Ordinal, DisplayName, ConfigSpelling)     \
  {ResolvedPnrViolationKind::Name, DisplayName},
#include "Common/MappingObjectiveKinds.def"
    }};

constexpr std::array<MappingMeasureDescriptor, mappingMeasureKindCount>
    measures{{
#define LOOM_MAPPING_MEASURE(Name, Ordinal, DisplayName)                       \
  {MappingMeasureKind::Name, DisplayName},
#include "Common/MappingObjectiveKinds.def"
    }};

llvm::Error objectiveError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_objective_invalid: " + message);
}

llvm::Error unavailable(llvm::StringRef source) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "objective_unavailable: required Mapping objective source '%s' is "
      "absent",
      source.str().c_str());
}

llvm::Error invalidObjectiveReference(llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "dse_objective_invalid: selected Mapping %s reference is out of range",
      detail.str().c_str());
}

llvm::Error checkedAdd(std::uint64_t &value, std::uint64_t increment,
                       llvm::StringRef what) {
  if (increment > std::numeric_limits<std::uint64_t>::max() - value)
    return objectiveError(what + " exceeds u64");
  value += increment;
  return llvm::Error::success();
}

} // namespace

const MappingObjectiveRegistryDescriptor &
loom::pnr::mappingObjectiveRegistryDescriptor() {
  return registry;
}

llvm::ArrayRef<MappingViolationDescriptor>
loom::pnr::mappingViolationDescriptors() {
  return violations;
}

llvm::ArrayRef<MappingMeasureDescriptor>
loom::pnr::mappingMeasureDescriptors() {
  return measures;
}

bool loom::pnr::spatialMappingViolationAvailable(
    ResolvedPnrViolationKind kind) {
  switch (kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
  case ResolvedPnrViolationKind::CapacityOveruse:
  case ResolvedPnrViolationKind::TagUnassigned:
  case ResolvedPnrViolationKind::TagConflict:
  case ResolvedPnrViolationKind::HardProgressViolation:
    return true;
  }
  llvm_unreachable("unknown Mapping violation kind");
}

llvm::Expected<std::uint64_t>
loom::pnr::spatialMappingViolationValue(const SpatialCandidateState &candidate,
                                        ResolvedPnrViolationKind kind) {
  if (!spatialMappingViolationAvailable(kind)) {
    const auto ordinal = static_cast<std::uint32_t>(kind);
    if (ordinal >= violations.size())
      llvm_unreachable("unknown Mapping violation kind");
    return llvm::createStringError(
        std::make_error_code(std::errc::operation_not_supported),
        "objective_unavailable: required Spatial violation owner '%s' is "
        "absent",
        violations[ordinal].spelling.str().c_str());
  }
  switch (kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
    return candidate.unroutedObligationCount();
  case ResolvedPnrViolationKind::CapacityOveruse: {
    const std::uint64_t atomic = candidate.atomicCapacityOveruse();
    const std::uint64_t route = candidate.routeCapacityOveruse();
    if (route > std::numeric_limits<std::uint64_t>::max() - atomic)
      return llvm::createStringError(
          std::make_error_code(std::errc::value_too_large),
          "Spatial CapacityOveruse exceeds u64");
    return atomic + route;
  }
  case ResolvedPnrViolationKind::TagUnassigned:
    return candidate.tagUnassignedCount();
  case ResolvedPnrViolationKind::TagConflict:
    return candidate.tagConflictCount();
  case ResolvedPnrViolationKind::HardProgressViolation:
    switch (candidate.problem().progressClosure().kind) {
    case ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet:
      return spatialCandidateClosedWaitCount(candidate);
    case ::loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet:
      return 1;
    case ::loom::mapping::MappingProgressClosureKind::ProofNotEstablished:
      return llvm::createStringError(
          std::make_error_code(std::errc::operation_not_supported),
          "proof_not_established: Spatial progress closure is unavailable");
    }
    llvm_unreachable("unknown Spatial progress closure kind");
  }
  llvm_unreachable("unknown Mapping violation kind");
}

llvm::Expected<bool> loom::pnr::spatialMappingViolationsAreZero(
    const SpatialCandidateState &candidate) {
  for (std::uint32_t ordinal = 0; ordinal != resolvedPnrViolationKindCount;
       ++ordinal) {
    auto value = spatialMappingViolationValue(
        candidate, static_cast<ResolvedPnrViolationKind>(ordinal));
    if (!value)
      return value.takeError();
    if (*value != 0)
      return false;
  }
  return true;
}

std::uint64_t
loom::pnr::spatialMappingMeasureValue(const SpatialCandidateState &candidate,
                                      MappingMeasureKind kind) {
  switch (kind) {
  case MappingMeasureKind::TotalSelectedTraversalClaim:
    return candidate.totalSelectedTraversalClaim();
  }
  llvm_unreachable("unknown Mapping measure kind");
}

llvm::Expected<std::uint64_t>
loom::pnr::systemMappingViolationValue(const SystemCandidateState &candidate,
                                       ResolvedPnrViolationKind kind) {
  switch (kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
  case ResolvedPnrViolationKind::TagUnassigned:
  case ResolvedPnrViolationKind::TagConflict:
    return 0;
  case ResolvedPnrViolationKind::CapacityOveruse:
    return candidate.capacityOveruse();
  case ResolvedPnrViolationKind::HardProgressViolation:
    switch (candidate.problem().progressClosure().kind) {
    case ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet:
      return 0;
    case ::loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet:
      return 1;
    case ::loom::mapping::MappingProgressClosureKind::ProofNotEstablished:
      return llvm::createStringError(
          std::make_error_code(std::errc::operation_not_supported),
          "proof_not_established: System progress closure is unavailable");
    }
    llvm_unreachable("unknown System progress closure kind");
  }
  llvm_unreachable("unknown Mapping violation kind");
}

llvm::Expected<std::uint64_t>
loom::pnr::systemMappingMeasureValue(const SystemCandidateState &candidate,
                                     MappingMeasureKind kind) {
  if (kind != MappingMeasureKind::TotalSelectedTraversalClaim)
    llvm_unreachable("unknown Mapping measure kind");
  return detail::measureSystemServiceRouteTraversalClaim(
      candidate.problem().routingTopology(),
      {candidate.serviceRoutes(), candidate.serviceRouteNodes(),
       candidate.serviceRouteSinks()});
}

llvm::Expected<MappingObjectiveProgram>
MappingObjectiveProgram::get(const ResolvedObjectiveCatalogs &catalogs,
                             const ResolvedPnrObjectiveSelection &selection) {
  static_assert(resolvedPnrViolationKindCount <= 64);
  static_assert(mappingMeasureKindCount <= 64);
  auto program = dse::ObjectiveProgram::get(catalogs);
  if (!program)
    return program.takeError();
  if (selection.selectedTotalOrdering >= program->totalOrderingCount())
    return invalidObjectiveReference("total ordering");
  if (selection.selectedSearchEnergy >= program->weightedLevelCount())
    return invalidObjectiveReference("search energy");

  std::uint64_t selectedViolations = 0;
  std::uint64_t selectedMeasures = 0;
  const auto violations = mappingViolationDescriptors();
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    if (const auto *source =
            std::get_if<ResolvedMappingViolationObjectiveSource>(
                &dimension.source)) {
      const std::uint32_t ordinal = static_cast<std::uint32_t>(source->kind);
      if (ordinal >= violations.size() ||
          !spatialMappingViolationAvailable(source->kind))
        return unavailable(ordinal < violations.size()
                               ? violations[ordinal].spelling
                               : llvm::StringRef("Mapping violation"));
      selectedViolations |= UINT64_C(1) << ordinal;
      continue;
    }
    if (const auto *source = std::get_if<ResolvedMappingMeasureObjectiveSource>(
            &dimension.source)) {
      if (source->ordinal >= mappingMeasureKindCount)
        return unavailable("Mapping measure");
      selectedMeasures |= UINT64_C(1) << source->ordinal;
      continue;
    }
    return unavailable("Evaluation metric interaction");
  }
  return MappingObjectiveProgram(
      std::move(*program), selectedViolations, selectedMeasures,
      selection.selectedTotalOrdering, selection.selectedSearchEnergy);
}

llvm::Expected<dse::ObjectiveVector> MappingObjectiveProgram::evaluate(
    const SpatialCandidateState &candidate) const {
  std::array<std::uint64_t, resolvedPnrViolationKindCount> violations{};
  for (std::uint32_t ordinal = 0; ordinal != violations.size(); ++ordinal) {
    if ((selectedViolations_ & (UINT64_C(1) << ordinal)) == 0)
      continue;
    auto value = spatialMappingViolationValue(
        candidate, static_cast<ResolvedPnrViolationKind>(ordinal));
    if (!value)
      return value.takeError();
    violations[ordinal] = *value;
  }
  std::array<std::uint64_t, mappingMeasureKindCount> measures{};
  for (std::uint32_t ordinal = 0; ordinal != measures.size(); ++ordinal)
    if ((selectedMeasures_ & (UINT64_C(1) << ordinal)) != 0)
      measures[ordinal] = spatialMappingMeasureValue(
          candidate, static_cast<MappingMeasureKind>(ordinal));
  dse::ObjectiveVector result = program_.makeVector();
  if (llvm::Error error = program_.evaluate({violations, measures, {}}, result))
    return std::move(error);
  return result;
}

llvm::Expected<dse::ObjectiveVector>
MappingObjectiveProgram::evaluate(const SystemCandidateState &candidate) const {
  auto traversalClaim = systemMappingMeasureValue(
      candidate, MappingMeasureKind::TotalSelectedTraversalClaim);
  if (!traversalClaim)
    return traversalClaim.takeError();
  return evaluateSystemProjection(candidate.problem(),
                                  candidate.capacityOveruse(), *traversalClaim);
}

llvm::Expected<dse::ObjectiveVector>
MappingObjectiveProgram::evaluateSystemProjection(
    const FrozenSystemPnrProblem &problem, std::uint64_t capacityOveruse,
    std::uint64_t totalSelectedTraversalClaim) const {
  std::array<std::uint64_t, resolvedPnrViolationKindCount> violations{};
  for (std::uint32_t ordinal = 0; ordinal != violations.size(); ++ordinal) {
    if ((selectedViolations_ & (UINT64_C(1) << ordinal)) == 0)
      continue;
    const auto kind = static_cast<ResolvedPnrViolationKind>(ordinal);
    if (kind == ResolvedPnrViolationKind::CapacityOveruse) {
      violations[ordinal] = capacityOveruse;
      continue;
    }
    switch (kind) {
    case ResolvedPnrViolationKind::UnroutedObligation:
    case ResolvedPnrViolationKind::TagUnassigned:
    case ResolvedPnrViolationKind::TagConflict:
      violations[ordinal] = 0;
      break;
    case ResolvedPnrViolationKind::CapacityOveruse:
      llvm_unreachable("CapacityOveruse was projected above");
    case ResolvedPnrViolationKind::HardProgressViolation:
      switch (problem.progressClosure().kind) {
      case ::loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet:
        violations[ordinal] = 0;
        break;
      case ::loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet:
        violations[ordinal] = 1;
        break;
      case ::loom::mapping::MappingProgressClosureKind::ProofNotEstablished:
        return llvm::createStringError(
            std::make_error_code(std::errc::operation_not_supported),
            "proof_not_established: System progress closure is unavailable");
      }
      break;
    }
  }
  std::array<std::uint64_t, mappingMeasureKindCount> measures{};
  for (std::uint32_t ordinal = 0; ordinal != measures.size(); ++ordinal) {
    if ((selectedMeasures_ & (UINT64_C(1) << ordinal)) == 0)
      continue;
    if (static_cast<MappingMeasureKind>(ordinal) !=
        MappingMeasureKind::TotalSelectedTraversalClaim)
      llvm_unreachable("unknown Mapping measure kind");
    measures[ordinal] = totalSelectedTraversalClaim;
  }
  dse::ObjectiveVector result = program_.makeVector();
  if (llvm::Error error = program_.evaluate({violations, measures, {}}, result))
    return std::move(error);
  return result;
}

llvm::Expected<dse::ObjectiveWideValue> MappingObjectiveProgram::selectedEnergy(
    const dse::ObjectiveVector &vector) const {
  return program_.weightedLevelValue(vector, selectedSearchEnergy_);
}

llvm::Expected<dse::ObjectiveSignedDifference>
MappingObjectiveProgram::selectedEnergyDifference(
    const dse::ObjectiveVector &left, const dse::ObjectiveVector &right) const {
  return program_.signedWeightedLevelDifference(left, right,
                                                selectedSearchEnergy_);
}

llvm::Expected<int> MappingObjectiveProgram::compareSelectedRank(
    const dse::ObjectiveVector &left,
    llvm::ArrayRef<std::uint8_t> leftCandidateKey,
    const dse::ObjectiveVector &right,
    llvm::ArrayRef<std::uint8_t> rightCandidateKey) const {
  return program_.compareTotalOrdering(
      left, leftCandidateKey, right, rightCandidateKey, selectedTotalOrdering_);
}

llvm::Expected<SpatialMappingTraversalClaimProjection>
loom::pnr::projectSpatialMappingTraversalClaims(
    const FrozenSpatialPnrProblem &problem,
    const ::loom::mapping::SpatialMappingView &mapping) {
  if (mapping.dataflowIdentity() != problem.dataflowIdentity() ||
      mapping.techMappingIdentity() != problem.techMappingIdentity() ||
      mapping.fabricIdentity() != problem.fabricIdentity())
    return objectiveError("Spatial Mapping and FrozenModel owners differ");

  const auto frozenNets = problem.transfers().logicalNets();
  if (mapping.routeTrees().size() != frozenNets.size())
    return objectiveError("Spatial Mapping route inventory is incomplete");

  std::map<std::vector<std::uint8_t>, PnrIndex> netOrdinals;
  for (auto indexed : llvm::enumerate(frozenNets)) {
    auto key = ::dataflow::encodeDataflowReference(problem.dataflowIdentity(),
                                                   indexed.value().producer);
    if (!key)
      return key.takeError();
    if (!netOrdinals
             .try_emplace(std::move(*key),
                          static_cast<PnrIndex>(indexed.index()))
             .second)
      return objectiveError("FrozenModel repeats a logical net");
  }

  const FrozenSpatialRoutingGraph &routing = problem.routing();
  std::map<std::vector<std::uint8_t>, PnrIndex> traversalOrdinals;
  for (auto indexed : llvm::enumerate(routing.traversals())) {
    auto key = ::loom::fabric::canonicalFabricBytes(indexed.value().reference);
    if (!traversalOrdinals
             .try_emplace(std::move(key),
                          static_cast<PnrIndex>(indexed.index()))
             .second)
      return objectiveError("FrozenModel repeats a physical traversal");
  }

  SpatialMappingTraversalClaimProjection projection;
  projection.logicalNets.reserve(mapping.routeTrees().size());
  std::vector<bool> seenNets(frozenNets.size(), false);
  const std::size_t claimWordCount = (routing.routeClaims().size() + 63) / 64;
  std::vector<std::uint64_t> selectedClaims(claimWordCount, 0);

  for (const ::loom::mapping::SpatialRouteTreeView &route :
       mapping.routeTrees()) {
    auto netKey = ::dataflow::encodeDataflowReference(
        problem.dataflowIdentity(), route.logicalNet);
    if (!netKey)
      return netKey.takeError();
    const auto net = netOrdinals.find(*netKey);
    if (net == netOrdinals.end() || seenNets[net->second])
      return objectiveError("Spatial Mapping has a foreign or repeated net");
    seenNets[net->second] = true;
    std::fill(selectedClaims.begin(), selectedClaims.end(), 0);

    std::uint64_t value = 0;
    for (const ::loom::mapping::SpatialRouteNodeView &node : route.nodes) {
      if (!node.incomingTraversal)
        continue;
      auto traversalKey =
          ::loom::fabric::canonicalFabricBytes(*node.incomingTraversal);
      const auto traversal = traversalOrdinals.find(traversalKey);
      if (traversal == traversalOrdinals.end())
        return objectiveError("Spatial Mapping selects a foreign traversal");
      const FrozenSpatialTraversal &record =
          routing.traversals()[traversal->second];
      for (PnrIndex claim : routing.traversalClaimKeys().slice(
               record.routeClaimOffset, record.routeClaimCount)) {
        if (claim >= routing.routeClaims().size())
          return objectiveError("Frozen traversal has an invalid claim");
        const std::uint64_t mask = std::uint64_t{1} << (claim % 64);
        std::uint64_t &word = selectedClaims[claim / 64];
        if ((word & mask) != 0)
          continue;
        word |= mask;
        if (llvm::Error error =
                checkedAdd(value, routing.routeClaims()[claim].qCost,
                           "logical-net selected traversal claim"))
          return std::move(error);
      }
    }
    if (llvm::Error error = checkedAdd(projection.total, value,
                                       "total selected traversal claim"))
      return std::move(error);
    projection.logicalNets.push_back({route.logicalNet, value});
  }
  if (std::find(seenNets.begin(), seenNets.end(), false) != seenNets.end())
    return objectiveError("Spatial Mapping omits a FrozenModel logical net");
  return projection;
}
