#include "PnR/MappingObjective.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <system_error>
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
      return 0;
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

std::uint64_t
loom::pnr::spatialMappingMeasureValue(const SpatialCandidateState &candidate,
                                      MappingMeasureKind kind) {
  switch (kind) {
  case MappingMeasureKind::TotalSelectedTraversalClaim:
    return candidate.totalSelectedTraversalClaim();
  }
  llvm_unreachable("unknown Mapping measure kind");
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
