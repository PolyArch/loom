#include "PnR/SpatialRootSupplyAdmission.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/STLExtras.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <system_error>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_root_supply_admission_invalid: " + message);
}

void saturatingAdd(std::uint64_t source, std::uint64_t &target) {
  if (source > std::numeric_limits<std::uint64_t>::max() - target)
    target = std::numeric_limits<std::uint64_t>::max();
  else
    target += source;
}

} // namespace

llvm::Expected<SpatialRootSupplyAdmission>
analyzeSpatialRootSupply(const ::loom::mapping::TechMappingView &techMapping,
                         const ::dataflow::CanonicalDataflowProgramView &dataflow,
                         const ::loom::fabric::FabricArtifactView &fabric) {
  if (techMapping.fabricIdentity() != fabric.identity())
    return invalid("TechMapping is bound to a foreign Fabric");
  if (techMapping.dataflowIdentity() != dataflow.identity())
    return invalid("TechMapping is bound to a foreign Dataflow");

  SpatialRootSupplyAdmission result;
  auto computeDemands =
      ::loom::mapping::deriveSpatialComputeContextDemands(techMapping, fabric);
  if (!computeDemands)
    return computeDemands.takeError();
  result.computeDemandCount = computeDemands->size();

  std::map<std::vector<std::uint8_t>, std::size_t> contextOrdinals;
  std::vector<std::vector<std::size_t>> domains;
  domains.reserve(computeDemands->size());
  for (const auto &demand : *computeDemands) {
    std::vector<std::size_t> domain;
    for (const auto &placement : demand.placements) {
      for (const auto &context : placement.contexts) {
        const std::vector<std::uint8_t> key =
            ::loom::fabric::canonicalFabricBytes(context);
        auto [found, inserted] =
            contextOrdinals.try_emplace(key, contextOrdinals.size());
        (void)inserted;
        domain.push_back(found->second);
      }
    }
    llvm::sort(domain);
    domain.erase(std::unique(domain.begin(), domain.end()), domain.end());
    if (domain.size() > std::numeric_limits<std::uint64_t>::max() -
                            result.computeContextEdgeCount)
      result.computeContextEdgeCount =
          std::numeric_limits<std::uint64_t>::max();
    else
      result.computeContextEdgeCount += domain.size();
    domains.push_back(std::move(domain));
  }
  auto supply = ::loom::mapping::analyzeSpatialComputeContextSupply(
      domains, contextOrdinals.size());
  if (!supply)
    return supply.takeError();
  result.computeContextValueCount = supply->valueCount;
  result.computeContextEdgeCount = supply->edgeCount;
  result.computeContextMaximumMatching = supply->maximumMatching;
  result.computeHallDemandCount = supply->hallDemands.size();
  result.computeHallContextValueCount = supply->hallValueCount;
  result.computeHallRealizations = std::move(supply->hallDemands);
  saturatingAdd(supply->deterministicWork, result.deterministicWork);
  if (!supply->admissible()) {
    result.disposition = SpatialRootSupplyDisposition::ProvenInfeasible;
    result.diagnostic =
        "compute resident-context supply has maximum matching " +
        std::to_string(result.computeContextMaximumMatching) + "/" +
        std::to_string(result.computeDemandCount) + "; Hall witness has " +
        std::to_string(result.computeHallDemandCount) + " demands and " +
        std::to_string(result.computeHallContextValueCount) + " contexts";
    return result;
  }

  auto memoryDemands = ::loom::mapping::deriveSpatialMemoryOccurrenceDemands(
      techMapping, dataflow, fabric);
  if (!memoryDemands)
    return memoryDemands.takeError();
  result.memoryDemandCount = memoryDemands->size();
  std::vector<const ::loom::mapping::SpatialMemoryOccurrenceDemandView *>
      memoryDemandPointers;
  memoryDemandPointers.reserve(memoryDemands->size());
  for (const auto &demand : *memoryDemands) {
    saturatingAdd(demand.projectionWork, result.deterministicWork);
    memoryDemandPointers.push_back(&demand);
  }
  auto memorySupply =
      ::loom::mapping::analyzeSpatialMemoryOccurrenceSupply(
          memoryDemandPointers);
  if (!memorySupply)
    return memorySupply.takeError();
  result.memoryOccurrenceValueCount = memorySupply->occurrenceValueCount;
  result.memoryOccurrenceChoiceCount = memorySupply->occurrenceChoiceCount;
  result.memoryExclusiveRelationCount =
      memorySupply->exclusiveRelationCount;
  result.memoryAssignmentAttempts = memorySupply->assignmentAttempts;
  result.memoryFailure = memorySupply->failure;
  saturatingAdd(memorySupply->deterministicWork, result.deterministicWork);
  if (!memorySupply->admissible()) {
    result.disposition = SpatialRootSupplyDisposition::ProvenInfeasible;
    result.diagnostic =
        "memory occurrence supply is " +
        ::loom::mapping::spatialMemoryOccurrenceSupplyFailureKindSpelling(
            memorySupply->failure)
            .str() +
        ": " + std::to_string(memorySupply->failingDemandCount) +
        " demands, " +
        std::to_string(memorySupply->failingOccurrenceCount) +
        " occurrences";
    if (memorySupply->failingResourceKind)
      result.diagnostic +=
          ", resource " +
          ::loom::mapping::spatialMemoryExclusiveResourceKindSpelling(
              *memorySupply->failingResourceKind)
              .str();
    if (memorySupply->failure ==
        ::loom::mapping::SpatialMemoryOccurrenceSupplyFailureKind::
            ResidentCapacityDeficit)
      result.diagnostic +=
          ", resident demand/capacity " +
          std::to_string(memorySupply->failingResidentDemand) + "/" +
          std::to_string(memorySupply->failingResidentCapacity);
  }
  return result;
}

} // namespace loom::pnr
