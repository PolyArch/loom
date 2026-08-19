#include "DSE/TechMappingHardwareFeedback.h"

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tech_mapping_hardware_feedback_invalid: " +
                                     message);
}

} // namespace

llvm::Expected<std::vector<SpatialMicroarchitectureDecisionDomain>>
projectTechMappingComputeContextGrowthDomains(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module) {
  if (module.rootKind() != fabric::FabricRootKind::Module)
    return invalid("hardware feedback target is not a Module");
  if (feedback.deficit() == 0)
    return invalid("compute-context feedback has no positive deficit");

  std::map<std::uint64_t, std::uint64_t> currentCapacityByPe;
  for (const auto &group : feedback.groups()) {
    for (const fabric::InstructionContextRef context :
         group.compatibleContexts) {
      const auto schedule = module.peSchedule(context.pe);
      if (!schedule)
        return invalid("compatible context has no PE schedule");
      if (*schedule != ::fabric::Schedule::Temporal)
        continue;
      const std::uint64_t current = module.peResidentContextCount(context.pe);
      if (current == 0 || current > std::numeric_limits<std::uint32_t>::max() -
                                        feedback.deficit())
        return invalid("Temporal PE context growth exceeds u32");
      auto [found, inserted] =
          currentCapacityByPe.emplace(context.pe.id(), current);
      if (!inserted && found->second != current)
        return invalid("Temporal PE has inconsistent context capacity");
    }
  }

  std::vector<SpatialMicroarchitectureDecisionDomain> domains;
  domains.reserve(currentCapacityByPe.size());
  for (const auto &[pe, current] : currentCapacityByPe) {
    std::vector<std::uint32_t> capacities{
        static_cast<std::uint32_t>(current + 1)};
    if (feedback.deficit() != 1)
      capacities.push_back(
          static_cast<std::uint32_t>(current + feedback.deficit()));
    domains.push_back(ResizeInstructionStoreDomain{
        fabric::FabricPeOccurrenceRef(pe), std::move(capacities)});
  }
  return domains;
}

llvm::Expected<TechMappingComputeContextJointGrowthPlan>
projectTechMappingComputeContextJointGrowthPlan(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module) {
  if (module.rootKind() != fabric::FabricRootKind::Module)
    return invalid("hardware feedback target is not a Module");
  if (feedback.deficit() == 0)
    return invalid("compute-context feedback has no positive deficit");

  struct PeSupply final {
    fabric::FabricPeOccurrenceRef pe;
    std::uint64_t currentCapacity = 0;
    std::vector<std::uint8_t> key;
  };
  std::map<std::vector<std::uint8_t>, std::size_t> peOrdinalByKey;
  std::vector<PeSupply> pes;
  for (const auto &group : feedback.groups())
    for (fabric::InstructionContextRef context : group.compatibleContexts) {
      auto schedule = module.peSchedule(context.pe);
      if (!schedule)
        return invalid("Hall feedback contains a context without a schedule");
      if (*schedule != ::fabric::Schedule::Temporal)
        continue;
      const std::vector<std::uint8_t> key =
          fabric::canonicalFabricBytes(context.pe);
      if (peOrdinalByKey.count(key))
        continue;
      const std::uint64_t capacity = module.peResidentContextCount(context.pe);
      if (capacity == 0 || capacity > std::numeric_limits<std::uint32_t>::max())
        return invalid("Temporal PE context capacity is outside u32");
      peOrdinalByKey.emplace(key, pes.size());
      pes.push_back({context.pe, capacity, key});
    }
  llvm::sort(pes, [](const PeSupply &lhs, const PeSupply &rhs) {
    return lhs.key < rhs.key;
  });
  peOrdinalByKey.clear();
  for (const auto indexed : llvm::enumerate(pes))
    peOrdinalByKey.emplace(indexed.value().key, indexed.index());
  if (pes.empty())
    return invalid("Hall feedback has no compatible Temporal PE");

  std::map<std::vector<std::uint8_t>, std::size_t> contextOrdinalByKey;
  for (const auto &group : feedback.groups())
    for (fabric::InstructionContextRef context : group.compatibleContexts) {
      const std::vector<std::uint8_t> key =
          fabric::canonicalFabricBytes(context);
      contextOrdinalByKey.emplace(key, 0);
    }
  std::size_t contextOrdinal = 0;
  for (auto &entry : contextOrdinalByKey)
    entry.second = contextOrdinal++;

  struct DemandGroup final {
    std::uint64_t count = 0;
    std::vector<std::size_t> baseValues;
    std::vector<std::size_t> compatiblePes;
  };
  std::vector<DemandGroup> groups;
  groups.reserve(feedback.groups().size());
  std::uint64_t demandCount = 0;
  for (const auto &group : feedback.groups()) {
    DemandGroup projected;
    projected.count = group.demandCount;
    if (projected.count >
        std::numeric_limits<std::uint64_t>::max() - demandCount)
      return invalid("Hall demand count overflows u64");
    demandCount += projected.count;
    std::set<std::size_t> compatiblePes;
    for (fabric::InstructionContextRef context : group.compatibleContexts) {
      const auto value =
          contextOrdinalByKey.find(fabric::canonicalFabricBytes(context));
      const auto pe =
          peOrdinalByKey.find(fabric::canonicalFabricBytes(context.pe));
      if (value == contextOrdinalByKey.end())
        return invalid("Hall feedback projection lost a context value");
      projected.baseValues.push_back(value->second);
      if (pe != peOrdinalByKey.end())
        compatiblePes.insert(pe->second);
    }
    llvm::sort(projected.baseValues);
    projected.baseValues.erase(
        std::unique(projected.baseValues.begin(), projected.baseValues.end()),
        projected.baseValues.end());
    projected.compatiblePes.assign(compatiblePes.begin(), compatiblePes.end());
    if (projected.baseValues.empty())
      return invalid("Hall demand group has no compatible context");
    groups.push_back(std::move(projected));
  }

  const auto analyze = [&](llvm::ArrayRef<std::uint64_t> growth)
      -> llvm::Expected<mapping::SpatialComputeContextSupplyAnalysis> {
    std::vector<std::size_t> growthOffsets(growth.size());
    std::size_t valueCount = contextOrdinalByKey.size();
    for (std::size_t pe = 0; pe != growth.size(); ++pe) {
      growthOffsets[pe] = valueCount;
      if (growth[pe] > std::numeric_limits<std::size_t>::max() - valueCount)
        return invalid("joint context growth exceeds size_t");
      valueCount += static_cast<std::size_t>(growth[pe]);
    }
    std::vector<std::vector<std::size_t>> domains;
    if (demandCount > std::numeric_limits<std::size_t>::max())
      return invalid("Hall demand count exceeds size_t");
    domains.reserve(static_cast<std::size_t>(demandCount));
    for (const DemandGroup &group : groups)
      for (std::uint64_t demand = 0; demand != group.count; ++demand) {
        std::vector<std::size_t> domain = group.baseValues;
        for (std::size_t pe : group.compatiblePes)
          for (std::uint64_t added = 0; added != growth[pe]; ++added)
            domain.push_back(growthOffsets[pe] +
                             static_cast<std::size_t>(added));
        llvm::sort(domain);
        domain.erase(std::unique(domain.begin(), domain.end()), domain.end());
        domains.push_back(std::move(domain));
      }
    return mapping::analyzeSpatialComputeContextSupply(domains, valueCount);
  };

  std::vector<std::uint64_t> growth(pes.size(), 0);
  auto initial = analyze(growth);
  if (!initial)
    return initial.takeError();
  mapping::SpatialComputeContextSupplyAnalysis current = std::move(*initial);
  if (current.admissible() ||
      current.maximumMatching != feedback.hallContextValueCount())
    return invalid("Hall feedback disagrees with its reconstructed relation");
  while (!current.admissible()) {
    std::optional<std::size_t> selectedPe;
    std::optional<mapping::SpatialComputeContextSupplyAnalysis> selected;
    for (std::size_t pe = 0; pe != pes.size(); ++pe) {
      if (growth[pe] ==
          std::numeric_limits<std::uint32_t>::max() - pes[pe].currentCapacity)
        continue;
      ++growth[pe];
      auto candidateOrError = analyze(growth);
      --growth[pe];
      if (!candidateOrError)
        return candidateOrError.takeError();
      mapping::SpatialComputeContextSupplyAnalysis candidate =
          std::move(*candidateOrError);
      if (candidate.maximumMatching <= current.maximumMatching)
        continue;
      const std::uint64_t resultingCapacity =
          pes[pe].currentCapacity + growth[pe] + 1;
      const std::uint64_t selectedCapacity =
          selectedPe
              ? pes[*selectedPe].currentCapacity + growth[*selectedPe] + 1
              : std::numeric_limits<std::uint64_t>::max();
      if (!selected || candidate.maximumMatching > selected->maximumMatching ||
          (candidate.maximumMatching == selected->maximumMatching &&
           resultingCapacity < selectedCapacity)) {
        selectedPe = pe;
        selected = std::move(candidate);
      }
    }
    if (!selectedPe || !selected)
      return invalid("no bounded PE growth improves the Hall matching");
    ++growth[*selectedPe];
    current = std::move(*selected);
  }

  TechMappingComputeContextJointGrowthPlan plan;
  for (std::size_t pe = 0; pe != pes.size(); ++pe) {
    if (growth[pe] == 0)
      continue;
    if (growth[pe] >
        std::numeric_limits<std::uint32_t>::max() - pes[pe].currentCapacity)
      return invalid("joint context growth exceeds u32");
    plan.decisions.push_back(
        {pes[pe].pe,
         static_cast<std::uint32_t>(pes[pe].currentCapacity + growth[pe])});
    plan.addedContextCount += growth[pe];
  }
  if (plan.decisions.empty() || plan.addedContextCount != feedback.deficit())
    return invalid("joint context growth is not the minimal Hall closure");
  return plan;
}

} // namespace loom::dse
