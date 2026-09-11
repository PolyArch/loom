#include "DSE/TechMappingHardwareFeedback.h"

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tech_mapping_hardware_feedback_invalid: " +
                                     message);
}

/// Ordered canonical types of one PE occurrence's token inputs. The ADG
/// Builder admits an FU inventory prototype only when the prototype's parent
/// PE presents the target PE exactly the same inputs by ordinal and type, so
/// this signature is the derivable admission key for prototype selection.
std::vector<std::vector<std::uint8_t>>
peInputSignature(const fabric::FabricArtifactView &module,
                 fabric::FabricPeOccurrenceRef pe) {
  const auto owner = fabric::FabricTransportEndpointOwnerRef::of(pe);
  const std::uint64_t endpoints = module.transportEndpointCount(owner);
  std::vector<std::vector<std::uint8_t>> inputs;
  for (fabric::FabricOrdinal ordinal = 0; ordinal != endpoints; ++ordinal) {
    const fabric::FabricTransportEndpointRef endpoint{owner, ordinal};
    if (module.transportEndpointDirection(endpoint) !=
        fabric::FabricPortDirection::Input)
      continue;
    const auto type = module.transportEndpointType(endpoint);
    inputs.emplace_back(type.begin(), type.end());
  }
  return inputs;
}

/// One admissible Spatial FU occurrence growth unit: giving `target` a clone
/// of `prototype` makes every resident context of `target` compatible with
/// the demand groups that admit `capability`.
struct SpatialGrowthUnit final {
  fabric::FabricPeOccurrenceRef target;
  fabric::FabricFuOccurrenceRef prototype;
  fabric::FabricFuCapabilityTemplateRef capability;
  std::vector<std::uint8_t> targetKey;
  std::vector<std::uint8_t> capabilityKey;
  std::vector<std::size_t> contextValues;
  std::vector<std::size_t> groups;
  /// Context values the unit adds that the observed relation does not already
  /// hold. A maximum matching counts values, not their identities, so two
  /// units that add the same number of new values to the same demand groups
  /// reach the same matching.
  std::uint64_t freshValueCount = 0;
};

bool spatialGrowthUnitLess(const SpatialGrowthUnit &lhs,
                           const SpatialGrowthUnit &rhs) {
  if (lhs.targetKey != rhs.targetKey)
    return lhs.targetKey < rhs.targetKey;
  return lhs.capabilityKey < rhs.capabilityKey;
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

namespace {

struct TemporalInstructionStoreClosure final {
  std::vector<ResizeInstructionStore> decisions;
  std::uint64_t addedContextCount = 0;
};

/// Closes the exact observed Hall relation with minimum total new Temporal
/// context capacity. This is the supply of last resort: it lengthens every
/// hosted loop's initiation interval because a Temporal PE issues its
/// residents in rotation.
///
/// An absent result means the observed relation admits no Temporal context
/// supply at all, because no compatible context of any demand group lies on a
/// Temporal PE. That is an ordinary Hall relation over capability classes the
/// Module only realizes spatially, not a malformed observation.
llvm::Expected<std::optional<TemporalInstructionStoreClosure>>
projectTemporalInstructionStoreClosure(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module) {
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
    return std::optional<TemporalInstructionStoreClosure>();

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

  TemporalInstructionStoreClosure closure;
  for (std::size_t pe = 0; pe != pes.size(); ++pe) {
    if (growth[pe] == 0)
      continue;
    if (growth[pe] >
        std::numeric_limits<std::uint32_t>::max() - pes[pe].currentCapacity)
      return invalid("joint context growth exceeds u32");
    closure.decisions.push_back(
        {pes[pe].pe,
         static_cast<std::uint32_t>(pes[pe].currentCapacity + growth[pe])});
    closure.addedContextCount += growth[pe];
  }
  if (closure.decisions.empty() ||
      closure.addedContextCount != feedback.deficit())
    return invalid("joint context growth is not the minimal Hall closure");
  return std::optional<TemporalInstructionStoreClosure>(std::move(closure));
}

struct SpatialFuOccurrenceGrowthStep final {
  std::optional<TechMappingComputeContextSpatialFuGrowth> growth;
  std::uint64_t contextSupplyBound = 0;
};

/// Chooses the one Spatial FU occurrence decision that makes the complete
/// observed relation admissible, and reports how much resident-context supply
/// the exact parent Module's other admissible Spatial PEs hold.
///
/// The microarchitecture vocabulary changes one PE's FU inventory per child
/// Module while a Hall closure child must be atomic, so a partial step would
/// publish a child that is still deficient: its TechMapping covers nothing and
/// the chain would have to repeat one PE at a time. This therefore admits only
/// a single closing decision, and the caller keeps the atomic Temporal closure
/// for every relation no one Spatial PE can close.
///
/// `searchClosure` selects whether the bipartite matchings run at all. The
/// structural enumeration and its bound are cheap and always available; the
/// matchings only run when the caller is actually considering this direction,
/// and then once per equivalence class of units rather than once per unit.
llvm::Expected<SpatialFuOccurrenceGrowthStep>
projectSpatialFuOccurrenceGrowthStep(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module, bool searchClosure) {
  struct CapabilityPrototype final {
    fabric::FabricFuOccurrenceRef fu;
    std::vector<std::vector<std::uint8_t>> parentSignature;
    std::vector<std::uint8_t> key;
  };
  struct CapabilityRecord final {
    fabric::FabricFuCapabilityTemplateRef capability;
    std::vector<std::uint8_t> key;
    std::vector<std::size_t> groups;
    std::vector<CapabilityPrototype> prototypes;
    std::set<std::vector<std::uint8_t>> hosts;
  };
  std::map<std::vector<std::uint8_t>, std::size_t> capabilityOrdinalByKey;
  std::vector<CapabilityRecord> capabilities;
  for (const auto indexed : llvm::enumerate(feedback.groups()))
    for (const fabric::FabricFuCapabilityTemplateRef capability :
         indexed.value().capabilities) {
      std::vector<std::uint8_t> key = fabric::canonicalFabricBytes(capability);
      const auto found =
          capabilityOrdinalByKey.emplace(key, capabilities.size());
      if (found.second)
        capabilities.push_back({capability, std::move(key), {}, {}, {}});
      capabilities[found.first->second].groups.push_back(indexed.index());
    }

  // Values the observed relation already offers. A Spatial PE that hosts one
  // of these contexts adds no supply beyond the capability it gains.
  std::map<std::vector<std::uint8_t>, std::size_t> contextOrdinalByKey;
  for (const auto &group : feedback.groups())
    for (const fabric::InstructionContextRef context : group.compatibleContexts)
      contextOrdinalByKey.emplace(fabric::canonicalFabricBytes(context), 0);
  const std::size_t baseValueCount = contextOrdinalByKey.size();

  std::map<std::vector<std::uint8_t>, std::vector<std::vector<std::uint8_t>>>
      signatureByPe;
  const auto signatureOf =
      [&](fabric::FabricPeOccurrenceRef pe)
      -> const std::vector<std::vector<std::uint8_t>> & {
    std::vector<std::uint8_t> key = fabric::canonicalFabricBytes(pe);
    const auto found = signatureByPe.find(key);
    if (found != signatureByPe.end())
      return found->second;
    return signatureByPe
        .emplace(std::move(key), peInputSignature(module, pe))
        .first->second;
  };

  using FuOccurrenceInventory = std::vector<fabric::FabricFuOccurrenceRef>;
  std::map<std::vector<std::uint8_t>, FuOccurrenceInventory> inventoryByPe;
  for (const fabric::FabricFuOccurrenceRef fu : module.fuOccurrences()) {
    const auto parent = module.parentPeOf(fu);
    if (!parent)
      return invalid("a Fabric FU occurrence has no parent PE relation");
    std::vector<std::uint8_t> parentKey = fabric::canonicalFabricBytes(*parent);
    inventoryByPe[parentKey].push_back(fu);
    const auto definition = module.fuTemplateOf(fu);
    if (!definition)
      continue;
    for (CapabilityRecord &record : capabilities) {
      if (record.capability.fu != *definition)
        continue;
      record.hosts.insert(parentKey);
      record.prototypes.push_back(
          {fu, signatureOf(*parent), fabric::canonicalFabricBytes(fu)});
    }
  }

  SpatialFuOccurrenceGrowthStep step;
  std::vector<SpatialGrowthUnit> units;
  std::map<std::vector<std::uint8_t>, std::uint64_t> contextsByTarget;
  for (const fabric::FabricPeOccurrenceRef pe : module.peOccurrences()) {
    const auto schedule = module.peSchedule(pe);
    if (!schedule)
      return invalid("a Fabric PE occurrence has no scheduling contract");
    if (*schedule != ::fabric::Schedule::Spatial)
      continue;
    const std::uint64_t contexts = module.peResidentContextCount(pe);
    if (contexts == 0)
      continue;
    std::vector<std::uint8_t> peKey = fabric::canonicalFabricBytes(pe);
    const auto inventory = inventoryByPe.find(peKey);
    if (inventory == inventoryByPe.end() || inventory->second.empty())
      continue;
    const std::vector<std::vector<std::uint8_t>> &signature = signatureOf(pe);
    for (const CapabilityRecord &record : capabilities) {
      if (record.hosts.count(peKey) != 0)
        continue;
      const CapabilityPrototype *prototype = nullptr;
      for (const CapabilityPrototype &candidate : record.prototypes) {
        if (candidate.parentSignature != signature)
          continue;
        if (!prototype || candidate.key < prototype->key)
          prototype = &candidate;
      }
      if (!prototype)
        continue;
      units.push_back({pe, prototype->fu, record.capability, peKey, record.key,
                       {}, record.groups, 0});
      contextsByTarget.emplace(peKey, contexts);
    }
  }
  const auto publishBound =
      [&](const std::vector<std::uint8_t> *selectedTarget) -> llvm::Error {
    for (const auto &target : contextsByTarget) {
      if (selectedTarget && target.first == *selectedTarget)
        continue;
      if (target.second >
          std::numeric_limits<std::uint64_t>::max() - step.contextSupplyBound)
        return invalid("Spatial FU context supply bound overflows u64");
      step.contextSupplyBound += target.second;
    }
    return llvm::Error::success();
  };
  if (units.empty() || !searchClosure) {
    if (llvm::Error error = publishBound(nullptr))
      return std::move(error);
    return step;
  }
  llvm::sort(units, spatialGrowthUnitLess);

  std::set<std::vector<std::uint8_t>> baseContextKeys;
  for (const auto &entry : contextOrdinalByKey)
    baseContextKeys.insert(entry.first);
  for (SpatialGrowthUnit &unit : units) {
    const std::uint64_t contexts = module.peResidentContextCount(unit.target);
    for (std::uint64_t ordinal = 0; ordinal != contexts; ++ordinal) {
      std::vector<std::uint8_t> key = fabric::canonicalFabricBytes(
          fabric::InstructionContextRef{unit.target, ordinal});
      unit.freshValueCount += baseContextKeys.count(key) == 0;
      contextOrdinalByKey.emplace(std::move(key), 0);
    }
  }
  std::size_t nextValue = 0;
  for (auto &entry : contextOrdinalByKey)
    entry.second = nextValue++;
  for (SpatialGrowthUnit &unit : units) {
    const std::uint64_t contexts = module.peResidentContextCount(unit.target);
    for (std::uint64_t ordinal = 0; ordinal != contexts; ++ordinal)
      unit.contextValues.push_back(contextOrdinalByKey.at(
          fabric::canonicalFabricBytes(
              fabric::InstructionContextRef{unit.target, ordinal})));
    llvm::sort(unit.contextValues);
  }

  struct DemandGroup final {
    std::uint64_t count = 0;
    std::vector<std::size_t> baseValues;
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
    for (const fabric::InstructionContextRef context :
         group.compatibleContexts) {
      const auto value =
          contextOrdinalByKey.find(fabric::canonicalFabricBytes(context));
      if (value == contextOrdinalByKey.end())
        return invalid("Hall feedback projection lost a context value");
      projected.baseValues.push_back(value->second);
    }
    llvm::sort(projected.baseValues);
    projected.baseValues.erase(
        std::unique(projected.baseValues.begin(), projected.baseValues.end()),
        projected.baseValues.end());
    if (projected.baseValues.empty())
      return invalid("Hall demand group has no compatible context");
    groups.push_back(std::move(projected));
  }
  if (demandCount > std::numeric_limits<std::size_t>::max())
    return invalid("Hall demand count exceeds size_t");

  const auto analyze = [&](const SpatialGrowthUnit *applied)
      -> llvm::Expected<mapping::SpatialComputeContextSupplyAnalysis> {
    std::vector<std::vector<std::size_t>> domains;
    domains.reserve(static_cast<std::size_t>(demandCount));
    for (const auto indexed : llvm::enumerate(groups)) {
      std::vector<std::size_t> domain = indexed.value().baseValues;
      if (applied && llvm::is_contained(applied->groups, indexed.index())) {
        domain.insert(domain.end(), applied->contextValues.begin(),
                      applied->contextValues.end());
        llvm::sort(domain);
        domain.erase(std::unique(domain.begin(), domain.end()), domain.end());
      }
      for (std::uint64_t demand = 0; demand != indexed.value().count; ++demand)
        domains.push_back(domain);
    }
    return mapping::analyzeSpatialComputeContextSupply(
        domains, contextOrdinalByKey.size());
  };

  auto base = analyze(nullptr);
  if (!base)
    return base.takeError();
  if (base->admissible() ||
      base->maximumMatching != feedback.hallContextValueCount() ||
      baseValueCount != feedback.hallContextValueCount())
    return invalid("Hall feedback disagrees with its reconstructed relation");

  // A new edge can only enlarge a maximum matching when its demand side lies
  // on the alternating structure the Hall witness already reached, so a unit
  // that touches no witness group cannot close the relation. Skipping those
  // units is exact, not heuristic, and keeps the search linear in the witness.
  std::vector<std::uint8_t> witnessGroups(groups.size(), 0);
  std::vector<std::size_t> groupOfDemand;
  groupOfDemand.reserve(static_cast<std::size_t>(demandCount));
  for (const auto indexed : llvm::enumerate(groups))
    for (std::uint64_t demand = 0; demand != indexed.value().count; ++demand)
      groupOfDemand.push_back(indexed.index());
  for (const std::uint64_t demand : base->hallDemands) {
    if (demand >= groupOfDemand.size())
      return invalid("Hall witness names a demand outside its relation");
    witnessGroups[groupOfDemand[static_cast<std::size_t>(demand)]] = 1;
  }

  // Units that add the same number of new values to the same demand groups
  // are interchangeable for the matching, so one representative decides the
  // whole class. Units are in canonical order, so the representative is the
  // canonically least member and the selected decision is unchanged; only the
  // number of matchings drops from one per admissible PE and capability to one
  // per class.
  std::set<std::pair<std::vector<std::size_t>, std::uint64_t>> evaluatedClasses;
  const SpatialGrowthUnit *selected = nullptr;
  for (const SpatialGrowthUnit &unit : units) {
    if (llvm::none_of(unit.groups, [&](std::size_t group) {
          return witnessGroups[group] != 0;
        }))
      continue;
    if (!evaluatedClasses.emplace(unit.groups, unit.freshValueCount).second)
      continue;
    auto candidate = analyze(&unit);
    if (!candidate)
      return candidate.takeError();
    if (!candidate->admissible())
      continue;
    selected = &unit;
    break;
  }
  if (!selected) {
    if (llvm::Error error = publishBound(nullptr))
      return std::move(error);
    return step;
  }

  if (llvm::Error error = publishBound(&selected->targetKey))
    return std::move(error);
  std::vector<fabric::FabricFuOccurrenceRef> prototypes =
      inventoryByPe.at(selected->targetKey);
  prototypes.push_back(selected->prototype);
  step.growth = TechMappingComputeContextSpatialFuGrowth{
      ChangeFuInventory{selected->target, std::move(prototypes)},
      selected->capability, contextsByTarget.at(selected->targetKey)};
  return step;
}

} // namespace

llvm::StringRef techMappingComputeContextGrowthDirectionSpelling(
    TechMappingComputeContextGrowthDirection direction) {
  switch (direction) {
  case TechMappingComputeContextGrowthDirection::SpatialFuOccurrence:
    return "spatial_fu_occurrence";
  case TechMappingComputeContextGrowthDirection::TemporalInstructionStore:
    return "temporal_instruction_store";
  }
  llvm_unreachable("unknown compute-context growth direction");
}

llvm::Expected<std::optional<TechMappingComputeContextJointGrowthPlan>>
projectTechMappingComputeContextJointGrowthPlan(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module,
    bool preferTemporalInstructionStore) {
  if (module.rootKind() != fabric::FabricRootKind::Module)
    return invalid("hardware feedback target is not a Module");
  if (feedback.deficit() == 0)
    return invalid("compute-context feedback has no positive deficit");

  TechMappingComputeContextJointGrowthPlan plan;
  const auto adoptSpatialStep = [&](SpatialFuOccurrenceGrowthStep &step) {
    plan.spatialFuContextSupplyBound = step.contextSupplyBound;
    plan.spatialFuUnclosedDeficit =
        feedback.deficit() > step.contextSupplyBound
            ? feedback.deficit() - step.contextSupplyBound
            : 0;
    if (!step.growth)
      return false;
    plan.direction =
        TechMappingComputeContextGrowthDirection::SpatialFuOccurrence;
    plan.addedContextCount = step.growth->addedContextCount;
    plan.spatialFuGrowth = std::move(step.growth);
    return true;
  };

  auto spatial = projectSpatialFuOccurrenceGrowthStep(
      feedback, module, !preferTemporalInstructionStore);
  if (!spatial)
    return spatial.takeError();
  if (adoptSpatialStep(*spatial))
    return std::optional<TechMappingComputeContextJointGrowthPlan>(
        std::move(plan));

  auto temporal = projectTemporalInstructionStoreClosure(feedback, module);
  if (!temporal)
    return temporal.takeError();
  if (*temporal) {
    plan.direction =
        TechMappingComputeContextGrowthDirection::TemporalInstructionStore;
    plan.decisions = std::move((*temporal)->decisions);
    plan.addedContextCount = (*temporal)->addedContextCount;
    return std::optional<TechMappingComputeContextJointGrowthPlan>(
        std::move(plan));
  }

  // The relation admits no Temporal context supply, so the preference has
  // nothing to prefer: the Spatial FU occurrence direction is the only
  // compatible supply and the owner searches it even when the chain has not
  // withdrawn the preference.
  if (preferTemporalInstructionStore) {
    auto only = projectSpatialFuOccurrenceGrowthStep(feedback, module, true);
    if (!only)
      return only.takeError();
    if (adoptSpatialStep(*only))
      return std::optional<TechMappingComputeContextJointGrowthPlan>(
          std::move(plan));
  }
  return std::optional<TechMappingComputeContextJointGrowthPlan>();
}

} // namespace loom::dse
