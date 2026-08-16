#include "SystemPnrDerivedContextInternal.h"

#include "../SpatialPhysicalTiming.h"

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SystemPnrFreezeFailure>(
      SystemPnrFreezeFailureKind::Invalid, message.str());
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendBytes(std::vector<std::uint8_t> &preimage,
                 llvm::ArrayRef<std::uint8_t> bytes) {
  appendU64Be(preimage, bytes.size());
  preimage.insert(preimage.end(), bytes.begin(), bytes.end());
}

void appendText(std::vector<std::uint8_t> &preimage, llvm::StringRef text) {
  appendBytes(preimage, llvm::ArrayRef<std::uint8_t>(
                            reinterpret_cast<const std::uint8_t *>(text.data()),
                            text.size()));
}

std::array<std::uint8_t, 32>
deriveSystemStaticContextKey(const FabricSystemRootView &system) {
  std::vector<std::uint8_t> preimage;
  appendText(preimage, loom::pnr::detail::systemStaticContextAlgorithmIdentity);
  appendText(preimage, fabricArtifactSchema.identity);
  appendU32Be(preimage, fabricArtifactSchema.version.major);
  appendU32Be(preimage, fabricArtifactSchema.version.minor);
  appendU32Be(preimage,
              static_cast<std::uint32_t>(system.artifact().rootKind()));
  appendBytes(preimage, system.artifact().identity().bytes());
  appendU32Be(preimage, sizeof(PnrIndex) * 8);
  return llvm::SHA256::hash(preimage);
}

std::vector<std::vector<std::uint8_t>>
canonicalTimingKeys(llvm::ArrayRef<FabricPhysicalTimingProfileView> profiles) {
  std::vector<std::vector<std::uint8_t>> keys;
  keys.reserve(profiles.size());
  for (const FabricPhysicalTimingProfileView &profile : profiles) {
    std::vector<std::uint8_t> key;
    appendBytes(key, profile.fabricIdentity().bytes());
    appendBytes(key, profile.schemaDescriptorBytes());
    appendBytes(key, profile.digest().bytes());
    appendU32Be(key, static_cast<std::uint32_t>(profile.kind()));
    keys.push_back(std::move(key));
  }
  llvm::sort(keys);
  return keys;
}

std::array<std::uint8_t, 32> deriveSystemActiveContextKey(
    const loom::pnr::detail::SystemStaticContextStorage &staticContext,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<FabricPhysicalTimingProfileView> profiles,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings) {
  std::vector<std::uint8_t> preimage;
  appendText(preimage, loom::pnr::detail::systemActiveContextAlgorithmIdentity);
  appendBytes(preimage, staticContext.key);
  appendText(preimage, ::dataflow::canonicalDataflowSchema.identity);
  appendU32Be(preimage, ::dataflow::canonicalDataflowSchema.version.major);
  appendU32Be(preimage, ::dataflow::canonicalDataflowSchema.version.minor);
  appendBytes(preimage, dataflow.identity().bytes());
  appendBytes(preimage, encodeArtifactRootReference(constraints.reference()));
  appendText(preimage, ::loom::mapping::mappingArtifactSchema.identity);
  appendU32Be(preimage, ::loom::mapping::mappingArtifactSchema.version.major);
  appendU32Be(preimage, ::loom::mapping::mappingArtifactSchema.version.minor);
  appendU32Be(preimage, static_cast<std::uint32_t>(spatialMappings.size()));
  for (const ArtifactRootReference &reference : spatialMappings)
    appendBytes(preimage, encodeArtifactRootReference(reference));
  appendText(preimage, fabricPhysicalTimingProfileArtifactSchema.identity);
  appendU32Be(preimage,
              fabricPhysicalTimingProfileArtifactSchema.version.major);
  appendU32Be(preimage,
              fabricPhysicalTimingProfileArtifactSchema.version.minor);
  const auto timingKeys = canonicalTimingKeys(profiles);
  appendU32Be(preimage, static_cast<std::uint32_t>(timingKeys.size()));
  for (const auto &key : timingKeys)
    appendBytes(preimage, key);
  return llvm::SHA256::hash(preimage);
}

std::string bytesKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::vector<FabricUsePatternRef>
patternRefs(FabricInventoryOwnerRef owner,
            const ::fabric::ResourceContract &contract) {
  std::vector<FabricUsePatternRef> result;
  result.reserve(contract.usePatternCount());
  const auto patternOwner = FabricUsePatternOwnerRef(owner);
  for (std::uint32_t ordinal = 0; ordinal < contract.usePatternCount();
       ++ordinal)
    result.push_back({patternOwner, ordinal});
  return result;
}

llvm::Expected<std::vector<FrozenSystemInstructionUsePatternDomain>>
buildInstructionUsePatterns(const FabricSystemRootView &system,
                            llvm::ArrayRef<AccCoreOccurrenceRef> cores) {
  std::vector<FrozenSystemInstructionUsePatternDomain> result;
  result.reserve(cores.size());
  for (AccCoreOccurrenceRef core : cores) {
    const InstructionCoreContextRef context{core};
    const auto *microarchitecture =
        system.instructionCoreMicroarchitecture(context);
    if (!microarchitecture)
      return invalid("AccCore has no InstructionCore microarchitecture");
    auto patterns = patternRefs(FabricInventoryOwnerRef::of(context),
                                microarchitecture->resourceContract());
    if (patterns.empty())
      return invalid("InstructionCore exposes no occupancy use pattern");
    result.push_back({context, std::move(patterns)});
  }
  return result;
}

std::vector<FrozenSystemConsistencyUsePatternDomain>
buildConsistencyUsePatterns(const FabricSystemRootView &system) {
  std::vector<FrozenSystemConsistencyUsePatternDomain> result;
  for (const auto domain : system.hardwareDomains()) {
    const auto *record = system.hardwareDomainContract(domain);
    if (!record)
      continue;
    const auto *consistency =
        std::get_if<::fabric::MemoryConsistencyContract>(&record->contract());
    if (!consistency)
      continue;
    result.push_back({MemoryConsistencyDomainRef(domain),
                      patternRefs(FabricInventoryOwnerRef::of(domain),
                                  consistency->resourceContract())});
  }
  return result;
}

std::uint64_t
retainedTopologyBytes(const FrozenEndpointRoutingTopology &topology) {
  return sizeof(topology) +
         topology.endpoints().size() * sizeof(EndpointRoutingEndpoint) +
         topology.traversals().size() * sizeof(EndpointRoutingTraversal) +
         topology.traversalEndpoints().size() * sizeof(PnrIndex) +
         topology.traversalReplicationGroups().size() * sizeof(PnrIndex) +
         topology.arcs().size() * sizeof(EndpointRoutingArc) +
         topology.arcSources().size() * sizeof(PnrIndex) +
         topology.adjacencyOffsets().size() * sizeof(PnrIndex) +
         topology.reverseAdjacencyOffsets().size() * sizeof(PnrIndex) +
         topology.reverseArcOrdinals().size() * sizeof(PnrIndex) +
         topology.capacityCells().size() * sizeof(EndpointRoutingCapacityCell) +
         topology.capacityClaims().size() *
             sizeof(EndpointRoutingCapacityClaim);
}

template <typename Domain>
std::uint64_t patternDomainBytes(llvm::ArrayRef<Domain> domains) {
  std::uint64_t bytes = domains.size() * sizeof(Domain);
  for (const Domain &domain : domains)
    bytes += domain.patterns.size() * sizeof(FabricUsePatternRef);
  return bytes;
}

std::uint64_t retainedSpatialCatalogBytes(
    llvm::ArrayRef<loom::pnr::detail::SpatialCatalogEntry> catalog) {
  std::uint64_t bytes =
      catalog.size() * sizeof(loom::pnr::detail::SpatialCatalogEntry);
  for (const auto &entry : catalog) {
    bytes += entry.covers.size() * sizeof(::dataflow::GraphRef);
    bytes += entry.graphProgress.size() *
             sizeof(loom::pnr::detail::SpatialCatalogGraphProgress);
    for (const auto &progress : entry.graphProgress)
      bytes +=
          progress.routeObligations.size() *
          sizeof(::loom::mapping::MappingRouteProgressObligationProjection);
    bytes += entry.graphStaticSchedulePressures.size() * sizeof(std::uint64_t);
    bytes += entry.graphRecurrenceTimings.size() *
             sizeof(SpatialRecurrenceTimingProjection);
  }
  return bytes;
}

struct ValidatedSpatialCatalog final {
  std::vector<ArtifactRootReference> canonicalMappings;
  ::loom::mapping::SpatialMappingImportContext imports;
  std::vector<loom::pnr::detail::SpatialCatalogEntry> catalog;
  loom::pnr::detail::SpatialCatalogImportStatistics statistics;
};

llvm::Expected<ValidatedSpatialCatalog> buildValidatedSpatialCatalog(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const FabricSystemRootView &system,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store) {
  std::vector<ArtifactRootReference> canonicalMappings(spatialMappings.begin(),
                                                       spatialMappings.end());
  llvm::sort(canonicalMappings, artifactRootReferenceLess);
  if (std::adjacent_find(canonicalMappings.begin(), canonicalMappings.end()) !=
      canonicalMappings.end())
    return invalid("System SpatialMapping set has a duplicate");

  auto imports = ::loom::mapping::buildSpatialMappingImportContext(
      canonicalMappings, store);
  if (!imports)
    return imports.takeError();
  loom::pnr::detail::SpatialCatalogImportStatistics statistics;
  auto catalog = loom::pnr::detail::importSpatialCatalog(
      canonicalMappings, dataflow, system, store, &*imports, &statistics);
  if (!catalog)
    return catalog.takeError();
  if (statistics.techMappingImportRequests != canonicalMappings.size() ||
      statistics.techMappingImportHits + statistics.techMappingImportMisses !=
          statistics.techMappingImportRequests)
    return invalid("System SpatialMapping TechMapping import accounting is "
                   "inconsistent");
  return ValidatedSpatialCatalog{std::move(canonicalMappings),
                                 std::move(*imports), std::move(*catalog),
                                 statistics};
}

llvm::Error addWork(std::uint64_t &work, std::uint64_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - work)
    return invalid("SystemStaticContext deterministic work overflows u64");
  work += amount;
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FrozenSystemSpatialTargetClass>
loom::pnr::detail::deriveSystemSpatialTargetClass(
    const FabricArtifactView &module) {
  auto root = module.moduleRootTemplate();
  if (module.rootKind() != FabricRootKind::Module || !root)
    return invalid("System SpatialCore dependency is not an exact Module root");
  return FrozenSystemSpatialTargetClass{module.identity(), *root};
}

std::string loom::pnr::detail::systemSpatialTargetClassKey(
    const FrozenSystemSpatialTargetClass &targetClass) {
  std::string key = bytesKey(targetClass.moduleIdentity.bytes());
  const auto moduleBytes = canonicalFabricBytes(targetClass.moduleTemplate);
  key.append(reinterpret_cast<const char *>(moduleBytes.data()),
             moduleBytes.size());
  return key;
}

const loom::pnr::detail::SystemStaticContextStorage &
loom::pnr::detail::systemStaticContextStorage(
    const SystemStaticContext &context) {
  return *context.storage_;
}

const loom::pnr::detail::SystemActiveContextStorage &
loom::pnr::detail::systemActiveContextStorage(
    const SystemActiveContext &context) {
  return *context.storage_;
}

const ArtifactIdentity &SystemStaticContext::systemIdentity() const {
  return storage_->systemIdentity;
}

const SystemStaticContextStatistics &SystemStaticContext::statistics() const {
  return storage_->statistics;
}

const ArtifactIdentity &SystemActiveContext::dataflowIdentity() const {
  return storage_->dataflowIdentity;
}

const ArtifactIdentity &SystemActiveContext::systemIdentity() const {
  return storage_->systemIdentity;
}

const ArtifactIdentity &SystemActiveContext::constraintIdentity() const {
  return storage_->constraintIdentity;
}

llvm::ArrayRef<ArtifactRootReference>
SystemActiveContext::spatialMappings() const {
  return storage_->spatialMappings;
}

const SystemActiveContextStatistics &SystemActiveContext::statistics() const {
  return storage_->statistics;
}

llvm::Expected<SystemStaticContext>
loom::pnr::buildSystemStaticContext(const FabricSystemRootView &system) {
  if (system.artifact().rootKind() != FabricRootKind::System)
    return invalid("SystemStaticContext requires one System root");
  const auto begin = std::chrono::steady_clock::now();

  std::vector<AccCoreOccurrenceRef> cores(
      system.artifact().accCoreOccurrences().begin(),
      system.artifact().accCoreOccurrences().end());
  llvm::sort(cores, [](AccCoreOccurrenceRef lhs, AccCoreOccurrenceRef rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  std::vector<FrozenSystemSpatialTargetClass> perCoreClasses;
  perCoreClasses.reserve(cores.size());
  for (AccCoreOccurrenceRef core : cores) {
    auto target = system.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= system.artifact().importedModules().size())
      return invalid("AccCore has no exact imported SpatialCore target");
    const FabricArtifactView &module =
        system.artifact().importedModules()[target->dependencyOrdinal];
    auto targetClass =
        loom::pnr::detail::deriveSystemSpatialTargetClass(module);
    if (!targetClass)
      return targetClass.takeError();
    if (targetClass->moduleTemplate != target->target)
      return invalid(
          "AccCore SpatialCore target disagrees with its Module root");
    perCoreClasses.push_back(std::move(*targetClass));
  }

  std::vector<FrozenSystemSpatialTargetClass> targetClasses = perCoreClasses;
  llvm::sort(targetClasses, [](const auto &lhs, const auto &rhs) {
    return loom::pnr::detail::systemSpatialTargetClassKey(lhs) <
           loom::pnr::detail::systemSpatialTargetClassKey(rhs);
  });
  targetClasses.erase(
      std::unique(targetClasses.begin(), targetClasses.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return loom::pnr::detail::systemSpatialTargetClassKey(
                               lhs) ==
                           loom::pnr::detail::systemSpatialTargetClassKey(rhs);
                  }),
      targetClasses.end());
  if (targetClasses.size() > std::numeric_limits<PnrIndex>::max())
    return invalid("System target-class inventory exceeds PnrIndex");
  std::map<std::string, PnrIndex> classOrdinals;
  for (const auto &[ordinal, targetClass] : llvm::enumerate(targetClasses))
    classOrdinals.emplace(
        loom::pnr::detail::systemSpatialTargetClassKey(targetClass),
        static_cast<PnrIndex>(ordinal));
  std::vector<PnrIndex> coreTargetClasses;
  coreTargetClasses.reserve(perCoreClasses.size());
  for (const auto &targetClass : perCoreClasses)
    coreTargetClasses.push_back(classOrdinals.at(
        loom::pnr::detail::systemSpatialTargetClassKey(targetClass)));

  auto topology = freezeEndpointRoutingTopology(system.artifact());
  if (!topology)
    return topology.takeError();
  auto instructionPatterns = buildInstructionUsePatterns(system, cores);
  if (!instructionPatterns)
    return instructionPatterns.takeError();
  auto consistencyPatterns = buildConsistencyUsePatterns(system);

  auto topologyOwner = std::make_shared<const FrozenEndpointRoutingTopology>(
      std::move(*topology));
  auto targetClassOwner =
      std::make_shared<const std::vector<FrozenSystemSpatialTargetClass>>(
          std::move(targetClasses));
  auto coreOwner = std::make_shared<const std::vector<AccCoreOccurrenceRef>>(
      std::move(cores));
  auto coreClassOwner = std::make_shared<const std::vector<PnrIndex>>(
      std::move(coreTargetClasses));
  auto instructionOwner = std::make_shared<
      const std::vector<FrozenSystemInstructionUsePatternDomain>>(
      std::move(*instructionPatterns));
  auto consistencyOwner = std::make_shared<
      const std::vector<FrozenSystemConsistencyUsePatternDomain>>(
      std::move(consistencyPatterns));

  SystemStaticContextStatistics statistics;
  statistics.context.constructionCount = 1;
  statistics.context.constructionNanoseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
  statistics.accCoreCount = coreOwner->size();
  statistics.targetClassCount = targetClassOwner->size();
  statistics.endpointCount = topologyOwner->endpoints().size();
  statistics.traversalCount = topologyOwner->traversals().size();
  statistics.routingArcCount = topologyOwner->arcs().size();
  for (const auto &domain : *instructionOwner)
    statistics.instructionUsePatternCount += domain.patterns.size();
  for (const auto &domain : *consistencyOwner)
    statistics.consistencyUsePatternCount += domain.patterns.size();
  statistics.context.retainedBytes =
      sizeof(loom::pnr::detail::SystemStaticContextStorage) +
      retainedTopologyBytes(*topologyOwner) +
      targetClassOwner->size() * sizeof(FrozenSystemSpatialTargetClass) +
      coreOwner->size() * sizeof(AccCoreOccurrenceRef) +
      coreClassOwner->size() * sizeof(PnrIndex) +
      patternDomainBytes<FrozenSystemInstructionUsePatternDomain>(
          *instructionOwner) +
      patternDomainBytes<FrozenSystemConsistencyUsePatternDomain>(
          *consistencyOwner);
  if (llvm::Error error = addWork(statistics.context.deterministicWork,
                                  statistics.accCoreCount))
    return std::move(error);
  if (llvm::Error error = addWork(statistics.context.deterministicWork,
                                  statistics.targetClassCount))
    return std::move(error);
  if (llvm::Error error = addWork(statistics.context.deterministicWork,
                                  statistics.endpointCount))
    return std::move(error);
  if (llvm::Error error = addWork(statistics.context.deterministicWork,
                                  statistics.traversalCount))
    return std::move(error);
  if (llvm::Error error = addWork(statistics.context.deterministicWork,
                                  statistics.routingArcCount))
    return std::move(error);
  if (llvm::Error error = addWork(statistics.context.deterministicWork,
                                  statistics.instructionUsePatternCount +
                                      statistics.consistencyUsePatternCount))
    return std::move(error);

  auto storage =
      std::make_shared<const loom::pnr::detail::SystemStaticContextStorage>(
          loom::pnr::detail::SystemStaticContextStorage{
              deriveSystemStaticContextKey(system),
              system.artifact().identity(), std::move(topologyOwner),
              std::move(targetClassOwner), std::move(coreOwner),
              std::move(coreClassOwner), std::move(instructionOwner),
              std::move(consistencyOwner), statistics});
  return SystemStaticContext(std::move(storage));
}

llvm::Error
loom::pnr::revalidateSystemStaticContext(const SystemStaticContext &context,
                                         const FabricSystemRootView &system) {
  if (!context.storage_ || !context.storage_->routingTopology ||
      !context.storage_->targetClasses || !context.storage_->accCores ||
      !context.storage_->accCoreTargetClasses ||
      !context.storage_->instructionUsePatterns ||
      !context.storage_->consistencyUsePatterns)
    return invalid("SystemStaticContext is incomplete");
  if (context.storage_->systemIdentity != system.artifact().identity() ||
      context.storage_->key != deriveSystemStaticContextKey(system))
    return invalid("SystemStaticContext key does not match its System input");
  return llvm::Error::success();
}

void loom::pnr::emitSystemStaticContextStatistics(
    const SystemStaticContext &context, mapping_debug::Stage stage,
    std::uint64_t hits, std::uint64_t misses) {
  const SystemStaticContextStatistics &statistics = context.statistics();
  mapping_debug::emit(
      mapping_debug::Level::Summary, stage,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "system_static";
        fields["cache_hits"] = hits;
        fields["cache_misses"] = misses;
        fields["construction_count"] = statistics.context.constructionCount;
        fields["construction_time_ns"] =
            statistics.context.constructionNanoseconds;
        fields["retained_bytes"] = statistics.context.retainedBytes;
        fields["deterministic_work"] = statistics.context.deterministicWork;
        fields["acc_core_count"] = statistics.accCoreCount;
        fields["target_class_count"] = statistics.targetClassCount;
        fields["endpoint_count"] = statistics.endpointCount;
        fields["traversal_count"] = statistics.traversalCount;
        fields["routing_arc_count"] = statistics.routingArcCount;
        fields["instruction_use_pattern_count"] =
            statistics.instructionUsePatternCount;
        fields["consistency_use_pattern_count"] =
            statistics.consistencyUsePatternCount;
      });
}

llvm::Error loom::pnr::validateSystemSpatialMappingSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const FabricSystemRootView &system,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store) {
  auto catalog =
      buildValidatedSpatialCatalog(dataflow, system, spatialMappings, store);
  if (!catalog)
    return catalog.takeError();
  return llvm::Error::success();
}

llvm::Expected<SystemActiveContext> loom::pnr::buildSystemActiveContext(
    const SystemStaticContext &staticContext,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const FabricSystemRootView &system,
    llvm::ArrayRef<FabricPhysicalTimingProfileView> physicalTimingProfiles,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store) {
  if (llvm::Error error = revalidateSystemStaticContext(staticContext, system))
    return std::move(error);
  if (constraints.view().dataflowIdentity() != dataflow.identity() ||
      constraints.view().fabricIdentity() != system.artifact().identity())
    return invalid("SystemActiveContext constraint owners do not match D/F");

  for (const ArtifactRootReference &required :
       constraints.view().spatialMappingReferences())
    if (!llvm::is_contained(spatialMappings, required))
      return invalid(
          "SystemActiveContext omits a constraint-owned SpatialMapping");

  const auto begin = std::chrono::steady_clock::now();
  auto validated =
      buildValidatedSpatialCatalog(dataflow, system, spatialMappings, store);
  if (!validated)
    return validated.takeError();
  auto &canonicalMappings = validated->canonicalMappings;
  auto &catalog = validated->catalog;

  std::map<ArtifactIdentity::Storage, const FabricPhysicalTimingProfileView *>
      timingByModule;
  std::map<ArtifactIdentity::Storage, const FabricArtifactView *>
      attachedModules;
  for (AccCoreOccurrenceRef core : system.artifact().accCoreOccurrences()) {
    const auto target = system.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= system.artifact().importedModules().size())
      return invalid("SystemActiveContext AccCore timing target does not "
                     "resolve");
    const FabricArtifactView &module =
        system.artifact().importedModules()[target->dependencyOrdinal];
    attachedModules.emplace(module.identity().bytes(), &module);
  }
  for (const FabricPhysicalTimingProfileView &profile :
       physicalTimingProfiles) {
    const auto module = attachedModules.find(profile.fabricIdentity().bytes());
    if (module == attachedModules.end())
      return invalid("SystemActiveContext timing profile targets an "
                     "unattached Module");
    if (!timingByModule.emplace(profile.fabricIdentity().bytes(), &profile)
             .second)
      return invalid("SystemActiveContext has multiple timing profiles for "
                     "one Module");
    if (llvm::Error error =
            validateFabricPhysicalTimingProfile(*module->second, profile))
      return std::move(error);
  }
  if (timingByModule.size() != attachedModules.size())
    return invalid("SystemActiveContext omits an attached Module timing "
                   "profile");

  const auto &staticStorage =
      loom::pnr::detail::systemStaticContextStorage(staticContext);
  std::map<std::string, PnrIndex> classOrdinals;
  for (const auto &[ordinal, targetClass] :
       llvm::enumerate(*staticStorage.targetClasses)) {
    if (ordinal > std::numeric_limits<PnrIndex>::max())
      return invalid("SystemActiveContext target-class index overflows "
                     "PnrIndex");
    classOrdinals.emplace(
        loom::pnr::detail::systemSpatialTargetClassKey(targetClass),
        static_cast<PnrIndex>(ordinal));
  }
  std::vector<PnrIndex> mappingTargetClasses;
  mappingTargetClasses.reserve(catalog.size());
  for (loom::pnr::detail::SpatialCatalogEntry &entry : catalog) {
    const auto timing =
        timingByModule.find(entry.mapping->view().fabricIdentity().bytes());
    if (timing == timingByModule.end())
      return invalid("SystemActiveContext SpatialMapping has no exact timing "
                     "profile");
    auto projected = loom::pnr::detail::projectSpatialMappingPhysicalTiming(
        entry.mapping->view(), *timing->second);
    if (!projected)
      return projected.takeError();
    entry.worstRouteArrivalDelayQuanta = projected->worstArrivalDelayQuanta;
    entry.totalRouteNegativeSlackQuanta = projected->totalNegativeSlackQuanta;
    entry.physicalTimingProfileDigest = timing->second->digest().bytes();
    entry.physicalTimingProfileKind = timing->second->kind();

    const FabricArtifactView *module = nullptr;
    for (const FabricArtifactView &candidate :
         system.artifact().importedModules())
      if (candidate.identity() == entry.mapping->view().fabricIdentity()) {
        module = &candidate;
        break;
      }
    if (!module)
      return invalid("SystemActiveContext SpatialMapping Module is not "
                     "imported by System");
    auto targetClass =
        loom::pnr::detail::deriveSystemSpatialTargetClass(*module);
    if (!targetClass)
      return targetClass.takeError();
    const auto found = classOrdinals.find(
        loom::pnr::detail::systemSpatialTargetClassKey(*targetClass));
    if (found == classOrdinals.end())
      return invalid("SystemActiveContext SpatialMapping target class is not "
                     "attached");
    mappingTargetClasses.push_back(found->second);
  }

  auto importOwner =
      std::make_shared<const ::loom::mapping::SpatialMappingImportContext>(
          std::move(validated->imports));
  auto catalogOwner = std::make_shared<
      const std::vector<loom::pnr::detail::SpatialCatalogEntry>>(
      std::move(catalog));
  auto classOwner = std::make_shared<const std::vector<PnrIndex>>(
      std::move(mappingTargetClasses));

  SystemActiveContextStatistics statistics;
  statistics.context.constructionCount = 1;
  statistics.context.constructionNanoseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
  statistics.spatialMappingCount = catalogOwner->size();
  statistics.timingProfileCount = physicalTimingProfiles.size();
  statistics.techMappingImportRequests =
      validated->statistics.techMappingImportRequests;
  statistics.techMappingImportHits =
      validated->statistics.techMappingImportHits;
  statistics.techMappingImportMisses =
      validated->statistics.techMappingImportMisses;
  for (const auto &entry : *catalogOwner) {
    statistics.coveredGraphCount += entry.covers.size();
    statistics.schedulePressureCount +=
        entry.graphStaticSchedulePressures.size();
    statistics.recurrenceProjectionCount += entry.graphRecurrenceTimings.size();
    for (const auto &progress : entry.graphProgress)
      statistics.routeProgressObligationCount +=
          progress.routeObligations.size();
  }
  statistics.context.retainedBytes =
      sizeof(loom::pnr::detail::SystemActiveContextStorage) +
      canonicalMappings.capacity() * sizeof(ArtifactRootReference) +
      importOwner->statistics().retainedBytes +
      retainedSpatialCatalogBytes(*catalogOwner) +
      classOwner->size() * sizeof(PnrIndex);
  statistics.context.deterministicWork =
      importOwner->statistics().deterministicWork;
  if (llvm::Error error = addWork(statistics.context.deterministicWork,
                                  statistics.spatialMappingCount +
                                      statistics.coveredGraphCount +
                                      statistics.routeProgressObligationCount +
                                      statistics.schedulePressureCount +
                                      statistics.recurrenceProjectionCount +
                                      statistics.timingProfileCount +
                                      statistics.techMappingImportRequests +
                                      statistics.techMappingImportMisses))
    return std::move(error);

  auto storage =
      std::make_shared<const loom::pnr::detail::SystemActiveContextStorage>(
          loom::pnr::detail::SystemActiveContextStorage{
              deriveSystemActiveContextKey(staticStorage, dataflow,
                                           physicalTimingProfiles, constraints,
                                           canonicalMappings),
              dataflow.identity(), system.artifact().identity(),
              constraints.view().identity(), std::move(canonicalMappings),
              std::move(importOwner), std::move(catalogOwner),
              std::move(classOwner), statistics});
  return SystemActiveContext(std::move(storage));
}

llvm::Error loom::pnr::revalidateSystemActiveContext(
    const SystemActiveContext &context,
    const SystemStaticContext &staticContext,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const FabricSystemRootView &system,
    llvm::ArrayRef<FabricPhysicalTimingProfileView> physicalTimingProfiles,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings) {
  if (llvm::Error error = revalidateSystemStaticContext(staticContext, system))
    return error;
  if (!context.storage_ || !context.storage_->spatialMappingImports ||
      !context.storage_->spatialCatalog ||
      !context.storage_->spatialMappingTargetClasses)
    return invalid("SystemActiveContext is incomplete");
  std::vector<ArtifactRootReference> canonicalMappings(spatialMappings.begin(),
                                                       spatialMappings.end());
  llvm::sort(canonicalMappings, artifactRootReferenceLess);
  if (std::adjacent_find(canonicalMappings.begin(), canonicalMappings.end()) !=
      canonicalMappings.end())
    return invalid("SystemActiveContext revalidation found duplicate "
                   "SpatialMappings");
  const auto &staticStorage =
      loom::pnr::detail::systemStaticContextStorage(staticContext);
  if (context.storage_->dataflowIdentity != dataflow.identity() ||
      context.storage_->systemIdentity != system.artifact().identity() ||
      context.storage_->constraintIdentity != constraints.view().identity() ||
      context.storage_->spatialMappings != canonicalMappings ||
      context.storage_->spatialMappingImports->references() !=
          llvm::ArrayRef<ArtifactRootReference>(canonicalMappings) ||
      context.storage_->spatialCatalog->size() != canonicalMappings.size() ||
      context.storage_->spatialMappingTargetClasses->size() !=
          canonicalMappings.size() ||
      context.storage_->statistics.techMappingImportRequests !=
          canonicalMappings.size() ||
      context.storage_->statistics.techMappingImportHits +
              context.storage_->statistics.techMappingImportMisses !=
          context.storage_->statistics.techMappingImportRequests ||
      context.storage_->key !=
          deriveSystemActiveContextKey(staticStorage, dataflow,
                                       physicalTimingProfiles, constraints,
                                       canonicalMappings))
    return invalid("SystemActiveContext key does not match its exact inputs");
  return llvm::Error::success();
}

void loom::pnr::emitSystemActiveContextStatistics(
    const SystemActiveContext &context, mapping_debug::Stage stage,
    std::uint64_t hits, std::uint64_t misses) {
  const SystemActiveContextStatistics &statistics = context.statistics();
  mapping_debug::emit(
      mapping_debug::Level::Summary, stage,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "system_active";
        fields["cache_hits"] = hits;
        fields["cache_misses"] = misses;
        fields["construction_count"] = statistics.context.constructionCount;
        fields["construction_time_ns"] =
            statistics.context.constructionNanoseconds;
        fields["retained_bytes"] = statistics.context.retainedBytes;
        fields["deterministic_work"] = statistics.context.deterministicWork;
        fields["spatial_mapping_count"] = statistics.spatialMappingCount;
        fields["covered_graph_count"] = statistics.coveredGraphCount;
        fields["route_progress_obligation_count"] =
            statistics.routeProgressObligationCount;
        fields["schedule_pressure_count"] = statistics.schedulePressureCount;
        fields["recurrence_projection_count"] =
            statistics.recurrenceProjectionCount;
        fields["timing_profile_count"] = statistics.timingProfileCount;
        fields["tech_mapping_import_requests"] =
            statistics.techMappingImportRequests;
        fields["tech_mapping_import_hits"] = statistics.techMappingImportHits;
        fields["tech_mapping_import_misses"] =
            statistics.techMappingImportMisses;
      });
}
