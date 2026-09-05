#include "JointDesignMutationTest.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/HardwareDecision.h"
#include "DSE/HardwareMutationRepairRecord.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/JointDesignPolicy.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/JointMappingMigration.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "joint design mutation anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

struct MutationFamilyContract {
  llvm::StringLiteral label;
  loom::dse::JointMappingReuseDisposition rebase;
  loom::dse::JointSystemMappingReuseDisposition system;
};

constexpr MutationFamilyContract mutationFamilyContracts[] = {
    {"fu", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"memory", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"fifo", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"operand", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"switch", loom::dse::JointMappingReuseDisposition::LocalRepair,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"spatial-core", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"acc-core-add", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::Preserved},
    {"acc-core-remove", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::Reopened},
    {"transport", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"service", loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"spatial-topology", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    // A Temporal instruction-store resize changes the Module's internal
    // instruction-context inventory, which the local Module correspondence
    // does not yet map; the family therefore witnesses the cold fallback.
    {"instruction-capacity",
     loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
    {"system-instruction-context",
     loom::dse::JointMappingReuseDisposition::Preserved,
     loom::dse::JointSystemMappingReuseDisposition::Reopened},
    {"combined", loom::dse::JointMappingReuseDisposition::ColdFallback,
     loom::dse::JointSystemMappingReuseDisposition::ColdFallback},
};

const MutationFamilyContract &mutationFamilyContract(llvm::StringRef label) {
  for (const MutationFamilyContract &contract : mutationFamilyContracts)
    if (contract.label == label)
      return contract;
  fail("mutation family has no declared disposition contract: " + label.str());
}

/// Proves, without executing any Mapping work, that the declared family
/// contracts still span every repair disposition the matrix must witness.
void requireMutationDispositionCoverage() {
  bool lowerLayerPreservation = false;
  bool systemPreservation = false;
  bool systemReopen = false;
  bool coldFallback = false;
  for (const MutationFamilyContract &contract : mutationFamilyContracts) {
    lowerLayerPreservation |=
        contract.rebase !=
        loom::dse::JointMappingReuseDisposition::ColdFallback;
    systemPreservation |=
        contract.system ==
        loom::dse::JointSystemMappingReuseDisposition::Preserved;
    systemReopen |= contract.system ==
                    loom::dse::JointSystemMappingReuseDisposition::Reopened;
    coldFallback |=
        contract.rebase ==
            loom::dse::JointMappingReuseDisposition::ColdFallback ||
        contract.system ==
            loom::dse::JointSystemMappingReuseDisposition::ColdFallback;
  }
  if (!lowerLayerPreservation || !systemPreservation || !systemReopen ||
      !coldFallback)
    fail("mutation family contracts do not span every repair disposition");
}

} // namespace

namespace loom::dse::joint_test {

void exerciseJointDesignMutationFamilies(
    llvm::StringRef mutationFamily, llvm::StringRef temporaryPath,
    const JointDesignExplorationPlan &plan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const ArtifactRootReference &parentMapping,
    const ResolvedConfig &config, const ArtifactRootReference &system,
    const ArtifactStore &store, const BlobStore &blobs) {
  const auto targetModules =
      take(projectJointDesignTargetModules(system, store));
  if (targetModules.empty())
    fail("mutation fixture has no target Module");
  auto systemArtifact = take(fabric::importEntireFabricRoot(system, store));
  auto systemView = take(fabric::requireSystemRoot(systemArtifact.view()));
  auto targetModule =
      take(fabric::importEntireFabricRoot(targetModules.front(), store));
  std::optional<fabric::FabricPeOccurrenceRef> operandPe;
  for (const auto pe : targetModule.view().peOccurrences())
    if (targetModule.view().peSchedule(pe) == ::fabric::Schedule::Temporal) {
      operandPe = pe;
      break;
    }
  if (!operandPe)
    fail("mutation fixture has no Temporal PE");
  const std::uint32_t operandEntries =
      targetModule.view().peOperandBufferSize(*operandPe);
  if (operandEntries == 0 ||
      operandEntries == std::numeric_limits<std::uint32_t>::max())
    fail("mutation fixture has no growable Temporal operand buffer");

  const bool runEveryFamily = mutationFamily == allJointDesignTestSections;
  requireMutationDispositionCoverage();
  std::uint64_t repairOrdinal = 0;
  std::uint64_t executedFamilies = 0;
  const auto executeMutation = [&](llvm::StringRef label,
                                   loom::dse::JointHardwareMutationChild child)
      -> std::optional<loom::dse::JointHardwareMutationRepair> {
    if (child.impacts.empty())
      fail(label + " mutation has no typed impact");
    // Child materialization stays on every path so the cross-family fixture
    // lineage is identical regardless of which family this process executes.
    if (!runEveryFamily && label != mutationFamily)
      return std::nullopt;
    ++executedFamilies;
    llvm::SmallString<128> journal(temporaryPath);
    llvm::sys::path::append(journal, "mutation-" +
                                         std::to_string(repairOrdinal++) + "-" +
                                         label.str());
    const std::string producer =
        "loom.test.hardware_mutation_matrix." + label.str() + ".v1";
    loom::dse::JointHardwareReopenRequest repairRequest{
        take(loom::dse::DseProducerSemanticBuildIdentity::get(producer)),
        journal.str().str(),
        {},
        loom::dse::JointDesignStoppingPolicy::FirstVerified,
        std::nullopt,
        std::nullopt,
        take(loom::dse::SiteCapacity::get(2, 0, 0)),
        take(loom::dse::PlanExecutionPolicy::get(
            2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))};
    // The matrix is the evidence owner for cold versus preserve-first
    // repair, so it asks for the independent cold oracle that ordinary
    // production repair does not run.
    repairRequest.coldComparisonBaseline = true;
    auto repair = take(loom::dse::executeJointHardwareMutationRepair(
        plan, parentExecution, policy, parentMapping, std::move(child),
        std::move(repairRequest), store, blobs));
    const auto verified = [](const auto &statistics, std::size_t count) {
      return statistics.importRequests == count &&
             statistics.cacheMisses == count &&
             statistics.uniqueConstructions == count &&
             statistics.deterministicWork != 0 && statistics.retainedBytes != 0;
    };
    if (repair.coldMappings.empty() || repair.incrementalMappings.empty() ||
        repair.coldExecution->summary.techMappingDispatchCount == 0 ||
        repair.coldExecution->summary.spatialPnrDispatchCount == 0 ||
        repair.coldExecution->summary.systemPnrDispatchCount == 0 ||
        repair.incrementalExecution.summary.systemPnrDispatchCount == 0 ||
        repair.coldExecution->summary.coldReopenWallTimeNanoseconds == 0 ||
        (repair.rebase.disposition ==
                 loom::dse::JointMappingReuseDisposition::ColdFallback
             ? repair.incrementalExecution.summary
                       .coldReopenWallTimeNanoseconds == 0
             : repair.incrementalExecution.summary
                       .incrementalReopenWallTimeNanoseconds == 0) ||
        !verified(repair.coldVerification, repair.coldMappings.size()) ||
        !verified(repair.incrementalVerification,
                  repair.incrementalMappings.size()))
      fail(label + " mutation did not execute and independently verify both "
                   "cold and preserve-first Mapping paths");
    if (llvm::Error error = loom::dse::validateJointMappingRebaseAccounting(
            repair.rebase.accounting))
      fail(label + " mutation has an open repair cone: " +
           llvm::toString(std::move(error)));
    const MutationFamilyContract &contract = mutationFamilyContract(label);
    if (repair.rebase.disposition != contract.rebase ||
        repair.systemDisposition != contract.system) {
      std::string failures;
      for (const auto &failure : repair.rebase.failures)
        failures +=
            " failure=" +
            loom::dse::jointMappingRebaseFailureReasonSpelling(failure.reason)
                .str() +
            ":" + failure.diagnostic;
      fail(label + " mutation reached " +
           loom::dse::jointMappingReuseDispositionSpelling(
               repair.rebase.disposition) +
           "/" +
           loom::dse::jointSystemMappingReuseDispositionSpelling(
               repair.systemDisposition) +
           " instead of its declared " +
           loom::dse::jointMappingReuseDispositionSpelling(contract.rebase) +
           "/" +
           loom::dse::jointSystemMappingReuseDispositionSpelling(
               contract.system) +
           failures);
    }
    if (contract.rebase !=
            loom::dse::JointMappingReuseDisposition::ColdFallback &&
        repair.rebase.accounting.preservedTechMappings == 0 &&
        repair.rebase.accounting.preservedSpatialMappings == 0 &&
        repair.rebase.accounting.repairedTechMappings == 0 &&
        repair.rebase.accounting.repairedSpatialMappings == 0)
      fail(label + " mutation reported reuse without a preserved or "
                   "repaired lower-layer Mapping");
    for (const auto *roots :
         {&repair.coldMappings, &repair.incrementalMappings})
      for (const loom::ArtifactRootReference &reference : *roots) {
        auto mapping =
            take(loom::mapping::importSystemMapping(reference, store));
        if (mapping.view().fabricIdentity() != repair.child.system.artifact)
          fail(label + " mutation Mapping names a foreign System");
      }
    // The executor publishes the durable per-family record; the strict
    // re-import must agree with the executed repair it summarizes.
    auto record = take(
        loom::dse::importHardwareMutationRepairRecord(repair.record, store));
    const loom::dse::HardwareMutationRepairRecord &durable = record.record();
    if (durable.parentMapping != repair.parentMapping ||
        durable.childSystem != repair.child.system ||
        durable.impacts.size() != repair.child.impacts.size() ||
        durable.mappingReuseDisposition != repair.rebase.disposition ||
        durable.systemMappingReuseDisposition != repair.systemDisposition ||
        durable.cold.has_value() != repair.coldExecution.has_value() ||
        (durable.cold && durable.cold->mappings != repair.coldMappings) ||
        durable.incremental.mappings != repair.incrementalMappings ||
        durable.incremental.systemPnrDispatches !=
            repair.incrementalExecution.summary.systemPnrDispatchCount)
      fail(label + " mutation record disagrees with the executed repair");
    for (const auto indexed : llvm::enumerate(repair.child.impacts))
      if (durable.impacts[indexed.index()].family != indexed.value().family ||
          durable.impacts[indexed.index()].locality != indexed.value().locality)
        fail(label + " mutation record lost its typed impact family");
    // One durable line per family so a sharded run still reports the typed
    // dispositions its family reached.
    llvm::outs() << "mutation-family " << label << " rebase="
                 << loom::dse::jointMappingReuseDispositionSpelling(
                        repair.rebase.disposition)
                 << " system="
                 << loom::dse::jointSystemMappingReuseDispositionSpelling(
                        repair.systemDisposition)
                 << " record="
                 << loom::formatArtifactIdentityHex(repair.record.artifact);
    for (const auto &failure : repair.rebase.failures)
      llvm::outs() << " failure=" << static_cast<int>(failure.reason) << ":"
                   << failure.diagnostic;
    llvm::outs() << "\n";
    return repair;
  };

  const auto materializeModule =
      [&](llvm::StringRef label,
          loom::dse::SpatialMicroarchitectureDecisionDomain decision) {
        auto child = take(loom::dse::materializeJointModuleHardwareMutation(
            config, system, targetModules.front(), std::move(decision), store,
            blobs));
        if (child.system == system || child.impacts.size() != 1)
          fail(label + " did not materialize one distinct hardware child");
        return child;
      };
  const auto materializeTopology =
      [&](llvm::StringRef label,
          loom::dse::SpatialTopologyDecisionDomain decision) {
        auto child = take(loom::dse::materializeJointModuleHardwareMutation(
            config, system, targetModules.front(), std::move(decision), store,
            blobs));
        if (child.system == system || child.impacts.size() != 1 ||
            child.impacts.front().family !=
                loom::dse::HardwareMutationFamily::SpatialTopology)
          fail(label + " did not materialize one distinct topology child");
        return child;
      };
  const auto materializeSystem =
      [&](llvm::StringRef label, const loom::ArtifactRootReference &parent,
          loom::dse::SystemCompositionDecisionDomain decision,
          llvm::ArrayRef<loom::ArtifactRootReference> modules) {
        auto child = take(loom::dse::materializeJointSystemHardwareMutation(
            config, parent, std::move(decision), modules, store, blobs));
        if (child.system == parent || child.impacts.size() != 1)
          fail(label + " did not materialize one distinct hardware child");
        return child;
      };

  std::optional<loom::dse::JointHardwareMutationChild> fuChild;
  for (const auto pe : targetModule.view().peOccurrences()) {
    std::vector<loom::fabric::FabricFuOccurrenceRef> inventory;
    for (const auto fu : targetModule.view().fuOccurrences())
      if (targetModule.view().parentPeOf(fu) == pe)
        inventory.push_back(fu);
    for (const auto prototype : targetModule.view().fuOccurrences()) {
      if (targetModule.view().parentPeOf(prototype) == pe)
        continue;
      auto candidateInventory = inventory;
      candidateInventory.push_back(prototype);
      auto candidate = loom::dse::materializeJointModuleHardwareMutation(
          config, system, targetModules.front(),
          loom::dse::ChangeFuInventoryDomain{pe,
                                             {std::move(candidateInventory)}},
          store, blobs);
      if (!candidate) {
        llvm::consumeError(candidate.takeError());
        continue;
      }
      fuChild = std::move(*candidate);
      break;
    }
    if (fuChild)
      break;
  }
  if (!fuChild)
    fail("FU mutation fixture has no valid inventory rewrite");
  executeMutation("fu", std::move(*fuChild));

  const auto memory = targetModule.view().memoryOccurrences().front();
  const std::uint64_t memoryCapacity =
      targetModule.view().localMemoryServiceCapacityBytes(memory);
  if (memoryCapacity == 0 ||
      memoryCapacity == std::numeric_limits<std::uint64_t>::max())
    fail("memory mutation fixture has no growable Local Memory Service");
  auto memoryChild = materializeModule(
      "memory", loom::dse::ResizeMemoryDomain{memory, {memoryCapacity + 1}});
  const loom::dse::JointHardwareMutationChild memorySpatialChild = memoryChild;
  executeMutation("memory", std::move(memoryChild));

  auto fifoChild = materializeModule(
      "fifo", loom::dse::ResizeFifoDomain{
                  targetModule.view().fifoOccurrences().front(), {257}});
  const loom::dse::JointHardwareMutationChild combinedFirst = fifoChild;
  executeMutation("fifo", std::move(fifoChild));

  executeMutation(
      "operand",
      materializeModule("operand", loom::dse::ResizeTemporalOperandBufferDomain{
                                       *operandPe, {operandEntries + 1}}));

  std::optional<loom::fabric::FabricSwitchOccurrenceRef> temporalSwitch;
  for (const auto target : targetModule.view().switchOccurrences()) {
    if (targetModule.view().switchSchedule(target) ==
        ::fabric::Schedule::Temporal) {
      temporalSwitch = target;
      break;
    }
  }
  if (!temporalSwitch ||
      targetModule.view().switchRouteTableSize(*temporalSwitch) ==
          std::numeric_limits<std::uint32_t>::max())
    fail("switch mutation fixture has no growable Temporal switch");
  executeMutation(
      "switch",
      materializeModule(
          "switch",
          loom::dse::ResizeSwitchRouteTableDomain{
              *temporalSwitch,
              {static_cast<std::uint32_t>(
                  targetModule.view().switchRouteTableSize(*temporalSwitch) +
                  1)}}));

  const auto &moduleLineage =
      memorySpatialChild.executionBindingCorrespondence->modules();
  const auto replacedModule =
      llvm::find_if(moduleLineage, [&](const auto &entry) {
        return entry.parent == targetModules.front() &&
               entry.child != entry.parent;
      });
  if (replacedModule == moduleLineage.end())
    fail("SpatialCore mutation fixture lost replacement Module lineage");
  const std::array replacementModules = {replacedModule->child};
  executeMutation(
      "spatial-core",
      materializeSystem("spatial-core", system,
                        loom::dse::ReplaceSpatialAttachmentDomain{
                            systemView.artifact().accCoreOccurrences().front(),
                            {replacedModule->child}},
                        replacementModules));

  executeMutation(
      "acc-core-add",
      materializeSystem("acc-core-add", system,
                        loom::dse::AddAccCoreDomain{
                            systemView.artifact().accCoreOccurrences().front(),
                            {targetModules.front()}},
                        targetModules));

  executeMutation("acc-core-remove",
                  materializeSystem(
                      "acc-core-remove", system,
                      loom::dse::RemoveAccCoreDomain{
                          {systemView.artifact().accCoreOccurrences().front()}},
                      targetModules));

  std::optional<loom::dse::JointHardwareMutationChild> transportChild;
  for (auto indexedFirst :
       llvm::enumerate(systemView.artifact().pointConnections())) {
    for (const auto &second :
         systemView.artifact().pointConnections().drop_front(
             indexedFirst.index() + 1)) {
      const auto &first = indexedFirst.value();
      if (!llvm::equal(
              systemView.artifact().transportEndpointType(first.source),
              systemView.artifact().transportEndpointType(
                  second.destination)) ||
          !llvm::equal(
              systemView.artifact().transportEndpointType(second.source),
              systemView.artifact().transportEndpointType(first.destination)))
        continue;
      auto firstDestination = first.destination;
      auto secondDestination = second.destination;
      if (loom::fabric::canonicalFabricBytes(secondDestination) <
          loom::fabric::canonicalFabricBytes(firstDestination))
        std::swap(firstDestination, secondDestination);
      auto candidate = loom::dse::materializeJointSystemHardwareMutation(
          config, system,
          loom::dse::SwapTransportConnectionSourcesDomain{firstDestination,
                                                          {secondDestination}},
          targetModules, store, blobs);
      if (!candidate) {
        llvm::consumeError(candidate.takeError());
        continue;
      }
      transportChild = std::move(*candidate);
      break;
    }
    if (transportChild)
      break;
  }
  if (!transportChild)
    fail("transport mutation fixture has no legal alternate connection");
  executeMutation("transport", std::move(*transportChild));

  std::optional<loom::fabric::SystemServiceEndpointRef> memoryEndpoint;
  for (const auto &attachment : systemView.spatialAttachments()) {
    if (attachment.spatialEndpoint.memory() && attachment.serviceEndpoint) {
      memoryEndpoint = *attachment.serviceEndpoint;
      break;
    }
  }
  if (!memoryEndpoint)
    fail("service mutation fixture has no memory endpoint");
  const auto *endpointOwner = systemView.serviceEndpointOwner(*memoryEndpoint);
  const auto *memoryOwner =
      endpointOwner ? std::get_if<loom::fabric::FabricMemoryServiceRef>(
                          &endpointOwner->owner().payload)
                    : nullptr;
  const auto *memoryService =
      memoryOwner ? std::get_if<loom::fabric::SystemMemoryServiceRef>(
                        &memoryOwner->payload)
                  : nullptr;
  const auto *memoryContract =
      memoryService ? systemView.memoryService(*memoryService) : nullptr;
  if (!memoryContract || memoryContract->regions().empty() ||
      memoryContract->regions().front().sizeBytes ==
          std::numeric_limits<std::uint64_t>::max())
    fail("service mutation fixture has no growable memory region");
  executeMutation(
      "service",
      materializeSystem("service", system,
                        loom::dse::ResizeSystemMemoryRegionDomain{
                            *memoryService,
                            0,
                            {memoryContract->regions().front().sizeBytes + 1}},
                        targetModules));

  // The three dedicated families named by the acceptance matrix: a Module
  // topology rewrite (one more FU occurrence), a Temporal instruction-store
  // capacity change, and a System InstructionCore realization change.
  executeMutation("spatial-topology",
                  materializeTopology(
                      "spatial-topology",
                      loom::dse::AddOccurrenceDomain{{take(
                          loom::fabric::FabricModulePhysicalOwnerRef::create(
                              targetModule.view().fuOccurrences().front()))}}));

  const std::uint64_t instructionCapacity =
      targetModule.view().peResidentContextCount(*operandPe);
  if (instructionCapacity == 0 ||
      instructionCapacity >=
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    fail("instruction-capacity fixture has no growable instruction store");
  executeMutation("instruction-capacity",
                  materializeModule("instruction-capacity",
                                    loom::dse::ResizeInstructionStoreDomain{
                                        *operandPe,
                                        {static_cast<std::uint32_t>(
                                            instructionCapacity + 1)}}));

  std::optional<loom::fabric::AccCoreOccurrenceRef> realizationTarget;
  std::optional<loom::fabric::AccCoreOccurrenceRef> realizationPrototype;
  std::optional<loom::fabric::InstructionCoreRealizationKind> targetKind;
  for (const auto core : systemView.artifact().accCoreOccurrences()) {
    const auto *realization = systemView.instructionCoreMicroarchitecture(
        loom::fabric::InstructionCoreContextRef{core});
    if (!realization)
      continue;
    if (!realizationTarget) {
      realizationTarget = core;
      targetKind = realization->kind();
      continue;
    }
    if (realization->kind() != *targetKind) {
      realizationPrototype = core;
      break;
    }
  }
  if (!realizationTarget || !realizationPrototype)
    fail("system-instruction-context fixture has no alternative "
         "InstructionCore realization");
  executeMutation(
      "system-instruction-context",
      materializeSystem(
          "system-instruction-context", system,
          loom::dse::SelectInstructionCoreRealizationDomain{
              loom::fabric::InstructionCoreContextRef{*realizationTarget},
              {loom::fabric::InstructionCoreContextRef{*realizationPrototype}}},
          targetModules));

  auto combinedSystem =
      take(loom::fabric::importEntireFabricRoot(combinedFirst.system, store));
  auto combinedSystemView =
      take(loom::fabric::requireSystemRoot(combinedSystem.view()));
  auto combinedModules = take(
      loom::dse::projectJointDesignTargetModules(combinedFirst.system, store));
  auto combinedSecond = materializeSystem(
      "combined-tail", combinedFirst.system,
      loom::dse::AddAccCoreDomain{
          combinedSystemView.artifact().accCoreOccurrences().front(),
          {combinedModules.front()}},
      combinedModules);
  auto combined = take(loom::dse::composeJointHardwareMutationChildren(
      std::move(combinedFirst), std::move(combinedSecond), store));
  if (combined.impacts.size() != 2)
    fail("combined mutation lost ordered component impacts");
  executeMutation("combined", std::move(combined));
  const std::uint64_t expectedFamilies =
      runEveryFamily ? std::size(mutationFamilyContracts) : 1;
  if (executedFamilies != expectedFamilies)
    fail("mutation family selector matched " +
         std::to_string(executedFamilies) + " of " +
         std::to_string(expectedFamilies) +
         " families: " + mutationFamily.str());
  return;
}

} // namespace loom::dse::joint_test
