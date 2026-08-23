#include "DSE/JointDesignPolicy.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <map>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_frontier_invalid: " + message);
}

llvm::Error canonicalizeUnique(std::vector<ArtifactRootReference> &roots,
                               llvm::StringRef subject) {
  if (roots.empty())
    return invalid(subject + " frontier is empty");
  llvm::sort(roots, artifactRootReferenceLess);
  if (std::adjacent_find(roots.begin(), roots.end()) != roots.end())
    return invalid(subject + " frontier contains a duplicate root");
  return llvm::Error::success();
}

bool scopeLess(const JointSoftwareScope &lhs, const JointSoftwareScope &rhs) {
  return std::lexicographical_compare(
      lhs.workloads.begin(), lhs.workloads.end(), rhs.workloads.begin(),
      rhs.workloads.end(), artifactRootReferenceLess);
}

struct SystemProjection final {
  ArtifactRootReference system;
  JointPairAnalyticProjection features;
};

llvm::Expected<std::uint64_t> checkedEstimateWork(
    std::uint64_t waves, std::uint64_t actors, std::uint64_t graphs,
    std::uint64_t memoryRoots, std::uint64_t transportResources) {
  // This is deliberately a monotone screening score, not a calibrated
  // runtime model. Saturating arithmetic keeps a malformed huge artifact
  // from wrapping into an apparently attractive candidate.
  const unsigned __int128 work =
      static_cast<unsigned __int128>(waves) * 1024u +
      static_cast<unsigned __int128>(actors) * 8u +
      static_cast<unsigned __int128>(graphs) * 16u +
      static_cast<unsigned __int128>(memoryRoots) * 32u +
      static_cast<unsigned __int128>(transportResources) * 4u;
  if (work > std::numeric_limits<std::uint64_t>::max())
    return std::numeric_limits<std::uint64_t>::max();
  return static_cast<std::uint64_t>(work);
}

} // namespace

llvm::StringRef jointPairEstimateConfidenceSpelling(
    JointPairEstimateConfidence confidence) {
  switch (confidence) {
  case JointPairEstimateConfidence::LowerBound:
    return "lower_bound";
  case JointPairEstimateConfidence::LowConfidence:
    return "low_confidence";
  }
  llvm_unreachable("unknown joint pair estimate confidence");
}

llvm::Expected<BoundedJointFrontier>
buildBoundedJointFrontier(JointDesignInputs inputs,
                          const JointDesignPolicy &policy,
                          const ArtifactStore &artifactStore) {
  if (inputs.applicationScopes.empty())
    return invalid("application scope frontier is empty");
  if (llvm::Error error = canonicalizeUnique(inputs.systemFrontier, "System"))
    return std::move(error);
  if (inputs.applicationScopes.size() > policy.maximumSoftwareFrontier())
    return invalid("software frontier exceeds its resolved bound");
  if (inputs.systemFrontier.size() > policy.maximumSystemFrontier())
    return invalid("System frontier exceeds its resolved bound");

  std::vector<JointSoftwareScope> softwareFrontier;
  softwareFrontier.reserve(inputs.applicationScopes.size());
  std::map<ArtifactIdentity::Storage, JointPairAnalyticProjection>
      dataflowFeatures;
  std::map<ArtifactIdentity::Storage, ArtifactRootReference>
      workloadDataflowOwners;
  for (std::vector<ArtifactRootReference> &workloads :
       inputs.applicationScopes) {
    if (llvm::Error error = canonicalizeUnique(workloads, "workload scope"))
      return std::move(error);
    std::optional<ArtifactRootReference> dataflowReference;
    for (const ArtifactRootReference &workload : workloads) {
      std::optional<ArtifactRootReference> owner;
      const auto workloadKey = workload.artifact.bytes();
      auto cachedOwner = workloadDataflowOwners.find(workloadKey);
      if (cachedOwner != workloadDataflowOwners.end()) {
        owner = cachedOwner->second;
      } else {
        auto imported =
            sim::importSpatialSimulationWorkload(workload, artifactStore);
        if (!imported)
          return imported.takeError();
        owner = ArtifactRootReference{
            dataflow::canonicalDataflowSchema.identity.str(),
            dataflow::canonicalDataflowSchema.version,
            imported->dataflow.identity()};
        workloadDataflowOwners.emplace(workloadKey, *owner);
      }
      if (dataflowReference && *dataflowReference != *owner)
        return invalid("one application scope has multiple Dataflow owners");
      dataflowReference = std::move(*owner);
    }
    if (dataflowFeatures.find(dataflowReference->artifact.bytes()) ==
        dataflowFeatures.end()) {
      auto dataflowArtifact =
          dataflow::importCanonicalDataflow(*dataflowReference, artifactStore);
      if (!dataflowArtifact)
        return dataflowArtifact.takeError();
      auto dataflow = dataflowArtifact->view();
      if (!dataflow)
        return dataflow.takeError();
      JointPairAnalyticProjection features;
      features.softwareActorCount = dataflow->actors().size();
      features.softwareGraphCount = dataflow->graphs().size();
      features.softwareMemoryRootCount = dataflow->logicalMemoryRoots().size();
      features.confidence = JointPairEstimateConfidence::LowConfidence;
      dataflowFeatures.emplace(dataflowReference->artifact.bytes(), features);
    }
    softwareFrontier.push_back(
        JointSoftwareScope{std::move(*dataflowReference),
                           std::move(workloads)});
  }
  llvm::sort(softwareFrontier, scopeLess);
  if (std::adjacent_find(softwareFrontier.begin(), softwareFrontier.end()) !=
      softwareFrontier.end())
    return invalid("application frontier contains a duplicate scope");
  std::vector<SystemProjection> systemProjections;
  systemProjections.reserve(inputs.systemFrontier.size());
  for (const ArtifactRootReference &root : inputs.systemFrontier) {
    auto artifact = fabric::importEntireFabricRoot(root, artifactStore);
    if (!artifact)
      return artifact.takeError();
    auto system = fabric::requireSystemRoot(artifact->view());
    if (!system)
      return system.takeError();
    if (system->artifact().accCoreOccurrences().empty())
      return invalid("System frontier contains no AccCore occurrence");
    JointPairAnalyticProjection features;
    features.systemAccCoreCount = system->artifact().accCoreOccurrences().size();
    features.systemTransportResourceCount =
        system->transportResources().size();
    features.confidence = JointPairEstimateConfidence::LowConfidence;
    systemProjections.push_back({root, features});
  }

  if (softwareFrontier.size() >
      std::numeric_limits<std::uint64_t>::max() /
          inputs.systemFrontier.size())
    return invalid("eligible pair count overflows u64");
  const std::uint64_t eligible =
      static_cast<std::uint64_t>(softwareFrontier.size()) *
      static_cast<std::uint64_t>(inputs.systemFrontier.size());
  struct RankedPair final {
    JointDesignPair pair;
    JointPairAnalyticProjection projection;
    std::uint64_t lowerBound = 0;
    std::uint64_t estimatedWork = 0;
  };
  const auto rankedLess = [](const RankedPair &lhs, const RankedPair &rhs) {
    if (lhs.lowerBound != rhs.lowerBound)
      return lhs.lowerBound < rhs.lowerBound;
    if (lhs.estimatedWork != rhs.estimatedWork)
      return lhs.estimatedWork < rhs.estimatedWork;
    if (lhs.pair.software.dataflow != rhs.pair.software.dataflow)
      return artifactRootReferenceLess(lhs.pair.software.dataflow,
                                       rhs.pair.software.dataflow);
    return artifactRootReferenceLess(lhs.pair.system, rhs.pair.system);
  };
  std::vector<RankedPair> retainedRanked;
  const std::uint64_t pairLimit = policy.maximumPairEvaluations();
  if (pairLimit > std::numeric_limits<std::size_t>::max())
    return invalid("pair evaluation bound exceeds host container capacity");
  const std::size_t retainedCapacity = static_cast<std::size_t>(std::min(
      pairLimit, eligible));
  retainedRanked.reserve(retainedCapacity);
  for (const JointSoftwareScope &software : softwareFrontier) {
    for (const ArtifactRootReference &system : inputs.systemFrontier) {
      const auto softwareFeatures = dataflowFeatures.find(
          software.dataflow.artifact.bytes());
      const auto systemFeatures = llvm::find_if(
          systemProjections,
          [&](const SystemProjection &projection) {
            return projection.system == system;
          });
      if (softwareFeatures == dataflowFeatures.end() ||
          systemFeatures == systemProjections.end())
        return invalid("analytic pair projection lost an exact root owner");
      JointPairAnalyticProjection projection = softwareFeatures->second;
      projection.systemAccCoreCount =
          systemFeatures->features.systemAccCoreCount;
      projection.systemTransportResourceCount =
          systemFeatures->features.systemTransportResourceCount;
      const std::uint64_t capacity =
          std::max<std::uint64_t>(1, projection.systemAccCoreCount);
      const unsigned __int128 work =
          static_cast<unsigned __int128>(projection.softwareActorCount) +
          static_cast<unsigned __int128>(projection.softwareGraphCount) * 2u +
          static_cast<unsigned __int128>(projection.softwareMemoryRootCount) *
              2u;
      const unsigned __int128 waves =
          (work + capacity - 1u) / capacity;
      projection.minimumExecutionWaves =
          waves > std::numeric_limits<std::uint64_t>::max()
              ? std::numeric_limits<std::uint64_t>::max()
              : static_cast<std::uint64_t>(waves);
      auto estimated = checkedEstimateWork(
          projection.minimumExecutionWaves, projection.softwareActorCount,
          projection.softwareGraphCount, projection.softwareMemoryRootCount,
          projection.systemTransportResourceCount);
      if (!estimated)
        return estimated.takeError();
      projection.estimatedWorkUnits = *estimated;
      RankedPair candidate{{software, system}, projection,
                           projection.minimumExecutionWaves,
                           projection.estimatedWorkUnits};
      if (retainedRanked.size() < pairLimit) {
        retainedRanked.push_back(std::move(candidate));
      } else {
        const auto worst =
            std::max_element(retainedRanked.begin(), retainedRanked.end(),
                             rankedLess);
        if (rankedLess(candidate, *worst))
          *worst = std::move(candidate);
      }
    }
  }
  llvm::sort(retainedRanked, rankedLess);
  const std::uint64_t retained = retainedRanked.size();
  std::vector<JointDesignPair> pairs;
  std::vector<JointPairAnalyticProjection> projections;
  pairs.reserve(static_cast<std::size_t>(retained));
  projections.reserve(static_cast<std::size_t>(retained));
  for (RankedPair &candidate : retainedRanked) {
    pairs.push_back(std::move(candidate.pair));
    projections.push_back(std::move(candidate.projection));
  }
  return BoundedJointFrontier{std::move(softwareFrontier),
                              std::move(inputs.systemFrontier),
                              std::move(pairs), std::move(projections),
                              eligible, eligible, eligible - retained,
                              retained != eligible};
}

} // namespace loom::dse
