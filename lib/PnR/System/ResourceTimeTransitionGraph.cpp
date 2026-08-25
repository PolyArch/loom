#include "PnR/System/SystemMappingMigration.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Deployment/Deployment.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <system_error>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_mapping_migration_invalid: " + message);
}

std::optional<std::size_t>
endpointIndex(const ResourceTimeTransitionGraph &graph,
              const ResourceTimeTransitionEndpointReference &endpoint) {
  for (std::size_t index = 0; index != graph.endpoints.size(); ++index)
    if (graph.endpoints[index] == endpoint)
      return index;
  return std::nullopt;
}

bool sameRootSet(llvm::ArrayRef<::dataflow::RootThreadLaunchRef> lhs,
                 llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rhs) {
  return lhs.size() == rhs.size() && llvm::all_of(lhs, [&](auto root) {
           return llvm::is_contained(rhs, root);
         });
}

bool rootSubset(llvm::ArrayRef<::dataflow::RootThreadLaunchRef> subset,
                llvm::ArrayRef<::dataflow::RootThreadLaunchRef> superset) {
  return llvm::all_of(
      subset, [&](auto root) { return llvm::is_contained(superset, root); });
}

bool rootLess(const ::dataflow::RootThreadLaunchRef &lhs,
              const ::dataflow::RootThreadLaunchRef &rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

std::vector<::dataflow::RootThreadLaunchRef>
completedAfter(const ResourceTimeTransition &transition) {
  std::vector<::dataflow::RootThreadLaunchRef> completed =
      transition.completedBefore;
  completed.push_back(transition.beforeActive.front().region);
  llvm::sort(completed, rootLess);
  return completed;
}

llvm::Error verifyCompletionFrontierPaths(
    const ResourceTimeTransitionGraph &graph,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootScope) {
  const std::optional<std::size_t> entry = endpointIndex(graph, graph.entry);
  if (!entry)
    return invalid("resource-time transition graph lost its entry endpoint");

  using CompletionFrontier = std::vector<::dataflow::RootThreadLaunchRef>;
  std::vector<std::vector<CompletionFrontier>> reachable(
      graph.endpoints.size());
  reachable[*entry].push_back({});
  std::vector<bool> reachableEdges(graph.transitions.size(), false);

  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t edge = 0; edge != graph.transitions.size(); ++edge) {
      if (reachableEdges[edge])
        continue;
      const ResourceTimeTransition &transition = graph.transitions[edge];
      if (!transition.safePoint ||
          transition.safePoint->kind != ResourceTimeSafePointKind::Completion ||
          transition.beforeActive.size() != 1)
        return invalid("resource-time transition graph has no bounded "
                       "completion frontier");
      if (!rootSubset(transition.completedBefore, rootScope) ||
          !llvm::is_contained(rootScope,
                              transition.beforeActive.front().region))
        return invalid("resource-time transition graph completion frontier "
                       "is outside its endpoint root scope");
      const std::size_t parent = *endpointIndex(graph, transition.parent);
      const std::size_t child = *endpointIndex(graph, transition.child);
      if (!llvm::any_of(reachable[parent], [&](const auto &frontier) {
            return rootSubset(frontier, transition.completedBefore);
          }))
        continue;
      reachableEdges[edge] = true;
      CompletionFrontier after = completedAfter(transition);
      if (!llvm::is_contained(reachable[child], after)) {
        reachable[child].push_back(std::move(after));
        changed = true;
      }
    }
  }
  if (llvm::is_contained(reachableEdges, false))
    return invalid("resource-time transition graph has an unrealizable "
                   "completion-frontier edge");
  return llvm::Error::success();
}

} // namespace

llvm::Error
validateResourceTimeTransitionGraph(const ResourceTimeTransitionGraph &graph) {
  if (graph.endpoints.empty())
    return invalid("resource-time transition graph has no Mapping state");
  const auto entryIndex = endpointIndex(graph, graph.entry);
  if (!entryIndex)
    return invalid("resource-time transition graph entry is not a catalog "
                   "endpoint");
  for (std::size_t index = 0; index != graph.endpoints.size(); ++index) {
    const auto &endpoint = graph.endpoints[index];
    if (endpoint.mapping.schemaIdentity !=
            ::loom::mapping::mappingArtifactSchema.identity ||
        endpoint.mapping.schemaVersion !=
            ::loom::mapping::mappingArtifactSchema.version)
      return invalid("resource-time transition graph has a non-Mapping "
                     "endpoint");
    if (!endpoint.deployment ||
        endpoint.deployment->schemaIdentity !=
            ::loom::deployment::deploymentSchema.identity ||
        endpoint.deployment->schemaVersion !=
            ::loom::deployment::deploymentSchema.version)
      return invalid("resource-time transition graph endpoint has no exact "
                     "Deployment");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (graph.endpoints[prior].mapping == endpoint.mapping)
        return invalid("resource-time transition graph repeats one Mapping "
                       "state");
  }

  std::vector<std::vector<std::size_t>> outgoing(graph.endpoints.size());
  for (std::size_t index = 0; index != graph.transitions.size(); ++index) {
    const ResourceTimeTransition &transition = graph.transitions[index];
    if (llvm::Error error = validateResourceTimeTransition(transition))
      return error;
    const auto parent = endpointIndex(graph, transition.parent);
    const auto child = endpointIndex(graph, transition.child);
    if (!parent || !child)
      return invalid("resource-time transition graph edge names a foreign "
                     "endpoint");
    if (*parent == *child)
      return invalid("resource-time transition graph contains a self edge");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (graph.transitions[prior].parent == transition.parent &&
          graph.transitions[prior].child == transition.child &&
          graph.transitions[prior].trigger == transition.trigger)
        return invalid("resource-time transition graph repeats one typed "
                       "edge");
    outgoing[*parent].push_back(*child);
  }

  std::vector<bool> reachable(graph.endpoints.size(), false);
  std::vector<std::size_t> worklist = {*entryIndex};
  reachable[*entryIndex] = true;
  while (!worklist.empty()) {
    const std::size_t current = worklist.back();
    worklist.pop_back();
    for (std::size_t child : outgoing[current]) {
      if (reachable[child])
        continue;
      reachable[child] = true;
      worklist.push_back(child);
    }
  }
  if (llvm::any_of(reachable, [](bool value) { return !value; }))
    return invalid("resource-time transition graph has an unreachable Mapping "
                   "state");
  return llvm::Error::success();
}

llvm::Error
verifyResourceTimeTransitionGraph(const ResourceTimeTransitionGraph &graph,
                                  const ArtifactStore &artifacts,
                                  const BlobStore &blobs) {
  if (llvm::Error error = validateResourceTimeTransitionGraph(graph))
    return error;
  std::optional<ArtifactIdentity> dataflowIdentity;
  std::optional<ArtifactIdentity> fabricIdentity;
  std::optional<std::vector<::dataflow::RootThreadLaunchRef>> rootScope;
  for (const ResourceTimeTransitionEndpointReference &endpoint :
       graph.endpoints) {
    auto mapping =
        ::loom::mapping::importSystemMapping(endpoint.mapping, artifacts);
    if (!mapping)
      return mapping.takeError();
    auto deployment = ::loom::deployment::importDeployment(*endpoint.deployment,
                                                           artifacts, blobs);
    if (!deployment)
      return deployment.takeError();
    if (deployment->deployment().systemMapping() != endpoint.mapping)
      return invalid("resource-time transition graph Deployment selects "
                     "another Mapping");
    if (dataflowIdentity &&
        *dataflowIdentity != mapping->view().dataflowIdentity())
      return invalid("resource-time transition graph spans multiple Dataflow "
                     "identities");
    if (fabricIdentity && *fabricIdentity != mapping->view().fabricIdentity())
      return invalid("resource-time transition graph spans multiple Fabric "
                     "identities");
    const auto roots = mapping->view().executionBindings().rootThreadLaunches();
    if (rootScope && !sameRootSet(*rootScope, roots))
      return invalid("resource-time transition graph endpoint root scopes "
                     "differ");
    dataflowIdentity = mapping->view().dataflowIdentity();
    fabricIdentity = mapping->view().fabricIdentity();
    if (!rootScope)
      rootScope.emplace(roots.begin(), roots.end());
  }
  for (const ResourceTimeTransition &transition : graph.transitions)
    if (llvm::Error error =
            verifyResourceTimeTransitionClosure(transition, artifacts, blobs))
      return error;
  if (!rootScope)
    return invalid("resource-time transition graph has no endpoint root "
                   "scope");
  if (llvm::Error error = verifyCompletionFrontierPaths(graph, *rootScope))
    return error;
  return llvm::Error::success();
}

} // namespace loom::pnr
