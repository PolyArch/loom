#include "PnR/System/SystemCandidateState.h"

#include "PnR/InitializerRelationSolver.h"
#include "SystemCandidateServiceResolver.h"
#include "SystemPnrSearchDomainInternal.h"
#include "SystemServiceRouter.h"

#include "llvm/Support/Error.h"

#include <cassert>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_candidate_invalid: " + message);
}

llvm::Expected<std::vector<PnrIndex>>
relationChoices(const FrozenSystemPnrProblem &problem,
                llvm::ArrayRef<PnrIndex> threadChoices,
                llvm::ArrayRef<PnrIndex> graphChoices) {
  if (threadChoices.size() != problem.threadDecisions().size())
    return invalid("thread choice count does not match H");
  if (graphChoices.size() != problem.graphDecisions().size())
    return invalid("graph choice count does not match H");
  std::vector<PnrIndex> choices;
  choices.reserve(threadChoices.size() + graphChoices.size());
  for (const auto &[decision, choice] : llvm::enumerate(threadChoices)) {
    if (choice >= problem.threadChoiceCatalogOrdinals(decision).size())
      return invalid("thread choice is outside its H domain");
    choices.push_back(choice);
  }
  for (const auto &[decision, choice] : llvm::enumerate(graphChoices)) {
    if (choice >= problem.graphChoiceCatalogOrdinals(decision).size())
      return invalid("graph choice is outside its H domain");
    choices.push_back(choice);
  }
  return choices;
}

} // namespace

llvm::Expected<SystemCandidateStateHandle>
SystemCandidateState::create(FrozenSystemPnrProblemHandle problem,
                             SystemCandidateInitialization initialization) {
  if (!problem)
    return invalid("FrozenSystemPnrProblem owner is null");
  auto choices = relationChoices(*problem, initialization.threadChoices,
                                 initialization.graphChoices);
  if (!choices)
    return choices.takeError();
  if (llvm::Error error =
          problem->initializerRelations_->verifyChoices(*choices))
    return llvm::joinErrors(
        invalid("thread and graph target classes are incompatible"),
        std::move(error));
  if (llvm::Error error = detail::verifySystemServiceTargetDomains(
          *problem, initialization.threadChoices, initialization.graphChoices))
    return std::move(error);
  if (llvm::Error error = detail::verifySystemServiceRoutes(
          *problem, initialization.threadChoices, initialization.graphChoices,
          initialization.serviceRoutes, initialization.serviceRouteNodes,
          initialization.serviceRouteSinks))
    return std::move(error);
  auto state = SystemCandidateStateHandle(new SystemCandidateState(
      std::move(problem),
      std::vector<PnrIndex>(initialization.threadChoices.begin(),
                            initialization.threadChoices.end()),
      std::vector<PnrIndex>(initialization.graphChoices.begin(),
                            initialization.graphChoices.end()),
      std::vector<SystemServiceRouteSelection>(
          initialization.serviceRoutes.begin(),
          initialization.serviceRoutes.end()),
      std::vector<SystemServiceRouteNodeSelection>(
          initialization.serviceRouteNodes.begin(),
          initialization.serviceRouteNodes.end()),
      std::vector<SystemServiceRouteSinkSelection>(
          initialization.serviceRouteSinks.begin(),
          initialization.serviceRouteSinks.end())));
  if (llvm::Error error = state->verify())
    return std::move(error);
  return state;
}

PnrIndex SystemCandidateState::threadChoice(PnrIndex decision) const {
  assert(decision < threadChoices_.size());
  return threadChoices_[decision];
}

PnrIndex SystemCandidateState::graphChoice(PnrIndex decision) const {
  assert(decision < graphChoices_.size());
  return graphChoices_[decision];
}

::loom::fabric::AccCoreOccurrenceRef
SystemCandidateState::selectedAccCore(PnrIndex decision) const {
  const auto domain = problem_->threadChoiceCatalogOrdinals(decision);
  return problem_->accCores()[domain[threadChoice(decision)]];
}

const ArtifactRootReference &
SystemCandidateState::selectedSpatialMapping(PnrIndex decision) const {
  const auto domain = problem_->graphChoiceCatalogOrdinals(decision);
  return problem_->spatialMappings()[domain[graphChoice(decision)]];
}

llvm::Expected<SystemServiceTargetDomain>
SystemCandidateState::serviceTargetDomain(PnrIndex context) const {
  return detail::resolveSystemServiceTargetDomain(
      *problem_, context, threadChoices_, graphChoices_);
}

llvm::Error SystemCandidateState::verify() const {
  auto choices = relationChoices(*problem_, threadChoices_, graphChoices_);
  if (!choices)
    return choices.takeError();
  if (llvm::Error error = detail::verifySystemServiceTargetDomains(
          *problem_, threadChoices_, graphChoices_))
    return error;
  if (llvm::Error error = detail::verifySystemServiceRoutes(
          *problem_, threadChoices_, graphChoices_, serviceRoutes_,
          serviceRouteNodes_, serviceRouteSinks_))
    return error;

  std::map<std::uint64_t, std::vector<PnrIndex>> threadsByRoot;
  for (const auto &[threadOrdinal, thread] :
       llvm::enumerate(problem_->threadDecisions()))
    threadsByRoot[thread.root.entity.value()].push_back(
        static_cast<PnrIndex>(threadOrdinal));
  for (const auto &[graphOrdinal, graph] :
       llvm::enumerate(problem_->graphDecisions())) {
    const auto graphDomain = problem_->graphChoiceCatalogOrdinals(graphOrdinal);
    const PnrIndex mapping = graphDomain[graphChoices_[graphOrdinal]];
    const PnrIndex mappingClass = problem_->spatialMappingTargetClass(mapping);
    bool intersectsParent = false;
    const auto rootThreads =
        threadsByRoot.find(graph.launch.rootThreadLaunch.entity.value());
    if (rootThreads == threadsByRoot.end())
      return invalid("graph atom has no parent thread domain");
    for (PnrIndex threadOrdinal : rootThreads->second) {
      const auto &thread = problem_->threadDecisions()[threadOrdinal];
      if (thread.root != graph.launch.rootThreadLaunch)
        continue;
      if (!(thread.cell == graph.cell)) {
        auto intersects =
            detail::systemPresburgerCellsIntersect(thread.cell, graph.cell);
        if (!intersects)
          return intersects.takeError();
        if (!*intersects)
          continue;
      }
      intersectsParent = true;
      const auto threadDomain =
          problem_->threadChoiceCatalogOrdinals(threadOrdinal);
      const PnrIndex core = threadDomain[threadChoices_[threadOrdinal]];
      if (problem_->accCoreTargetClass(core) != mappingClass)
        return invalid(
            "selected graph SpatialMapping does not target the selected "
            "AccCore SpatialCore");
    }
    if (!intersectsParent)
      return invalid("graph atom does not intersect its parent thread domain");
  }
  return llvm::Error::success();
}

llvm::Expected<InitializedSystemCandidate>
loom::pnr::initializeCanonicalSystemCandidate(
    FrozenSystemPnrProblemHandle problem) {
  if (!problem)
    return invalid("FrozenSystemPnrProblem owner is null");
  detail::InitializerRelationSolver solver(*problem->initializerRelations_);
  auto solved = solver.solveCanonical(
      problem->config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed);
  if (!solved)
    return solved.takeError();
  const std::size_t threadCount = problem->threadDecisions().size();
  auto state = initializeSystemCandidate(
      problem, llvm::ArrayRef(solved->choices).take_front(threadCount),
      llvm::ArrayRef(solved->choices).drop_front(threadCount));
  if (!state)
    return state.takeError();
  return InitializedSystemCandidate{*state, solved->assignmentAttempts};
}

llvm::Expected<SystemCandidateStateHandle>
loom::pnr::initializeSystemCandidate(FrozenSystemPnrProblemHandle problem,
                                     llvm::ArrayRef<PnrIndex> threadChoices,
                                     llvm::ArrayRef<PnrIndex> graphChoices) {
  if (!problem)
    return invalid("FrozenSystemPnrProblem owner is null");
  auto choices = relationChoices(*problem, threadChoices, graphChoices);
  if (!choices)
    return choices.takeError();
  if (llvm::Error error =
          problem->initializerRelations_->verifyChoices(*choices))
    return llvm::joinErrors(
        invalid("thread and graph target classes are incompatible"),
        std::move(error));
  if (llvm::Error error = detail::verifySystemServiceTargetDomains(
          *problem, threadChoices, graphChoices))
    return std::move(error);
  auto routes = detail::buildCanonicalSystemServiceRoutes(
      *problem, threadChoices, graphChoices);
  if (!routes)
    return routes.takeError();
  return SystemCandidateState::create(
      std::move(problem), {threadChoices, graphChoices, routes->routes,
                           routes->nodes, routes->sinks});
}
