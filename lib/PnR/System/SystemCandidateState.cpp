#include "PnR/System/SystemCandidateState.h"

#include "PnR/InitializerRelationSolver.h"
#include "SystemCandidateServiceResolver.h"
#include "SystemPnrSearchDomainInternal.h"
#include "SystemServiceRouter.h"

#include "llvm/ADT/STLExtras.h"
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

bool targetInDomain(const SystemServiceTargetSelection &target,
                    const SystemServiceTargetDomain &domain) {
  if (const auto *plan = std::get_if<SystemMemoryServiceTargetPlan>(&target)) {
    const auto *plans =
        std::get_if<std::vector<SystemMemoryServiceTargetPlan>>(&domain);
    return plans && llvm::is_contained(*plans, *plan);
  }
  if (const auto *consistency =
          std::get_if<::loom::fabric::MemoryConsistencyDomainRef>(&target)) {
    const auto *domains =
        std::get_if<std::vector<::loom::fabric::MemoryConsistencyDomainRef>>(
            &domain);
    return domains && llvm::is_contained(*domains, *consistency);
  }
  return false;
}

llvm::Error
verifyServiceTargets(const FrozenSystemPnrProblem &problem,
                     llvm::ArrayRef<PnrIndex> threadChoices,
                     llvm::ArrayRef<PnrIndex> graphChoices,
                     llvm::ArrayRef<SystemServiceTargetSelection> targets) {
  if (targets.size() != problem.serviceContexts().size())
    return invalid("service target count does not match its context closure");
  for (const auto &[ordinal, context] :
       llvm::enumerate(problem.serviceContexts())) {
    if (context.service >= problem.serviceDomains().size())
      return invalid("service target has no H service domain");
    const bool transfer =
        std::holds_alternative<::loom::mapping::TransferObligationFamilyKey>(
            problem.serviceDomains()[context.service].key);
    const auto &target = targets[ordinal];
    if (transfer) {
      if (!std::holds_alternative<std::monostate>(target))
        return invalid("transfer service context selects an operation target");
      continue;
    }
    auto domain = detail::resolveSystemServiceTargetDomain(
        problem, static_cast<PnrIndex>(ordinal), threadChoices, graphChoices);
    if (!domain)
      return domain.takeError();
    if (!targetInDomain(target, *domain))
      return invalid("selected service target is outside its exact H domain");
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<SystemServiceTargetSelection>>
selectCanonicalServiceTargets(const FrozenSystemPnrProblem &problem,
                              llvm::ArrayRef<PnrIndex> threadChoices,
                              llvm::ArrayRef<PnrIndex> graphChoices) {
  std::vector<SystemServiceTargetSelection> result;
  result.reserve(problem.serviceContexts().size());
  for (const auto &[ordinal, context] :
       llvm::enumerate(problem.serviceContexts())) {
    if (context.service >= problem.serviceDomains().size())
      return invalid("service target has no H service domain");
    if (std::holds_alternative<::loom::mapping::TransferObligationFamilyKey>(
            problem.serviceDomains()[context.service].key)) {
      result.emplace_back(std::monostate{});
      continue;
    }
    auto domain = detail::resolveSystemServiceTargetDomain(
        problem, static_cast<PnrIndex>(ordinal), threadChoices, graphChoices);
    if (!domain)
      return domain.takeError();
    std::visit(
        [&](const auto &values) {
          assert(!values.empty());
          result.emplace_back(values.front());
        },
        *domain);
  }
  return result;
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
  if (llvm::Error error = verifyServiceTargets(
          *problem, initialization.threadChoices, initialization.graphChoices,
          initialization.serviceTargets))
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
          initialization.serviceRouteSinks.end()),
      std::vector<SystemServiceTargetSelection>(
          initialization.serviceTargets.begin(),
          initialization.serviceTargets.end())));
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

const SystemServiceTargetSelection &
SystemCandidateState::serviceTarget(PnrIndex context) const {
  assert(context < serviceTargets_.size());
  return serviceTargets_[context];
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
  if (llvm::Error error = verifyServiceTargets(*problem_, threadChoices_,
                                               graphChoices_, serviceTargets_))
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
  SystemCandidateStateHandle accepted;
  auto solved = solver.solveCanonical(
      problem->config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed,
      [&](llvm::ArrayRef<PnrIndex> choices) -> llvm::Expected<bool> {
        const std::size_t threadCount = problem->threadDecisions().size();
        auto candidate =
            initializeSystemCandidate(problem, choices.take_front(threadCount),
                                      choices.drop_front(threadCount));
        if (candidate) {
          accepted = std::move(*candidate);
          return true;
        }
        bool infeasible = false;
        llvm::Error remaining =
            llvm::handleErrors(candidate.takeError(),
                               [&](const detail::SystemCandidateInfeasible &) {
                                 infeasible = true;
                               });
        if (remaining)
          return std::move(remaining);
        if (!infeasible)
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "System candidate rejection lost its cause");
        return false;
      });
  if (!solved)
    return solved.takeError();
  if (!accepted)
    return invalid("initializer accepted no System candidate");
  return InitializedSystemCandidate{std::move(accepted),
                                    solved->assignmentAttempts};
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
  auto targets =
      selectCanonicalServiceTargets(*problem, threadChoices, graphChoices);
  if (!targets)
    return targets.takeError();
  return SystemCandidateState::create(
      std::move(problem), {threadChoices, graphChoices, routes->routes,
                           routes->nodes, routes->sinks, *targets});
}
