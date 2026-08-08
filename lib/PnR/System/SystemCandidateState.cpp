#include "PnR/System/SystemCandidateState.h"

#include "PnR/EndpointRouter.h"
#include "PnR/InitializerRelationSolver.h"
#include "SystemCandidateServiceResolver.h"
#include "SystemPnrSearchDomainInternal.h"
#include "SystemServiceRouter.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

char SystemCandidateInitializationFailure::ID;

void SystemCandidateInitializationFailure::log(
    llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code
SystemCandidateInitializationFailure::convertToErrorCode() const {
  return std::make_error_code(
      kind_ == SystemCandidateInitializationFailureKind::SemanticLimitReached
          ? std::errc::resource_unavailable_try_again
          : std::errc::invalid_argument);
}

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_candidate_invalid: " + message);
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

llvm::Error initializationFailure(llvm::Error error,
                                  std::uint64_t assignmentAttempts,
                                  std::uint64_t endpointExpansions) {
  SystemCandidateInitializationFailureKind kind =
      SystemCandidateInitializationFailureKind::Internal;
  std::string diagnostic;
  llvm::handleAllErrors(
      std::move(error),
      [&](const detail::InitializerRelationSolveFailure &failure) {
        switch (failure.kind()) {
        case detail::InitializerRelationSolveFailureKind::ProvenInfeasible:
        case detail::InitializerRelationSolveFailureKind::FixedRootInfeasible:
          kind = SystemCandidateInitializationFailureKind::ProvenInfeasible;
          break;
        case detail::InitializerRelationSolveFailureKind::WorkLimit:
          kind = SystemCandidateInitializationFailureKind::SemanticLimitReached;
          break;
        }
        diagnostic = errorMessage(failure);
      },
      [&](const EndpointRouteSearchFailure &failure) {
        kind =
            failure.kind() == EndpointRouteSearchFailureKind::WorkLimit
                ? SystemCandidateInitializationFailureKind::SemanticLimitReached
                : SystemCandidateInitializationFailureKind::Internal;
        diagnostic = errorMessage(failure);
      },
      [&](const llvm::ErrorInfoBase &failure) {
        kind = SystemCandidateInitializationFailureKind::Internal;
        diagnostic = errorMessage(failure);
      });
  return llvm::make_error<SystemCandidateInitializationFailure>(
      kind, assignmentAttempts, endpointExpansions, std::move(diagnostic));
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

struct RequiredInstructionUse final {
  ::dataflow::RootThreadLaunchRef root;
  ::loom::fabric::InstructionCoreContextRef context;
  llvm::ArrayRef<::loom::fabric::FabricUsePatternRef> patterns;
};

llvm::Expected<std::vector<RequiredInstructionUse>>
requiredInstructionUses(const FrozenSystemPnrProblem &problem,
                        llvm::ArrayRef<PnrIndex> threadChoices) {
  std::map<std::string, RequiredInstructionUse> unique;
  for (const auto &[decision, thread] :
       llvm::enumerate(problem.threadDecisions())) {
    const ::loom::fabric::InstructionCoreContextRef context{
        problem.accCores()[problem.threadChoiceCatalogOrdinals(
            decision)[threadChoices[decision]]]};
    const auto domain = llvm::find_if(
        problem.instructionUsePatternDomains(),
        [&](const auto &candidate) { return candidate.context == context; });
    if (domain == problem.instructionUsePatternDomains().end() ||
        domain->patterns.empty())
      return invalid("selected InstructionCore has no use-pattern domain");
    auto rootBytes = ::dataflow::encodeDataflowReference(
        problem.dataflowIdentity(), thread.root);
    if (!rootBytes)
      return rootBytes.takeError();
    std::string key(reinterpret_cast<const char *>(rootBytes->data()),
                    rootBytes->size());
    const auto contextBytes = ::loom::fabric::canonicalFabricBytes(context);
    key.append(reinterpret_cast<const char *>(contextBytes.data()),
               contextBytes.size());
    unique.try_emplace(
        std::move(key),
        RequiredInstructionUse{thread.root, context, domain->patterns});
  }
  std::vector<RequiredInstructionUse> result;
  result.reserve(unique.size());
  for (const auto &[key, use] : unique) {
    (void)key;
    result.push_back(use);
  }
  return result;
}

struct RequiredServiceUse final {
  PnrIndex context = getInvalidPnrIndex();
  PnrIndex subject = getInvalidPnrIndex();
  PnrIndex branch = 0;
  llvm::ArrayRef<::loom::fabric::FabricUsePatternRef> patterns;
};

llvm::Expected<std::vector<RequiredServiceUse>>
requiredServiceUses(const FrozenSystemPnrProblem &problem,
                    llvm::ArrayRef<PnrIndex> threadChoices,
                    llvm::ArrayRef<PnrIndex> graphChoices,
                    llvm::ArrayRef<SystemServiceTargetSelection> targets) {
  std::vector<RequiredServiceUse> result;
  for (const auto &[contextOrdinal, context] :
       llvm::enumerate(problem.serviceContexts())) {
    if (contextOrdinal >= targets.size())
      return invalid("service target closure is incomplete");
    for (const auto &[subjectOrdinal, subject] :
         llvm::enumerate(context.subjects)) {
      if (!std::holds_alternative<SystemServiceMemberTargetSubject>(subject))
        continue;
      if (const auto *plan = std::get_if<SystemMemoryServiceTargetPlan>(
              &targets[contextOrdinal])) {
        auto binding = detail::resolveSystemMemoryServiceBinding(
            problem, static_cast<PnrIndex>(contextOrdinal), subject,
            threadChoices, graphChoices);
        if (!binding)
          return binding.takeError();
        for (const auto &[branchOrdinal, branch] :
             llvm::enumerate(plan->branches)) {
          const auto branchRegion = branch.region;
          const auto domain = llvm::find_if(
              (*binding)->usePatternDomains, [&](const auto &candidate) {
                return candidate.region == branchRegion;
              });
          if (domain == (*binding)->usePatternDomains.end() ||
              domain->patterns.empty())
            return invalid(
                "selected memory target has no admissible use pattern");
          result.push_back({static_cast<PnrIndex>(contextOrdinal),
                            static_cast<PnrIndex>(subjectOrdinal),
                            static_cast<PnrIndex>(branchOrdinal),
                            domain->patterns});
        }
        continue;
      }
      const auto *consistency =
          std::get_if<::loom::fabric::MemoryConsistencyDomainRef>(
              &targets[contextOrdinal]);
      if (!consistency)
        continue;
      const auto domain = llvm::find_if(
          problem.consistencyUsePatternDomains(), [&](const auto &candidate) {
            return candidate.domain == *consistency;
          });
      if (domain == problem.consistencyUsePatternDomains().end() ||
          domain->patterns.empty())
        return invalid(
            "selected consistency target has no admissible use pattern");
      result.push_back({static_cast<PnrIndex>(contextOrdinal),
                        static_cast<PnrIndex>(subjectOrdinal), 0,
                        domain->patterns});
    }
  }
  return result;
}

llvm::Expected<std::vector<SystemInstructionResourceUseSelection>>
selectCanonicalInstructionUses(const FrozenSystemPnrProblem &problem,
                               llvm::ArrayRef<PnrIndex> threadChoices) {
  auto required = requiredInstructionUses(problem, threadChoices);
  if (!required)
    return required.takeError();
  std::vector<SystemInstructionResourceUseSelection> result;
  result.reserve(required->size());
  for (const auto &use : *required)
    result.push_back({use.root, use.context, use.patterns.front()});
  return result;
}

llvm::Expected<std::vector<SystemServiceResourceUseSelection>>
selectCanonicalServiceUses(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceTargetSelection> targets) {
  auto required =
      requiredServiceUses(problem, threadChoices, graphChoices, targets);
  if (!required)
    return required.takeError();
  std::vector<SystemServiceResourceUseSelection> result;
  result.reserve(required->size());
  for (const auto &use : *required)
    result.push_back(
        {use.context, use.subject, use.branch, use.patterns.front()});
  return result;
}

llvm::Error verifyResourceUses(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceTargetSelection> targets,
    llvm::ArrayRef<SystemInstructionResourceUseSelection> instructionUses,
    llvm::ArrayRef<SystemServiceResourceUseSelection> serviceUses) {
  auto requiredInstructions = requiredInstructionUses(problem, threadChoices);
  if (!requiredInstructions)
    return requiredInstructions.takeError();
  if (instructionUses.size() != requiredInstructions->size())
    return invalid("InstructionCore ResourceUse count is incomplete");
  for (const auto &[selected, required] :
       llvm::zip_equal(instructionUses, *requiredInstructions)) {
    if (selected.root != required.root ||
        selected.context != required.context ||
        !llvm::is_contained(required.patterns, selected.pattern))
      return invalid("InstructionCore ResourceUse is foreign or inadmissible");
  }
  auto requiredServices =
      requiredServiceUses(problem, threadChoices, graphChoices, targets);
  if (!requiredServices)
    return requiredServices.takeError();
  if (serviceUses.size() != requiredServices->size())
    return invalid("service ResourceUse count is incomplete");
  for (const auto &[selected, required] :
       llvm::zip_equal(serviceUses, *requiredServices)) {
    if (selected.context != required.context ||
        selected.subject != required.subject ||
        selected.branch != required.branch ||
        !llvm::is_contained(required.patterns, selected.pattern))
      return invalid("service ResourceUse is foreign or inadmissible");
  }
  return llvm::Error::success();
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
  std::vector<SystemInstructionResourceUseSelection> instructionUses(
      initialization.instructionResourceUses.begin(),
      initialization.instructionResourceUses.end());
  std::vector<SystemServiceResourceUseSelection> serviceUses(
      initialization.serviceResourceUses.begin(),
      initialization.serviceResourceUses.end());
  if (llvm::Error error = verifyResourceUses(
          *problem, initialization.threadChoices, initialization.graphChoices,
          initialization.serviceTargets, instructionUses, serviceUses))
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
          initialization.serviceTargets.end()),
      std::move(instructionUses), std::move(serviceUses)));
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
  if (llvm::Error error = verifyResourceUses(
          *problem_, threadChoices_, graphChoices_, serviceTargets_,
          instructionResourceUses_, serviceResourceUses_))
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
  std::uint64_t endpointExpansions = 0;
  auto solved = solver.solveCanonical(
      problem->config()
          .policy()
          .search.initializer.assignmentAttemptLimitPerSeed,
      [&](llvm::ArrayRef<PnrIndex> choices) -> llvm::Expected<bool> {
        const std::size_t threadCount = problem->threadDecisions().size();
        std::uint64_t candidateEndpointExpansions = 0;
        auto candidate = initializeSystemCandidate(
            problem, choices.take_front(threadCount),
            choices.drop_front(threadCount), &candidateEndpointExpansions);
        if (candidateEndpointExpansions >
            std::numeric_limits<std::uint64_t>::max() - endpointExpansions)
          return llvm::createStringError(
              std::make_error_code(std::errc::value_too_large),
              "System initializer endpoint expansion accounting overflow");
        endpointExpansions += candidateEndpointExpansions;
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
    return initializationFailure(
        solved.takeError(), solver.assignmentAttempts(), endpointExpansions);
  if (!accepted)
    return llvm::make_error<SystemCandidateInitializationFailure>(
        SystemCandidateInitializationFailureKind::Internal,
        solved->assignmentAttempts, endpointExpansions,
        "initializer accepted no System candidate");
  return InitializedSystemCandidate{
      std::move(accepted), solved->assignmentAttempts, endpointExpansions};
}

llvm::Expected<SystemCandidateStateHandle>
loom::pnr::initializeSystemCandidate(FrozenSystemPnrProblemHandle problem,
                                     llvm::ArrayRef<PnrIndex> threadChoices,
                                     llvm::ArrayRef<PnrIndex> graphChoices,
                                     std::uint64_t *endpointExpansions) {
  if (endpointExpansions)
    *endpointExpansions = 0;
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
  std::uint64_t routeEndpointExpansions = 0;
  auto routes = detail::buildCanonicalSystemServiceRoutes(
      *problem, threadChoices, graphChoices, routeEndpointExpansions);
  if (endpointExpansions)
    *endpointExpansions = routeEndpointExpansions;
  if (!routes)
    return routes.takeError();
  auto targets =
      selectCanonicalServiceTargets(*problem, threadChoices, graphChoices);
  if (!targets)
    return targets.takeError();
  auto instructionUses =
      selectCanonicalInstructionUses(*problem, threadChoices);
  if (!instructionUses)
    return instructionUses.takeError();
  auto serviceUses = selectCanonicalServiceUses(*problem, threadChoices,
                                                graphChoices, *targets);
  if (!serviceUses)
    return serviceUses.takeError();
  return SystemCandidateState::create(
      std::move(problem),
      {threadChoices, graphChoices, routes->routes, routes->nodes,
       routes->sinks, *targets, *instructionUses, *serviceUses});
}
