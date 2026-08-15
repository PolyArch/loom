#include "PnR/System/SystemCandidateState.h"

#include "Common/MappingDebugLog.h"
#include "PnR/EndpointRouter.h"
#include "PnR/InitializerRelationSolver.h"
#include "SystemCandidateMutation.h"
#include "SystemCandidateServiceResolver.h"
#include "SystemCapacityProjection.h"
#include "SystemNegotiatedRouter.h"
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
#include <type_traits>
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
                                  std::uint64_t endpointExpansions,
                                  std::uint64_t negotiationIterations) {
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
      [&](const detail::SystemRoutingClosureFailure &failure) {
        kind = SystemCandidateInitializationFailureKind::SemanticLimitReached;
        diagnostic = errorMessage(failure);
      },
      [&](const llvm::ErrorInfoBase &failure) {
        kind = SystemCandidateInitializationFailureKind::Internal;
        diagnostic = errorMessage(failure);
      });
  return llvm::make_error<SystemCandidateInitializationFailure>(
      kind, assignmentAttempts, endpointExpansions, negotiationIterations,
      std::move(diagnostic));
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

struct RouteCapacityOveruseProjection final {
  std::uint64_t total = 0;
  std::vector<SystemRouteCapacityOveruseWitness> witnesses;
};

llvm::Expected<RouteCapacityOveruseProjection> projectRouteCapacityOveruse(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<SystemServiceRouteSelection> routes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> nodes,
    llvm::ArrayRef<SystemServiceRouteSinkSelection> sinks) {
  const FrozenEndpointRoutingTopology &topology = problem.routingTopology();
  auto usage = detail::measureSystemServiceRouteCapacityUsage(
      topology, {routes, nodes, sinks}, /*enforceCapacity=*/false);
  if (!usage)
    return usage.takeError();
  if (usage->size() != topology.capacityCells().size())
    return invalid("route capacity projection has the wrong width");
  RouteCapacityOveruseProjection result;
  for (PnrIndex cell = 0; cell < usage->size(); ++cell) {
    const std::uint64_t capacity = topology.capacityCells()[cell].capacity;
    if ((*usage)[cell] <= capacity)
      continue;
    const std::uint64_t overuse = (*usage)[cell] - capacity;
    if (overuse > std::numeric_limits<std::uint64_t>::max() - result.total)
      return invalid("route CapacityOveruse exceeds u64");
    result.total += overuse;
    result.witnesses.push_back({cell, (*usage)[cell], capacity, overuse});
  }
  return result;
}

bool admitsCapacityOveruse(const FrozenSystemPnrProblem &problem) {
  return llvm::is_contained(
      problem.config().policy().temporaryViolations.admitted,
      ResolvedPnrViolationKind::CapacityOveruse);
}

} // namespace

llvm::Expected<SystemCandidateStateHandle>
SystemCandidateState::create(FrozenSystemPnrProblemHandle problem,
                             SystemCandidateInitialization initialization) {
  return createImpl(std::move(problem), initialization, nullptr, std::nullopt,
                    true);
}

llvm::Expected<SystemCandidateStateHandle> SystemCandidateState::createMutation(
    const SystemCandidateState &source,
    SystemCandidateInitialization initialization,
    SystemCandidateMutationDomain domain) {
  return createImpl(source.problemHandle(), initialization, &source, domain,
                    false);
}

llvm::Expected<SystemCandidateStateHandle> SystemCandidateState::createImpl(
    FrozenSystemPnrProblemHandle problem,
    SystemCandidateInitialization initialization,
    const SystemCandidateState *source,
    std::optional<SystemCandidateMutationDomain> domain, bool runFullOracle) {
  if (!problem)
    return invalid("FrozenSystemPnrProblem owner is null");
  if ((source == nullptr) != !domain)
    return invalid("mutation source and domain must be supplied together");
  if (source && source->problemHandle() != problem)
    return invalid("mutation source has a foreign FrozenSystemPnrProblem");
  auto choices = relationChoices(*problem, initialization.threadChoices,
                                 initialization.graphChoices);
  if (!choices)
    return choices.takeError();
  if (llvm::Error error =
          problem->initializerRelations().verifyChoices(*choices))
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
  auto routeCapacity = projectRouteCapacityOveruse(
      *problem, initialization.serviceRoutes, initialization.serviceRouteNodes,
      initialization.serviceRouteSinks);
  if (!routeCapacity)
    return routeCapacity.takeError();
  const detail::SystemCandidateCapacityProjectionView capacityView{
      initialization.threadChoices,
      initialization.graphChoices,
      initialization.serviceRoutes,
      initialization.serviceRouteNodes,
      initialization.serviceRouteSinks,
      instructionUses,
      serviceUses};
  llvm::Expected<detail::SystemCandidateProjectionResult> capacity =
      source && *domain == SystemCandidateMutationDomain::TransportRoutes
          ? problem->capacityModel().projectRouteDelta(
                *problem, capacityView, source->projectionCache())
      : source
          ? problem->capacityModel().projectResourceDelta(
                *problem, capacityView, source->projectionCache())
          : problem->capacityModel().projectWithCache(*problem, capacityView);
  if (!capacity)
    return capacity.takeError();
  auto recurrence = projectSystemRecurrenceTiming(
      *problem, initialization.graphChoices);
  if (!recurrence)
    return recurrence.takeError();
  if (capacity->demand.capacity.total != 0 && !admitsCapacityOveruse(*problem))
    return llvm::make_error<detail::SystemCandidateInfeasible>(
        "CapacityOveruse is not policy-admitted");
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
      std::move(instructionUses), std::move(serviceUses),
      std::move(capacity->cache), capacity->demand.capacity.total,
      capacity->demand.progress, std::move(*recurrence),
      capacity->demand.timing.minimumInitiationIntervalCycles,
      capacity->demand.timing.transportBitCycleDemand, routeCapacity->total,
      std::move(routeCapacity->witnesses)));
  if (runFullOracle)
    if (llvm::Error error = state->verify())
      return std::move(error);
  return state;
}

SystemCandidateState::~SystemCandidateState() = default;

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
  auto routeCapacity = projectRouteCapacityOveruse(
      *problem_, serviceRoutes_, serviceRouteNodes_, serviceRouteSinks_);
  if (!routeCapacity)
    return routeCapacity.takeError();
  if (routeCapacity->total != routeCapacityOveruse_ ||
      routeCapacity->witnesses != routeCapacityOveruseWitnesses_)
    return invalid("cached route CapacityOveruse projection diverged");
  auto capacity = problem_->capacityModel().project(
      *problem_,
      {threadChoices_, graphChoices_, serviceRoutes_, serviceRouteNodes_,
       serviceRouteSinks_, instructionResourceUses_, serviceResourceUses_});
  if (!capacity)
    return capacity.takeError();
  if (capacity->capacity.total != capacityOveruse_)
    return invalid("cached CapacityOveruse projection diverged");
  if (capacity->progress.kind != progressClosure_.kind)
    return invalid("cached progress projection diverged");
  auto recurrence = projectSystemRecurrenceTiming(*problem_, graphChoices_);
  if (!recurrence)
    return recurrence.takeError();
  if (!(*recurrence == recurrenceTiming_))
    return invalid("cached recurrence timing projection diverged");
  if (capacity->timing.minimumInitiationIntervalCycles !=
          resourceMinimumInitiationIntervalCycles_ ||
      capacity->timing.transportBitCycleDemand != transportBitCycleDemand_)
    return invalid("cached intrinsic timing projection diverged");
  if (capacityOveruse_ != 0 && !admitsCapacityOveruse(*problem_))
    return invalid("CapacityOveruse is not policy-admitted");

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

namespace {

llvm::Expected<SystemCandidateStateHandle> initializeSystemCandidateWithClosure(
    FrozenSystemPnrProblemHandle problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    detail::SystemRoutingClosureRequirement closureRequirement,
    std::uint64_t *endpointExpansions, std::uint64_t *negotiationIterations);

template <typename Solve>
llvm::Expected<InitializedSystemCandidate>
solveSystemCandidate(FrozenSystemPnrProblemHandle problem, Solve &&solve) {
  detail::InitializerRelationSolver solver(problem->initializerRelations());
  SystemCandidateStateHandle accepted;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
  const auto validate =
      [&](llvm::ArrayRef<PnrIndex> choices) -> llvm::Expected<bool> {
    const std::size_t threadCount = problem->threadDecisions().size();
    const auto emitChoice = [&](llvm::StringRef status,
                                llvm::StringRef diagnostic = {}) {
      mapping_debug::emit(
          mapping_debug::Level::Decision, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::ContextChoice, [&](llvm::json::Object &fields) {
            llvm::json::Array threadChoices;
            for (PnrIndex choice : choices.take_front(threadCount))
              threadChoices.push_back(choice);
            llvm::json::Array graphChoices;
            for (PnrIndex choice : choices.drop_front(threadCount))
              graphChoices.push_back(choice);
            fields["operation"] = "initializer_execution_assignment";
            fields["assignment_attempt"] = solver.assignmentAttempts();
            fields["thread_choices"] = std::move(threadChoices);
            fields["graph_choices"] = std::move(graphChoices);
            fields["status"] = status;
            if (!diagnostic.empty())
              fields["diagnostic"] = diagnostic;
          });
    };
    std::uint64_t candidateEndpointExpansions = 0;
    std::uint64_t candidateNegotiationIterations = 0;
    auto candidate = initializeSystemCandidateWithClosure(
        problem, choices.take_front(threadCount),
        choices.drop_front(threadCount),
        detail::SystemRoutingClosureRequirement::Strict,
        &candidateEndpointExpansions, &candidateNegotiationIterations);
    if (candidateEndpointExpansions >
        std::numeric_limits<std::uint64_t>::max() - endpointExpansions)
      return llvm::createStringError(
          std::make_error_code(std::errc::value_too_large),
          "System initializer endpoint expansion accounting overflow");
    endpointExpansions += candidateEndpointExpansions;
    if (candidateNegotiationIterations >
        std::numeric_limits<std::uint64_t>::max() - negotiationIterations)
      return llvm::createStringError(
          std::make_error_code(std::errc::value_too_large),
          "System initializer negotiation iteration accounting overflow");
    negotiationIterations += candidateNegotiationIterations;
    if (candidate) {
      emitChoice("accepted");
      accepted = std::move(*candidate);
      return true;
    }
    bool infeasible = false;
    std::string diagnostic;
    llvm::Error remaining = llvm::handleErrors(
        candidate.takeError(),
        [&](const detail::SystemCandidateInfeasible &failure) {
          infeasible = true;
          diagnostic = errorMessage(failure);
        },
        [&](const detail::SystemRoutingClosureFailure &failure) -> llvm::Error {
          if (failure.kind() == detail::SystemRoutingClosureFailureKind::
                                    FixedTerminalCapacityCut) {
            infeasible = true;
            diagnostic = errorMessage(failure);
            return llvm::Error::success();
          }
          return llvm::make_error<detail::SystemRoutingClosureFailure>(
              failure.kind(), errorMessage(failure));
        });
    if (remaining)
      return std::move(remaining);
    if (!infeasible)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "System candidate rejection lost its cause");
    emitChoice("rejected", diagnostic);
    return false;
  };
  auto solved = solve(solver, validate);
  if (!solved)
    return initializationFailure(solved.takeError(),
                                 solver.assignmentAttempts(),
                                 endpointExpansions, negotiationIterations);
  if (!accepted)
    return llvm::make_error<SystemCandidateInitializationFailure>(
        SystemCandidateInitializationFailureKind::Internal,
        solved->assignmentAttempts, endpointExpansions, negotiationIterations,
        "initializer accepted no System candidate");
  return InitializedSystemCandidate{std::move(accepted),
                                    solved->assignmentAttempts,
                                    endpointExpansions, negotiationIterations};
}

} // namespace

llvm::Expected<InitializedSystemCandidate>
loom::pnr::initializeCanonicalSystemCandidate(
    FrozenSystemPnrProblemHandle problem) {
  return initializeSystemCandidateAttempt(std::move(problem), 0);
}

llvm::Expected<InitializedSystemCandidate>
loom::pnr::initializeSystemCandidateAttempt(
    FrozenSystemPnrProblemHandle problem, std::uint32_t attemptOrdinal) {
  if (!problem)
    return invalid("FrozenSystemPnrProblem owner is null");
  const auto &policy = problem->config().policy();
  if (attemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return invalid("System initializer attempt ordinal is out of range");
  return solveSystemCandidate(
      problem, [&](detail::InitializerRelationSolver &solver,
                   auto validateCompleteAssignment) {
        if (attemptOrdinal == 0)
          return solver.solveCanonical(
              policy.search.initializer.assignmentAttemptLimitPerSeed,
              validateCompleteAssignment);
        auto stream = DeterministicPnrRandomStream::create(
            policy.determinism.masterSeed, attemptOrdinal,
            PnrRandomStreamPurpose::InitializerDiversification);
        return solver.solveDiversified(
            policy.search.initializer.assignmentAttemptLimitPerSeed, stream,
            validateCompleteAssignment);
      });
}

llvm::Expected<InitializedSystemCandidate>
loom::pnr::initializeSystemCandidateWithFixedChoices(
    FrozenSystemPnrProblemHandle problem,
    llvm::ArrayRef<PnrIndex> fixedChoices) {
  if (!problem)
    return invalid("FrozenSystemPnrProblem owner is null");
  return solveSystemCandidate(
      problem, [&](detail::InitializerRelationSolver &solver,
                   auto validateCompleteAssignment) {
        return solver.solveCanonicalWithFixedChoices(
            problem->config()
                .policy()
                .search.initializer.assignmentAttemptLimitPerSeed,
            fixedChoices, validateCompleteAssignment);
      });
}

namespace {

llvm::Expected<SystemCandidateStateHandle> initializeSystemCandidateWithClosure(
    FrozenSystemPnrProblemHandle problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    detail::SystemRoutingClosureRequirement closureRequirement,
    std::uint64_t *endpointExpansions, std::uint64_t *negotiationIterations) {
  if (endpointExpansions)
    *endpointExpansions = 0;
  if (negotiationIterations)
    *negotiationIterations = 0;
  if (!problem)
    return invalid("FrozenSystemPnrProblem owner is null");
  auto choices = relationChoices(*problem, threadChoices, graphChoices);
  if (!choices)
    return choices.takeError();
  if (llvm::Error error =
          problem->initializerRelations().verifyChoices(*choices))
    return llvm::joinErrors(
        invalid("thread and graph target classes are incompatible"),
        std::move(error));
  if (llvm::Error error = detail::verifySystemServiceTargetDomains(
          *problem, threadChoices, graphChoices))
    return std::move(error);
  for (PnrIndex leg = 0; leg < problem->serviceLegs().size(); ++leg) {
    const FrozenSystemServiceLeg &record = problem->serviceLegs()[leg];
    auto source = detail::resolveSystemServiceTerminalDomain(
        *problem, leg, record.sourceTerminal, threadChoices, graphChoices);
    if (!source)
      return source.takeError();
    for (PnrIndex terminal : problem->serviceLegSinkTerminals(leg)) {
      auto sink = detail::resolveSystemServiceTerminalDomain(
          *problem, leg, terminal, threadChoices, graphChoices);
      if (!sink)
        return sink.takeError();
    }
  }
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
  std::uint64_t routeEndpointExpansions = 0;
  std::uint64_t routeNegotiationIterations = 0;
  auto routes = detail::negotiateSystemServiceRoutes(
      *problem, threadChoices, graphChoices, *instructionUses, *serviceUses,
      routeEndpointExpansions, routeNegotiationIterations, {}, std::nullopt,
      std::nullopt, std::nullopt, closureRequirement);
  if (endpointExpansions)
    *endpointExpansions = routeEndpointExpansions;
  if (negotiationIterations)
    *negotiationIterations = routeNegotiationIterations;
  if (!routes)
    return routes.takeError();
  return SystemCandidateState::create(
      std::move(problem),
      {threadChoices, graphChoices, routes->routes, routes->nodes,
       routes->sinks, *targets, *instructionUses, *serviceUses});
}

} // namespace

llvm::Expected<SystemCandidateStateHandle>
loom::pnr::initializeSystemCandidate(FrozenSystemPnrProblemHandle problem,
                                     llvm::ArrayRef<PnrIndex> threadChoices,
                                     llvm::ArrayRef<PnrIndex> graphChoices,
                                     std::uint64_t *endpointExpansions,
                                     std::uint64_t *negotiationIterations) {
  return initializeSystemCandidateWithClosure(
      std::move(problem), threadChoices, graphChoices,
      detail::SystemRoutingClosureRequirement::PolicyAdmittedTemporary,
      endpointExpansions, negotiationIterations);
}

llvm::Expected<std::vector<SystemServiceTargetSelection>>
loom::pnr::detail::systemServiceTargetChoices(
    const SystemCandidateState &candidate, PnrIndex context) {
  if (context >= candidate.problem().serviceContexts().size())
    return invalid("service target Action context is out of range");
  const FrozenSystemServiceContext &record =
      candidate.problem().serviceContexts()[context];
  if (record.service >= candidate.problem().serviceDomains().size())
    return invalid("service target Action context has no H service domain");
  if (std::holds_alternative<::loom::mapping::TransferObligationFamilyKey>(
          candidate.problem().serviceDomains()[record.service].key))
    return std::vector<SystemServiceTargetSelection>{std::monostate{}};
  auto domain = candidate.serviceTargetDomain(context);
  if (!domain)
    return domain.takeError();
  return std::visit(
      [](const auto &values) {
        return std::vector<SystemServiceTargetSelection>(values.begin(),
                                                         values.end());
      },
      *domain);
}

llvm::Expected<std::vector<::loom::fabric::FabricUsePatternRef>>
loom::pnr::detail::systemInstructionUsePatternChoices(
    const SystemCandidateState &candidate, PnrIndex use) {
  auto required =
      requiredInstructionUses(candidate.problem(), candidate.threadChoices());
  if (!required)
    return required.takeError();
  if (use >= required->size())
    return invalid("InstructionCore ResourceUse Action is out of range");
  return std::vector<::loom::fabric::FabricUsePatternRef>(
      (*required)[use].patterns.begin(), (*required)[use].patterns.end());
}

llvm::Expected<std::vector<::loom::fabric::FabricUsePatternRef>>
loom::pnr::detail::systemServiceUsePatternChoices(
    const SystemCandidateState &candidate, PnrIndex use) {
  auto required =
      requiredServiceUses(candidate.problem(), candidate.threadChoices(),
                          candidate.graphChoices(), candidate.serviceTargets());
  if (!required)
    return required.takeError();
  if (use >= required->size())
    return invalid("service ResourceUse Action is out of range");
  return std::vector<::loom::fabric::FabricUsePatternRef>(
      (*required)[use].patterns.begin(), (*required)[use].patterns.end());
}

llvm::Expected<SystemCandidateStateHandle>
loom::pnr::detail::rebuildSystemCandidateWithServiceTarget(
    const SystemCandidateState &candidate, PnrIndex context, PnrIndex choice) {
  auto choices = systemServiceTargetChoices(candidate, context);
  if (!choices)
    return choices.takeError();
  if (choice >= choices->size())
    return invalid("service target Action choice is out of range");
  std::vector<SystemServiceTargetSelection> targets(
      candidate.serviceTargets().begin(), candidate.serviceTargets().end());
  targets[context] = (*choices)[choice];
  auto serviceUses =
      selectCanonicalServiceUses(candidate.problem(), candidate.threadChoices(),
                                 candidate.graphChoices(), targets);
  if (!serviceUses)
    return serviceUses.takeError();
  return SystemCandidateState::createMutation(
      candidate,
      {candidate.threadChoices(), candidate.graphChoices(),
       candidate.serviceRoutes(), candidate.serviceRouteNodes(),
       candidate.serviceRouteSinks(), targets,
       candidate.instructionResourceUses(), *serviceUses},
      SystemCandidateMutationDomain::ResourceSelection);
}

llvm::Expected<SystemCandidateStateHandle>
loom::pnr::detail::rebuildSystemCandidateWithInstructionUsePattern(
    const SystemCandidateState &candidate, PnrIndex use, PnrIndex choice) {
  auto choices = systemInstructionUsePatternChoices(candidate, use);
  if (!choices)
    return choices.takeError();
  if (choice >= choices->size())
    return invalid("InstructionCore ResourceUse Action choice is out of range");
  std::vector<SystemInstructionResourceUseSelection> instructionUses(
      candidate.instructionResourceUses().begin(),
      candidate.instructionResourceUses().end());
  instructionUses[use].pattern = (*choices)[choice];
  return SystemCandidateState::createMutation(
      candidate,
      {candidate.threadChoices(), candidate.graphChoices(),
       candidate.serviceRoutes(), candidate.serviceRouteNodes(),
       candidate.serviceRouteSinks(), candidate.serviceTargets(),
       instructionUses, candidate.serviceResourceUses()},
      SystemCandidateMutationDomain::ResourceSelection);
}

llvm::Expected<SystemCandidateStateHandle>
loom::pnr::detail::rebuildSystemCandidateWithServiceUsePattern(
    const SystemCandidateState &candidate, PnrIndex use, PnrIndex choice) {
  auto choices = systemServiceUsePatternChoices(candidate, use);
  if (!choices)
    return choices.takeError();
  if (choice >= choices->size())
    return invalid("service ResourceUse Action choice is out of range");
  std::vector<SystemServiceResourceUseSelection> serviceUses(
      candidate.serviceResourceUses().begin(),
      candidate.serviceResourceUses().end());
  serviceUses[use].pattern = (*choices)[choice];
  return SystemCandidateState::createMutation(
      candidate,
      {candidate.threadChoices(), candidate.graphChoices(),
       candidate.serviceRoutes(), candidate.serviceRouteNodes(),
       candidate.serviceRouteSinks(), candidate.serviceTargets(),
       candidate.instructionResourceUses(), serviceUses},
      SystemCandidateMutationDomain::ResourceSelection);
}

llvm::Expected<SystemCandidateStateHandle>
loom::pnr::detail::rebuildSystemCandidateRoutes(
    const SystemCandidateState &candidate,
    const SystemTransportRoutingAction &action,
    std::uint64_t &endpointExpansions, std::uint64_t &negotiationIterations,
    bool requireCapacityClosure,
    std::optional<SystemRoutingReopenWitness> *reopenWitness) {
  std::optional<SystemServiceRouteTraversalExclusion> exclusion;
  std::vector<PnrIndex> reroutedLegs;
  std::optional<SystemServiceRouteRepairRegion> repairRegion;
  if (llvm::Error error = std::visit(
          [&](const auto &value) -> llvm::Error {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, SystemGlobalRoutingAction>) {
              return llvm::Error::success();
            } else if constexpr (std::is_same_v<
                                     T, SystemWitnessRegionRoutingAction>) {
              if (value.witnessKind !=
                  ResolvedPnrViolationKind::CapacityOveruse)
                return invalid(
                    "WitnessRegion Action has no live System witness");
              const auto witness = llvm::find_if(
                  candidate.routeCapacityOveruseWitnesses(),
                  [&](const SystemRouteCapacityOveruseWitness &record) {
                    return record.capacityCell == value.witnessOrdinal;
                  });
              if (witness == candidate.routeCapacityOveruseWitnesses().end())
                return invalid(
                    "WitnessRegion Action has no live System witness");
              const FrozenEndpointRoutingTopology &topology =
                  candidate.problem().routingTopology();
              for (const SystemServiceRouteSelection &route :
                   candidate.serviceRoutes()) {
                const auto nodes = candidate.serviceRouteNodes().slice(
                    route.nodeOffset, route.nodeCount);
                const bool participates = llvm::any_of(
                    nodes, [&](const SystemServiceRouteNodeSelection &node) {
                      if (node.incomingTraversal == getInvalidPnrIndex())
                        return false;
                      if (node.incomingTraversal >=
                          topology.traversals().size())
                        return false;
                      const EndpointRoutingTraversal &traversal =
                          topology.traversals()[node.incomingTraversal];
                      if (traversal.capacityClaimOffset >
                              topology.capacityClaims().size() ||
                          traversal.capacityClaimCount >
                              topology.capacityClaims().size() -
                                  traversal.capacityClaimOffset)
                        return false;
                      return llvm::any_of(
                          topology.capacityClaims().slice(
                              traversal.capacityClaimOffset,
                              traversal.capacityClaimCount),
                          [&](const EndpointRoutingCapacityClaim &claim) {
                            return claim.cell == witness->capacityCell;
                          });
                    });
                if (participates)
                  reroutedLegs.push_back(route.leg);
              }
              if (reroutedLegs.empty())
                return invalid(
                    "WitnessRegion Action has no selected service leg");
              return llvm::Error::success();
            } else {
              const auto route = llvm::find_if(
                  candidate.serviceRoutes(), [&](const auto &candidateRoute) {
                    return candidateRoute.leg == value.leg;
                  });
              if (route == candidate.serviceRoutes().end())
                return invalid("routing Action names a foreign service leg");
              const auto nodes = candidate.serviceRouteNodes().slice(
                  route->nodeOffset, route->nodeCount);
              if constexpr (std::is_same_v<T, SystemWholeLegRoutingAction>) {
                const auto selected =
                    llvm::find_if(nodes, [](const auto &node) {
                      return node.incomingTraversal != getInvalidPnrIndex();
                    });
                if (selected == nodes.end())
                  return invalid("WholeLeg Action has no current traversal");
                exclusion = SystemServiceRouteTraversalExclusion{
                    value.leg, selected->incomingTraversal};
              } else if constexpr (std::is_same_v<
                                       T, SystemSingleSinkRoutingAction>) {
                const auto sinks = candidate.serviceRouteSinks().slice(
                    route->sinkOffset, route->sinkCount);
                if (value.sinkObligation >= sinks.size() ||
                    sinks[value.sinkObligation].node >= nodes.size())
                  return invalid("SingleSink Action names a foreign sink");
                const PnrIndex traversal =
                    nodes[sinks[value.sinkObligation].node].incomingTraversal;
                if (traversal == getInvalidPnrIndex())
                  return invalid("SingleSink Action names the route root");
                repairRegion = SystemServiceRouteRepairRegion{
                    SystemServiceRouteRepairRegionKind::SingleSink, value.leg,
                    value.sinkObligation};
              } else {
                const auto node =
                    llvm::find_if(nodes, [&](const auto &candidateNode) {
                      return candidateNode.endpoint == value.rootEndpoint;
                    });
                if (node == nodes.end() ||
                    node->incomingTraversal == getInvalidPnrIndex())
                  return invalid(
                      "RootedSubtree Action names a foreign route node");
                repairRegion = SystemServiceRouteRepairRegion{
                    SystemServiceRouteRepairRegionKind::RootedSubtree,
                    value.leg, value.rootEndpoint};
              }
              reroutedLegs.push_back(value.leg);
              return llvm::Error::success();
            }
          },
          action))
    return std::move(error);
  auto routes = negotiateSystemServiceRoutes(
      candidate.problem(), candidate.threadChoices(), candidate.graphChoices(),
      candidate.instructionResourceUses(), candidate.serviceResourceUses(),
      endpointExpansions, negotiationIterations, reroutedLegs,
      SystemServiceRoutesView{candidate.serviceRoutes(),
                              candidate.serviceRouteNodes(),
                              candidate.serviceRouteSinks()},
      exclusion, repairRegion,
      requireCapacityClosure
          ? SystemRoutingClosureRequirement::Strict
          : SystemRoutingClosureRequirement::PolicyAdmittedTemporary,
      reopenWitness);
  if (!routes)
    return routes.takeError();
  return SystemCandidateState::createMutation(
      candidate,
      {candidate.threadChoices(), candidate.graphChoices(), routes->routes,
       routes->nodes, routes->sinks, candidate.serviceTargets(),
       candidate.instructionResourceUses(), candidate.serviceResourceUses()},
      SystemCandidateMutationDomain::TransportRoutes);
}
