#include "PnR/System/SystemMappingMaterializer.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "SystemCandidateServiceResolver.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_materialization_invalid: " +
                                     message);
}

mlir::DenseI8ArrayAttr bytesAttr(mlir::MLIRContext *context,
                                 llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

template <typename Attr, typename Ref>
llvm::Expected<Attr> dataflowRefAttr(mlir::MLIRContext *context,
                                     const ArtifactIdentity &owner,
                                     const Ref &reference) {
  auto bytes = ::dataflow::encodeDataflowReference(owner, reference);
  if (!bytes)
    return bytes.takeError();
  return Attr::get(context, bytesAttr(context, *bytes));
}

::mapping::ArtifactIdentityAttr identityAttr(mlir::MLIRContext *context,
                                             const ArtifactIdentity &identity) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, bytesAttr(context, identity.bytes()));
}

::mapping::ArtifactRootReferenceAttr
rootReferenceAttr(mlir::MLIRContext *context,
                  const ArtifactRootReference &reference) {
  return ::mapping::ArtifactRootReferenceAttr::get(
      context, bytesAttr(context, encodeArtifactRootReference(reference)));
}

::mapping::FabricAccCoreOccurrenceRefAttr
accCoreAttr(mlir::MLIRContext *context,
            ::loom::fabric::AccCoreOccurrenceRef core) {
  return ::mapping::FabricAccCoreOccurrenceRefAttr::get(
      context, bytesAttr(context, ::loom::fabric::canonicalFabricBytes(core)));
}

::mapping::FabricTransportEndpointRefAttr transportEndpointAttr(
    mlir::MLIRContext *context,
    const ::loom::fabric::FabricTransportEndpointRef &endpoint) {
  return ::mapping::FabricTransportEndpointRefAttr::get(
      context,
      bytesAttr(context, ::loom::fabric::canonicalFabricBytes(endpoint)));
}

::mapping::FabricPhysicalTraversalRefAttr physicalTraversalAttr(
    mlir::MLIRContext *context,
    const ::loom::fabric::FabricPhysicalTraversalRef &traversal) {
  return ::mapping::FabricPhysicalTraversalRefAttr::get(
      context,
      bytesAttr(context, ::loom::fabric::canonicalFabricBytes(traversal)));
}

template <typename Attr, typename Ref>
Attr fabricRefAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(
      context,
      bytesAttr(context, ::loom::fabric::canonicalFabricBytes(reference)));
}

mlir::Attribute
intervalAttr(mlir::MLIRContext *context,
             const ::loom::mapping::SpatialMemoryIntervalView &interval) {
  if (std::holds_alternative<::loom::mapping::SpatialMemoryWholeIntervalView>(
          interval))
    return ::mapping::MemoryWholeIntervalAttr::get(context);
  const auto &range =
      std::get<::loom::mapping::SpatialMemoryByteRangeView>(interval);
  return ::mapping::MemoryByteRangeAttr::get(context, range.offsetBytes,
                                             range.sizeBytes);
}

std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>
intervalKey(const ::loom::mapping::SpatialMemoryIntervalView &interval) {
  if (std::holds_alternative<::loom::mapping::SpatialMemoryWholeIntervalView>(
          interval))
    return {0, 0, 0};
  const auto &range =
      std::get<::loom::mapping::SpatialMemoryByteRangeView>(interval);
  return {1, range.offsetBytes, range.sizeBytes};
}

std::vector<std::vector<std::uint8_t>>
targetRegionKey(const SystemMemoryServiceTargetPlan &plan) {
  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(plan.branches.size());
  for (const auto &branch : plan.branches)
    result.push_back(::loom::fabric::canonicalFabricBytes(branch.region));
  llvm::sort(result);
  return result;
}

llvm::Expected<bool>
requiresExplicitTransformPaths(const SystemCandidateState &candidate,
                               PnrIndex context,
                               const SystemMemoryServiceTargetPlan &selected) {
  auto domain = candidate.serviceTargetDomain(context);
  if (!domain)
    return domain.takeError();
  const auto *plans =
      std::get_if<std::vector<SystemMemoryServiceTargetPlan>>(&*domain);
  if (!plans)
    return invalid("memory target has a non-memory target domain");
  const auto selectedRegions = targetRegionKey(selected);
  std::size_t matchingRegionSets = 0;
  for (const SystemMemoryServiceTargetPlan &plan : *plans)
    if (targetRegionKey(plan) == selectedRegions)
      ++matchingRegionSets;
  return matchingRegionSets != 1;
}

llvm::Expected<::mapping::EventFamilyKeyAttr>
eventFamilyAttr(mlir::MLIRContext *context,
                const ArtifactIdentity &dataflowIdentity,
                const ::dataflow::EventFamilyKey &event) {
  return dataflowRefAttr<::mapping::EventFamilyKeyAttr>(
      context, dataflowIdentity, event);
}

llvm::Error emitSystemResourceUse(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &body,
    mlir::Attribute owner, const ::loom::fabric::FabricUsePatternRef &pattern,
    const ArtifactIdentity &dataflowIdentity,
    const ::dataflow::EventFamilyKey &triggerEvent,
    const std::optional<::dataflow::EventFamilyKey> &releaseEvent) {
  auto triggerAttr =
      eventFamilyAttr(builder.getContext(), dataflowIdentity, triggerEvent);
  if (!triggerAttr)
    return triggerAttr.takeError();
  auto trigger = ::mapping::SystemEventPointAttr::get(
      builder.getContext(), *triggerAttr, ::mapping::OwnerTypedValueAttr());
  ::mapping::SystemEventPointAttr release;
  if (releaseEvent) {
    auto releaseAttr =
        eventFamilyAttr(builder.getContext(), dataflowIdentity, *releaseEvent);
    if (!releaseAttr)
      return releaseAttr.takeError();
    release = ::mapping::SystemEventPointAttr::get(
        builder.getContext(), *releaseAttr, ::mapping::OwnerTypedValueAttr());
  }
  auto activation = ::mapping::SystemRelativeActivationAttr::get(
      builder.getContext(), trigger, release);
  builder.setInsertionPointToEnd(&body);
  ::mapping::ResourceUseOp::create(
      builder, location, owner,
      fabricRefAttr<::mapping::FabricUsePatternRefAttr>(builder.getContext(),
                                                        pattern),
      activation, builder.getArrayAttr({}), builder.getArrayAttr({}));
  return llvm::Error::success();
}

::mapping::SystemPresburgerCellAttr
cellAttr(mlir::MLIRContext *context,
         const ::loom::mapping::SystemPresburgerCell &cell) {
  llvm::SmallVector<mlir::Attribute> equalities;
  llvm::SmallVector<mlir::Attribute> inequalities;
  for (const auto &row : cell.equalities)
    equalities.push_back(mlir::DenseI64ArrayAttr::get(context, row));
  for (const auto &row : cell.inequalities)
    inequalities.push_back(mlir::DenseI64ArrayAttr::get(context, row));
  return ::mapping::SystemPresburgerCellAttr::get(
      context, cell.dimensionCount, cell.symbolCount, cell.localCount,
      mlir::ArrayAttr::get(context, equalities),
      mlir::ArrayAttr::get(context, inequalities));
}

template <typename BindingOp, typename KeyAttr, typename ClauseOp,
          typename TargetAttr>
llvm::Expected<BindingOp> createBinding(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &parent,
    KeyAttr key,
    llvm::ArrayRef<std::pair<::loom::mapping::SystemPresburgerCell, TargetAttr>>
        clauses) {
  mlir::OperationState bindingState(location, BindingOp::getOperationName());
  bindingState.addAttribute("key", key);
  bindingState.addAttribute(
      "relation_kind",
      ::mapping::SystemBindingRelationKindAttr::get(
          builder.getContext(),
          ::mapping::SystemBindingRelationKind::PresburgerPartition));
  bindingState.addRegion();
  builder.setInsertionPointToEnd(&parent);
  auto binding = mlir::cast<BindingOp>(builder.create(bindingState));
  binding.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&binding.getBody().front());
  for (const auto &[cell, target] : clauses) {
    mlir::OperationState clauseState(location, ClauseOp::getOperationName());
    clauseState.addAttribute(
        "cells", mlir::ArrayAttr::get(builder.getContext(),
                                      {cellAttr(builder.getContext(), cell)}));
    clauseState.addAttribute("target", target);
    builder.create(clauseState);
  }
  return binding;
}

} // namespace

llvm::Expected<mlir::OwningOpRef<mlir::Operation *>>
materializeSystemCandidateDraft(const SystemCandidateState &candidate,
                                mlir::MLIRContext &context) {
  if (llvm::Error error = candidate.verify())
    return std::move(error);
  context.getOrLoadDialect<::mapping::MappingDialect>();
  const FrozenSystemPnrProblem &problem = candidate.problem();

  std::vector<ArtifactRootReference> imports;
  imports.reserve(problem.graphDecisions().size());
  for (PnrIndex decision = 0; decision != problem.graphDecisions().size();
       ++decision)
    imports.push_back(candidate.selectedSpatialMapping(decision));
  llvm::sort(imports, artifactRootReferenceLess);
  imports.erase(std::unique(imports.begin(), imports.end()), imports.end());
  std::map<ArtifactRootReference, std::uint64_t,
           decltype(&artifactRootReferenceLess)>
      importOrdinals(&artifactRootReferenceLess);
  for (const auto &[ordinal, reference] : llvm::enumerate(imports))
    importOrdinals.emplace(reference, ordinal);

  mlir::OpBuilder builder(&context);
  mlir::Location location = builder.getUnknownLoc();
  mlir::OperationState rootState(location,
                                 ::mapping::SystemOp::getOperationName());
  rootState.addAttribute("dataflow",
                         identityAttr(&context, problem.dataflowIdentity()));
  rootState.addAttribute("fabric",
                         identityAttr(&context, problem.fabricIdentity()));
  llvm::SmallVector<mlir::Attribute> importAttrs;
  for (const ArtifactRootReference &reference : imports)
    importAttrs.push_back(rootReferenceAttr(&context, reference));
  rootState.addAttribute("spatial_mapping_imports",
                         builder.getArrayAttr(importAttrs));
  llvm::SmallVector<mlir::Attribute> rootAttrs;
  for (const auto &root : problem.rootThreadLaunches()) {
    auto attribute = dataflowRefAttr<::mapping::RootThreadLaunchRefAttr>(
        &context, problem.dataflowIdentity(), root);
    if (!attribute)
      return attribute.takeError();
    rootAttrs.push_back(*attribute);
  }
  rootState.addAttribute("root_thread_launches",
                         builder.getArrayAttr(rootAttrs));
  rootState.addRegion();
  mlir::OwningOpRef<mlir::Operation *> result(builder.create(rootState));
  auto root = mlir::cast<::mapping::SystemOp>(result.get());
  root.getBody().emplaceBlock();

  std::map<std::vector<std::uint8_t>,
           std::vector<std::pair<::loom::mapping::SystemPresburgerCell,
                                 ::mapping::FabricAccCoreOccurrenceRefAttr>>>
      threadClauses;
  std::map<std::vector<std::uint8_t>, ::dataflow::RootThreadLaunchRef>
      threadKeys;
  for (const auto &[decision, frozen] :
       llvm::enumerate(problem.threadDecisions())) {
    auto key = ::dataflow::encodeDataflowReference(problem.dataflowIdentity(),
                                                   frozen.root);
    if (!key)
      return key.takeError();
    threadKeys.emplace(*key, frozen.root);
    threadClauses[*key].push_back(
        {frozen.cell,
         accCoreAttr(&context, candidate.selectedAccCore(
                                   static_cast<PnrIndex>(decision)))});
  }
  for (const auto &[keyBytes, clauses] : threadClauses) {
    auto reference = threadKeys.find(keyBytes);
    if (reference == threadKeys.end())
      return invalid("thread binding lost its Dataflow-owned key");
    auto key = dataflowRefAttr<::mapping::RootThreadLaunchRefAttr>(
        &context, problem.dataflowIdentity(), reference->second);
    if (!key)
      return key.takeError();
    auto binding = createBinding<::mapping::ThreadExecutionBindingOp,
                                 ::mapping::RootThreadLaunchRefAttr,
                                 ::mapping::ThreadPresburgerClauseOp,
                                 ::mapping::FabricAccCoreOccurrenceRefAttr>(
        builder, location, root.getBody().front(), *key, clauses);
    if (!binding)
      return binding.takeError();
  }

  std::map<std::vector<std::uint8_t>,
           std::vector<std::pair<::loom::mapping::SystemPresburgerCell,
                                 ::mapping::SpatialMappingImportRefAttr>>>
      graphClauses;
  std::map<std::vector<std::uint8_t>, ::dataflow::RootedGraphLaunchRef>
      graphKeys;
  for (const auto &[decision, frozen] :
       llvm::enumerate(problem.graphDecisions())) {
    auto key = ::dataflow::encodeDataflowReference(problem.dataflowIdentity(),
                                                   frozen.launch);
    if (!key)
      return key.takeError();
    graphKeys.emplace(*key, frozen.launch);
    const ArtifactRootReference &selected =
        candidate.selectedSpatialMapping(static_cast<PnrIndex>(decision));
    auto imported = importOrdinals.find(selected);
    if (imported == importOrdinals.end())
      return invalid("selected SpatialMapping is absent from the import set");
    graphClauses[*key].push_back(
        {frozen.cell, ::mapping::SpatialMappingImportRefAttr::get(
                          &context, imported->second)});
  }
  for (const auto &[keyBytes, clauses] : graphClauses) {
    auto reference = graphKeys.find(keyBytes);
    if (reference == graphKeys.end())
      return invalid("graph binding lost its Dataflow-owned key");
    auto key = dataflowRefAttr<::mapping::RootedGraphLaunchRefAttr>(
        &context, problem.dataflowIdentity(), reference->second);
    if (!key)
      return key.takeError();
    auto binding = createBinding<::mapping::GraphExecutionBindingOp,
                                 ::mapping::RootedGraphLaunchRefAttr,
                                 ::mapping::GraphPresburgerClauseOp,
                                 ::mapping::SpatialMappingImportRefAttr>(
        builder, location, root.getBody().front(), *key, clauses);
    if (!binding)
      return binding.takeError();
  }

  struct PlanGroup final {
    std::vector<PnrIndex> routes;
  };
  struct ServiceGroup final {
    ::loom::mapping::SystemServiceObligationKey key;
    std::map<PnrIndex, PlanGroup> plans;
  };
  std::map<PnrIndex, std::pair<::loom::mapping::SystemServiceObligationKey,
                               std::uint64_t>>
      persistentPlanByContext;
  std::map<std::vector<std::uint8_t>, ServiceGroup> serviceGroups;
  for (const auto &[contextOrdinal, context] :
       llvm::enumerate(problem.serviceContexts())) {
    if (context.service >= problem.serviceDomains().size())
      return invalid("service context has no H service domain");
    const auto &obligation = problem.serviceDomains()[context.service].key;
    auto keyBytes = ::loom::mapping::encodeSystemServiceObligationKey(
        problem.dataflowIdentity(), obligation);
    if (!keyBytes)
      return keyBytes.takeError();
    auto service =
        serviceGroups.try_emplace(*keyBytes, ServiceGroup{obligation, {}})
            .first;
    service->second.plans.try_emplace(static_cast<PnrIndex>(contextOrdinal),
                                      PlanGroup{});
  }
  for (const auto &[routeOrdinal, route] :
       llvm::enumerate(candidate.serviceRoutes())) {
    if (route.leg >= problem.serviceLegs().size())
      return invalid("service route leg is out of range");
    const auto &leg = problem.serviceLegs()[route.leg].key;
    const PnrIndex serviceContext =
        problem.serviceLegs()[route.leg].serviceContext;
    if (serviceContext >= problem.serviceContexts().size())
      return invalid("service route leg has no execution context");
    auto keyBytes = ::loom::mapping::encodeSystemServiceObligationKey(
        problem.dataflowIdentity(), leg.obligation);
    if (!keyBytes)
      return keyBytes.takeError();
    auto found =
        serviceGroups.try_emplace(*keyBytes, ServiceGroup{leg.obligation, {}})
            .first;
    auto plan =
        found->second.plans.try_emplace(serviceContext, PlanGroup{}).first;
    plan->second.routes.push_back(static_cast<PnrIndex>(routeOrdinal));
  }
  for (const auto &[keyBytes, group] : serviceGroups) {
    (void)keyBytes;
    auto obligationBytes = ::loom::mapping::encodeSystemServiceObligationKey(
        problem.dataflowIdentity(), group.key);
    if (!obligationBytes)
      return obligationBytes.takeError();
    builder.setInsertionPointToEnd(&root.getBody().front());
    auto service = ::mapping::ServiceRealizationOp::create(
        builder, location,
        ::mapping::SystemServiceObligationKeyAttr::get(
            &context, bytesAttr(&context, *obligationBytes)));
    service.getBody().emplaceBlock();
    std::map<PnrIndex, std::uint64_t> planOrdinals;
    for (const auto &[contextOrdinal, groupedPlan] : group.plans) {
      const std::uint64_t authoredOrdinal = planOrdinals.size();
      planOrdinals.emplace(contextOrdinal, authoredOrdinal);
      persistentPlanByContext.emplace(
          contextOrdinal, std::make_pair(group.key, authoredOrdinal));
      builder.setInsertionPointToEnd(&service.getBody().front());
      auto plan =
          ::mapping::ServicePlanOp::create(builder, location, authoredOrdinal);
      plan.getBody().emplaceBlock();
      const auto &selectedTarget = candidate.serviceTarget(contextOrdinal);
      if (const auto *targetPlan =
              std::get_if<SystemMemoryServiceTargetPlan>(&selectedTarget)) {
        if (targetPlan->branches.empty())
          return invalid("selected memory target plan has no terminal branch");
        const auto *operation =
            std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
                &group.key);
        const auto *logicalMemory =
            operation
                ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
                : nullptr;
        if (!logicalMemory)
          return invalid(
              "memory region target belongs to a non-memory service");
        struct MemoryTargetDraft final {
          ::loom::mapping::SpatialMemoryIntervalView interval;
          std::vector<std::pair<::dataflow::MemoryExposureRef,
                                ::loom::fabric::SubordinateEndpointRef>>
              exposures;
        };
        std::map<std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>,
                 MemoryTargetDraft>
            targets;
        const auto &serviceContext = problem.serviceContexts()[contextOrdinal];
        for (const SystemServiceTargetSubject &subject :
             serviceContext.subjects) {
          auto binding = detail::resolveSystemMemoryServiceBinding(
              problem, contextOrdinal, subject, candidate.threadChoices(),
              candidate.graphChoices());
          if (!binding)
            return binding.takeError();
          if (!(*binding)->interval)
            return invalid("memory target subject has no logical interval");
          const auto key = intervalKey(*(*binding)->interval);
          auto inserted = targets.try_emplace(
              key, MemoryTargetDraft{*(*binding)->interval, {}});
          if (const auto *exposure =
                  std::get_if<SystemMemoryExposureTargetSubject>(&subject)) {
            if (!(*binding)->exposureTerminal)
              return invalid("memory exposure target has no provider terminal");
            inserted.first->second.exposures.push_back(
                {exposure->exposure, *(*binding)->exposureTerminal});
          }
        }
        auto logicalAttr =
            dataflowRefAttr<::mapping::LogicalMemoryRootOrViewRefAttr>(
                &context, problem.dataflowIdentity(), *logicalMemory);
        if (!logicalAttr)
          return logicalAttr.takeError();
        auto explicitPaths = requiresExplicitTransformPaths(
            candidate, contextOrdinal, *targetPlan);
        if (!explicitPaths)
          return explicitPaths.takeError();
        for (const auto &branch : targetPlan->branches) {
          llvm::SmallVector<mlir::Attribute> transformPath;
          if (*explicitPaths)
            for (const auto transform : branch.transformPath)
              transformPath.push_back(
                  fabricRefAttr<::mapping::SystemServiceTransformRefAttr>(
                      &context, transform));
          for (auto &[key, target] : targets) {
            (void)key;
            builder.setInsertionPointToEnd(&plan.getBody().front());
            auto targetOp = ::mapping::MemoryRegionTargetOp::create(
                builder, location, *logicalAttr,
                intervalAttr(&context, target.interval),
                fabricRefAttr<::mapping::FabricMemoryServiceRegionRefAttr>(
                    &context, branch.region),
                builder.getArrayAttr(transformPath));
            targetOp.getBody().emplaceBlock();
            builder.setInsertionPointToEnd(&targetOp.getBody().front());
            for (const auto &[exposure, terminal] : target.exposures) {
              auto exposureAttr =
                  dataflowRefAttr<::mapping::MemoryExposureRefAttr>(
                      &context, problem.dataflowIdentity(), exposure);
              if (!exposureAttr)
                return exposureAttr.takeError();
              ::mapping::SystemMemoryExposureOp::create(
                  builder, location, *exposureAttr,
                  fabricRefAttr<::mapping::SubordinateEndpointRefAttr>(
                      &context, terminal));
            }
          }
        }
      } else if (const auto *domain =
                     std::get_if<::loom::fabric::MemoryConsistencyDomainRef>(
                         &selectedTarget)) {
        const auto *operation =
            std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
                &group.key);
        const auto *fence =
            operation ? std::get_if<::dataflow::FenceActorFamilyRef>(operation)
                      : nullptr;
        if (!fence)
          return invalid("consistency target belongs to a non-fence service");
        auto fenceAttr = dataflowRefAttr<::mapping::FenceActorFamilyRefAttr>(
            &context, problem.dataflowIdentity(), *fence);
        if (!fenceAttr)
          return fenceAttr.takeError();
        builder.setInsertionPointToEnd(&plan.getBody().front());
        ::mapping::ConsistencyTargetOp::create(
            builder, location, *fenceAttr,
            fabricRefAttr<::mapping::MemoryConsistencyDomainRefAttr>(&context,
                                                                     *domain));
      } else if (!std::holds_alternative<std::monostate>(selectedTarget)) {
        return invalid("service context has an unknown selected target kind");
      }
      for (PnrIndex routeOrdinal : groupedPlan.routes) {
        const SystemServiceRouteSelection &selected =
            candidate.serviceRoutes()[routeOrdinal];
        const FrozenSystemServiceLeg &leg = problem.serviceLegs()[selected.leg];
        auto legBytes = ::loom::mapping::encodeCanonicalServiceLegKey(
            problem.dataflowIdentity(), leg.key);
        if (!legBytes)
          return legBytes.takeError();
        if (selected.rootEndpoint >=
            problem.routingTopology().endpoints().size())
          return invalid("service route root endpoint is out of range");
        builder.setInsertionPointToEnd(&plan.getBody().front());
        auto route = ::mapping::TransferLegRealizationOp::create(
            builder, location,
            ::mapping::CanonicalServiceLegKeyAttr::get(
                &context, bytesAttr(&context, *legBytes)),
            transportEndpointAttr(&context,
                                  problem.routingTopology()
                                      .endpoints()[selected.rootEndpoint]
                                      .reference));
        route.getBody().emplaceBlock();
        builder.setInsertionPointToEnd(&route.getBody().front());
        const auto nodes = candidate.serviceRouteNodes().slice(
            selected.nodeOffset, selected.nodeCount);
        for (PnrIndex nodeOrdinal = 1; nodeOrdinal < nodes.size();
             ++nodeOrdinal) {
          const auto &node = nodes[nodeOrdinal];
          if (node.incomingTraversal >=
              problem.routingTopology().traversals().size())
            return invalid("service route traversal is out of range");
          ::mapping::SystemRouteNodeOp::create(
              builder, location, nodeOrdinal, node.parentNode,
              physicalTraversalAttr(&context,
                                    problem.routingTopology()
                                        .traversals()[node.incomingTraversal]
                                        .reference));
        }
        const auto sinks = candidate.serviceRouteSinks().slice(
            selected.sinkOffset, selected.sinkCount);
        for (const SystemServiceRouteSinkSelection &sink : sinks) {
          if (sink.terminal >= problem.serviceTerminals().size())
            return invalid("service route sink terminal is out of range");
          auto terminalBytes = ::loom::mapping::encodeSystemTransferTerminalKey(
              problem.dataflowIdentity(),
              problem.serviceTerminals()[sink.terminal].key);
          if (!terminalBytes)
            return terminalBytes.takeError();
          ::mapping::SystemRouteSinkOp::create(
              builder, location,
              ::mapping::SystemTransferTerminalKeyAttr::get(
                  &context, bytesAttr(&context, *terminalBytes)),
              sink.node);
        }
      }
    }

    struct SelectionDraft final {
      std::vector<
          std::pair<::loom::mapping::SystemPresburgerCell, std::uint64_t>>
          clauses;
    };
    std::map<std::vector<std::uint8_t>, SelectionDraft> selections;
    for (const auto &[contextOrdinal, authoredOrdinal] : planOrdinals) {
      const auto &serviceContext = problem.serviceContexts()[contextOrdinal];
      if (serviceContext.threadDecision >= problem.threadDecisions().size())
        return invalid("service selection has an invalid thread dependency");
      const auto core =
          candidate.selectedAccCore(serviceContext.threadDecision);
      ::loom::mapping::ExecutionContextKey executionContext;
      if (serviceContext.graphDecision != getInvalidPnrIndex()) {
        if (serviceContext.graphDecision >= problem.graphDecisions().size())
          return invalid("service selection has an invalid graph dependency");
        executionContext = ::loom::mapping::SpatialExecutionContextKey{
            core, candidate.selectedSpatialMapping(serviceContext.graphDecision)
                      .artifact};
      } else {
        executionContext =
            ::loom::mapping::InstructionExecutionContextKey{core};
      }
      if (serviceContext.cells.empty())
        return invalid("service context has no selection relation cell");
      for (const SystemServiceTargetSubject &subject :
           serviceContext.subjects) {
        ::loom::mapping::ServicePlanSelectionAnchor anchor;
        if (const auto *member =
                std::get_if<SystemServiceMemberTargetSubject>(&subject))
          anchor =
              ::loom::mapping::ServiceMemberPlanSelectionAnchor{member->member};
        else
          anchor = ::loom::mapping::MemoryExposurePlanSelectionAnchor{
              std::get<SystemMemoryExposureTargetSubject>(subject).exposure};
        ::loom::mapping::ServicePlanSelectionKey selectionKey{std::move(anchor),
                                                              executionContext};
        auto selectionBytes = ::loom::mapping::encodeServicePlanSelectionKey(
            problem.dataflowIdentity(), selectionKey);
        if (!selectionBytes)
          return selectionBytes.takeError();
        auto selection =
            selections.try_emplace(*selectionBytes, SelectionDraft{}).first;
        for (const auto &cell : serviceContext.cells)
          selection->second.clauses.push_back({cell, authoredOrdinal});
      }
    }
    for (const auto &[selectionBytes, selection] : selections) {
      builder.setInsertionPointToEnd(&service.getBody().front());
      mlir::OperationState selectionState(
          location, ::mapping::ServicePlanSelectionOp::getOperationName());
      selectionState.addAttribute(
          "key", ::mapping::ServicePlanSelectionKeyAttr::get(
                     &context, bytesAttr(&context, selectionBytes)));
      selectionState.addAttribute(
          "relation_kind",
          ::mapping::SystemBindingRelationKindAttr::get(
              &context,
              ::mapping::SystemBindingRelationKind::PresburgerPartition));
      selectionState.addRegion();
      auto selectionOp = mlir::cast<::mapping::ServicePlanSelectionOp>(
          builder.create(selectionState));
      selectionOp.getBody().emplaceBlock();
      builder.setInsertionPointToEnd(&selectionOp.getBody().front());
      for (const auto &[cell, target] : selection.clauses)
        ::mapping::ServicePlanPresburgerClauseOp::create(
            builder, location,
            mlir::ArrayAttr::get(&context, {cellAttr(&context, cell)}), target);
    }
  }

  for (const auto &selected : candidate.instructionResourceUses()) {
    auto rootAttr = dataflowRefAttr<::mapping::RootThreadLaunchRefAttr>(
        &context, problem.dataflowIdentity(), selected.root);
    if (!rootAttr)
      return rootAttr.takeError();
    auto owner = ::mapping::InstructionExecutionResourceOwnerRefAttr::get(
        &context, *rootAttr,
        fabricRefAttr<::mapping::InstructionCoreContextRefAttr>(
            &context, selected.context));
    const ::dataflow::RootThreadBoundaryTransferRef startTransfer(
        ::dataflow::RootThreadStartTransferRef{selected.root});
    const ::dataflow::RootThreadBoundaryTransferRef completionTransfer(
        ::dataflow::RootThreadCompletionTransferRef{selected.root});
    const ::dataflow::EventFamilyKey trigger(
        ::dataflow::StaticTransferEventRef(::dataflow::ConsumedTransferEventRef{
            ::dataflow::CanonicalSinkTerminalRef(
                ::dataflow::RootThreadBoundarySinkRef{startTransfer})}));
    const ::dataflow::EventFamilyKey release(
        ::dataflow::StaticTransferEventRef(::dataflow::ProducedTransferEventRef{
            ::dataflow::CanonicalProducerTerminalRef(
                ::dataflow::RootThreadBoundarySourceRef{completionTransfer})}));
    if (llvm::Error error = emitSystemResourceUse(
            builder, location, root.getBody().front(), owner, selected.pattern,
            problem.dataflowIdentity(), trigger, release))
      return std::move(error);
  }

  for (const auto &selected : candidate.serviceResourceUses()) {
    if (selected.context >= problem.serviceContexts().size())
      return invalid("service ResourceUse context is out of range");
    const auto &serviceContext = problem.serviceContexts()[selected.context];
    if (selected.subject >= serviceContext.subjects.size())
      return invalid("service ResourceUse subject is out of range");
    const auto *memberSubject = std::get_if<SystemServiceMemberTargetSubject>(
        &serviceContext.subjects[selected.subject]);
    if (!memberSubject)
      return invalid("memory exposure unexpectedly selected a ResourceUse");
    const auto persisted = persistentPlanByContext.find(selected.context);
    if (persisted == persistentPlanByContext.end())
      return invalid("service ResourceUse has no persistent ServicePlan");
    auto serviceBytes = ::loom::mapping::encodeSystemServiceObligationKey(
        problem.dataflowIdentity(), persisted->second.first);
    if (!serviceBytes)
      return serviceBytes.takeError();
    auto serviceAttr = ::mapping::SystemServiceObligationKeyAttr::get(
        &context, bytesAttr(&context, *serviceBytes));

    mlir::Attribute element;
    const auto &target = candidate.serviceTarget(selected.context);
    if (const auto *plan =
            std::get_if<SystemMemoryServiceTargetPlan>(&target)) {
      if (selected.branch >= plan->branches.size())
        return invalid("service ResourceUse branch is out of range");
      auto binding = detail::resolveSystemMemoryServiceBinding(
          problem, selected.context, serviceContext.subjects[selected.subject],
          candidate.threadChoices(), candidate.graphChoices());
      if (!binding)
        return binding.takeError();
      if (!(*binding)->interval)
        return invalid("service ResourceUse has no logical interval");
      const auto *operation =
          std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
              &persisted->second.first);
      const auto *logicalMemory =
          operation
              ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
              : nullptr;
      if (!logicalMemory)
        return invalid("memory ResourceUse belongs to a non-memory service");
      auto logicalAttr =
          dataflowRefAttr<::mapping::LogicalMemoryRootOrViewRefAttr>(
              &context, problem.dataflowIdentity(), *logicalMemory);
      if (!logicalAttr)
        return logicalAttr.takeError();
      llvm::SmallVector<mlir::Attribute> transforms;
      auto explicitPaths =
          requiresExplicitTransformPaths(candidate, selected.context, *plan);
      if (!explicitPaths)
        return explicitPaths.takeError();
      if (*explicitPaths)
        for (const auto transform :
             plan->branches[selected.branch].transformPath)
          transforms.push_back(
              fabricRefAttr<::mapping::SystemServiceTransformRefAttr>(
                  &context, transform));
      element = ::mapping::MemoryRegionElementKeyAttr::get(
          &context, *logicalAttr, intervalAttr(&context, *(*binding)->interval),
          fabricRefAttr<::mapping::FabricMemoryServiceRegionRefAttr>(
              &context, plan->branches[selected.branch].region),
          builder.getArrayAttr(transforms));
    } else {
      const auto *domain =
          std::get_if<::loom::fabric::MemoryConsistencyDomainRef>(&target);
      const auto *operation =
          std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
              &persisted->second.first);
      const auto *fence =
          operation ? std::get_if<::dataflow::FenceActorFamilyRef>(operation)
                    : nullptr;
      if (!domain || !fence)
        return invalid("consistency ResourceUse has no fence target");
      auto fenceAttr = dataflowRefAttr<::mapping::FenceActorFamilyRefAttr>(
          &context, problem.dataflowIdentity(), *fence);
      if (!fenceAttr)
        return fenceAttr.takeError();
      element = ::mapping::ConsistencyElementKeyAttr::get(
          &context, *fenceAttr,
          fabricRefAttr<::mapping::MemoryConsistencyDomainRefAttr>(&context,
                                                                   *domain));
    }
    auto owner = ::mapping::ServicePlanElementRefAttr::get(
        &context, serviceAttr, persisted->second.second, element);
    const ::dataflow::ContextualActorRef *actor = nullptr;
    if (const auto *addressed =
            std::get_if<::dataflow::AddressedMemoryActorMemberRef>(
                &memberSubject->member))
      actor = &addressed->actor;
    else if (const auto *fence = std::get_if<::dataflow::FenceActorMemberRef>(
                 &memberSubject->member))
      actor = &fence->actor;
    if (!actor)
      return invalid("service ResourceUse member has no contextual actor");
    const ::dataflow::EventFamilyKey trigger(
        ::dataflow::ContextualActorTransitionEventRef{*actor, 0});
    if (llvm::Error error = emitSystemResourceUse(
            builder, location, root.getBody().front(), owner, selected.pattern,
            problem.dataflowIdentity(), trigger, std::nullopt))
      return std::move(error);
  }
  return result;
}

} // namespace loom::pnr
