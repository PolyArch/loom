#include "TechMappingCandidateTestSupport.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialActionExecutor.h"
#include "PnR/SpatialCandidateInitializer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial route constraint test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes) {
  std::string text = "[";
  for (auto [ordinal, byte] : llvm::enumerate(bytes)) {
    if (ordinal)
      text += ", ";
    text += std::to_string(static_cast<std::int8_t>(byte));
  }
  return text + "]";
}

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
}

template <typename Ref>
std::string dataflowAttr(llvm::StringRef spelling,
                         const loom::ArtifactIdentity &owner, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(take(dataflow::encodeDataflowReference(owner, ref))) + ">";
}

template <typename Ref>
std::string fabricAttr(llvm::StringRef spelling, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(loom::fabric::canonicalFabricBytes(ref)) + ">";
}

struct SelectedRouteSets final {
  std::vector<loom::pnr::PnrIndex> traversals;
  std::vector<loom::pnr::PnrIndex> resourceStates;
};

SelectedRouteSets
selectedRouteSets(const loom::pnr::FrozenSpatialPnrProblem &problem,
                  const loom::pnr::SpatialCandidateState &candidate,
                  loom::pnr::PnrIndex logicalNet) {
  SelectedRouteSets result;
  for (const loom::pnr::RouteTreeNode &node :
       candidate.routeTree(logicalNet).nodeStorage()) {
    if (!node.isActive() || node.parentArc == loom::pnr::getInvalidPnrIndex())
      continue;
    const auto &routing = problem.routing();
    const loom::pnr::PnrIndex traversal =
        routing.routingArcs()[node.parentArc].traversal;
    result.traversals.push_back(traversal);
    const auto &record = routing.traversals()[traversal];
    const auto states = routing.traversalResourceStates().slice(
        record.resourceStateOffset, record.resourceStateCount);
    result.resourceStates.insert(result.resourceStates.end(), states.begin(),
                                 states.end());
  }
  for (auto *values : {&result.traversals, &result.resourceStates}) {
    llvm::sort(*values);
    values->erase(std::unique(values->begin(), values->end()), values->end());
  }
  return result;
}

bool intersects(llvm::ArrayRef<loom::pnr::PnrIndex> lhs,
                llvm::ArrayRef<loom::pnr::PnrIndex> rhs) {
  for (loom::pnr::PnrIndex value : lhs)
    if (llvm::is_contained(rhs, value))
      return true;
  return false;
}

} // namespace

void loom::test::exerciseSpatialRouteConstraintRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store) {
  std::vector<dataflow::CanonicalGraphProducerEndpointRef> producers;
  for (const auto &net : techMapping.residualLogicalNets())
    producers.push_back(net.producer);
  if (producers.size() < 2)
    fail("fixture has fewer than two logical nets");

  const auto config = take(pnr::projectResolvedSpatialPnrConfigView(
      buildSpatialPnrTestResolvedConfig()));
  const auto makeConstraints = [&](llvm::StringRef clauses) {
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(techMapping.identity()) + ") fabric(" +
        identityAttr(fabric.identity()) + ") {\n" + clauses.str() + "  }\n}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
      fail("cannot parse fixture constraints");
    auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
    return take(mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, techMapping, fabric, store));
  };
  const auto producerAttr = [&](pnr::PnrIndex net) {
    return dataflowAttr("graph_producer_endpoint_ref", dataflow.identity(),
                        producers[net]);
  };
  const auto traversalDomain =
      [&](llvm::ArrayRef<pnr::PnrIndex> traversals,
          const pnr::FrozenSpatialPnrProblem &problem) {
        std::string text = "[";
        for (auto [ordinal, traversal] : llvm::enumerate(traversals)) {
          if (ordinal)
            text += ", ";
          text +=
              fabricAttr("fabric_physical_traversal_ref",
                         problem.routing().traversals()[traversal].reference);
        }
        return text + "]";
      };
  const auto resourceDomain = [&](llvm::ArrayRef<pnr::PnrIndex> states,
                                  const pnr::FrozenSpatialPnrProblem &problem) {
    std::string text = "[";
    for (auto [ordinal, state] : llvm::enumerate(states)) {
      if (ordinal)
        text += ", ";
      text += fabricAttr("fabric_resource_state_ref",
                         problem.resources().resourceStates()[state].reference);
    }
    return text + "]";
  };

  const auto unconstrained = buildSpatialMappingConstraints(
      context, dataflow, techMapping, fabric, store);
  auto unconstrainedProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, config, unconstrained.view()));
  auto unconstrainedCandidate =
      take(pnr::createCanonicalSpatialCandidate(unconstrainedProblem));
  pnr::SpatialActionExecutorScratch unconstrainedExecutor;
  requireSuccess(unconstrainedExecutor.prepare(*unconstrainedCandidate));
  std::vector<pnr::PnrIndex> routableNets;
  std::optional<pnr::PnrIndex> statefulNet;
  std::optional<pnr::PnrIndex> multicastNet;
  SelectedRouteSets statefulRoute;
  for (pnr::PnrIndex net = 0; net < producers.size(); ++net) {
    const pnr::SpatialMappingAction action = pnr::SpatialTransportRoutingAction{
        pnr::SpatialWholeNetRoutingAction{net}};
    auto probe = unconstrainedExecutor.probe(*unconstrainedCandidate, action);
    if (!probe) {
      llvm::consumeError(probe.takeError());
      continue;
    }
    const SelectedRouteSets selected =
        selectedRouteSets(*unconstrainedProblem, *unconstrainedCandidate, net);
    requireSuccess(probe->discard());
    if (selected.traversals.empty())
      continue;
    routableNets.push_back(net);
    if (!multicastNet &&
        unconstrainedProblem->transfers().logicalNets()[net].sinkCount > 1)
      multicastNet = net;
    if (!statefulNet && !selected.resourceStates.empty()) {
      statefulNet = net;
      statefulRoute = selected;
    }
  }
  if (routableNets.size() < 2 || !statefulNet || !multicastNet)
    fail("fixture lacks two routes, one stateful route, and one multicast");

  auto localCandidate =
      take(pnr::createCanonicalSpatialCandidate(unconstrainedProblem));
  pnr::SpatialActionExecutorScratch localExecutor;
  requireSuccess(localExecutor.prepare(*localCandidate));
  auto initialRoute = take(localExecutor.probe(
      *localCandidate, pnr::SpatialTransportRoutingAction{
                           pnr::SpatialWholeNetRoutingAction{*multicastNet}}));
  requireSuccess(initialRoute.commit());
  const std::vector<pnr::RouteTreeNode> routedNodes(
      localCandidate->routeTree(*multicastNet).nodeStorage().begin(),
      localCandidate->routeTree(*multicastNet).nodeStorage().end());

  pnr::SpatialActionDomainScratch localDomain;
  requireSuccess(localDomain.prepare(*unconstrainedProblem));
  requireSuccess(localDomain.rebuild(*localCandidate));
  std::optional<pnr::SpatialTransportRoutingAction> singleSink;
  std::optional<pnr::SpatialTransportRoutingAction> rootedSubtree;
  std::optional<pnr::SpatialTransportRoutingAction> witnessRegion;
  for (const auto &action : localDomain.view().transportChoices) {
    if (const auto *choice =
            std::get_if<pnr::SpatialSingleSinkRoutingAction>(&action);
        choice && choice->logicalNet == *multicastNet && !singleSink)
      singleSink = action;
    if (const auto *choice =
            std::get_if<pnr::SpatialRootedSubtreeRoutingAction>(&action);
        choice && choice->logicalNet == *multicastNet && !rootedSubtree)
      rootedSubtree = action;
    if (const auto *choice =
            std::get_if<pnr::SpatialWitnessRegionRoutingAction>(&action);
        choice &&
        choice->witnessKind == ResolvedPnrViolationKind::UnroutedObligation &&
        !witnessRegion)
      witnessRegion = action;
  }
  if (!singleSink || !rootedSubtree || !witnessRegion)
    fail("dynamic Action domain omitted a closed local routing scope");

  for (const pnr::SpatialTransportRoutingAction &action :
       {*singleSink, *rootedSubtree}) {
    auto probe = take(localExecutor.probe(*localCandidate, action));
    requireSuccess(probe.discard());
    requireSuccess(localCandidate->verify());
    if (!llvm::equal(localCandidate->routeTree(*multicastNet).nodeStorage(),
                     routedNodes))
      fail("discarded local routing Action changed its RouteTree");
  }
  const auto *witness =
      std::get_if<pnr::SpatialWitnessRegionRoutingAction>(&*witnessRegion);
  auto witnessProbe =
      take(localExecutor.probe(*localCandidate, *witnessRegion));
  bool routedWitnessNet = false;
  for (pnr::PnrIndex net : routableNets)
    routedWitnessNet |=
        net != *multicastNet && localCandidate->routeTree(net).isRouted();
  if (!witness || !routedWitnessNet)
    fail("WitnessRegion did not route its unresolved dependency closure");
  requireSuccess(witnessProbe.discard());
  for (pnr::PnrIndex net : routableNets)
    if (net != *multicastNet && localCandidate->routeTree(net).isRouted())
      fail("discarded WitnessRegion retained a routed dependency");
  requireSuccess(localCandidate->verify());

  const pnr::PnrIndex routedNet = *statefulNet;
  const pnr::PnrIndex peerNet = *llvm::find_if(
      routableNets, [&](pnr::PnrIndex net) { return net != routedNet; });
  const pnr::SpatialMappingAction routeFirst =
      pnr::SpatialTransportRoutingAction{
          pnr::SpatialWholeNetRoutingAction{routedNet}};

  const auto emptyRoute =
      makeConstraints("    mapping.constraint.domain_restriction "
                      "projection(net_selected_physical_traversals) subject(" +
                      producerAttr(routedNet) + ") admissible_domain([])\n");
  auto emptyProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, config, emptyRoute.view()));
  auto emptyCandidate =
      take(pnr::createCanonicalSpatialCandidate(emptyProblem));
  pnr::SpatialActionExecutorScratch emptyExecutor;
  requireSuccess(emptyExecutor.prepare(*emptyCandidate));
  auto forbidden = emptyExecutor.probe(*emptyCandidate, routeFirst);
  if (forbidden) {
    if (!selectedRouteSets(*emptyProblem, *emptyCandidate, routedNet)
             .traversals.empty())
      fail("empty traversal domain admitted selected traversals");
    requireSuccess(forbidden->discard());
  } else {
    llvm::consumeError(forbidden.takeError());
  }
  if (!selectedRouteSets(*emptyProblem, *emptyCandidate, routedNet)
           .traversals.empty())
    fail("rejected traversal-domain Action retained a route");
  requireSuccess(emptyCandidate->verify());

  const auto resourceLimited = makeConstraints(
      "    mapping.constraint.domain_restriction "
      "projection(net_selected_physical_traversals) subject(" +
      producerAttr(routedNet) + ") admissible_domain(" +
      traversalDomain(statefulRoute.traversals, *unconstrainedProblem) +
      ")\n    mapping.constraint.domain_restriction "
      "projection(net_traversal_resource_states) subject(" +
      producerAttr(routedNet) + ") admissible_domain(" +
      resourceDomain(statefulRoute.resourceStates, *unconstrainedProblem) +
      ")\n");
  auto resourceProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, config, resourceLimited.view()));
  auto resourceCandidate =
      take(pnr::createCanonicalSpatialCandidate(resourceProblem));
  pnr::SpatialActionExecutorScratch resourceExecutor;
  requireSuccess(resourceExecutor.prepare(*resourceCandidate));
  auto resourceProbe = resourceExecutor.probe(*resourceCandidate, routeFirst);
  if (!resourceProbe)
    fail(
        "exact traversal/resource-state domains rejected their source route: " +
        llvm::toString(resourceProbe.takeError()));
  const SelectedRouteSets constrainedRoute =
      selectedRouteSets(*resourceProblem, *resourceCandidate, routedNet);
  if (constrainedRoute.traversals.empty() ||
      constrainedRoute.resourceStates.empty() ||
      !llvm::all_of(constrainedRoute.traversals,
                    [&](pnr::PnrIndex value) {
                      return llvm::is_contained(statefulRoute.traversals,
                                                value);
                    }) ||
      !llvm::all_of(constrainedRoute.resourceStates, [&](pnr::PnrIndex value) {
        return llvm::is_contained(statefulRoute.resourceStates, value);
      }))
    fail("route escaped its exact traversal/resource-state domains");
  requireSuccess(resourceProbe->discard());
  requireSuccess(resourceCandidate->verify());

  const auto equalRoutes = makeConstraints(
      "    mapping.constraint.equal "
      "projection(net_selected_physical_traversals) subjects([" +
      producerAttr(routedNet) + ", " + producerAttr(peerNet) + "])\n");
  auto equalProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, config, equalRoutes.view()));
  auto equalCandidate =
      take(pnr::createCanonicalSpatialCandidate(equalProblem));
  pnr::SpatialActionExecutorScratch equalExecutor;
  requireSuccess(equalExecutor.prepare(*equalCandidate));
  auto equalProbe = equalExecutor.probe(*equalCandidate, routeFirst);
  if (equalProbe) {
    if (selectedRouteSets(*equalProblem, *equalCandidate, routedNet)
            .traversals !=
        selectedRouteSets(*equalProblem, *equalCandidate, peerNet).traversals)
      fail("one Action exposed unequal selected traversal sets");
    requireSuccess(equalProbe->discard());
  } else {
    llvm::consumeError(equalProbe.takeError());
    if (!equalCandidate->routeTree(routedNet).isUnrouted() ||
        !equalCandidate->routeTree(peerNet).isUnrouted())
      fail("rejected traversal equality Action changed one RouteTree");
  }
  requireSuccess(equalCandidate->verify());

  const auto disjointRoutes = makeConstraints(
      "    mapping.constraint.disjoint "
      "projection(net_selected_physical_traversals) subjects([" +
      producerAttr(routedNet) + ", " + producerAttr(peerNet) + "])\n");
  auto disjointProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, config, disjointRoutes.view()));
  auto disjointCandidate =
      take(pnr::createCanonicalSpatialCandidate(disjointProblem));
  pnr::SpatialActionExecutorScratch disjointExecutor;
  requireSuccess(disjointExecutor.prepare(*disjointCandidate));
  auto firstProbe = disjointExecutor.probe(*disjointCandidate, routeFirst);
  if (!firstProbe)
    fail("first member of traversal Disjoint was not independently routable: " +
         llvm::toString(firstProbe.takeError()));
  requireSuccess(firstProbe->commit());
  const SelectedRouteSets firstRoute =
      selectedRouteSets(*disjointProblem, *disjointCandidate, routedNet);
  const pnr::SpatialMappingAction routePeer =
      pnr::SpatialTransportRoutingAction{
          pnr::SpatialWholeNetRoutingAction{peerNet}};
  auto peerProbe = disjointExecutor.probe(*disjointCandidate, routePeer);
  if (peerProbe) {
    const SelectedRouteSets secondRoute =
        selectedRouteSets(*disjointProblem, *disjointCandidate, peerNet);
    if (intersects(firstRoute.traversals, secondRoute.traversals))
      fail("Disjoint Action selected a shared traversal");
    requireSuccess(peerProbe->commit());
  } else {
    llvm::consumeError(peerProbe.takeError());
    if (selectedRouteSets(*disjointProblem, *disjointCandidate, routedNet)
                .traversals != firstRoute.traversals ||
        !disjointCandidate->routeTree(peerNet).isUnrouted())
      fail("rejected Disjoint Action changed an existing route");
  }
  requireSuccess(disjointCandidate->verify());
}
