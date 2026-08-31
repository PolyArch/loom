#include "SpatialRuntimeCounterexampleTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/MappingConstraintSetMigration.h"
#include "PnR/FrozenConstraintIndex.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::test {
namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial runtime counterexample test: " << message << '\n';
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

bool rejected(llvm::Error error) {
  if (!error)
    return false;
  llvm::consumeError(std::move(error));
  return true;
}

ArtifactRootReference syntheticRootReference(llvm::StringRef schemaIdentity,
                                             SchemaVersion schemaVersion,
                                             std::uint8_t identityByte) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes;
  bytes.fill(identityByte);
  return ArtifactRootReference{schemaIdentity.str(), schemaVersion,
                               take(ArtifactIdentity::fromBytes(bytes))};
}

ComponentViewDigest syntheticDigest(std::uint8_t digestByte) {
  std::array<std::uint8_t, ComponentViewDigest::byteSize> bytes;
  bytes.fill(digestByte);
  return take(ComponentViewDigest::fromBytes(bytes));
}

} // namespace

void exerciseSpatialRuntimeCounterexampleNoGood(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const fabric::FabricArtifactView &foreignFabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parent,
    const mapping::FinalizedSpatialMapping &mapping,
    const fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const pnr::ResolvedPnrConfigView &pnrConfig, const ArtifactStore &store) {
  // Quote exact choices the sealed Mapping really made. A RouteTree can select
  // a traversal at its source attachment, on a routed node, or at a sink
  // attachment; all three positions belong to NetUsesTraversal.
  if (mapping.view().routeTrees().empty())
    fail("sealed Spatial Mapping published no RouteTree to quote");
  const mapping::SpatialRouteTreeView *selectedRoute = nullptr;
  std::optional<fabric::FabricPhysicalTraversalRef> selectedTraversal;
  for (const auto &route : mapping.view().routeTrees()) {
    if (route.localTraversal) {
      selectedRoute = &route;
      selectedTraversal = *route.localTraversal;
      break;
    }
    for (const auto &node : route.nodes) {
      if (!node.incomingTraversal)
        continue;
      selectedRoute = &route;
      selectedTraversal = *node.incomingTraversal;
      break;
    }
    if (selectedTraversal)
      break;
    for (const auto &sink : route.sinks) {
      if (!sink.localTraversal)
        continue;
      selectedRoute = &route;
      selectedTraversal = *sink.localTraversal;
      break;
    }
    if (selectedTraversal)
      break;
  }
  if (!selectedTraversal)
    fail("sealed Spatial Mapping selected no physical traversal to quote");

  const mapping::SpatialNoGoodLiteral attachment =
      mapping::SpatialTransferAttachmentEqualsLiteral{
          mapping::SpatialConstraintTransferTerminal{selectedRoute->logicalNet,
                                                     std::nullopt},
          selectedRoute->rootEndpoint};
  const mapping::SpatialNoGoodLiteral usesTraversal =
      mapping::SpatialNetUsesTraversalLiteral{selectedRoute->logicalNet,
                                              std::nullopt, *selectedTraversal};
  std::optional<mapping::SpatialNoGoodLiteral> selectedTag;
  for (const mapping::SpatialPhysicalTagSegmentView &segment :
       mapping.view().physicalTagSegments()) {
    if (segment.routeTreeOrdinal >= mapping.view().routeTrees().size() ||
        segment.resourceUseOrdinal >= mapping.view().resourceUses().size())
      fail("sealed Spatial Mapping has a malformed Physical Tag segment");
    const auto &assignments = mapping.view()
                                  .resourceUses()[segment.resourceUseOrdinal]
                                  .sharingAssignments;
    if (assignments.size() != 1)
      continue;
    const auto *tag =
        std::get_if<::fabric::PhysicalTagPatternValue>(&assignments.front());
    if (!tag)
      continue;
    selectedTag =
        mapping::SpatialNoGoodLiteral{mapping::SpatialNetTagEqualsLiteral{
            mapping.view().routeTrees()[segment.routeTreeOrdinal].logicalNet,
            segment.segmentOrdinal, tag->value}};
    break;
  }
  const mapping::SpatialNoGoodLiteral migrationTagLiteral =
      mapping::SpatialNetTagEqualsLiteral{selectedRoute->logicalNet, 0,
                                          llvm::APInt(4, 3)};

  const auto publish =
      [&](llvm::ArrayRef<mapping::SpatialNoGoodLiteral> literals) {
        return take(mapping::finalizeSpatialRuntimeCounterexampleConstraintSet(
            parent.reference(), literals, store));
      };

  // Every literal still holds of the exact parent Mapping, so the parent is
  // rejected: it repeats the recorded counterexample.
  const auto rejectsMapping = [&](const auto &constraint) {
    bool rejected = false;
    llvm::handleAllErrors(
        mapping::admitSpatialMappingConstraints(
            dataflow, techMapping, fabric, constraint.view(), mapping.view()),
        [&](const mapping::SpatialMappingConstraintRejection &) {
          rejected = true;
        },
        [&](const llvm::ErrorInfoBase &error) { fail(error.message()); });
    return rejected;
  };
  const auto bothHold = publish({usesTraversal, attachment});
  if (!rejectsMapping(bothHold))
    fail("a no-good whose literals all hold did not reject its own parent "
         "Spatial Mapping");
  if (selectedTag && !rejectsMapping(publish({*selectedTag})))
    fail("an exact Physical Tag segment literal did not reject its parent");

  const mapping::SpatialNoGoodLiteral parentIdentity =
      mapping::SpatialMappingIdentityEqualsLiteral{mapping.reference(),
                                                   nullptr};
  const auto exactParent = publish({parentIdentity});
  if (!rejectsMapping(exactParent))
    fail("an exact SpatialMapping identity literal did not reject its parent");
  if (!(publish({parentIdentity, usesTraversal, parentIdentity}).reference() ==
        publish({usesTraversal, parentIdentity}).reference()))
    fail("SpatialMapping identity literal ordering or duplication changed "
         "constraint identity");

  const mapping::SpatialRuntimeCounterexampleNoGoodView::Lineage lineage{
      mapping.reference(),
      syntheticRootReference("loom.evaluation_evidence", {3, 1}, 41),
      syntheticRootReference("loom.evaluation_request", {4, 0}, 42),
      syntheticRootReference("loom.simulation_execution", {2, 0}, 43),
      syntheticDigest(44)};
  const auto learned =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(), {parentIdentity, usesTraversal}, lineage, store));
  const auto learnedNoGood =
      llvm::find_if(learned.view().clauses(), [](const auto &clause) {
        const auto *noGood =
            std::get_if<mapping::SpatialRuntimeCounterexampleNoGoodView>(
                &clause);
        return noGood && noGood->lineage.has_value();
      });
  if (learnedNoGood == learned.view().clauses().end() ||
      !(*std::get<mapping::SpatialRuntimeCounterexampleNoGoodView>(
             *learnedNoGood)
             .lineage == lineage))
    fail("runtime-counterexample lineage did not survive strict import");

  const auto strictParent = take(
      mapping::importSpatialMappingConstraintSet(parent.reference(), store));
  auto cachedParentMapping =
      std::make_shared<const mapping::FinalizedSpatialMapping>(
          take(mapping::importSpatialMapping(mapping.reference(), store)));
  const mapping::SpatialNoGoodLiteral cachedParentIdentity =
      mapping::SpatialMappingIdentityEqualsLiteral{mapping.reference(),
                                                   cachedParentMapping};
  const auto requireSameConstraint = [&](const auto &lhs, const auto &rhs,
                                         llvm::StringRef diagnostic) {
    if (lhs.reference() != rhs.reference() ||
        !lhs.canonicalBytes().bytes().equals(rhs.canonicalBytes().bytes()))
      fail(diagnostic);
  };

  const auto incrementalLearned =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          strictParent, {cachedParentIdentity, usesTraversal}, lineage, store));
  requireSameConstraint(
      incrementalLearned, learned,
      "incremental promoted no-good finalization diverged from cold rebuild");

  const auto coldRepeated =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          learned.reference(), {cachedParentIdentity, usesTraversal}, lineage,
          store));
  const auto incrementalRepeated =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          incrementalLearned, {cachedParentIdentity, usesTraversal}, lineage,
          store));
  requireSameConstraint(coldRepeated, learned,
                        "cold promoted clause repetition changed identity");
  requireSameConstraint(
      incrementalRepeated, learned,
      "incremental promoted clause repetition changed identity");

  const auto coldAccumulated =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          learned.reference(), {cachedParentIdentity, attachment}, lineage,
          store));
  const auto incrementalAccumulated =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          incrementalLearned, {cachedParentIdentity, attachment}, lineage,
          store));
  const auto coldSecondFirst =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(), {cachedParentIdentity, attachment}, lineage,
          store));
  const auto coldReverseAccumulated =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          coldSecondFirst.reference(), {cachedParentIdentity, usesTraversal},
          lineage, store));
  requireSameConstraint(
      incrementalAccumulated, coldAccumulated,
      "incremental two-clause accumulation diverged from cold rebuild");
  requireSameConstraint(
      incrementalAccumulated, coldReverseAccumulated,
      "promoted clause discovery order changed accumulated identity");

  auto tamperedDigest = lineage;
  tamperedDigest.certificateDigest = syntheticDigest(45);
  const auto digestChanged =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(), {parentIdentity, usesTraversal}, tamperedDigest,
          store));
  if (digestChanged.reference() == learned.reference())
    fail("tampering a runtime certificate digest did not change constraint "
         "identity");
  auto tamperedEvidence = lineage;
  tamperedEvidence.runtimeEvidence =
      syntheticRootReference("loom.evaluation_evidence", {3, 1}, 46);
  const auto evidenceChanged =
      take(mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(), {parentIdentity, usesTraversal}, tamperedEvidence,
          store));
  if (evidenceChanged.reference() == learned.reference())
    fail("tampering a runtime evidence reference did not change constraint "
         "identity");

  auto foreignParentLineage = lineage;
  foreignParentLineage.parentMapping =
      syntheticRootReference(mapping::mappingArtifactSchema.identity,
                             mapping::mappingArtifactSchema.version, 47);
  auto mismatchedLineage =
      mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(), {parentIdentity, usesTraversal},
          foreignParentLineage, store);
  if (mismatchedLineage)
    fail("runtime lineage accepted a parent that differs from its exact "
         "SpatialMapping literal");
  llvm::consumeError(mismatchedLineage.takeError());
  auto missingParentLiteral =
      mapping::finalizePromotedSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(), {usesTraversal}, lineage, store);
  if (missingParentLiteral)
    fail("runtime lineage verified without one exact parent Mapping literal");
  llvm::consumeError(missingParentLiteral.takeError());

  {
    auto unconstrainedProblem = take(
        pnr::freezeSpatialPnrProblem(dataflow, techMapping, fabric,
                                     physicalTiming, pnrConfig, parent.view()));
    auto unconstrainedCandidate =
        take(pnr::createCanonicalSpatialCandidate(unconstrainedProblem));
    pnr::SpatialActionDomainScratch actionDomain;
    requireSuccess(actionDomain.prepare(*unconstrainedProblem));
    requireSuccess(actionDomain.rebuild(*unconstrainedCandidate));

    std::optional<mapping::SpatialConstraintTransferTerminal> terminal;
    std::optional<pnr::SpatialResourceAllocationAction> alternate;
    const auto logicalNets = unconstrainedProblem->transfers().logicalNets();
    const auto sinkBindings =
        unconstrainedProblem->transfers().logicalNetSinkBindings();
    const auto sinks = unconstrainedProblem->transfers().logicalNetSinks();
    const auto choices = actionDomain.view().resourceChoices;
    const auto findAlternate = [&](pnr::FrozenSpatialTerminalBinding binding,
                                   pnr::PnrIndex selectedEndpoint)
        -> std::optional<pnr::SpatialResourceAllocationAction> {
      for (const auto &choice : choices) {
        if (binding.kind == pnr::FrozenSpatialTerminalBindingKind::PortDemand) {
          const auto *port =
              std::get_if<pnr::SpatialPortAttachmentAction>(&choice);
          if (!port || port->demand != binding.index)
            continue;
          if (unconstrainedProblem->ports()
                  .attachmentOptions()[port->attachmentOption]
                  .endpoint != selectedEndpoint)
            return choice;
          continue;
        }
        const auto *boundary =
            std::get_if<pnr::SpatialGraphBoundaryAttachmentAction>(&choice);
        if (!boundary || boundary->boundary != binding.index)
          continue;
        if (unconstrainedProblem->ports()
                .attachmentOptions()[boundary->attachmentOption]
                .endpoint != selectedEndpoint)
          return choice;
      }
      return std::nullopt;
    };

    for (pnr::PnrIndex logicalNet = 0;
         logicalNet < logicalNets.size() && !alternate; ++logicalNet) {
      const auto source = unconstrainedProblem->transfers()
                              .logicalNetSourceBindings()[logicalNet];
      alternate = findAlternate(
          source, unconstrainedCandidate->logicalNetSourceEndpoint(logicalNet));
      if (alternate) {
        terminal = mapping::SpatialConstraintTransferTerminal{
            logicalNets[logicalNet].producer, std::nullopt};
        break;
      }
      for (pnr::PnrIndex sink = 0;
           sink < logicalNets[logicalNet].sinkCount && !alternate; ++sink) {
        const auto binding =
            sinkBindings[logicalNets[logicalNet].sinkOffset + sink];
        alternate = findAlternate(
            binding,
            unconstrainedCandidate->logicalNetSinkEndpoint(logicalNet, sink));
        if (alternate)
          terminal = mapping::SpatialConstraintTransferTerminal{
              logicalNets[logicalNet].producer,
              sinks[logicalNets[logicalNet].sinkOffset + sink]};
      }
    }
    if (!terminal || !alternate)
      fail("Spatial candidate fixture has no terminal with a distinct "
           "endpoint");

    pnr::PnrIndex selectedEndpoint = 0;
    pnr::PnrIndex logicalNet = 0;
    for (; logicalNet < logicalNets.size(); ++logicalNet) {
      if (!(logicalNets[logicalNet].producer == terminal->producer))
        continue;
      if (!terminal->consumer) {
        selectedEndpoint =
            unconstrainedCandidate->logicalNetSourceEndpoint(logicalNet);
        break;
      }
      const auto netSinks = sinks.slice(logicalNets[logicalNet].sinkOffset,
                                        logicalNets[logicalNet].sinkCount);
      const auto found = llvm::find(netSinks, *terminal->consumer);
      if (found == netSinks.end())
        continue;
      selectedEndpoint = unconstrainedCandidate->logicalNetSinkEndpoint(
          logicalNet, static_cast<pnr::PnrIndex>(found - netSinks.begin()));
      break;
    }
    if (logicalNet == logicalNets.size())
      fail("runtime-counterexample terminal lost its logical-net owner");
    const auto endpointReference = unconstrainedProblem->routing()
                                       .routingEndpoints()[selectedEndpoint]
                                       .reference;
    const mapping::SpatialNoGoodLiteral attachmentOnly =
        mapping::SpatialTransferAttachmentEqualsLiteral{*terminal,
                                                        endpointReference};
    const auto attachmentConstraint = publish({attachmentOnly});
    auto constrainedProblem = take(pnr::freezeSpatialPnrProblem(
        dataflow, techMapping, fabric, physicalTiming, pnrConfig,
        attachmentConstraint.view()));
    auto constrainedCandidate =
        take(pnr::createCanonicalSpatialCandidate(constrainedProblem));
    if (constrainedCandidate->runtimeCounterexampleViolation() != 1 ||
        constrainedCandidate->firstRuntimeCounterexampleViolation() != 0)
      fail("canonical candidate did not expose its exact no-good violation");

    pnr::SpatialCandidateScratch scratch;
    requireSuccess(scratch.prepare(*constrainedProblem));
    auto move = take(constrainedCandidate->beginMove(scratch));
    if (const auto *port =
            std::get_if<pnr::SpatialPortAttachmentAction>(&*alternate))
      requireSuccess(
          move.setPortAttachment(port->demand, port->attachmentOption));
    else {
      const auto &boundary =
          std::get<pnr::SpatialGraphBoundaryAttachmentAction>(*alternate);
      requireSuccess(move.setGraphBoundaryAttachment(
          boundary.boundary, boundary.attachmentOption));
    }
    const bool closed = take(move.close());
    if (!closed || constrainedCandidate->runtimeCounterexampleViolation() != 0)
      fail("changing one exact attachment did not satisfy the no-good");
    move.rollback();
    if (constrainedCandidate->runtimeCounterexampleViolation() != 1)
      fail("runtime-counterexample rollback did not restore the violation");
    requireSuccess(constrainedCandidate->verify());
  }

  // The ordinary Spatial provider must enforce the same clause while it
  // searches, then independently admit every published Mapping. This reaches
  // prospective route projection, committed incremental state, rollback, cold
  // verification, and the publication gate without a test-only state hook.
  const pnr::SpatialPnrGenerationInputs noGoodInputs{
      dataflow,  techMapping,     fabric, physicalTiming,
      pnrConfig, bothHold.view(), store};
  const auto noGoodSearch = pnr::generateSpatialMappings(noGoodInputs);
  const auto *noGoodMappings =
      std::get_if<pnr::GeneratedSpatialMappings>(&noGoodSearch);
  if (!noGoodMappings || noGoodMappings->candidates.empty())
    fail("Spatial PnR did not find a candidate satisfying an exact runtime "
         "counterexample clause");
  for (const ArtifactRootReference &candidate : noGoodMappings->candidates) {
    const auto repaired = take(mapping::importSpatialMapping(candidate, store));
    requireSuccess(mapping::admitSpatialMappingConstraints(
        dataflow, techMapping, fabric, bothHold.view(), repaired.view()));
    if (candidate == mapping.reference())
      fail("no-good search republished the rejected parent Mapping");
    requireSuccess(mapping::admitSpatialMappingConstraints(
        dataflow, techMapping, fabric, exactParent.view(), repaired.view()));
  }

  // Changing one literal satisfies the clause. A traversal the route does not
  // select is independently verifiable as not-holding, so the same Mapping is
  // admitted again.
  const auto routeSelects =
      [&](const fabric::FabricPhysicalTraversalRef &candidate) {
        return mapping::spatialRouteTreeSelectsTraversal(*selectedRoute,
                                                         candidate);
      };
  std::optional<mapping::SpatialNoGoodLiteral> changedLiteral;
  for (const auto &traversal : fabric.physicalTraversals()) {
    if (routeSelects(traversal.reference))
      continue;
    changedLiteral =
        mapping::SpatialNoGoodLiteral{mapping::SpatialNetUsesTraversalLiteral{
            selectedRoute->logicalNet, std::nullopt, traversal.reference}};
    break;
  }
  if (!changedLiteral)
    fail("Fabric offers no unselected traversal to change a literal to");

  const auto oneChanged = publish({*changedLiteral, attachment});
  requireSuccess(mapping::admitSpatialMappingConstraints(
      dataflow, techMapping, fabric, oneChanged.view(), mapping.view()));

  // An unrelated exact choice must not reject: a clause naming only choices
  // this Mapping did not make is satisfied.
  const auto unrelated = publish({*changedLiteral});
  requireSuccess(mapping::admitSpatialMappingConstraints(
      dataflow, techMapping, fabric, unrelated.view(), mapping.view()));

  if (selectedTag) {
    auto changedTag =
        std::get<mapping::SpatialNetTagEqualsLiteral>(*selectedTag);
    changedTag.value.flipBit(0);
    const auto differentTag =
        publish({mapping::SpatialNoGoodLiteral{std::move(changedTag)}});
    requireSuccess(mapping::admitSpatialMappingConstraints(
        dataflow, techMapping, fabric, differentTag.view(), mapping.view()));
  }

  // A sink-qualified traversal is evaluated only on that sink's exact branch,
  // including the source- and sink-local traversal positions. A sink-qualified
  // attachment likewise resolves to that sink node rather than the route root.
  const mapping::SpatialRouteTreeView *qualifiedRoute = nullptr;
  const mapping::SpatialRouteSinkView *qualifiedSink = nullptr;
  std::optional<fabric::FabricPhysicalTraversalRef> branchTraversal;
  for (const auto &route : mapping.view().routeTrees()) {
    for (const auto &sink : route.sinks) {
      if (route.localTraversal)
        branchTraversal = *route.localTraversal;
      for (std::optional<std::uint64_t> cursor = sink.nodeOrdinal;
           !branchTraversal && cursor;
           cursor = route.nodes[*cursor].parentOrdinal)
        if (route.nodes[*cursor].incomingTraversal)
          branchTraversal = *route.nodes[*cursor].incomingTraversal;
      if (!branchTraversal && sink.localTraversal)
        branchTraversal = *sink.localTraversal;
      if (!branchTraversal)
        continue;
      qualifiedRoute = &route;
      qualifiedSink = &sink;
      break;
    }
    if (qualifiedSink)
      break;
  }
  if (!qualifiedRoute || !qualifiedSink || !branchTraversal)
    fail("sealed Spatial Mapping has no sink branch traversal to quote");

  const mapping::SpatialNoGoodLiteral qualifiedTraversal =
      mapping::SpatialNetUsesTraversalLiteral{
          qualifiedRoute->logicalNet, qualifiedSink->sink, *branchTraversal};
  if (!rejectsMapping(publish({qualifiedTraversal})))
    fail("a sink-qualified traversal did not hold on its selected branch");
  const mapping::SpatialNoGoodLiteral qualifiedAttachment =
      mapping::SpatialTransferAttachmentEqualsLiteral{
          mapping::SpatialConstraintTransferTerminal{qualifiedRoute->logicalNet,
                                                     qualifiedSink->sink},
          qualifiedRoute->nodes[qualifiedSink->nodeOrdinal].endpoint};
  if (!rejectsMapping(publish({qualifiedAttachment})))
    fail("a sink-qualified attachment did not resolve to its selected node");

  const auto branchSelects =
      [&](const fabric::FabricPhysicalTraversalRef &candidate) {
        if (qualifiedRoute->localTraversal &&
            *qualifiedRoute->localTraversal == candidate)
          return true;
        for (std::optional<std::uint64_t> cursor = qualifiedSink->nodeOrdinal;
             cursor; cursor = qualifiedRoute->nodes[*cursor].parentOrdinal)
          if (qualifiedRoute->nodes[*cursor].incomingTraversal &&
              *qualifiedRoute->nodes[*cursor].incomingTraversal == candidate)
            return true;
        return qualifiedSink->localTraversal &&
               *qualifiedSink->localTraversal == candidate;
      };
  std::optional<fabric::FabricPhysicalTraversalRef> absentBranchTraversal;
  for (const auto &traversal : fabric.physicalTraversals())
    if (!branchSelects(traversal.reference)) {
      absentBranchTraversal = traversal.reference;
      break;
    }
  if (!absentBranchTraversal)
    fail("Fabric offers no traversal outside the selected sink branch");
  const auto changedBranch = publish(
      {mapping::SpatialNoGoodLiteral{mapping::SpatialNetUsesTraversalLiteral{
          qualifiedRoute->logicalNet, qualifiedSink->sink,
          *absentBranchTraversal}}});
  requireSuccess(mapping::admitSpatialMappingConstraints(
      dataflow, techMapping, fabric, changedBranch.view(), mapping.view()));

  // Republishing the same counterexample is identity-idempotent, and literal
  // discovery order is not identity.
  if (!(publish({usesTraversal, attachment}).reference() ==
        bothHold.reference()))
    fail("republishing an identical no-good changed the Artifact identity");
  if (!(publish({attachment, usesTraversal}).reference() ==
        bothHold.reference()))
    fail("no-good literal discovery order changed the Artifact identity");

  // Two distinct counterexamples form a canonical union whose identity does
  // not depend on the order they were recorded in. Recording {A} then B must
  // equal recording {B} then A.
  const auto onlyAttachment = publish({attachment});
  const auto unionA =
      take(mapping::finalizeSpatialRuntimeCounterexampleConstraintSet(
          bothHold.reference(), {attachment}, store));
  const auto unionB =
      take(mapping::finalizeSpatialRuntimeCounterexampleConstraintSet(
          onlyAttachment.reference(), {usesTraversal, attachment}, store));
  if (!(unionA.reference() == unionB.reference()))
    fail("no-good clause discovery order changed the canonical union identity");
  if (unionA.reference() == bothHold.reference())
    fail("a distinct no-good clause did not change the Artifact identity");

  // Foreign owners are rejected before any clause is evaluated.
  llvm::Error foreign = mapping::admitSpatialMappingConstraints(
      dataflow, techMapping, foreignFabric, bothHold.view(), mapping.view());
  if (!foreign)
    fail("no-good admission accepted a foreign Fabric owner");
  llvm::consumeError(std::move(foreign));

  // Freeze preserves the clause, and a set carrying only no-goods is not
  // empty: a consumer that early-outs on empty() must not skip it.
  auto frozen = take(pnr::detail::buildFrozenConstraintIndex(bothHold.view()));
  if (frozen.noGoods().size() != 1 ||
      frozen.noGoods().front().literals.size() != 2)
    fail("FrozenConstraintIndex lost a runtime-counterexample no-good");
  if (frozen.empty())
    fail("a constraint index carrying a no-good reported itself empty");

  // The 1.0 family is reachable only through the explicit migration owner.
  // `parent` predates the extension, so its canonical bytes are exactly what a
  // 1.0 payload would hold; republishing them under the 1.0 descriptor yields a
  // genuine legacy reference with a different identity.
  const ArtifactRootReference legacyParent{
      mapping::mappingConstraintSetSchemaV1_0.identity.str(),
      mapping::mappingConstraintSetSchemaV1_0.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_0,
                     parent.canonicalBytes()))};
  if (legacyParent.artifact == parent.reference().artifact)
    fail("the 1.0 and 1.3 identities of identical bytes collided");

  llvm::Error strict =
      mapping::importSpatialMappingConstraintSet(legacyParent, store)
          .takeError();
  if (!strict)
    fail("the strict 1.3 Spatial importer accepted a 1.0 reference");
  llvm::consumeError(std::move(strict));

  const auto migratedV1_1 = take(
      mapping::migrateSpatialConstraintRootV1_0ToV1_1(legacyParent, store));
  const ArtifactRootReference nativeV1_1{
      mapping::mappingConstraintSetSchemaV1_1.identity.str(),
      mapping::mappingConstraintSetSchemaV1_1.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_1,
                     parent.canonicalBytes()))};
  if (!(migratedV1_1 == nativeV1_1))
    fail("Spatial 1.0-to-1.1 migration did not reproduce the native 1.1 "
         "identity");
  const auto migratedV1_2 = take(
      mapping::migrateSpatialConstraintRootV1_1ToV1_2(migratedV1_1, store));
  const ArtifactRootReference nativeV1_2{
      mapping::mappingConstraintSetSchemaV1_2.identity.str(),
      mapping::mappingConstraintSetSchemaV1_2.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_2,
                     parent.canonicalBytes()))};
  if (!(migratedV1_2 == nativeV1_2))
    fail("Spatial 1.1-to-1.2 migration did not reproduce the native 1.2 "
         "identity");
  if (!(take(mapping::migrateSpatialConstraintRootV1_0ToV1_2(
            legacyParent, store)) == nativeV1_2))
    fail("Spatial 1.0-to-1.2 migration chain changed the cold identity");
  if (!(take(mapping::migrateSpatialConstraintRootV1_2ToV1_3(
            nativeV1_2, store)) == parent.reference()))
    fail("Spatial 1.2-to-1.3 migration changed the native 1.3 identity");
  if (!(take(mapping::migrateSpatialConstraintRootV1_0ToV1_3(
            legacyParent, store)) == parent.reference()))
    fail("Spatial 1.0-to-1.3 migration chain changed the cold identity");

  const ArtifactRootReference legacyNoGoodV1_1{
      mapping::mappingConstraintSetSchemaV1_1.identity.str(),
      mapping::mappingConstraintSetSchemaV1_1.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_1,
                     bothHold.canonicalBytes()))};
  const ArtifactRootReference nativeNoGoodV1_2{
      mapping::mappingConstraintSetSchemaV1_2.identity.str(),
      mapping::mappingConstraintSetSchemaV1_2.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_2,
                     bothHold.canonicalBytes()))};
  if (!(take(mapping::migrateSpatialConstraintRootV1_1ToV1_2(
            legacyNoGoodV1_1, store)) == nativeNoGoodV1_2))
    fail("Spatial 1.1 no-good migration changed the native 1.2 identity");
  if (!(take(mapping::migrateSpatialConstraintRootV1_2ToV1_3(
            nativeNoGoodV1_2, store)) == bothHold.reference()))
    fail("Spatial 1.2 no-good migration changed the native 1.3 identity");
  auto legacyNoGoodBytes = take(store.get(legacyNoGoodV1_1));
  if (!legacyNoGoodBytes.bytes().equals(bothHold.canonicalBytes().bytes()))
    fail("Spatial 1.1 no-good fixture changed its canonical payload bytes");

  // A 1.0 reference to bytes that already carry the 1.1-only clause kind is
  // mislabelled, not due an upgrade.
  const ArtifactRootReference mislabelled{
      mapping::mappingConstraintSetSchemaV1_0.identity.str(),
      mapping::mappingConstraintSetSchemaV1_0.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_0,
                     bothHold.canonicalBytes()))};
  llvm::Error mislabelledError =
      mapping::migrateSpatialConstraintRootV1_0ToV1_1(mislabelled, store)
          .takeError();
  if (!mislabelledError)
    fail("Spatial migration accepted a 1.0 payload holding a 1.1-only clause");
  llvm::consumeError(std::move(mislabelledError));

  const auto tagOnly = publish({migrationTagLiteral});
  const ArtifactRootReference mislabelledV1_1{
      mapping::mappingConstraintSetSchemaV1_1.identity.str(),
      mapping::mappingConstraintSetSchemaV1_1.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_1,
                     tagOnly.canonicalBytes()))};
  auto tagMigration =
      mapping::migrateSpatialConstraintRootV1_1ToV1_2(mislabelledV1_1, store);
  if (tagMigration)
    fail("Spatial migration accepted a 1.1 payload holding a 1.2-only tag "
         "literal");
  llvm::consumeError(tagMigration.takeError());

  const ArtifactRootReference mislabelledV1_2{
      mapping::mappingConstraintSetSchemaV1_2.identity.str(),
      mapping::mappingConstraintSetSchemaV1_2.version,
      take(store.put(mapping::mappingConstraintSetSchemaV1_2,
                     exactParent.canonicalBytes()))};
  auto identityMigration =
      mapping::migrateSpatialConstraintRootV1_2ToV1_3(mislabelledV1_2, store);
  if (identityMigration)
    fail("Spatial migration accepted a 1.2 payload holding a 1.3-only exact "
         "Mapping literal");
  llvm::consumeError(identityMigration.takeError());

  // Migration accepts only what the strict 1.0 importer could have accepted.
  // Perturbing the stored payload so it is no longer canonical under its own
  // family must be rejected, not silently normalized into a valid 1.1
  // artifact. Duplicating the trailing newline keeps the payload parseable and
  // semantically identical while making it noncanonical.
  {
    const auto &bytes = parent.canonicalBytes().bytes();
    std::vector<std::uint8_t> perturbed(bytes.begin(), bytes.end());
    perturbed.push_back('\n');
    const ArtifactRootReference noncanonical{
        mapping::mappingConstraintSetSchemaV1_0.identity.str(),
        mapping::mappingConstraintSetSchemaV1_0.version,
        take(store.put(mapping::mappingConstraintSetSchemaV1_0,
                       CanonicalSemanticBytes(std::move(perturbed))))};
    auto migratedNoncanonical =
        mapping::migrateSpatialConstraintRootV1_0ToV1_1(noncanonical, store);
    if (migratedNoncanonical)
      fail("Spatial migration normalized a noncanonical 1.0 payload instead "
           "of rejecting it");
    const std::string diagnostic =
        llvm::toString(migratedNoncanonical.takeError());
    if (!llvm::StringRef(diagnostic).contains("is not canonical"))
      fail("noncanonical 1.0 payload was rejected for the wrong reason: " +
           diagnostic);
  }
}

void exerciseSpatialPhysicalTagRuntimeCounterexampleNoGood(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parent,
    const mapping::FinalizedSpatialMapping &mapping,
    const pnr::ResolvedPnrConfigView &pnrConfig,
    pnr::SpatialCandidateState &candidate, const ArtifactStore &store) {
  if (mapping.view().physicalTagSegments().empty())
    fail("tagged SpatialMapping has no canonical tag segment");
  const auto &segment = mapping.view().physicalTagSegments().front();
  if (segment.routeTreeOrdinal >= mapping.view().routeTrees().size() ||
      segment.resourceUseOrdinal >= mapping.view().resourceUses().size())
    fail("tagged SpatialMapping has a malformed tag segment owner");
  const auto &tagAssignments = mapping.view()
                                   .resourceUses()[segment.resourceUseOrdinal]
                                   .sharingAssignments;
  const auto *selectedValue =
      tagAssignments.size() == 1
          ? std::get_if<::fabric::PhysicalTagPatternValue>(
                &tagAssignments.front())
          : nullptr;
  if (!selectedValue)
    fail("tagged SpatialMapping segment has no typed tag value");
  const mapping::SpatialNoGoodLiteral exactTag =
      mapping::SpatialNetTagEqualsLiteral{
          mapping.view().routeTrees()[segment.routeTreeOrdinal].logicalNet,
          segment.segmentOrdinal, selectedValue->value};
  const mapping::SpatialNoGoodLiteral exactMapping =
      mapping::SpatialMappingIdentityEqualsLiteral{mapping.reference(),
                                                   nullptr};
  const auto exactTagConstraint =
      take(mapping::finalizeSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(), {exactTag, exactMapping}, store));
  if (!rejected(mapping::admitSpatialMappingConstraints(
          dataflow, techMapping, fabric, exactTagConstraint.view(),
          mapping.view())))
    fail("sealed tag literal did not reject its exact SpatialMapping");
  auto differentTag = std::get<mapping::SpatialNetTagEqualsLiteral>(exactTag);
  differentTag.value.flipBit(0);
  const auto differentTagConstraint =
      take(mapping::finalizeSpatialRuntimeCounterexampleConstraintSet(
          parent.reference(),
          {mapping::SpatialNoGoodLiteral{std::move(differentTag)}}, store));
  requireSuccess(mapping::admitSpatialMappingConstraints(
      dataflow, techMapping, fabric, differentTagConstraint.view(),
      mapping.view()));

  auto tagConstrainedProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, exactTagConstraint.view()));
  auto tagConstrainedSnapshot = take(candidate.snapshotFullyRouted());
  tagConstrainedSnapshot.problem = tagConstrainedProblem;
  auto tagConstrainedCandidate =
      take(pnr::SpatialCandidateState::materializeFullyRouted(
          tagConstrainedSnapshot));
  if (tagConstrainedCandidate->runtimeCounterexampleViolation() != 1)
    fail("warm candidate reconstruction lost an exact Physical Tag no-good "
         "violation");
  auto tagConstrainedClone = take(tagConstrainedCandidate->cloneFullyRouted());
  if (tagConstrainedClone->runtimeCounterexampleViolation() != 1)
    fail("candidate cloning lost an exact Physical Tag no-good violation");

  const auto &tagLiteral =
      std::get<mapping::SpatialNetTagEqualsLiteral>(exactTag);
  const auto tagNet =
      llvm::find_if(tagConstrainedProblem->transfers().logicalNets(),
                    [&](const pnr::FrozenSpatialLogicalNet &logicalNet) {
                      return logicalNet.producer == tagLiteral.producer;
                    });
  if (tagNet == tagConstrainedProblem->transfers().logicalNets().end())
    fail("Physical Tag no-good producer is absent from the frozen problem");
  const auto tagNetOrdinal = static_cast<pnr::PnrIndex>(
      tagNet - tagConstrainedProblem->transfers().logicalNets().begin());
  pnr::SpatialCandidateScratch tagNoGoodScratch;
  requireSuccess(tagNoGoodScratch.prepare(*tagConstrainedProblem));
  auto tagNoGoodMove =
      take(tagConstrainedCandidate->beginMove(tagNoGoodScratch));
  requireSuccess(tagNoGoodMove.ripUpWholeRoute(tagNetOrdinal));
  const auto tagNoGoodProjection = take(tagNoGoodMove.projectCurrentRoutes());
  if (tagNoGoodProjection.runtimeCounterexampleViolation != 0)
    fail("provisional route-local tag removal retained its exact no-good");
  if (!take(tagNoGoodMove.close()))
    fail("route-local tag no-good move closed a handshake cycle");
  if (tagConstrainedCandidate->runtimeCounterexampleViolation() != 0)
    fail("closed route-local tag removal retained its exact no-good");
  tagNoGoodMove.rollback();
  if (tagConstrainedCandidate->runtimeCounterexampleViolation() != 1)
    fail("Physical Tag no-good rollback did not restore its violation");
  requireSuccess(tagConstrainedCandidate->verify());

  pnr::SpatialCandidateScratch tagNoGoodCommitScratch;
  requireSuccess(tagNoGoodCommitScratch.prepare(*tagConstrainedProblem));
  auto tagNoGoodCommit =
      take(tagConstrainedClone->beginMove(tagNoGoodCommitScratch));
  requireSuccess(tagNoGoodCommit.ripUpWholeRoute(tagNetOrdinal));
  requireSuccess(tagNoGoodCommit.commit());
  if (tagConstrainedClone->runtimeCounterexampleViolation() != 0)
    fail("committed route-local tag removal retained its no-good");
  requireSuccess(tagConstrainedClone->verify());
}

} // namespace loom::test
