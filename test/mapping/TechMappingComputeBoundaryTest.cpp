#include "TechMappingArtifactTestSupport.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "TechMappingCandidateTestSupport.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <system_error>
#include <utility>

namespace loom::test::tech_mapping_artifact {
namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping compute boundary test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-tech-mapping-compute-boundary", path))
      fail("cannot create ArtifactStore directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove test directory: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

} // namespace

void temporalDispatchProjectionFollowsFabricPolicy() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  auto design = loom::test::buildTemporalCapacityFabric(store);
  const auto &fabric = design.roots().front().view();
  if (fabric.peOccurrences().empty() || fabric.fuOccurrences().empty())
    fail("temporal dispatch fixture has no PE or FU occurrence");
  const auto pe = fabric.peOccurrences().front();
  const auto fu = llvm::find_if(fabric.fuOccurrences(), [&](const auto &item) {
    return fabric.parentPeOf(item) == pe;
  });
  if (fu == fabric.fuOccurrences().end() ||
      fabric.peResidentContextCount(pe) != 2)
    fail("temporal dispatch fixture changed its resident domain");

  const std::vector<loom::mapping::SpatialComputeBindingView> bindings = {
      {1, *fu, {pe, 1}, {}},
      {0, *fu, {pe, 0}, {}},
  };
  const auto domains = take(
      loom::mapping::deriveSpatialTemporalPeDispatchDomains(fabric, bindings));
  if (domains.size() != 1 || domains.front().pe != pe ||
      domains.front().allocationUnit != 0 ||
      domains.front().resetPosition != 0 ||
      domains.front().candidates.size() != 2)
    fail("temporal dispatch projection changed its shared service domain");
  for (std::uint32_t position = 0; position != 2; ++position) {
    const auto &candidate = domains.front().candidates[position];
    if (candidate.context != loom::fabric::InstructionContextRef{pe, position} ||
        candidate.fu != *fu || candidate.realization != position)
      fail("temporal dispatch projection followed Mapping insertion order");
  }
}

void temporalIngressServiceAdmission() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeComputeBoundaryContext();
  auto dataflowArtifact = buildComputeFanoutDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto design = loom::test::buildTemporalCapacityFabric(store);
  const auto &fabric = design.roots().front();
  auto mapping = parseTechMapping(
      context, computeFanoutMappingText(dataflow, fabric.view()));
  if (!mapping)
    fail("fanout TechMapping fixture did not parse");
  auto techRoots = mapping->getOps<::mapping::TechOp>();
  auto tech = take(loom::mapping::finalizeTechMapping(
      *techRoots.begin(), dataflow, fabric.view(), store));
  auto constraintModule = parseTechMapping(
      context, spatialConstraintMappingText(dataflow, tech.view(),
                                            fabric.view(), /*clauses=*/""));
  if (!constraintModule)
    fail("fanout Spatial constraint fixture did not parse");
  auto constraintRoots =
      constraintModule->getOps<::mapping::ConstraintsSpatialOp>();
  auto constraints = take(loom::mapping::finalizeSpatialMappingConstraintSet(
      *constraintRoots.begin(), dataflow, tech.view(), fabric.view(), store));
  const auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(
      loom::test::buildSpatialPnrTestResolvedConfig()));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), config, constraints.view()));
  auto candidate = take(loom::pnr::createCanonicalSpatialCandidate(problem));
  loom::pnr::SpatialCandidateScratch scratch;
  requireSuccess(scratch.prepare(*problem));

  std::optional<loom::pnr::PnrIndex> fanoutNet;
  for (auto [ordinal, net] :
       llvm::enumerate(problem->transfers().logicalNets()))
    if (net.sinkCount == 2) {
      if (fanoutNet)
        fail("fanout fixture has more than one multicast logical net");
      fanoutNet = static_cast<loom::pnr::PnrIndex>(ordinal);
    }
  if (!fanoutNet)
    fail("fanout fixture has no two-sink logical net");
  const auto &net = problem->transfers().logicalNets()[*fanoutNet];
  const auto sinkBindings = problem->transfers().logicalNetSinkBindings();
  const auto lhsBinding = sinkBindings[net.sinkOffset];
  const auto rhsBinding = sinkBindings[net.sinkOffset + 1];
  if (lhsBinding.kind !=
          loom::pnr::FrozenSpatialTerminalBindingKind::PortDemand ||
      rhsBinding.kind !=
          loom::pnr::FrozenSpatialTerminalBindingKind::PortDemand)
    fail("fanout sinks are not compute PortDemands");

  const auto &realization =
      problem->realizations().computeRealizations().front();
  std::optional<loom::pnr::SpatialComputeBindingSelection> selectedBinding;
  std::vector<loom::pnr::PnrIndex> conflicting;
  std::vector<loom::pnr::PnrIndex> legal;
  for (loom::pnr::PnrIndex placement = realization.placementOffset;
       placement != realization.placementOffset + realization.placementCount &&
       !selectedBinding;
       ++placement) {
    const auto &placementRecord =
        problem->realizations().computePlacements()[placement];
    const auto domainFor = [&](loom::pnr::PnrIndex demand) -> const auto & {
      const auto &record = problem->ports().portDemands()[demand];
      return problem->ports()
          .placementDomains()[record.placementDomainOffset + placement -
                              realization.placementOffset];
    };
    const auto &lhsDomain = domainFor(lhsBinding.index);
    const auto &rhsDomain = domainFor(rhsBinding.index);
    const auto options = problem->ports().attachmentOptions();
    std::optional<std::pair<loom::pnr::PnrIndex, loom::pnr::PnrIndex>> conflict;
    for (loom::pnr::PnrIndex lhs = lhsDomain.attachmentOptionOffset;
         lhs != lhsDomain.attachmentOptionOffset +
                    lhsDomain.attachmentOptionCount &&
         !conflict;
         ++lhs)
      for (loom::pnr::PnrIndex rhs = rhsDomain.attachmentOptionOffset;
           rhs !=
           rhsDomain.attachmentOptionOffset + rhsDomain.attachmentOptionCount;
           ++rhs)
        if (options[lhs].sharedOperandEnqueueUnit &&
            options[lhs].sharedOperandEnqueueUnit ==
                options[rhs].sharedOperandEnqueueUnit &&
            options[lhs].endpoint == options[rhs].endpoint) {
          conflict = {lhs, rhs};
          break;
        }
    if (!conflict)
      continue;

    std::vector<loom::pnr::PnrIndex> candidateLegal(
        problem->ports().portDemands().size());
    std::vector<loom::pnr::PnrIndex> usedInputEndpoints;
    std::vector<loom::pnr::PnrIndex> usedOutputEndpoints;
    bool complete = true;
    for (auto [demandOrdinal, demand] :
         llvm::enumerate(problem->ports().portDemands())) {
      const auto &domain =
          domainFor(static_cast<loom::pnr::PnrIndex>(demandOrdinal));
      auto &used = std::holds_alternative<dataflow::ActorTokenOperandRef>(
                       demand.terminal)
                       ? usedInputEndpoints
                       : usedOutputEndpoints;
      const auto available = options.slice(domain.attachmentOptionOffset,
                                           domain.attachmentOptionCount);
      const auto found = llvm::find_if(available, [&](const auto &option) {
        return !llvm::is_contained(used, option.endpoint);
      });
      if (found == available.end()) {
        complete = false;
        break;
      }
      candidateLegal[demandOrdinal] =
          domain.attachmentOptionOffset +
          static_cast<loom::pnr::PnrIndex>(found - available.begin());
      used.push_back(found->endpoint);
    }
    if (!complete)
      continue;
    selectedBinding = loom::pnr::SpatialComputeBindingSelection{
        placement, placementRecord.contextOffset};
    legal = std::move(candidateLegal);
    conflicting = legal;
    conflicting[lhsBinding.index] = conflict->first;
    conflicting[rhsBinding.index] = conflict->second;
  }
  if (!selectedBinding)
    fail("fanout fixture has no shared-service placement with legal fallback");

  {
    auto move = take(candidate->beginMove(scratch));
    requireSuccess(move.setComputeBinding(0, selectedBinding->placement,
                                          selectedBinding->instructionContext));
    for (auto [demand, option] : llvm::enumerate(conflicting))
      requireSuccess(move.setPortAttachment(
          static_cast<loom::pnr::PnrIndex>(demand), option));
    auto closed = move.close();
    if (closed)
      fail("same ingress and enqueue unit escaped PnR admission");
    const std::string message = llvm::toString(closed.takeError());
    if (message.find("hard equality or disjoint relation") == std::string::npos)
      fail("same-service rejection lost its typed binding relation");
    move.rollback();
  }
  {
    auto move = take(candidate->beginMove(scratch));
    requireSuccess(move.setComputeBinding(0, selectedBinding->placement,
                                          selectedBinding->instructionContext));
    for (auto [demand, option] : llvm::enumerate(legal))
      requireSuccess(move.setPortAttachment(
          static_cast<loom::pnr::PnrIndex>(demand), option));
    auto closed = take(move.close());
    if (!closed)
      fail("distinct Temporal ingresses formed a handshake cycle");
    requireSuccess(move.commit());
  }
  requireSuccess(candidate->verify());
}

void computeBoundaryClosure() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeComputeBoundaryContext();

  auto dataflowArtifact = buildComputeBoundaryDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflowView = take(dataflowArtifact.view());

  auto design = loom::test::buildTemporalCapacityFabric(store);
  const auto &fabricRoot = design.roots().front();

  auto complete = parseTechMapping(
      context,
      computeBoundaryMappingText(dataflowView, fabricRoot.view(), true));
  if (!complete)
    fail("complete compute TechMapping fixture did not parse");
  auto completeRoots = complete->getOps<::mapping::TechOp>();
  auto finalized = take(loom::mapping::finalizeTechMapping(
      *completeRoots.begin(), dataflowView, fabricRoot.view(), store));

  auto emptyConstraints = parseTechMapping(
      context, spatialConstraintMappingText(dataflowView, finalized.view(),
                                            fabricRoot.view(), /*clauses=*/""));
  if (!emptyConstraints)
    fail("compute Spatial MappingConstraintSet fixture did not parse");
  auto constraintRoots =
      emptyConstraints->getOps<::mapping::ConstraintsSpatialOp>();
  auto constraints = take(loom::mapping::finalizeSpatialMappingConstraintSet(
      *constraintRoots.begin(), dataflowView, finalized.view(),
      fabricRoot.view(), store));
  const loom::pnr::ResolvedPnrConfigView spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(
          loom::test::buildSpatialPnrTestResolvedConfig()));
  auto frozen = take(loom::pnr::freezeSpatialPnrProblem(
      dataflowView, finalized.view(), fabricRoot.view(), spatialConfig,
      constraints.view()));
  const auto &handshake = frozen->handshake();
  if (handshake.computePlacementFragmentOffsets().size() !=
          frozen->realizations().computePlacements().size() + 1 ||
      handshake.computePlacementFragments().empty())
    fail("compute freeze omitted exact placement handshake fragments");
  loom::test::exerciseHandshakeCandidateRefcounts(frozen);
  loom::test::exerciseCapacityOveruseCandidate(dataflowView, finalized.view(),
                                               fabricRoot.view(), frozen);
  loom::test::exerciseTemporalComputeUseProjection(
      dataflowView, finalized.view(), fabricRoot.view(), frozen);
  loom::test::exerciseCanonicalCandidateInitialization(frozen);

  const auto freezeWithRepairLimits = [&](std::uint64_t regionDecisions,
                                          std::uint64_t solverCalls)
      -> loom::pnr::FrozenSpatialPnrProblemHandle {
    loom::ResolvedConfig config =
        loom::test::buildSpatialPnrTestResolvedConfig();
    config.dse.spatialPnr.search.exactRepair.maxRegionDecisions =
        regionDecisions;
    config.dse.spatialPnr.search.exactRepair.maxSolverCalls = solverCalls;
    const loom::pnr::ResolvedPnrConfigView view =
        take(loom::pnr::projectResolvedSpatialPnrConfigView(config));
    return take(loom::pnr::freezeSpatialPnrProblem(
        dataflowView, finalized.view(), fabricRoot.view(), view,
        constraints.view()));
  };
  loom::test::exerciseCapacityExactRepairNoMutation(
      freezeWithRepairLimits(1, 128),
      loom::pnr::SpatialExactRepairResultKind::RegionTooLarge);
  loom::test::exerciseCapacityExactRepairNoMutation(
      freezeWithRepairLimits(256, 1),
      loom::pnr::SpatialExactRepairResultKind::UnknownBudgetExhausted);
  if (frozen->ports().portDemands().size() != 4 ||
      frozen->ports().graphBoundaries().size() != 4)
    fail("compute freeze omitted actor or graph-boundary demands");
  for (const auto &demand : frozen->ports().portDemands()) {
    if (demand.kind != loom::pnr::FrozenSpatialPortDemandKind::Compute ||
        demand.placementDomainCount == 0)
      fail("compute PortDemand lost its factorized placement domain");
    for (const auto &domain : frozen->ports().placementDomains().slice(
             demand.placementDomainOffset, demand.placementDomainCount)) {
      const auto &placement =
          frozen->realizations().computePlacements()[domain.placement];
      for (const auto &option : frozen->ports().attachmentOptions().slice(
               domain.attachmentOptionOffset, domain.attachmentOptionCount)) {
        if (!option.localTraversal)
          fail("compute PortDemand omitted its exact PE selector traversal");
        const auto &traversal =
            frozen->routing().traversals()[*option.localTraversal].reference;
        const auto *selector =
            std::get_if<loom::fabric::FabricPeSelectorPayload>(
                &traversal.payload);
        if (!selector || selector->owner != placement.parentPe)
          fail("compute PortDemand selected a foreign local traversal");
      }
    }
  }

  auto missing = parseTechMapping(
      context,
      computeBoundaryMappingText(dataflowView, fabricRoot.view(), false));
  if (!missing)
    fail("missing-boundary TechMapping fixture did not parse");
  auto missingRoots = missing->getOps<::mapping::TechOp>();
  if (!rejected(
          loom::mapping::finalizeTechMapping(*missingRoots.begin(), store)))
    fail("compute realization without its FU boundaries was published");
}

} // namespace loom::test::tech_mapping_artifact

int main() {
  loom::test::tech_mapping_artifact::
      temporalDispatchProjectionFollowsFabricPolicy();
  loom::test::tech_mapping_artifact::temporalIngressServiceAdmission();
  loom::test::tech_mapping_artifact::computeBoundaryClosure();
  llvm::outs() << "tech mapping compute boundary tests passed\n";
  return 0;
}
