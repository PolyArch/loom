#include "TechMappingArtifactTestSupport.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/PnrConfig.h"
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
  loom::test::exerciseCapacityOveruseCandidate(frozen);
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
