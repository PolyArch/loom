#include "RootCompleteSpatialPnrExecutionTest.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "PnR/SpatialPnrGenerator.h"
#include "RootCompleteSpatialFeedbackTestSupport.h"
#include "RootCompleteSpatialPnrTestSupport.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <utility>
#include <variant>

namespace loom::test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "root-complete Spatial PnR execution failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
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

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-root-complete-spatial-pnr", path_);
    if (error)
      fail("cannot create ArtifactStore directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

} // namespace

void candidateWorkerCountPreservesFormalResult() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto fixture = buildRootCompleteSpatialPnrFixture(context, store, blobs);
  const auto &dataflow = fixture.dataflow.view();
  auto tech = take(
      loom::mapping::importTechMapping(fixture.techMappingReference, store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fixture.fabric.view(), store));
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 4;
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fixture.fabric.view()));
  const auto run = [&](std::uint32_t workerCount,
                       loom::ExecutionResourceBudget executionBudget,
                       std::optional<std::uint64_t> maximumPublications =
                           std::nullopt) {
    loom::pnr::SpatialPnrGenerationInputs inputs{
        dataflow,       tech.view(), fixture.fabric.view(),
        physicalTiming, config,      constraints.view(),
        store,          workerCount};
    inputs.executionBudget = executionBudget;
    inputs.maximumCandidatePublications = maximumPublications;
    return loom::pnr::generateSpatialMappings(inputs);
  };
  const auto single = run(4, {1, std::nullopt});
  const auto memoryConstrained = run(4, {4, 1});
  const auto parallel = run(4, {3, 0});
  const auto publicationBounded = run(4, {3, 0}, 1);
  if (single.index() != memoryConstrained.index() ||
      single.index() != parallel.index() ||
      single.index() != publicationBounded.index())
    fail("candidate worker count changed the Spatial PnR outcome kind");
  const auto *singleGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&single);
  const auto *memoryConstrainedGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&memoryConstrained);
  const auto *parallelGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&parallel);
  const auto *publicationBoundedGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&publicationBounded);
  if (!singleGenerated || !memoryConstrainedGenerated || !parallelGenerated ||
      !publicationBoundedGenerated)
    fail("worker-invariance fixture did not produce Spatial Mappings");
  if (singleGenerated->termination != memoryConstrainedGenerated->termination ||
      singleGenerated->termination != parallelGenerated->termination ||
      !(singleGenerated->accounting ==
        memoryConstrainedGenerated->accounting) ||
      !(singleGenerated->accounting == parallelGenerated->accounting) ||
      singleGenerated->candidates != memoryConstrainedGenerated->candidates ||
      singleGenerated->candidates != parallelGenerated->candidates)
    fail("candidate worker count changed formal Spatial PnR output or work");
  loom::pnr::SpatialPnrGenerationAccounting publicationWork =
      publicationBoundedGenerated->accounting;
  publicationWork.finalizedRestarts =
      parallelGenerated->accounting.finalizedRestarts;
  publicationWork.publicationSlots =
      parallelGenerated->accounting.publicationSlots;
  if (publicationBoundedGenerated->termination !=
          parallelGenerated->termination ||
      publicationBoundedGenerated->accounting.seedAttemptSlots != 4 ||
      publicationBoundedGenerated->accounting.publicationSlots != 1 ||
      publicationBoundedGenerated->accounting.finalizedRestarts != 1 ||
      publicationBoundedGenerated->candidates.size() > 1 ||
      !(publicationWork == parallelGenerated->accounting))
    fail("publication demand serialized, truncated, or reclassified exhaustive "
         "Spatial work");
  llvm::outs() << "spatial_worker_budget constrained_workers=1"
               << " parallel_workers=3 exhaustive_restart_slots="
               << publicationBoundedGenerated->accounting.seedAttemptSlots
               << " bounded_publications="
               << publicationBoundedGenerated->accounting.publicationSlots
               << '\n';
}

void sharedFrontierWorkersPreserveFormalResult() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto dataflow = loom::test::buildRootCompleteSpatialDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto fabric = loom::test::buildAlternativeTechSpatialCore(store);
  const auto physicalTiming =
      normalizedTimingProfile(fabric.reference(), store);
  const auto techMappings = generateTechMappingSet(
      dataflowReference, fabric.reference(), store, blobs);
  if (techMappings.size() < 2)
    fail("fixture did not expose alternative TechMappings for one graph");

  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          techMappings, fabric.reference(), physicalTiming));
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 4;
  const auto config =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  const auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  const auto run = [&](loom::ExecutionResourceBudget budget,
                       llvm::ArrayRef<loom::dse::CandidateGeneratorOutputDemand>
                           outputDemands = {}) {
    const loom::dse::CandidateGeneratorInvocationView invocation(
        {}, outputDemands, budget);
    return take(loom::dse::invokeCandidateGenerator(inputs, binding, store,
                                                    blobs, invocation));
  };
  const auto serial = run({1, std::nullopt});
  const auto shared = run({3, std::nullopt});
  const auto calibrated = run({3, 1});
  const auto requireCompleted = [&](const auto &result)
      -> const loom::dse::CompletedCandidateGeneratorResult & {
    const auto *value =
        std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
            &result.outcome);
    if (!value || value->outputBindings.size() != 1 ||
        value->outputBindings.front().artifacts.empty() || !result.dispatched)
      fail("frontier resource guard did not execute a complete real provider");
    requireSpatialWorkSummary(result.workSummary, true);
    if (result.workSummary.front().planned != techMappings.size() * 4 ||
        result.workSummary.front().consumed != techMappings.size() * 4)
      fail("frontier scheduling skipped or repeated a configured seed");
    return *value;
  };
  const auto &serialCompleted = requireCompleted(serial);
  const auto &serialArtifacts =
      serialCompleted.outputBindings.front().artifacts;
  for (const auto *result : {&shared, &calibrated}) {
    const auto &value = requireCompleted(*result);
    if (result->workSummary != serial.workSummary ||
        result->ownerFeedback != serial.ownerFeedback ||
        value.outputBindings.front().slot !=
            serialCompleted.outputBindings.front().slot ||
        value.outputBindings.front().artifacts != serialArtifacts ||
        value.lineageEdges != serialCompleted.lineageEdges)
      fail("shared frontier pool changed canonical outputs, lineage, or full "
           "work");
  }
  const std::array<loom::dse::CandidateGeneratorOutputDemand, 1> demand{
      {{loom::dse::CandidateGeneratorOutputSlotRef(0), 1}}};
  const auto limited = run({3, std::nullopt}, demand);
  const auto &limitedCompleted = requireCompleted(limited);
  if (limited.workSummary != serial.workSummary ||
      limitedCompleted.outputBindings.front().artifacts.size() != 1 ||
      limitedCompleted.outputBindings.front().artifacts.front() !=
          serialArtifacts.front())
    fail("parallel frontier publication demand truncated work or reordered "
         "outputs");
}

void finalizedRestartSurvivesUnfinishedPeer() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  requireSuccess(
      llvm::errorCodeToError(llvm::sys::fs::create_directories(blobPath)));
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto fixture = buildRootCompleteSpatialPnrFixture(context, store, blobs);
  const auto &dataflow = fixture.dataflow.view();
  auto tech = take(
      loom::mapping::importTechMapping(fixture.techMappingReference, store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fixture.fabric.view(), store));
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fixture.fabric.view()));
  auto resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 4;
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));

  // One real restart remains inside its first public execution-control query
  // until an independent peer durably finalizes. The old peer barrier cannot
  // satisfy this dependency; a guard deadline makes that failure terminal.
  struct StopAfterPublication {
    std::string root;
    std::set<std::string> originalObjects;
    std::thread::id caller = std::this_thread::get_id();
    mutable std::mutex mutex;
    mutable std::optional<std::thread::id> heldWorker;
    mutable std::atomic_bool requested{false};
    mutable std::atomic_bool deadlineExpired{false};
    mutable std::atomic_bool publicationObserved{false};

    bool hasPublishedObject() const {
      std::error_code error;
      for (llvm::sys::fs::directory_iterator it(root, error), end;
           !error && it != end; it.increment(error)) {
        const llvm::StringRef name = llvm::sys::path::filename(it->path());
        if (name.size() == 64 && !originalObjects.count(name.str()))
          return true;
      }
      if (error)
        fail("retention guard could not observe its own ArtifactStore");
      return false;
    }
    static bool query(const void *opaque) {
      const auto &self = *static_cast<const StopAfterPublication *>(opaque);
      if (self.requested.load())
        return true;
      const auto current = std::this_thread::get_id();
      if (current == self.caller)
        return false;
      bool held = false;
      {
        std::lock_guard<std::mutex> lock(self.mutex);
        if (!self.heldWorker)
          self.heldWorker = current;
        held = *self.heldWorker == current;
      }
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      do {
        if (self.hasPublishedObject()) {
          self.publicationObserved.store(true);
          self.requested.store(true);
          return true;
        }
        if (!held)
          return false;
        if (std::chrono::steady_clock::now() >= deadline) {
          self.deadlineExpired.store(true);
          self.requested.store(true);
          return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (true);
    }
  } stop;
  stop.root = directory.path().str();
  {
    std::error_code error;
    for (llvm::sys::fs::directory_iterator it(stop.root, error), end;
         !error && it != end; it.increment(error))
      stop.originalObjects.insert(llvm::sys::path::filename(it->path()).str());
    if (error)
      fail("retention guard could not snapshot its own ArtifactStore");
  }
  loom::pnr::SpatialPnrGenerationInputs inputs{
      dataflow,
      tech.view(),
      fixture.fabric.view(),
      physicalTiming,
      config,
      constraints.view(),
      store,
      2,
      loom::ExecutionControlView(&stop, StopAfterPublication::query)};
  inputs.executionBudget = {2, std::nullopt};
  llvm::DefaultThreadPool workers(llvm::heavyweight_hardware_concurrency(2));
  auto outcome = loom::pnr::generateSpatialMappings(inputs, workers);
  if (stop.deadlineExpired || !stop.publicationObserved)
    fail("finalized restart remained behind the peer completion barrier");
  const auto *interrupted =
      std::get_if<loom::pnr::InterruptedSpatialPnrGeneration>(&outcome);
  if (!interrupted || interrupted->candidates.empty() ||
      interrupted->accounting.finalizedRestarts != 1 ||
      interrupted->accounting.publicationSlots != 1 ||
      interrupted->snapshot.frontier.finalizedRestarts != 1 ||
      interrupted->snapshot.closureResidual.retainedCandidates != 1 ||
      !interrupted->snapshot.bestSelectedRank ||
      !interrupted->snapshot.closureResidual.violationValues ||
      llvm::any_of(*interrupted->snapshot.closureResidual.violationValues,
                   [](const auto &value) { return !value || *value != 0; }))
    fail("interruption lost a verified reference or its compact "
         "completed-state summary");
  requireSuccess(loom::pnr::verifySpatialPnrWorkAccounting(
      interrupted->accounting, false));
  loom::ArtifactStore reopened(directory.path());
  for (const auto &reference : interrupted->candidates) {
    const auto finalized =
        take(loom::mapping::importSpatialMapping(reference, reopened));
    const auto &view = finalized.view();
    const auto progress =
        take(loom::mapping::deriveSpatialMappingProgressClosure(
            dataflow, tech.view(), fixture.fabric.view(),
            view.computeBindings(), view.registerFifoTransfers(),
            view.routeTrees(), view.resourceUses(),
            view.physicalTagSegments()));
    if (progress.kind !=
        loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
      fail("retention guard exposed an unproven route as a finalized Mapping");
  }
}

} // namespace loom::test
