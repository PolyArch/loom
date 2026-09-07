#include "SimulationExecutionInternal.h"

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::sim::detail {
namespace {

bool isRetired(const ExecutionTerminal &terminal) {
  return std::holds_alternative<RetiredExecution>(terminal);
}

int compareActorRefs(const dataflow::ActorRef &lhs,
                     const dataflow::ActorRef &rhs) {
  const int identity = compareIdentities(lhs.artifact, rhs.artifact);
  if (identity != 0)
    return identity;
  if (lhs.entity.value() < rhs.entity.value())
    return -1;
  if (lhs.entity.value() > rhs.entity.value())
    return 1;
  return 0;
}

std::vector<dataflow::ActorRef>
graphActors(const SpatialExecutionContext &context) {
  std::vector<dataflow::ActorRef> actors;
  for (const dataflow::CanonicalActorView &actor :
       context.dataflowView.actors())
    if (actor.graph == context.launch.graph)
      actors.push_back(actor.ref);
  return actors;
}

llvm::Expected<fabric::FabricArtifactView>
resolveActivityFabric(const SpatialExecutionContext &context) {
  if (context.fabricView)
    return *context.fabricView;
  if (!context.request || !context.artifactStore)
    return llvm::createStringError(std::errc::not_supported,
                                   "simulation execution: Fabric activity "
                                   "requires its exact Fabric context");
  const ArtifactRootReference *subject = nullptr;
  for (const evaluation::CaseRoleBinding &binding :
       context.request->subjectBindings().roleBindings())
    for (const ArtifactRootReference &candidate : binding.subjects) {
      if (candidate.schemaIdentity != fabric::fabricArtifactSchema.identity)
        continue;
      if (subject && *subject != candidate)
        return invalid("simulation execution: Fabric activity has multiple "
                       "bound hardware subjects");
      subject = &candidate;
    }
  if (!subject)
    return invalid("simulation execution: Fabric activity has no bound "
                   "hardware subject");
  auto root = fabric::importEntireFabricRoot(*subject, *context.artifactStore);
  if (!root)
    return root.takeError();
  return root->view();
}

template <typename Ref>
llvm::Error validateKeyOrder(const Ref &current,
                             std::vector<std::uint8_t> &previous) {
  auto bytes = fabric::canonicalFabricBytes(current);
  if (!previous.empty() && bytes <= previous)
    return invalid("simulation execution: Fabric activity table is not "
                   "canonical or contains a duplicate");
  previous = std::move(bytes);
  return llvm::Error::success();
}

bool exceedsWindowPeak(evaluation::ExactRatio integral,
                       evaluation::ExactRatio begin, evaluation::ExactRatio end,
                       std::uint64_t peak) {
  // The comparison can contain four encoded u64 factors. It does not require
  // the transient duration or capacity bound to fit a stored ExactRatio.
  constexpr unsigned width = 4 * std::numeric_limits<std::uint64_t>::digits;
  const auto wide = [](std::uint64_t value) {
    return llvm::APInt(width, value);
  };
  const auto durationNumerator =
      wide(end.numerator()) * wide(begin.denominator()) -
      wide(begin.numerator()) * wide(end.denominator());
  const auto left = wide(integral.numerator()) * wide(end.denominator()) *
                    wide(begin.denominator());
  const auto right =
      durationNumerator * wide(peak) * wide(integral.denominator());
  return left.ugt(right);
}

llvm::Error validateFabricActivity(const ActivitySummary &summary,
                                   const FabricResourcesActivity &activity,
                                   const SpatialProgressObservations &progress,
                                   const fabric::FabricArtifactView &view) {
  if (view.rootKind() != fabric::FabricRootKind::Module)
    return invalid("simulation execution: Fabric activity requires a Module");
  if (activity.useCounts.empty() && activity.resourceOccupancy.empty())
    return invalid(
        "simulation execution: empty activity summary must be omitted");
  const auto &end = summary.window == ActivityWindow::LaunchToGraphRetirement
                        ? *progress.graphRetirementVisible
                        : progress.terminalObserved;
  if (evaluation::compareExactRatio(end.referenceCycle,
                                    progress.launchAccepted.referenceCycle) < 0)
    return invalid(
        "simulation execution: activity window has negative duration");

  std::vector<std::uint8_t> previous;
  for (const FabricUseCountEntry &entry : activity.useCounts) {
    if (llvm::Error error = validateKeyOrder(entry.pattern, previous))
      return error;
    if (llvm::Error error = fabric::validateFabricRef(view, entry.pattern))
      return error;
    if (!llvm::is_contained(view.moduleResourceOwners(),
                            entry.pattern.owner.catalog()))
      return invalid("simulation execution: activity use is not a physical "
                     "resource in the mapped Module");
  }
  previous.clear();
  for (const FabricResourceOccupancyEntry &entry : activity.resourceOccupancy) {
    if (llvm::Error error = validateKeyOrder(entry.resource, previous))
      return error;
    if (llvm::Error error = fabric::validateFabricRef(view, entry.resource))
      return error;
    if (!llvm::is_contained(view.moduleResourceOwners(),
                            entry.resource.owner.catalog()))
      return invalid("simulation execution: activity state is not a physical "
                     "resource in the mapped Module");
    const auto *contract =
        view.resourceContract(entry.resource.owner.catalog());
    if (!contract)
      return invalid("simulation execution: activity state has no contract");
    const auto dimensions = contract->capacityDimensions(
        ::fabric::StateKey(static_cast<std::uint32_t>(entry.resource.ordinal)));
    if (entry.dimensions.empty())
      return invalid("simulation execution: empty resource dimension table");
    std::optional<std::uint32_t> previousDimension;
    for (const FabricCapacityOccupancy &occupancy : entry.dimensions) {
      const auto ordinal = occupancy.dimension.ordinal();
      if (ordinal >= dimensions.size() ||
          (previousDimension && ordinal <= *previousDimension))
        return invalid("simulation execution: resource dimension table is "
                       "outside its canonical capacity inventory");
      previousDimension = ordinal;
      const auto capacity = dimensions[ordinal].capacity.value();
      if (occupancy.peakOccupiedCapacity > capacity)
        return invalid("simulation execution: resource peak exceeds capacity");
      if (exceedsWindowPeak(occupancy.occupiedCapacityReferenceCycles,
                            progress.launchAccepted.referenceCycle,
                            end.referenceCycle, occupancy.peakOccupiedCapacity))
        return invalid("simulation execution: resource integral exceeds "
                       "its observed window peak");
    }
    if (summary.coverage == ActivityCoverage::Complete &&
        entry.dimensions.size() != dimensions.size())
      return invalid("simulation execution: complete capacity dimension "
                     "table is not total");
  }

  if (summary.coverage != ActivityCoverage::Complete)
    return llvm::Error::success();
  std::vector<std::vector<std::uint8_t>> expectedUses, expectedStates;
  for (const auto &owner : view.moduleResourceOwners()) {
    const auto *contract = view.resourceContract(owner);
    if (!contract)
      return invalid("simulation execution: physical owner has no contract");
    for (std::uint32_t pattern = 0; pattern != contract->usePatternCount();
         ++pattern)
      expectedUses.push_back(
          fabric::canonicalFabricBytes(fabric::FabricUsePatternRef{
              fabric::FabricUsePatternOwnerRef(owner), pattern}));
    for (std::uint32_t state = 0; state != contract->stateCount(); ++state)
      if (!contract->capacityDimensions(::fabric::StateKey(state)).empty())
        expectedStates.push_back(
            fabric::canonicalFabricBytes(fabric::FabricResourceStateRef{
                fabric::FabricResourceStateOwnerRef(owner), state}));
  }
  llvm::sort(expectedUses);
  llvm::sort(expectedStates);
  if (expectedUses.size() != activity.useCounts.size() ||
      expectedStates.size() != activity.resourceOccupancy.size())
    return invalid(
        "simulation execution: complete Fabric activity is not total");
  for (auto [index, entry] : llvm::enumerate(activity.useCounts))
    if (fabric::canonicalFabricBytes(entry.pattern) != expectedUses[index])
      return invalid("simulation execution: complete use inventory differs");
  for (auto [index, entry] : llvm::enumerate(activity.resourceOccupancy))
    if (fabric::canonicalFabricBytes(entry.resource) != expectedStates[index])
      return invalid("simulation execution: complete state inventory differs");
  return llvm::Error::success();
}

template <typename Ref>
void encodeReference(WireWriter &writer, const Ref &ref) {
  auto bytes = fabric::canonicalFabricBytes(ref);
  writer.u64(bytes.size());
  writer.bytes(bytes);
}

template <typename Ref>
llvm::Expected<Ref> decodeReference(WireReader &reader) {
  auto size = reader.u64();
  if (!size)
    return size.takeError();
  auto bytes = reader.bytes(*size);
  if (!bytes)
    return bytes.takeError();
  return fabric::decodeFabricRef<Ref>(*bytes);
}

} // namespace

llvm::Error
validateActivitySummaries(llvm::ArrayRef<ActivitySummary> summaries,
                          const ExecutionTerminal &terminal,
                          const SpatialProgressObservations &progress,
                          const SpatialExecutionContext &context) {
  std::optional<fabric::FabricArtifactView> fabricView;
  const std::vector<dataflow::ActorRef> completeActors = graphActors(context);
  for (std::size_t summaryIndex = 0; summaryIndex < summaries.size();
       ++summaryIndex) {
    const ActivitySummary &summary = summaries[summaryIndex];
    if (static_cast<std::uint32_t>(summary.window) >
            static_cast<std::uint32_t>(ActivityWindow::LaunchToTerminal) ||
        static_cast<std::uint32_t>(summary.coverage) >
            static_cast<std::uint32_t>(ActivityCoverage::Partial))
      return invalid(
          "simulation execution: activity summary enum is out of domain");
    const auto key = [](const ActivitySummary &value) {
      return std::make_pair(value.window, value.payload.index());
    };
    if (summaryIndex && key(summary) <= key(summaries[summaryIndex - 1]))
      return invalid("simulation execution: activity summaries are not "
                     "canonical or contain a duplicate");
    if (summary.window == ActivityWindow::LaunchToGraphRetirement &&
        !progress.graphRetirementVisible)
      return invalid("simulation execution: graph-retirement activity window "
                     "has no retirement anchor");
    if (const auto *resources =
            std::get_if<FabricResourcesActivity>(&summary.payload)) {
      if (!fabricView) {
        auto resolved = resolveActivityFabric(context);
        if (!resolved)
          return resolved.takeError();
        fabricView.emplace(std::move(*resolved));
      }
      if (llvm::Error error = validateFabricActivity(summary, *resources,
                                                     progress, *fabricView))
        return error;
      continue;
    }
    const auto &activity = std::get<ActorTransitionsActivity>(summary.payload);
    if (activity.transitions.empty())
      return invalid("simulation execution: empty activity summary must be "
                     "omitted");
    for (std::size_t index = 0; index < activity.transitions.size(); ++index) {
      const ActorTransitionEntry &entry = activity.transitions[index];
      if (index != 0 &&
          compareActorRefs(entry.actor,
                           activity.transitions[index - 1].actor) <= 0)
        return invalid("simulation execution: actor activity table is not "
                       "canonical or contains a duplicate");
      llvm::Expected<dataflow::CanonicalActorView> actor =
          context.dataflowView.resolve(entry.actor);
      if (!actor)
        return actor.takeError();
      if (actor->graph != context.launch.graph)
        return invalid("simulation execution: activity actor is outside the "
                       "rooted launch graph");
      if (entry.counts.committedFirings < entry.counts.retiredFirings)
        return invalid("simulation execution: actor retired count exceeds "
                       "committed count");
      if (isRetired(terminal) &&
          summary.window == ActivityWindow::LaunchToTerminal &&
          entry.counts.committedFirings != entry.counts.retiredFirings)
        return invalid("simulation execution: Retired terminal activity has "
                       "unretired actor firings");
    }
    if (summary.coverage == ActivityCoverage::Complete) {
      if (activity.transitions.size() != completeActors.size())
        return invalid("simulation execution: complete actor activity table "
                       "is not total");
      for (std::size_t index = 0; index < completeActors.size(); ++index)
        if (compareActorRefs(activity.transitions[index].actor,
                             completeActors[index]) != 0)
          return invalid("simulation execution: complete actor activity table "
                         "does not match the rooted graph inventory");
    }
  }
  return llvm::Error::success();
}

void encodeActivitySummaries(WireWriter &writer,
                             llvm::ArrayRef<ActivitySummary> summaries) {
  writer.u64(summaries.size());
  for (const ActivitySummary &summary : summaries) {
    writer.u32(static_cast<std::uint32_t>(summary.window));
    writer.u32(static_cast<std::uint32_t>(summary.coverage));
    writer.u32(static_cast<std::uint32_t>(summary.payload.index()));
    if (const auto *activity =
            std::get_if<ActorTransitionsActivity>(&summary.payload)) {
      writer.u64(activity->transitions.size());
      for (const ActorTransitionEntry &entry : activity->transitions) {
        writer.identity(entry.actor.artifact);
        writer.u64(entry.actor.entity.value());
        writer.u64(entry.counts.committedFirings);
        writer.u64(entry.counts.retiredFirings);
      }
      continue;
    }
    const auto &activity = std::get<FabricResourcesActivity>(summary.payload);
    writer.u64(activity.useCounts.size());
    for (const FabricUseCountEntry &entry : activity.useCounts) {
      encodeReference(writer, entry.pattern);
      writer.u64(entry.activations);
    }
    writer.u64(activity.resourceOccupancy.size());
    for (const FabricResourceOccupancyEntry &entry :
         activity.resourceOccupancy) {
      encodeReference(writer, entry.resource);
      writer.u64(entry.dimensions.size());
      for (const FabricCapacityOccupancy &occupancy : entry.dimensions) {
        writer.u32(occupancy.dimension.ordinal());
        writer.u64(occupancy.occupiedCapacityReferenceCycles.numerator());
        writer.u64(occupancy.occupiedCapacityReferenceCycles.denominator());
        writer.u64(occupancy.peakOccupiedCapacity);
      }
    }
  }
}

llvm::Expected<std::vector<ActivitySummary>>
decodeActivitySummaries(WireReader &reader) {
  auto count = reader.u64();
  if (!count)
    return count.takeError();
  if (llvm::Error error = reader.guardCount(*count, 20))
    return std::move(error);
  std::vector<ActivitySummary> summaries;
  summaries.reserve(*count);
  for (std::uint64_t index = 0; index != *count; ++index) {
    auto window = reader.u32();
    if (!window)
      return window.takeError();
    auto coverage = reader.u32();
    if (!coverage)
      return coverage.takeError();
    auto payload = reader.u32();
    if (!payload)
      return payload.takeError();
    ActivitySummary summary{static_cast<ActivityWindow>(*window),
                            static_cast<ActivityCoverage>(*coverage),
                            {}};
    if (*payload == 0) {
      auto transitionCount = reader.u64();
      if (!transitionCount)
        return transitionCount.takeError();
      if (llvm::Error error = reader.guardCount(*transitionCount, 56))
        return std::move(error);
      auto &activity = std::get<ActorTransitionsActivity>(summary.payload);
      activity.transitions.reserve(*transitionCount);
      for (std::uint64_t entry = 0; entry != *transitionCount; ++entry) {
        auto artifact = reader.identity();
        if (!artifact)
          return artifact.takeError();
        auto entity = reader.u64();
        if (!entity)
          return entity.takeError();
        auto committed = reader.u64();
        if (!committed)
          return committed.takeError();
        auto retired = reader.u64();
        if (!retired)
          return retired.takeError();
        activity.transitions.push_back(
            {dataflow::ActorRef{*artifact, dataflow::ActorId(*entity)},
             {*committed, *retired}});
      }
    } else if (*payload == 1) {
      FabricResourcesActivity activity;
      auto useCount = reader.u64();
      if (!useCount)
        return useCount.takeError();
      if (llvm::Error error = reader.guardCount(*useCount, 16))
        return std::move(error);
      activity.useCounts.reserve(*useCount);
      for (std::uint64_t index = 0; index != *useCount; ++index) {
        auto pattern = decodeReference<fabric::FabricUsePatternRef>(reader);
        if (!pattern)
          return pattern.takeError();
        auto activations = reader.u64();
        if (!activations)
          return activations.takeError();
        activity.useCounts.push_back({std::move(*pattern), *activations});
      }
      auto stateCount = reader.u64();
      if (!stateCount)
        return stateCount.takeError();
      if (llvm::Error error = reader.guardCount(*stateCount, 16))
        return std::move(error);
      activity.resourceOccupancy.reserve(*stateCount);
      for (std::uint64_t index = 0; index != *stateCount; ++index) {
        auto resource = decodeReference<fabric::FabricResourceStateRef>(reader);
        if (!resource)
          return resource.takeError();
        auto dimensionCount = reader.u64();
        if (!dimensionCount)
          return dimensionCount.takeError();
        if (llvm::Error error = reader.guardCount(*dimensionCount, 28))
          return std::move(error);
        FabricResourceOccupancyEntry state{std::move(*resource), {}};
        state.dimensions.reserve(*dimensionCount);
        for (std::uint64_t index = 0; index != *dimensionCount; ++index) {
          auto dimension = reader.u32();
          if (!dimension)
            return dimension.takeError();
          auto numerator = reader.u64();
          if (!numerator)
            return numerator.takeError();
          auto denominator = reader.u64();
          if (!denominator)
            return denominator.takeError();
          auto ratio = evaluation::ExactRatio::get(*numerator, *denominator);
          if (!ratio)
            return ratio.takeError();
          if (ratio->numerator() != *numerator ||
              ratio->denominator() != *denominator)
            return invalid(
                "simulation execution: activity ratio is not canonical");
          auto peak = reader.u64();
          if (!peak)
            return peak.takeError();
          state.dimensions.push_back(
              {::fabric::CapacityDimensionKey(*dimension), *ratio, *peak});
        }
        activity.resourceOccupancy.push_back(std::move(state));
      }
      summary.payload = std::move(activity);
    } else {
      return llvm::createStringError(
          std::errc::not_supported,
          "simulation execution activity owner is unavailable");
    }
    summaries.push_back(std::move(summary));
  }
  return summaries;
}

} // namespace loom::sim::detail
