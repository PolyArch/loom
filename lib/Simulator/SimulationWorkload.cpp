//===- SimulationWorkload.cpp - spatial workload artifact ----------------===//
//
// Schema-1.0 Spatial SimulationWorkload: semantic validation against the
// exact Dataflow owner view, the one strict canonical encoder/decoder, and
// failure-atomic finalization/import framed by the Common finalizer.
//
//===----------------------------------------------------------------------===//

#include "SimulationWireInternal.h"

#include "Common/ArtifactFinalizer.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"

#include <algorithm>
#include <utility>

using namespace mlir;

namespace loom::sim {
namespace {

constexpr std::uint32_t kSpatialRootTag = 0;

//===----------------------------------------------------------------------===//
// Semantic validation
//===----------------------------------------------------------------------===//

llvm::Error checkAscendingUnique(llvm::ArrayRef<std::uint64_t> ordinals,
                                 std::uint64_t ownerCount,
                                 const llvm::Twine &what) {
  for (std::size_t index = 0; index < ordinals.size(); ++index) {
    if (index > 0 && ordinals[index] <= ordinals[index - 1])
      return detail::invalid(what + ": ordinals are not sorted or contain a "
                                    "duplicate");
    if (ordinals[index] >= ownerCount)
      return detail::invalid(what + ": ordinal out of range");
  }
  return llvm::Error::success();
}

bool containsRoot(llvm::ArrayRef<dataflow::LogicalMemoryRootRef> sortedRoots,
                  const dataflow::LogicalMemoryRootRef &root) {
  return std::binary_search(sortedRoots.begin(), sortedRoots.end(), root,
                            [](const auto &lhs, const auto &rhs) {
                              return detail::compareRootKeys(lhs, rhs) < 0;
                            });
}

llvm::Error
validateObservableContract(const SpatialSimulationWorkload &workload,
                           const detail::ResolvedLaunchContext &context,
                           const dataflow::CanonicalDataflowProgramView &view) {
  const SpatialObservableContract &contract = workload.observableContract;
  if (llvm::Error error =
          checkAscendingUnique(contract.valueResults, context.numValueResults,
                               "simulation workload: value results"))
    return error;
  if (llvm::Error error =
          checkAscendingUnique(contract.streamOutputs, context.numStreamOutputs,
                               "simulation workload: stream outputs"))
    return error;
  for (std::size_t index = 0; index < contract.memories.size(); ++index) {
    const SpatialMemoryObservable &observable = contract.memories[index];
    if (static_cast<std::uint32_t>(observable.form) >
        static_cast<std::uint32_t>(MemoryObservationForm::DiffFromRuntimeInput))
      return detail::invalid(
          "simulation workload: memory observation form is out of domain");
    if (index > 0 &&
        detail::compareObservableTargets(
            observable.target, contract.memories[index - 1].target) <= 0)
      return detail::invalid("simulation workload: memory observables are "
                             "not sorted or contain a duplicate");
    if (const auto *rootOrView =
            std::get_if<dataflow::LogicalMemoryRootOrViewRef>(
                &observable.target)) {
      dataflow::LogicalMemoryRootRef root =
          std::holds_alternative<dataflow::LogicalMemoryRootRef>(*rootOrView)
              ? std::get<dataflow::LogicalMemoryRootRef>(*rootOrView)
              : std::get<dataflow::LogicalMemoryViewRef>(*rootOrView).root;
      llvm::Expected<dataflow::CanonicalLogicalMemoryRootView> resolved =
          view.resolve(root);
      if (!resolved)
        return resolved.takeError();
      if (const auto *viewRef =
              std::get_if<dataflow::LogicalMemoryViewRef>(rootOrView)) {
        llvm::Expected<llvm::ArrayRef<dataflow::LogicalMemoryViewRef>> views =
            view.views(viewRef->root);
        if (!views)
          return views.takeError();
        if (viewRef->viewOrdinal >= views->size())
          return detail::invalid(
              "simulation workload: memory view ordinal out of range");
      }
      if (!containsRoot(context.observableRoots, root))
        return detail::invalid("simulation workload: observable memory root "
                               "is not reachable from the rooted launch");
      continue;
    }
    llvm::Expected<dataflow::LogicalMemoryRootOrViewRef> exposure =
        view.resolveExposure(dataflow::MemoryExposureRef{
            workload.launchRef,
            std::get<MemoryExposureTarget>(observable.target)
                .memoryResultOrdinal});
    if (!exposure)
      return exposure.takeError();
  }
  return llvm::Error::success();
}

} // namespace

namespace detail {

llvm::Error
validateSpatialWorkload(const SpatialSimulationWorkload &workload,
                        const ResolvedLaunchContext &context,
                        const dataflow::CanonicalDataflowProgramView &view) {
  // Dense coordinates: exactly the root thread domain rank, each inside any
  // statically known grid bound.
  if (workload.denseCoordinates.size() != context.threadRank)
    return invalid("simulation workload: dense coordinate count does not "
                   "equal the root thread domain rank");
  dataflow::ThreadLaunchOp rootLaunchOp = context.rootLaunchOp;
  mlir::ValueRange gridBounds = rootLaunchOp.getGridUpperBounds();
  for (std::size_t dimension = 0; dimension < workload.denseCoordinates.size();
       ++dimension) {
    llvm::APInt bound;
    if (matchPattern(gridBounds[dimension], m_ConstantInt(&bound)) &&
        workload.denseCoordinates[dimension] >= bound.getLimitedValue())
      return invalid("simulation workload: dense coordinate outside the "
                     "static grid bound");
  }

  // Total Fixed/Runtime classification with exact lane states.
  if (workload.valueInputPlan.size() != context.numValueInputs)
    return invalid("simulation workload: value-input plan is not total over "
                   "the graph value inputs");
  for (std::uint64_t ordinal = 0; ordinal < context.numValueInputs; ++ordinal) {
    const auto *fixed =
        std::get_if<CanonicalValueSequence>(&workload.valueInputPlan[ordinal]);
    if (!fixed)
      continue;
    if (fixed->tokenCount != 1)
      return invalid(
          "simulation workload: a fixed value input holds exactly one token");
    if (llvm::Error error =
            validateValueSequence(*fixed, context.valueInputShapes[ordinal],
                                  "simulation workload: fixed value input"))
      return error;
  }

  return validateObservableContract(workload, context, view);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Canonical encoding
//===----------------------------------------------------------------------===//

namespace {

template <typename EntityId>
void encodeEntityRef(detail::WireWriter &writer,
                     const ::loom::ArtifactReference<EntityId> &reference) {
  writer.identity(reference.artifact);
  writer.u64(reference.entity.value());
}

llvm::Expected<dataflow::RootThreadLaunchRef>
decodeRootThreadLaunchRef(detail::WireReader &reader) {
  llvm::Expected<::loom::ArtifactIdentity> artifact = reader.identity();
  if (!artifact)
    return artifact.takeError();
  llvm::Expected<std::uint64_t> entity = reader.u64();
  if (!entity)
    return entity.takeError();
  return dataflow::RootThreadLaunchRef{*artifact,
                                       dataflow::RootThreadLaunchId(*entity)};
}

llvm::Expected<dataflow::StaticGraphLaunchRef>
decodeStaticGraphLaunchRef(detail::WireReader &reader) {
  llvm::Expected<::loom::ArtifactIdentity> artifact = reader.identity();
  if (!artifact)
    return artifact.takeError();
  llvm::Expected<std::uint64_t> entity = reader.u64();
  if (!entity)
    return entity.takeError();
  return dataflow::StaticGraphLaunchRef{*artifact,
                                        dataflow::StaticGraphLaunchId(*entity)};
}

llvm::Expected<dataflow::LogicalMemoryRootRef>
decodeLogicalMemoryRootRef(detail::WireReader &reader) {
  llvm::Expected<::loom::ArtifactIdentity> artifact = reader.identity();
  if (!artifact)
    return artifact.takeError();
  llvm::Expected<std::uint64_t> entity = reader.u64();
  if (!entity)
    return entity.takeError();
  return dataflow::LogicalMemoryRootRef{*artifact,
                                        dataflow::LogicalMemoryRootId(*entity)};
}

void encodeRootOrView(detail::WireWriter &writer,
                      const dataflow::LogicalMemoryRootOrViewRef &reference) {
  if (const auto *root =
          std::get_if<dataflow::LogicalMemoryRootRef>(&reference)) {
    writer.u32(0);
    encodeEntityRef(writer, *root);
    return;
  }
  const auto &viewRef = std::get<dataflow::LogicalMemoryViewRef>(reference);
  writer.u32(1);
  encodeEntityRef(writer, viewRef.root);
  writer.u64(viewRef.viewOrdinal);
}

llvm::Expected<dataflow::LogicalMemoryRootOrViewRef>
decodeRootOrView(detail::WireReader &reader) {
  llvm::Expected<std::uint32_t> tag = reader.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0) {
    llvm::Expected<dataflow::LogicalMemoryRootRef> root =
        decodeLogicalMemoryRootRef(reader);
    if (!root)
      return root.takeError();
    return dataflow::LogicalMemoryRootOrViewRef{*root};
  }
  if (*tag != 1)
    return detail::invalid(
        "simulation wire: unknown root-or-view discriminant");
  llvm::Expected<dataflow::LogicalMemoryRootRef> root =
      decodeLogicalMemoryRootRef(reader);
  if (!root)
    return root.takeError();
  llvm::Expected<std::uint64_t> viewOrdinal = reader.u64();
  if (!viewOrdinal)
    return viewOrdinal.takeError();
  return dataflow::LogicalMemoryRootOrViewRef{
      dataflow::LogicalMemoryViewRef{*root, *viewOrdinal}};
}

void encodeObservableTarget(detail::WireWriter &writer,
                            const SpatialMemoryObservableTarget &target) {
  if (const auto *rootOrView =
          std::get_if<dataflow::LogicalMemoryRootOrViewRef>(&target)) {
    writer.u32(0);
    encodeRootOrView(writer, *rootOrView);
    return;
  }
  writer.u32(1);
  writer.u64(std::get<MemoryExposureTarget>(target).memoryResultOrdinal);
}

llvm::Expected<SpatialMemoryObservableTarget>
decodeObservableTarget(detail::WireReader &reader) {
  llvm::Expected<std::uint32_t> tag = reader.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0) {
    llvm::Expected<dataflow::LogicalMemoryRootOrViewRef> rootOrView =
        decodeRootOrView(reader);
    if (!rootOrView)
      return rootOrView.takeError();
    return SpatialMemoryObservableTarget{*rootOrView};
  }
  if (*tag != 1)
    return detail::invalid(
        "simulation wire: unknown observable-target discriminant");
  llvm::Expected<std::uint64_t> ordinal = reader.u64();
  if (!ordinal)
    return ordinal.takeError();
  return SpatialMemoryObservableTarget{MemoryExposureTarget{*ordinal}};
}

std::vector<std::uint8_t>
encodeSpatialWorkload(const SpatialSimulationWorkload &workload) {
  detail::WireWriter writer;
  writer.u32(kSpatialRootTag);
  encodeEntityRef(writer, workload.launchRef.rootThreadLaunch);
  encodeEntityRef(writer, workload.launchRef.staticGraphLaunch);
  writer.u64(workload.denseCoordinates.size());
  for (std::uint64_t coordinate : workload.denseCoordinates)
    writer.u64(coordinate);
  writer.u64(workload.valueInputPlan.size());
  for (std::uint64_t ordinal = 0; ordinal < workload.valueInputPlan.size();
       ++ordinal) {
    writer.u64(ordinal);
    const SpatialValueInputSource &source = workload.valueInputPlan[ordinal];
    if (const auto *fixed = std::get_if<CanonicalValueSequence>(&source)) {
      writer.u32(0);
      detail::encodeValueSequence(writer, *fixed);
    } else {
      writer.u32(1);
    }
  }
  const SpatialObservableContract &contract = workload.observableContract;
  writer.u64(contract.valueResults.size());
  for (std::uint64_t ordinal : contract.valueResults)
    writer.u64(ordinal);
  writer.u64(contract.streamOutputs.size());
  for (std::uint64_t ordinal : contract.streamOutputs)
    writer.u64(ordinal);
  writer.u64(contract.memories.size());
  for (const SpatialMemoryObservable &observable : contract.memories) {
    encodeObservableTarget(writer, observable.target);
    writer.u32(static_cast<std::uint32_t>(observable.form));
  }
  return writer.take();
}

llvm::Expected<std::vector<std::uint64_t>>
decodeOrdinalSet(detail::WireReader &reader, const llvm::Twine &what) {
  llvm::Expected<std::uint64_t> count = reader.u64();
  if (!count)
    return count.takeError();
  if (llvm::Error error = reader.guardCount(*count, 8))
    return std::move(error);
  std::vector<std::uint64_t> ordinals;
  ordinals.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    llvm::Expected<std::uint64_t> ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (index > 0 && *ordinal <= ordinals.back())
      return detail::invalid(what + ": ordinals are not sorted or contain a "
                                    "duplicate");
    ordinals.push_back(*ordinal);
  }
  return ordinals;
}

// The decoded model plus the one launch context resolved while parsing; the
// caller reuses the context for semantic validation instead of resolving the
// same launch again.
struct DecodedSpatialWorkload {
  SpatialSimulationWorkload workload;
  detail::ResolvedLaunchContext context;
};

llvm::Expected<DecodedSpatialWorkload>
decodeSpatialWorkload(llvm::ArrayRef<std::uint8_t> bytes,
                      const dataflow::CanonicalDataflowProgramView &view) {
  detail::WireReader reader(bytes);
  llvm::Expected<std::uint32_t> root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root == 1)
    return detail::invalid(
        "simulation workload: the System root is fail-closed in schema 1.0");
  if (*root != kSpatialRootTag)
    return detail::invalid("simulation workload: unknown root discriminant");

  llvm::Expected<dataflow::RootThreadLaunchRef> rootLaunch =
      decodeRootThreadLaunchRef(reader);
  if (!rootLaunch)
    return rootLaunch.takeError();
  llvm::Expected<dataflow::StaticGraphLaunchRef> staticLaunch =
      decodeStaticGraphLaunchRef(reader);
  if (!staticLaunch)
    return staticLaunch.takeError();
  SpatialSimulationWorkload workload{
      dataflow::RootedGraphLaunchRef{*rootLaunch, *staticLaunch}};

  // The remaining fields decode under the exact recovered launch context.
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(view, workload.launchRef);
  if (!context)
    return context.takeError();

  llvm::Expected<std::uint64_t> coordinateCount = reader.u64();
  if (!coordinateCount)
    return coordinateCount.takeError();
  if (llvm::Error error = reader.guardCount(*coordinateCount, 8))
    return std::move(error);
  workload.denseCoordinates.reserve(*coordinateCount);
  for (std::uint64_t index = 0; index < *coordinateCount; ++index) {
    llvm::Expected<std::uint64_t> coordinate = reader.u64();
    if (!coordinate)
      return coordinate.takeError();
    workload.denseCoordinates.push_back(*coordinate);
  }

  llvm::Expected<std::uint64_t> planCount = reader.u64();
  if (!planCount)
    return planCount.takeError();
  if (llvm::Error error = reader.guardCount(*planCount, 12))
    return std::move(error);
  workload.valueInputPlan.reserve(*planCount);
  for (std::uint64_t index = 0; index < *planCount; ++index) {
    llvm::Expected<std::uint64_t> ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (*ordinal != index)
      return detail::invalid("simulation workload: value-input plan keys are "
                             "not the dense sorted ordinals");
    if (*ordinal >= context->numValueInputs)
      return detail::invalid(
          "simulation workload: value-input ordinal out of range");
    llvm::Expected<std::uint32_t> tag = reader.u32();
    if (!tag)
      return tag.takeError();
    if (*tag == 0) {
      llvm::Expected<CanonicalValueSequence> fixed =
          detail::decodeValueSequence(reader,
                                      context->valueInputShapes[*ordinal]);
      if (!fixed)
        return fixed.takeError();
      workload.valueInputPlan.emplace_back(std::move(*fixed));
      continue;
    }
    if (*tag != 1)
      return detail::invalid(
          "simulation workload: unknown value-input source discriminant");
    workload.valueInputPlan.emplace_back(RuntimeValueInput{});
  }

  llvm::Expected<std::vector<std::uint64_t>> valueResults =
      decodeOrdinalSet(reader, "simulation workload: value results");
  if (!valueResults)
    return valueResults.takeError();
  workload.observableContract.valueResults = std::move(*valueResults);
  llvm::Expected<std::vector<std::uint64_t>> streamOutputs =
      decodeOrdinalSet(reader, "simulation workload: stream outputs");
  if (!streamOutputs)
    return streamOutputs.takeError();
  workload.observableContract.streamOutputs = std::move(*streamOutputs);

  llvm::Expected<std::uint64_t> memoryCount = reader.u64();
  if (!memoryCount)
    return memoryCount.takeError();
  if (llvm::Error error = reader.guardCount(*memoryCount, 8))
    return std::move(error);
  workload.observableContract.memories.reserve(*memoryCount);
  for (std::uint64_t index = 0; index < *memoryCount; ++index) {
    llvm::Expected<SpatialMemoryObservableTarget> target =
        decodeObservableTarget(reader);
    if (!target)
      return target.takeError();
    llvm::Expected<std::uint32_t> form = reader.u32();
    if (!form)
      return form.takeError();
    if (*form >
        static_cast<std::uint32_t>(MemoryObservationForm::DiffFromRuntimeInput))
      return detail::invalid(
          "simulation workload: unknown memory observation form");
    if (index > 0 &&
        detail::compareObservableTargets(
            *target, workload.observableContract.memories.back().target) <= 0)
      return detail::invalid("simulation workload: memory observables are "
                             "not sorted or contain a duplicate");
    workload.observableContract.memories.push_back(SpatialMemoryObservable{
        std::move(*target), static_cast<MemoryObservationForm>(*form)});
  }

  if (!reader.atEnd())
    return detail::invalid("simulation workload: trailing bytes");
  return DecodedSpatialWorkload{std::move(workload), std::move(*context)};
}

} // namespace

//===----------------------------------------------------------------------===//
// Finalization and import
//===----------------------------------------------------------------------===//

llvm::Expected<CanonicalSimulationWorkload>
finalizeSimulationWorkload(const SpatialSimulationWorkload &workload,
                           const dataflow::CanonicalDataflowProgramView &view) {
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(view, workload.launchRef);
  if (!context)
    return context.takeError();
  if (llvm::Error error =
          detail::validateSpatialWorkload(workload, *context, view))
    return std::move(error);
  ::loom::CanonicalSemanticBytes bytes(encodeSpatialWorkload(workload));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationWorkloadSchema, bytes);
  return CanonicalSimulationWorkload(identity, workload, std::move(bytes));
}

llvm::Expected<CanonicalSimulationWorkload>
importSimulationWorkload(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                         const dataflow::CanonicalDataflowProgramView &view,
                         const ::loom::ArtifactIdentity &expectedIdentity) {
  llvm::Expected<DecodedSpatialWorkload> decoded =
      decodeSpatialWorkload(canonicalBytes, view);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = detail::validateSpatialWorkload(
          decoded->workload, decoded->context, view))
    return std::move(error);
  const std::vector<std::uint8_t> reencoded =
      encodeSpatialWorkload(decoded->workload);
  if (!llvm::ArrayRef<std::uint8_t>(reencoded).equals(canonicalBytes))
    return detail::invalid(
        "simulation workload: noncanonical bytes do not re-encode exactly");
  ::loom::CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(canonicalBytes.begin(), canonicalBytes.end()));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationWorkloadSchema, bytes);
  if (identity != expectedIdentity)
    return detail::invalid(
        "simulation workload: identity does not match the expected artifact");
  return CanonicalSimulationWorkload(identity, std::move(decoded->workload),
                                     std::move(bytes));
}

} // namespace loom::sim
