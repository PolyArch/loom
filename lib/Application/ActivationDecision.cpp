#include "Application/ActivationDecision.h"

#include "ActivationRepairLineage.h"

#include "Application/Build.h"
#include "ApplicationRuntimeValidationInternal.h"
#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/InvocationManifest.h"
#include "DSE/PreMappingExploration.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Evaluation/Request.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application {
namespace {

llvm::Error reject(ApplicationActivationDecisionErrorReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<ApplicationActivationDecisionError>(reason,
                                                              message.str());
}

llvm::Error malformed(const llvm::Twine &message) {
  return reject(ApplicationActivationDecisionErrorReason::MalformedEncoding,
                message);
}

class Encoder final {
public:
  void u32(std::uint32_t value) {
    bytes_.push_back(static_cast<std::uint8_t>(value >> 24));
    bytes_.push_back(static_cast<std::uint8_t>(value >> 16));
    bytes_.push_back(static_cast<std::uint8_t>(value >> 8));
    bytes_.push_back(static_cast<std::uint8_t>(value));
  }

  void u64(std::uint64_t value) {
    for (unsigned shift = 56; shift != 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
    bytes_.push_back(static_cast<std::uint8_t>(value));
  }

  void fixed(llvm::ArrayRef<std::uint8_t> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  void bytes(llvm::ArrayRef<std::uint8_t> value) {
    u64(value.size());
    fixed(value);
  }

  void text(llvm::StringRef value) {
    bytes(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(value.data()), value.size()));
  }

  void root(const ArtifactRootReference &reference) {
    fixed(encodeArtifactRootReference(reference));
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef field) {
    auto raw = fixed(sizeof(std::uint32_t), field);
    if (!raw)
      return raw.takeError();
    std::uint32_t value = 0;
    for (std::uint8_t byte : *raw)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    auto raw = fixed(sizeof(std::uint64_t), field);
    if (!raw)
      return raw.takeError();
    std::uint64_t value = 0;
    for (std::uint8_t byte : *raw)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> fixed(std::size_t width,
                                                     llvm::StringRef field) {
    if (width > remaining())
      return malformed("truncated " + field);
    llvm::ArrayRef<std::uint8_t> value = bytes_.slice(offset_, width);
    offset_ += width;
    return value;
  }

  llvm::Expected<std::vector<std::uint8_t>> bytes(llvm::StringRef field) {
    auto width = u64((field + " length").str());
    if (!width)
      return width.takeError();
    if (*width > std::numeric_limits<std::size_t>::max() ||
        *width > remaining())
      return malformed(field + " length exceeds the remaining wire");
    auto raw = fixed(static_cast<std::size_t>(*width), field);
    if (!raw)
      return raw.takeError();
    return raw->vec();
  }

  llvm::Expected<std::string> text(llvm::StringRef field) {
    auto raw = bytes(field);
    if (!raw)
      return raw.takeError();
    return std::string(raw->begin(), raw->end());
  }

  llvm::Expected<std::size_t> count(llvm::StringRef field) {
    auto value = u64(field);
    if (!value)
      return value.takeError();
    if (*value > std::numeric_limits<std::size_t>::max() ||
        *value > remaining())
      return malformed(field + " is not representable by the remaining wire");
    return static_cast<std::size_t>(*value);
  }

  llvm::Expected<ArtifactRootReference> root(llvm::StringRef field) {
    auto decoded =
        decodeArtifactRootReferencePrefix(bytes_.drop_front(offset_));
    if (!decoded)
      return malformed(field + ": " + llvm::toString(decoded.takeError()));
    offset_ += decoded->byteCount;
    return std::move(decoded->reference);
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }
  bool atEnd() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

bool rootLaunchLess(dataflow::RootThreadLaunchRef lhs,
                    dataflow::RootThreadLaunchRef rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

void encodeRootLaunch(Encoder &encoder,
                      dataflow::RootThreadLaunchRef reference) {
  encoder.fixed(reference.artifact.bytes());
  encoder.u64(reference.entity.value());
}

llvm::Expected<dataflow::RootThreadLaunchRef>
decodeRootLaunch(Decoder &decoder, llvm::StringRef field) {
  auto identityBytes =
      decoder.fixed(ArtifactIdentity::byteSize, (field + " artifact").str());
  if (!identityBytes)
    return identityBytes.takeError();
  auto identity = ArtifactIdentity::fromBytes(*identityBytes);
  if (!identity)
    return malformed(field + " has an invalid Artifact identity");
  auto entity = decoder.u64((field + " entity").str());
  if (!entity)
    return entity.takeError();
  return dataflow::RootThreadLaunchRef{std::move(*identity),
                                       dataflow::RootThreadLaunchId(*entity)};
}

void encodeRootLaunches(
    Encoder &encoder,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> references) {
  encoder.u64(references.size());
  for (dataflow::RootThreadLaunchRef reference : references)
    encodeRootLaunch(encoder, reference);
}

llvm::Expected<std::vector<dataflow::RootThreadLaunchRef>>
decodeRootLaunches(Decoder &decoder, llvm::StringRef field) {
  auto count = decoder.count((field + " count").str());
  if (!count)
    return count.takeError();
  std::vector<dataflow::RootThreadLaunchRef> result;
  result.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto reference = decodeRootLaunch(decoder, field);
    if (!reference)
      return reference.takeError();
    result.push_back(std::move(*reference));
  }
  return result;
}

void encodeOptionalU64(Encoder &encoder,
                       const std::optional<std::uint64_t> &value) {
  encoder.u32(value ? 1 : 0);
  if (value)
    encoder.u64(*value);
}

llvm::Expected<std::optional<std::uint64_t>>
decodeOptionalU64(Decoder &decoder, llvm::StringRef field) {
  auto present = decoder.u32((field + " presence").str());
  if (!present)
    return present.takeError();
  if (*present > 1)
    return malformed(field + " presence is not boolean");
  if (*present == 0)
    return std::optional<std::uint64_t>{};
  auto value = decoder.u64(field);
  if (!value)
    return value.takeError();
  return std::optional<std::uint64_t>{*value};
}

void encodeScheduleHint(Encoder &encoder,
                        const dse::ResourceTimeScheduleHint &hint) {
  encoder.u64(hint.actions.size());
  for (const dse::ResourceTimeActionDelta &action : hint.actions) {
    encoder.u32(static_cast<std::uint32_t>(action.kind));
    encoder.u32(action.admittedRegion ? 1 : 0);
    if (action.admittedRegion)
      encodeRootLaunch(encoder, *action.admittedRegion);
    encodeOptionalU64(encoder, action.speedupPointOrdinal);
    encoder.u64(action.beforeTimePicoseconds);
    encoder.u64(action.afterTimePicoseconds);
    encodeRootLaunches(encoder, action.completedRegions);
    encodeRootLaunches(encoder, action.tokenReadyProducers);
    encodeRootLaunches(encoder, action.newlyReadyRegions);
  }
  encoder.u64(hint.states.size());
  for (const dse::ResourceTimeHintState &state : hint.states) {
    encoder.u64(state.timePicoseconds);
    encoder.u64(state.active.size());
    for (const dse::ResourceTimeHintAllocation &allocation : state.active) {
      encodeRootLaunch(encoder, allocation.region);
      encoder.u64(allocation.speedupPointOrdinal);
      encoder.u64(allocation.resourceUnits.size());
      for (std::uint64_t units : allocation.resourceUnits)
        encoder.u64(units);
      encoder.u64(allocation.completionTimePicoseconds);
    }
    encodeRootLaunches(encoder, state.ready);
    encodeRootLaunches(encoder, state.completed);
    encoder.u64(state.optimisticMakespanLowerBoundPicoseconds);
  }
  encoder.u64(hint.estimatedMakespanPicoseconds);
  encoder.u64(hint.optimisticMakespanLowerBoundPicoseconds);
  encoder.u64(hint.peakConcurrentRegions);
  encoder.u64(hint.totalAllocatedResourceTime);
  encoder.u32(static_cast<std::uint32_t>(hint.support));
}

llvm::Expected<dse::ResourceTimeScheduleHint>
decodeScheduleHint(Decoder &decoder) {
  dse::ResourceTimeScheduleHint hint;
  auto actionCount = decoder.count("schedule action count");
  if (!actionCount)
    return actionCount.takeError();
  hint.actions.reserve(*actionCount);
  for (std::size_t index = 0; index != *actionCount; ++index) {
    auto kind = decoder.u32("schedule action kind");
    if (!kind)
      return kind.takeError();
    if (*kind >
        static_cast<std::uint32_t>(dse::ResourceTimeActionKind::AdvanceEvent))
      return malformed("schedule action has an unknown kind");
    auto hasRegion = decoder.u32("schedule admitted-region presence");
    if (!hasRegion)
      return hasRegion.takeError();
    if (*hasRegion > 1)
      return malformed("schedule admitted-region presence is not boolean");
    std::optional<dataflow::RootThreadLaunchRef> admitted;
    if (*hasRegion == 1) {
      auto region = decodeRootLaunch(decoder, "schedule admitted region");
      if (!region)
        return region.takeError();
      admitted = std::move(*region);
    }
    auto speedup = decodeOptionalU64(decoder, "schedule speedup point");
    if (!speedup)
      return speedup.takeError();
    auto before = decoder.u64("schedule action before time");
    if (!before)
      return before.takeError();
    auto after = decoder.u64("schedule action after time");
    if (!after)
      return after.takeError();
    auto completed = decodeRootLaunches(decoder, "completed regions");
    if (!completed)
      return completed.takeError();
    auto tokenReady = decodeRootLaunches(decoder, "token-ready producers");
    if (!tokenReady)
      return tokenReady.takeError();
    auto newlyReady = decodeRootLaunches(decoder, "newly-ready regions");
    if (!newlyReady)
      return newlyReady.takeError();
    hint.actions.push_back({static_cast<dse::ResourceTimeActionKind>(*kind),
                            std::move(admitted), std::move(*speedup), *before,
                            *after, std::move(*completed),
                            std::move(*tokenReady), std::move(*newlyReady)});
  }

  auto stateCount = decoder.count("schedule state count");
  if (!stateCount)
    return stateCount.takeError();
  hint.states.reserve(*stateCount);
  for (std::size_t index = 0; index != *stateCount; ++index) {
    auto time = decoder.u64("schedule state time");
    if (!time)
      return time.takeError();
    auto allocationCount = decoder.count("active allocation count");
    if (!allocationCount)
      return allocationCount.takeError();
    std::vector<dse::ResourceTimeHintAllocation> active;
    active.reserve(*allocationCount);
    for (std::size_t allocationIndex = 0; allocationIndex != *allocationCount;
         ++allocationIndex) {
      auto region = decodeRootLaunch(decoder, "active allocation region");
      if (!region)
        return region.takeError();
      auto speedup = decoder.u64("active allocation speedup point");
      if (!speedup)
        return speedup.takeError();
      auto unitCount = decoder.count("active allocation resource-unit count");
      if (!unitCount)
        return unitCount.takeError();
      std::vector<std::uint64_t> units;
      units.reserve(*unitCount);
      for (std::size_t unit = 0; unit != *unitCount; ++unit) {
        auto value = decoder.u64("active allocation resource units");
        if (!value)
          return value.takeError();
        units.push_back(*value);
      }
      auto completion = decoder.u64("active allocation completion time");
      if (!completion)
        return completion.takeError();
      active.push_back(
          {std::move(*region), *speedup, std::move(units), *completion});
    }
    auto ready = decodeRootLaunches(decoder, "ready regions");
    if (!ready)
      return ready.takeError();
    auto completed = decodeRootLaunches(decoder, "completed regions");
    if (!completed)
      return completed.takeError();
    auto lowerBound = decoder.u64("schedule state lower bound");
    if (!lowerBound)
      return lowerBound.takeError();
    hint.states.push_back({*time, std::move(active), std::move(*ready),
                           std::move(*completed), *lowerBound});
  }
  auto makespan = decoder.u64("schedule estimated makespan");
  if (!makespan)
    return makespan.takeError();
  auto lowerBound = decoder.u64("schedule optimistic lower bound");
  if (!lowerBound)
    return lowerBound.takeError();
  auto peak = decoder.u64("schedule peak concurrent regions");
  if (!peak)
    return peak.takeError();
  auto resourceTime = decoder.u64("schedule allocated resource time");
  if (!resourceTime)
    return resourceTime.takeError();
  auto support = decoder.u32("schedule estimate support");
  if (!support)
    return support.takeError();
  if (*support >
      static_cast<std::uint32_t>(dse::ResourceTimeEstimateSupport::Unsupported))
    return malformed("schedule has an unknown estimate support");
  hint.estimatedMakespanPicoseconds = *makespan;
  hint.optimisticMakespanLowerBoundPicoseconds = *lowerBound;
  hint.peakConcurrentRegions = *peak;
  hint.totalAllocatedResourceTime = *resourceTime;
  hint.support = static_cast<dse::ResourceTimeEstimateSupport>(*support);
  return hint;
}

void encodeRoots(Encoder &encoder,
                 llvm::ArrayRef<ArtifactRootReference> roots) {
  encoder.u64(roots.size());
  for (const ArtifactRootReference &root : roots)
    encoder.root(root);
}

void encodeOptionalRoot(Encoder &encoder,
                        const std::optional<ArtifactRootReference> &reference) {
  encoder.u32(reference ? 1 : 0);
  if (reference)
    encoder.root(*reference);
}

llvm::Expected<std::vector<ArtifactRootReference>>
decodeRoots(Decoder &decoder, llvm::StringRef field) {
  auto count = decoder.count((field + " count").str());
  if (!count)
    return count.takeError();
  std::vector<ArtifactRootReference> roots;
  roots.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto root = decoder.root(field);
    if (!root)
      return root.takeError();
    roots.push_back(std::move(*root));
  }
  return roots;
}

llvm::Expected<std::optional<ArtifactRootReference>>
decodeOptionalRoot(Decoder &decoder, llvm::StringRef field) {
  auto present = decoder.u32((field + " presence").str());
  if (!present)
    return present.takeError();
  if (*present > 1)
    return malformed(field + " has a noncanonical presence tag");
  if (*present == 0)
    return std::optional<ArtifactRootReference>{};
  auto root = decoder.root(field);
  if (!root)
    return root.takeError();
  return std::optional<ArtifactRootReference>(std::move(*root));
}

llvm::Expected<ComponentViewDigest> decodeDigest(Decoder &decoder,
                                                 llvm::StringRef field) {
  auto bytes = decoder.fixed(ComponentViewDigest::byteSize, field);
  if (!bytes)
    return bytes.takeError();
  auto digest = ComponentViewDigest::fromBytes(*bytes);
  if (!digest)
    return malformed(field + " is not a component-view digest");
  return std::move(*digest);
}

llvm::Expected<BlobDigest> decodeBlobDigest(Decoder &decoder,
                                            llvm::StringRef field) {
  auto bytes = decoder.fixed(BlobDigest::byteSize, field);
  if (!bytes)
    return bytes.takeError();
  auto digest = BlobDigest::fromBytes(*bytes);
  if (!digest)
    return malformed(field + " is not a Blob digest");
  return std::move(*digest);
}

std::vector<std::uint8_t>
encodedScheduleHint(const dse::ResourceTimeScheduleHint &hint) {
  Encoder encoder;
  encodeScheduleHint(encoder, hint);
  return encoder.take();
}

std::vector<std::uint8_t>
encodeDecision(const ApplicationActivationDecisionDraft &draft) {
  Encoder encoder;
  encoder.text(applicationActivationDecisionSchema.identity);
  encoder.u32(applicationActivationDecisionSchema.version.major);
  encoder.u32(applicationActivationDecisionSchema.version.minor);
  encoder.root(draft.sourceProgram);
  encoder.root(draft.fabric);
  encoder.root(draft.workload);
  encoder.root(draft.runtimeInput);
  encoder.u64(draft.sourceBackedReplayCases.size());
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       draft.sourceBackedReplayCases) {
    encoder.root(replay.workload);
    encoder.root(replay.runtimeInput);
  }
  encoder.root(draft.dseInvocation.resolvedConfig());
  encoder.fixed(draft.dseInvocation.blob().bytes());
  encoder.fixed(draft.dseInvocation.occurrence().runKey.bytes());
  encoder.u64(draft.dseInvocation.occurrence().occurrenceOrdinal);
  encoder.u64(draft.supportingDseInvocations.size());
  for (const dse::JointDesignInvocationManifestReference &supporting :
       draft.supportingDseInvocations) {
    encoder.root(supporting.resolvedConfig());
    encoder.fixed(supporting.blob().bytes());
    encoder.fixed(supporting.occurrence().runKey.bytes());
    encoder.u64(supporting.occurrence().occurrenceOrdinal);
  }
  encoder.root(draft.planning.structuredProgram);
  encoder.root(draft.planning.canonicalDataflow);
  encoder.u64(draft.planning.ownedProtocolRoots.size());
  for (const frontend::StructuredEntityRef &root :
       draft.planning.ownedProtocolRoots)
    encoder.bytes(frontend::encodeStructuredEntityRef(root));
  encoder.fixed(draft.planning.projectionIdentity.bytes());
  encoder.fixed(draft.planning.frontierPolicyDigest.bytes());
  encoder.u64(draft.selectedPlanOrdinal);
  encoder.u64(draft.selectedScheduleHints.size());
  for (const dse::ResourceTimeScheduleHint &hint : draft.selectedScheduleHints)
    encodeScheduleHint(encoder, hint);
  encoder.root(draft.selectedSystem);
  encoder.root(draft.selectedMapping);
  encoder.u32(static_cast<std::uint32_t>(draft.disposition));
  encodeRoots(encoder, draft.runtimeEvidence);
  encodeRoots(encoder, draft.oracleEvidence);
  encodeOptionalRoot(encoder, draft.selectedHardwareMutationRepairRecord);
  encodeRoots(encoder, draft.hardwareMutationRepairRecords);
  return encoder.take();
}

llvm::Expected<ComponentViewDigest> deriveSelectedCandidateIdentity(
    const ApplicationActivationPlanningPreimage &planning,
    const ArtifactRootReference &sourceProgram,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  dse::PreMappingCandidatePlanningRecord record;
  record.structuredProgram = planning.structuredProgram;
  record.canonicalDataflow = planning.canonicalDataflow;
  record.ownedProtocolRoots = planning.ownedProtocolRoots;
  record.projection.emplace(planning.projectionIdentity);
  return dse::computePreMappingCandidateIdentity(record, sourceProgram, fabric,
                                                 workload, runtimeInput,
                                                 planning.frontierPolicyDigest);
}

llvm::Expected<ApplicationActivationDecisionDraft>
decodeDecision(llvm::ArrayRef<std::uint8_t> bytes,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  Decoder decoder(bytes);
  auto schema = decoder.text("activation decision schema identity");
  if (!schema)
    return schema.takeError();
  auto major = decoder.u32("activation decision schema major");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("activation decision schema minor");
  if (!minor)
    return minor.takeError();
  if (*schema != applicationActivationDecisionSchema.identity ||
      SchemaVersion{*major, *minor} !=
          applicationActivationDecisionSchema.version)
    return reject(ApplicationActivationDecisionErrorReason::ForeignSchema,
                  "unsupported Application activation decision schema");
  auto source = decoder.root("source Program");
  if (!source)
    return source.takeError();
  auto fabric = decoder.root("source Fabric");
  if (!fabric)
    return fabric.takeError();
  auto workload = decoder.root("source workload");
  if (!workload)
    return workload.takeError();
  auto runtimeInput = decoder.root("source runtime input");
  if (!runtimeInput)
    return runtimeInput.takeError();
  auto replayCount = decoder.count("source-backed replay case count");
  if (!replayCount)
    return replayCount.takeError();
  std::vector<sim::SourceBackedDfgReplayCaseReference> replayCases;
  replayCases.reserve(*replayCount);
  for (std::size_t index = 0; index != *replayCount; ++index) {
    auto replayWorkload = decoder.root("replay workload");
    if (!replayWorkload)
      return replayWorkload.takeError();
    auto replayRuntimeInput = decoder.root("replay runtime input");
    if (!replayRuntimeInput)
      return replayRuntimeInput.takeError();
    replayCases.push_back(
        {std::move(*replayWorkload), std::move(*replayRuntimeInput)});
  }
  auto resolvedConfig = decoder.root("DSE invocation ResolvedConfig");
  if (!resolvedConfig)
    return resolvedConfig.takeError();
  auto invocationBlob = decodeBlobDigest(decoder, "DSE InvocationManifest");
  if (!invocationBlob)
    return invocationBlob.takeError();
  auto invocationRunKeyBytes =
      decoder.fixed(dse::DseRunKey::byteSize, "DSE invocation run key");
  if (!invocationRunKeyBytes)
    return invocationRunKeyBytes.takeError();
  auto invocationRunKey = dse::DseRunKey::fromBytes(*invocationRunKeyBytes);
  if (!invocationRunKey)
    return malformed("DSE invocation run key is malformed");
  auto invocationOrdinal = decoder.u64("DSE invocation occurrence ordinal");
  if (!invocationOrdinal)
    return invocationOrdinal.takeError();
  auto invocation = dse::JointDesignInvocationManifestReference::get(
      std::move(*resolvedConfig), std::move(*invocationBlob),
      dse::InvocationOccurrenceRef{std::move(*invocationRunKey),
                                   *invocationOrdinal},
      artifacts, blobs);
  if (!invocation)
    return reject(ApplicationActivationDecisionErrorReason::InvocationMismatch,
                  "DSE invocation reference failed strict import: " +
                      llvm::toString(invocation.takeError()));
  auto supportingCount = decoder.count("supporting DSE invocation count");
  if (!supportingCount)
    return supportingCount.takeError();
  std::vector<dse::JointDesignInvocationManifestReference>
      supportingInvocations;
  supportingInvocations.reserve(*supportingCount);
  for (std::size_t index = 0; index != *supportingCount; ++index) {
    auto supportingConfig =
        decoder.root("supporting DSE invocation ResolvedConfig");
    if (!supportingConfig)
      return supportingConfig.takeError();
    auto supportingBlob =
        decodeBlobDigest(decoder, "supporting DSE InvocationManifest");
    if (!supportingBlob)
      return supportingBlob.takeError();
    auto supportingRunKeyBytes = decoder.fixed(
        dse::DseRunKey::byteSize, "supporting DSE invocation run key");
    if (!supportingRunKeyBytes)
      return supportingRunKeyBytes.takeError();
    auto supportingRunKey = dse::DseRunKey::fromBytes(*supportingRunKeyBytes);
    if (!supportingRunKey)
      return malformed("supporting DSE invocation run key is malformed");
    auto supportingOrdinal =
        decoder.u64("supporting DSE invocation occurrence ordinal");
    if (!supportingOrdinal)
      return supportingOrdinal.takeError();
    auto supporting = dse::JointDesignInvocationManifestReference::get(
        std::move(*supportingConfig), std::move(*supportingBlob),
        dse::InvocationOccurrenceRef{std::move(*supportingRunKey),
                                     *supportingOrdinal},
        artifacts, blobs);
    if (!supporting)
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          "supporting DSE invocation failed strict import: " +
              llvm::toString(supporting.takeError()));
    supportingInvocations.push_back(std::move(*supporting));
  }
  auto planningProgram = decoder.root("selected planning StructuredProgram");
  if (!planningProgram)
    return planningProgram.takeError();
  auto planningDataflow = decoder.root("selected planning CanonicalDataflow");
  if (!planningDataflow)
    return planningDataflow.takeError();
  auto rootCount = decoder.count("selected owned protocol root count");
  if (!rootCount)
    return rootCount.takeError();
  std::vector<frontend::StructuredEntityRef> ownedRoots;
  ownedRoots.reserve(*rootCount);
  for (std::size_t index = 0; index != *rootCount; ++index) {
    auto encoded = decoder.bytes("selected owned protocol root");
    if (!encoded)
      return encoded.takeError();
    auto root = frontend::decodeStructuredEntityRef(*encoded);
    if (!root || frontend::encodeStructuredEntityRef(*root) != *encoded)
      return malformed("selected owned protocol root is not canonical");
    ownedRoots.push_back(std::move(*root));
  }
  auto projectionIdentity =
      decodeDigest(decoder, "selected planning projection identity");
  if (!projectionIdentity)
    return projectionIdentity.takeError();
  auto frontierPolicyDigest =
      decodeDigest(decoder, "selected frontier policy digest");
  if (!frontierPolicyDigest)
    return frontierPolicyDigest.takeError();
  auto selectedPlan = decoder.u64("selected plan ordinal");
  if (!selectedPlan)
    return selectedPlan.takeError();
  auto hintCount = decoder.count("selected schedule hint count");
  if (!hintCount)
    return hintCount.takeError();
  std::vector<dse::ResourceTimeScheduleHint> hints;
  hints.reserve(*hintCount);
  for (std::size_t index = 0; index != *hintCount; ++index) {
    auto hint = decodeScheduleHint(decoder);
    if (!hint)
      return hint.takeError();
    hints.push_back(std::move(*hint));
  }
  auto selectedSystem = decoder.root("selected System");
  if (!selectedSystem)
    return selectedSystem.takeError();
  auto selectedMapping = decoder.root("selected SystemMapping");
  if (!selectedMapping)
    return selectedMapping.takeError();
  auto disposition = decoder.u32("activation disposition");
  if (!disposition)
    return disposition.takeError();
  if (*disposition >
      static_cast<std::uint32_t>(
          ApplicationPairDecisionDisposition::HardwareDseAlternative))
    return malformed("activation decision has an unknown disposition");
  auto runtimeEvidence = decodeRoots(decoder, "runtime Evidence");
  if (!runtimeEvidence)
    return runtimeEvidence.takeError();
  auto oracleEvidence = decodeRoots(decoder, "oracle Evidence");
  if (!oracleEvidence)
    return oracleEvidence.takeError();
  auto selectedRepair =
      decodeOptionalRoot(decoder, "selected hardware mutation repair record");
  if (!selectedRepair)
    return selectedRepair.takeError();
  auto repairRecords = decodeRoots(decoder, "hardware mutation repair records");
  if (!repairRecords)
    return repairRecords.takeError();
  if (!decoder.atEnd())
    return malformed("activation decision has trailing bytes");

  ApplicationActivationPlanningPreimage planning{
      std::move(*planningProgram), std::move(*planningDataflow),
      std::move(ownedRoots), std::move(*projectionIdentity),
      std::move(*frontierPolicyDigest)};
  auto candidate = deriveSelectedCandidateIdentity(planning, *source, *fabric,
                                                   *workload, *runtimeInput);
  if (!candidate)
    return candidate.takeError();
  return ApplicationActivationDecisionDraft{
      std::move(*source),
      std::move(*fabric),
      std::move(*workload),
      std::move(*runtimeInput),
      std::move(replayCases),
      std::move(*invocation),
      std::move(supportingInvocations),
      std::move(planning),
      std::move(*candidate),
      *selectedPlan,
      std::move(hints),
      std::move(*selectedSystem),
      std::move(*selectedMapping),
      static_cast<ApplicationPairDecisionDisposition>(*disposition),
      std::move(*runtimeEvidence),
      std::move(*oracleEvidence),
      std::move(*selectedRepair),
      std::move(*repairRecords)};
}

template <typename Range, typename Less>
llvm::Error requireCanonicalUnique(const Range &values, Less less,
                                   const llvm::Twine &field) {
  if (!llvm::is_sorted(values, less) ||
      std::adjacent_find(values.begin(), values.end(),
                         [&](const auto &lhs, const auto &rhs) {
                           return !less(lhs, rhs) && !less(rhs, lhs);
                         }) != values.end())
    return reject(
        ApplicationActivationDecisionErrorReason::NonCanonicalEncoding,
        field + " must be a canonical set");
  return llvm::Error::success();
}

bool allocationLess(const dse::ResourceTimeHintAllocation &lhs,
                    const dse::ResourceTimeHintAllocation &rhs) {
  return rootLaunchLess(lhs.region, rhs.region);
}

bool allocationsEqual(llvm::ArrayRef<dse::ResourceTimeHintAllocation> lhs,
                      llvm::ArrayRef<dse::ResourceTimeHintAllocation> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                    [](const auto &left, const auto &right) {
                      return left.region == right.region &&
                             left.speedupPointOrdinal ==
                                 right.speedupPointOrdinal &&
                             left.resourceUnits == right.resourceUnits &&
                             left.completionTimePicoseconds ==
                                 right.completionTimePicoseconds;
                    });
}

llvm::Error
validateRootLaunchSet(llvm::ArrayRef<dataflow::RootThreadLaunchRef> references,
                      const dataflow::CanonicalDataflowProgramView &view,
                      const ArtifactIdentity &dataflowIdentity,
                      const llvm::Twine &field) {
  if (llvm::Error error =
          requireCanonicalUnique(references, rootLaunchLess, field))
    return error;
  for (dataflow::RootThreadLaunchRef reference : references) {
    if (reference.artifact != dataflowIdentity)
      return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                    field + " contains a foreign CanonicalDataflow reference");
    auto entity = view.resolve(reference);
    if (!entity)
      return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                    field + " contains an unresolved root launch: " +
                        llvm::toString(entity.takeError()));
  }
  return llvm::Error::success();
}

llvm::Error
validateScheduleHint(const dse::ResourceTimeScheduleHint &hint,
                     const dataflow::CanonicalDataflowProgramView &view,
                     const ArtifactIdentity &dataflowIdentity) {
  if (hint.states.empty() || hint.states.size() != hint.actions.size() + 1)
    return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                  "selected schedule hint has incomplete action/state lineage");
  if (!hint.states.front().active.empty() ||
      !hint.states.front().completed.empty())
    return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                  "selected schedule hint does not begin before execution");

  std::uint64_t computedPeak = 0;
  std::uint64_t computedResourceTime = 0;
  std::vector<dataflow::RootThreadLaunchRef> admitted;
  for (const dse::ResourceTimeHintState &state : hint.states) {
    computedPeak = std::max<std::uint64_t>(computedPeak, state.active.size());
    if (llvm::Error error = requireCanonicalUnique(
            state.active, allocationLess, "active schedule allocations"))
      return error;
    if (llvm::Error error = validateRootLaunchSet(
            state.ready, view, dataflowIdentity, "ready schedule regions"))
      return error;
    if (llvm::Error error =
            validateRootLaunchSet(state.completed, view, dataflowIdentity,
                                  "completed schedule regions"))
      return error;
    for (const dse::ResourceTimeHintAllocation &allocation : state.active) {
      if (allocation.region.artifact != dataflowIdentity ||
          allocation.resourceUnits.empty() ||
          llvm::all_of(allocation.resourceUnits,
                       [](std::uint64_t units) { return units == 0; }) ||
          allocation.completionTimePicoseconds <= state.timePicoseconds)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "selected schedule hint has an invalid active allocation");
      auto entity = view.resolve(allocation.region);
      if (!entity)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "selected schedule hint has an unresolved active region: " +
                llvm::toString(entity.takeError()));
      if (llvm::is_contained(state.ready, allocation.region) ||
          llvm::is_contained(state.completed, allocation.region))
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "one schedule region has overlapping lifecycle states");
    }
    for (dataflow::RootThreadLaunchRef completed : state.completed)
      if (llvm::is_contained(state.ready, completed))
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "one schedule region is both ready and completed");
  }

  for (std::size_t index = 0; index != hint.actions.size(); ++index) {
    const dse::ResourceTimeActionDelta &action = hint.actions[index];
    const dse::ResourceTimeHintState &before = hint.states[index];
    const dse::ResourceTimeHintState &after = hint.states[index + 1];
    if (action.beforeTimePicoseconds != before.timePicoseconds ||
        action.afterTimePicoseconds != after.timePicoseconds)
      return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                    "schedule action times differ from adjacent states");
    for (const auto *references :
         {&action.completedRegions, &action.tokenReadyProducers,
          &action.newlyReadyRegions})
      if (llvm::Error error =
              validateRootLaunchSet(*references, view, dataflowIdentity,
                                    "schedule action region set"))
        return error;

    if (action.kind == dse::ResourceTimeActionKind::AdmitRegion) {
      if (!action.admittedRegion || !action.speedupPointOrdinal ||
          action.beforeTimePicoseconds != action.afterTimePicoseconds ||
          !action.completedRegions.empty() ||
          !action.tokenReadyProducers.empty() ||
          !action.newlyReadyRegions.empty() ||
          action.admittedRegion->artifact != dataflowIdentity ||
          llvm::is_contained(admitted, *action.admittedRegion))
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule admission action is malformed");
      auto resolved = view.resolve(*action.admittedRegion);
      if (!resolved)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule admission names an unresolved region: " +
                llvm::toString(resolved.takeError()));
      if (!llvm::is_contained(before.ready, *action.admittedRegion))
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule admits a region which is not ready");
      const auto allocation = llvm::find_if(
          after.active, [&](const dse::ResourceTimeHintAllocation &candidate) {
            return candidate.region == *action.admittedRegion;
          });
      if (allocation == after.active.end() ||
          allocation->speedupPointOrdinal != *action.speedupPointOrdinal)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule admission has no matching allocation");
      std::vector<dse::ResourceTimeHintAllocation> expectedActive =
          before.active;
      expectedActive.push_back(*allocation);
      llvm::sort(expectedActive, allocationLess);
      std::vector<dataflow::RootThreadLaunchRef> expectedReady = before.ready;
      expectedReady.erase(llvm::find(expectedReady, *action.admittedRegion));
      if (!allocationsEqual(after.active, expectedActive) ||
          after.ready != expectedReady || after.completed != before.completed)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule admission changes unrelated lifecycle state");
      std::uint64_t allocationMagnitude = 0;
      for (std::uint64_t units : allocation->resourceUnits) {
        auto sum = llvm::checkedAddUnsigned(allocationMagnitude, units);
        if (!sum)
          return reject(
              ApplicationActivationDecisionErrorReason::ScheduleMismatch,
              "schedule allocation magnitude overflows");
        allocationMagnitude = *sum;
      }
      const std::uint64_t duration =
          allocation->completionTimePicoseconds - action.beforeTimePicoseconds;
      auto work = llvm::checkedMulUnsigned(duration, allocationMagnitude);
      if (!work)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule allocated resource time overflows");
      auto total = llvm::checkedAddUnsigned(computedResourceTime, *work);
      if (!total)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule allocated resource time overflows");
      computedResourceTime = *total;
      admitted.push_back(*action.admittedRegion);
    } else {
      if (action.admittedRegion || action.speedupPointOrdinal ||
          action.afterTimePicoseconds <= action.beforeTimePicoseconds ||
          (action.completedRegions.empty() &&
           action.tokenReadyProducers.empty()))
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule event-advance action is malformed");
      for (dataflow::RootThreadLaunchRef completed : action.completedRegions) {
        const auto active = llvm::find_if(
            before.active,
            [&](const dse::ResourceTimeHintAllocation &allocation) {
              return allocation.region == completed;
            });
        if (active == before.active.end() ||
            active->completionTimePicoseconds != action.afterTimePicoseconds ||
            !llvm::is_contained(after.completed, completed))
          return reject(
              ApplicationActivationDecisionErrorReason::ScheduleMismatch,
              "schedule completion does not match an active allocation");
      }
      for (dataflow::RootThreadLaunchRef producer : action.tokenReadyProducers)
        if (llvm::none_of(before.active, [&](const auto &allocation) {
              return allocation.region == producer;
            }))
          return reject(
              ApplicationActivationDecisionErrorReason::ScheduleMismatch,
              "schedule token event has no active producer");
      std::vector<dse::ResourceTimeHintAllocation> expectedActive;
      for (const dse::ResourceTimeHintAllocation &allocation : before.active)
        if (!llvm::is_contained(action.completedRegions, allocation.region))
          expectedActive.push_back(allocation);
      std::vector<dataflow::RootThreadLaunchRef> expectedReady = before.ready;
      expectedReady.insert(expectedReady.end(),
                           action.newlyReadyRegions.begin(),
                           action.newlyReadyRegions.end());
      llvm::sort(expectedReady, rootLaunchLess);
      std::vector<dataflow::RootThreadLaunchRef> expectedCompleted =
          before.completed;
      expectedCompleted.insert(expectedCompleted.end(),
                               action.completedRegions.begin(),
                               action.completedRegions.end());
      llvm::sort(expectedCompleted, rootLaunchLess);
      if (!allocationsEqual(after.active, expectedActive) ||
          after.ready != expectedReady || after.completed != expectedCompleted)
        return reject(
            ApplicationActivationDecisionErrorReason::ScheduleMismatch,
            "schedule event delta changes unrelated lifecycle state");
    }
  }
  llvm::sort(admitted, rootLaunchLess);
  if (hint.states.back().timePicoseconds != hint.estimatedMakespanPicoseconds ||
      hint.states.back().optimisticMakespanLowerBoundPicoseconds !=
          hint.optimisticMakespanLowerBoundPicoseconds ||
      hint.optimisticMakespanLowerBoundPicoseconds >
          hint.estimatedMakespanPicoseconds ||
      hint.peakConcurrentRegions != computedPeak ||
      hint.totalAllocatedResourceTime != computedResourceTime ||
      !hint.states.back().active.empty() || !hint.states.back().ready.empty() ||
      hint.states.back().completed != admitted)
    return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                  "selected schedule summary differs from its exact lineage");
  return llvm::Error::success();
}

llvm::Error canonicalizeReplayCases(
    std::vector<sim::SourceBackedDfgReplayCaseReference> &replayCases) {
  const auto less = [](const auto &lhs, const auto &rhs) {
    if (lhs.workload != rhs.workload)
      return artifactRootReferenceLess(lhs.workload, rhs.workload);
    return artifactRootReferenceLess(lhs.runtimeInput, rhs.runtimeInput);
  };
  llvm::sort(replayCases, less);
  return requireCanonicalUnique(replayCases, less,
                                "source-backed replay cases");
}

llvm::Error canonicalizeRoots(std::vector<ArtifactRootReference> &roots,
                              const llvm::Twine &field) {
  llvm::sort(roots, artifactRootReferenceLess);
  return requireCanonicalUnique(roots, artifactRootReferenceLess, field);
}

llvm::Error
canonicalizePlanningRoots(std::vector<frontend::StructuredEntityRef> &roots) {
  const auto less = [](const frontend::StructuredEntityRef &lhs,
                       const frontend::StructuredEntityRef &rhs) {
    return frontend::encodeStructuredEntityRef(lhs) <
           frontend::encodeStructuredEntityRef(rhs);
  };
  llvm::sort(roots, less);
  return requireCanonicalUnique(roots, less, "selected owned protocol roots");
}

llvm::Error
canonicalizeScheduleHints(std::vector<dse::ResourceTimeScheduleHint> &hints) {
  struct Key final {
    ComponentViewDigest digest;
    std::vector<std::uint8_t> bytes;
  };
  std::vector<Key> keys;
  keys.reserve(hints.size());
  for (const dse::ResourceTimeScheduleHint &hint : hints) {
    auto digest = dse::deriveResourceTimeScheduleHintDigest(hint);
    if (!digest)
      return digest.takeError();
    keys.push_back({std::move(*digest), encodedScheduleHint(hint)});
  }
  std::vector<std::size_t> order(hints.size());
  for (std::size_t index = 0; index != order.size(); ++index)
    order[index] = index;
  llvm::sort(order, [&](std::size_t lhs, std::size_t rhs) {
    if (keys[lhs].digest != keys[rhs].digest)
      return keys[lhs].digest.bytes() < keys[rhs].digest.bytes();
    return keys[lhs].bytes < keys[rhs].bytes;
  });
  for (std::size_t index = 1; index != order.size(); ++index)
    if (keys[order[index - 1]].digest == keys[order[index]].digest)
      return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                    keys[order[index - 1]].bytes == keys[order[index]].bytes
                        ? "selected schedule hint is repeated"
                        : "selected schedule hints have a digest collision");
  std::vector<dse::ResourceTimeScheduleHint> canonical;
  canonical.reserve(hints.size());
  for (std::size_t index : order)
    canonical.push_back(std::move(hints[index]));
  hints = std::move(canonical);
  return llvm::Error::success();
}

bool invocationReferenceLess(
    const dse::JointDesignInvocationManifestReference &lhs,
    const dse::JointDesignInvocationManifestReference &rhs) {
  if (lhs.occurrence().runKey.bytes() != rhs.occurrence().runKey.bytes())
    return lhs.occurrence().runKey.bytes() < rhs.occurrence().runKey.bytes();
  if (lhs.occurrence().occurrenceOrdinal != rhs.occurrence().occurrenceOrdinal)
    return lhs.occurrence().occurrenceOrdinal <
           rhs.occurrence().occurrenceOrdinal;
  if (lhs.blob() != rhs.blob())
    return lhs.blob().bytes() < rhs.blob().bytes();
  return artifactRootReferenceLess(lhs.resolvedConfig(), rhs.resolvedConfig());
}

bool sameInvocationReference(
    const dse::JointDesignInvocationManifestReference &lhs,
    const dse::JointDesignInvocationManifestReference &rhs) {
  return !invocationReferenceLess(lhs, rhs) &&
         !invocationReferenceLess(rhs, lhs);
}

bool sameInvocationOccurrence(
    const dse::JointDesignInvocationManifestReference &lhs,
    const dse::JointDesignInvocationManifestReference &rhs) {
  return lhs.occurrence() == rhs.occurrence();
}

llvm::Error canonicalizeSupportingInvocations(
    std::vector<dse::JointDesignInvocationManifestReference> &supporting,
    const dse::JointDesignInvocationManifestReference &primary) {
  llvm::sort(supporting, invocationReferenceLess);
  for (std::size_t index = 0; index != supporting.size(); ++index) {
    if (sameInvocationOccurrence(supporting[index], primary))
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          sameInvocationReference(supporting[index], primary)
              ? "primary DSE invocation is repeated as supporting"
              : "primary DSE occurrence has a conflicting manifest");
    if (index != 0 &&
        sameInvocationOccurrence(supporting[index - 1], supporting[index]))
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          sameInvocationReference(supporting[index - 1], supporting[index])
              ? "supporting DSE invocation is repeated"
              : "supporting DSE occurrence has conflicting manifests");
  }
  return llvm::Error::success();
}

bool isEvidenceRoot(const ArtifactRootReference &root) {
  return root.schemaIdentity ==
             evaluation::EvaluationEvidence::artifactSchema.identity &&
         root.schemaVersion ==
             evaluation::EvaluationEvidence::artifactSchema.version;
}

llvm::Error
addDependencyRoot(ApplicationActivationDecisionDependencyProjection &projection,
                  const ArtifactRootReference &root,
                  const ArtifactStore &artifacts) {
  auto stored = artifacts.get(root);
  if (!stored)
    return reject(ApplicationActivationDecisionErrorReason::DependencyMismatch,
                  "activation dependency is unavailable: " +
                      llvm::toString(stored.takeError()));
  if (!llvm::is_contained(projection.artifacts, root))
    projection.artifacts.push_back(root);
  return llvm::Error::success();
}

llvm::Error addRequestPayloadBlobs(
    ApplicationActivationDecisionDependencyProjection &projection,
    llvm::ArrayRef<ArtifactRootReference> requestDependencies,
    const ArtifactStore &artifacts) {
  for (const ArtifactRootReference &root : requestDependencies) {
    if (root.schemaIdentity !=
            evaluation::modelParameterBundleSchema.identity ||
        root.schemaVersion != evaluation::modelParameterBundleSchema.version)
      continue;
    auto bundle = evaluation::importModelParameterBundleRoot(root, artifacts);
    if (!bundle)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "model parameter bundle failed strict root import: " +
                        llvm::toString(bundle.takeError()));
    if (!llvm::is_contained(projection.blobs, bundle->payloadDigest()))
      projection.blobs.push_back(bundle->payloadDigest());
  }
  return llvm::Error::success();
}

llvm::Error addEvidenceDependencies(
    ApplicationActivationDecisionDependencyProjection &projection,
    const ArtifactRootReference &evidence, const ArtifactStore &artifacts) {
  if (!isEvidenceRoot(evidence))
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "activation Evidence has a foreign schema");
  if (llvm::Error error = addDependencyRoot(projection, evidence, artifacts))
    return error;
  auto facts = evaluation::importEvaluationEvidenceDependencyProjection(
      evidence, artifacts);
  if (!facts)
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "activation Evidence failed dependency projection: " +
                      llvm::toString(facts.takeError()));
  if (facts->outcomeKind != evaluation::EvidenceOutcomeKind::Completed)
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "activation Evidence is not completed");
  if (llvm::Error error =
          addDependencyRoot(projection, facts->request, artifacts))
    return error;
  auto requestDependencies =
      evaluation::importEvaluationRequestArtifactReferences(facts->request,
                                                            artifacts);
  if (!requestDependencies)
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "activation Evidence Request failed dependency projection: " +
                      llvm::toString(requestDependencies.takeError()));
  for (const ArtifactRootReference &root : *requestDependencies)
    if (llvm::Error error = addDependencyRoot(projection, root, artifacts))
      return error;
  if (llvm::Error error =
          addRequestPayloadBlobs(projection, *requestDependencies, artifacts))
    return error;
  for (const evaluation::ModelOutputBinding &binding : facts->outputBindings)
    for (const ArtifactRootReference &root : binding.artifacts)
      if (llvm::Error error = addDependencyRoot(projection, root, artifacts))
        return error;
  return llvm::Error::success();
}

llvm::Error addInvocationDependencies(
    ApplicationActivationDecisionDependencyProjection &projection,
    const dse::InvocationManifest &manifest, const ArtifactStore &artifacts) {
  for (const ArtifactRootReference &root : manifest.closure().semanticInputs())
    if (llvm::Error error = addDependencyRoot(projection, root, artifacts))
      return error;
  for (const ArtifactRootReference &root :
       manifest.closure().preexistingEvidence())
    if (llvm::Error error =
            addEvidenceDependencies(projection, root, artifacts))
      return error;
  for (const dse::InvocationGenerateRecord &record :
       manifest.generateRecords()) {
    for (const dse::CandidateGeneratorInputBinding &binding :
         record.invocation.inputBindings)
      for (const ArtifactRootReference &root : binding.artifacts)
        if (llvm::Error error = addDependencyRoot(projection, root, artifacts))
          return error;
    for (const dse::CandidateGeneratorOutputBinding &binding :
         record.invocation.outputBindings)
      for (const ArtifactRootReference &root : binding.artifacts) {
        llvm::Error error =
            isEvidenceRoot(root)
                ? addEvidenceDependencies(projection, root, artifacts)
                : addDependencyRoot(projection, root, artifacts);
        if (error)
          return error;
      }
    for (const dse::CandidateGeneratorLineageEdge &edge :
         record.invocation.lineageEdges) {
      if (llvm::Error error =
              addDependencyRoot(projection, edge.output, artifacts))
        return error;
      for (const ArtifactRootReference &root : edge.parents)
        if (llvm::Error error = addDependencyRoot(projection, root, artifacts))
          return error;
    }
  }
  return std::visit(
      [&](const auto &outcome) -> llvm::Error {
        using T = std::decay_t<decltype(outcome)>;
        if constexpr (std::is_same_v<T, dse::InvocationCompletedSelection>) {
          for (const ArtifactRootReference &root : outcome.selected)
            if (llvm::Error error =
                    addDependencyRoot(projection, root, artifacts))
              return error;
          for (const ArtifactRootReference &root : outcome.satisfiedEvidence)
            if (llvm::Error error =
                    addEvidenceDependencies(projection, root, artifacts))
              return error;
        } else if constexpr (std::is_same_v<
                                 T,
                                 dse::InvocationCompletedNoFeasibleCandidate>) {
          for (const ArtifactRootReference &root : outcome.satisfiedEvidence)
            if (llvm::Error error =
                    addEvidenceDependencies(projection, root, artifacts))
              return error;
        } else {
          for (const ArtifactRootReference &root : outcome.retainedArtifacts)
            if (llvm::Error error =
                    addDependencyRoot(projection, root, artifacts))
              return error;
          for (const ArtifactRootReference &root : outcome.retainedEvidence)
            if (llvm::Error error =
                    addEvidenceDependencies(projection, root, artifacts))
              return error;
        }
        return llvm::Error::success();
      },
      manifest.outcome());
}

} // namespace

namespace detail {

llvm::Expected<ApplicationRuntimeEvidenceJoin>
resolveApplicationRuntimeEvidenceJoin(
    llvm::ArrayRef<ArtifactRootReference> runtimeEvidence,
    llvm::ArrayRef<ArtifactRootReference> oracleEvidence,
    const ArtifactRootReference &dataflow,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    llvm::ArrayRef<sim::SourceBackedDfgReplayCaseReference> replayCases,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (runtimeEvidence.empty() || oracleEvidence.empty())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "application runtime requires runtime and oracle Evidence");
  const auto hasDuplicateRoot =
      [](llvm::ArrayRef<ArtifactRootReference> roots) {
        std::vector<ArtifactRootReference> ordered(roots.begin(), roots.end());
        llvm::sort(ordered, artifactRootReferenceLess);
        return std::adjacent_find(ordered.begin(), ordered.end()) !=
               ordered.end();
      };
  if (hasDuplicateRoot(runtimeEvidence))
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence repeats an Evidence root");
  if (hasDuplicateRoot(oracleEvidence))
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "oracle Evidence repeats an Evidence root");
  for (const ArtifactRootReference &oracle : oracleEvidence) {
    if (!llvm::is_contained(runtimeEvidence, oracle))
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "oracle Evidence is outside the runtime Evidence set");
  }

  enum class ExecutionKind : std::uint8_t { Dfg, Cgra };
  struct ExecutionRecord final {
    ArtifactRootReference evidence;
    ArtifactRootReference execution;
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;
    evaluation::CaseArtifactResolution resolution;
    ExecutionKind kind;
  };
  struct EvidenceFacts final {
    ArtifactRootReference evidence;
    evaluation::EvaluationEvidenceDependencyProjection projection;
    std::vector<ArtifactRootReference> requestReferences;
  };
  std::vector<EvidenceFacts> evidenceFacts;
  std::vector<ExecutionRecord> executions;
  ApplicationRuntimeEvidenceJoin result;
  evidenceFacts.reserve(runtimeEvidence.size());
  executions.reserve(runtimeEvidence.size());
  for (const ArtifactRootReference &evidence : runtimeEvidence) {
    auto projection = evaluation::importEvaluationEvidenceDependencyProjection(
        evidence, artifacts);
    if (!projection)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence failed dependency projection: " +
                        llvm::toString(projection.takeError()));
    if (projection->outcomeKind != evaluation::EvidenceOutcomeKind::Completed)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence is not completed");
    auto requestReferences =
        evaluation::importEvaluationRequestArtifactReferences(
            projection->request, artifacts);
    if (!requestReferences)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence Request cannot be imported: " +
                        llvm::toString(requestReferences.takeError()));
    for (const evaluation::ModelOutputBinding &binding :
         projection->outputBindings)
      for (const ArtifactRootReference &root : binding.artifacts) {
        auto stored = artifacts.get(root);
        if (!stored)
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "runtime Evidence output is unavailable: " +
                  llvm::toString(stored.takeError()));
      }
    EvidenceFacts row{evidence, std::move(*projection),
                      std::move(*requestReferences)};
    std::optional<ArtifactRootReference> workload;
    std::optional<ArtifactRootReference> runtimeInput;
    for (const ArtifactRootReference &reference : row.requestReferences) {
      if (reference.schemaIdentity == sim::simulationWorkloadSchema.identity &&
          reference.schemaVersion == sim::simulationWorkloadSchema.version) {
        if (workload)
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "runtime Evidence Request repeats its SimulationWorkload");
        workload = reference;
      }
      if (reference.schemaIdentity ==
              sim::simulationRuntimeInputSchema.identity &&
          reference.schemaVersion ==
              sim::simulationRuntimeInputSchema.version) {
        if (runtimeInput)
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "runtime Evidence Request repeats its SimulationRuntimeInput");
        runtimeInput = reference;
      }
    }
    const bool hasDataflow =
        llvm::is_contained(row.requestReferences, dataflow);
    std::vector<ArtifactRootReference> selectedMappings;
    for (const ArtifactRootReference &mapping : spatialMappings)
      if (llvm::is_contained(row.requestReferences, mapping))
        selectedMappings.push_back(mapping);
    if (!hasDataflow && selectedMappings.empty()) {
      if (!llvm::is_contained(oracleEvidence, evidence))
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "non-execution Evidence is not declared as oracle Evidence");
      evidenceFacts.push_back(std::move(row));
      continue;
    }
    if (!workload || !runtimeInput)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "runtime Evidence Request has no exact workload and runtime input");
    if (selectedMappings.size() > 1)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "runtime Evidence Request repeats a selected SpatialMapping");
    std::optional<evaluation::CaseArtifactResolution> resolution;
    ExecutionKind kind = ExecutionKind::Dfg;
    if (!selectedMappings.empty()) {
      auto resolved = evaluation::models::resolveCgraSimulationCase(
          selectedMappings.front(), *workload, *runtimeInput, artifacts);
      if (!resolved)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "cannot resolve selected CGRA runtime case: " +
                llvm::toString(resolved.takeError()));
      if (resolved->canonicalDataflow != dataflow)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "CGRA runtime case names a foreign canonical Dataflow");
      resolution.emplace(std::move(resolved->resolution));
      kind = ExecutionKind::Cgra;
    } else {
      auto resolved = evaluation::models::resolveDfgSimulationCase(
          dataflow, *workload, *runtimeInput, artifacts);
      if (!resolved)
        return reject(
            ApplicationActivationDecisionErrorReason::EvidenceMismatch,
            "cannot resolve DFG runtime case: " +
                llvm::toString(resolved.takeError()));
      resolution.emplace(std::move(*resolved));
    }
    auto strict = evaluation::importEvaluationEvidence(evidence, *resolution,
                                                       artifacts, blobs);
    if (!strict)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence failed strict import: " +
                        llvm::toString(strict.takeError()));
    if (strict->requestRef() != row.projection.request ||
        strict->outcomeKind() != evaluation::EvidenceOutcomeKind::Completed)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "strict runtime Evidence differs from its dependency projection");
    const auto *completed =
        std::get_if<evaluation::CompletedEvidence>(&strict->outcome());
    const auto *point =
        completed && completed->metricResults.size() == 1
            ? std::get_if<evaluation::PointObservation>(
                  &completed->metricResults.front().observation)
            : nullptr;
    const auto *cycles =
        point ? std::get_if<evaluation::IntegerValue>(&point->value) : nullptr;
    if (!cycles || cycles->value() < 0)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence has no nonnegative cycle metric");
    std::uint64_t &cycleTotal =
        kind == ExecutionKind::Dfg ? result.dfgCycles : result.cgraCycles;
    const std::uint64_t cycleValue =
        static_cast<std::uint64_t>(cycles->value());
    if (cycleValue > std::numeric_limits<std::uint64_t>::max() - cycleTotal)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence cycle total overflowed");
    cycleTotal += cycleValue;
    std::optional<ArtifactRootReference> execution;
    for (const evaluation::ModelOutputBinding &binding :
         strict->outputBindings())
      for (const ArtifactRootReference &output : binding.artifacts)
        if (output.schemaIdentity == sim::simulationExecutionSchema.identity &&
            output.schemaVersion == sim::simulationExecutionSchema.version) {
          if (execution)
            return reject(
                ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                "runtime Evidence repeats its SimulationExecution output");
          execution = output;
        }
    if (!execution)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "runtime Evidence has no SimulationExecution output");
    auto executionRequest =
        sim::simulationExecutionRequestReference(*execution, artifacts);
    if (!executionRequest || *executionRequest != row.projection.request)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "SimulationExecution and Evidence name different Requests");
    executions.push_back({evidence, *execution, *workload, *runtimeInput,
                          std::move(*resolution), kind});
    evidenceFacts.push_back(std::move(row));
  }
  if (executions.empty() ||
      !llvm::any_of(executions,
                    [](const ExecutionRecord &record) {
                      return record.kind == ExecutionKind::Dfg;
                    }) ||
      !llvm::any_of(executions, [](const ExecutionRecord &record) {
        return record.kind == ExecutionKind::Cgra;
      }))
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence does not bind both DFG and CGRA execution");

  struct InputPair final {
    ArtifactRootReference workload;
    ArtifactRootReference runtimeInput;
    bool operator==(const InputPair &other) const {
      return workload == other.workload && runtimeInput == other.runtimeInput;
    }
  };
  const auto pairLess = [](const InputPair &lhs, const InputPair &rhs) {
    if (lhs.workload != rhs.workload)
      return artifactRootReferenceLess(lhs.workload, rhs.workload);
    return artifactRootReferenceLess(lhs.runtimeInput, rhs.runtimeInput);
  };
  auto canonicalizePairs = [&](std::vector<InputPair> &pairs) {
    llvm::sort(pairs, pairLess);
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  };
  std::vector<InputPair> expectedPairs;
  for (const sim::SourceBackedDfgReplayCaseReference &replay : replayCases)
    expectedPairs.push_back({replay.workload, replay.runtimeInput});
  const std::size_t expectedPairCount = expectedPairs.size();
  std::vector<InputPair> dfgPairs;
  std::vector<InputPair> cgraPairs;
  for (const ExecutionRecord &record : executions)
    (record.kind == ExecutionKind::Dfg ? dfgPairs : cgraPairs)
        .push_back({record.workload, record.runtimeInput});
  const std::size_t dfgExecutionCount = dfgPairs.size();
  const std::size_t cgraExecutionCount = cgraPairs.size();
  canonicalizePairs(expectedPairs);
  canonicalizePairs(dfgPairs);
  canonicalizePairs(cgraPairs);
  if (expectedPairs.empty() || expectedPairs.size() != expectedPairCount ||
      dfgPairs != expectedPairs || cgraPairs != expectedPairs ||
      dfgExecutionCount != expectedPairs.size() ||
      cgraExecutionCount != expectedPairs.size())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "runtime Evidence does not join through exact source-backed "
                  "replay inputs");

  std::vector<InputPair> comparisonPairs;
  std::vector<ArtifactRootReference> comparisonEvidence;
  for (const EvidenceFacts &row : evidenceFacts) {
    if (!llvm::is_contained(oracleEvidence, row.evidence))
      continue;
    std::vector<const ExecutionRecord *> compared;
    for (const ArtifactRootReference &reference : row.requestReferences)
      if (reference.schemaIdentity == sim::simulationExecutionSchema.identity &&
          reference.schemaVersion == sim::simulationExecutionSchema.version) {
        auto found =
            llvm::find_if(executions, [&](const ExecutionRecord &record) {
              return record.execution == reference;
            });
        if (found == executions.end())
          return reject(
              ApplicationActivationDecisionErrorReason::EvidenceMismatch,
              "oracle Evidence names a foreign SimulationExecution");
        compared.push_back(&*found);
      }
    if (compared.size() != 2 || compared[0]->kind == compared[1]->kind)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence does not compare one DFG and one CGRA execution");
    const ExecutionRecord *dfg =
        compared[0]->kind == ExecutionKind::Dfg ? compared[0] : compared[1];
    const ExecutionRecord *cgra =
        compared[0]->kind == ExecutionKind::Cgra ? compared[0] : compared[1];
    const InputPair dfgPair{dfg->workload, dfg->runtimeInput};
    const InputPair cgraPair{cgra->workload, cgra->runtimeInput};
    if (!(dfgPair == cgraPair))
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence compares executions from different replay inputs");
    auto resolution = evaluation::models::resolveSimulationComparisonCase(
        dfg->execution, dfg->resolution, cgra->execution, cgra->resolution,
        artifacts, blobs);
    if (!resolution)
      return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                    "cannot resolve SimulationComparison Evidence: " +
                        llvm::toString(resolution.takeError()));
    auto strict = evaluation::importEvaluationEvidence(
        row.evidence, *resolution, artifacts, blobs);
    if (!strict || strict->requestRef() != row.projection.request ||
        strict->outcomeKind() != evaluation::EvidenceOutcomeKind::Completed)
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence failed strict SimulationComparison import");
    const auto *completed =
        std::get_if<evaluation::CompletedEvidence>(&strict->outcome());
    if (!completed || completed->findingResults.size() != 1 ||
        !std::holds_alternative<evaluation::AbsentFinding>(
            completed->findingResults.front().result))
      return reject(
          ApplicationActivationDecisionErrorReason::EvidenceMismatch,
          "oracle Evidence did not establish an absent comparison finding");
    comparisonPairs.push_back(dfgPair);
    comparisonEvidence.push_back(row.evidence);
  }
  const std::size_t comparisonCount = comparisonPairs.size();
  canonicalizePairs(comparisonPairs);
  llvm::sort(comparisonEvidence, artifactRootReferenceLess);
  std::vector<ArtifactRootReference> expectedOracleEvidence(
      oracleEvidence.begin(), oracleEvidence.end());
  llvm::sort(expectedOracleEvidence, artifactRootReferenceLess);
  if (comparisonEvidence != expectedOracleEvidence ||
      comparisonPairs != expectedPairs ||
      comparisonCount != expectedPairs.size())
    return reject(ApplicationActivationDecisionErrorReason::EvidenceMismatch,
                  "oracle Evidence does not provide exact one-to-one "
                  "SimulationComparison coverage for the source-backed "
                  "replay inputs");
  return result;
}

} // namespace detail

namespace {

llvm::Error validateDecision(ApplicationActivationDecisionDraft &draft,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  switch (draft.disposition) {
  case ApplicationPairDecisionDisposition::VerifiedAcceleration:
  case ApplicationPairDecisionDisposition::VerifiedFeasibleButNotBeneficial:
  case ApplicationPairDecisionDisposition::HardwareDseAlternative:
    break;
  default:
    return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                  "activation decision has no completed pair disposition");
  }
  const bool selectedDifferentSystem = draft.selectedSystem != draft.fabric;
  if (selectedDifferentSystem !=
      (draft.disposition ==
       ApplicationPairDecisionDisposition::HardwareDseAlternative))
    return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                  "hardware-alternative disposition differs from the selected "
                  "System");

  auto sourceInputs = sim::importStructuredProgramSimulationInputs(
      draft.workload, draft.runtimeInput, artifacts);
  if (!sourceInputs)
    return reject(ApplicationActivationDecisionErrorReason::DependencyMismatch,
                  "source activation inputs failed strict import: " +
                      llvm::toString(sourceInputs.takeError()));
  if (sourceInputs->structuredProgram.identity() !=
      draft.sourceProgram.artifact)
    return reject(ApplicationActivationDecisionErrorReason::DependencyMismatch,
                  "source activation inputs name a foreign StructuredProgram");
  auto sourceProgram =
      frontend::importStructuredProgram(draft.sourceProgram, artifacts);
  if (!sourceProgram)
    return reject(ApplicationActivationDecisionErrorReason::DependencyMismatch,
                  "source StructuredProgram failed strict import: " +
                      llvm::toString(sourceProgram.takeError()));
  auto sourceFabric = fabric::importEntireFabricRoot(draft.fabric, artifacts);
  if (!sourceFabric)
    return reject(ApplicationActivationDecisionErrorReason::DependencyMismatch,
                  "source Fabric failed strict import: " +
                      llvm::toString(sourceFabric.takeError()));
  auto selectedSystem =
      fabric::importEntireFabricRoot(draft.selectedSystem, artifacts);
  if (!selectedSystem)
    return reject(ApplicationActivationDecisionErrorReason::MappingMismatch,
                  "selected System failed strict import: " +
                      llvm::toString(selectedSystem.takeError()));

  auto planningProgram = frontend::importStructuredProgram(
      draft.planning.structuredProgram, artifacts);
  if (!planningProgram)
    return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                  "selected planning StructuredProgram failed strict import: " +
                      llvm::toString(planningProgram.takeError()));
  auto planningView = planningProgram->view();
  if (!planningView)
    return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                  "selected planning StructuredProgram has no strict view: " +
                      llvm::toString(planningView.takeError()));
  if (draft.planning.ownedProtocolRoots.empty())
    return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                  "selected planning preimage has no owned protocol root");
  for (const frontend::StructuredEntityRef &root :
       draft.planning.ownedProtocolRoots) {
    auto entity = planningView->resolve(root);
    if (!entity)
      return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                    "selected planning preimage has an unresolved root: " +
                        llvm::toString(entity.takeError()));
  }
  auto dataflow = dataflow::importCanonicalDataflow(
      draft.planning.canonicalDataflow, artifacts);
  if (!dataflow)
    return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                  "selected CanonicalDataflow failed strict import: " +
                      llvm::toString(dataflow.takeError()));
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return reject(ApplicationActivationDecisionErrorReason::PlanningMismatch,
                  "selected CanonicalDataflow has no strict view: " +
                      llvm::toString(dataflowView.takeError()));

  auto candidate = deriveSelectedCandidateIdentity(
      draft.planning, draft.sourceProgram, draft.fabric, draft.workload,
      draft.runtimeInput);
  if (!candidate)
    return candidate.takeError();
  if (*candidate != draft.selectedCandidateIdentity)
    return reject(
        ApplicationActivationDecisionErrorReason::PlanningMismatch,
        "selected candidate differs from its exact planning preimage");

  auto mapping = mapping::importSystemMapping(draft.selectedMapping, artifacts);
  if (!mapping)
    return reject(ApplicationActivationDecisionErrorReason::MappingMismatch,
                  "selected SystemMapping failed strict import: " +
                      llvm::toString(mapping.takeError()));
  if (mapping->view().fabricIdentity() != draft.selectedSystem.artifact ||
      mapping->view().dataflowIdentity() !=
          draft.planning.canonicalDataflow.artifact)
    return reject(ApplicationActivationDecisionErrorReason::MappingMismatch,
                  "selected SystemMapping differs from the selected System or "
                  "CanonicalDataflow");
  if (llvm::Error error =
          activation_detail::validateHardwareMutationRepairs(draft, artifacts))
    return error;

  if (draft.sourceBackedReplayCases.empty())
    return reject(ApplicationActivationDecisionErrorReason::DependencyMismatch,
                  "activation decision has no source-backed replay case");
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       draft.sourceBackedReplayCases) {
    auto imported = sim::importSpatialSimulationInputs(
        replay.workload, replay.runtimeInput, artifacts);
    if (!imported)
      return reject(
          ApplicationActivationDecisionErrorReason::DependencyMismatch,
          "source-backed replay case failed strict import: " +
              llvm::toString(imported.takeError()));
    if (imported->dataflow.identity() !=
        draft.planning.canonicalDataflow.artifact)
      return reject(
          ApplicationActivationDecisionErrorReason::DependencyMismatch,
          "source-backed replay case names a foreign CanonicalDataflow");
  }

  if (draft.selectedScheduleHints.empty())
    return reject(ApplicationActivationDecisionErrorReason::ScheduleMismatch,
                  "activation decision has no selected schedule hint");
  for (const dse::ResourceTimeScheduleHint &hint : draft.selectedScheduleHints)
    if (llvm::Error error = validateScheduleHint(
            hint, *dataflowView, draft.planning.canonicalDataflow.artifact))
      return error;

  auto invocation = dse::importJointDesignInvocationManifest(
      draft.dseInvocation, artifacts, blobs);
  if (!invocation)
    return reject(ApplicationActivationDecisionErrorReason::InvocationMismatch,
                  "DSE invocation failed strict import: " +
                      llvm::toString(invocation.takeError()));
  // An invocation that stopped because its bounded completion policy was
  // satisfied still owns a verified Mapping, so it can activate. Only that
  // reason is admitted here: a timeout, an unestablished proof, or any other
  // incompleteness leaves the selected Mapping unproven and stays rejected.
  llvm::ArrayRef<ArtifactRootReference> ownedMappings;
  if (const auto *selected =
          std::get_if<dse::InvocationCompletedSelection>(&invocation->outcome()))
    ownedMappings = selected->selected;
  else if (const auto *incomplete =
               std::get_if<dse::InvocationIncomplete>(&invocation->outcome())) {
    const auto *generatorReason =
        std::get_if<dse::CandidateGeneratorIncompleteReason>(
            &incomplete->reason);
    if (!generatorReason ||
        *generatorReason !=
            dse::CandidateGeneratorIncompleteReason::SemanticLimitReached)
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          "DSE invocation did not complete with a selection: " +
              dse::toString(incomplete->reason));
    ownedMappings = incomplete->retainedArtifacts;
  } else
    return reject(ApplicationActivationDecisionErrorReason::InvocationMismatch,
                  "DSE invocation found no feasible candidate");
  if (!llvm::is_contained(ownedMappings, draft.selectedMapping))
    return reject(ApplicationActivationDecisionErrorReason::InvocationMismatch,
                  "DSE invocation does not own the selected Mapping");
  for (const auto &[owned, subject] :
       {std::pair<const ArtifactRootReference &, llvm::StringRef>{
            draft.sourceProgram, "source program"},
        {draft.fabric, "Fabric"},
        {draft.workload, "workload"},
        {draft.runtimeInput, "runtime input"},
        {draft.planning.canonicalDataflow, "CanonicalDataflow"},
        {draft.selectedSystem, "selected System"}})
    if (!llvm::is_contained(invocation->closure().semanticInputs(), owned))
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          "DSE invocation does not own the exact application " + subject);
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       draft.sourceBackedReplayCases)
    if (!llvm::is_contained(invocation->closure().semanticInputs(),
                            replay.workload) ||
        !llvm::is_contained(invocation->closure().semanticInputs(),
                            replay.runtimeInput))
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          "DSE invocation omits a source-backed replay input");
  for (const dse::JointDesignInvocationManifestReference &supporting :
       draft.supportingDseInvocations) {
    auto imported =
        dse::importJointDesignInvocationManifest(supporting, artifacts, blobs);
    if (!imported)
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          "supporting DSE invocation failed strict import: " +
              llvm::toString(imported.takeError()));
  }

  auto spatialMappings =
      mapping->view().executionBindings().spatialMappingImports();
  auto evidenceJoin = detail::resolveApplicationRuntimeEvidenceJoin(
      draft.runtimeEvidence, draft.oracleEvidence,
      draft.planning.canonicalDataflow, spatialMappings,
      draft.sourceBackedReplayCases, artifacts, blobs);
  if (!evidenceJoin)
    return evidenceJoin.takeError();
  return llvm::Error::success();
}

} // namespace

char ApplicationActivationDecisionError::ID = 0;

void ApplicationActivationDecisionError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code ApplicationActivationDecisionError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<ApplicationActivationDecision>
ApplicationActivationDecision::get(ApplicationActivationDecisionDraft draft,
                                   const ArtifactStore &artifacts,
                                   const BlobStore &blobs) {
  if (llvm::Error error =
          canonicalizeReplayCases(draft.sourceBackedReplayCases))
    return std::move(error);
  if (llvm::Error error =
          canonicalizePlanningRoots(draft.planning.ownedProtocolRoots))
    return std::move(error);
  if (llvm::Error error =
          canonicalizeScheduleHints(draft.selectedScheduleHints))
    return std::move(error);
  if (llvm::Error error = canonicalizeSupportingInvocations(
          draft.supportingDseInvocations, draft.dseInvocation))
    return std::move(error);
  if (llvm::Error error =
          canonicalizeRoots(draft.runtimeEvidence, "runtime Evidence"))
    return std::move(error);
  if (llvm::Error error =
          canonicalizeRoots(draft.oracleEvidence, "oracle Evidence"))
    return std::move(error);
  if (llvm::Error error = canonicalizeRoots(draft.hardwareMutationRepairRecords,
                                            "hardware mutation repair records"))
    return std::move(error);
  if (llvm::Error error = validateDecision(draft, artifacts, blobs))
    return std::move(error);
  std::vector<std::uint8_t> encoded = encodeDecision(draft);
  return ApplicationActivationDecision(
      std::move(draft), CanonicalSemanticBytes(std::move(encoded)));
}

llvm::Expected<ApplicationActivationDecisionDependencyProjection>
projectApplicationActivationDecisionDependencies(
    const ApplicationActivationDecision &decision,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ApplicationActivationDecisionDependencyProjection projection;
  for (const ArtifactRootReference *root :
       {&decision.sourceProgram(), &decision.fabric(), &decision.workload(),
        &decision.runtimeInput(), &decision.dseInvocation().resolvedConfig(),
        &decision.planning().structuredProgram,
        &decision.planning().canonicalDataflow, &decision.selectedSystem(),
        &decision.selectedMapping()})
    if (llvm::Error error = addDependencyRoot(projection, *root, artifacts))
      return std::move(error);
  for (const sim::SourceBackedDfgReplayCaseReference &replay :
       decision.sourceBackedReplayCases())
    for (const ArtifactRootReference *root :
         {&replay.workload, &replay.runtimeInput})
      if (llvm::Error error = addDependencyRoot(projection, *root, artifacts))
        return std::move(error);
  for (const ArtifactRootReference &evidence : decision.runtimeEvidence())
    if (llvm::Error error =
            addEvidenceDependencies(projection, evidence, artifacts))
      return std::move(error);
  for (const ArtifactRootReference &repair :
       decision.hardwareMutationRepairRecords())
    if (llvm::Error error = addDependencyRoot(projection, repair, artifacts))
      return std::move(error);
  auto invocation = dse::importJointDesignInvocationManifest(
      decision.dseInvocation(), artifacts, blobs);
  if (!invocation)
    return reject(ApplicationActivationDecisionErrorReason::InvocationMismatch,
                  "DSE invocation failed strict dependency projection: " +
                      llvm::toString(invocation.takeError()));
  if (llvm::Error error =
          addInvocationDependencies(projection, *invocation, artifacts))
    return std::move(error);
  projection.blobs.push_back(decision.dseInvocation().blob());
  for (const dse::JointDesignInvocationManifestReference &supporting :
       decision.supportingDseInvocations()) {
    if (llvm::Error error = addDependencyRoot(
            projection, supporting.resolvedConfig(), artifacts))
      return std::move(error);
    auto imported =
        dse::importJointDesignInvocationManifest(supporting, artifacts, blobs);
    if (!imported)
      return reject(
          ApplicationActivationDecisionErrorReason::InvocationMismatch,
          "supporting DSE invocation failed strict dependency projection: " +
              llvm::toString(imported.takeError()));
    if (llvm::Error error =
            addInvocationDependencies(projection, *imported, artifacts))
      return std::move(error);
    projection.blobs.push_back(supporting.blob());
  }
  llvm::sort(projection.artifacts, artifactRootReferenceLess);
  projection.artifacts.erase(
      std::unique(projection.artifacts.begin(), projection.artifacts.end()),
      projection.artifacts.end());
  llvm::sort(projection.blobs,
             [](const BlobDigest &lhs, const BlobDigest &rhs) {
               return lhs.bytes() < rhs.bytes();
             });
  projection.blobs.erase(
      std::unique(projection.blobs.begin(), projection.blobs.end()),
      projection.blobs.end());
  return projection;
}

llvm::Expected<FinalizedApplicationActivationDecision>
publishApplicationActivationDecision(ApplicationActivationDecision decision,
                                     const ArtifactStore &artifacts) {
  auto identity = artifacts.put(applicationActivationDecisionSchema,
                                decision.canonicalBytes());
  if (!identity)
    return identity.takeError();
  const ArtifactIdentity expected = finalizeArtifactIdentity(
      applicationActivationDecisionSchema, decision.canonicalBytes());
  if (*identity != expected)
    return reject(
        ApplicationActivationDecisionErrorReason::NonCanonicalEncoding,
        "ArtifactStore returned a foreign activation decision identity");
  ArtifactRootReference reference{
      applicationActivationDecisionSchema.identity.str(),
      applicationActivationDecisionSchema.version, std::move(*identity)};
  return FinalizedApplicationActivationDecision(std::move(reference),
                                                std::move(decision));
}

llvm::Expected<FinalizedApplicationActivationDecision>
importApplicationActivationDecision(const ArtifactRootReference &reference,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs) {
  if (reference.schemaIdentity !=
          applicationActivationDecisionSchema.identity ||
      reference.schemaVersion != applicationActivationDecisionSchema.version)
    return reject(ApplicationActivationDecisionErrorReason::ForeignSchema,
                  "foreign Application activation decision reference schema");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto draft = decodeDecision(bytes->bytes(), artifacts, blobs);
  if (!draft)
    return draft.takeError();
  auto decision =
      ApplicationActivationDecision::get(std::move(*draft), artifacts, blobs);
  if (!decision)
    return decision.takeError();
  if (decision->canonicalBytes().bytes() != bytes->bytes())
    return reject(
        ApplicationActivationDecisionErrorReason::NonCanonicalEncoding,
        "Application activation decision encoding is not canonical");
  const ArtifactIdentity expected = finalizeArtifactIdentity(
      applicationActivationDecisionSchema, decision->canonicalBytes());
  if (expected != reference.artifact)
    return reject(
        ApplicationActivationDecisionErrorReason::NonCanonicalEncoding,
        "Application activation decision identity is not canonical");
  return FinalizedApplicationActivationDecision(reference,
                                                std::move(*decision));
}

} // namespace loom::application
