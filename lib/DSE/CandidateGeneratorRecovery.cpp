#include "DSE/CandidateGeneratorRecovery.h"

#include "CandidateGeneratorCanonical.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr std::uint64_t maximumRecordBytes = 64ULL * 1024ULL * 1024ULL;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "candidate_generator_recovery_invalid: " +
                                     message);
}

class Encoder final {
public:
  void u32(std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
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
      return invalid("truncated " + field);
    llvm::ArrayRef<std::uint8_t> value = bytes_.slice(offset_, width);
    offset_ += width;
    return value;
  }

  llvm::Expected<std::vector<std::uint8_t>> bytes(llvm::StringRef field) {
    auto width = u64((field + " length").str());
    if (!width)
      return width.takeError();
    if (*width > maximumRecordBytes ||
        *width > std::numeric_limits<std::size_t>::max())
      return invalid(field + " length exceeds the recovery-record bound");
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

  llvm::Expected<std::size_t> count(llvm::StringRef field,
                                    std::size_t minimumWidth) {
    auto value = u64(field);
    if (!value)
      return value.takeError();
    if (*value > maximumRecordBytes ||
        *value > std::numeric_limits<std::size_t>::max() ||
        (minimumWidth != 0 && *value > remaining() / minimumWidth))
      return invalid(field + " exceeds the recovery-record bound");
    return static_cast<std::size_t>(*value);
  }

  llvm::Expected<ArtifactRootReference> root(llvm::StringRef field) {
    auto decoded =
        decodeArtifactRootReferencePrefix(bytes_.drop_front(offset_));
    if (!decoded)
      return invalid(field + ": " + llvm::toString(decoded.takeError()));
    offset_ += decoded->byteCount;
    return std::move(decoded->reference);
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }
  bool empty() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

void encodeWorkUnit(Encoder &encoder, const WorkUnitKey &key) {
  encoder.u64(key.planNodeOrdinal());
  encoder.text(key.descriptor().ownerRegistryIdentity());
  encoder.u32(key.descriptor().ownerRegistryVersion().major);
  encoder.u32(key.descriptor().ownerRegistryVersion().minor);
  encoder.u32(key.descriptor().ownerLocalKind());
  encoder.u64(key.stableOrdinal());
}

llvm::Expected<WorkUnitKey> decodeWorkUnit(Decoder &decoder) {
  auto node = decoder.u64("work unit plan node");
  if (!node)
    return node.takeError();
  auto owner = decoder.text("work unit owner registry");
  if (!owner)
    return owner.takeError();
  auto major = decoder.u32("work unit owner major version");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("work unit owner minor version");
  if (!minor)
    return minor.takeError();
  auto kind = decoder.u32("work unit owner-local kind");
  if (!kind)
    return kind.takeError();
  auto ordinal = decoder.u64("work unit stable ordinal");
  if (!ordinal)
    return ordinal.takeError();
  auto descriptor = WorkUnitDescriptorRef::get(*owner, {*major, *minor}, *kind);
  if (!descriptor)
    return descriptor.takeError();
  return WorkUnitKey::get(*node, std::move(*descriptor), *ordinal);
}

void encodeRoots(Encoder &encoder,
                 llvm::ArrayRef<ArtifactRootReference> roots) {
  encoder.u64(roots.size());
  for (const ArtifactRootReference &root : roots)
    encoder.root(root);
}

llvm::Expected<std::vector<ArtifactRootReference>>
decodeRoots(Decoder &decoder, llvm::StringRef field) {
  constexpr std::size_t minimumRootWidth =
      3 * sizeof(std::uint32_t) + ArtifactIdentity::byteSize;
  auto count = decoder.count((field + " count").str(), minimumRootWidth);
  if (!count)
    return count.takeError();
  std::vector<ArtifactRootReference> roots;
  roots.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto root = decoder.root((field + " root").str());
    if (!root)
      return root.takeError();
    roots.push_back(std::move(*root));
  }
  return roots;
}

void encodeOutputs(Encoder &encoder,
                   llvm::ArrayRef<CandidateGeneratorOutputBinding> outputs) {
  encoder.u64(outputs.size());
  for (const CandidateGeneratorOutputBinding &output : outputs) {
    encoder.u32(output.slot.ordinal());
    encodeRoots(encoder, output.artifacts);
  }
}

llvm::Expected<std::vector<CandidateGeneratorOutputBinding>>
decodeOutputs(Decoder &decoder) {
  constexpr std::size_t minimumOutputBindingWidth =
      sizeof(std::uint32_t) + sizeof(std::uint64_t);
  auto count = decoder.count("output binding count", minimumOutputBindingWidth);
  if (!count)
    return count.takeError();
  std::vector<CandidateGeneratorOutputBinding> outputs;
  outputs.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto slot = decoder.u32("output binding slot");
    if (!slot)
      return slot.takeError();
    auto roots = decodeRoots(decoder, "output binding");
    if (!roots)
      return roots.takeError();
    outputs.push_back(
        {CandidateGeneratorOutputSlotRef(*slot), std::move(*roots)});
  }
  return outputs;
}

void encodeLineage(Encoder &encoder,
                   llvm::ArrayRef<CandidateGeneratorLineageEdge> edges) {
  encoder.u64(edges.size());
  for (const CandidateGeneratorLineageEdge &edge : edges) {
    encoder.u32(static_cast<std::uint32_t>(edge.kind));
    encoder.u32(edge.outputSlot.ordinal());
    encoder.root(edge.output);
    encodeRoots(encoder, edge.parents);
    encoder.bytes(edge.ownerPayload);
  }
}

llvm::Expected<std::vector<CandidateGeneratorLineageEdge>>
decodeLineage(Decoder &decoder) {
  constexpr std::size_t minimumRootWidth =
      3 * sizeof(std::uint32_t) + ArtifactIdentity::byteSize;
  constexpr std::size_t minimumLineageEdgeWidth =
      2 * sizeof(std::uint32_t) + minimumRootWidth + 2 * sizeof(std::uint64_t);
  auto count = decoder.count("lineage edge count", minimumLineageEdgeWidth);
  if (!count)
    return count.takeError();
  std::vector<CandidateGeneratorLineageEdge> edges;
  edges.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto kind = decoder.u32("lineage edge kind");
    if (!kind)
      return kind.takeError();
    if (*kind > static_cast<std::uint32_t>(
                    CandidateGeneratorLineageEdgeKind::CandidateDecision))
      return invalid("lineage edge kind is unknown");
    auto slot = decoder.u32("lineage edge output slot");
    if (!slot)
      return slot.takeError();
    auto output = decoder.root("lineage edge output");
    if (!output)
      return output.takeError();
    auto parents = decodeRoots(decoder, "lineage edge parent");
    if (!parents)
      return parents.takeError();
    auto payload = decoder.bytes("lineage edge owner payload");
    if (!payload)
      return payload.takeError();
    edges.push_back({static_cast<CandidateGeneratorLineageEdgeKind>(*kind),
                     CandidateGeneratorOutputSlotRef(*slot), std::move(*output),
                     std::move(*parents), std::move(*payload)});
  }
  return edges;
}

void encodeWorkSummary(
    Encoder &encoder,
    llvm::ArrayRef<CandidateGeneratorWorkUnitSummary> summary) {
  encoder.u64(summary.size());
  for (const CandidateGeneratorWorkUnitSummary &unit : summary) {
    encoder.u32(unit.unit.ordinal());
    encoder.u64(unit.planned);
    encoder.u64(unit.consumed);
  }
}

llvm::Expected<std::vector<CandidateGeneratorWorkUnitSummary>>
decodeWorkSummary(Decoder &decoder) {
  constexpr std::size_t encodedUnitWidth =
      sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t);
  auto count = decoder.count("work summary count", encodedUnitWidth);
  if (!count)
    return count.takeError();
  std::vector<CandidateGeneratorWorkUnitSummary> summary;
  summary.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto unit = decoder.u32("work summary unit");
    if (!unit)
      return unit.takeError();
    auto planned = decoder.u64("work summary planned count");
    if (!planned)
      return planned.takeError();
    auto consumed = decoder.u64("work summary consumed count");
    if (!consumed)
      return consumed.takeError();
    summary.push_back(
        {CandidateGeneratorWorkUnitRef(*unit), *planned, *consumed});
  }
  return summary;
}

std::vector<std::uint8_t>
encodeRecord(const DseRunKey &runKey, const WorkUnitKey &workUnit,
             llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
             const ResolvedCandidateGeneratorBinding &binding,
             const CandidateGeneratorProviderResult &result,
             SchemaVersion schemaVersion =
                 candidateGeneratorFinalizedWorkRecordSchemaVersion) {
  Encoder encoder;
  encoder.text(candidateGeneratorFinalizedWorkRecordSchemaIdentity);
  encoder.u32(schemaVersion.major);
  encoder.u32(schemaVersion.minor);
  encoder.fixed(runKey.bytes());
  encodeWorkUnit(encoder, workUnit);
  encoder.bytes(detail::encodeCanonicalCandidateGeneratorInputBindings(inputs));
  encoder.bytes(
      detail::encodeCanonicalResolvedCandidateGeneratorBinding(binding));
  std::visit(
      [&](const auto &outcome) {
        using T = std::decay_t<decltype(outcome)>;
        if constexpr (std::is_same_v<T, CompletedCandidateGeneratorResult>) {
          encoder.u32(0);
          encodeOutputs(encoder, outcome.outputBindings);
          encodeLineage(encoder, outcome.lineageEdges);
        } else if constexpr (std::is_same_v<
                                 T, ProvenInfeasibleCandidateGeneratorResult>) {
          assert(schemaVersion ==
                 candidateGeneratorFinalizedWorkRecordSchemaVersion);
          encoder.u32(2);
          encoder.u32(outcome.proof.kind.ordinal());
          encoder.bytes(outcome.proof.witness);
          encodeOutputs(encoder, outcome.outputBindings);
          encodeLineage(encoder, {});
        } else {
          encoder.u32(1);
          encoder.u32(static_cast<std::uint32_t>(outcome.reason));
          encodeOutputs(encoder, outcome.retainedOutputBindings);
          encodeLineage(encoder, outcome.lineageEdges);
        }
      },
      result.outcome);
  encodeWorkSummary(encoder, result.workSummary);
  return encoder.take();
}

llvm::Expected<CandidateGeneratorProviderResult>
decodeRecord(llvm::ArrayRef<std::uint8_t> bytes, const DseRunKey &runKey,
             const WorkUnitKey &workUnit,
             llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
             const ResolvedCandidateGeneratorBinding &binding) {
  if (bytes.size() > maximumRecordBytes)
    return invalid("recovery record exceeds the size bound");
  Decoder decoder(bytes);
  auto schema = decoder.text("recovery record schema");
  if (!schema)
    return schema.takeError();
  auto major = decoder.u32("recovery record major version");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("recovery record minor version");
  if (!minor)
    return minor.takeError();
  const SchemaVersion sourceVersion{*major, *minor};
  if (*schema != candidateGeneratorFinalizedWorkRecordSchemaIdentity ||
      (sourceVersion !=
           candidateGeneratorLegacyFinalizedWorkRecordSchemaVersion &&
       sourceVersion != candidateGeneratorFinalizedWorkRecordSchemaVersion))
    return invalid("recovery record schema is unsupported");
  auto storedRunKey = decoder.fixed(DseRunKey::byteSize, "run key");
  if (!storedRunKey)
    return storedRunKey.takeError();
  if (*storedRunKey != llvm::ArrayRef<std::uint8_t>(runKey.bytes()))
    return invalid("recovery record belongs to another run key");
  auto storedWorkUnit = decodeWorkUnit(decoder);
  if (!storedWorkUnit)
    return storedWorkUnit.takeError();
  if (!(*storedWorkUnit == workUnit))
    return invalid("recovery record belongs to another WorkUnitKey");
  auto storedInputs = decoder.bytes("generator input closure");
  if (!storedInputs)
    return storedInputs.takeError();
  if (*storedInputs !=
      detail::encodeCanonicalCandidateGeneratorInputBindings(inputs))
    return invalid("recovery record has another generator input closure");
  auto storedBinding = decoder.bytes("resolved generator binding");
  if (!storedBinding)
    return storedBinding.takeError();
  if (*storedBinding !=
      detail::encodeCanonicalResolvedCandidateGeneratorBinding(binding))
    return invalid("recovery record has another resolved generator binding");
  auto outcomeKind = decoder.u32("provider outcome kind");
  if (!outcomeKind)
    return outcomeKind.takeError();
  if (*outcomeKind >
      (sourceVersion == candidateGeneratorLegacyFinalizedWorkRecordSchemaVersion
           ? 1U
           : 2U))
    return invalid("provider outcome kind is unknown");
  std::optional<CandidateGeneratorIncompleteReason> incompleteReason;
  std::optional<CandidateGeneratorInfeasibilityProof> infeasibilityProof;
  if (*outcomeKind == 1) {
    auto reason = decoder.u32("incomplete outcome reason");
    if (!reason)
      return reason.takeError();
    if (*reason > static_cast<std::uint32_t>(
                      CandidateGeneratorIncompleteReason::CancelledOrTimeout))
      return invalid("incomplete outcome reason is unknown");
    incompleteReason = static_cast<CandidateGeneratorIncompleteReason>(*reason);
  } else if (*outcomeKind == 2) {
    auto kind = decoder.u32("infeasibility proof kind");
    if (!kind)
      return kind.takeError();
    auto witness = decoder.bytes("infeasibility proof witness");
    if (!witness)
      return witness.takeError();
    infeasibilityProof.emplace(CandidateGeneratorInfeasibilityProof{
        CandidateGeneratorInfeasibilityProofKindRef(*kind),
        std::move(*witness)});
  }
  auto outputs = decodeOutputs(decoder);
  if (!outputs)
    return outputs.takeError();
  auto lineage = decodeLineage(decoder);
  if (!lineage)
    return lineage.takeError();
  auto workSummary = decodeWorkSummary(decoder);
  if (!workSummary)
    return workSummary.takeError();
  if (!decoder.empty())
    return invalid("recovery record has trailing bytes");
  if (incompleteReason)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            *incompleteReason, std::move(*outputs), std::move(*lineage)},
        std::move(*workSummary)};
  if (infeasibilityProof)
    return CandidateGeneratorProviderResult{
        ProvenInfeasibleCandidateGeneratorResult{
            std::move(*outputs), std::move(*infeasibilityProof)},
        std::move(*workSummary)};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{std::move(*outputs),
                                        std::move(*lineage)},
      std::move(*workSummary)};
}

} // namespace

llvm::Expected<BlobDigest> publishCandidateGeneratorFinalizedWorkRecord(
    const DseRunKey &runKey, const WorkUnitKey &workUnit,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const CandidateGeneratorProviderResult &result,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  CandidateGeneratorProviderResult canonical = result;
  if (llvm::Error error = validateCandidateGeneratorProviderResult(
          inputs, binding, canonical, artifactStore, blobStore))
    return std::move(error);
  const std::vector<std::uint8_t> bytes =
      encodeRecord(runKey, workUnit, inputs, binding, canonical);
  if (bytes.size() > maximumRecordBytes)
    return invalid("recovery record exceeds the size bound");
  return blobStore.put(bytes);
}

llvm::Expected<CandidateGeneratorProviderResult>
importCandidateGeneratorFinalizedWorkRecord(
    SchemaVersion recordVersion, const BlobDigest &recordDigest,
    const DseRunKey &runKey,
    const WorkUnitKey &workUnit,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto bytes = blobStore.get(recordDigest);
  if (!bytes)
    return bytes.takeError();
  auto result = decodeRecord(*bytes, runKey, workUnit, inputs, binding);
  if (!result)
    return result.takeError();
  if (llvm::Error error = validateCandidateGeneratorProviderResult(
          inputs, binding, *result, artifactStore, blobStore))
    return std::move(error);
  if (recordVersion !=
          candidateGeneratorLegacyFinalizedWorkRecordSchemaVersion &&
      recordVersion != candidateGeneratorFinalizedWorkRecordSchemaVersion)
    return invalid("recovery record reference has an unsupported version");
  if (recordVersion ==
          candidateGeneratorLegacyFinalizedWorkRecordSchemaVersion &&
      std::holds_alternative<ProvenInfeasibleCandidateGeneratorResult>(
          result->outcome))
    return invalid(
        "legacy recovery record reference cannot carry an infeasibility proof");
  if (encodeRecord(runKey, workUnit, inputs, binding, *result, recordVersion) !=
      *bytes)
    return invalid("recovery record bytes are not canonical");
  return result;
}

} // namespace loom::dse
