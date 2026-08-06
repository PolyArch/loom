#include "DSE/InvocationManifest.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/Evidence.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {
namespace {

constexpr char runKeyDomain[] = "loom.dse.run_key.1.0\0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invocation_manifest_invalid: " + message);
}

bool isCanonicalAscii(llvm::StringRef value) {
  return !value.empty() && llvm::all_of(value, [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
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
    auto raw = fixed(4, field);
    if (!raw)
      return raw.takeError();
    std::uint32_t value = 0;
    for (std::uint8_t byte : *raw)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    auto raw = fixed(8, field);
    if (!raw)
      return raw.takeError();
    std::uint64_t value = 0;
    for (std::uint8_t byte : *raw)
      value = (value << 8) | byte;
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> fixed(std::size_t width,
                                                     llvm::StringRef field) {
    if (width > bytes_.size() - offset_)
      return invalid("truncated " + field);
    llvm::ArrayRef<std::uint8_t> value = bytes_.slice(offset_, width);
    offset_ += width;
    return value;
  }

  llvm::Expected<std::vector<std::uint8_t>> bytes(llvm::StringRef field) {
    auto width = u64((field + " length").str());
    if (!width)
      return width.takeError();
    if (*width > std::numeric_limits<std::size_t>::max())
      return invalid(field + " length exceeds host size_t");
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
      return invalid(field + " is not representable by the remaining wire");
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
  bool atEnd() const { return offset_ == bytes_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

void canonicalizeRoots(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

bool rootsAreCanonical(llvm::ArrayRef<ArtifactRootReference> references) {
  return llvm::is_sorted(references, artifactRootReferenceLess) &&
         std::adjacent_find(references.begin(), references.end()) ==
             references.end();
}

bool isEvidence(const ArtifactRootReference &reference) {
  return reference.schemaIdentity ==
             evaluation::EvaluationEvidence::artifactSchema.identity &&
         reference.schemaVersion ==
             evaluation::EvaluationEvidence::artifactSchema.version;
}

llvm::Error validateStoredRoots(llvm::ArrayRef<ArtifactRootReference> roots,
                                llvm::StringRef field,
                                const ArtifactStore &store,
                                bool requireEvidence) {
  if (!rootsAreCanonical(roots))
    return invalid(field + " must be a canonical set");
  for (const ArtifactRootReference &root : roots) {
    if (requireEvidence && !isEvidence(root))
      return invalid(field + " contains a non-Evidence reference");
    if (requireEvidence) {
      auto request =
          evaluation::importEvaluationEvidenceRequestReference(root, store);
      if (!request)
        return request.takeError();
      if (request->schemaIdentity !=
              evaluation::EvaluationRequest::artifactSchema.identity ||
          request->schemaVersion !=
              evaluation::EvaluationRequest::artifactSchema.version)
        return invalid(field + " contains Evidence with a foreign Request");
      auto storedRequest = store.get(
          evaluation::EvaluationRequest::artifactSchema, request->artifact);
      if (!storedRequest)
        return storedRequest.takeError();
    } else {
      auto stored = store.get(root);
      if (!stored)
        return stored.takeError();
    }
  }
  return llvm::Error::success();
}

void encodeRoots(Encoder &encoder,
                 llvm::ArrayRef<ArtifactRootReference> references) {
  encoder.u64(references.size());
  for (const ArtifactRootReference &reference : references)
    encoder.root(reference);
}

llvm::Expected<std::vector<ArtifactRootReference>>
decodeRoots(Decoder &decoder, llvm::StringRef field) {
  auto count = decoder.count((field + " count").str());
  if (!count)
    return count.takeError();
  std::vector<ArtifactRootReference> references;
  references.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto reference = decoder.root((field + " reference").str());
    if (!reference)
      return reference.takeError();
    references.push_back(std::move(*reference));
  }
  return references;
}

llvm::Expected<ArtifactIdentity> decodeIdentity(Decoder &decoder,
                                                llvm::StringRef field) {
  auto bytes = decoder.fixed(ArtifactIdentity::byteSize, field);
  if (!bytes)
    return bytes.takeError();
  return ArtifactIdentity::fromBytes(*bytes);
}

llvm::Expected<ComponentViewDigest> decodeDigest(Decoder &decoder,
                                                 llvm::StringRef field) {
  auto bytes = decoder.fixed(ComponentViewDigest::byteSize, field);
  if (!bytes)
    return bytes.takeError();
  return ComponentViewDigest::fromBytes(*bytes);
}

llvm::Expected<DseRunKey> decodeRunKey(Decoder &decoder,
                                       llvm::StringRef field) {
  auto bytes = decoder.fixed(DseRunKey::byteSize, field);
  if (!bytes)
    return bytes.takeError();
  return DseRunKey::fromBytes(*bytes);
}

std::vector<std::uint8_t>
runKeyPreimage(const DseProducerSemanticBuildIdentity &producer,
               llvm::ArrayRef<ArtifactRootReference> semanticInputs,
               const ArtifactIdentity &resolvedConfigIdentity,
               llvm::ArrayRef<ArtifactRootReference> preexistingEvidence) {
  Encoder encoder;
  encoder.fixed(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(runKeyDomain),
      sizeof(runKeyDomain) - 1));
  encoder.text(producer.spelling());
  encodeRoots(encoder, semanticInputs);
  encoder.fixed(resolvedConfigIdentity.bytes());
  encodeRoots(encoder, preexistingEvidence);
  return encoder.take();
}

llvm::Error validateConfigObject(const ResolvedConfig &config,
                                 const ArtifactStore &store) {
  const ArtifactIdentity identity = loom::resolvedConfigIdentity(config);
  auto stored = store.get(ResolvedConfig::artifactSchema, identity);
  if (!stored)
    return stored.takeError();
  if (stored->bytes() != canonicalResolvedConfigBytes(config).bytes())
    return invalid("stored ResolvedConfig bytes differ from the supplied "
                   "exact configuration");
  return llvm::Error::success();
}

void canonicalizeObligations(
    std::vector<EvidenceObligationTemplateRef> &obligations) {
  llvm::sort(obligations, [](EvidenceObligationTemplateRef lhs,
                             EvidenceObligationTemplateRef rhs) {
    return lhs.ordinal() < rhs.ordinal();
  });
  obligations.erase(std::unique(obligations.begin(), obligations.end()),
                    obligations.end());
}

llvm::Error canonicalizeOutcome(InvocationControllerOutcome &outcome) {
  return std::visit(
      [](auto &value) -> llvm::Error {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, InvocationCompletedSelection>) {
          canonicalizeRoots(value.selected);
          canonicalizeRoots(value.satisfiedEvidence);
          if (value.selected.empty())
            return invalid("CompletedSelection requires a nonempty selected "
                           "Artifact set");
        } else if constexpr (std::is_same_v<
                                 T, InvocationCompletedNoFeasibleCandidate>) {
          canonicalizeRoots(value.satisfiedEvidence);
        } else {
          canonicalizeObligations(value.unsatisfiedObligations);
          canonicalizeRoots(value.retainedArtifacts);
          canonicalizeRoots(value.retainedEvidence);
        }
        return llvm::Error::success();
      },
      outcome);
}

llvm::Error validateIncompleteReason(const DsePlanIncompleteReason &reason) {
  return std::visit(
      [](const auto &value) -> llvm::Error {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, CandidateGeneratorIncompleteReason>) {
          if (static_cast<std::uint32_t>(value) >
              static_cast<std::uint32_t>(
                  CandidateGeneratorIncompleteReason::CancelledOrTimeout))
            return invalid("unknown candidate-generator incomplete reason");
        } else if constexpr (std::is_same_v<
                                 T, PromotionAcquisitionIncompleteReason>) {
          if (static_cast<std::uint32_t>(value) >
              static_cast<std::uint32_t>(
                  PromotionAcquisitionIncompleteReason::Unsupported))
            return invalid("unknown acquisition incomplete reason");
        } else {
          if (static_cast<std::uint32_t>(value) >
              static_cast<std::uint32_t>(
                  IncompleteSelectionReason::ObjectiveUnavailable))
            return invalid("unknown selection incomplete reason");
        }
        return llvm::Error::success();
      },
      reason);
}

llvm::Error validateGenerateRecord(const InvocationGenerateRecord &record,
                                   const ResolvedGeneratePlanNode &planNode,
                                   const ArtifactStore &store) {
  if (record.invocation.planNodeOrdinal != record.workSummary.planNodeOrdinal)
    return invalid("Generate record and work summary name different plan "
                   "nodes");
  const ResolvedCandidateGeneratorBinding &binding =
      record.invocation.generatorBinding;
  if (binding.descriptorRef() != planNode.descriptorRef() ||
      binding.canonicalConfigBytes() != planNode.canonicalConfigBytes() ||
      binding.configDigest() != planNode.configDigest())
    return invalid("Generate record does not match its exact resolved plan "
                   "node");
  if (llvm::Error error = validateCandidateGeneratorWorkSummary(
          binding.descriptorRef(), record.workSummary.units))
    return error;
  return validateCanonicalCandidateGeneratorInvocation(
      record.invocation.inputBindings, binding,
      record.invocation.outputBindings, record.invocation.lineageEdges,
      record.completed, store);
}

llvm::Error
validateGenerateSequence(llvm::ArrayRef<InvocationGenerateRecord> records,
                         const ResolvedDseConfigView &view,
                         const InvocationControllerOutcome &outcome,
                         const ArtifactStore &store) {
  const auto *incomplete = std::get_if<InvocationIncomplete>(&outcome);
  if (incomplete && incomplete->planNodeOrdinal >= view.plan().nodes().size())
    return invalid("Incomplete outcome references an unknown plan node");

  std::size_t recordOrdinal = 0;
  for (std::size_t nodeOrdinal = 0; nodeOrdinal != view.plan().nodes().size();
       ++nodeOrdinal) {
    const bool nodeExecuted =
        !incomplete || nodeOrdinal <= incomplete->planNodeOrdinal;
    if (!nodeExecuted)
      break;
    const auto *generate = std::get_if<ResolvedGeneratePlanNode>(
        &view.plan().nodes()[nodeOrdinal]);
    if (!generate)
      continue;
    if (recordOrdinal == records.size())
      return invalid("Manifest omits an executed Generate plan node");
    const InvocationGenerateRecord &record = records[recordOrdinal++];
    if (record.invocation.planNodeOrdinal != nodeOrdinal)
      return invalid("Generate records do not follow exact PlanNodeRef order");
    const bool shouldComplete =
        !incomplete || nodeOrdinal < incomplete->planNodeOrdinal;
    if (record.completed != shouldComplete)
      return invalid("Generate completion state disagrees with the controller "
                     "outcome");
    if (llvm::Error error = validateGenerateRecord(record, *generate, store))
      return error;
  }
  if (recordOrdinal != records.size())
    return invalid("Manifest contains an unexecuted Generate plan node");
  if (incomplete) {
    const bool failedAtGenerate =
        std::holds_alternative<ResolvedGeneratePlanNode>(
            view.plan().nodes()[incomplete->planNodeOrdinal]);
    if (failedAtGenerate && (records.empty() || records.back().completed ||
                             records.back().invocation.planNodeOrdinal !=
                                 incomplete->planNodeOrdinal))
      return invalid("Incomplete Generate node lacks its partial invocation "
                     "record");
  } else if (llvm::any_of(records, [](const InvocationGenerateRecord &record) {
               return !record.completed;
             })) {
    return invalid("completed controller outcome contains an incomplete "
                   "Generate record");
  }
  return llvm::Error::success();
}

bool containsRoot(llvm::ArrayRef<ArtifactRootReference> roots,
                  const ArtifactRootReference &root) {
  return std::binary_search(roots.begin(), roots.end(), root,
                            artifactRootReferenceLess);
}

llvm::Error
validateOutcome(const InvocationControllerOutcome &outcome,
                llvm::ArrayRef<InvocationGenerateRecord> generateRecords,
                const DseRunClosure &closure, const ResolvedDseConfigView &view,
                const ArtifactStore &store) {
  std::vector<ArtifactRootReference> candidateClosure(
      closure.semanticInputs().begin(), closure.semanticInputs().end());
  for (const InvocationGenerateRecord &record : generateRecords) {
    for (const CandidateGeneratorInputBinding &binding :
         record.invocation.inputBindings)
      candidateClosure.insert(candidateClosure.end(), binding.artifacts.begin(),
                              binding.artifacts.end());
    for (const CandidateGeneratorOutputBinding &binding :
         record.invocation.outputBindings)
      candidateClosure.insert(candidateClosure.end(), binding.artifacts.begin(),
                              binding.artifacts.end());
  }
  canonicalizeRoots(candidateClosure);

  return std::visit(
      [&](const auto &value) -> llvm::Error {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, InvocationCompletedSelection>) {
          if (value.selected.empty() || !rootsAreCanonical(value.selected))
            return invalid("CompletedSelection selected set is not nonempty "
                           "and canonical");
          if (llvm::Error error = validateStoredRoots(
                  value.selected, "selected Artifacts", store, false))
            return error;
          for (const ArtifactRootReference &selected : value.selected)
            if (!containsRoot(candidateClosure, selected))
              return invalid("selected Artifact is outside the invocation "
                             "candidate closure");
          return validateStoredRoots(value.satisfiedEvidence,
                                     "satisfied Evidence", store, true);
        } else if constexpr (std::is_same_v<
                                 T, InvocationCompletedNoFeasibleCandidate>) {
          return validateStoredRoots(value.satisfiedEvidence,
                                     "satisfied Evidence", store, true);
        } else {
          if (llvm::Error error = validateIncompleteReason(value.reason))
            return error;
          for (EvidenceObligationTemplateRef obligation :
               value.unsatisfiedObligations)
            if (obligation.ordinal() >=
                view.evidenceObligationTemplates().size())
              return invalid("Incomplete outcome references an unknown "
                             "Evidence obligation");
          if (llvm::Error error = validateStoredRoots(
                  value.retainedArtifacts, "retained Artifacts", store, false))
            return error;
          for (const ArtifactRootReference &retained : value.retainedArtifacts)
            if (isEvidence(retained))
              return invalid("retained Artifact set contains Evidence");
          return validateStoredRoots(value.retainedEvidence,
                                     "retained Evidence", store, true);
        }
      },
      outcome);
}

void encodeIncompleteReason(Encoder &encoder,
                            const DsePlanIncompleteReason &reason) {
  std::visit(
      [&](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, CandidateGeneratorIncompleteReason>)
          encoder.u32(0);
        else if constexpr (std::is_same_v<T,
                                          PromotionAcquisitionIncompleteReason>)
          encoder.u32(1);
        else
          encoder.u32(2);
        encoder.u32(static_cast<std::uint32_t>(value));
      },
      reason);
}

llvm::Expected<DsePlanIncompleteReason>
decodeIncompleteReason(Decoder &decoder) {
  auto family = decoder.u32("incomplete reason family");
  if (!family)
    return family.takeError();
  auto ordinal = decoder.u32("incomplete reason ordinal");
  if (!ordinal)
    return ordinal.takeError();
  switch (*family) {
  case 0:
    if (*ordinal > static_cast<std::uint32_t>(
                       CandidateGeneratorIncompleteReason::CancelledOrTimeout))
      return invalid("unknown candidate-generator incomplete reason");
    return DsePlanIncompleteReason{
        static_cast<CandidateGeneratorIncompleteReason>(*ordinal)};
  case 1:
    if (*ordinal > static_cast<std::uint32_t>(
                       PromotionAcquisitionIncompleteReason::Unsupported))
      return invalid("unknown acquisition incomplete reason");
    return DsePlanIncompleteReason{
        static_cast<PromotionAcquisitionIncompleteReason>(*ordinal)};
  case 2:
    if (*ordinal > static_cast<std::uint32_t>(
                       IncompleteSelectionReason::ObjectiveUnavailable))
      return invalid("unknown selection incomplete reason");
    return DsePlanIncompleteReason{
        static_cast<IncompleteSelectionReason>(*ordinal)};
  default:
    return invalid("unknown incomplete reason family");
  }
}

void encodeGeneratorBinding(Encoder &encoder,
                            const ResolvedCandidateGeneratorBinding &binding) {
  const CandidateGeneratorDescriptorRef reference = binding.descriptorRef();
  encoder.text(reference.descriptorSchema().identity);
  encoder.u32(reference.descriptorSchema().version.major);
  encoder.u32(reference.descriptorSchema().version.minor);
  encoder.u32(reference.kind().ordinal());
  encoder.bytes(binding.canonicalConfigBytes());
  encoder.fixed(binding.configDigest().bytes());
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
decodeGeneratorBinding(Decoder &decoder) {
  auto schema = decoder.text("generator descriptor schema identity");
  if (!schema)
    return schema.takeError();
  auto major = decoder.u32("generator descriptor schema major");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("generator descriptor schema minor");
  if (!minor)
    return minor.takeError();
  if (*schema != candidateGeneratorDescriptorSchema.identity ||
      SchemaVersion{*major, *minor} !=
          candidateGeneratorDescriptorSchema.version)
    return invalid("unsupported candidate-generator descriptor schema");
  auto kind = decoder.u32("generator descriptor kind");
  if (!kind)
    return kind.takeError();
  auto config = decoder.bytes("generator canonical config");
  if (!config)
    return config.takeError();
  auto digest = decodeDigest(decoder, "generator config digest");
  if (!digest)
    return digest.takeError();
  auto reference = CandidateGeneratorDescriptorRef::get(
      candidateGeneratorDescriptorSchema, CandidateGeneratorKind(*kind));
  if (!reference)
    return reference.takeError();
  return ResolvedCandidateGeneratorBinding::get(*reference, *config, *digest);
}

void encodeInputBindings(
    Encoder &encoder, llvm::ArrayRef<CandidateGeneratorInputBinding> bindings) {
  encoder.u64(bindings.size());
  for (const CandidateGeneratorInputBinding &binding : bindings) {
    encoder.u32(binding.slot.ordinal());
    encodeRoots(encoder, binding.artifacts);
  }
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
decodeInputBindings(Decoder &decoder) {
  auto count = decoder.count("generator input binding count");
  if (!count)
    return count.takeError();
  std::vector<CandidateGeneratorInputBinding> bindings;
  bindings.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto slot = decoder.u32("generator input slot");
    if (!slot)
      return slot.takeError();
    auto roots = decodeRoots(decoder, "generator input artifacts");
    if (!roots)
      return roots.takeError();
    bindings.push_back(
        {CandidateGeneratorInputSlotRef(*slot), std::move(*roots)});
  }
  return bindings;
}

void encodeOutputBindings(
    Encoder &encoder,
    llvm::ArrayRef<CandidateGeneratorOutputBinding> bindings) {
  encoder.u64(bindings.size());
  for (const CandidateGeneratorOutputBinding &binding : bindings) {
    encoder.u32(binding.slot.ordinal());
    encodeRoots(encoder, binding.artifacts);
  }
}

llvm::Expected<std::vector<CandidateGeneratorOutputBinding>>
decodeOutputBindings(Decoder &decoder) {
  auto count = decoder.count("generator output binding count");
  if (!count)
    return count.takeError();
  std::vector<CandidateGeneratorOutputBinding> bindings;
  bindings.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto slot = decoder.u32("generator output slot");
    if (!slot)
      return slot.takeError();
    auto roots = decodeRoots(decoder, "generator output artifacts");
    if (!roots)
      return roots.takeError();
    bindings.push_back(
        {CandidateGeneratorOutputSlotRef(*slot), std::move(*roots)});
  }
  return bindings;
}

void encodeLineageEdges(Encoder &encoder,
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
decodeLineageEdges(Decoder &decoder) {
  auto count = decoder.count("generator lineage edge count");
  if (!count)
    return count.takeError();
  std::vector<CandidateGeneratorLineageEdge> edges;
  edges.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto kind = decoder.u32("generator lineage edge kind");
    if (!kind)
      return kind.takeError();
    if (*kind > static_cast<std::uint32_t>(
                    CandidateGeneratorLineageEdgeKind::CandidateDecision))
      return invalid("unknown generator lineage edge kind");
    auto slot = decoder.u32("generator lineage output slot");
    if (!slot)
      return slot.takeError();
    auto output = decoder.root("generator lineage output");
    if (!output)
      return output.takeError();
    auto parents = decodeRoots(decoder, "generator lineage parents");
    if (!parents)
      return parents.takeError();
    auto payload = decoder.bytes("generator owner lineage payload");
    if (!payload)
      return payload.takeError();
    edges.push_back({static_cast<CandidateGeneratorLineageEdgeKind>(*kind),
                     CandidateGeneratorOutputSlotRef(*slot), std::move(*output),
                     std::move(*parents), std::move(*payload)});
  }
  return edges;
}

void encodeWorkSummary(Encoder &encoder,
                       const GenerateInvocationWorkSummary &summary) {
  encoder.u64(summary.planNodeOrdinal);
  encoder.u64(summary.units.size());
  for (const CandidateGeneratorWorkUnitSummary &unit : summary.units) {
    encoder.u32(unit.unit.ordinal());
    encoder.u64(unit.planned);
    encoder.u64(unit.consumed);
  }
}

llvm::Expected<GenerateInvocationWorkSummary>
decodeWorkSummary(Decoder &decoder) {
  auto planNode = decoder.u64("work summary plan node");
  if (!planNode)
    return planNode.takeError();
  auto count = decoder.count("work summary unit count");
  if (!count)
    return count.takeError();
  std::vector<CandidateGeneratorWorkUnitSummary> units;
  units.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto unit = decoder.u32("work summary unit");
    if (!unit)
      return unit.takeError();
    auto planned = decoder.u64("planned logical work slots");
    if (!planned)
      return planned.takeError();
    auto consumed = decoder.u64("consumed logical work slots");
    if (!consumed)
      return consumed.takeError();
    units.push_back(
        {CandidateGeneratorWorkUnitRef(*unit), *planned, *consumed});
  }
  return GenerateInvocationWorkSummary{*planNode, std::move(units)};
}

void encodeGenerateRecord(Encoder &encoder,
                          const InvocationGenerateRecord &record) {
  encoder.u32(record.completed ? 1 : 0);
  encoder.u64(record.invocation.planNodeOrdinal);
  encodeInputBindings(encoder, record.invocation.inputBindings);
  encodeGeneratorBinding(encoder, record.invocation.generatorBinding);
  encodeOutputBindings(encoder, record.invocation.outputBindings);
  encodeLineageEdges(encoder, record.invocation.lineageEdges);
  encodeWorkSummary(encoder, record.workSummary);
}

llvm::Expected<InvocationGenerateRecord>
decodeGenerateRecord(Decoder &decoder) {
  auto completed = decoder.u32("Generate completion flag");
  if (!completed)
    return completed.takeError();
  if (*completed > 1)
    return invalid("Generate completion flag is not boolean");
  auto planNode = decoder.u64("Generate plan node");
  if (!planNode)
    return planNode.takeError();
  auto inputs = decodeInputBindings(decoder);
  if (!inputs)
    return inputs.takeError();
  auto binding = decodeGeneratorBinding(decoder);
  if (!binding)
    return binding.takeError();
  auto outputs = decodeOutputBindings(decoder);
  if (!outputs)
    return outputs.takeError();
  auto edges = decodeLineageEdges(decoder);
  if (!edges)
    return edges.takeError();
  auto work = decodeWorkSummary(decoder);
  if (!work)
    return work.takeError();
  return InvocationGenerateRecord{
      *completed == 1,
      GenerateInvocationRecord{*planNode, std::move(*inputs),
                               std::move(*binding), std::move(*outputs),
                               std::move(*edges)},
      std::move(*work)};
}

void encodeOutcome(Encoder &encoder,
                   const InvocationControllerOutcome &outcome) {
  std::visit(
      [&](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, InvocationCompletedSelection>) {
          encoder.u32(0);
          encodeRoots(encoder, value.selected);
          encodeRoots(encoder, value.satisfiedEvidence);
        } else if constexpr (std::is_same_v<
                                 T, InvocationCompletedNoFeasibleCandidate>) {
          encoder.u32(1);
          encodeRoots(encoder, value.satisfiedEvidence);
        } else {
          encoder.u32(2);
          encoder.u64(value.planNodeOrdinal);
          encodeIncompleteReason(encoder, value.reason);
          encoder.u64(value.unsatisfiedObligations.size());
          for (EvidenceObligationTemplateRef obligation :
               value.unsatisfiedObligations)
            encoder.u32(obligation.ordinal());
          encodeRoots(encoder, value.retainedArtifacts);
          encodeRoots(encoder, value.retainedEvidence);
        }
      },
      outcome);
}

llvm::Expected<InvocationControllerOutcome> decodeOutcome(Decoder &decoder) {
  auto kind = decoder.u32("controller outcome kind");
  if (!kind)
    return kind.takeError();
  if (*kind == 0) {
    auto selected = decodeRoots(decoder, "selected Artifacts");
    if (!selected)
      return selected.takeError();
    auto evidence = decodeRoots(decoder, "satisfied Evidence");
    if (!evidence)
      return evidence.takeError();
    return InvocationControllerOutcome{InvocationCompletedSelection{
        std::move(*selected), std::move(*evidence)}};
  }
  if (*kind == 1) {
    auto evidence = decodeRoots(decoder, "satisfied Evidence");
    if (!evidence)
      return evidence.takeError();
    return InvocationControllerOutcome{
        InvocationCompletedNoFeasibleCandidate{std::move(*evidence)}};
  }
  if (*kind != 2)
    return invalid("unknown controller outcome kind");
  auto planNode = decoder.u64("incomplete plan node");
  if (!planNode)
    return planNode.takeError();
  auto reason = decodeIncompleteReason(decoder);
  if (!reason)
    return reason.takeError();
  auto count = decoder.count("unsatisfied obligation count");
  if (!count)
    return count.takeError();
  std::vector<EvidenceObligationTemplateRef> obligations;
  obligations.reserve(*count);
  for (std::size_t index = 0; index != *count; ++index) {
    auto ordinal = decoder.u32("unsatisfied obligation ordinal");
    if (!ordinal)
      return ordinal.takeError();
    obligations.emplace_back(*ordinal);
  }
  auto artifacts = decodeRoots(decoder, "retained Artifacts");
  if (!artifacts)
    return artifacts.takeError();
  auto evidence = decodeRoots(decoder, "retained Evidence");
  if (!evidence)
    return evidence.takeError();
  return InvocationControllerOutcome{InvocationIncomplete{
      *planNode, std::move(*reason), std::move(obligations),
      std::move(*artifacts), std::move(*evidence)}};
}

std::vector<std::uint8_t>
encodeManifest(const InvocationOccurrenceRef &occurrence,
               const DseRunClosure &closure,
               const std::optional<InvocationOccurrenceRef> &resumedFrom,
               llvm::ArrayRef<std::uint8_t> dseViewDescriptor,
               const ComponentViewDigest &dseViewDigest,
               llvm::ArrayRef<InvocationGenerateRecord> generateRecords,
               const InvocationControllerOutcome &outcome) {
  Encoder encoder;
  encoder.text(InvocationManifest::schemaIdentity);
  encoder.u32(InvocationManifest::schemaVersion.major);
  encoder.u32(InvocationManifest::schemaVersion.minor);
  encoder.text(closure.producer().spelling());
  encodeRoots(encoder, closure.semanticInputs());
  encoder.fixed(closure.resolvedConfigIdentity().bytes());
  encodeRoots(encoder, closure.preexistingEvidence());
  encoder.fixed(occurrence.runKey.bytes());
  encoder.u64(occurrence.occurrenceOrdinal);
  encoder.u32(resumedFrom ? 1 : 0);
  if (resumedFrom) {
    encoder.fixed(resumedFrom->runKey.bytes());
    encoder.u64(resumedFrom->occurrenceOrdinal);
  }
  encoder.bytes(dseViewDescriptor);
  encoder.fixed(dseViewDigest.bytes());
  encoder.u64(generateRecords.size());
  for (const InvocationGenerateRecord &record : generateRecords)
    encodeGenerateRecord(encoder, record);
  encodeOutcome(encoder, outcome);
  return encoder.take();
}

llvm::Error validateManifest(
    const InvocationOccurrenceRef &occurrence, const DseRunClosure &closure,
    const std::optional<InvocationOccurrenceRef> &resumedFrom,
    llvm::ArrayRef<std::uint8_t> dseViewDescriptor,
    const ComponentViewDigest &dseViewDigest,
    llvm::ArrayRef<InvocationGenerateRecord> generateRecords,
    const InvocationControllerOutcome &outcome,
    const ResolvedDseConfigView &view, const ArtifactStore &store) {
  if (occurrence.runKey != closure.runKey())
    return invalid("occurrence run key disagrees with its semantic closure");
  if (resumedFrom &&
      (resumedFrom->runKey != occurrence.runKey ||
       resumedFrom->occurrenceOrdinal >= occurrence.occurrenceOrdinal))
    return invalid("resume provenance must name an earlier occurrence of the "
                   "same run key");
  if (dseViewDescriptor != view.schemaDescriptorBytes() ||
      dseViewDigest != view.digest())
    return invalid("resolved DSE component view verification record does not "
                   "match the exact ResolvedConfig");
  if (llvm::Error error =
          validateGenerateSequence(generateRecords, view, outcome, store))
    return error;
  return validateOutcome(outcome, generateRecords, closure, view, store);
}

llvm::Expected<std::vector<InvocationGenerateRecord>>
flattenGenerateRecords(const DsePlanGenerateInvocationRecords &records) {
  if (records.completed().size() != records.completedWorkSummaries().size())
    return invalid("completed Generate records and work summaries differ in "
                   "width");
  if (records.incomplete().has_value() !=
      records.incompleteWorkSummary().has_value())
    return invalid("incomplete Generate record and work summary presence "
                   "differs");
  std::vector<InvocationGenerateRecord> flattened;
  flattened.reserve(records.completed().size() +
                    (records.incomplete() ? 1 : 0));
  for (auto [record, work] :
       llvm::zip_equal(records.completed(), records.completedWorkSummaries()))
    flattened.push_back({true, record, work});
  if (records.incomplete())
    flattened.push_back(
        {false, *records.incomplete(), *records.incompleteWorkSummary()});
  return flattened;
}

} // namespace

llvm::Expected<DseProducerSemanticBuildIdentity>
DseProducerSemanticBuildIdentity::get(llvm::StringRef spelling) {
  if (!isCanonicalAscii(spelling))
    return invalid("producer semantic/build identity must be nonempty "
                   "canonical ASCII");
  return DseProducerSemanticBuildIdentity(spelling.str());
}

llvm::Expected<DseRunKey>
DseRunKey::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return invalid("DSE run key requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return DseRunKey(storage);
}

llvm::Expected<DseRunClosure>
DseRunClosure::get(DseProducerSemanticBuildIdentity producer,
                   llvm::ArrayRef<ArtifactRootReference> semanticInputs,
                   const ResolvedConfig &resolvedConfig,
                   llvm::ArrayRef<ArtifactRootReference> preexistingEvidence,
                   const ArtifactStore &artifactStore) {
  if (llvm::Error error = validateConfigObject(resolvedConfig, artifactStore))
    return std::move(error);
  std::vector<ArtifactRootReference> canonicalInputs(semanticInputs.begin(),
                                                     semanticInputs.end());
  std::vector<ArtifactRootReference> canonicalEvidence(
      preexistingEvidence.begin(), preexistingEvidence.end());
  canonicalizeRoots(canonicalInputs);
  canonicalizeRoots(canonicalEvidence);
  if (llvm::Error error = validateStoredRoots(
          canonicalInputs, "semantic inputs", artifactStore, false))
    return std::move(error);
  if (llvm::Error error = validateStoredRoots(
          canonicalEvidence, "preexisting Evidence", artifactStore, true))
    return std::move(error);
  const ArtifactIdentity configIdentity =
      loom::resolvedConfigIdentity(resolvedConfig);
  const std::vector<std::uint8_t> preimage = runKeyPreimage(
      producer, canonicalInputs, configIdentity, canonicalEvidence);
  DseRunKey runKey =
      llvm::cantFail(DseRunKey::fromBytes(llvm::SHA256::hash(preimage)));
  return DseRunClosure(std::move(producer), std::move(canonicalInputs),
                       configIdentity, std::move(canonicalEvidence),
                       std::move(runKey));
}

llvm::Expected<InvocationManifest>
InvocationManifest::get(DseRunClosure closure, std::uint64_t occurrenceOrdinal,
                        std::optional<InvocationOccurrenceRef> resumedFrom,
                        const ResolvedConfig &resolvedConfig,
                        const DsePlanGenerateInvocationRecords &generateRecords,
                        InvocationControllerOutcome outcome,
                        const ArtifactStore &artifactStore) {
  if (closure.resolvedConfigIdentity() !=
      loom::resolvedConfigIdentity(resolvedConfig))
    return invalid("run closure names a different ResolvedConfig identity");
  auto view = projectResolvedDseConfigView(resolvedConfig);
  if (!view)
    return view.takeError();
  if (generateRecords.resolvedDseConfigViewDigest() != view->digest())
    return invalid("Generate records name a different resolved DSE view");
  auto flattened = flattenGenerateRecords(generateRecords);
  if (!flattened)
    return flattened.takeError();
  if (llvm::Error error = canonicalizeOutcome(outcome))
    return std::move(error);
  InvocationOccurrenceRef occurrence{closure.runKey(), occurrenceOrdinal};
  if (llvm::Error error = validateManifest(
          occurrence, closure, resumedFrom, view->schemaDescriptorBytes(),
          view->digest(), *flattened, outcome, *view, artifactStore))
    return std::move(error);
  std::vector<std::uint8_t> canonical = encodeManifest(
      occurrence, closure, resumedFrom, view->schemaDescriptorBytes(),
      view->digest(), *flattened, outcome);
  return InvocationManifest(
      std::move(occurrence), std::move(closure), std::move(resumedFrom),
      view->schemaDescriptorBytes().vec(), view->digest(),
      std::move(*flattened), std::move(outcome), std::move(canonical));
}

llvm::Expected<InvocationManifest>
adoptInvocationManifest(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                        const ResolvedConfig &resolvedConfig,
                        const ArtifactStore &artifactStore) {
  Decoder decoder(canonicalBytes);
  auto schema = decoder.text("InvocationManifest schema identity");
  if (!schema)
    return schema.takeError();
  auto major = decoder.u32("InvocationManifest schema major");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("InvocationManifest schema minor");
  if (!minor)
    return minor.takeError();
  if (*schema != InvocationManifest::schemaIdentity ||
      SchemaVersion{*major, *minor} != InvocationManifest::schemaVersion)
    return invalid("unsupported InvocationManifest schema");

  auto producerSpelling = decoder.text("producer semantic/build identity");
  if (!producerSpelling)
    return producerSpelling.takeError();
  auto producer = DseProducerSemanticBuildIdentity::get(*producerSpelling);
  if (!producer)
    return producer.takeError();
  auto semanticInputs = decodeRoots(decoder, "semantic inputs");
  if (!semanticInputs)
    return semanticInputs.takeError();
  auto configIdentity = decodeIdentity(decoder, "ResolvedConfig identity");
  if (!configIdentity)
    return configIdentity.takeError();
  auto preexistingEvidence = decodeRoots(decoder, "preexisting Evidence");
  if (!preexistingEvidence)
    return preexistingEvidence.takeError();
  auto encodedRunKey = decodeRunKey(decoder, "DSE run key");
  if (!encodedRunKey)
    return encodedRunKey.takeError();
  auto occurrenceOrdinal = decoder.u64("occurrence ordinal");
  if (!occurrenceOrdinal)
    return occurrenceOrdinal.takeError();
  auto hasResume = decoder.u32("resume provenance presence");
  if (!hasResume)
    return hasResume.takeError();
  if (*hasResume > 1)
    return invalid("resume provenance presence is not boolean");
  std::optional<InvocationOccurrenceRef> resumedFrom;
  if (*hasResume == 1) {
    auto resumeKey = decodeRunKey(decoder, "resume run key");
    if (!resumeKey)
      return resumeKey.takeError();
    auto resumeOrdinal = decoder.u64("resume occurrence ordinal");
    if (!resumeOrdinal)
      return resumeOrdinal.takeError();
    resumedFrom.emplace(
        InvocationOccurrenceRef{std::move(*resumeKey), *resumeOrdinal});
  }
  auto dseViewDescriptor = decoder.bytes("resolved DSE view descriptor");
  if (!dseViewDescriptor)
    return dseViewDescriptor.takeError();
  auto dseViewDigest = decodeDigest(decoder, "resolved DSE view digest");
  if (!dseViewDigest)
    return dseViewDigest.takeError();
  auto recordCount = decoder.count("Generate record count");
  if (!recordCount)
    return recordCount.takeError();
  std::vector<InvocationGenerateRecord> records;
  records.reserve(*recordCount);
  for (std::size_t index = 0; index != *recordCount; ++index) {
    auto record = decodeGenerateRecord(decoder);
    if (!record)
      return record.takeError();
    records.push_back(std::move(*record));
  }
  auto outcome = decodeOutcome(decoder);
  if (!outcome)
    return outcome.takeError();
  if (!decoder.atEnd())
    return invalid("canonical InvocationManifest has trailing bytes");

  auto closure =
      DseRunClosure::get(std::move(*producer), *semanticInputs, resolvedConfig,
                         *preexistingEvidence, artifactStore);
  if (!closure)
    return closure.takeError();
  if (*configIdentity != closure->resolvedConfigIdentity())
    return invalid("InvocationManifest ResolvedConfig identity disagrees with "
                   "the supplied exact configuration");
  if (*encodedRunKey != closure->runKey())
    return invalid("InvocationManifest run key does not match its closure");
  auto view = projectResolvedDseConfigView(resolvedConfig);
  if (!view)
    return view.takeError();
  if (llvm::Error error = canonicalizeOutcome(*outcome))
    return std::move(error);
  InvocationOccurrenceRef occurrence{closure->runKey(), *occurrenceOrdinal};
  if (llvm::Error error = validateManifest(
          occurrence, *closure, resumedFrom, *dseViewDescriptor, *dseViewDigest,
          records, *outcome, *view, artifactStore))
    return std::move(error);
  std::vector<std::uint8_t> reencoded =
      encodeManifest(occurrence, *closure, resumedFrom, *dseViewDescriptor,
                     *dseViewDigest, records, *outcome);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalBytes)
    return invalid("InvocationManifest bytes are not canonical");
  return InvocationManifest(
      std::move(occurrence), std::move(*closure), std::move(resumedFrom),
      std::move(*dseViewDescriptor), std::move(*dseViewDigest),
      std::move(records), std::move(*outcome), std::move(reencoded));
}

} // namespace loom::dse
