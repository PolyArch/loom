#include "DSE/EvidenceObligation.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

using namespace evaluation;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dse_evidence_obligation_invalid: " + message);
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

  void framed(llvm::ArrayRef<std::uint8_t> value) {
    u64(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  void text(llvm::StringRef value) {
    framed(
        {reinterpret_cast<const std::uint8_t *>(value.data()), value.size()});
  }

  void root(const ArtifactRootReference &reference) {
    const std::vector<std::uint8_t> encoded =
        encodeArtifactRootReference(reference);
    bytes_.insert(bytes_.end(), encoded.begin(), encoded.end());
  }

  void optionalRoot(const std::optional<ArtifactRootReference> &reference) {
    u32(reference ? 1 : 0);
    if (reference)
      root(*reference);
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32() {
    if (remaining() < 4)
      return invalid("truncated u32 field");
    std::uint32_t value = 0;
    for (unsigned index = 0; index != 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> u64() {
    if (remaining() < 8)
      return invalid("truncated u64 field");
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::size_t> count(std::size_t minimumBytesPerItem = 0) {
    auto value = u64();
    if (!value)
      return value.takeError();
    if (*value > std::numeric_limits<std::size_t>::max())
      return invalid("sequence count is not host-representable");
    const std::size_t count = static_cast<std::size_t>(*value);
    if (minimumBytesPerItem != 0 && count > remaining() / minimumBytesPerItem)
      return invalid("sequence count exceeds remaining bytes");
    return count;
  }

  llvm::Expected<std::string> text() {
    auto size = count();
    if (!size)
      return size.takeError();
    if (*size > remaining())
      return invalid("framed text exceeds remaining bytes");
    const char *begin = reinterpret_cast<const char *>(bytes_.data() + offset_);
    std::string value(begin, *size);
    offset_ += *size;
    return value;
  }

  llvm::Expected<ArtifactRootReference> root() {
    auto decoded =
        decodeArtifactRootReferencePrefix(bytes_.drop_front(offset_));
    if (!decoded)
      return decoded.takeError();
    offset_ += decoded->byteCount;
    return std::move(decoded->reference);
  }

  llvm::Expected<std::optional<ArtifactRootReference>> optionalRoot() {
    auto present = u32();
    if (!present)
      return present.takeError();
    if (*present > 1)
      return invalid("invalid optional root discriminant");
    if (*present == 0)
      return std::optional<ArtifactRootReference>{};
    auto value = root();
    if (!value)
      return value.takeError();
    return std::optional<ArtifactRootReference>{std::move(*value)};
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

struct TemplateParts final {
  ResolvedModelBinding modelBinding;
  std::vector<CaseRoleBinding> fixedSubjectBindings;
  std::optional<ArtifactRootReference> workload;
  std::optional<ArtifactRootReference> runtimeInput;
  std::vector<EvaluationCondition> baseConditions;
  std::vector<MetricRequestTemplate> metricRequests;
  std::vector<FindingRequestTemplate> findingRequests;
  CaseSubjectRoleRef candidateRole;
  std::vector<InputSubjectBinding> inputSubjectBindings;
  std::optional<CalibrationPartitionRole> calibrationPartitionRole;
};

bool roleLess(CaseSubjectRoleRef lhs, CaseSubjectRoleRef rhs) {
  return lhs.ordinal() < rhs.ordinal();
}

bool acceptsSchema(llvm::ArrayRef<const ArtifactSchemaDescriptor *> schemas,
                   const ArtifactRootReference &artifact) {
  return llvm::any_of(schemas, [&](const ArtifactSchemaDescriptor *schema) {
    return schema && artifact.schemaIdentity == schema->identity &&
           artifact.schemaVersion == schema->version;
  });
}

bool dynamicRole(const TemplateParts &parts, CaseSubjectRoleRef role) {
  if (role == parts.candidateRole)
    return true;
  return llvm::any_of(
      parts.inputSubjectBindings,
      [&](const InputSubjectBinding &binding) { return binding.role == role; });
}

const CaseRoleBinding *findFixedBinding(const TemplateParts &parts,
                                        CaseSubjectRoleRef role) {
  auto found = llvm::lower_bound(
      parts.fixedSubjectBindings, role,
      [](const CaseRoleBinding &binding, CaseSubjectRoleRef sought) {
        return roleLess(binding.role, sought);
      });
  if (found == parts.fixedSubjectBindings.end() || found->role != role)
    return nullptr;
  return &*found;
}

llvm::Error validateTarget(const TemplateParts &parts,
                           const SubjectTargetRef &target) {
  if (dynamicRole(parts, target.caseSubjectRole))
    return invalid("request target references an unresolved dynamic subject "
                   "role");
  const CaseRoleBinding *binding =
      findFixedBinding(parts, target.caseSubjectRole);
  if (!binding ||
      !llvm::is_contained(binding->subjects, target.anchorSubjectArtifact))
    return invalid("request target anchor is not a fixed bound subject");
  return llvm::Error::success();
}

llvm::Error validateTargets(const TemplateParts &parts) {
  const auto validateConditions =
      [&](llvm::ArrayRef<EvaluationCondition> conditions) -> llvm::Error {
    for (const EvaluationCondition &condition : conditions)
      for (const SubjectTargetRef *target : conditionOrderedTargets(condition))
        if (llvm::Error error = validateTarget(parts, *target))
          return error;
    return llvm::Error::success();
  };

  if (llvm::Error error = validateConditions(parts.baseConditions))
    return error;
  for (const MetricRequestTemplate &request : parts.metricRequests) {
    for (const SubjectTargetRef &target : request.query.scope.targets)
      if (llvm::Error error = validateTarget(parts, target))
        return error;
    if (llvm::Error error = validateConditions(request.conditions))
      return error;
  }
  for (const FindingRequestTemplate &request : parts.findingRequests) {
    for (const SubjectTargetRef &target : request.query.scope.targets)
      if (llvm::Error error = validateTarget(parts, target))
        return error;
    if (llvm::Error error = validateConditions(request.conditions))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error validateOptionalCaseReference(
    ArtifactRequirement requirement,
    llvm::ArrayRef<const ArtifactSchemaDescriptor *> schemas,
    const std::optional<ArtifactRootReference> &reference,
    llvm::StringRef field) {
  if (requirement == ArtifactRequirement::Forbidden && reference)
    return invalid(field + " is forbidden by the case signature");
  if (requirement == ArtifactRequirement::Required && !reference)
    return invalid(field + " is required by the case signature");
  if (reference && !acceptsSchema(schemas, *reference))
    return invalid(field + " has an unaccepted artifact schema");
  return llvm::Error::success();
}

std::vector<std::uint8_t>
metricRequestKey(const MetricRequestTemplate &request) {
  return canonicalMetricRequestKey(request.query, request.conditions);
}

std::vector<std::uint8_t>
findingRequestKey(const FindingRequestTemplate &request) {
  return canonicalFindingRequestKey(request.query, request.conditions);
}

llvm::Error validateParts(TemplateParts &parts) {
  const EvaluationModelDescriptor *model =
      parts.modelBinding.descriptorRef().descriptor();
  if (!model)
    return invalid("model binding references an unregistered descriptor");
  if (llvm::Error error = validateResolvedModelBinding(parts.modelBinding))
    return error;
  const EvaluationCaseSignatureDescriptor *signature =
      model->caseSignature.descriptor();
  if (!signature)
    return invalid("model descriptor references an unregistered case "
                   "signature");
  const CaseSubjectRoleDescriptor *candidate =
      signature->findSubjectRole(parts.candidateRole);
  if (!candidate)
    return invalid("candidate role is not owned by the case signature");
  if (candidate->cardinality != SubjectRoleCardinality::ExactlyOne)
    return invalid("candidate role must have exactly-one cardinality");
  if (parts.calibrationPartitionRole &&
      static_cast<std::uint32_t>(*parts.calibrationPartitionRole) >
          static_cast<std::uint32_t>(CalibrationPartitionRole::HeldOut))
    return invalid("unknown calibration partition role");

  llvm::sort(parts.inputSubjectBindings, [](const InputSubjectBinding &lhs,
                                            const InputSubjectBinding &rhs) {
    return roleLess(lhs.role, rhs.role);
  });
  for (std::size_t index = 0; index < parts.inputSubjectBindings.size();
       ++index) {
    const InputSubjectBinding &binding = parts.inputSubjectBindings[index];
    if (binding.role == parts.candidateRole)
      return invalid("candidate role cannot be input-bound");
    if (!signature->findSubjectRole(binding.role))
      return invalid("input-bound role is not owned by the case signature");
    if (index != 0 &&
        parts.inputSubjectBindings[index - 1].role == binding.role)
      return invalid("duplicate input-bound subject role");
  }

  auto canonicalFixed =
      EvaluationSubjectBindings::get(std::move(parts.fixedSubjectBindings));
  if (!canonicalFixed)
    return canonicalFixed.takeError();
  parts.fixedSubjectBindings.assign(canonicalFixed->roleBindings().begin(),
                                    canonicalFixed->roleBindings().end());

  for (const CaseSubjectRoleDescriptor &role : signature->subjectRoles) {
    const CaseRoleBinding *fixed = findFixedBinding(parts, role.role);
    if (dynamicRole(parts, role.role)) {
      if (fixed)
        return invalid("dynamic subject role also has a fixed binding");
      continue;
    }
    if (!fixed)
      return invalid("non-dynamic subject role has no fixed binding");
    if (role.cardinality == SubjectRoleCardinality::ExactlyOne &&
        fixed->subjects.size() != 1)
      return invalid("fixed exactly-one role has the wrong cardinality");
    if (fixed->subjects.empty())
      return invalid("fixed subject role is empty");
    for (const ArtifactRootReference &subject : fixed->subjects)
      if (!acceptsSchema(role.acceptedSchemas, subject))
        return invalid("fixed subject has an unaccepted artifact schema");
  }
  for (const CaseRoleBinding &fixed : parts.fixedSubjectBindings)
    if (!signature->findSubjectRole(fixed.role))
      return invalid("fixed binding contains a foreign case role");

  if (llvm::Error error = validateOptionalCaseReference(
          signature->workload, signature->acceptedWorkloadSchemas,
          parts.workload, "workload"))
    return error;
  if (llvm::Error error = validateOptionalCaseReference(
          signature->runtimeInput, signature->acceptedRuntimeInputSchemas,
          parts.runtimeInput, "runtime input"))
    return error;
  if (parts.metricRequests.empty() && parts.findingRequests.empty())
    return invalid("template requires a metric or finding request");

  for (const MetricRequestTemplate &request : parts.metricRequests) {
    if (llvm::Error error = validateMetricQuery(request.query))
      return error;
    if (!model->supportsMetricQuery(request.query))
      return invalid("model descriptor rejects a metric query");
  }
  for (const FindingRequestTemplate &request : parts.findingRequests) {
    if (llvm::Error error = validateFindingQuery(request.query))
      return error;
    if (!model->supportsFindingQuery(request.query))
      return invalid("model descriptor rejects a finding query");
  }

  std::vector<std::vector<std::uint8_t>> metricKeys;
  metricKeys.reserve(parts.metricRequests.size());
  for (const MetricRequestTemplate &request : parts.metricRequests)
    metricKeys.push_back(metricRequestKey(request));
  if (!llvm::is_sorted(metricKeys))
    return invalid("metric requests are not in canonical order");
  for (std::size_t index = 1; index < metricKeys.size(); ++index)
    if (metricKeys[index - 1] == metricKeys[index])
      return invalid("duplicate metric request");

  std::vector<std::vector<std::uint8_t>> findingKeys;
  findingKeys.reserve(parts.findingRequests.size());
  for (const FindingRequestTemplate &request : parts.findingRequests)
    findingKeys.push_back(findingRequestKey(request));
  if (!llvm::is_sorted(findingKeys))
    return invalid("finding requests are not in canonical order");
  for (std::size_t index = 1; index < findingKeys.size(); ++index)
    if (findingKeys[index - 1] == findingKeys[index])
      return invalid("duplicate finding request");

  return validateTargets(parts);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeParts(const TemplateParts &parts) {
  Encoder encoder;
  encoder.text(serializeResolvedModelBinding(parts.modelBinding));
  encoder.u64(parts.fixedSubjectBindings.size());
  for (const CaseRoleBinding &binding : parts.fixedSubjectBindings) {
    encoder.u32(binding.role.ordinal());
    encoder.u64(binding.subjects.size());
    for (const ArtifactRootReference &subject : binding.subjects)
      encoder.root(subject);
  }
  encoder.optionalRoot(parts.workload);
  encoder.optionalRoot(parts.runtimeInput);
  encoder.text(serializeEvaluationConditions(parts.baseConditions));
  encoder.u64(parts.metricRequests.size());
  for (const MetricRequestTemplate &request : parts.metricRequests) {
    auto query = serializeMetricQuery(request.query);
    if (!query)
      return query.takeError();
    encoder.text(*query);
    encoder.text(serializeEvaluationConditions(request.conditions));
  }
  encoder.u64(parts.findingRequests.size());
  for (const FindingRequestTemplate &request : parts.findingRequests) {
    auto query = serializeFindingQuery(request.query);
    if (!query)
      return query.takeError();
    encoder.text(*query);
    encoder.text(serializeEvaluationConditions(request.conditions));
  }
  encoder.u32(parts.candidateRole.ordinal());
  encoder.u64(parts.inputSubjectBindings.size());
  for (const InputSubjectBinding &binding : parts.inputSubjectBindings) {
    encoder.u32(binding.role.ordinal());
    encoder.u32(binding.inputSlot.ordinal());
  }
  encoder.u32(parts.calibrationPartitionRole ? 1 : 0);
  if (parts.calibrationPartitionRole)
    encoder.u32(static_cast<std::uint32_t>(*parts.calibrationPartitionRole));
  return encoder.take();
}

llvm::Expected<std::vector<EvaluationCondition>>
decodeConditions(Decoder &decoder) {
  auto text = decoder.text();
  if (!text)
    return text.takeError();
  return parseEvaluationConditions(*text);
}

llvm::Expected<TemplateParts> decodeParts(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto modelText = decoder.text();
  if (!modelText)
    return modelText.takeError();
  auto model = parseResolvedModelBinding(*modelText);
  if (!model)
    return model.takeError();

  auto fixedCount = decoder.count(12);
  if (!fixedCount)
    return fixedCount.takeError();
  std::vector<CaseRoleBinding> fixed;
  fixed.reserve(*fixedCount);
  for (std::size_t index = 0; index != *fixedCount; ++index) {
    auto role = decoder.u32();
    auto subjectCount = decoder.count(1);
    if (!role)
      return role.takeError();
    if (!subjectCount)
      return subjectCount.takeError();
    CaseRoleBinding binding{CaseSubjectRoleRef(*role), {}};
    binding.subjects.reserve(*subjectCount);
    for (std::size_t subject = 0; subject != *subjectCount; ++subject) {
      auto root = decoder.root();
      if (!root)
        return root.takeError();
      binding.subjects.push_back(std::move(*root));
    }
    fixed.push_back(std::move(binding));
  }
  auto workload = decoder.optionalRoot();
  auto runtimeInput = decoder.optionalRoot();
  auto baseConditions = decodeConditions(decoder);
  if (!workload)
    return workload.takeError();
  if (!runtimeInput)
    return runtimeInput.takeError();
  if (!baseConditions)
    return baseConditions.takeError();

  auto metricCount = decoder.count(16);
  if (!metricCount)
    return metricCount.takeError();
  std::vector<MetricRequestTemplate> metrics;
  metrics.reserve(*metricCount);
  for (std::size_t index = 0; index != *metricCount; ++index) {
    auto queryText = decoder.text();
    if (!queryText)
      return queryText.takeError();
    auto query = parseMetricQuery(*queryText);
    if (!query)
      return query.takeError();
    auto conditions = decodeConditions(decoder);
    if (!conditions)
      return conditions.takeError();
    metrics.push_back({std::move(*query), std::move(*conditions)});
  }

  auto findingCount = decoder.count(16);
  if (!findingCount)
    return findingCount.takeError();
  std::vector<FindingRequestTemplate> findings;
  findings.reserve(*findingCount);
  for (std::size_t index = 0; index != *findingCount; ++index) {
    auto queryText = decoder.text();
    if (!queryText)
      return queryText.takeError();
    auto query = parseFindingQuery(*queryText);
    if (!query)
      return query.takeError();
    auto conditions = decodeConditions(decoder);
    if (!conditions)
      return conditions.takeError();
    findings.push_back({std::move(*query), std::move(*conditions)});
  }

  auto candidate = decoder.u32();
  auto inputCount = decoder.count(8);
  if (!candidate)
    return candidate.takeError();
  if (!inputCount)
    return inputCount.takeError();
  std::vector<InputSubjectBinding> inputs;
  inputs.reserve(*inputCount);
  for (std::size_t index = 0; index != *inputCount; ++index) {
    auto role = decoder.u32();
    auto slot = decoder.u32();
    if (!role)
      return role.takeError();
    if (!slot)
      return slot.takeError();
    inputs.push_back(
        {CaseSubjectRoleRef(*role), EvidenceAcquisitionInputSlotRef(*slot)});
  }
  auto partitionPresent = decoder.u32();
  if (!partitionPresent)
    return partitionPresent.takeError();
  if (*partitionPresent > 1)
    return invalid("invalid calibration partition optional tag");
  std::optional<CalibrationPartitionRole> partition;
  if (*partitionPresent == 1) {
    auto tag = decoder.u32();
    if (!tag)
      return tag.takeError();
    if (*tag > static_cast<std::uint32_t>(CalibrationPartitionRole::HeldOut))
      return invalid("unknown calibration partition role");
    partition = static_cast<CalibrationPartitionRole>(*tag);
  }
  if (decoder.remaining() != 0)
    return invalid("template has trailing bytes");

  return TemplateParts{
      std::move(*model),          std::move(fixed),
      std::move(*workload),       std::move(*runtimeInput),
      std::move(*baseConditions), std::move(metrics),
      std::move(findings),        CaseSubjectRoleRef(*candidate),
      std::move(inputs),          partition};
}

TemplateParts
partsFromPrototype(const EvaluationRequest &prototype,
                   CaseSubjectRoleRef candidateRole,
                   std::vector<InputSubjectBinding> inputSubjectBindings,
                   std::optional<CalibrationPartitionRole> partition) {
  TemplateParts parts{
      prototype.modelBinding(),
      {},
      prototype.workload(),
      prototype.runtimeInput(),
      {prototype.baseConditions().begin(), prototype.baseConditions().end()},
      {},
      {},
      candidateRole,
      std::move(inputSubjectBindings),
      partition};
  for (const CaseRoleBinding &binding :
       prototype.subjectBindings().roleBindings()) {
    if (!dynamicRole(parts, binding.role))
      parts.fixedSubjectBindings.push_back(binding);
  }
  for (const MetricRequest &request : prototype.metricRequests())
    parts.metricRequests.push_back(
        {request.query(),
         {request.conditions().begin(), request.conditions().end()}});
  for (const FindingRequest &request : prototype.findingRequests())
    parts.findingRequests.push_back(
        {request.query(),
         {request.conditions().begin(), request.conditions().end()}});
  return parts;
}

} // namespace

llvm::Expected<EvidenceObligationTemplate> EvidenceObligationTemplate::get(
    const EvaluationRequest &prototype, CaseSubjectRoleRef candidateRole,
    std::vector<InputSubjectBinding> inputSubjectBindings,
    std::optional<CalibrationPartitionRole> calibrationPartitionRole) {
  TemplateParts parts = partsFromPrototype(prototype, candidateRole,
                                           std::move(inputSubjectBindings),
                                           calibrationPartitionRole);
  if (llvm::Error error = validateParts(parts))
    return std::move(error);
  auto bytes = encodeParts(parts);
  if (!bytes)
    return bytes.takeError();
  return EvidenceObligationTemplate(
      std::move(parts.modelBinding), std::move(parts.fixedSubjectBindings),
      std::move(parts.workload), std::move(parts.runtimeInput),
      std::move(parts.baseConditions), std::move(parts.metricRequests),
      std::move(parts.findingRequests), parts.candidateRole,
      std::move(parts.inputSubjectBindings), parts.calibrationPartitionRole,
      std::move(*bytes));
}

llvm::Expected<EvidenceObligationTemplate>
adoptEvidenceObligationTemplate(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parts = decodeParts(bytes);
  if (!parts)
    return parts.takeError();
  if (llvm::Error error = validateParts(*parts))
    return std::move(error);
  auto canonical = encodeParts(*parts);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("template bytes are not canonical");
  return EvidenceObligationTemplate(
      std::move(parts->modelBinding), std::move(parts->fixedSubjectBindings),
      std::move(parts->workload), std::move(parts->runtimeInput),
      std::move(parts->baseConditions), std::move(parts->metricRequests),
      std::move(parts->findingRequests), parts->candidateRole,
      std::move(parts->inputSubjectBindings), parts->calibrationPartitionRole,
      std::move(*canonical));
}

llvm::Expected<EvaluationRequest> instantiateEvidenceObligation(
    const EvidenceObligationTemplate &obligation,
    const ArtifactRootReference &candidate,
    llvm::ArrayRef<EvidenceAcquisitionInputBinding> inputBindings,
    std::uint64_t replicateIndex, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore) {
  std::vector<EvidenceAcquisitionInputBinding> canonicalInputs(
      inputBindings.begin(), inputBindings.end());
  llvm::sort(canonicalInputs, [](const EvidenceAcquisitionInputBinding &lhs,
                                 const EvidenceAcquisitionInputBinding &rhs) {
    return lhs.slot.ordinal() < rhs.slot.ordinal();
  });
  for (std::size_t index = 0; index < canonicalInputs.size(); ++index) {
    EvidenceAcquisitionInputBinding &binding = canonicalInputs[index];
    if (index != 0 && canonicalInputs[index - 1].slot == binding.slot)
      return invalid("duplicate acquisition input slot");
    llvm::sort(binding.artifacts, artifactRootReferenceLess);
    if (std::adjacent_find(binding.artifacts.begin(),
                           binding.artifacts.end()) != binding.artifacts.end())
      return invalid("duplicate artifact in acquisition input slot");
  }
  std::vector<std::uint32_t> expectedSlots;
  expectedSlots.reserve(obligation.inputSubjectBindings_.size());
  for (const InputSubjectBinding &binding : obligation.inputSubjectBindings_)
    expectedSlots.push_back(binding.inputSlot.ordinal());
  llvm::sort(expectedSlots);
  expectedSlots.erase(std::unique(expectedSlots.begin(), expectedSlots.end()),
                      expectedSlots.end());
  for (const EvidenceAcquisitionInputBinding &binding : canonicalInputs)
    if (!std::binary_search(expectedSlots.begin(), expectedSlots.end(),
                            binding.slot.ordinal()))
      return invalid("acquisition input contains an unreferenced slot " +
                     llvm::Twine(binding.slot.ordinal()));

  const auto findInput = [&](EvidenceAcquisitionInputSlotRef slot)
      -> const EvidenceAcquisitionInputBinding * {
    auto found =
        llvm::lower_bound(canonicalInputs, slot,
                          [](const EvidenceAcquisitionInputBinding &binding,
                             EvidenceAcquisitionInputSlotRef sought) {
                            return binding.slot.ordinal() < sought.ordinal();
                          });
    if (found == canonicalInputs.end() || found->slot != slot)
      return nullptr;
    return &*found;
  };

  std::vector<CaseRoleBinding> roles = obligation.fixedSubjectBindings_;
  roles.push_back({obligation.candidateRole_, {candidate}});
  for (const InputSubjectBinding &source : obligation.inputSubjectBindings_) {
    const EvidenceAcquisitionInputBinding *input = findInput(source.inputSlot);
    if (!input)
      return invalid("required acquisition input slot " +
                     llvm::Twine(source.inputSlot.ordinal()) + " is absent");
    roles.push_back({source.role, input->artifacts});
  }
  auto bindings = EvaluationSubjectBindings::get(std::move(roles));
  if (!bindings)
    return bindings.takeError();

  const EvaluationModelDescriptor *model =
      obligation.modelBinding_.descriptorRef().descriptor();
  if (!model)
    return invalid("template model descriptor became unavailable");
  auto evaluationCase = EvaluationCase::get(
      model->caseSignature, std::move(*bindings), obligation.workload_,
      obligation.runtimeInput_, obligation.baseConditions_, resolution,
      artifactStore);
  if (!evaluationCase)
    return evaluationCase.takeError();

  std::vector<MetricRequest> metrics;
  metrics.reserve(obligation.metricRequests_.size());
  for (const MetricRequestTemplate &request : obligation.metricRequests_) {
    auto resolved =
        MetricRequest::get(request.query, request.conditions, *evaluationCase,
                           resolution, artifactStore);
    if (!resolved)
      return resolved.takeError();
    metrics.push_back(std::move(*resolved));
  }
  std::vector<FindingRequest> findings;
  findings.reserve(obligation.findingRequests_.size());
  for (const FindingRequestTemplate &request : obligation.findingRequests_) {
    auto resolved =
        FindingRequest::get(request.query, request.conditions, *evaluationCase,
                            resolution, artifactStore);
    if (!resolved)
      return resolved.takeError();
    findings.push_back(std::move(*resolved));
  }
  return EvaluationRequest::get(*evaluationCase, metrics, findings,
                                obligation.modelBinding_, replicateIndex,
                                resolution, artifactStore);
}

} // namespace loom::dse
