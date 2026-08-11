#include "DSE/PromotionAcquisition.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelProvider.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace loom::dse {
namespace {

std::vector<const PromotionAcquisitionDescriptor *> &descriptors() {
  static std::vector<const PromotionAcquisitionDescriptor *> records;
  return records;
}

std::shared_mutex &descriptorMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

std::vector<PromotionAcquisitionProvider> &providers() {
  static std::vector<PromotionAcquisitionProvider> records;
  return records;
}

std::shared_mutex &providerMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "promotion_acquisition_invalid: " + message);
}

llvm::Error validateSelectedTaskInputs(
    llvm::ArrayRef<EvidenceAcquisitionInputBinding> bound,
    llvm::ArrayRef<EvidenceAcquisitionInputBinding> selected) {
  if (selected.size() != bound.size())
    return invalid("provider did not select every bound task input slot");
  for (std::size_t index = 0; index < selected.size(); ++index) {
    const EvidenceAcquisitionInputBinding &source = bound[index];
    const EvidenceAcquisitionInputBinding &choice = selected[index];
    if (choice.slot != source.slot)
      return invalid("provider changed a bound task input slot");
    if (!llvm::is_sorted(choice.artifacts, artifactRootReferenceLess) ||
        std::adjacent_find(choice.artifacts.begin(), choice.artifacts.end()) !=
            choice.artifacts.end())
      return invalid(
          "provider task input selection is not canonical and unique");
    for (const ArtifactRootReference &artifact : choice.artifacts)
      if (!llvm::binary_search(source.artifacts, artifact,
                               artifactRootReferenceLess))
        return invalid(
            "provider selected an artifact outside the bound task input");
  }
  return llvm::Error::success();
}

bool isCanonicalAscii(llvm::StringRef value) {
  return !value.empty() && llvm::all_of(value, [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

bool acceptsSchema(const PromotionAcquisitionInputSlotDescriptor &slot,
                   const ArtifactRootReference &artifact) {
  return slot.schema && slot.schema->identity == artifact.schemaIdentity &&
         slot.schema->version == artifact.schemaVersion;
}

bool validRole(PlanValueRole role) {
  return static_cast<std::uint32_t>(role) <=
         static_cast<std::uint32_t>(PlanValueRole::SimulationExecutionSet);
}

bool validCardinality(PlanValueCardinality cardinality) {
  return static_cast<std::uint32_t>(cardinality) <=
         static_cast<std::uint32_t>(PlanValueCardinality::FiniteSet);
}

class InProcessPromotionEvidenceExecutor final
    : public PromotionEvidenceExecutor {
public:
  llvm::Expected<std::vector<PromotionEvidenceExecutionResult>>
  execute(llvm::ArrayRef<PromotionEvidenceExecutionTask> tasks,
          const ArtifactStore &store, const BlobStore &blobs) override {
    std::vector<PromotionEvidenceExecutionResult> results;
    results.reserve(tasks.size());
    for (const PromotionEvidenceExecutionTask &task : tasks) {
      auto evidence = evaluation::evaluateRequest(
          task.request, *task.resolution, store, blobs);
      if (!evidence)
        return evidence.takeError();
      results.emplace_back(std::move(*evidence));
    }
    return results;
  }
};

llvm::Error
validateDescriptor(const PromotionAcquisitionDescriptor &descriptor) {
  if (!isCanonicalAscii(descriptor.spelling) ||
      !isCanonicalAscii(descriptor.stableIdentity))
    return invalid("descriptor identities must be nonempty canonical ASCII");
  if (descriptor.inputSlots.empty())
    return invalid("descriptor requires typed input slots");
  if (descriptor.resolvedConfigView.schemaDescriptorBytes.empty() ||
      !descriptor.resolvedConfigView.validateCanonical)
    return invalid("descriptor requires an exact resolved config contract");
  if (!descriptor.resolveEvidenceObligations)
    return invalid("descriptor requires an Evidence obligation resolver");
  for (std::size_t index = 0; index < descriptor.inputSlots.size(); ++index) {
    const PromotionAcquisitionInputSlotDescriptor &slot =
        descriptor.inputSlots[index];
    if (slot.ref.ordinal() != index)
      return invalid("input slots must be dense and canonical");
    if (!isCanonicalAscii(slot.spelling) || !slot.schema ||
        !validRole(slot.role) || !validCardinality(slot.cardinality))
      return invalid("input slot has an invalid typed contract");
    const bool isParameterBundle =
        *slot.schema == evaluation::modelParameterBundleSchema;
    if (isParameterBundle != (slot.modelParameterContract != nullptr))
      return invalid("input slot must declare one parameter contract iff it "
                     "accepts ModelParameterBundle");
    if (slot.modelParameterContract &&
        !evaluation::findModelParameterContract(*slot.modelParameterContract))
      return invalid("input slot references an unregistered model parameter "
                     "contract");
    const bool isEvidence =
        slot.role == PlanValueRole::EvidenceSet &&
        *slot.schema == evaluation::EvaluationEvidence::artifactSchema;
    if (slot.calibrationPartitionRole && !isEvidence)
      return invalid("calibration partition is permitted only on an exact "
                     "Evidence input slot");
    if (slot.calibrationPartitionRole &&
        static_cast<std::uint32_t>(*slot.calibrationPartitionRole) >
            static_cast<std::uint32_t>(CalibrationPartitionRole::HeldOut))
      return invalid("input slot has an unknown calibration partition");
  }
  const PromotionAcquisitionInputSlotDescriptor *candidate =
      descriptor.findInputSlot(descriptor.candidateInputSlot);
  if (!candidate || candidate->role != PlanValueRole::CandidateSet)
    return invalid("candidate input slot is absent or has the wrong role");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<PromotionAcquisitionDescriptorRef>
PromotionAcquisitionDescriptorRef::get(
    ArtifactSchemaDescriptor descriptorSchema, PromotionAcquisitionKind kind) {
  if (descriptorSchema != PromotionAcquisitionDescriptor::schema)
    return invalid("promotion acquisition descriptor schema is unsupported");
  return PromotionAcquisitionDescriptorRef(descriptorSchema, kind);
}

const PromotionAcquisitionDescriptor *
PromotionAcquisitionDescriptorRef::descriptor() const {
  if (descriptorSchema_ != PromotionAcquisitionDescriptor::schema)
    return nullptr;
  return findPromotionAcquisitionDescriptor(kind_);
}

PromotionAcquisitionDescriptorRef
PromotionAcquisitionDescriptor::reference() const {
  return llvm::cantFail(PromotionAcquisitionDescriptorRef::get(schema, kind));
}

const PromotionAcquisitionInputSlotDescriptor *
PromotionAcquisitionDescriptor::findInputSlot(
    PromotionAcquisitionInputSlotRef ref) const {
  if (ref.ordinal() >= inputSlots.size())
    return nullptr;
  return &inputSlots[ref.ordinal()];
}

llvm::Error registerPromotionAcquisitionDescriptor(
    const PromotionAcquisitionDescriptor &descriptor) {
  if (llvm::Error error = validateDescriptor(descriptor))
    return error;
  std::unique_lock<std::shared_mutex> lock(descriptorMutex());
  for (const PromotionAcquisitionDescriptor *existing : descriptors()) {
    if (existing == &descriptor)
      return llvm::Error::success();
    if (existing->kind == descriptor.kind)
      return invalid("conflicting promotion acquisition kind " +
                     std::to_string(descriptor.kind.ordinal()));
    if (existing->spelling == descriptor.spelling)
      return invalid("conflicting promotion acquisition spelling '" +
                     descriptor.spelling + "'");
  }
  descriptors().push_back(&descriptor);
  llvm::sort(descriptors(), [](const PromotionAcquisitionDescriptor *lhs,
                               const PromotionAcquisitionDescriptor *rhs) {
    return lhs->kind < rhs->kind;
  });
  return llvm::Error::success();
}

const PromotionAcquisitionDescriptor *
findPromotionAcquisitionDescriptor(PromotionAcquisitionKind kind) {
  std::shared_lock<std::shared_mutex> lock(descriptorMutex());
  auto found =
      llvm::lower_bound(descriptors(), kind,
                        [](const PromotionAcquisitionDescriptor *descriptor,
                           PromotionAcquisitionKind requested) {
                          return descriptor->kind < requested;
                        });
  if (found == descriptors().end() || (*found)->kind != kind)
    return nullptr;
  return *found;
}

llvm::Expected<ResolvedPromotionAcquisitionBinding>
ResolvedPromotionAcquisitionBinding::get(
    PromotionAcquisitionDescriptorRef descriptorRef,
    llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
    const ComponentViewDigest &configDigest) {
  const PromotionAcquisitionDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");
  if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
          canonicalConfigBytes, configDigest))
    return std::move(error);
  auto obligations =
      descriptor->resolveEvidenceObligations(canonicalConfigBytes);
  if (!obligations)
    return obligations.takeError();
  if (!llvm::is_sorted(*obligations,
                       [](EvidenceObligationTemplateRef lhs,
                          EvidenceObligationTemplateRef rhs) {
                         return lhs.ordinal() < rhs.ordinal();
                       }) ||
      std::adjacent_find(obligations->begin(), obligations->end()) !=
          obligations->end())
    return invalid("Evidence obligation references are not canonical");
  return ResolvedPromotionAcquisitionBinding(
      descriptorRef, canonicalConfigBytes.vec(), configDigest,
      std::move(*obligations));
}

llvm::Error validatePromotionAcquisitionInputBindings(
    PromotionAcquisitionDescriptorRef descriptorRef,
    llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings) {
  const PromotionAcquisitionDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("input bindings reference an unregistered descriptor");
  if (inputBindings.size() != descriptor->inputSlots.size())
    return invalid("binding does not provide every descriptor input slot");
  for (std::size_t index = 0; index < inputBindings.size(); ++index) {
    const PromotionAcquisitionInputBinding &binding = inputBindings[index];
    const PromotionAcquisitionInputSlotDescriptor &slot =
        descriptor->inputSlots[index];
    if (binding.slot.ordinal() != index)
      return invalid("input bindings must be dense and canonical");
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (!acceptsSchema(slot, artifact))
        return invalid("input artifact schema does not match slot '" +
                       slot.spelling + "'");
    if (!llvm::is_sorted(binding.artifacts, artifactRootReferenceLess) ||
        std::adjacent_find(binding.artifacts.begin(),
                           binding.artifacts.end()) != binding.artifacts.end())
      return invalid("input artifact sets must be canonical");
    if (!planCardinalityContains(slot.cardinality, binding.artifacts.size()))
      return invalid("canonical input set violates descriptor cardinality");
  }
  return llvm::Error::success();
}

llvm::Error registerPromotionAcquisitionProvider(
    const PromotionAcquisitionProvider &provider) {
  if (!provider.resolve || !provider.descriptor.descriptor())
    return invalid("provider requires a registered descriptor and callback");
  std::unique_lock<std::shared_mutex> lock(providerMutex());
  for (const PromotionAcquisitionProvider &existing : providers()) {
    if (existing.descriptor != provider.descriptor)
      continue;
    if (existing.resolve == provider.resolve)
      return llvm::Error::success();
    return invalid("conflicting provider registration");
  }
  providers().push_back(provider);
  llvm::sort(providers(), [](const PromotionAcquisitionProvider &lhs,
                             const PromotionAcquisitionProvider &rhs) {
    return lhs.descriptor.kind() < rhs.descriptor.kind();
  });
  return llvm::Error::success();
}

llvm::Expected<PromotionAcquisitionOutcome> invokePromotionAcquisition(
    llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings,
    const ResolvedPromotionAcquisitionBinding &binding,
    llvm::ArrayRef<EvidenceObligationTemplate> evidenceObligationTemplates,
    PromotionAcquisitionTaskDomain taskDomain, const ArtifactStore &store,
    const BlobStore &blobs, PromotionEvidenceExecutor *executor) {
  const PromotionAcquisitionDescriptor *descriptor =
      binding.descriptorRef().descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");
  if (llvm::Error error = validatePromotionAcquisitionInputBindings(
          binding.descriptorRef(), inputBindings))
    return std::move(error);
  if (descriptor->candidateInputSlot.ordinal() >= inputBindings.size())
    return invalid("candidate input slot is unavailable");

  if (!llvm::is_sorted(taskDomain.candidates, artifactRootReferenceLess) ||
      std::adjacent_find(taskDomain.candidates.begin(),
                         taskDomain.candidates.end()) !=
          taskDomain.candidates.end())
    return invalid("task-domain candidates are not canonical and unique");
  const PromotionAcquisitionInputBinding &boundCandidates =
      inputBindings[descriptor->candidateInputSlot.ordinal()];
  for (const ArtifactRootReference &candidate : taskDomain.candidates)
    if (!llvm::binary_search(boundCandidates.artifacts, candidate,
                             artifactRootReferenceLess))
      return invalid("task-domain candidate is outside the bound input");

  const auto obligationLess = [](EvidenceObligationTemplateRef lhs,
                                 EvidenceObligationTemplateRef rhs) {
    return lhs.ordinal() < rhs.ordinal();
  };
  if (!llvm::is_sorted(taskDomain.evidenceObligations, obligationLess) ||
      std::adjacent_find(taskDomain.evidenceObligations.begin(),
                         taskDomain.evidenceObligations.end()) !=
          taskDomain.evidenceObligations.end())
    return invalid("task-domain obligations are not canonical and unique");
  for (EvidenceObligationTemplateRef obligation :
       taskDomain.evidenceObligations)
    if (!llvm::binary_search(binding.evidenceObligations(), obligation,
                             obligationLess))
      return invalid("task-domain obligation is outside the resolved binding");

  std::vector<PromotionAcquisitionInputBinding> providerInputs(
      inputBindings.begin(), inputBindings.end());
  providerInputs[descriptor->candidateInputSlot.ordinal()].artifacts.assign(
      taskDomain.candidates.begin(), taskDomain.candidates.end());

  std::vector<PromotionEvidenceAcquisitionTask> tasks;
  const std::size_t obligationCount = taskDomain.evidenceObligations.size();
  if (obligationCount != 0 &&
      taskDomain.candidates.size() >
          std::numeric_limits<std::size_t>::max() / obligationCount)
    return invalid("acquisition task count overflows size_t");

  std::vector<std::vector<EvidenceAcquisitionInputBinding>> obligationInputs;
  obligationInputs.reserve(obligationCount);
  for (EvidenceObligationTemplateRef obligationRef :
       taskDomain.evidenceObligations) {
    if (obligationRef.ordinal() >= evidenceObligationTemplates.size())
      return invalid("acquisition references a foreign Evidence obligation");
    const EvidenceObligationTemplate &obligation =
        evidenceObligationTemplates[obligationRef.ordinal()];
    std::vector<std::uint32_t> slots;
    slots.reserve(obligation.inputSubjectBindings().size());
    for (const InputSubjectBinding &subject : obligation.inputSubjectBindings())
      slots.push_back(subject.inputSlot.ordinal());
    llvm::sort(slots);
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());

    std::vector<EvidenceAcquisitionInputBinding> taskInputs;
    taskInputs.reserve(slots.size());
    for (std::uint32_t slot : slots) {
      if (slot >= inputBindings.size())
        return invalid("template references an unavailable acquisition slot");
      taskInputs.push_back({EvidenceAcquisitionInputSlotRef(slot),
                            providerInputs[slot].artifacts});
    }
    obligationInputs.push_back(std::move(taskInputs));
  }

  tasks.reserve(taskDomain.candidates.size() * obligationCount);
  for (const ArtifactRootReference &candidate : taskDomain.candidates) {
    for (std::size_t index = 0; index < obligationCount; ++index) {
      const EvidenceObligationTemplateRef obligationRef =
          taskDomain.evidenceObligations[index];
      tasks.push_back({obligationRef,
                       &evidenceObligationTemplates[obligationRef.ordinal()],
                       candidate, obligationInputs[index]});
    }
  }

  if (taskDomain.candidates.empty())
    return PromotionAcquisitionOutcome{CompletedPromotionAcquisition{}};

  PromotionAcquisitionProviderFunction resolve = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(providerMutex());
    auto found =
        llvm::lower_bound(providers(), binding.descriptorRef().kind(),
                          [](const PromotionAcquisitionProvider &provider,
                             PromotionAcquisitionKind kind) {
                            return provider.descriptor.kind() < kind;
                          });
    if (found != providers().end() &&
        found->descriptor == binding.descriptorRef())
      resolve = found->resolve;
  }
  if (!resolve)
    return PromotionAcquisitionOutcome{IncompletePromotionAcquisition{
        PromotionAcquisitionIncompleteReason::ProviderUnavailable, {}}};

  auto resolution = resolve(binding, providerInputs, tasks, store);
  if (!resolution)
    return resolution.takeError();
  if (auto *incomplete =
          std::get_if<IncompletePromotionAcquisitionResolution>(&*resolution)) {
    if (static_cast<std::uint32_t>(incomplete->reason) >
        static_cast<std::uint32_t>(
            PromotionAcquisitionIncompleteReason::Unsupported))
      return invalid("provider returned an invalid Incomplete reason");
    return PromotionAcquisitionOutcome{
        IncompletePromotionAcquisition{incomplete->reason, {}}};
  }

  auto &completed =
      std::get<CompletedPromotionAcquisitionResolution>(*resolution);
  if (completed.tasks.size() != tasks.size())
    return invalid("provider did not resolve every acquisition task");
  std::vector<PromotionEvidenceExecutionTask> executionTasks;
  executionTasks.reserve(tasks.size());
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    const PromotionEvidenceAcquisitionTask &task = tasks[index];
    ResolvedPromotionEvidenceAcquisitionTask &resolved = completed.tasks[index];
    if (!resolved.resolution)
      return invalid("provider returned an absent case resolution");
    llvm::ArrayRef<EvidenceAcquisitionInputBinding> requestInputs =
        task.inputBindings;
    if (resolved.selectedInputs) {
      if (llvm::Error error = validateSelectedTaskInputs(
              task.inputBindings, *resolved.selectedInputs))
        return std::move(error);
      requestInputs = *resolved.selectedInputs;
    }
    auto request = instantiateEvidenceObligation(
        *task.obligation, task.candidate, requestInputs,
        resolved.replicateIndex, *resolved.resolution, store, blobs);
    if (!request)
      return request.takeError();
    auto requestReference =
        evaluation::publishEvaluationRequest(*request, store);
    if (!requestReference)
      return requestReference.takeError();
    executionTasks.push_back({task.candidate, task.obligationTemplate,
                              std::move(*request), resolved.resolution});
  }

  InProcessPromotionEvidenceExecutor inProcess;
  PromotionEvidenceExecutor &selectedExecutor =
      executor ? *executor
               : static_cast<PromotionEvidenceExecutor &>(inProcess);
  auto executionResults =
      selectedExecutor.execute(executionTasks, store, blobs);
  if (!executionResults)
    return executionResults.takeError();
  if (executionResults->size() != executionTasks.size())
    return invalid("Evidence executor returned the wrong result count");

  std::vector<PromotionEvidence> evidence;
  evidence.reserve(executionTasks.size());
  for (std::size_t index = 0; index != executionTasks.size(); ++index) {
    PromotionEvidenceExecutionTask &task = executionTasks[index];
    PromotionEvidenceExecutionResult &result = (*executionResults)[index];
    if (auto *incomplete =
            std::get_if<PromotionAcquisitionIncompleteReason>(&result))
      return PromotionAcquisitionOutcome{
          IncompletePromotionAcquisition{*incomplete, std::move(evidence)}};
    evidence.emplace_back(
        std::move(task.request),
        std::get<evaluation::EvaluationEvidence>(std::move(result)),
        task.obligationTemplate.ordinal());
  }
  return PromotionAcquisitionOutcome{
      CompletedPromotionAcquisition{std::move(evidence)}};
}

} // namespace loom::dse
