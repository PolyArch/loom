#include "DSE/CandidateGenerator.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace loom::dse {
namespace {

std::vector<const CandidateGeneratorDescriptor *> &descriptors() {
  static std::vector<const CandidateGeneratorDescriptor *> records;
  return records;
}

std::shared_mutex &descriptorMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

std::vector<CandidateGeneratorProvider> &providers() {
  static std::vector<CandidateGeneratorProvider> records;
  return records;
}

std::shared_mutex &providerMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "candidate_generator_invalid: " + message);
}

bool isCanonicalAscii(llvm::StringRef value) {
  return !value.empty() && llvm::all_of(value, [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

bool acceptsSchema(const CandidateGeneratorInputSlotDescriptor &slot,
                   const ArtifactRootReference &artifact) {
  return slot.schema && slot.schema->identity == artifact.schemaIdentity &&
         slot.schema->version == artifact.schemaVersion;
}

bool matchesSchema(const CandidateGeneratorOutputSlotDescriptor &slot,
                   const ArtifactRootReference &artifact) {
  return slot.schema && slot.schema->identity == artifact.schemaIdentity &&
         slot.schema->version == artifact.schemaVersion;
}

llvm::Error canonicalizeOutputBindings(
    const CandidateGeneratorDescriptor &descriptor,
    std::vector<CandidateGeneratorOutputBinding> &bindings,
    bool requireFinalCardinality) {
  if (bindings.size() != descriptor.outputSlots.size())
    return invalid("provider does not bind every output slot");
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    CandidateGeneratorOutputBinding &binding = bindings[index];
    const CandidateGeneratorOutputSlotDescriptor &slot =
        descriptor.outputSlots[index];
    if (binding.slot.ordinal() != index)
      return invalid("provider output bindings must be dense and canonical");
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (!matchesSchema(slot, artifact))
        return invalid("provider output artifact schema does not match slot '" +
                       slot.semanticRole + "'");
    llvm::sort(binding.artifacts, artifactRootReferenceLess);
    binding.artifacts.erase(
        std::unique(binding.artifacts.begin(), binding.artifacts.end()),
        binding.artifacts.end());
    const PlanCardinalityBounds bounds =
        planCardinalityBounds(slot.cardinality);
    if (binding.artifacts.size() > bounds.maximum ||
        (requireFinalCardinality && binding.artifacts.size() < bounds.minimum))
      return invalid("provider output violates slot cardinality");
  }
  return llvm::Error::success();
}

llvm::Error validateDescriptor(const CandidateGeneratorDescriptor &descriptor) {
  if (!isCanonicalAscii(descriptor.spelling))
    return invalid("descriptor spelling must be nonempty canonical ASCII");
  if (!isCanonicalAscii(descriptor.implementationSemanticIdentity))
    return invalid("implementation semantic identity must be nonempty "
                   "canonical ASCII");
  if (descriptor.inputSlots.empty())
    return invalid("descriptor requires at least one typed input slot");
  if (descriptor.outputSlots.empty())
    return invalid("descriptor requires at least one typed output slot");
  if (descriptor.resolvedConfigView.schemaDescriptorBytes.empty() ||
      !descriptor.resolvedConfigView.validateCanonical)
    return invalid("descriptor requires an exact resolved config contract");
  if (static_cast<std::uint32_t>(descriptor.determinism) >
      static_cast<std::uint32_t>(
          CandidateGeneratorDeterminism::IndependentReplicates))
    return invalid("descriptor has an invalid determinism contract");

  for (std::size_t index = 0; index < descriptor.inputSlots.size(); ++index) {
    const CandidateGeneratorInputSlotDescriptor &slot =
        descriptor.inputSlots[index];
    if (slot.slot.ordinal() != index)
      return invalid("input slots must be dense and canonical");
    if (!isCanonicalAscii(slot.semanticRole))
      return invalid("input slot role must be nonempty canonical ASCII");
    if (!slot.schema)
      return invalid("input slot requires one exact schema");
    if (static_cast<std::uint32_t>(slot.role) >
            static_cast<std::uint32_t>(PlanValueRole::SimulationExecutionSet) ||
        static_cast<std::uint32_t>(slot.cardinality) >
            static_cast<std::uint32_t>(PlanValueCardinality::FiniteSet))
      return invalid("input slot has an invalid plan value contract");
  }

  for (std::size_t index = 0; index < descriptor.outputSlots.size(); ++index) {
    const CandidateGeneratorOutputSlotDescriptor &slot =
        descriptor.outputSlots[index];
    if (slot.slot.ordinal() != index)
      return invalid("output slots must be dense and canonical");
    if (!isCanonicalAscii(slot.semanticRole) || !slot.schema)
      return invalid("output slot requires a role and exact schema");
    if (static_cast<std::uint32_t>(slot.role) >
            static_cast<std::uint32_t>(PlanValueRole::SimulationExecutionSet) ||
        static_cast<std::uint32_t>(slot.cardinality) >
            static_cast<std::uint32_t>(PlanValueCardinality::FiniteSet))
      return invalid("output slot has an invalid plan value contract");
  }

  for (std::size_t index = 0; index < descriptor.workUnits.size(); ++index) {
    const CandidateGeneratorWorkUnitDescriptor &unit =
        descriptor.workUnits[index];
    if (unit.unit.ordinal() != index || !isCanonicalAscii(unit.spelling))
      return invalid("work units must be dense canonical records");
  }
  for (std::size_t index = 0; index < descriptor.projectionSlots.size();
       ++index) {
    const CandidateGeneratorProjectionSlotDescriptor &slot =
        descriptor.projectionSlots[index];
    if (slot.slot.ordinal() != index || !isCanonicalAscii(slot.semanticRole))
      return invalid("projection slots must be dense canonical records");
    if (static_cast<std::uint32_t>(slot.kind) >
        static_cast<std::uint32_t>(
            CandidateGeneratorProjectionKind::EvaluationFinding))
      return invalid("projection slot has an invalid kind");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CandidateGeneratorDescriptorRef>
CandidateGeneratorDescriptorRef::get(
    const ArtifactSchemaDescriptor &descriptorSchema,
    CandidateGeneratorKind kind) {
  if (descriptorSchema != candidateGeneratorDescriptorSchema)
    return invalid("candidate generator descriptor schema is unsupported");
  return CandidateGeneratorDescriptorRef(descriptorSchema, kind);
}

const CandidateGeneratorDescriptor *
CandidateGeneratorDescriptorRef::descriptor() const {
  if (descriptorSchema_ != candidateGeneratorDescriptorSchema)
    return nullptr;
  return findCandidateGeneratorDescriptor(kind_);
}

CandidateGeneratorDescriptorRef
CandidateGeneratorDescriptor::reference() const {
  return llvm::cantFail(CandidateGeneratorDescriptorRef::get(
      candidateGeneratorDescriptorSchema, kind));
}

const CandidateGeneratorInputSlotDescriptor *
CandidateGeneratorDescriptor::findInputSlot(
    CandidateGeneratorInputSlotRef slot) const {
  if (slot.ordinal() >= inputSlots.size())
    return nullptr;
  return &inputSlots[slot.ordinal()];
}

const CandidateGeneratorOutputSlotDescriptor *
CandidateGeneratorDescriptor::findOutputSlot(
    CandidateGeneratorOutputSlotRef slot) const {
  if (slot.ordinal() >= outputSlots.size())
    return nullptr;
  return &outputSlots[slot.ordinal()];
}

llvm::Error registerCandidateGeneratorDescriptor(
    const CandidateGeneratorDescriptor &descriptor) {
  if (llvm::Error error = validateDescriptor(descriptor))
    return error;

  std::unique_lock<std::shared_mutex> lock(descriptorMutex());
  for (const CandidateGeneratorDescriptor *existing : descriptors()) {
    if (existing == &descriptor)
      return llvm::Error::success();
    if (existing->kind == descriptor.kind)
      return invalid("conflicting registration for candidate generator kind " +
                     std::to_string(descriptor.kind.ordinal()));
    if (existing->spelling == descriptor.spelling)
      return invalid("conflicting registration for candidate generator '" +
                     descriptor.spelling + "'");
  }
  descriptors().push_back(&descriptor);
  std::sort(descriptors().begin(), descriptors().end(),
            [](const CandidateGeneratorDescriptor *lhs,
               const CandidateGeneratorDescriptor *rhs) {
              return lhs->kind < rhs->kind;
            });
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor *
findCandidateGeneratorDescriptor(CandidateGeneratorKind kind) {
  std::shared_lock<std::shared_mutex> lock(descriptorMutex());
  auto found =
      std::lower_bound(descriptors().begin(), descriptors().end(), kind,
                       [](const CandidateGeneratorDescriptor *descriptor,
                          CandidateGeneratorKind requested) {
                         return descriptor->kind < requested;
                       });
  if (found == descriptors().end() || (*found)->kind != kind)
    return nullptr;
  return *found;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
ResolvedCandidateGeneratorBinding::get(
    CandidateGeneratorDescriptorRef descriptorRef,
    llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
    const ComponentViewDigest &configDigest) {
  const CandidateGeneratorDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");

  if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
          canonicalConfigBytes, configDigest))
    return std::move(error);

  return ResolvedCandidateGeneratorBinding(
      descriptorRef, canonicalConfigBytes.vec(), configDigest);
}

llvm::Error validateCandidateGeneratorInputBindings(
    CandidateGeneratorDescriptorRef descriptorRef,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings) {
  const CandidateGeneratorDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("input bindings reference an unregistered descriptor");
  if (inputBindings.size() != descriptor->inputSlots.size())
    return invalid("binding does not provide every descriptor input slot");

  for (std::size_t index = 0; index < inputBindings.size(); ++index) {
    const CandidateGeneratorInputBinding &binding = inputBindings[index];
    const CandidateGeneratorInputSlotDescriptor &slot =
        descriptor->inputSlots[index];
    if (binding.slot.ordinal() != index)
      return invalid("input bindings must be dense and canonical");
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (!acceptsSchema(slot, artifact))
        return invalid("input slot '" + slot.semanticRole +
                       "' does not accept artifact schema '" +
                       artifact.schemaIdentity + "'");
    if (!llvm::is_sorted(binding.artifacts, artifactRootReferenceLess) ||
        std::adjacent_find(binding.artifacts.begin(),
                           binding.artifacts.end()) != binding.artifacts.end())
      return invalid("input artifact sets must be canonical");
    if (!planCardinalityContains(slot.cardinality, binding.artifacts.size()))
      return invalid("canonical input set violates descriptor cardinality");
  }
  return llvm::Error::success();
}

llvm::Error
registerCandidateGeneratorProvider(const CandidateGeneratorProvider &provider) {
  if (!provider.invoke || !provider.descriptor.descriptor())
    return invalid("provider requires a registered descriptor and callback");
  std::unique_lock<std::shared_mutex> lock(providerMutex());
  for (const CandidateGeneratorProvider &existing : providers()) {
    if (existing.descriptor != provider.descriptor)
      continue;
    if (existing.invoke == provider.invoke)
      return llvm::Error::success();
    return invalid("conflicting provider registration for candidate generator");
  }
  providers().push_back(provider);
  llvm::sort(providers(), [](const CandidateGeneratorProvider &lhs,
                             const CandidateGeneratorProvider &rhs) {
    return lhs.descriptor.kind() < rhs.descriptor.kind();
  });
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorInvocationOutcome> invokeCandidateGenerator(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store) {
  const CandidateGeneratorDescriptor *descriptor =
      binding.descriptorRef().descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          binding.descriptorRef(), inputBindings))
    return std::move(error);

  CandidateGeneratorProviderFunction invoke = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(providerMutex());
    auto found =
        llvm::lower_bound(providers(), binding.descriptorRef().kind(),
                          [](const CandidateGeneratorProvider &provider,
                             CandidateGeneratorKind kind) {
                            return provider.descriptor.kind() < kind;
                          });
    if (found != providers().end() &&
        found->descriptor == binding.descriptorRef())
      invoke = found->invoke;
  }
  if (!invoke) {
    std::vector<CandidateGeneratorOutputBinding> outputs;
    outputs.reserve(descriptor->outputSlots.size());
    for (const CandidateGeneratorOutputSlotDescriptor &slot :
         descriptor->outputSlots)
      outputs.push_back({slot.slot, {}});
    return CandidateGeneratorInvocationOutcome{
        IncompleteCandidateGeneratorInvocation{
            CandidateGeneratorIncompleteReason::ProviderUnavailable,
            std::move(outputs)}};
  }

  auto outcome = invoke(inputBindings, binding, store);
  if (!outcome)
    return outcome.takeError();
  if (auto *completed =
          std::get_if<CompletedCandidateGeneratorInvocation>(&*outcome)) {
    if (llvm::Error error = canonicalizeOutputBindings(
            *descriptor, completed->outputBindings, true))
      return std::move(error);
  } else {
    auto &incomplete =
        std::get<IncompleteCandidateGeneratorInvocation>(*outcome);
    if (static_cast<std::uint32_t>(incomplete.reason) >
        static_cast<std::uint32_t>(
            CandidateGeneratorIncompleteReason::Unsupported))
      return invalid("provider returned an invalid Incomplete reason");
    if (llvm::Error error = canonicalizeOutputBindings(
            *descriptor, incomplete.retainedOutputBindings, false))
      return std::move(error);
  }
  return outcome;
}

} // namespace loom::dse
