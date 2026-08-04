#include "DSE/PromotionAcquisition.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
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
  for (std::size_t index = 0; index < descriptor.inputSlots.size(); ++index) {
    const PromotionAcquisitionInputSlotDescriptor &slot =
        descriptor.inputSlots[index];
    if (slot.ref.ordinal() != index)
      return invalid("input slots must be dense and canonical");
    if (!isCanonicalAscii(slot.spelling) || !slot.schema ||
        !validRole(slot.role) || !validCardinality(slot.cardinality))
      return invalid("input slot has an invalid typed contract");
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
    std::vector<PromotionAcquisitionInputBinding> inputBindings,
    llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
    const ComponentViewDigest &configDigest) {
  const PromotionAcquisitionDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");
  if (inputBindings.size() != descriptor->inputSlots.size())
    return invalid("binding does not provide every descriptor input slot");

  for (std::size_t index = 0; index < inputBindings.size(); ++index) {
    PromotionAcquisitionInputBinding &binding = inputBindings[index];
    const PromotionAcquisitionInputSlotDescriptor &slot =
        descriptor->inputSlots[index];
    if (binding.slot.ordinal() != index)
      return invalid("input bindings must be dense and canonical");
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (!acceptsSchema(slot, artifact))
        return invalid("input artifact schema does not match slot '" +
                       slot.spelling + "'");
    llvm::sort(binding.artifacts, artifactRootReferenceLess);
    binding.artifacts.erase(
        std::unique(binding.artifacts.begin(), binding.artifacts.end()),
        binding.artifacts.end());
    if (!planCardinalityContains(slot.cardinality, binding.artifacts.size()))
      return invalid("canonical input set violates descriptor cardinality");
  }
  if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
          canonicalConfigBytes, configDigest))
    return std::move(error);
  return ResolvedPromotionAcquisitionBinding(
      descriptorRef, std::move(inputBindings), canonicalConfigBytes.vec(),
      configDigest);
}

const PromotionAcquisitionInputBinding *
ResolvedPromotionAcquisitionBinding::findInputBinding(
    PromotionAcquisitionInputSlotRef slot) const {
  if (slot.ordinal() >= inputBindings_.size())
    return nullptr;
  return &inputBindings_[slot.ordinal()];
}

llvm::Error registerPromotionAcquisitionProvider(
    const PromotionAcquisitionProvider &provider) {
  if (!provider.acquire || !provider.descriptor.descriptor())
    return invalid("provider requires a registered descriptor and callback");
  std::unique_lock<std::shared_mutex> lock(providerMutex());
  for (const PromotionAcquisitionProvider &existing : providers()) {
    if (existing.descriptor != provider.descriptor)
      continue;
    if (existing.acquire == provider.acquire)
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

llvm::Expected<PromotionAcquisitionOutcome>
invokePromotionAcquisition(const ResolvedPromotionAcquisitionBinding &binding,
                           const ArtifactStore &store) {
  if (!binding.descriptorRef().descriptor())
    return invalid("binding references an unregistered descriptor");
  PromotionAcquisitionProviderFunction acquire = nullptr;
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
      acquire = found->acquire;
  }
  if (!acquire)
    return PromotionAcquisitionOutcome{IncompletePromotionAcquisition{
        PromotionAcquisitionIncompleteReason::ProviderUnavailable, {}}};

  auto outcome = acquire(binding, store);
  if (!outcome)
    return outcome.takeError();
  if (auto *incomplete =
          std::get_if<IncompletePromotionAcquisition>(&*outcome)) {
    if (static_cast<std::uint32_t>(incomplete->reason) >
        static_cast<std::uint32_t>(
            PromotionAcquisitionIncompleteReason::Unsupported))
      return invalid("provider returned an invalid Incomplete reason");
  }
  return outcome;
}

} // namespace loom::dse
