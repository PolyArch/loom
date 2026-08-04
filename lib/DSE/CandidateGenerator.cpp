#include "DSE/CandidateGenerator.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace loom::dse {
namespace {

std::vector<const CandidateGeneratorDescriptor *> &descriptors() {
  static std::vector<const CandidateGeneratorDescriptor *> records;
  return records;
}

std::mutex &descriptorMutex() {
  static std::mutex mutex;
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

bool schemaLess(const ArtifactSchemaDescriptor *lhs,
                const ArtifactSchemaDescriptor *rhs) {
  if (lhs->identity != rhs->identity)
    return lhs->identity < rhs->identity;
  if (lhs->version.major != rhs->version.major)
    return lhs->version.major < rhs->version.major;
  return lhs->version.minor < rhs->version.minor;
}

bool acceptsSchema(const CandidateGeneratorInputSlotDescriptor &slot,
                   const ArtifactRootReference &artifact) {
  return llvm::any_of(slot.acceptedSchemas,
                      [&](const ArtifactSchemaDescriptor *schema) {
                        return schema->identity == artifact.schemaIdentity &&
                               schema->version == artifact.schemaVersion;
                      });
}

llvm::Error validateBounds(llvm::StringRef owner,
                           ArtifactCollectionBounds bounds) {
  if (bounds.minimum > bounds.maximum)
    return invalid(owner + " has an inverted artifact cardinality");
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
    if (slot.acceptedSchemas.empty())
      return invalid("input slot requires at least one accepted schema");
    if (llvm::any_of(slot.acceptedSchemas,
                     [](const ArtifactSchemaDescriptor *schema) {
                       return schema == nullptr;
                     }))
      return invalid("input slot contains a null schema");
    if (!std::is_sorted(slot.acceptedSchemas.begin(),
                        slot.acceptedSchemas.end(), schemaLess) ||
        std::adjacent_find(slot.acceptedSchemas.begin(),
                           slot.acceptedSchemas.end(),
                           [](const ArtifactSchemaDescriptor *lhs,
                              const ArtifactSchemaDescriptor *rhs) {
                             return *lhs == *rhs;
                           }) != slot.acceptedSchemas.end())
      return invalid("input schemas must be canonical without duplicates");
    if (llvm::Error error = validateBounds("input slot", slot.cardinality))
      return error;
  }

  for (std::size_t index = 0; index < descriptor.outputSlots.size(); ++index) {
    const CandidateGeneratorOutputSlotDescriptor &slot =
        descriptor.outputSlots[index];
    if (slot.slot.ordinal() != index)
      return invalid("output slots must be dense and canonical");
    if (!isCanonicalAscii(slot.semanticRole) || !slot.schema)
      return invalid("output slot requires a role and exact schema");
    if (llvm::Error error = validateBounds("output slot", slot.cardinality))
      return error;
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

  std::lock_guard<std::mutex> lock(descriptorMutex());
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
  std::lock_guard<std::mutex> lock(descriptorMutex());
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
    std::vector<CandidateGeneratorInputBinding> inputBindings,
    llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
    const ComponentViewDigest &configDigest) {
  const CandidateGeneratorDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");
  if (inputBindings.size() != descriptor->inputSlots.size())
    return invalid("binding does not provide every descriptor input slot");

  for (std::size_t index = 0; index < inputBindings.size(); ++index) {
    const CandidateGeneratorInputBinding &binding = inputBindings[index];
    const CandidateGeneratorInputSlotDescriptor &slot =
        descriptor->inputSlots[index];
    if (binding.slot.ordinal() != index)
      return invalid("input bindings must be dense and canonical");
    if (!slot.cardinality.contains(binding.artifacts.size()))
      return invalid("input binding violates descriptor cardinality");
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (!acceptsSchema(slot, artifact))
        return invalid("input slot '" + slot.semanticRole +
                       "' does not accept artifact schema '" +
                       artifact.schemaIdentity + "'");
  }

  if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
          canonicalConfigBytes, configDigest))
    return std::move(error);

  return ResolvedCandidateGeneratorBinding(
      descriptorRef, std::move(inputBindings), canonicalConfigBytes.vec(),
      configDigest);
}

const CandidateGeneratorInputBinding *
ResolvedCandidateGeneratorBinding::findInputBinding(
    CandidateGeneratorInputSlotRef slot) const {
  if (slot.ordinal() >= inputBindings_.size())
    return nullptr;
  return &inputBindings_[slot.ordinal()];
}

} // namespace loom::dse
