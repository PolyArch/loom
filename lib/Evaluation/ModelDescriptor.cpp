#include "Evaluation/ModelDescriptor.h"

#include "CanonicalSupport.h"
#include "Evaluation/Request.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::appendFramedString;
using detail::appendSchemaVersion;
using detail::appendU32Be;
using detail::appendU64Be;
using detail::evaluationError;

std::vector<const EvaluationModelDescriptor *> &modelDescriptors() {
  static std::vector<const EvaluationModelDescriptor *> descriptors;
  return descriptors;
}

std::mutex &modelDescriptorMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<const EvaluationInteractionDomainDescriptor *> &
interactionDomains() {
  static std::vector<const EvaluationInteractionDomainDescriptor *> domains;
  return domains;
}

std::mutex &interactionDomainMutex() {
  static std::mutex mutex;
  return mutex;
}

bool isCanonicalAscii(llvm::StringRef value) {
  if (value.empty())
    return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

bool isValidModeledPhenomenon(ModeledPhenomenon phenomenon) {
  return static_cast<std::uint32_t>(phenomenon) <=
         static_cast<std::uint32_t>(ModeledPhenomenon::PhysicalImplementation);
}

bool isValidExecutionMethod(EvaluationExecutionMethod method) {
  return static_cast<std::uint32_t>(method) <=
         static_cast<std::uint32_t>(
             EvaluationExecutionMethod::PhysicalMeasurement);
}

bool isValidInteractionMode(EvaluationInteractionMode mode) {
  return static_cast<std::uint32_t>(mode) <=
         static_cast<std::uint32_t>(EvaluationInteractionMode::Guidance);
}

bool supportsInteractionMode(
    const EvaluationInteractionDomainDescriptor &descriptor,
    EvaluationInteractionMode requested) {
  return std::find(descriptor.implementedModes.begin(),
                   descriptor.implementedModes.end(),
                   requested) != descriptor.implementedModes.end();
}

llvm::Error validateInteractionModes(
    llvm::StringRef owner, llvm::ArrayRef<EvaluationInteractionMode> modes) {
  if (modes.empty())
    return evaluationError(owner + " requires at least one interaction mode");
  for (std::size_t index = 0; index < modes.size(); ++index) {
    if (!isValidInteractionMode(modes[index]))
      return evaluationError(owner + " has an invalid interaction mode");
    if (index != 0 &&
        static_cast<std::uint32_t>(modes[index - 1]) >=
            static_cast<std::uint32_t>(modes[index]))
      return evaluationError(owner +
                             " interaction modes must be canonical");
  }
  return llvm::Error::success();
}

bool schemaLess(const ArtifactSchemaDescriptor *lhs,
                const ArtifactSchemaDescriptor *rhs) {
  if (lhs->identity != rhs->identity)
    return lhs->identity < rhs->identity;
  if (lhs->version.major != rhs->version.major)
    return lhs->version.major < rhs->version.major;
  return lhs->version.minor < rhs->version.minor;
}

bool acceptsSchema(const ModelInputSlotDescriptor &slot,
                   const ArtifactRootReference &artifact) {
  return std::any_of(slot.acceptedSchemas.begin(), slot.acceptedSchemas.end(),
                     [&](const ArtifactSchemaDescriptor *schema) {
                       return schema->identity == artifact.schemaIdentity &&
                              schema->version == artifact.schemaVersion;
                     });
}

bool supportsScopeForm(llvm::ArrayRef<ScopeFormRef> forms,
                       ScopeFormRef requested) {
  return std::find(forms.begin(), forms.end(), requested) != forms.end();
}

llvm::Error validateScopeCapabilities(const llvm::Twine &owner,
                                      llvm::ArrayRef<ScopeFormRef> forms,
                                      std::size_t availableForms) {
  if (forms.empty())
    return evaluationError(owner + " requires at least one scope form");
  for (std::size_t index = 0; index < forms.size(); ++index) {
    if (forms[index].ordinal() >= availableForms)
      return evaluationError(owner + " references an unknown scope form");
    if (index != 0 && forms[index - 1].ordinal() >= forms[index].ordinal())
      return evaluationError(owner +
                             " scope forms must be in canonical order without "
                             "duplicates");
  }
  return llvm::Error::success();
}

bool patternPermittedByOwners(const EvaluationModelDescriptor &descriptor,
                              const ConditionApplicabilityPattern &pattern) {
  const EvaluationCaseSignatureDescriptor *signature =
      descriptor.caseSignature.descriptor();
  if (std::find(signature->permittedBaseConditions.begin(),
                signature->permittedBaseConditions.end(),
                pattern) != signature->permittedBaseConditions.end())
    return true;

  for (const MetricCapability &capability : descriptor.metricCapabilities) {
    const auto patterns =
        metricDescriptor(capability.kind).permittedRequestConditionPatterns;
    if (std::find(patterns.begin(), patterns.end(), pattern) != patterns.end())
      return true;
  }
  for (const FindingCapability &capability : descriptor.findingCapabilities) {
    const FindingDescriptor *finding = findFindingDescriptor(capability.kind);
    if (!finding)
      continue;
    if (std::find(finding->permittedRequestConditionPatterns.begin(),
                  finding->permittedRequestConditionPatterns.end(),
                  pattern) != finding->permittedRequestConditionPatterns.end())
      return true;
  }
  return false;
}

llvm::Error validateDescriptor(const EvaluationModelDescriptor &descriptor) {
  if (descriptor.spelling.empty())
    return evaluationError("an Evaluation model descriptor requires a "
                           "spelling");
  if (descriptor.implementationSemanticIdentity.empty())
    return evaluationError("model '" + descriptor.spelling +
                           "' requires an implementation semantic identity");
  if (!descriptor.caseSignature.descriptor())
    return evaluationError("model '" + descriptor.spelling +
                           "' references an unregistered case signature");
  if (descriptor.resolvedConfigView.schemaDescriptorBytes.empty() ||
      !descriptor.resolvedConfigView.project ||
      !descriptor.resolvedConfigView.encode ||
      !descriptor.resolvedConfigView.adopt)
    return evaluationError("model '" + descriptor.spelling +
                           "' requires a complete resolved-config view "
                           "contract");

  for (std::size_t index = 0; index < descriptor.modeledPhenomena.size();
       ++index) {
    const ModeledPhenomenon phenomenon = descriptor.modeledPhenomena[index];
    if (!isValidModeledPhenomenon(phenomenon))
      return evaluationError("model '" + descriptor.spelling +
                             "' has an invalid modeled phenomenon");
    if (index != 0 &&
        static_cast<std::uint32_t>(descriptor.modeledPhenomena[index - 1]) >=
            static_cast<std::uint32_t>(phenomenon))
      return evaluationError("model '" + descriptor.spelling +
                             "' modeled phenomena must be canonical");
  }
  if (!isValidExecutionMethod(descriptor.executionMethod))
    return evaluationError("model '" + descriptor.spelling +
                           "' has an invalid execution method");
  for (std::size_t index = 0;
       index < descriptor.interactionCapabilities.size(); ++index) {
    const EvaluationInteractionCapability &capability =
        descriptor.interactionCapabilities[index];
    if (index != 0 &&
        !(descriptor.interactionCapabilities[index - 1].domain <
          capability.domain))
      return evaluationError("model '" + descriptor.spelling +
                             "' interaction capabilities must be canonical");
    if (llvm::Error error = validateInteractionModes(
            "model interaction capability", capability.modes))
      return error;
    const EvaluationInteractionDomainDescriptor *domain =
        findEvaluationInteractionDomain(capability.domain);
    if (!domain)
      return evaluationError("model '" + descriptor.spelling +
                             "' references an unregistered interaction "
                             "domain");
    for (EvaluationInteractionMode mode : capability.modes) {
      if (!supportsInteractionMode(*domain, mode))
        return evaluationError("interaction domain '" +
                               capability.domain.ownerRegistryIdentity() +
                               "' does not implement interaction mode '" +
                               toString(mode) + "'");
      if (llvm::Error error = domain->validateTypedProtocol(mode))
        return error;
    }
  }

  for (std::size_t index = 0; index < descriptor.conditionCapabilities.size();
       ++index) {
    const ModelConditionCapability &capability =
        descriptor.conditionCapabilities[index];
    if (index != 0 && !conditionApplicabilityPatternLess(
                          descriptor.conditionCapabilities[index - 1].pattern,
                          capability.pattern))
      return evaluationError("model '" + descriptor.spelling +
                             "' condition capabilities must be canonical");
    if (capability.pattern.targets.caseSignature != descriptor.caseSignature)
      return evaluationError("model '" + descriptor.spelling +
                             "' declares a condition capability for a foreign "
                             "case signature");
    if (!patternPermittedByOwners(descriptor, capability.pattern))
      return evaluationError("model '" + descriptor.spelling +
                             "' widens condition applicability beyond its "
                             "case, metric, and finding owners");
  }

  for (std::size_t index = 0; index < descriptor.metricCapabilities.size();
       ++index) {
    const MetricCapability &capability = descriptor.metricCapabilities[index];
    if (index != 0 &&
        !(descriptor.metricCapabilities[index - 1].kind < capability.kind))
      return evaluationError("model '" + descriptor.spelling +
                             "' metric capabilities must be in canonical "
                             "order without duplicates");
    const MetricDescriptor &metric = metricDescriptor(capability.kind);
    if (llvm::Error error = validateScopeFormDescriptors(metric.scopeForms))
      return error;
    if (llvm::Error error = validateScopeCapabilities(
            "metric capability '" + metric.spelling + "'",
            capability.scopeForms, metric.scopeForms.size()))
      return error;
    for (ScopeFormRef form : capability.scopeForms)
      if (llvm::Error error = validateMetricScopeAdmissibility(
              capability.kind, form, *descriptor.caseSignature.descriptor()))
        return error;
    if (capability.permittedObservationForms == 0 ||
        (capability.permittedObservationForms &
         ~metric.permittedObservationForms) != 0)
      return evaluationError("model '" + descriptor.spelling +
                             "' declares invalid observation forms for metric "
                             "'" +
                             metric.spelling + "'");
  }

  for (std::size_t index = 0; index < descriptor.findingCapabilities.size();
       ++index) {
    const FindingCapability &capability = descriptor.findingCapabilities[index];
    if (index != 0 &&
        !(descriptor.findingCapabilities[index - 1].kind < capability.kind))
      return evaluationError("model '" + descriptor.spelling +
                             "' finding capabilities must be in canonical "
                             "order without duplicates");
    const FindingDescriptor *finding = findFindingDescriptor(capability.kind);
    if (!finding)
      return evaluationError("model '" + descriptor.spelling +
                             "' references an unregistered finding kind");
    if (llvm::Error error = validateScopeCapabilities(
            "finding capability '" + finding->spelling + "'",
            capability.scopeForms, finding->scopeForms.size()))
      return error;
    if (capability.permittedResultForms == 0 ||
        (capability.permittedResultForms & ~allFindingResultFormsMask()) != 0)
      return evaluationError("model '" + descriptor.spelling +
                             "' declares invalid result forms for finding '" +
                             finding->spelling + "'");
  }

  for (std::size_t index = 0; index < descriptor.inputSlots.size(); ++index) {
    const ModelInputSlotDescriptor &slot = descriptor.inputSlots[index];
    if (slot.slot.ordinal() != index)
      return evaluationError("model '" + descriptor.spelling +
                             "' input-slot ordinals must be contiguous");
    if (slot.semanticRole.empty() || slot.acceptedSchemas.empty())
      return evaluationError("model '" + descriptor.spelling +
                             "' input slots require a role and accepted "
                             "schemas");
    if (slot.cardinality == ArtifactCollectionCardinality::Forbidden)
      return evaluationError("model '" + descriptor.spelling +
                             "' must omit forbidden input slots");
    for (std::size_t schema = 0; schema < slot.acceptedSchemas.size();
         ++schema) {
      if (!slot.acceptedSchemas[schema])
        return evaluationError("model '" + descriptor.spelling +
                               "' input slot has a null schema");
      if (schema != 0 && !schemaLess(slot.acceptedSchemas[schema - 1],
                                     slot.acceptedSchemas[schema]))
        return evaluationError("model '" + descriptor.spelling +
                               "' input schemas must be canonical");
    }
    for (std::size_t previous = 0; previous < index; ++previous)
      if (descriptor.inputSlots[previous].semanticRole == slot.semanticRole)
        return evaluationError("model '" + descriptor.spelling +
                               "' input-slot roles must be unique");
  }

  for (std::size_t index = 0; index < descriptor.outputSlots.size(); ++index) {
    const ModelOutputSlotDescriptor &slot = descriptor.outputSlots[index];
    if (slot.slot.ordinal() != index)
      return evaluationError("model '" + descriptor.spelling +
                             "' output-slot ordinals must be contiguous");
    if (slot.semanticRole.empty() || !slot.schema)
      return evaluationError("model '" + descriptor.spelling +
                             "' output slots require a role and exact schema");
    for (std::size_t previous = 0; previous < index; ++previous)
      if (descriptor.outputSlots[previous].semanticRole == slot.semanticRole)
        return evaluationError("model '" + descriptor.spelling +
                               "' output-slot roles must be unique");
  }

  for (std::size_t index = 0;
       index < descriptor.mandatoryTerminalFindings.size(); ++index) {
    const FindingQuery &query = descriptor.mandatoryTerminalFindings[index];
    if (llvm::Error error = validateFindingQuery(query))
      return error;
    if (!descriptor.supportsFindingQuery(query))
      return evaluationError("model '" + descriptor.spelling +
                             "' requires an unsupported terminal finding");
    if (index != 0 && !(canonicalFindingQueryKey(
                            descriptor.mandatoryTerminalFindings[index - 1]) <
                        canonicalFindingQueryKey(query)))
      return evaluationError("model '" + descriptor.spelling +
                             "' mandatory terminal findings must be "
                             "canonical");
  }
  return llvm::Error::success();
}

llvm::Error requireRecognizedConditions(
    const EvaluationModelDescriptor &descriptor,
    llvm::ArrayRef<EvaluationCondition> conditions,
    EvaluationCaseSignatureRef signature,
    llvm::SmallVectorImpl<ConditionApplicabilityPattern> &present) {
  for (const EvaluationCondition &condition : conditions) {
    const ConditionApplicabilityPattern derived =
        deriveConditionApplicabilityPattern(condition, signature);
    if (!descriptor.findConditionCapability(derived))
      return evaluationError("model '" + descriptor.spelling +
                             "' does not recognize condition '" +
                             toString(condition.kind()) + "'");
    present.push_back(derived);
  }
  return llvm::Error::success();
}

} // namespace

llvm::StringRef toString(ModeledPhenomenon phenomenon) {
  switch (phenomenon) {
  case ModeledPhenomenon::StructuredProgram:
    return "structured_program";
  case ModeledPhenomenon::CanonicalDataflow:
    return "canonical_dataflow";
  case ModeledPhenomenon::SpatialResources:
    return "spatial_resources";
  case ModeledPhenomenon::RoutedTransport:
    return "routed_transport";
  case ModeledPhenomenon::FiniteBuffering:
    return "finite_buffering";
  case ModeledPhenomenon::MemoryContention:
    return "memory_contention";
  case ModeledPhenomenon::ClockTiming:
    return "clock_timing";
  case ModeledPhenomenon::SystemMemoryHierarchy:
    return "system_memory_hierarchy";
  case ModeledPhenomenon::Coherence:
    return "coherence";
  case ModeledPhenomenon::RTLBehavior:
    return "rtl_behavior";
  case ModeledPhenomenon::PhysicalImplementation:
    return "physical_implementation";
  }
  llvm_unreachable("unknown ModeledPhenomenon");
}

llvm::StringRef toString(EvaluationExecutionMethod method) {
  switch (method) {
  case EvaluationExecutionMethod::Analytic:
    return "analytic";
  case EvaluationExecutionMethod::Simulation:
    return "simulation";
  case EvaluationExecutionMethod::Emulation:
    return "emulation";
  case EvaluationExecutionMethod::ToolMeasurement:
    return "tool_measurement";
  case EvaluationExecutionMethod::PhysicalMeasurement:
    return "physical_measurement";
  }
  llvm_unreachable("unknown EvaluationExecutionMethod");
}

llvm::StringRef toString(EvaluationInteractionMode mode) {
  switch (mode) {
  case EvaluationInteractionMode::Incremental:
    return "incremental";
  case EvaluationInteractionMode::Guidance:
    return "guidance";
  }
  llvm_unreachable("unknown EvaluationInteractionMode");
}

llvm::Expected<EvaluationInteractionDomainRef>
EvaluationInteractionDomainRef::get(llvm::StringRef ownerRegistryIdentity,
                                    SchemaVersion ownerRegistryVersion,
                                    std::uint32_t ownerLocalDomainKind) {
  if (!isCanonicalAscii(ownerRegistryIdentity))
    return evaluationError(
        "interaction domain owner identity must be nonempty canonical ASCII");
  return EvaluationInteractionDomainRef(ownerRegistryIdentity.str(),
                                        ownerRegistryVersion,
                                        ownerLocalDomainKind);
}

bool operator<(const EvaluationInteractionDomainRef &lhs,
               const EvaluationInteractionDomainRef &rhs) {
  if (lhs.ownerRegistryIdentity_ != rhs.ownerRegistryIdentity_)
    return lhs.ownerRegistryIdentity_ < rhs.ownerRegistryIdentity_;
  if (lhs.ownerRegistryVersion_.major != rhs.ownerRegistryVersion_.major)
    return lhs.ownerRegistryVersion_.major < rhs.ownerRegistryVersion_.major;
  if (lhs.ownerRegistryVersion_.minor != rhs.ownerRegistryVersion_.minor)
    return lhs.ownerRegistryVersion_.minor < rhs.ownerRegistryVersion_.minor;
  return lhs.ownerLocalDomainKind_ < rhs.ownerLocalDomainKind_;
}

llvm::Error registerEvaluationInteractionDomain(
    const EvaluationInteractionDomainDescriptor &descriptor) {
  if (descriptor.semanticDefinition.empty() ||
      !descriptor.validateTypedProtocol)
    return evaluationError(
        "an interaction domain requires a semantic definition and typed "
        "protocol validator");
  if (llvm::Error error = validateInteractionModes(
          "interaction domain", descriptor.implementedModes))
    return error;
  for (EvaluationInteractionMode mode : descriptor.implementedModes)
    if (llvm::Error error = descriptor.validateTypedProtocol(mode))
      return error;

  std::lock_guard<std::mutex> lock(interactionDomainMutex());
  for (const EvaluationInteractionDomainDescriptor *existing :
       interactionDomains()) {
    if (existing->reference == descriptor.reference) {
      if (existing == &descriptor)
        return llvm::Error::success();
      return evaluationError("conflicting registration for interaction "
                             "domain '" +
                             descriptor.reference.ownerRegistryIdentity() +
                             "'");
    }
  }
  interactionDomains().push_back(&descriptor);
  std::sort(interactionDomains().begin(), interactionDomains().end(),
            [](const EvaluationInteractionDomainDescriptor *lhs,
               const EvaluationInteractionDomainDescriptor *rhs) {
              return lhs->reference < rhs->reference;
            });
  return llvm::Error::success();
}

const EvaluationInteractionDomainDescriptor *
findEvaluationInteractionDomain(const EvaluationInteractionDomainRef &domain) {
  std::lock_guard<std::mutex> lock(interactionDomainMutex());
  for (const EvaluationInteractionDomainDescriptor *descriptor :
       interactionDomains())
    if (descriptor->reference == domain)
      return descriptor;
  return nullptr;
}

llvm::Expected<EvaluationModelDescriptorRef>
EvaluationModelDescriptorRef::get(SchemaVersion schemaVersion,
                                  EvaluationModelKind modelKind) {
  if (schemaVersion != evaluationSchemaVersion())
    return evaluationError("unsupported Evaluation model descriptor version");
  return EvaluationModelDescriptorRef(schemaVersion, modelKind);
}

const EvaluationModelDescriptor *
EvaluationModelDescriptorRef::descriptor() const {
  return findEvaluationModelDescriptor(modelKind_);
}

llvm::StringRef toString(EvidenceOutcomeKind outcome) {
  switch (outcome) {
  case EvidenceOutcomeKind::Completed:
    return "completed";
  case EvidenceOutcomeKind::Unsupported:
    return "unsupported";
  case EvidenceOutcomeKind::ExecutionFailed:
    return "execution_failed";
  case EvidenceOutcomeKind::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown EvidenceOutcomeKind");
}

llvm::Error
validateArtifactCollectionCardinality(ArtifactCollectionCardinality cardinality,
                                      std::size_t count,
                                      llvm::StringRef owner) {
  bool valid = false;
  switch (cardinality) {
  case ArtifactCollectionCardinality::Forbidden:
    valid = count == 0;
    break;
  case ArtifactCollectionCardinality::ZeroOrOne:
    valid = count <= 1;
    break;
  case ArtifactCollectionCardinality::ExactlyOne:
    valid = count == 1;
    break;
  case ArtifactCollectionCardinality::OneOrMore:
    valid = count >= 1;
    break;
  }
  if (!valid)
    return evaluationError(owner + " violates its declared cardinality");
  return llvm::Error::success();
}

EvaluationModelDescriptorRef EvaluationModelDescriptor::reference() const {
  return llvm::cantFail(
      EvaluationModelDescriptorRef::get(evaluationSchemaVersion(), modelKind));
}

const ModelConditionCapability *
EvaluationModelDescriptor::findConditionCapability(
    const ConditionApplicabilityPattern &pattern) const {
  for (const ModelConditionCapability &capability : conditionCapabilities)
    if (capability.pattern == pattern)
      return &capability;
  return nullptr;
}

const MetricCapability *
EvaluationModelDescriptor::findMetricCapability(MetricKind metric) const {
  for (const MetricCapability &capability : metricCapabilities)
    if (capability.kind == metric)
      return &capability;
  return nullptr;
}

const FindingCapability *
EvaluationModelDescriptor::findFindingCapability(FindingKind finding) const {
  for (const FindingCapability &capability : findingCapabilities)
    if (capability.kind == finding)
      return &capability;
  return nullptr;
}

const ModelInputSlotDescriptor *
EvaluationModelDescriptor::findInputSlot(ModelInputSlotRef slot) const {
  if (slot.ordinal() >= inputSlots.size())
    return nullptr;
  return &inputSlots[slot.ordinal()];
}

const ModelOutputSlotDescriptor *
EvaluationModelDescriptor::findOutputSlot(ModelOutputSlotRef slot) const {
  return outputSlotByOrdinal(slot.ordinal());
}

const ModelOutputSlotDescriptor *
EvaluationModelDescriptor::outputSlotByOrdinal(std::uint32_t ordinal) const {
  if (ordinal >= outputSlots.size())
    return nullptr;
  return &outputSlots[ordinal];
}

bool EvaluationModelDescriptor::supportsMetricQuery(
    const MetricQuery &query) const {
  const MetricCapability *capability = findMetricCapability(query.metric);
  return capability &&
         supportsScopeForm(capability->scopeForms, query.scope.form);
}

bool EvaluationModelDescriptor::supportsFindingQuery(
    const FindingQuery &query) const {
  const FindingCapability *capability = findFindingCapability(query.kind);
  return capability &&
         supportsScopeForm(capability->scopeForms, query.scope.form);
}

llvm::Error
registerEvaluationModelDescriptor(const EvaluationModelDescriptor &descriptor) {
  if (llvm::Error error = validateDescriptor(descriptor))
    return error;

  std::lock_guard<std::mutex> lock(modelDescriptorMutex());
  for (const EvaluationModelDescriptor *existing : modelDescriptors()) {
    if (existing->modelKind == descriptor.modelKind) {
      if (existing == &descriptor)
        return llvm::Error::success();
      return evaluationError("conflicting registration for Evaluation model "
                             "kind " +
                             std::to_string(descriptor.modelKind.ordinal()));
    }
    if (existing->spelling == descriptor.spelling)
      return evaluationError("conflicting registration for Evaluation model '" +
                             descriptor.spelling + "'");
  }
  modelDescriptors().push_back(&descriptor);
  std::sort(modelDescriptors().begin(), modelDescriptors().end(),
            [](const EvaluationModelDescriptor *lhs,
               const EvaluationModelDescriptor *rhs) {
              return lhs->modelKind < rhs->modelKind;
            });
  return llvm::Error::success();
}

const EvaluationModelDescriptor *
findEvaluationModelDescriptor(EvaluationModelKind modelKind) {
  std::lock_guard<std::mutex> lock(modelDescriptorMutex());
  for (const EvaluationModelDescriptor *descriptor : modelDescriptors())
    if (descriptor->modelKind == modelKind)
      return descriptor;
  return nullptr;
}

std::vector<std::uint8_t> canonicalEvaluationModelCapabilityBytes(
    const EvaluationModelDescriptor &descriptor) {
  std::vector<std::uint8_t> bytes;
  appendU64Be(bytes, descriptor.modeledPhenomena.size());
  for (ModeledPhenomenon phenomenon : descriptor.modeledPhenomena)
    appendU32Be(bytes, static_cast<std::uint32_t>(phenomenon));
  appendU32Be(bytes, static_cast<std::uint32_t>(descriptor.executionMethod));
  appendU64Be(bytes, descriptor.interactionCapabilities.size());
  for (const EvaluationInteractionCapability &capability :
       descriptor.interactionCapabilities) {
    appendFramedString(bytes, capability.domain.ownerRegistryIdentity());
    appendSchemaVersion(bytes, capability.domain.ownerRegistryVersion());
    appendU32Be(bytes, capability.domain.ownerLocalDomainKind());
    appendU64Be(bytes, capability.modes.size());
    for (EvaluationInteractionMode mode : capability.modes)
      appendU32Be(bytes, static_cast<std::uint32_t>(mode));
  }
  return bytes;
}

llvm::Expected<ResolvedModelBinding>
ResolvedModelBinding::get(EvaluationModelDescriptorRef descriptor,
                          std::vector<ModelInputBinding> inputBindings,
                          ResolvedModelConfigView resolvedModelConfig) {
  std::sort(inputBindings.begin(), inputBindings.end(),
            [](const ModelInputBinding &lhs, const ModelInputBinding &rhs) {
              return lhs.slot < rhs.slot;
            });
  for (std::size_t index = 0; index < inputBindings.size(); ++index) {
    ModelInputBinding &binding = inputBindings[index];
    if (index != 0 && inputBindings[index - 1].slot == binding.slot)
      return evaluationError("duplicate model input-slot binding");
    std::sort(binding.artifacts.begin(), binding.artifacts.end(),
              artifactRootReferenceLess);
    for (std::size_t artifact = 1; artifact < binding.artifacts.size();
         ++artifact)
      if (binding.artifacts[artifact - 1] == binding.artifacts[artifact])
        return evaluationError("duplicate artifact in model input binding");
  }

  ResolvedModelBinding binding(descriptor, std::move(inputBindings),
                               std::move(resolvedModelConfig));
  if (llvm::Error error = validateResolvedModelBinding(binding))
    return std::move(error);
  return binding;
}

llvm::Expected<ResolvedModelBinding>
ResolvedModelBinding::project(EvaluationModelDescriptorRef descriptor,
                              std::vector<ModelInputBinding> inputBindings,
                              const ResolvedConfig &config) {
  auto view = ResolvedModelConfigView::project(descriptor, config);
  if (!view)
    return view.takeError();
  return get(descriptor, std::move(inputBindings), std::move(*view));
}

llvm::Expected<ResolvedModelBinding>
ResolvedModelBinding::adopt(EvaluationModelDescriptorRef descriptor,
                            std::vector<ModelInputBinding> inputBindings,
                            std::vector<std::uint8_t> canonicalViewBytes,
                            ComponentViewDigest digest) {
  auto view = ResolvedModelConfigView::adopt(
      descriptor, std::move(canonicalViewBytes), digest);
  if (!view)
    return view.takeError();
  return get(descriptor, std::move(inputBindings), std::move(*view));
}

llvm::Expected<ResolvedModelConfigView>
ResolvedModelConfigView::project(EvaluationModelDescriptorRef descriptorRef,
                                 const ResolvedConfig &config) {
  const EvaluationModelDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return evaluationError(
        "resolved model config references an unregistered descriptor");
  auto projected = descriptor->resolvedConfigView.project(config);
  if (!projected)
    return projected.takeError();
  if (!*projected)
    return evaluationError("resolved model config projector returned no "
                           "owner-typed view");
  auto bytes = descriptor->resolvedConfigView.encode(*projected);
  if (!bytes)
    return bytes.takeError();
  auto digest = computeComponentViewDigest(
      descriptor->resolvedConfigView.schemaDescriptorBytes, *bytes);
  if (!digest)
    return digest.takeError();
  return adopt(descriptorRef, std::move(*bytes), *digest);
}

llvm::Expected<ResolvedModelConfigView>
ResolvedModelConfigView::adopt(EvaluationModelDescriptorRef descriptorRef,
                               std::vector<std::uint8_t> canonicalViewBytes,
                               ComponentViewDigest digest) {
  const EvaluationModelDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return evaluationError(
        "resolved model config references an unregistered descriptor");
  if (llvm::Error error = validateComponentViewDigest(
          descriptor->resolvedConfigView.schemaDescriptorBytes,
          canonicalViewBytes, digest))
    return std::move(error);
  auto adopted =
      descriptor->resolvedConfigView.adopt(canonicalViewBytes, digest);
  if (!adopted)
    return adopted.takeError();
  if (!*adopted)
    return evaluationError(
        "resolved model config adopter returned no owner-typed view");
  auto reencoded = descriptor->resolvedConfigView.encode(*adopted);
  if (!reencoded)
    return reencoded.takeError();
  if (*reencoded != canonicalViewBytes)
    return evaluationError(
        "resolved model config decode/re-encode changed canonical bytes");
  return ResolvedModelConfigView(descriptorRef,
                                 std::move(canonicalViewBytes), digest,
                                 std::move(*adopted));
}

const ModelInputBinding *
ResolvedModelBinding::findInputBinding(ModelInputSlotRef slot) const {
  for (const ModelInputBinding &binding : inputBindings_)
    if (binding.slot == slot)
      return &binding;
  return nullptr;
}

llvm::Error validateResolvedModelBinding(const ResolvedModelBinding &binding) {
  const EvaluationModelDescriptor *descriptor =
      binding.descriptorRef().descriptor();
  if (!descriptor)
    return evaluationError("model binding references an unregistered "
                           "descriptor");
  if (binding.resolvedModelConfig().descriptorRef() !=
      binding.descriptorRef())
    return evaluationError(
        "model binding carries a config view from a foreign descriptor");
  if (binding.inputBindings().size() != descriptor->inputSlots.size())
    return evaluationError("model input binding is not total over descriptor "
                           "slots");

  for (std::size_t index = 0; index < descriptor->inputSlots.size(); ++index) {
    const ModelInputSlotDescriptor &slot = descriptor->inputSlots[index];
    const ModelInputBinding &input = binding.inputBindings()[index];
    if (input.slot != slot.slot)
      return evaluationError("model input binding has a foreign slot ordinal");
    if (llvm::Error error = validateArtifactCollectionCardinality(
            slot.cardinality, input.artifacts.size(), slot.semanticRole))
      return error;
    for (const ArtifactRootReference &artifact : input.artifacts)
      if (!acceptsSchema(slot, artifact))
        return evaluationError("model input slot '" + slot.semanticRole +
                               "' rejects artifact schema '" +
                               artifact.schemaIdentity + "'");
  }

  return llvm::Error::success();
}

llvm::Error
validateModelCapability(const EvaluationModelDescriptor &descriptor,
                        const EvaluationCase &evaluationCase,
                        llvm::ArrayRef<MetricRequest> metricRequests,
                        llvm::ArrayRef<FindingRequest> findingRequests) {
  if (llvm::Error error = validateDescriptor(descriptor))
    return error;
  const EvaluationCaseSignatureRef signature = evaluationCase.signature();
  if (signature != descriptor.caseSignature)
    return evaluationError("model '" + descriptor.spelling +
                           "' does not evaluate the case's exact case "
                           "signature");

  for (const MetricRequest &request : metricRequests)
    if (!descriptor.supportsMetricQuery(request.query()))
      return evaluationError("model '" + descriptor.spelling +
                             "' does not support requested metric query");
  for (const FindingRequest &request : findingRequests)
    if (!descriptor.supportsFindingQuery(request.query()))
      return evaluationError("model '" + descriptor.spelling +
                             "' does not support requested finding query");

  llvm::SmallVector<ConditionApplicabilityPattern, 4> present;
  if (llvm::Error error = requireRecognizedConditions(
          descriptor, evaluationCase.baseConditions(), signature, present))
    return error;
  for (const MetricRequest &request : metricRequests)
    if (llvm::Error error = requireRecognizedConditions(
            descriptor, request.conditions(), signature, present))
      return error;
  for (const FindingRequest &request : findingRequests)
    if (llvm::Error error = requireRecognizedConditions(
            descriptor, request.conditions(), signature, present))
      return error;

  for (const ModelConditionCapability &capability :
       descriptor.conditionCapabilities) {
    if (capability.disposition != ConditionDisposition::Required)
      continue;
    if (std::find(present.begin(), present.end(), capability.pattern) ==
        present.end())
      return evaluationError("model '" + descriptor.spelling +
                             "' requires condition '" +
                             toString(capability.pattern.kind) + "'");
  }
  return llvm::Error::success();
}

} // namespace loom::evaluation
