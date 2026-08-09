#include "Evaluation/ModelProvider.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace loom::evaluation {
namespace {

std::vector<const EvaluationModelProvider *> &providers() {
  static std::vector<const EvaluationModelProvider *> records;
  return records;
}

std::mutex &providerMutex() {
  static std::mutex mutex;
  return mutex;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "evaluation_provider_invalid: " + message);
}

std::vector<ModelOutputBinding>
emptyOutputBindings(const EvaluationModelDescriptor &descriptor) {
  std::vector<ModelOutputBinding> bindings;
  bindings.reserve(descriptor.outputSlots.size());
  for (const ModelOutputSlotDescriptor &slot : descriptor.outputSlots)
    bindings.push_back(ModelOutputBinding{slot.slot, {}});
  return bindings;
}

} // namespace

llvm::Error
registerEvaluationModelProvider(const EvaluationModelProvider &provider) {
  const EvaluationModelDescriptor *descriptor =
      provider.descriptor.descriptor();
  if (!descriptor)
    return invalid("provider references an unregistered model descriptor");
  if (const auto *inProcess =
          std::get_if<EvaluationModelInProcessProvider>(
              &provider.implementation)) {
    if (descriptor->providerForm != ProviderForm::InProcess)
      return invalid("provider form does not match the descriptor");
    if (!inProcess->evaluate)
      return invalid("in-process provider requires an evaluate callback");
  } else if (const auto *external =
                 std::get_if<EvaluationModelExternalPrepareImportProvider>(
                     &provider.implementation)) {
    if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
      return invalid("provider form does not match the descriptor");
    if (!external->prepare || !external->import)
      return invalid("external provider requires both prepare and import");
  } else {
    return invalid("provider has an unknown implementation form");
  }

  std::lock_guard<std::mutex> lock(providerMutex());
  for (const EvaluationModelProvider *existing : providers()) {
    if (existing == &provider)
      return llvm::Error::success();
    if (existing->descriptor == provider.descriptor)
      return invalid("an exact model descriptor already has a provider");
  }
  providers().push_back(&provider);
  return llvm::Error::success();
}

llvm::Expected<external_tool::ExternalToolSemanticContract>
deriveExternalToolSemanticContract(const EvaluationRequest &request) {
  const EvaluationModelDescriptorRef reference =
      request.modelBinding().descriptorRef();
  const EvaluationModelDescriptor *descriptor = reference.descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid(
        "external semantic contract requires ExternalPrepareImport");

  std::vector<std::uint8_t> descriptorReference;
  descriptorReference.reserve(12);
  const auto appendU32 = [&descriptorReference](std::uint32_t value) {
    descriptorReference.push_back(static_cast<std::uint8_t>(value >> 24));
    descriptorReference.push_back(static_cast<std::uint8_t>(value >> 16));
    descriptorReference.push_back(static_cast<std::uint8_t>(value >> 8));
    descriptorReference.push_back(static_cast<std::uint8_t>(value));
  };
  appendU32(reference.schemaVersion().major);
  appendU32(reference.schemaVersion().minor);
  appendU32(reference.modelKind().ordinal());
  auto importer = external_tool::deriveExternalToolResultImporterIdentity(
      descriptorReference, descriptor->providerForm);
  if (!importer)
    return importer.takeError();
  return external_tool::ExternalToolSemanticContract{
      descriptor->implementationSemanticIdentity.str(),
      evaluationRequestReference(request), std::move(*importer)};
}

llvm::Expected<EvaluationEvidence>
evaluateRequest(const EvaluationRequest &request,
                const CaseArtifactResolution &resolution,
                const ArtifactStore &artifactStore,
                const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);

  // The descriptor form rules before any provider lookup: the ordinary
  // facade is defined only for InProcess models.
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  if (descriptor->providerForm != ProviderForm::InProcess)
    return invalid(
        "external prepare/import model cannot be evaluated in-process");

  std::optional<EvaluationModelProviderImplementation> implementation;
  {
    std::lock_guard<std::mutex> lock(providerMutex());
    for (const EvaluationModelProvider *provider : providers())
      if (provider->descriptor == request.modelBinding().descriptorRef()) {
        implementation = provider->implementation;
        break;
      }
  }
  if (!implementation) {
    return EvaluationEvidence::get(
        request, emptyOutputBindings(*descriptor),
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable},
        resolution, artifactStore);
  }

  auto result = std::get<EvaluationModelInProcessProvider>(*implementation)
                    .evaluate(request, resolution, artifactStore, blobStore);
  if (!result)
    return result.takeError();
  return EvaluationEvidence::get(request, std::move(result->outputBindings),
                                 std::move(result->outcome), resolution,
                                 artifactStore);
}

namespace {

std::optional<EvaluationModelProviderImplementation>
lookupProviderImplementation(EvaluationModelDescriptorRef descriptor) {
  std::lock_guard<std::mutex> lock(providerMutex());
  for (const EvaluationModelProvider *provider : providers())
    if (provider->descriptor == descriptor)
      return provider->implementation;
  return std::nullopt;
}

} // namespace

llvm::Expected<EvaluationModelPreparation>
prepareEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const external_tool::ExternalToolPreparationContext &context) {
  RequestVerifier verifier(resolution, artifactStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  // The descriptor form rules before any provider lookup.
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid("in-process model cannot prepare an external invocation");
  std::optional<EvaluationModelProviderImplementation> implementation =
      lookupProviderImplementation(request.modelBinding().descriptorRef());
  if (!implementation)
    return invalid("external prepare/import model provider is unavailable");
  auto preparation =
      std::get<EvaluationModelExternalPrepareImportProvider>(*implementation)
          .prepare(request, resolution, artifactStore, blobStore, context);
  if (!preparation)
    return preparation.takeError();
  if (auto *prepared =
          std::get_if<external_tool::PreparedExternalToolInvocation>(
              &*preparation))
    return EvaluationModelPreparation{std::move(*prepared)};

  auto evidence = EvaluationEvidence::get(
      request, emptyOutputBindings(*descriptor),
      std::get<UnsupportedEvidence>(std::move(*preparation)), resolution,
      artifactStore);
  if (!evidence)
    return evidence.takeError();
  return EvaluationModelPreparation{std::move(*evidence)};
}

llvm::Expected<EvaluationEvidence> importEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(resolution, artifactStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid("in-process model cannot import an external invocation");
  std::optional<EvaluationModelProviderImplementation> implementation =
      lookupProviderImplementation(request.modelBinding().descriptorRef());
  if (!implementation)
    return invalid("external prepare/import model provider is unavailable");
  auto result = std::get<EvaluationModelExternalPrepareImportProvider>(
                    *implementation)
                    .import(request, resolution, prepared, artifactStore,
                            blobStore);
  if (!result)
    return result.takeError();
  return EvaluationEvidence::get(request, std::move(result->outputBindings),
                                 std::move(result->outcome), resolution,
                                 artifactStore);
}

} // namespace loom::evaluation
