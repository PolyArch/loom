#include "Evaluation/ModelProvider.h"

#include "Evaluation/ArtifactImportCache.h"
#include "EvidenceInternal.h"

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

namespace detail {

struct EvaluationModelPreparedInvocationBuilder final {
  static EvaluationModelPreparedInvocation
  create(ArtifactRootReference request, CaseArtifactResolution resolution,
         external_tool::PreparedExternalToolInvocation externalInvocation,
         std::shared_ptr<const EvaluationModelInvocationContext> context) {
    return EvaluationModelPreparedInvocation(
        std::move(request), std::move(resolution),
        std::move(externalInvocation), std::move(context));
  }

  static const ArtifactRootReference &
  request(const EvaluationModelPreparedInvocation &prepared) {
    return prepared.request_;
  }

  static const CaseArtifactResolution &
  resolution(const EvaluationModelPreparedInvocation &prepared) {
    return prepared.resolution_;
  }

  static const std::shared_ptr<const EvaluationModelInvocationContext> &
  context(const EvaluationModelPreparedInvocation &prepared) {
    return prepared.context_;
  }
};

} // namespace detail

llvm::Error
registerEvaluationModelProvider(const EvaluationModelProvider &provider) {
  const EvaluationModelDescriptor *descriptor =
      provider.descriptor.descriptor();
  if (!descriptor)
    return invalid("provider references an unregistered model descriptor");
  if (const auto *inProcess = std::get_if<EvaluationModelInProcessProvider>(
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
    return invalid("external semantic contract requires ExternalPrepareImport");

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

llvm::Expected<EvaluationEvidence> evaluateRequest(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  ArtifactImportCacheScope cacheScope(artifactStore, &blobStore);
  RequestVerifier verifier(resolution, artifactStore, blobStore);
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
    return detail::EvaluationEvidenceBuilder::getForVerifiedRequest(
        request, emptyOutputBindings(*descriptor),
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable},
        resolution, artifactStore, blobStore);
  }

  auto result = std::get<EvaluationModelInProcessProvider>(*implementation)
                    .evaluate(request, resolution, artifactStore, blobStore);
  if (!result)
    return result.takeError();
  return detail::EvaluationEvidenceBuilder::getForVerifiedRequest(
      request, std::move(result->outputBindings), std::move(result->outcome),
      resolution, artifactStore, blobStore);
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

llvm::Expected<EvaluationModelExternalPrepareImportProvider>
externalProvider(const EvaluationRequest &request) {
  std::optional<EvaluationModelProviderImplementation> implementation =
      lookupProviderImplementation(request.modelBinding().descriptorRef());
  if (!implementation)
    return invalid("external prepare/import model provider is unavailable");
  const auto *provider =
      std::get_if<EvaluationModelExternalPrepareImportProvider>(
          &*implementation);
  if (!provider)
    return invalid("external model descriptor resolved to an in-process "
                   "provider");
  return *provider;
}

llvm::Expected<std::shared_ptr<const EvaluationModelInvocationContext>>
openInvocationContext(
    const EvaluationModelExternalPrepareImportProvider &provider,
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (!provider.openInvocationContext)
    return std::shared_ptr<const EvaluationModelInvocationContext>{};
  auto context = provider.openInvocationContext(request, resolution,
                                                artifactStore, blobStore);
  if (!context)
    return context.takeError();
  if (!*context)
    return invalid("external provider returned an empty invocation context");
  return std::move(*context);
}

llvm::Expected<std::unique_ptr<EvaluationModelInvocationContext::Activation>>
activateInvocationContext(
    const std::shared_ptr<const EvaluationModelInvocationContext> &context,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (!context)
    return std::unique_ptr<EvaluationModelInvocationContext::Activation>{};
  auto activation = context->activate(artifactStore, blobStore);
  if (!activation)
    return activation.takeError();
  if (!*activation)
    return invalid("external provider returned an empty context activation");
  return std::move(*activation);
}

llvm::Expected<EvaluationEvidence> importVerifiedInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const EvaluationModelExternalPrepareImportProvider &provider,
    const std::shared_ptr<const EvaluationModelInvocationContext> &context,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const external_tool::ExternalToolInvocationExecutionObservation *execution =
        nullptr) {
  auto activation =
      activateInvocationContext(context, artifactStore, blobStore);
  if (!activation)
    return activation.takeError();
  llvm::Expected<EvaluationModelResult> result =
      execution && provider.importWithExecution
          ? provider.importWithExecution(request, resolution, prepared,
                                         *execution, artifactStore, blobStore)
          : provider.import(request, resolution, prepared, artifactStore,
                            blobStore);
  if (!result)
    return result.takeError();
  return detail::EvaluationEvidenceBuilder::getForVerifiedRequest(
      request, std::move(result->outputBindings), std::move(result->outcome),
      resolution, artifactStore, blobStore);
}

} // namespace

llvm::Expected<EvaluationModelPreparation> prepareEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const external_tool::ExternalToolPreparationContext &context) {
  ArtifactImportCacheScope cacheScope(artifactStore, &blobStore);
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  // The descriptor form rules before any provider lookup.
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid("in-process model cannot prepare an external invocation");
  auto provider = externalProvider(request);
  if (!provider)
    return provider.takeError();
  auto invocationContext = openInvocationContext(*provider, request, resolution,
                                                 artifactStore, blobStore);
  if (!invocationContext)
    return invocationContext.takeError();
  auto activation =
      activateInvocationContext(*invocationContext, artifactStore, blobStore);
  if (!activation)
    return activation.takeError();
  auto preparation =
      provider->prepare(request, resolution, artifactStore, blobStore, context);
  if (!preparation)
    return preparation.takeError();
  if (auto *prepared =
          std::get_if<external_tool::PreparedExternalToolInvocation>(
              &*preparation))
    return EvaluationModelPreparation{
        detail::EvaluationModelPreparedInvocationBuilder::create(
            evaluationRequestReference(request), resolution,
            std::move(*prepared), std::move(*invocationContext))};

  auto evidence = detail::EvaluationEvidenceBuilder::getForVerifiedRequest(
      request, emptyOutputBindings(*descriptor),
      std::get<UnsupportedEvidence>(std::move(*preparation)), resolution,
      artifactStore, blobStore);
  if (!evidence)
    return evidence.takeError();
  return EvaluationModelPreparation{std::move(*evidence)};
}

llvm::Expected<EvaluationModelPreparedInvocation>
bindPreparedEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  ArtifactImportCacheScope cacheScope(artifactStore, &blobStore);
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid("in-process model cannot bind an external invocation");
  auto provider = externalProvider(request);
  if (!provider)
    return provider.takeError();
  auto context = openInvocationContext(*provider, request, resolution,
                                       artifactStore, blobStore);
  if (!context)
    return context.takeError();
  return detail::EvaluationModelPreparedInvocationBuilder::create(
      evaluationRequestReference(request), resolution, prepared,
      std::move(*context));
}

llvm::Expected<EvaluationEvidence> importEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const EvaluationModelPreparedInvocation &prepared,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  ArtifactImportCacheScope cacheScope(artifactStore, &blobStore);
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid("in-process model cannot import an external invocation");
  if (detail::EvaluationModelPreparedInvocationBuilder::request(prepared) !=
      evaluationRequestReference(request))
    return invalid("live prepared invocation belongs to another Request");
  if (detail::EvaluationModelPreparedInvocationBuilder::resolution(prepared) !=
      resolution)
    return invalid(
        "live prepared invocation belongs to another artifact resolution");
  auto provider = externalProvider(request);
  if (!provider)
    return provider.takeError();
  return importVerifiedInvocation(
      request, resolution, prepared.externalInvocation(), *provider,
      detail::EvaluationModelPreparedInvocationBuilder::context(prepared),
      artifactStore, blobStore);
}

llvm::Expected<EvaluationEvidence> importEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const EvaluationModelPreparedInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  ArtifactImportCacheScope cacheScope(artifactStore, &blobStore);
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid("in-process model cannot import an external invocation");
  if (detail::EvaluationModelPreparedInvocationBuilder::request(prepared) !=
      evaluationRequestReference(request))
    return invalid("live prepared invocation belongs to another Request");
  if (detail::EvaluationModelPreparedInvocationBuilder::resolution(prepared) !=
      resolution)
    return invalid(
        "live prepared invocation belongs to another artifact resolution");
  auto provider = externalProvider(request);
  if (!provider)
    return provider.takeError();
  if (!provider->importWithExecution)
    return invalid("external provider has no receipt-bound import");
  return importVerifiedInvocation(
      request, resolution, prepared.externalInvocation(), *provider,
      detail::EvaluationModelPreparedInvocationBuilder::context(prepared),
      artifactStore, blobStore, &execution);
}

llvm::Expected<EvaluationEvidence> importEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  ArtifactImportCacheScope cacheScope(artifactStore, &blobStore);
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  const EvaluationModelDescriptor *descriptor =
      request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("request references an unregistered model descriptor");
  if (descriptor->providerForm != ProviderForm::ExternalPrepareImport)
    return invalid("in-process model cannot import an external invocation");
  auto provider = externalProvider(request);
  if (!provider)
    return provider.takeError();
  auto context = openInvocationContext(*provider, request, resolution,
                                       artifactStore, blobStore);
  if (!context)
    return context.takeError();
  return importVerifiedInvocation(request, resolution, prepared, *provider,
                                  *context, artifactStore, blobStore);
}

} // namespace loom::evaluation
