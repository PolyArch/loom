#include "Evaluation/ModelProvider.h"

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
    std::vector<ModelOutputBinding> emptyOutputs;
    emptyOutputs.reserve(descriptor->outputSlots.size());
    for (const ModelOutputSlotDescriptor &slot : descriptor->outputSlots)
      emptyOutputs.push_back(ModelOutputBinding{slot.slot, {}});
    return EvaluationEvidence::get(
        request, std::move(emptyOutputs),
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

llvm::Expected<external_tool::PreparedExternalToolInvocation>
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
  return std::get<EvaluationModelExternalPrepareImportProvider>(
             *implementation)
      .prepare(request, resolution, artifactStore, blobStore, context);
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
