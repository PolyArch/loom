#include "Evaluation/ModelProvider.h"

#include <mutex>
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
  if (!provider.evaluate)
    return invalid("provider has no evaluate implementation");

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

const EvaluationModelProvider *
findEvaluationModelProvider(EvaluationModelDescriptorRef descriptor) {
  std::lock_guard<std::mutex> lock(providerMutex());
  for (const EvaluationModelProvider *provider : providers())
    if (provider->descriptor == descriptor)
      return provider;
  return nullptr;
}

llvm::Expected<EvaluationEvidence>
evaluateRequest(const EvaluationRequest &request,
                const CaseArtifactResolution &resolution,
                const ArtifactStore &artifactStore) {
  RequestVerifier verifier(resolution, artifactStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);

  const EvaluationModelProvider *provider =
      findEvaluationModelProvider(request.modelBinding().descriptorRef());
  if (!provider) {
    std::vector<ModelOutputBinding> emptyOutputs;
    const EvaluationModelDescriptor *descriptor =
        request.modelBinding().descriptorRef().descriptor();
    emptyOutputs.reserve(descriptor->outputSlots.size());
    for (const ModelOutputSlotDescriptor &slot : descriptor->outputSlots)
      emptyOutputs.push_back(ModelOutputBinding{slot.slot, {}});
    return EvaluationEvidence::get(
        request, std::move(emptyOutputs),
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable},
        resolution, artifactStore);
  }

  auto result = provider->evaluate(request, resolution, artifactStore);
  if (!result)
    return result.takeError();
  return EvaluationEvidence::get(request, std::move(result->outputBindings),
                                 std::move(result->outcome), resolution,
                                 artifactStore);
}

} // namespace loom::evaluation
