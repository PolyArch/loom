#include "Runtime/RuntimeProvider.h"

#include "llvm/ADT/STLExtras.h"

#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace loom::runtime {
namespace {

std::vector<const RuntimeProviderDescriptor *> &providers() {
  static std::vector<const RuntimeProviderDescriptor *> records;
  return records;
}

std::mutex &providerMutex() {
  static std::mutex mutex;
  return mutex;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "runtime_provider_invalid: " + message);
}

bool flowCompatible(RuntimeEndpointFlow actual, RuntimeEndpointFlow expected) {
  return actual == expected || actual == RuntimeEndpointFlow::Bidirectional;
}

} // namespace

RuntimeProviderDescriptorRef
runtimeProviderDescriptorRef(const RuntimeProviderDescriptor &descriptor) {
  return RuntimeProviderDescriptorRef{descriptor.descriptor.identity.str(),
                                      descriptor.descriptor.version};
}

llvm::Error registerRuntimeProvider(const RuntimeProviderDescriptor &provider) {
  if (provider.descriptor.identity.empty())
    return invalid("descriptor identity is empty");
  if (provider.implementationSemanticIdentity.empty())
    return invalid("implementation semantic identity is empty");
  if (provider.runtimeAbiIdentity.empty())
    return invalid("runtime ABI identity is empty");
  if (provider.resourceTimeCostModel) {
    const RuntimeResourceTimeCostModel &cost = *provider.resourceTimeCostModel;
    if (cost.memoryCopySetupPicoseconds == 0 ||
        cost.memoryCopyBytePicoseconds == 0 ||
        cost.configurationWordPicoseconds == 0 ||
        cost.configurationCommitPicoseconds == 0)
      return invalid("resource-time cost model contains a zero component");
    if (!provider.supportsPreparedActivationReplacement)
      return invalid("resource-time cost model requires prepared activation "
                     "replacement");
  }

  std::set<std::uint32_t> kinds;
  std::set<std::string> names;
  bool hasIdentity = false;
  for (const RuntimeProviderEndpointKindDescriptor &endpoint :
       provider.endpointKinds) {
    if (endpoint.stableName.empty())
      return invalid("endpoint kind has an empty stable name");
    if (!endpoint.validateCanonicalPayload)
      return invalid("endpoint kind has no canonical payload validator");
    if (!kinds.insert(endpoint.kind).second)
      return invalid("endpoint kind ordinal is duplicated");
    if (!names.insert(endpoint.stableName.str()).second)
      return invalid("endpoint kind stable name is duplicated");
    hasIdentity |= endpoint.endpointClass == RuntimeEndpointClass::Identity;
  }
  if (provider.supportsHardwareReportedIdentity && !hasIdentity)
    return invalid(
        "hardware-reported identity requires an Identity endpoint kind");
  if (!provider.supportsHardwareReportedIdentity &&
      !provider.supportsTrustedImmutableIdentity)
    return invalid("provider supports no identity verification form");

  std::lock_guard<std::mutex> lock(providerMutex());
  const RuntimeProviderDescriptorRef reference =
      runtimeProviderDescriptorRef(provider);
  for (const RuntimeProviderDescriptor *existing : providers()) {
    if (existing == &provider)
      return llvm::Error::success();
    if (runtimeProviderDescriptorRef(*existing) == reference)
      return invalid("an exact descriptor already has a provider owner");
  }
  providers().push_back(&provider);
  return llvm::Error::success();
}

const RuntimeProviderDescriptor *
findRuntimeProvider(const RuntimeProviderDescriptorRef &reference) {
  std::lock_guard<std::mutex> lock(providerMutex());
  const auto found = llvm::find_if(providers(), [&](const auto *provider) {
    return runtimeProviderDescriptorRef(*provider) == reference;
  });
  return found == providers().end() ? nullptr : *found;
}

const RuntimeProviderEndpointKindDescriptor *
findRuntimeEndpointKind(const RuntimeProviderDescriptor &provider,
                        std::uint32_t kind) {
  const auto found =
      llvm::find_if(provider.endpointKinds,
                    [&](const auto &item) { return item.kind == kind; });
  return found == provider.endpointKinds.end() ? nullptr : &*found;
}

llvm::Error
validateRuntimeProviderEndpoint(const RuntimeProviderDescriptor &provider,
                                const RuntimeProviderEndpointRef &endpoint,
                                RuntimeEndpointClass expectedClass,
                                RuntimeEndpointFlow expectedFlow) {
  const RuntimeProviderEndpointKindDescriptor *kind =
      findRuntimeEndpointKind(provider, endpoint.kind);
  if (!kind)
    return invalid("endpoint references an unknown provider kind");
  if (kind->endpointClass != expectedClass)
    return invalid("endpoint class does not match the bound interface");
  if (!flowCompatible(kind->flow, expectedFlow))
    return invalid("endpoint flow does not match the bound interface");
  if (llvm::Error error = kind->validateCanonicalPayload(endpoint.payload))
    return invalid("endpoint payload is not canonical: " +
                   llvm::toString(std::move(error)));
  return llvm::Error::success();
}

} // namespace loom::runtime
