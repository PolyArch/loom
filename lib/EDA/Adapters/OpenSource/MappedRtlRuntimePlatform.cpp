#include "EDA/Adapters/OpenSource/MappedRtlRuntimePlatform.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/ConfigurationTransport.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::eda::open_source {
namespace {

enum class EndpointKind : std::uint32_t {
  Identity = 0,
  Programming = 1,
  Memory = 2,
  Completion = 3,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapped_rtl_runtime_platform_invalid: " + message);
}

llvm::Error validateIdentity(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != ArtifactIdentity::byteSize)
    return invalid("identity endpoint is not an ArtifactIdentity");
  return llvm::Error::success();
}

llvm::Error validateInterface(llvm::ArrayRef<std::uint8_t> payload) {
  constexpr std::size_t expected = 1 + ArtifactIdentity::byteSize + 8;
  if (payload.size() != expected || payload.front() != 1)
    return invalid("interface endpoint is not a scoped ordinal");
  return llvm::Error::success();
}

std::vector<std::uint8_t> scopedOrdinal(const ArtifactIdentity &implementation,
                                        std::uint64_t ordinal) {
  std::vector<std::uint8_t> result;
  result.reserve(1 + ArtifactIdentity::byteSize + 8);
  result.push_back(1);
  result.insert(result.end(), implementation.bytes().begin(),
                implementation.bytes().end());
  for (unsigned shift = 56;; shift -= 8) {
    result.push_back(static_cast<std::uint8_t>(ordinal >> shift));
    if (shift == 0)
      break;
  }
  return result;
}

runtime::RuntimeProviderEndpointRef endpoint(
    EndpointKind kind, const ArtifactIdentity &implementation,
    std::uint64_t ordinal) {
  return {static_cast<std::uint32_t>(kind),
          scopedOrdinal(implementation, ordinal)};
}

const runtime::RuntimeProviderEndpointKindDescriptor endpointKinds[] = {
    {static_cast<std::uint32_t>(EndpointKind::Identity), "identity",
     runtime::RuntimeEndpointClass::Identity,
     runtime::RuntimeEndpointFlow::ImplementationToRuntime, false,
     validateIdentity},
    {static_cast<std::uint32_t>(EndpointKind::Programming), "programming",
     runtime::RuntimeEndpointClass::Programming,
     runtime::RuntimeEndpointFlow::Bidirectional, false, validateInterface},
    {static_cast<std::uint32_t>(EndpointKind::Memory), "memory",
     runtime::RuntimeEndpointClass::Memory,
     runtime::RuntimeEndpointFlow::Bidirectional, false, validateInterface},
    {static_cast<std::uint32_t>(EndpointKind::Completion), "completion",
     runtime::RuntimeEndpointClass::Completion,
     runtime::RuntimeEndpointFlow::Bidirectional, false, validateInterface},
};

const runtime::RuntimeProviderDescriptor descriptor{
    {"loom.runtime.mapped_rtl", SchemaVersion{2, 0}},
    "loom.hardware.mapped_rtl.simulation_transport.v1",
    hardware::rtl::portableConfigurationRuntimeAbiIdentity,
    endpointKinds,
    true,
    false,
    false,
    false};

} // namespace

llvm::Expected<runtime::FinalizedRuntimePlatformBinding>
finalizeMappedRtlRuntimePlatformBinding(
    const hardware::FinalizedHardwareImplementation &implementation,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (implementation.implementation().representationRoot().variant !=
      hardware::RepresentationRootVariant::Rtl)
    return invalid("HardwareImplementation is not RTL");
  if (llvm::Error error = runtime::registerRuntimeProvider(descriptor))
    return std::move(error);

  std::vector<runtime::RuntimeProgrammingBinding> programming;
  std::vector<runtime::RuntimeInterfaceBinding> memory;
  std::vector<runtime::RuntimeInterfaceBinding> completion;
  for (const auto indexed :
       llvm::enumerate(implementation.implementation().interfaces())) {
    const std::uint64_t ordinal = indexed.index();
    const ArtifactReference<hardware::HardwareImplementationInterfaceRef>
        reference{implementation.reference().artifact,
                  hardware::HardwareImplementationInterfaceRef{ordinal}};
    const auto &semantic = indexed.value().semanticRef;
    if (const auto *configuration =
            std::get_if<hardware::ImplementationConfigurationInterfaceRef>(
                &semantic)) {
      programming.push_back(
          {configuration->programmingUnit, reference,
           endpoint(EndpointKind::Programming,
                    implementation.reference().artifact, ordinal)});
    } else if (std::holds_alternative<
                   hardware::ImplementationMemoryInterfaceRef>(semantic)) {
      memory.push_back(
          {reference, endpoint(EndpointKind::Memory,
                               implementation.reference().artifact, ordinal)});
    } else if (std::holds_alternative<
                   hardware::ImplementationDataInterfaceRef>(semantic) ||
               std::holds_alternative<
                   hardware::ImplementationExternalProtocolInterfaceRef>(
                   semantic)) {
      completion.push_back(
          {reference,
           endpoint(EndpointKind::Completion,
                    implementation.reference().artifact, ordinal)});
    }
  }
  runtime::RuntimeProviderEndpointRef identity{
      static_cast<std::uint32_t>(EndpointKind::Identity),
      std::vector<std::uint8_t>(
          implementation.reference().artifact.bytes().begin(),
          implementation.reference().artifact.bytes().end())};
  return runtime::finalizeRuntimePlatformBinding(
      runtime::RuntimePlatformBindingDraft{
          implementation.reference(),
          runtime::runtimeProviderDescriptorRef(descriptor),
          runtime::HardwareReportedIdentity{std::move(identity)},
          std::move(programming), std::move(memory), std::move(completion)},
      artifacts, blobs);
}

} // namespace loom::eda::open_source
