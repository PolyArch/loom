#include "Runtime/FabricModelPlatform.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::runtime {
namespace {

enum class FabricModelEndpointKind : std::uint32_t {
  Identity = 0,
  Programming = 1,
  Memory = 2,
  Completion = 3,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "fabric_model_platform_invalid: " + message);
}

llvm::Error validateIdentityPayload(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != ArtifactIdentity::byteSize)
    return invalid("identity endpoint payload is not an ArtifactIdentity");
  return llvm::Error::success();
}

llvm::Error validateInterfacePayload(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != sizeof(std::uint64_t))
    return invalid("interface endpoint payload is not one u64be ordinal");
  return llvm::Error::success();
}

std::vector<std::uint8_t> encodeOrdinal(std::uint64_t ordinal) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(sizeof(ordinal));
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(ordinal >> shift));
    if (shift == 0)
      break;
  }
  return bytes;
}

RuntimeProviderEndpointRef endpoint(FabricModelEndpointKind kind,
                                    std::uint64_t ordinal) {
  return {static_cast<std::uint32_t>(kind), encodeOrdinal(ordinal)};
}

const RuntimeProviderEndpointKindDescriptor endpointKinds[] = {
    {static_cast<std::uint32_t>(FabricModelEndpointKind::Identity),
     "identity", RuntimeEndpointClass::Identity,
     RuntimeEndpointFlow::ImplementationToRuntime, false,
     validateIdentityPayload},
    {static_cast<std::uint32_t>(FabricModelEndpointKind::Programming),
     "programming", RuntimeEndpointClass::Programming,
     RuntimeEndpointFlow::Bidirectional, false, validateInterfacePayload},
    {static_cast<std::uint32_t>(FabricModelEndpointKind::Memory), "memory",
     RuntimeEndpointClass::Memory, RuntimeEndpointFlow::Bidirectional, false,
     validateInterfacePayload},
    {static_cast<std::uint32_t>(FabricModelEndpointKind::Completion),
     "completion", RuntimeEndpointClass::Completion,
     RuntimeEndpointFlow::Bidirectional, false, validateInterfacePayload},
};

const RuntimeProviderDescriptor descriptor{
    {"loom.runtime.fabric_model", SchemaVersion{1, 0}},
    "loom.hardware.fabric_model.v1",
    "loom.runtime.fabric_model_abi.v1",
    endpointKinds,
    true,
    false,
    false,
    false};

} // namespace

const RuntimeProviderDescriptor &fabricModelRuntimeProviderDescriptor() {
  return descriptor;
}

llvm::Expected<FinalizedRuntimePlatformBinding>
finalizeFabricModelRuntimePlatformBinding(
    const hardware::FinalizedHardwareImplementation &implementation,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (implementation.implementation().representationRoot().variant !=
      hardware::RepresentationRootVariant::FabricModel)
    return invalid("HardwareImplementation is not a FabricModel");
  if (llvm::Error error = registerRuntimeProvider(descriptor))
    return std::move(error);

  std::vector<RuntimeProgrammingBinding> programming;
  std::vector<RuntimeInterfaceBinding> memory;
  std::vector<RuntimeInterfaceBinding> completion;
  for (const auto indexed :
       llvm::enumerate(implementation.implementation().interfaces())) {
    const std::uint64_t ordinal = indexed.index();
    const ArtifactReference<hardware::HardwareImplementationInterfaceRef>
        reference{implementation.reference().artifact,
                  hardware::HardwareImplementationInterfaceRef{ordinal}};
    const hardware::ImplementationInterfaceSemanticRef &semantic =
        indexed.value().semanticRef;
    if (const auto *configuration =
            std::get_if<hardware::ImplementationConfigurationInterfaceRef>(
                &semantic)) {
      programming.push_back(
          {configuration->programmingUnit, reference,
           endpoint(FabricModelEndpointKind::Programming, ordinal)});
      continue;
    }
    if (std::holds_alternative<hardware::ImplementationMemoryInterfaceRef>(
            semantic)) {
      memory.push_back(
          {reference, endpoint(FabricModelEndpointKind::Memory, ordinal)});
      continue;
    }
    if (std::holds_alternative<hardware::ImplementationDataInterfaceRef>(
            semantic) ||
        std::holds_alternative<
            hardware::ImplementationExternalProtocolInterfaceRef>(semantic))
      completion.push_back(
          {reference, endpoint(FabricModelEndpointKind::Completion, ordinal)});
  }

  RuntimeProviderEndpointRef identity{
      static_cast<std::uint32_t>(FabricModelEndpointKind::Identity),
      std::vector<std::uint8_t>(implementation.reference().artifact.bytes().begin(),
                                implementation.reference().artifact.bytes().end())};
  return finalizeRuntimePlatformBinding(
      RuntimePlatformBindingDraft{
          implementation.reference(), runtimeProviderDescriptorRef(descriptor),
          HardwareReportedIdentity{std::move(identity)},
          std::move(programming), std::move(memory), std::move(completion)},
      artifacts, blobs);
}

} // namespace loom::runtime
