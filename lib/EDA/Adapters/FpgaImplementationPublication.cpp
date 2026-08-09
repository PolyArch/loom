#include "EDA/Adapters/FpgaImplementationPublication.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda {
namespace {

using namespace hardware;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fpga_implementation_publication_invalid: " +
                                     message);
}

std::string encodeIdentifier(llvm::StringRef prefix, llvm::StringRef value) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result = prefix.str();
  result.reserve(result.size() + value.size() * 2);
  for (const unsigned char byte : value.bytes()) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<RepresentationLocator>
projectPhysicalLocator(const RepresentationLocator &locator,
                       const RepresentationLocator &sourceTop,
                       const RepresentationLocator &physicalTop,
                       bool retainExternalBindings) {
  if (locator == sourceTop)
    return physicalTop;
  const std::string rootedPrefix = sourceTop.canonicalName + ".";
  if (llvm::StringRef(locator.canonicalName).starts_with(rootedPrefix))
    return RepresentationLocator{
        locator.kind, physicalTop.canonicalName +
                          llvm::StringRef(locator.canonicalName)
                              .drop_front(sourceTop.canonicalName.size())
                              .str()};
  if (locator.kind == RepresentationObjectKind::Module &&
      retainExternalBindings)
    return locator;
  if (locator.kind == RepresentationObjectKind::Module)
    return RepresentationLocator{
        RepresentationObjectKind::DeviceResource,
        physicalTop.canonicalName + "." +
            encodeIdentifier("external_module_", locator.canonicalName)};
  return invalid("ordinary source locator is not rooted at the source top");
}

struct ProjectedImplementationMetadata final {
  std::vector<ImplementationInterface> interfaces;
  std::vector<ActivityPoint> activityPoints;
  std::vector<MemoryMacroBindingDraft> memoryBindings;
  std::vector<ExternalImplementationBindingDraft> externalBindings;
  std::vector<PhysicalRepresentationObject> objects;
  std::vector<RepresentationLocator> unresolvedDefinitions;
};

llvm::Expected<ProjectedImplementationMetadata>
projectImplementationMetadata(const FinalizedHardwareImplementation &source,
                              const RepresentationIndex &sourceIndex,
                              const RepresentationLocator &physicalTop,
                              bool retainExternalBindings) {
  const HardwareImplementation &implementation = source.implementation();
  const ImplementationRepresentationRoot &sourceRoot =
      implementation.representationRoot();
  ProjectedImplementationMetadata result;
  result.objects.push_back({physicalTop, std::nullopt});

  const auto addReferencedObject =
      [&](const RepresentationLocator &sourceLocator) -> llvm::Error {
    auto projected = projectPhysicalLocator(
        sourceLocator, sourceRoot.top, physicalTop, retainExternalBindings);
    if (!projected)
      return projected.takeError();
    if (llvm::any_of(result.objects, [&](const auto &object) {
          return object.locator == *projected;
        }))
      return llvm::Error::success();
    auto facts = sourceIndex.lookup(sourceLocator);
    if (!facts)
      return facts.takeError();
    if (!*facts)
      return invalid("referenced source object is absent from its exact "
                     "representation index");
    result.objects.push_back({*projected, (*facts)->signalGeometry});
    if (projected->kind == RepresentationObjectKind::Module)
      result.unresolvedDefinitions.push_back(*projected);
    return llvm::Error::success();
  };

  result.interfaces.assign(implementation.interfaces().begin(),
                           implementation.interfaces().end());
  for (ImplementationInterface &interface : result.interfaces) {
    if (llvm::Error error =
            addReferencedObject(interface.representationLocator))
      return std::move(error);
    auto projected =
        projectPhysicalLocator(interface.representationLocator, sourceRoot.top,
                               physicalTop, retainExternalBindings);
    if (!projected)
      return projected.takeError();
    interface.representationLocator = std::move(*projected);
  }

  result.activityPoints.assign(implementation.activityPoints().begin(),
                               implementation.activityPoints().end());
  for (ActivityPoint &point : result.activityPoints) {
    if (llvm::Error error = addReferencedObject(point.representationLocator))
      return std::move(error);
    auto projected =
        projectPhysicalLocator(point.representationLocator, sourceRoot.top,
                               physicalTop, retainExternalBindings);
    if (!projected)
      return projected.takeError();
    point.representationLocator = std::move(*projected);
  }

  if (!retainExternalBindings)
    return result;

  for (const ExternalImplementationBinding &binding :
       implementation.externalImplementationBindings()) {
    std::vector<RepresentationLocator> locators(
        binding.representationLocators.begin(),
        binding.representationLocators.end());
    for (RepresentationLocator &locator : locators) {
      if (llvm::Error error = addReferencedObject(locator))
        return std::move(error);
      auto projected = projectPhysicalLocator(
          locator, sourceRoot.top, physicalTop, retainExternalBindings);
      if (!projected)
        return projected.takeError();
      locator = std::move(*projected);
    }
    std::optional<ImplementationPayloadKey> blackBoxContract;
    if (binding.blackBoxContractPayloadRef) {
      if (binding.blackBoxContractPayloadRef->ordinal >=
          sourceRoot.payloads.size())
        return invalid("source black-box payload reference is out of range");
      const ImplementationPayload &payload =
          sourceRoot.payloads[binding.blackBoxContractPayloadRef->ordinal];
      blackBoxContract =
          ImplementationPayloadKey{payload.role, payload.canonicalLogicalName};
    }
    result.externalBindings.push_back(ExternalImplementationBindingDraft{
        binding.providerContractRef, binding.externalInputs,
        binding.fabricResourceRefs, std::move(locators),
        std::move(blackBoxContract)});
  }

  for (const MemoryMacroBinding &binding :
       implementation.memoryMacroBindings()) {
    if (binding.externalImplementationBindingRef.ordinal >=
        result.externalBindings.size())
      return invalid("source memory binding references an unknown external "
                     "implementation");
    if (llvm::Error error = addReferencedObject(binding.representationLocator))
      return std::move(error);
    auto projected =
        projectPhysicalLocator(binding.representationLocator, sourceRoot.top,
                               physicalTop, retainExternalBindings);
    if (!projected)
      return projected.takeError();
    result.memoryBindings.push_back(MemoryMacroBindingDraft{
        binding.fabricMemoryRef,
        binding.externalImplementationBindingRef.ordinal,
        std::move(*projected)});
  }

  for (const RepresentationLocator &locator :
       sourceIndex.unresolvedExternalDefinitions())
    if (llvm::Error error = addReferencedObject(locator))
      return std::move(error);
  return result;
}

llvm::Expected<FinalizedHardwareImplementation>
publishRepresentation(const FinalizedHardwareImplementation &source,
                      const ArtifactRootReference &implementationPlatform,
                      llvm::StringRef device, RepresentationRootVariant variant,
                      std::optional<RepresentationPhysicalStage> stage,
                      PayloadRole outputRole, llvm::StringRef outputLogicalName,
                      llvm::StringRef outputContents,
                      bool retainExternalBindings,
                      const ExternalImplementationContractCatalog &contracts,
                      const ArtifactStore &artifacts, const BlobStore &blobs) {
  const ImplementationRepresentationRoot &sourceRoot =
      source.implementation().representationRoot();
  auto targetPlatform =
      platform::importImplementationPlatform(implementationPlatform, artifacts);
  if (!targetPlatform)
    return targetPlatform.takeError();
  const auto *fpga =
      std::get_if<platform::FpgaTarget>(&targetPlatform->platform().target());
  if (!fpga || fpga->deviceOrderingCode != device)
    return invalid("device ordering code does not match the FPGA platform");
  if (source.implementation().implementationPlatform() &&
      *source.implementation().implementationPlatform() !=
          implementationPlatform)
    return invalid("source implementation is bound to a different platform");
  auto sourceIndex = indexRepresentationRoot(sourceRoot, blobs);
  if (!sourceIndex)
    return sourceIndex.takeError();
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::IndexedPhysical);
  if (!format)
    return format.takeError();
  const RepresentationLocator physicalTop{
      RepresentationObjectKind::DeviceResource,
      fpgaDeviceResourceLocatorName(device)};
  auto projected = projectImplementationMetadata(
      source, *sourceIndex, physicalTop, retainExternalBindings);
  if (!projected)
    return projected.takeError();

  std::vector<ImplementationPayload> payloads;
  if (variant == RepresentationRootVariant::FpgaPhysical)
    for (const ImplementationPayload &payload : sourceRoot.payloads)
      if (payload.role == PayloadRole::GenerationConstraint ||
          payload.role == PayloadRole::BlackBoxContract)
        payloads.push_back(payload);
  auto outputDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(outputContents.data()),
      outputContents.size()));
  if (!outputDigest)
    return outputDigest.takeError();
  payloads.push_back(
      {outputRole, outputLogicalName.str(), std::move(*outputDigest)});

  auto index = createPhysicalRepresentationIndexPayload(
      *format, variant, stage, physicalTop, "index/physical.json", payloads,
      std::move(projected->objects),
      std::move(projected->unresolvedDefinitions));
  if (!index)
    return index.takeError();
  auto indexBytes = serializePhysicalRepresentationIndexPayloadJson(*index);
  if (!indexBytes)
    return indexBytes.takeError();
  auto indexDigest = blobs.put(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(indexBytes->data()),
      indexBytes->size()));
  if (!indexDigest)
    return indexDigest.takeError();
  payloads.push_back({PayloadRole::RepresentationIndex, index->indexLogicalName,
                      std::move(*indexDigest)});

  auto representation = createImplementationRepresentationRoot(
      variant, stage, *format, physicalTop, std::move(payloads));
  if (!representation)
    return representation.takeError();
  return finalizeHardwareImplementation(
      HardwareImplementationDraft{
          source.implementation().fabric(),
          source.implementation().configurationAbi(),
          source.implementation().interconnectImplementations().vec(),
          std::move(*representation), implementationPlatform,
          std::move(projected->interfaces),
          std::move(projected->activityPoints),
          std::move(projected->memoryBindings),
          std::move(projected->externalBindings)},
      contracts, artifacts, blobs);
}

} // namespace

std::string fpgaDeviceResourceLocatorName(llvm::StringRef deviceOrderingCode) {
  return encodeIdentifier("device_", deviceOrderingCode);
}

llvm::Expected<FinalizedHardwareImplementation>
publishRoutedFpgaPhysicalImplementation(
    const FinalizedHardwareImplementation &source,
    const ArtifactRootReference &implementationPlatform,
    llvm::StringRef deviceOrderingCode, llvm::StringRef logicalName,
    llvm::StringRef contents,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  return publishRepresentation(
      source, implementationPlatform, deviceOrderingCode,
      RepresentationRootVariant::FpgaPhysical,
      RepresentationPhysicalStage::Routed, PayloadRole::PhysicalDatabase,
      logicalName, contents, true, contracts, artifacts, blobs);
}

llvm::Expected<FinalizedHardwareImplementation> publishFpgaImageImplementation(
    const FinalizedHardwareImplementation &routedPhysical,
    const ArtifactRootReference &implementationPlatform,
    llvm::StringRef deviceOrderingCode, llvm::StringRef logicalName,
    llvm::StringRef contents,
    const ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (routedPhysical.implementation().representationRoot().variant !=
          RepresentationRootVariant::FpgaPhysical ||
      routedPhysical.implementation().representationRoot().stage !=
          RepresentationPhysicalStage::Routed)
    return invalid("FpgaImage source is not a routed FpgaPhysical state");
  const RepresentationLocator expectedTop{
      RepresentationObjectKind::DeviceResource,
      fpgaDeviceResourceLocatorName(deviceOrderingCode)};
  if (!(routedPhysical.implementation().representationRoot().top ==
        expectedTop))
    return invalid("FpgaImage device does not match its routed physical state");
  return publishRepresentation(
      routedPhysical, implementationPlatform, deviceOrderingCode,
      RepresentationRootVariant::FpgaImage, std::nullopt,
      PayloadRole::DeviceImage, logicalName, contents, false, contracts,
      artifacts, blobs);
}

} // namespace loom::eda
