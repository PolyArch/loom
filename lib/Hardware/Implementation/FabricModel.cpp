#include "Hardware/Implementation/FabricModel.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom::hardware {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_model_invalid: " + message);
}

llvm::Expected<fabric::HardwareDomainRef>
findDomain(const fabric::FabricSystemRootView &system,
           fabric::SpatialCoreOccurrenceRef subject,
           fabric::FabricHardwareDomainKind kind) {
  const fabric::FabricClockResetKind resetKind =
      kind == fabric::FabricHardwareDomainKind::Clock
          ? fabric::FabricClockResetKind::Clock
          : fabric::FabricClockResetKind::Reset;
  return system.effectiveHardwareDomain(subject, resetKind);
}

llvm::Expected<fabric::SpatialCoreOccurrenceRef>
attachmentSubject(
    const fabric::FabricSpatialAttachmentEndpointRef &endpoint) {
  if (const auto *transport = endpoint.transport()) {
    if (transport->owner.kind() !=
        fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
      return invalid("transport attachment is not SpatialCore-owned");
    return std::get<fabric::SpatialCoreOccurrenceRef>(
        transport->owner.payload);
  }
  const auto *memory = endpoint.memory();
  if (!memory ||
      memory->owner.kind() !=
          fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return invalid("memory attachment is not SpatialCore-owned");
  return std::get<fabric::SpatialCoreOccurrenceRef>(memory->owner.payload);
}

} // namespace

llvm::Expected<std::vector<ImplementationInterfaceSemanticRef>>
deriveSpatialCoreImplementationInterfaceSemantics(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject) {
  const fabric::FabricSystemRootView &system =
      configurationAbi.abi().fabricSystem();
  auto clock =
      findDomain(system, subject, fabric::FabricHardwareDomainKind::Clock);
  if (!clock)
    return clock.takeError();
  auto reset =
      findDomain(system, subject, fabric::FabricHardwareDomainKind::Reset);
  if (!reset)
    return reset.takeError();

  std::vector<ImplementationInterfaceSemanticRef> semantics;
  semantics.emplace_back(ImplementationClockInterfaceRef{*clock});
  semantics.emplace_back(ImplementationResetInterfaceRef{*reset});

  for (const ProgrammingUnit &unit :
       configurationAbi.abi().programmingUnits()) {
    const ProgrammingUnitOccurrenceScope scope =
        deriveProgrammingUnitOccurrenceScope(unit);
    if (scope.includesDirectSystemResources || scope.spatialCores.size() != 1 ||
        scope.spatialCores.front() != subject)
      continue;
    semantics.emplace_back(ImplementationConfigurationInterfaceRef{
        ProgrammingUnitRef{configurationAbi.reference(), unit.id}});
  }

  for (const auto &attachment : system.spatialAttachments()) {
    auto owner = attachmentSubject(attachment.spatialEndpoint);
    if (!owner)
      return owner.takeError();
    if (*owner != subject)
      continue;
    const auto target = system.spatialCoreTarget(subject.core);
    if (!target ||
        target->dependencyOrdinal !=
            attachment.moduleEndpoint.dependencyOrdinal ||
        target->target != attachment.moduleEndpoint.target.module)
      return invalid("System attachment targets a foreign Module occurrence");
    if (attachment.spatialEndpoint.transport())
      semantics.emplace_back(
          ImplementationDataInterfaceRef{attachment.spatialEndpoint});
    else
      semantics.emplace_back(
          ImplementationMemoryInterfaceRef{attachment.spatialEndpoint});
  }
  return semantics;
}

llvm::Expected<FinalizedHardwareImplementation>
finalizeFabricModelHardwareImplementation(
    const FinalizedConfigurationABI &configurationAbi,
    fabric::SpatialCoreOccurrenceRef subject, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::FabricModel);
  if (!format)
    return format.takeError();
  auto representation = createImplementationRepresentationRoot(
      RepresentationRootVariant::FabricModel, std::nullopt, *format,
      {RepresentationObjectKind::Model, fabricModelRootCanonicalName.str()},
      {});
  if (!representation)
    return representation.takeError();
  auto semantics = deriveSpatialCoreImplementationInterfaceSemantics(
      configurationAbi, subject);
  if (!semantics)
    return semantics.takeError();
  std::vector<ImplementationInterface> interfaces;
  interfaces.reserve(semantics->size());
  for (ImplementationInterfaceSemanticRef &semantic : *semantics)
    interfaces.push_back(
        {std::move(semantic), representation->top, std::nullopt});
  return finalizeHardwareImplementation(
      HardwareImplementationDraft{
          configurationAbi.abi().fabric(), subject,
          configurationAbi.reference(), std::move(*representation),
          std::nullopt, std::move(interfaces), {}, {}, {}},
      artifacts, blobs);
}

} // namespace loom::hardware
