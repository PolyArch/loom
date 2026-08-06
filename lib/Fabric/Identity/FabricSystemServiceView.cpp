#include "Fabric/Artifact/FabricSystemRootView.h"

#include "FabricArtifactViewInternal.h"
#include "FabricArtifactViewStorage.h"

using namespace loom::fabric;

const ::fabric::MemoryServiceContractRecord *
FabricSystemRootView::memoryService(SystemMemoryServiceRef service) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(service);
  return entity && entity->systemMemoryService
             ? &*entity->systemMemoryService
             : nullptr;
}

const SystemServiceEndpointOwnerRef *FabricSystemRootView::serviceEndpointOwner(
    SystemServiceEndpointRef endpoint) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(endpoint);
  return entity && entity->systemServiceEndpointOwner
             ? &*entity->systemServiceEndpointOwner
             : nullptr;
}

const CanonicalServiceCapabilitySet *
FabricSystemRootView::serviceEndpointCapabilities(
    SystemServiceEndpointRef endpoint) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(endpoint);
  return entity && entity->systemServiceCapabilities
             ? &*entity->systemServiceCapabilities
             : nullptr;
}

const SystemServiceTransformRecord *FabricSystemRootView::serviceTransform(
    SystemServiceTransformRef transform) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(transform);
  return entity && entity->systemServiceTransform
             ? &*entity->systemServiceTransform
             : nullptr;
}
