#include "Fabric/Artifact/FabricModuleRootView.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "FabricArtifactViewInternal.h"
#include "FabricArtifactViewStorage.h"

using namespace loom::fabric;

llvm::ArrayRef<FabricInventoryOwnerRef>
FabricArtifactView::moduleResourceOwners() const {
  return storage_->moduleResourceOwners;
}

llvm::ArrayRef<FabricModuleDomainMemberRef>
FabricArtifactView::moduleDomainMembers() const {
  return storage_->moduleDomainMembers;
}

llvm::ArrayRef<FabricModuleDomainSlotRef>
FabricModuleRootView::domainSlots() const {
  return artifact_.storage_->data.moduleDomainSlots;
}

llvm::ArrayRef<ModuleDomainAssignment>
FabricModuleRootView::domainAssignments() const {
  return artifact_.storage_->data.moduleDomainAssignments;
}

llvm::Expected<FabricModuleRootView>
loom::fabric::requireModuleRoot(const FabricArtifactView &view) {
  if (view.rootKind() != FabricRootKind::Module)
    return makeFabricRefError(FabricRefErrorKind::WrongRootKind,
                              "Fabric root is not a Module");
  return FabricModuleRootView(view);
}

std::optional<FabricFuNodeKind>
FabricArtifactView::fuNodeKind(const FabricInventoryOwnerRef &owner,
                               FabricOrdinal ordinal) const {
  const std::vector<detail::FabricFuNodeViewData> *nodes = nullptr;
  if (owner.kind() == FabricInventoryOwnerKind::FuTemplate)
    nodes = storage_->fuNodes(std::get<FabricFuTemplateRef>(owner.payload));
  else if (owner.kind() == FabricInventoryOwnerKind::FuOccurrence)
    nodes = storage_->fuNodes(std::get<FabricFuOccurrenceRef>(owner.payload));
  if (!nodes || ordinal >= nodes->size())
    return std::nullopt;
  return (*nodes)[ordinal].kind;
}

bool FabricArtifactView::declaresLocalMemoryService(
    FabricMemoryOccurrenceRef memory) const {
  const detail::FabricEntityViewData *record = storage_->entity(memory);
  return record && record->localMemoryService.has_value();
}

const ::fabric::MemoryServiceContractRecord *
FabricArtifactView::localMemoryService(FabricMemoryOccurrenceRef memory) const {
  const detail::FabricEntityViewData *record = storage_->entity(memory);
  return record && record->localMemoryService
             ? &record->localMemoryService->record
             : nullptr;
}

std::optional<FabricMemoryEndpointRole> FabricArtifactView::memoryEndpointRole(
    const FabricMemoryEndpointRef &endpoint) const {
  const detail::FabricMemoryEndpointViewData *record =
      storage_->memoryEndpoint(endpoint);
  return record ? std::optional<FabricMemoryEndpointRole>(record->role)
                : std::nullopt;
}

llvm::ArrayRef<std::uint8_t> FabricArtifactView::memoryEndpointType(
    const FabricMemoryEndpointRef &endpoint) const {
  const detail::FabricMemoryEndpointViewData *record =
      storage_->memoryEndpoint(endpoint);
  return record ? llvm::ArrayRef<std::uint8_t>(record->canonicalType)
                : llvm::ArrayRef<std::uint8_t>();
}

std::optional<FabricHardwareDomainKind>
FabricArtifactView::hardwareDomainKind(HardwareDomainRef domain) const {
  const detail::FabricEntityViewData *record = storage_->entity(domain);
  return record ? record->hardwareDomainKind : std::nullopt;
}

std::optional<FabricPeOccurrenceRef>
FabricArtifactView::parentPeOf(FabricFuOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->parentPe : std::nullopt;
}

std::optional<::fabric::Schedule>
FabricArtifactView::peSchedule(FabricPeOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->peSchedule : std::nullopt;
}

std::uint64_t FabricArtifactView::peResidentContextCount(
    FabricPeOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->instructionContexts.size() : 0;
}

std::optional<FabricFuTemplateRef>
FabricArtifactView::fuTemplateOf(FabricFuOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->fuTemplate : std::nullopt;
}

llvm::ArrayRef<FabricFuTemplateRef> FabricArtifactView::fuTemplates() const {
  return storage_->fuTemplates;
}

llvm::ArrayRef<FabricMemoryEngineTemplateRef>
FabricArtifactView::memoryEngineTemplates() const {
  return storage_->memoryEngineTemplates;
}
