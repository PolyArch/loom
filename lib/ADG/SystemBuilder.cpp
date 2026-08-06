#include "ADG/Builder.h"

#include "BuilderInternal.h"

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom::adg {
namespace detail {

struct SystemHandleAccess final {
  template <typename Handle>
  static const std::weak_ptr<DesignState> &state(const Handle &handle) {
    return handle.state_;
  }

  template <typename Handle>
  static std::size_t rootOrdinal(const Handle &handle) {
    return handle.rootOrdinal_;
  }

  static HardwareDomainMember
  makeDomainMember(const std::weak_ptr<DesignState> &state,
                   std::size_t rootOrdinal,
                   loom::fabric::FabricInventoryOwnerRef owner) {
    return HardwareDomainMember(state.lock(), rootOrdinal, std::move(owner));
  }
};

} // namespace detail

namespace {

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext &context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(&context, signedBytes);
}

llvm::Expected<detail::SystemRootState *>
activeSystem(const std::shared_ptr<detail::DesignState> &state,
             std::size_t rootOrdinal, bool requireOpen = true) {
  if (rootOrdinal >= state->systemRoots.size())
    return detail::invalid("System handle has an invalid owner ordinal");
  detail::SystemRootState &root = state->systemRoots[rootOrdinal];
  if (requireOpen && root.closed)
    return detail::invalid("System is already closed");
  return &root;
}

llvm::Expected<detail::SystemEntityState *>
activeEntity(const std::shared_ptr<detail::DesignState> &state,
             std::size_t rootOrdinal, loom::fabric::FabricEntityId entity,
             loom::fabric::FabricEntityKind kind, bool requireOpenRoot = true) {
  auto root = activeSystem(state, rootOrdinal, requireOpenRoot);
  if (!root)
    return root.takeError();
  if (entity >= (*root)->entities.size())
    return detail::invalid("System entity handle has an invalid owner ordinal");
  detail::SystemEntityState &record = (*root)->entities[entity];
  if (record.kind != kind || !record.operation)
    return detail::invalid("System entity handle has the wrong typed owner");
  return &record;
}

template <typename Handle>
llvm::Expected<std::shared_ptr<detail::DesignState>>
ownedState(const Handle &handle,
           const std::shared_ptr<detail::DesignState> &expected,
           std::size_t rootOrdinal) {
  std::shared_ptr<detail::DesignState> state =
      detail::SystemHandleAccess::state(handle).lock();
  if (!state || state.get() != expected.get() ||
      detail::SystemHandleAccess::rootOrdinal(handle) != rootOrdinal ||
      state->consumed)
    return detail::invalid("foreign System handle cannot cross root owners");
  return state;
}

template <typename Handle>
llvm::Error checkOwned(const Handle &handle,
                       const std::shared_ptr<detail::DesignState> &expected,
                       std::size_t rootOrdinal) {
  auto state = ownedState(handle, expected, rootOrdinal);
  if (!state)
    return state.takeError();
  return llvm::Error::success();
}

mlir::OpBuilder systemInsertionBuilder(detail::DesignState &state,
                                       detail::SystemRootState &root) {
  mlir::OpBuilder builder(&state.context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  return builder;
}

::fabric::EntityIdAttr entityId(mlir::MLIRContext &context,
                                loom::fabric::FabricEntityId id) {
  return ::fabric::EntityIdAttr::get(&context, id);
}

llvm::Error verifyCreated(mlir::Operation *operation,
                          llvm::StringRef description) {
  if (mlir::succeeded(mlir::verify(operation)))
    return llvm::Error::success();
  operation->erase();
  return detail::invalid("Fabric rejected the typed " + description +
                         " operation");
}

llvm::Expected<loom::fabric::FabricModuleTemplateRef>
uniqueModuleTemplate(const loom::fabric::FabricArtifactView &view) {
  std::optional<loom::fabric::FabricModuleTemplateRef> result;
  for (loom::fabric::FabricEntityId id = 0;; ++id) {
    std::optional<loom::fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricModuleTemplate)
      continue;
    if (result)
      return detail::invalid(
          "imported Module contains more than one module template");
    result = loom::fabric::FabricModuleTemplateRef(id);
  }
  if (!result)
    return detail::invalid("imported Module has no module template");
  return *result;
}

HardwareDomainMember
makeDomainMember(const std::weak_ptr<detail::DesignState> &weak,
                 std::size_t rootOrdinal,
                 loom::fabric::FabricInventoryOwnerRef owner) {
  return detail::SystemHandleAccess::makeDomainMember(weak, rootOrdinal,
                                                      std::move(owner));
}

loom::fabric::FabricTransportEndpointOwnerRef
spatialTransportOwner(loom::fabric::FabricEntityId core) {
  return loom::fabric::FabricTransportEndpointOwnerRef::of(
      loom::fabric::SpatialCoreOccurrenceRef{
          loom::fabric::AccCoreOccurrenceRef(core)});
}

loom::fabric::FabricMemoryEndpointOwnerRef
spatialMemoryOwner(loom::fabric::FabricEntityId core) {
  return loom::fabric::FabricMemoryEndpointOwnerRef::of(
      loom::fabric::SpatialCoreOccurrenceRef{
          loom::fabric::AccCoreOccurrenceRef(core)});
}

} // namespace

HardwareDomainMember HostCore::domainMember() const {
  return makeDomainMember(state_, rootOrdinal_,
                          loom::fabric::FabricInventoryOwnerRef::of(
                              loom::fabric::HostCoreOccurrenceRef(entity_)));
}

HardwareDomainMember AccCore::domainMember() const {
  return makeDomainMember(state_, rootOrdinal_,
                          loom::fabric::FabricInventoryOwnerRef::of(
                              loom::fabric::AccCoreOccurrenceRef(entity_)));
}

HardwareDomainMember AccCore::instructionCoreDomainMember() const {
  return makeDomainMember(
      state_, rootOrdinal_,
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::InstructionCoreContextRef{
              loom::fabric::AccCoreOccurrenceRef(entity_)}));
}

HardwareDomainMember AccCore::spatialCoreDomainMember() const {
  return makeDomainMember(
      state_, rootOrdinal_,
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::SpatialCoreOccurrenceRef{
              loom::fabric::AccCoreOccurrenceRef(entity_)}));
}

HardwareDomainMember SystemMemoryService::domainMember() const {
  return makeDomainMember(
      state_, rootOrdinal_,
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::FabricMemoryServiceRef::system(
              loom::fabric::SystemMemoryServiceRef(entity_))));
}

HardwareDomainMember ExternalBoundary::domainMember() const {
  return makeDomainMember(state_, rootOrdinal_,
                          loom::fabric::FabricInventoryOwnerRef::of(
                              loom::fabric::ExternalBoundaryRef(entity_)));
}

HardwareDomainMember SystemServiceEndpoint::domainMember() const {
  return makeDomainMember(state_, rootOrdinal_,
                          loom::fabric::FabricInventoryOwnerRef::of(
                              loom::fabric::SystemServiceEndpointRef(entity_)));
}

HardwareDomainMember SystemTransportResource::domainMember() const {
  return makeDomainMember(
      state_, rootOrdinal_,
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::SystemTransportResourceRef(entity_)));
}

HardwareDomainMember SystemTransferPattern::domainMember() const {
  return makeDomainMember(
      state_, rootOrdinal_,
      loom::fabric::FabricInventoryOwnerRef::of(reference_));
}

HardwareDomainMember HardwareDomainBuilder::domainMember() const {
  return makeDomainMember(state_, rootOrdinal_,
                          loom::fabric::FabricInventoryOwnerRef::of(
                              loom::fabric::HardwareDomainRef(entity_)));
}

HardwareDomainMember ServiceTransformBuilder::domainMember() const {
  return makeDomainMember(
      state_, rootOrdinal_,
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::SystemServiceTransformRef(entity_)));
}

llvm::Expected<SystemTransportEndpoint>
AccCore::spatialTransportInput(std::size_t ordinal) const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity =
      activeEntity(*state, rootOrdinal_, entity_,
                   loom::fabric::FabricEntityKind::AccCoreOccurrence, false);
  if (!entity)
    return entity.takeError();
  const auto &root = (*state)->systemRoots[rootOrdinal_];
  if (!(*entity)->importedModule ||
      *(*entity)->importedModule >= root.importedModules.size())
    return detail::invalid("AccCore has no imported SpatialCore");
  const auto &module = root.importedModules[*(*entity)->importedModule];
  if (ordinal >= module.transportInputCount)
    return detail::invalid(
        "SpatialCore transport input ordinal is out of range");
  return SystemTransportEndpoint(
      *state, rootOrdinal_,
      {spatialTransportOwner(entity_), static_cast<std::uint64_t>(ordinal)},
      loom::fabric::FabricPortDirection::Input);
}

llvm::Expected<SystemTransportEndpoint>
AccCore::spatialTransportOutput(std::size_t ordinal) const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity =
      activeEntity(*state, rootOrdinal_, entity_,
                   loom::fabric::FabricEntityKind::AccCoreOccurrence, false);
  if (!entity)
    return entity.takeError();
  const auto &root = (*state)->systemRoots[rootOrdinal_];
  if (!(*entity)->importedModule ||
      *(*entity)->importedModule >= root.importedModules.size())
    return detail::invalid("AccCore has no imported SpatialCore");
  const auto &module = root.importedModules[*(*entity)->importedModule];
  if (ordinal >= module.transportOutputCount)
    return detail::invalid(
        "SpatialCore transport output ordinal is out of range");
  return SystemTransportEndpoint(
      *state, rootOrdinal_,
      {spatialTransportOwner(entity_),
       module.transportInputCount + static_cast<std::uint64_t>(ordinal)},
      loom::fabric::FabricPortDirection::Output);
}

llvm::Expected<SystemMemoryEndpoint>
AccCore::spatialMemoryManager(std::size_t ordinal) const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity =
      activeEntity(*state, rootOrdinal_, entity_,
                   loom::fabric::FabricEntityKind::AccCoreOccurrence, false);
  if (!entity)
    return entity.takeError();
  const auto &root = (*state)->systemRoots[rootOrdinal_];
  if (!(*entity)->importedModule ||
      *(*entity)->importedModule >= root.importedModules.size())
    return detail::invalid("AccCore has no imported SpatialCore");
  const auto &module = root.importedModules[*(*entity)->importedModule];
  if (ordinal >= module.memoryInputCount)
    return detail::invalid(
        "SpatialCore memory manager ordinal is out of range");
  return SystemMemoryEndpoint(
      *state, rootOrdinal_,
      {spatialMemoryOwner(entity_), static_cast<std::uint64_t>(ordinal)},
      loom::fabric::FabricMemoryEndpointRole::Manager);
}

llvm::Expected<SystemMemoryEndpoint>
AccCore::spatialMemorySubordinate(std::size_t ordinal) const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity =
      activeEntity(*state, rootOrdinal_, entity_,
                   loom::fabric::FabricEntityKind::AccCoreOccurrence, false);
  if (!entity)
    return entity.takeError();
  const auto &root = (*state)->systemRoots[rootOrdinal_];
  if (!(*entity)->importedModule ||
      *(*entity)->importedModule >= root.importedModules.size())
    return detail::invalid("AccCore has no imported SpatialCore");
  const auto &module = root.importedModules[*(*entity)->importedModule];
  if (ordinal >= module.memoryOutputCount)
    return detail::invalid(
        "SpatialCore memory subordinate ordinal is out of range");
  return SystemMemoryEndpoint(
      *state, rootOrdinal_,
      {spatialMemoryOwner(entity_),
       module.memoryInputCount + static_cast<std::uint64_t>(ordinal)},
      loom::fabric::FabricMemoryEndpointRole::Subordinate);
}

llvm::Expected<SystemTransportEndpoint>
SystemTransportResource::input(std::size_t ordinal) const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity = activeEntity(
      *state, rootOrdinal_, entity_,
      loom::fabric::FabricEntityKind::SystemTransportResource, false);
  if (!entity)
    return entity.takeError();
  if (ordinal >= (*entity)->inputCount)
    return detail::invalid("transport resource input ordinal is out of range");
  return SystemTransportEndpoint(
      *state, rootOrdinal_,
      {loom::fabric::FabricTransportEndpointOwnerRef::of(
           loom::fabric::SystemTransportResourceRef(entity_)),
       static_cast<std::uint64_t>(ordinal)},
      loom::fabric::FabricPortDirection::Input);
}

llvm::Expected<SystemTransportEndpoint>
SystemTransportResource::output(std::size_t ordinal) const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity = activeEntity(
      *state, rootOrdinal_, entity_,
      loom::fabric::FabricEntityKind::SystemTransportResource, false);
  if (!entity)
    return entity.takeError();
  if (ordinal >= (*entity)->outputCount)
    return detail::invalid("transport resource output ordinal is out of range");
  return SystemTransportEndpoint(
      *state, rootOrdinal_,
      {loom::fabric::FabricTransportEndpointOwnerRef::of(
           loom::fabric::SystemTransportResourceRef(entity_)),
       (*entity)->inputCount + static_cast<std::uint64_t>(ordinal)},
      loom::fabric::FabricPortDirection::Output);
}

llvm::Expected<SystemTransportEndpoint>
SystemServiceEndpoint::transport() const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity = activeEntity(
      *state, rootOrdinal_, entity_,
      loom::fabric::FabricEntityKind::SystemServiceEndpoint, false);
  if (!entity)
    return entity.takeError();
  if ((*entity)->endpointPlane !=
      loom::fabric::CanonicalServiceEndpointPlane::Transport)
    return detail::invalid("service endpoint is not a message transport");
  const auto direction =
      (*entity)->endpointRole ==
              loom::fabric::CanonicalServiceEndpointRole::Initiate
          ? loom::fabric::FabricPortDirection::Output
          : loom::fabric::FabricPortDirection::Input;
  return SystemTransportEndpoint(
      *state, rootOrdinal_,
      {loom::fabric::FabricTransportEndpointOwnerRef::of(
           loom::fabric::SystemServiceEndpointRef(entity_)),
       0},
      direction);
}

llvm::Expected<SystemMemoryEndpoint> SystemServiceEndpoint::memory() const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity = activeEntity(
      *state, rootOrdinal_, entity_,
      loom::fabric::FabricEntityKind::SystemServiceEndpoint, false);
  if (!entity)
    return entity.takeError();
  if ((*entity)->endpointPlane !=
      loom::fabric::CanonicalServiceEndpointPlane::Memory)
    return detail::invalid("service endpoint is not a memory capability");
  const auto role = (*entity)->endpointRole ==
                            loom::fabric::CanonicalServiceEndpointRole::Initiate
                        ? loom::fabric::FabricMemoryEndpointRole::Manager
                        : loom::fabric::FabricMemoryEndpointRole::Subordinate;
  return SystemMemoryEndpoint(
      *state, rootOrdinal_,
      {loom::fabric::FabricMemoryEndpointOwnerRef::of(
           loom::fabric::SystemServiceEndpointRef(entity_)),
       0},
      role);
}

llvm::Expected<ImportedSpatialCore> SystemBuilder::importSpatialCore(
    const loom::fabric::FinalizedFabricRoot &module) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  if (module.view().rootKind() != loom::fabric::FabricRootKind::Module)
    return detail::invalid("System can import only a finalized Module root");

  auto moduleTemplate = uniqueModuleTemplate(module.view());
  if (!moduleTemplate)
    return moduleTemplate.takeError();
  for (std::size_t index = 0; index < (*root)->importedModules.size(); ++index)
    if ((*root)->importedModules[index].reference == module.reference())
      return ImportedSpatialCore(*state, rootOrdinal_, index);

  detail::ImportedModuleState imported{module.reference(), *moduleTemplate};
  for (loom::fabric::FabricPortDirection direction :
       {loom::fabric::FabricPortDirection::Input,
        loom::fabric::FabricPortDirection::Output}) {
    const std::uint64_t count =
        module.view().moduleBoundaryEndpointCount(*moduleTemplate, direction);
    std::vector<detail::ImportedModuleBoundary> &boundaries =
        direction == loom::fabric::FabricPortDirection::Input
            ? imported.inputs
            : imported.outputs;
    boundaries.reserve(count);
    for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal) {
      loom::fabric::FabricModuleBoundaryEndpointRef endpoint{
          *moduleTemplate, direction, ordinal};
      auto plane = module.view().moduleBoundaryEndpointPlane(endpoint);
      auto occurrence =
          module.view().moduleBoundaryEndpointOccurrenceOrdinal(endpoint);
      if (!plane || !occurrence)
        return detail::invalid(
            "finalized Module has an incomplete boundary projection");
      boundaries.push_back({*plane, *occurrence});
      const bool input = direction == loom::fabric::FabricPortDirection::Input;
      if (*plane ==
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
        (input ? imported.transportInputCount
               : imported.transportOutputCount)++;
      else
        (input ? imported.memoryInputCount : imported.memoryOutputCount)++;
    }
  }
  const std::size_t ordinal = (*root)->importedModules.size();
  (*root)->importedModules.push_back(std::move(imported));
  return ImportedSpatialCore(*state, rootOrdinal_, ordinal);
}

llvm::Expected<HostCore> SystemBuilder::addHostCore(
    const loom::fabric::InstructionCoreArchitecturalContract &architecture,
    const loom::fabric::InstructionCoreMicroarchitecturalRealization
        &microarchitecture) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto architectureBytes =
      loom::fabric::encodeInstructionCoreArchitecturalContract(architecture);
  if (!architectureBytes)
    return architectureBytes.takeError();
  auto microarchitectureBytes =
      loom::fabric::encodeInstructionCoreMicroarchitecturalRealization(
          microarchitecture);
  if (!microarchitectureBytes)
    return microarchitectureBytes.takeError();

  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemHostCoreOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id),
      denseBytes((*state)->context, *architectureBytes),
      denseBytes((*state)->context, *microarchitectureBytes));
  if (llvm::Error error = verifyCreated(operation, "HostCore"))
    return error;
  (*root)->entities.push_back(
      {loom::fabric::FabricEntityKind::HostCoreOccurrence, operation});
  return HostCore(*state, rootOrdinal_, id);
}

llvm::Expected<AccCore> SystemBuilder::addAccCore(
    const loom::fabric::InstructionCoreArchitecturalContract &architecture,
    const loom::fabric::InstructionCoreMicroarchitecturalRealization
        &microarchitecture,
    const ImportedSpatialCore &spatialCore) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto importedState = ownedState(spatialCore, *state, rootOrdinal_);
  if (!importedState)
    return importedState.takeError();
  if (spatialCore.importOrdinal_ >= (*root)->importedModules.size())
    return detail::invalid("ImportedSpatialCore ordinal is out of range");
  const detail::ImportedModuleState &imported =
      (*root)->importedModules[spatialCore.importOrdinal_];

  auto architectureBytes =
      loom::fabric::encodeInstructionCoreArchitecturalContract(architecture);
  if (!architectureBytes)
    return architectureBytes.takeError();
  auto microarchitectureBytes =
      loom::fabric::encodeInstructionCoreMicroarchitecturalRealization(
          microarchitecture);
  if (!microarchitectureBytes)
    return microarchitectureBytes.takeError();
  const loom::fabric::FabricImportedModuleTargetRef target{
      spatialCore.importOrdinal_, imported.module};

  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemAccCoreOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id),
      denseBytes((*state)->context, *architectureBytes),
      denseBytes((*state)->context, *microarchitectureBytes),
      denseBytes((*state)->context,
                 loom::fabric::encodeFabricImportedModuleTargetRef(target)));
  if (llvm::Error error = verifyCreated(operation, "AccCore"))
    return error;

  std::vector<mlir::Operation *> attachments;
  auto emitAttachments =
      [&](llvm::ArrayRef<detail::ImportedModuleBoundary> boundaries,
          loom::fabric::FabricPortDirection direction) -> llvm::Error {
    for (auto indexedBoundary : llvm::enumerate(boundaries)) {
      const std::uint64_t ordinal = indexedBoundary.index();
      const detail::ImportedModuleBoundary &boundary = indexedBoundary.value();
      const loom::fabric::FabricImportedModuleBoundaryEndpointRef
          moduleEndpoint{spatialCore.importOrdinal_,
                         {imported.module, direction,
                          static_cast<std::uint64_t>(ordinal)}};
      llvm::Expected<loom::fabric::FabricSpatialAttachmentEndpointRef>
          spatialEndpoint = [&]()
          -> llvm::Expected<loom::fabric::FabricSpatialAttachmentEndpointRef> {
        if (boundary.plane ==
            loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport)
          return loom::fabric::FabricSpatialAttachmentEndpointRef::create(
              loom::fabric::FabricTransportEndpointRef{
                  spatialTransportOwner(id), boundary.occurrenceOrdinal});
        return loom::fabric::FabricSpatialAttachmentEndpointRef::create(
            loom::fabric::FabricMemoryEndpointRef{spatialMemoryOwner(id),
                                                  boundary.occurrenceOrdinal});
      }();
      if (!spatialEndpoint)
        return spatialEndpoint.takeError();
      auto attachment = ::fabric::SystemSpatialAttachmentOp::create(
          builder, (*root)->operation.getLoc(),
          denseBytes(
              (*state)->context,
              loom::fabric::encodeFabricImportedModuleBoundaryEndpointRef(
                  moduleEndpoint)),
          denseBytes((*state)->context,
                     loom::fabric::encodeFabricSpatialAttachmentEndpointRef(
                         *spatialEndpoint)));
      if (mlir::failed(mlir::verify(attachment))) {
        attachment->erase();
        return detail::invalid(
            "Fabric rejected the typed SpatialCore attachment operation");
      }
      attachments.push_back(attachment);
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = emitAttachments(
          imported.inputs, loom::fabric::FabricPortDirection::Input)) {
    for (mlir::Operation *attachment : llvm::reverse(attachments))
      attachment->erase();
    operation->erase();
    return error;
  }
  if (llvm::Error error = emitAttachments(
          imported.outputs, loom::fabric::FabricPortDirection::Output)) {
    for (mlir::Operation *attachment : llvm::reverse(attachments))
      attachment->erase();
    operation->erase();
    return error;
  }

  detail::SystemEntityState entity{
      loom::fabric::FabricEntityKind::AccCoreOccurrence, operation};
  entity.importedModule = spatialCore.importOrdinal_;
  (*root)->entities.push_back(std::move(entity));
  return AccCore(*state, rootOrdinal_, id);
}

llvm::Expected<SystemMemoryService> SystemBuilder::addMemoryService(
    const ::fabric::MemoryServiceContractRecord &contract) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto encoded = ::fabric::encodeMemoryServiceContractRecord(contract);
  if (!encoded)
    return encoded.takeError();
  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto attribute = ::fabric::MemoryServiceContractAttr::get(
      &(*state)->context, denseBytes((*state)->context, *encoded));
  auto operation = ::fabric::SystemMemoryServiceOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id),
      attribute);
  if (llvm::Error error = verifyCreated(operation, "System memory service"))
    return error;
  (*root)->entities.push_back(
      {loom::fabric::FabricEntityKind::SystemMemoryService, operation});
  return SystemMemoryService(*state, rootOrdinal_, id);
}

llvm::Expected<ExternalBoundary> SystemBuilder::addExternalBoundary() {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemExternalBoundaryOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id));
  if (llvm::Error error = verifyCreated(operation, "external boundary"))
    return error;
  (*root)->entities.push_back(
      {loom::fabric::FabricEntityKind::ExternalBoundary, operation});
  return ExternalBoundary(*state, rootOrdinal_, id);
}

llvm::Expected<HardwareDomainBuilder> SystemBuilder::createHardwareDomain() {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemHardwareDomainOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id),
      denseBytes((*state)->context, {}));
  detail::SystemEntityState entity{
      loom::fabric::FabricEntityKind::HardwareDomain, operation};
  entity.closed = false;
  (*root)->entities.push_back(std::move(entity));
  return HardwareDomainBuilder(*state, rootOrdinal_, id);
}

llvm::Expected<ServiceTransformBuilder>
SystemBuilder::createServiceTransform() {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemServiceTransformOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id),
      denseBytes((*state)->context, {}));
  detail::SystemEntityState entity{
      loom::fabric::FabricEntityKind::SystemServiceTransform, operation};
  entity.closed = false;
  (*root)->entities.push_back(std::move(entity));
  return ServiceTransformBuilder(*state, rootOrdinal_, id);
}

llvm::Expected<loom::fabric::ServiceRateContractRecord>
SystemBuilder::createServiceRate(const HardwareDomainBuilder &clock,
                                 std::uint64_t operationsPerWindow,
                                 std::uint64_t windowTicks,
                                 std::uint64_t maxOutstanding,
                                 loom::fabric::ServiceProgress progress) const {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto clockState = ownedState(clock, *state, rootOrdinal_);
  if (!clockState)
    return clockState.takeError();
  auto entity = activeEntity(*state, rootOrdinal_, clock.entity_,
                             loom::fabric::FabricEntityKind::HardwareDomain);
  if (!entity)
    return entity.takeError();
  return loom::fabric::ServiceRateContractRecord::create(
      loom::fabric::ClockDomainRef(
          loom::fabric::HardwareDomainRef(clock.entity_)),
      operationsPerWindow, windowTicks, maxOutstanding, std::move(progress));
}

llvm::Expected<SystemServiceEndpoint> SystemBuilder::addServiceEndpoint(
    const HostCore &owner,
    const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    std::optional<PortType> carrier) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(owner, *state, rootOrdinal_))
    return error;
  return addServiceEndpoint(
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::HostCoreOccurrenceRef(owner.entity_)),
      capabilities, std::move(carrier));
}

llvm::Expected<SystemServiceEndpoint> SystemBuilder::addServiceEndpoint(
    const AccCore &owner,
    const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    std::optional<PortType> carrier) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(owner, *state, rootOrdinal_))
    return error;
  return addServiceEndpoint(
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::AccCoreOccurrenceRef(owner.entity_)),
      capabilities, std::move(carrier));
}

llvm::Expected<SystemServiceEndpoint> SystemBuilder::addServiceEndpoint(
    const SystemMemoryService &owner,
    const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    std::optional<PortType> carrier) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(owner, *state, rootOrdinal_))
    return error;
  return addServiceEndpoint(
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::FabricMemoryServiceRef::system(
              loom::fabric::SystemMemoryServiceRef(owner.entity_))),
      capabilities, std::move(carrier));
}

llvm::Expected<SystemServiceEndpoint> SystemBuilder::addServiceEndpoint(
    const ServiceTransformBuilder &owner,
    const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    std::optional<PortType> carrier) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(owner, *state, rootOrdinal_))
    return error;
  return addServiceEndpoint(
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::SystemServiceTransformRef(owner.entity_)),
      capabilities, std::move(carrier));
}

llvm::Expected<SystemServiceEndpoint> SystemBuilder::addServiceEndpoint(
    const ExternalBoundary &owner,
    const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    std::optional<PortType> carrier) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(owner, *state, rootOrdinal_))
    return error;
  return addServiceEndpoint(
      loom::fabric::FabricInventoryOwnerRef::of(
          loom::fabric::ExternalBoundaryRef(owner.entity_)),
      capabilities, std::move(carrier));
}

llvm::Expected<SystemServiceEndpoint> SystemBuilder::addServiceEndpoint(
    loom::fabric::FabricInventoryOwnerRef owner,
    const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    std::optional<PortType> carrier) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto ownerRef =
      loom::fabric::SystemServiceEndpointOwnerRef::create(std::move(owner));
  if (!ownerRef)
    return ownerRef.takeError();
  auto capabilityBytes =
      loom::fabric::encodeCanonicalServiceCapabilitySet(capabilities);
  if (!capabilityBytes)
    return capabilityBytes.takeError();
  const auto plane = capabilities.plane();
  if (plane == loom::fabric::CanonicalServiceEndpointPlane::Transport) {
    if (!carrier || carrier->kind() == PortType::Kind::Memory)
      return detail::invalid(
          "message service endpoint requires a transport carrier");
  } else if (carrier) {
    return detail::invalid(
        "memory service endpoint cannot declare a message carrier");
  }

  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  mlir::TypeAttr carrierAttribute;
  if (carrier)
    carrierAttribute = mlir::TypeAttr::get(
        detail::materializePortType((*state)->context, *carrier));
  auto operation = ::fabric::SystemServiceEndpointOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id),
      denseBytes((*state)->context,
                 loom::fabric::encodeSystemServiceEndpointOwnerRef(*ownerRef)),
      denseBytes((*state)->context, *capabilityBytes), carrierAttribute);
  if (llvm::Error error = verifyCreated(operation, "service endpoint"))
    return error;
  detail::SystemEntityState entity{
      loom::fabric::FabricEntityKind::SystemServiceEndpoint, operation};
  entity.endpointPlane = plane;
  entity.endpointRole = capabilities.role();
  (*root)->entities.push_back(std::move(entity));
  return SystemServiceEndpoint(*state, rootOrdinal_, id);
}

llvm::Expected<SystemTransportResource>
SystemBuilder::addTransportResource(const SystemTransportResourceSpec &spec) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  if (spec.inputTypes.empty() || spec.outputTypes.empty())
    return detail::invalid(
        "System transport resource requires input and output ports");
  llvm::SmallVector<mlir::Type, 4> inputs;
  llvm::SmallVector<mlir::Type, 4> outputs;
  for (const PortType &type : spec.inputTypes) {
    if (type.kind() == PortType::Kind::Memory)
      return detail::invalid("transport resource cannot have a memory port");
    inputs.push_back(detail::materializePortType((*state)->context, type));
  }
  for (const PortType &type : spec.outputTypes) {
    if (type.kind() == PortType::Kind::Memory)
      return detail::invalid("transport resource cannot have a memory port");
    outputs.push_back(detail::materializePortType((*state)->context, type));
  }
  auto contractBytes =
      ::fabric::encodeResourceContractRecord(spec.resourceContract);
  if (!contractBytes)
    return contractBytes.takeError();
  const auto id =
      static_cast<loom::fabric::FabricEntityId>((*root)->entities.size());
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemTransportResourceOp::create(
      builder, (*root)->operation.getLoc(), entityId((*state)->context, id),
      mlir::TypeAttr::get(
          mlir::FunctionType::get(&(*state)->context, inputs, outputs)),
      denseBytes((*state)->context, *contractBytes), mlir::DenseI8ArrayAttr());
  if (llvm::Error error = verifyCreated(operation, "transport resource"))
    return error;
  detail::SystemEntityState entity{
      loom::fabric::FabricEntityKind::SystemTransportResource, operation};
  entity.inputCount = inputs.size();
  entity.outputCount = outputs.size();
  (*root)->entities.push_back(std::move(entity));
  return SystemTransportResource(*state, rootOrdinal_, id);
}

llvm::Expected<SystemTransferPattern>
SystemBuilder::addTransferPattern(const SystemTransportResource &resource,
                                  std::size_t inputOrdinal,
                                  llvm::ArrayRef<std::uint32_t> outputOrdinals,
                                  std::uint32_t usePatternOrdinal) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(resource, *state, rootOrdinal_))
    return error;
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto entity =
      activeEntity(*state, rootOrdinal_, resource.entity_,
                   loom::fabric::FabricEntityKind::SystemTransportResource);
  if (!entity)
    return entity.takeError();
  if (inputOrdinal >= (*entity)->inputCount)
    return detail::invalid("transfer pattern input ordinal is out of range");
  if (outputOrdinals.empty())
    return detail::invalid("transfer pattern requires an output");

  const loom::fabric::SystemTransportResourceRef resourceRef(resource.entity_);
  const auto endpointOwner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(resourceRef);
  std::vector<loom::fabric::FabricTransportEndpointRef> egresses;
  egresses.reserve(outputOrdinals.size());
  for (std::uint32_t ordinal : outputOrdinals) {
    if (ordinal >= (*entity)->outputCount)
      return detail::invalid("transfer pattern output ordinal is out of range");
    egresses.push_back({endpointOwner, (*entity)->inputCount + ordinal});
  }
  const loom::fabric::FabricTransferPatternRef pattern{
      resourceRef, (*entity)->nextTransferPatternOrdinal};
  const loom::fabric::FabricUsePatternRef usePattern{
      loom::fabric::FabricUsePatternOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(resourceRef)),
      usePatternOrdinal};
  auto record = loom::fabric::SystemTransferPatternRecord::create(
      pattern, {endpointOwner, inputOrdinal}, std::move(egresses), usePattern);
  if (!record)
    return record.takeError();
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemTransferPatternOp::create(
      builder, (*root)->operation.getLoc(),
      denseBytes((*state)->context,
                 loom::fabric::encodeSystemTransferPatternRecord(*record)));
  if (llvm::Error error = verifyCreated(operation, "transfer pattern"))
    return error;
  ++(*entity)->nextTransferPatternOrdinal;
  return SystemTransferPattern(*state, rootOrdinal_, pattern);
}

llvm::Error SystemBuilder::attachServiceLegCarriers(
    const SystemMemoryEndpoint &endpoint, dataflow::semantics::ServiceKind kind,
    dataflow::StructuralOrdinal legOrdinal,
    llvm::ArrayRef<SystemTransportEndpoint> carriers) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(endpoint, *state, rootOrdinal_))
    return error;

  std::vector<loom::fabric::FabricTransportEndpointRef> carrierRefs;
  carrierRefs.reserve(carriers.size());
  for (const SystemTransportEndpoint &carrier : carriers) {
    if (llvm::Error error = checkOwned(carrier, *state, rootOrdinal_))
      return error;
    carrierRefs.push_back(carrier.reference_);
  }
  auto record = loom::fabric::ServiceLegCarrierAttachmentRecord::create(
      endpoint.reference_, kind, legOrdinal, std::move(carrierRefs));
  if (!record)
    return record.takeError();
  auto encoded = loom::fabric::encodeServiceLegCarrierAttachmentRecord(*record);
  if (!encoded)
    return encoded.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemServiceLegCarrierAttachmentOp::create(
      builder, (*root)->operation.getLoc(),
      denseBytes((*state)->context, *encoded));
  return verifyCreated(operation, "service-leg carrier attachment");
}

llvm::Error
SystemBuilder::addClockCrossing(const SystemTransportResource &resource,
                                const SystemTransferPattern &pattern,
                                const HardwareDomainBuilder &sourceClock,
                                const HardwareDomainBuilder &destinationClock,
                                std::uint32_t depth,
                                std::uint32_t synchronizerStages) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(resource, *state, rootOrdinal_))
    return error;
  if (llvm::Error error = checkOwned(pattern, *state, rootOrdinal_))
    return error;
  if (llvm::Error error = checkOwned(sourceClock, *state, rootOrdinal_))
    return error;
  if (llvm::Error error = checkOwned(destinationClock, *state, rootOrdinal_))
    return error;
  auto entity =
      activeEntity(*state, rootOrdinal_, resource.entity_,
                   loom::fabric::FabricEntityKind::SystemTransportResource);
  if (!entity)
    return entity.takeError();
  if (pattern.reference_.resource !=
      loom::fabric::SystemTransportResourceRef(resource.entity_))
    return detail::invalid(
        "clock crossing pattern belongs to another transport resource");
  if ((*entity)->crossingDeclared)
    return detail::invalid("transport resource already has a clock crossing");
  auto crossing = loom::fabric::ClockCrossingContractRecord::createAsyncFifo(
      pattern.reference_,
      loom::fabric::ClockDomainRef(
          loom::fabric::HardwareDomainRef(sourceClock.entity_)),
      loom::fabric::ClockDomainRef(
          loom::fabric::HardwareDomainRef(destinationClock.entity_)),
      depth, synchronizerStages);
  if (!crossing)
    return crossing.takeError();
  auto encoded = loom::fabric::encodeClockCrossingContractRecord(*crossing);
  if (!encoded)
    return encoded.takeError();
  auto operation =
      llvm::cast<::fabric::SystemTransportResourceOp>((*entity)->operation);
  operation.setClockCrossingAttr(denseBytes((*state)->context, *encoded));
  (*entity)->crossingDeclared = true;
  return llvm::Error::success();
}

llvm::Error SystemBuilder::connect(const SystemTransportEndpoint &source,
                                   const SystemTransportEndpoint &destination) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  if (llvm::Error error = checkOwned(source, *state, rootOrdinal_))
    return error;
  if (llvm::Error error = checkOwned(destination, *state, rootOrdinal_))
    return error;
  if (source.direction_ != loom::fabric::FabricPortDirection::Output ||
      destination.direction_ != loom::fabric::FabricPortDirection::Input)
    return detail::invalid("System connection must be output-to-input");
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  mlir::OpBuilder builder = systemInsertionBuilder(**state, **root);
  auto operation = ::fabric::SystemConnectionOp::create(
      builder, (*root)->operation.getLoc(),
      denseBytes((*state)->context,
                 loom::fabric::canonicalFabricBytes(source.reference_)),
      denseBytes((*state)->context,
                 loom::fabric::canonicalFabricBytes(destination.reference_)));
  return verifyCreated(operation, "System connection");
}

llvm::Error
HardwareDomainBuilder::close(llvm::ArrayRef<HardwareDomainMember> members,
                             loom::fabric::HardwareDomainContract contract) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity = activeEntity(*state, rootOrdinal_, entity_,
                             loom::fabric::FabricEntityKind::HardwareDomain);
  if (!entity)
    return entity.takeError();
  if ((*entity)->closed)
    return detail::invalid("hardware domain is already closed");
  std::vector<loom::fabric::FabricInventoryOwnerRef> references;
  references.reserve(members.size());
  for (const HardwareDomainMember &member : members) {
    if (llvm::Error error = checkOwned(member, *state, rootOrdinal_))
      return error;
    references.push_back(member.owner_);
  }
  auto record = loom::fabric::HardwareDomainContractRecord::create(
      std::move(references), std::move(contract));
  if (!record)
    return record.takeError();
  auto encoded = loom::fabric::encodeHardwareDomainContractRecord(*record);
  if (!encoded)
    return encoded.takeError();
  auto operation =
      llvm::cast<::fabric::SystemHardwareDomainOp>((*entity)->operation);
  operation.setContractAttr(denseBytes((*state)->context, *encoded));
  if (mlir::failed(mlir::verify(operation)))
    return detail::invalid("Fabric rejected the typed hardware domain");
  (*entity)->closed = true;
  return llvm::Error::success();
}

llvm::Error ServiceTransformBuilder::close(
    llvm::ArrayRef<SystemMemoryEndpoint> inputs,
    llvm::ArrayRef<SystemMemoryEndpoint> outputs,
    loom::fabric::ServiceTransformContract contract) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto entity =
      activeEntity(*state, rootOrdinal_, entity_,
                   loom::fabric::FabricEntityKind::SystemServiceTransform);
  if (!entity)
    return entity.takeError();
  if ((*entity)->closed)
    return detail::invalid("service transform is already closed");
  std::vector<loom::fabric::FabricMemoryEndpointRef> inputRefs;
  std::vector<loom::fabric::FabricMemoryEndpointRef> outputRefs;
  inputRefs.reserve(inputs.size());
  outputRefs.reserve(outputs.size());
  for (const SystemMemoryEndpoint &input : inputs) {
    if (llvm::Error error = checkOwned(input, *state, rootOrdinal_))
      return error;
    if (input.role_ != loom::fabric::FabricMemoryEndpointRole::Manager)
      return detail::invalid("service transform input must be a manager");
    inputRefs.push_back(input.reference_);
  }
  for (const SystemMemoryEndpoint &output : outputs) {
    if (llvm::Error error = checkOwned(output, *state, rootOrdinal_))
      return error;
    if (output.role_ != loom::fabric::FabricMemoryEndpointRole::Subordinate)
      return detail::invalid("service transform output must be a subordinate");
    outputRefs.push_back(output.reference_);
  }
  auto record = loom::fabric::SystemServiceTransformRecord::create(
      std::move(inputRefs), std::move(outputRefs), std::move(contract));
  if (!record)
    return record.takeError();
  auto encoded = loom::fabric::encodeSystemServiceTransformRecord(*record);
  if (!encoded)
    return encoded.takeError();
  auto operation =
      llvm::cast<::fabric::SystemServiceTransformOp>((*entity)->operation);
  operation.setContractAttr(denseBytes((*state)->context, *encoded));
  if (mlir::failed(mlir::verify(operation)))
    return detail::invalid("Fabric rejected the typed service transform");
  (*entity)->closed = true;
  return llvm::Error::success();
}

llvm::Error SystemBuilder::close() {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = activeSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  for (const detail::SystemEntityState &entity : (*root)->entities)
    if (!entity.closed)
      return detail::invalid(
          "System contains an incomplete scoped entity definition");
  if (mlir::failed(mlir::verify((*root)->operation)))
    return detail::invalid("Fabric rejected the typed System root");
  (*root)->closed = true;
  return llvm::Error::success();
}

llvm::Expected<SystemBuilder>
DesignBuilder::createSystem(llvm::StringRef label) {
  if (!state_ || state_->consumed)
    return detail::invalid("DesignBuilder is already consumed");
  if (label.empty())
    return detail::invalid("System diagnostic label cannot be empty");
  if (!state_->labels.insert(label).second)
    return detail::invalid("duplicate Fabric root diagnostic label '" + label +
                           "'");
  mlir::OpBuilder builder(&state_->context);
  builder.setInsertionPointToEnd(state_->draft->getBody());
  auto operation =
      ::fabric::SystemOp::create(builder, state_->draft->getLoc(), label);
  operation.getBody().push_back(new mlir::Block());
  const std::size_t ordinal = state_->systemRoots.size();
  state_->systemRoots.push_back(
      detail::SystemRootState{operation, label.str()});
  return SystemBuilder(state_, ordinal);
}

} // namespace loom::adg
