#include "Fabric/Identity/FabricRefs.h"

#include "llvm/Support/ErrorHandling.h"

using namespace loom;
using namespace loom::fabric;

char FabricRefError::ID = 0;

namespace {

template <typename Ref>
FabricInventoryOwnerRef inventoryOwnerFor(const Ref &ref) {
  return FabricInventoryOwnerRef::of(ref);
}

llvm::Error modulePhysicalRoleError(const llvm::Twine &message) {
  return makeFabricRefError(FabricRefErrorKind::InvalidOwnerFamily, message);
}

template <typename Ref>
llvm::Error acceptFabricModulePhysicalOwner(const Ref &) {
  return llvm::Error::success();
}

llvm::Error validateFabricModuleLocalMemoryServiceOwner(
    const LocalMemoryServiceRef &service) {
  if (service.underlying().kind() == FabricMemoryServiceKind::Local)
    return llvm::Error::success();
  return modulePhysicalRoleError(
      "a Module-local memory service cannot select a System service");
}

template <typename Ref>
llvm::Error rejectFabricModuleInventoryOwner(const Ref &) {
  return modulePhysicalRoleError(
      "the inventory owner is not declared inside one reusable Module");
}

#define LOOM_FABRIC_MODULE_PHYSICAL_OWNER(Ordinal, Name, Type, Validator)      \
  llvm::Error validateFabricModuleInventoryOwner(const Type &value) {          \
    llvm::Expected<FabricModulePhysicalOwnerRef> owner =                       \
        FabricModulePhysicalOwnerRef::create(value);                           \
    if (!owner)                                                                \
      return owner.takeError();                                                \
    return llvm::Error::success();                                             \
  }
#include "Fabric/Identity/FabricRefs.def"

llvm::Error
validateFabricModuleInventoryOwner(const FabricMemoryServiceRef &service) {
  return validateFabricModuleInventoryOwner(LocalMemoryServiceRef(service));
}

template <typename Ref>
llvm::Error validateFabricModuleInventoryOwner(const Ref &value) {
  return rejectFabricModuleInventoryOwner(value);
}

llvm::Error
validateFabricModuleInventoryOwner(const FabricInventoryOwnerRef &owner) {
  return std::visit(
      [](const auto &value) {
        return validateFabricModuleInventoryOwner(value);
      },
      owner.payload);
}

template <typename Ref>
llvm::Error acceptFabricModulePhysicalTarget(const Ref &) {
  return llvm::Error::success();
}

llvm::Error validateFabricModuleTransportTarget(
    const FabricTransportEndpointRef &endpoint) {
  return validateFabricModuleInventoryOwner(
      projectFabricInventoryOwner(endpoint.owner));
}

llvm::Error validateFabricModuleMemoryEndpointTarget(
    const FabricMemoryEndpointRef &endpoint) {
  return validateFabricModuleInventoryOwner(
      projectFabricInventoryOwner(endpoint.owner));
}

llvm::Error validateFabricModuleMemoryServiceRegionTarget(
    const FabricMemoryServiceRegionRef &region) {
  return validateFabricModuleInventoryOwner(region.service);
}

template <typename Ref>
llvm::Error validateFabricModuleInventoryTarget(const Ref &ref) {
  return validateFabricModuleInventoryOwner(ref.owner.catalog());
}

llvm::Error validateFabricModulePhysicalTraversalTarget(
    const FabricPhysicalTraversalRef &traversal) {
  switch (traversal.kind()) {
  case FabricPhysicalTraversalKind::PointConnection: {
    const auto &connection =
        std::get<FabricPointConnectionPayload>(traversal.payload);
    if (llvm::Error error =
            validateFabricModuleTransportTarget(connection.source))
      return error;
    return validateFabricModuleTransportTarget(connection.destination);
  }
  case FabricPhysicalTraversalKind::PeSelectorTraversal: {
    const auto &selector = std::get<FabricPeSelectorPayload>(traversal.payload);
    if (llvm::Error error =
            validateFabricModuleTransportTarget(selector.source))
      return error;
    return validateFabricModuleTransportTarget(selector.destination);
  }
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal:
  case FabricPhysicalTraversalKind::SwitchTraversal:
  case FabricPhysicalTraversalKind::FifoTraversal:
  case FabricPhysicalTraversalKind::BoundaryTraversal:
    return llvm::Error::success();
  case FabricPhysicalTraversalKind::SystemTransferPatternLeg:
    return modulePhysicalRoleError(
        "a System transfer-pattern leg is not a Module-local traversal");
  }
  llvm_unreachable("closed traversal kind outside its declaration");
}

llvm::Error systemPhysicalRoleError(const llvm::Twine &message) {
  return makeFabricRefError(FabricRefErrorKind::InvalidOwnerFamily, message);
}

template <typename Ref> llvm::Error acceptFabricDirectSystemOwner(const Ref &) {
  return llvm::Error::success();
}

template <typename Ref>
llvm::Error acceptFabricClockResetDirectOwner(const Ref &) {
  return llvm::Error::success();
}

template <typename Ref>
llvm::Error rejectFabricClockResetDirectOwner(const Ref &) {
  return systemPhysicalRoleError(
      "the direct System owner is not a Clock/Reset domain member");
}

llvm::Error
validateFabricSystemMemoryServiceOwner(const FabricMemoryServiceRef &service) {
  if (service.kind() == FabricMemoryServiceKind::System)
    return llvm::Error::success();
  return systemPhysicalRoleError(
      "a direct System owner cannot select a Module-local memory service");
}

#define LOOM_FABRIC_DIRECT_SYSTEM_OWNER(Name, Type, DirectValidator,           \
                                        ClockResetValidator)                   \
  static_assert(                                                               \
      std::is_same_v<                                                          \
          std::variant_alternative_t<static_cast<std::size_t>(                 \
                                         FabricInventoryOwnerKind::Name),      \
                                     FabricInventoryOwnerRef::Payload>,        \
          Type>,                                                               \
      "direct System owner must reuse its inventory-owner constructor");
#include "Fabric/Identity/FabricRefs.def"

llvm::Error
validateFabricDirectSystemOwner(const FabricInventoryOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_DIRECT_SYSTEM_OWNER(Name, Type, DirectValidator,           \
                                        ClockResetValidator)                   \
  case FabricInventoryOwnerKind::Name:                                         \
    return DirectValidator(std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  default:
    return systemPhysicalRoleError(
        "the inventory owner is not declared by the System root");
  }
}

llvm::Error
validateFabricClockResetDirectOwner(const FabricInventoryOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_DIRECT_SYSTEM_OWNER(Name, Type, DirectValidator,           \
                                        ClockResetValidator)                   \
  case FabricInventoryOwnerKind::Name:                                         \
    return ClockResetValidator(std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  default:
    return systemPhysicalRoleError(
        "the inventory owner is not a direct System Clock/Reset owner");
  }
}

template <typename Ref>
llvm::Error acceptFabricSystemPhysicalRole(const Ref &) {
  return llvm::Error::success();
}

llvm::Error validateFabricSpatialCoreTransportBoundary(
    const FabricTransportEndpointRef &endpoint) {
  if (endpoint.owner.kind() ==
      FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
    return llvm::Error::success();
  return systemPhysicalRoleError(
      "a SpatialCore transport boundary must name its exact occurrence");
}

llvm::Error validateFabricSpatialCoreMemoryBoundary(
    const FabricMemoryEndpointRef &endpoint) {
  if (endpoint.owner.kind() ==
      FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return llvm::Error::success();
  return systemPhysicalRoleError(
      "a SpatialCore memory boundary must name its exact occurrence");
}

llvm::Error validateFabricSpatialCoreInternalOwner(
    const SpatialCoreInternalOccurrenceRef &occurrence) {
  if (occurrence.target.kind() == FabricModulePhysicalTargetKind::Owner)
    return llvm::Error::success();
  return systemPhysicalRoleError(
      "a physical occurrence owner must select a Module physical owner");
}

llvm::Error validateFabricDirectSystemConfigurationField(
    const FabricSemanticConfigFieldRef &field) {
  return validateFabricDirectSystemOwner(field.owner.catalog());
}

llvm::Error validateFabricSpatialCoreInternalConfigurationField(
    const SpatialCoreInternalOccurrenceRef &occurrence) {
  if (occurrence.target.kind() ==
      FabricModulePhysicalTargetKind::SemanticConfigurationField)
    return llvm::Error::success();
  return systemPhysicalRoleError(
      "a physical configuration field must select a Module semantic field");
}

} // namespace

#define LOOM_FABRIC_MODULE_PHYSICAL_OWNER(Ordinal, Name, Type, Validator)      \
  llvm::Expected<FabricModulePhysicalOwnerRef>                                 \
  FabricModulePhysicalOwnerRef::create(const Type &value) {                    \
    if (llvm::Error error = Validator(value))                                  \
      return std::move(error);                                                 \
    return FabricModulePhysicalOwnerRef(                                       \
        Payload(std::in_place_type<Type>, value));                             \
  }
#include "Fabric/Identity/FabricRefs.def"

#define LOOM_FABRIC_MODULE_PHYSICAL_TARGET(Ordinal, Name, Type, Validator)     \
  llvm::Expected<FabricModulePhysicalTargetRef>                                \
  FabricModulePhysicalTargetRef::create(const Type &value) {                   \
    if (llvm::Error error = Validator(value))                                  \
      return std::move(error);                                                 \
    return FabricModulePhysicalTargetRef(                                      \
        Payload(std::in_place_type<Type>, value));                             \
  }
#include "Fabric/Identity/FabricRefs.def"

#define LOOM_FABRIC_SPATIAL_CORE_DOMAIN_TARGET(Ordinal, Name, Type, Validator) \
  llvm::Expected<SpatialCorePhysicalDomainTargetRef>                           \
  SpatialCorePhysicalDomainTargetRef::create(const Type &value) {              \
    if (llvm::Error error = Validator(value))                                  \
      return std::move(error);                                                 \
    return SpatialCorePhysicalDomainTargetRef(                                 \
        Payload(std::in_place_type<Type>, value));                             \
  }
#include "Fabric/Identity/FabricRefs.def"

#define LOOM_FABRIC_PHYSICAL_OCCURRENCE_OWNER(Ordinal, Name, Type, Validator)  \
  llvm::Expected<FabricPhysicalOccurrenceOwnerRef>                             \
  FabricPhysicalOccurrenceOwnerRef::create(const Type &value) {                \
    if (llvm::Error error = Validator(value))                                  \
      return std::move(error);                                                 \
    return FabricPhysicalOccurrenceOwnerRef(                                   \
        Payload(std::in_place_type<Type>, value));                             \
  }
#include "Fabric/Identity/FabricRefs.def"

#define LOOM_FABRIC_PHYSICAL_CONFIGURATION_FIELD(Ordinal, Name, Type,          \
                                                 Validator)                    \
  llvm::Expected<FabricPhysicalConfigurationFieldRef>                          \
  FabricPhysicalConfigurationFieldRef::create(const Type &value) {             \
    if (llvm::Error error = Validator(value))                                  \
      return std::move(error);                                                 \
    return FabricPhysicalConfigurationFieldRef(                                \
        Payload(std::in_place_type<Type>, value));                             \
  }
#include "Fabric/Identity/FabricRefs.def"

llvm::Expected<FabricPhysicalConfigurationSlotRef>
FabricPhysicalConfigurationSlotRef::create(
    const FabricConfigurationSlotRef &value) {
  auto field = FabricPhysicalConfigurationFieldRef::create(value.field);
  if (!field)
    return field.takeError();
  return FabricPhysicalConfigurationSlotRef(
      Payload(std::in_place_type<FabricConfigurationSlotRef>, value));
}

llvm::Expected<FabricPhysicalConfigurationSlotRef>
FabricPhysicalConfigurationSlotRef::create(
    const SpatialCoreInternalConfigurationSlotRef &value) {
  auto target = FabricModulePhysicalTargetRef::create(value.slot.field);
  if (!target)
    return target.takeError();
  auto field = FabricPhysicalConfigurationFieldRef::create(
      SpatialCoreInternalOccurrenceRef{value.spatialCore, std::move(*target)});
  if (!field)
    return field.takeError();
  return FabricPhysicalConfigurationSlotRef(
      Payload(std::in_place_type<SpatialCoreInternalConfigurationSlotRef>,
              value));
}

llvm::Expected<FabricPhysicalConfigurationSlotRef>
loom::fabric::qualifyFabricConfigurationSlot(
    const FabricPhysicalConfigurationFieldRef &field,
    FabricConfigurationResidency residency) {
  switch (field.kind()) {
  case FabricPhysicalConfigurationFieldKind::DirectSystemField:
    return FabricPhysicalConfigurationSlotRef::create(
        FabricConfigurationSlotRef{
            std::get<FabricSemanticConfigFieldRef>(field.payload()),
            std::move(residency)});
  case FabricPhysicalConfigurationFieldKind::SpatialCoreInternalField: {
    const auto &internal =
        std::get<SpatialCoreInternalOccurrenceRef>(field.payload());
    const auto &local =
        std::get<FabricSemanticConfigFieldRef>(internal.target.payload());
    return FabricPhysicalConfigurationSlotRef::create(
        SpatialCoreInternalConfigurationSlotRef{
            internal.spatialCore,
            FabricConfigurationSlotRef{local, std::move(residency)}});
  }
  }
  llvm_unreachable("unknown physical configuration field kind");
}

FabricPhysicalConfigurationFieldRef loom::fabric::configurationField(
    const FabricPhysicalConfigurationSlotRef &slot) {
  switch (slot.kind()) {
  case FabricPhysicalConfigurationSlotKind::DirectSystemSlot:
    return llvm::cantFail(FabricPhysicalConfigurationFieldRef::create(
        std::get<FabricConfigurationSlotRef>(slot.payload()).field));
  case FabricPhysicalConfigurationSlotKind::SpatialCoreInternalSlot: {
    const auto &internal = std::get<SpatialCoreInternalConfigurationSlotRef>(
        slot.payload());
    auto target = llvm::cantFail(
        FabricModulePhysicalTargetRef::create(internal.slot.field));
    return llvm::cantFail(FabricPhysicalConfigurationFieldRef::create(
        SpatialCoreInternalOccurrenceRef{internal.spatialCore,
                                         std::move(target)}));
  }
  }
  llvm_unreachable("unknown physical configuration slot kind");
}

const FabricConfigurationSlotRef &loom::fabric::configurationSlot(
    const FabricPhysicalConfigurationSlotRef &slot) {
  switch (slot.kind()) {
  case FabricPhysicalConfigurationSlotKind::DirectSystemSlot:
    return std::get<FabricConfigurationSlotRef>(slot.payload());
  case FabricPhysicalConfigurationSlotKind::SpatialCoreInternalSlot:
    return std::get<SpatialCoreInternalConfigurationSlotRef>(slot.payload())
        .slot;
  }
  llvm_unreachable("unknown physical configuration slot kind");
}

#define LOOM_FABRIC_HARDWARE_DOMAIN_MEMBER(Ordinal, Name, Type, Validator)     \
  llvm::Expected<FabricHardwareDomainMemberRef>                                \
  FabricHardwareDomainMemberRef::create(const Type &value) {                   \
    if (llvm::Error error = Validator(value))                                  \
      return std::move(error);                                                 \
    return FabricHardwareDomainMemberRef(                                      \
        Payload(std::in_place_type<Type>, value));                             \
  }
#include "Fabric/Identity/FabricRefs.def"

llvm::Expected<FabricClockResetDirectOwnerRef>
FabricClockResetDirectOwnerRef::create(const FabricInventoryOwnerRef &owner) {
  if (llvm::Error error = validateFabricClockResetDirectOwner(owner))
    return std::move(error);
  return FabricClockResetDirectOwnerRef(owner);
}

// Every keyword table below is a projection of the one catalog declaration.

#define LOOM_FABRIC_ROOT_KIND(Name, Keyword)                                   \
  case FabricRootKind::Name:                                                   \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricRootKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_FU_NODE_KIND(Name, Keyword)                                \
  case FabricFuNodeKind::Name:                                                 \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricFuNodeKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_PORT_DIRECTION(Name, Keyword)                              \
  case FabricPortDirection::Name:                                              \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricPortDirection value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type)                        \
  case FabricMemoryServiceKind::Name:                                          \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricMemoryServiceKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_FIFO_MODE(Name, Keyword)                                   \
  case FabricFifoTraversalMode::Name:                                          \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricFifoTraversalMode value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_REGISTER_FIFO_PATH_ROLE(Name, Keyword)                     \
  case FabricRegisterFifoPathRole::Name:                                       \
    return Keyword;
llvm::StringRef
loom::fabric::fabricRefKeyword(FabricRegisterFifoPathRole value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type)                             \
  case FabricPhysicalTraversalKind::Name:                                      \
    return Keyword;
llvm::StringRef
loom::fabric::fabricRefKeyword(FabricPhysicalTraversalKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_INVENTORY(Name, Keyword)                                   \
  case FabricInventoryKind::Name:                                              \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricInventoryKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_HARDWARE_DOMAIN_KIND(Name, Keyword)                        \
  case FabricHardwareDomainKind::Name:                                         \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricHardwareDomainKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_CLOCK_RESET_KIND(Name, Keyword)                            \
  case FabricClockResetKind::Name:                                             \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricClockResetKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_MEMORY_ENDPOINT_ROLE(Name, Keyword)                        \
  case FabricMemoryEndpointRole::Name:                                         \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricMemoryEndpointRole value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

#define LOOM_FABRIC_REF_ERROR(Name, Keyword)                                   \
  case FabricRefErrorKind::Name:                                               \
    return Keyword;
llvm::StringRef loom::fabric::fabricRefKeyword(FabricRefErrorKind value) {
  switch (value) {
#include "Fabric/Identity/FabricRefs.def"
  }
  llvm_unreachable("closed sum value outside its declaration");
}

llvm::Error loom::fabric::makeFabricRefError(FabricRefErrorKind kind,
                                             const llvm::Twine &message) {
  return llvm::make_error<FabricRefError>(kind, message.str());
}

FabricRefErrorKind loom::fabric::takeFabricRefErrorKind(llvm::Error error) {
  FabricRefErrorKind kind = FabricRefErrorKind::MalformedSyntax;
  llvm::handleAllErrors(
      std::move(error),
      [&](const FabricRefError &typed) { kind = typed.kind(); },
      [](const llvm::ErrorInfoBase &) {});
  return kind;
}

FabricInventoryOwnerRef loom::fabric::projectFabricInventoryOwner(
    const FabricTransportEndpointOwnerRef &owner) {
  return std::visit([](const auto &value) { return inventoryOwnerFor(value); },
                    owner.payload);
}

FabricInventoryOwnerRef loom::fabric::projectFabricInventoryOwner(
    const FabricMemoryEndpointOwnerRef &owner) {
  return std::visit([](const auto &value) { return inventoryOwnerFor(value); },
                    owner.payload);
}
