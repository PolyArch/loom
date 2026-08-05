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
