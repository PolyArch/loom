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

} // namespace

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
