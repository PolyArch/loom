#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom::fabric;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invalid physical identity");
  llvm::consumeError(value.takeError());
}

template <typename T, typename = void>
struct IsModulePhysicalOwner : std::false_type {};

template <typename T>
struct IsModulePhysicalOwner<
    T, std::void_t<decltype(FabricModulePhysicalOwnerRef::create(
           std::declval<T>()))>> : std::true_type {};

template <typename T, typename = void>
struct IsModuleDomainMember : std::false_type {};

template <typename T>
struct IsModuleDomainMember<
    T,
    std::void_t<decltype(FabricModuleDomainMemberRef::of(std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct IsModulePhysicalTarget : std::false_type {};

template <typename T>
struct IsModulePhysicalTarget<
    T, std::void_t<decltype(FabricModulePhysicalTargetRef::create(
           std::declval<T>()))>> : std::true_type {};

void modulePhysicalOwnersFormOneClosedCatalog() {
  const llvm::StringRef test = __func__;
  static_assert(IsModulePhysicalOwner<FabricPeOccurrenceRef>::value);
  static_assert(IsModulePhysicalOwner<FabricFuOccurrenceRef>::value);
  static_assert(IsModulePhysicalOwner<FabricFuOccurrenceNodeRef>::value);
  static_assert(IsModulePhysicalOwner<FabricMemoryOccurrenceRef>::value);
  static_assert(IsModulePhysicalOwner<FabricMemoryOperationPortRef>::value);
  static_assert(IsModulePhysicalOwner<LocalMemoryServiceRef>::value);
  static_assert(IsModulePhysicalOwner<FabricSwitchOccurrenceRef>::value);
  static_assert(IsModulePhysicalOwner<FabricFifoOccurrenceRef>::value);
  static_assert(IsModulePhysicalOwner<FabricBoundaryOccurrenceRef>::value);
  static_assert(IsModulePhysicalOwner<InstructionContextRef>::value);
  static_assert(!IsModulePhysicalOwner<FabricModuleTemplateRef>::value);
  static_assert(!IsModulePhysicalOwner<SpatialCoreOccurrenceRef>::value);
  static_assert(!IsModulePhysicalOwner<AccCoreOccurrenceRef>::value);
  static_assert(!IsModulePhysicalOwner<FabricMemoryServiceRef>::value);
  static_assert(!IsModulePhysicalOwner<SystemMemoryServiceRef>::value);

  const auto requireOwner = [&](const auto &payload, std::uint8_t tag) {
    const FabricModulePhysicalOwnerRef owner =
        take(test, FabricModulePhysicalOwnerRef::create(payload));
    const std::vector<std::uint8_t> bytes = canonicalFabricBytes(owner);
    require(test,
            bytes.size() >= 4 && bytes[0] == 0 && bytes[1] == 0 &&
                bytes[2] == 0 && bytes[3] == tag,
            "Module physical owner discriminant changed");
    require(test,
            take(test, decodeFabricRef<FabricModulePhysicalOwnerRef>(bytes)) ==
                    owner &&
                take(test, parseFabricRef<FabricModulePhysicalOwnerRef>(
                               printFabricRef(owner))) == owner,
            "Module physical owner codec changed its payload");
    require(test, printFabricRef(owner) == printFabricRef(payload),
            "Module physical owner text duplicated its constructor");
    const std::vector<std::uint8_t> payloadBytes =
        canonicalFabricBytes(payload);
    require(test,
            bytes.size() == payloadBytes.size() + 4 &&
                std::equal(payloadBytes.begin(), payloadBytes.end(),
                           bytes.begin() + 4),
            "Module physical owner bytes changed its payload");
  };

  requireOwner(FabricPeOccurrenceRef(20), 0);
  requireOwner(FabricFuOccurrenceRef(21), 1);
  requireOwner(FabricFuOccurrenceNodeRef{FabricFuNodeKind::Op,
                                         FabricFuOccurrenceRef(21), 2},
               2);
  requireOwner(FabricMemoryOccurrenceRef(22), 3);
  requireOwner(FabricMemoryOperationPortRef{FabricMemoryOccurrenceRef(22), 1},
               4);
  requireOwner(LocalMemoryServiceRef(FabricMemoryServiceRef::local(
                   FabricMemoryOccurrenceRef(22))),
               5);
  requireOwner(FabricSwitchOccurrenceRef(23), 6);
  requireOwner(FabricFifoOccurrenceRef(24), 7);
  requireOwner(FabricBoundaryOccurrenceRef(25), 8);
  requireOwner(InstructionContextRef{FabricPeOccurrenceRef(20), 3}, 9);

  requireRejected(
      test, FabricModulePhysicalOwnerRef::create(LocalMemoryServiceRef(
                FabricMemoryServiceRef::system(SystemMemoryServiceRef(26)))));
  const FabricMemoryServiceRef systemService =
      FabricMemoryServiceRef::system(SystemMemoryServiceRef(26));
  std::vector<std::uint8_t> invalidLocalService = {0, 0, 0, 5};
  const std::vector<std::uint8_t> systemServiceBytes =
      canonicalFabricBytes(systemService);
  invalidLocalService.insert(invalidLocalService.end(),
                             systemServiceBytes.begin(),
                             systemServiceBytes.end());
  requireRejected(
      test, decodeFabricRef<FabricModulePhysicalOwnerRef>(invalidLocalService));
  requireRejected(test, parseFabricRef<FabricModulePhysicalOwnerRef>(
                            printFabricRef(systemService)));

  std::vector<std::uint8_t> unknown = canonicalFabricBytes(take(
      test, FabricModulePhysicalOwnerRef::create(FabricPeOccurrenceRef(20))));
  unknown[3] = 10;
  requireRejected(test, decodeFabricRef<FabricModulePhysicalOwnerRef>(unknown));
  requireRejected(test, parseFabricRef<FabricModulePhysicalOwnerRef>(
                            "fabric.acc_core_occurrence<30>"));
}

void moduleMembersAndTargetsPreserveOneLocalIdentity() {
  const llvm::StringRef test = __func__;
  static_assert(IsModuleDomainMember<FabricModuleBoundaryEndpointRef>::value);
  static_assert(IsModuleDomainMember<FabricModulePhysicalOwnerRef>::value);
  static_assert(!IsModuleDomainMember<FabricPeOccurrenceRef>::value);
  static_assert(!IsModuleDomainMember<SpatialCoreOccurrenceRef>::value);

  const FabricModuleBoundaryEndpointRef boundary{FabricModuleTemplateRef(10),
                                                 FabricPortDirection::Input, 0};
  const FabricModulePhysicalOwnerRef fuOwner = take(
      test, FabricModulePhysicalOwnerRef::create(FabricFuOccurrenceRef(21)));
  const FabricModuleDomainMemberRef boundaryMember =
      FabricModuleDomainMemberRef::of(boundary);
  const FabricModuleDomainMemberRef internalMember =
      FabricModuleDomainMemberRef::of(fuOwner);

  const auto requireMember = [&](const FabricModuleDomainMemberRef &member,
                                 std::uint8_t tag) {
    const std::vector<std::uint8_t> bytes = canonicalFabricBytes(member);
    require(test, bytes.size() >= 4 && bytes[3] == tag,
            "Module domain member discriminant changed");
    require(test,
            take(test, decodeFabricRef<FabricModuleDomainMemberRef>(bytes)) ==
                    member &&
                take(test, parseFabricRef<FabricModuleDomainMemberRef>(
                               printFabricRef(member))) == member,
            "Module domain member codec changed its payload");
  };
  requireMember(boundaryMember, 0);
  requireMember(internalMember, 1);
  std::vector<std::uint8_t> unknownMember =
      canonicalFabricBytes(boundaryMember);
  unknownMember[3] = 2;
  requireRejected(test,
                  decodeFabricRef<FabricModuleDomainMemberRef>(unknownMember));
  requireRejected(test, parseFabricRef<FabricModuleDomainMemberRef>(
                            "fabric.acc_core_occurrence<30>"));

  const ModuleDomainAssignment assignment{
      internalMember,
      FabricModuleDomainSlotRef{FabricModuleTemplateRef(10),
                                FabricClockResetKind::Clock, 1}};
  require(test,
          take(test, decodeFabricRef<ModuleDomainAssignment>(
                         canonicalFabricBytes(assignment))) == assignment &&
              take(test, parseFabricRef<ModuleDomainAssignment>(
                             printFabricRef(assignment))) == assignment,
          "Module domain assignment codec changed its relation");

  const FabricFuOccurrencePortRef fuPort{FabricFuOccurrenceRef(21),
                                         FabricPortDirection::Output, 0};
  const FabricTransportEndpointRef tokenEndpoint{
      FabricTransportEndpointOwnerRef::of(FabricFuOccurrenceRef(21)), 1};
  const FabricMemoryOperationPortRef memoryPort{FabricMemoryOccurrenceRef(22),
                                                0};
  const FabricMemoryEndpointRef memoryEndpoint{
      FabricMemoryEndpointOwnerRef::of(FabricMemoryOccurrenceRef(22)), 1};
  const FabricMemoryCapabilityAlternativeRef memoryAlternative{memoryPort, 2};
  const FabricMemoryOperationContextRef memoryContext{memoryPort, 3};
  const FabricMemoryServiceRef localService =
      FabricMemoryServiceRef::local(FabricMemoryOccurrenceRef(22));
  const FabricMemoryServiceRegionRef memoryRegion{localService, 4};
  const FabricInventoryOwnerRef inventoryOwner =
      FabricInventoryOwnerRef::of(FabricFuOccurrenceRef(21));
  const FabricResourceStateRef state{
      FabricResourceStateOwnerRef(inventoryOwner), 5};
  const FabricUsePatternRef use{FabricUsePatternOwnerRef(inventoryOwner), 6};
  const FabricSemanticConfigFieldRef field{
      FabricConfigurationOwnerRef(inventoryOwner), 7};
  const FabricPhysicalRefinementDomainRef refinement{
      FabricRefinementOwnerRef(inventoryOwner), 8};
  const FabricPhysicalTraversalRef traversal =
      FabricPhysicalTraversalRef::pointConnection(tokenEndpoint, tokenEndpoint);

  static_assert(IsModulePhysicalTarget<FabricModulePhysicalOwnerRef>::value);
  static_assert(IsModulePhysicalTarget<FabricFuOccurrencePortRef>::value);
  static_assert(IsModulePhysicalTarget<FabricTransportEndpointRef>::value);
  static_assert(IsModulePhysicalTarget<FabricMemoryEndpointRef>::value);
  static_assert(
      IsModulePhysicalTarget<FabricMemoryCapabilityAlternativeRef>::value);
  static_assert(IsModulePhysicalTarget<FabricMemoryOperationContextRef>::value);
  static_assert(IsModulePhysicalTarget<FabricMemoryServiceRegionRef>::value);
  static_assert(IsModulePhysicalTarget<FabricResourceStateRef>::value);
  static_assert(IsModulePhysicalTarget<FabricUsePatternRef>::value);
  static_assert(IsModulePhysicalTarget<FabricSemanticConfigFieldRef>::value);
  static_assert(
      IsModulePhysicalTarget<FabricPhysicalRefinementDomainRef>::value);
  static_assert(IsModulePhysicalTarget<FabricPhysicalTraversalRef>::value);
  static_assert(!IsModulePhysicalTarget<FabricModuleTemplateRef>::value);
  static_assert(!IsModulePhysicalTarget<HardwareDomainRef>::value);

  const auto requireTarget = [&](const auto &payload, std::uint8_t tag) {
    const FabricModulePhysicalTargetRef target =
        take(test, FabricModulePhysicalTargetRef::create(payload));
    const std::vector<std::uint8_t> bytes = canonicalFabricBytes(target);
    require(test, bytes.size() >= 4 && bytes[3] == tag,
            "Module physical target discriminant changed");
    require(test,
            take(test, decodeFabricRef<FabricModulePhysicalTargetRef>(bytes)) ==
                    target &&
                take(test, parseFabricRef<FabricModulePhysicalTargetRef>(
                               printFabricRef(target))) == target,
            "Module physical target codec changed its payload");
    require(test, printFabricRef(target) == printFabricRef(payload),
            "Module physical target text duplicated its constructor");
    const std::vector<std::uint8_t> payloadBytes =
        canonicalFabricBytes(payload);
    require(test,
            bytes.size() == payloadBytes.size() + 4 &&
                std::equal(payloadBytes.begin(), payloadBytes.end(),
                           bytes.begin() + 4),
            "Module physical target bytes changed its payload");
  };
  requireTarget(fuOwner, 0);
  requireTarget(fuPort, 1);
  requireTarget(tokenEndpoint, 2);
  requireTarget(memoryEndpoint, 3);
  requireTarget(memoryAlternative, 4);
  requireTarget(memoryContext, 5);
  requireTarget(memoryRegion, 6);
  requireTarget(state, 7);
  requireTarget(use, 8);
  requireTarget(field, 9);
  requireTarget(refinement, 10);
  requireTarget(traversal, 11);

  const FabricModulePhysicalTargetRef localTarget =
      take(test, FabricModulePhysicalTargetRef::create(field));
  const SpatialCoreInternalOccurrenceRef first{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(30)}, localTarget};
  const SpatialCoreInternalOccurrenceRef second{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(31)}, localTarget};
  require(test, canonicalFabricBytes(first) != canonicalFabricBytes(second),
          "internal occurrence identity lost its SpatialCore occurrence");
  require(test,
          take(test, decodeFabricRef<SpatialCoreInternalOccurrenceRef>(
                         canonicalFabricBytes(first))) == first &&
              take(test, parseFabricRef<SpatialCoreInternalOccurrenceRef>(
                             printFabricRef(first))) == first,
          "internal occurrence codec changed its local target");

  std::vector<std::uint8_t> unknown = canonicalFabricBytes(localTarget);
  unknown[3] = 12;
  requireRejected(test,
                  decodeFabricRef<FabricModulePhysicalTargetRef>(unknown));
  requireRejected(test, parseFabricRef<FabricModulePhysicalTargetRef>(
                            "fabric.hardware_domain<40>"));

  const FabricTransportEndpointRef systemTokenEndpoint{
      FabricTransportEndpointOwnerRef::of(SystemServiceEndpointRef(40)), 0};
  const FabricMemoryEndpointRef systemMemoryEndpoint{
      FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(40)), 0};
  const SpatialCoreOccurrenceRef spatialCore{AccCoreOccurrenceRef(42)};
  const FabricTransportEndpointRef spatialTokenEndpoint{
      FabricTransportEndpointOwnerRef::of(spatialCore), 0};
  const FabricMemoryEndpointRef spatialMemoryEndpoint{
      FabricMemoryEndpointOwnerRef::of(spatialCore), 0};
  const FabricMemoryServiceRef systemService =
      FabricMemoryServiceRef::system(SystemMemoryServiceRef(41));
  const FabricInventoryOwnerRef systemOwner =
      FabricInventoryOwnerRef::of(SystemServiceEndpointRef(40));
  const FabricInventoryOwnerRef spatialOwner =
      FabricInventoryOwnerRef::of(spatialCore);

  requireRejected(test,
                  FabricModulePhysicalTargetRef::create(systemTokenEndpoint));
  requireRejected(test,
                  FabricModulePhysicalTargetRef::create(spatialTokenEndpoint));
  requireRejected(test,
                  FabricModulePhysicalTargetRef::create(systemMemoryEndpoint));
  requireRejected(test,
                  FabricModulePhysicalTargetRef::create(spatialMemoryEndpoint));
  requireRejected(test, FabricModulePhysicalTargetRef::create(
                            FabricMemoryServiceRegionRef{systemService, 0}));
  requireRejected(test,
                  FabricModulePhysicalTargetRef::create(FabricResourceStateRef{
                      FabricResourceStateOwnerRef(systemOwner), 0}));
  requireRejected(test,
                  FabricModulePhysicalTargetRef::create(FabricUsePatternRef{
                      FabricUsePatternOwnerRef(spatialOwner), 0}));
  requireRejected(
      test, FabricModulePhysicalTargetRef::create(FabricSemanticConfigFieldRef{
                FabricConfigurationOwnerRef(systemOwner), 0}));
  requireRejected(test, FabricModulePhysicalTargetRef::create(
                            FabricPhysicalRefinementDomainRef{
                                FabricRefinementOwnerRef(spatialOwner), 0}));
  requireRejected(
      test,
      FabricModulePhysicalTargetRef::create(
          FabricPhysicalTraversalRef::transferPatternLeg(
              FabricTransferPatternRef{SystemTransportResourceRef(43), 0}, 0)));
  const FabricPhysicalTraversalRef systemTraversal =
      FabricPhysicalTraversalRef::pointConnection(systemTokenEndpoint,
                                                  systemTokenEndpoint);
  requireRejected(test, FabricModulePhysicalTargetRef::create(systemTraversal));

  std::vector<std::uint8_t> invalidTransport = {0, 0, 0, 2};
  const std::vector<std::uint8_t> systemEndpointBytes =
      canonicalFabricBytes(systemTokenEndpoint);
  invalidTransport.insert(invalidTransport.end(), systemEndpointBytes.begin(),
                          systemEndpointBytes.end());
  requireRejected(
      test, decodeFabricRef<FabricModulePhysicalTargetRef>(invalidTransport));
  requireRejected(test, parseFabricRef<FabricModulePhysicalTargetRef>(
                            printFabricRef(systemTokenEndpoint)));
}

} // namespace

int main() {
  modulePhysicalOwnersFormOneClosedCatalog();
  moduleMembersAndTargetsPreserveOneLocalIdentity();
  llvm::outs() << "fabric physical identity ok\n";
  return EXIT_SUCCESS;
}
