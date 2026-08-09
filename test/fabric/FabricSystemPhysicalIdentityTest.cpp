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
    fail(test, "accepted an invalid System physical identity");
  llvm::consumeError(value.takeError());
}

template <typename Union, typename Payload>
Union requireTaggedUnion(llvm::StringRef test, const Payload &payload,
                         std::uint8_t tag) {
  const Union value = take(test, Union::create(payload));
  const std::vector<std::uint8_t> bytes = canonicalFabricBytes(value);
  const std::vector<std::uint8_t> payloadBytes = canonicalFabricBytes(payload);
  require(test,
          bytes.size() == payloadBytes.size() + 4 && bytes[0] == 0 &&
              bytes[1] == 0 && bytes[2] == 0 && bytes[3] == tag &&
              std::equal(payloadBytes.begin(), payloadBytes.end(),
                         bytes.begin() + 4),
          "closed union changed its tag or payload bytes");
  require(test, printFabricRef(value) == printFabricRef(payload),
          "closed union duplicated its constructor in canonical text");
  require(test,
          take(test, decodeFabricRef<Union>(bytes)) == value &&
              take(test, parseFabricRef<Union>(printFabricRef(value))) == value,
          "closed union codec changed its payload");
  return value;
}

FabricModulePhysicalOwnerRef moduleFuOwner(llvm::StringRef test) {
  return take(test,
              FabricModulePhysicalOwnerRef::create(FabricFuOccurrenceRef(20)));
}

SpatialCoreInternalOccurrenceRef moduleInternalOwner(llvm::StringRef test,
                                                     std::uint64_t core) {
  const FabricModulePhysicalTargetRef target =
      take(test, FabricModulePhysicalTargetRef::create(moduleFuOwner(test)));
  return SpatialCoreInternalOccurrenceRef{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(core)}, target};
}

SpatialCoreInternalOccurrenceRef moduleInternalField(llvm::StringRef test,
                                                     std::uint64_t core) {
  const FabricInventoryOwnerRef owner =
      FabricInventoryOwnerRef::of(FabricFuOccurrenceRef(20));
  const FabricSemanticConfigFieldRef field{FabricConfigurationOwnerRef(owner),
                                           3};
  const FabricModulePhysicalTargetRef target =
      take(test, FabricModulePhysicalTargetRef::create(field));
  return SpatialCoreInternalOccurrenceRef{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(core)}, target};
}

void spatialCoreDomainTargetsAreOccurrenceExact() {
  const llvm::StringRef test = __func__;
  const SpatialCoreOccurrenceRef firstCore{AccCoreOccurrenceRef(10)};
  const SpatialCoreOccurrenceRef secondCore{AccCoreOccurrenceRef(11)};
  const FabricTransportEndpointRef transport{
      FabricTransportEndpointOwnerRef::of(firstCore), 2};
  const FabricMemoryEndpointRef memory{
      FabricMemoryEndpointOwnerRef::of(firstCore), 4};
  const SpatialCoreInternalOccurrenceRef internal =
      moduleInternalOwner(test, 10);

  requireTaggedUnion<SpatialCorePhysicalDomainTargetRef>(test, transport, 0);
  requireTaggedUnion<SpatialCorePhysicalDomainTargetRef>(test, memory, 1);
  const SpatialCorePhysicalDomainTargetRef internalTarget =
      requireTaggedUnion<SpatialCorePhysicalDomainTargetRef>(test, internal, 2);

  const FabricTransportEndpointRef moduleEndpoint{
      FabricTransportEndpointOwnerRef::of(FabricFuOccurrenceRef(20)), 2};
  const FabricMemoryEndpointRef systemEndpoint{
      FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(21)), 4};
  requireRejected(test,
                  SpatialCorePhysicalDomainTargetRef::create(moduleEndpoint));
  requireRejected(test,
                  SpatialCorePhysicalDomainTargetRef::create(systemEndpoint));
  requireRejected(test, parseFabricRef<SpatialCorePhysicalDomainTargetRef>(
                            printFabricRef(moduleEndpoint)));

  SpatialCoreInternalOccurrenceRef otherOccurrence = internal;
  otherOccurrence.spatialCore = secondCore;
  const SpatialCorePhysicalDomainTargetRef otherTarget =
      take(test, SpatialCorePhysicalDomainTargetRef::create(otherOccurrence));
  require(test,
          canonicalFabricBytes(internalTarget) !=
              canonicalFabricBytes(otherTarget),
          "domain target lost its exact SpatialCore occurrence");

  std::vector<std::uint8_t> unknown = canonicalFabricBytes(internalTarget);
  unknown[3] = 3;
  requireRejected(test,
                  decodeFabricRef<SpatialCorePhysicalDomainTargetRef>(unknown));
}

void physicalOwnerAndConfigurationFieldRolesStayDisjoint() {
  const llvm::StringRef test = __func__;
  const FabricInventoryOwnerRef systemOwner =
      FabricInventoryOwnerRef::of(SystemTransportResourceRef(30));
  const FabricInventoryOwnerRef moduleOwner =
      FabricInventoryOwnerRef::of(FabricFuOccurrenceRef(31));
  const FabricInventoryOwnerRef templateOwner =
      FabricInventoryOwnerRef::of(FabricFuTemplateRef(32));

  const auto requireDirectOwner = [&](const FabricInventoryOwnerRef &owner) {
    requireTaggedUnion<FabricPhysicalOccurrenceOwnerRef>(test, owner, 0);
  };
  requireDirectOwner(FabricInventoryOwnerRef::of(HostCoreOccurrenceRef(33)));
  requireDirectOwner(FabricInventoryOwnerRef::of(AccCoreOccurrenceRef(34)));
  requireDirectOwner(FabricInventoryOwnerRef::of(
      InstructionCoreContextRef{AccCoreOccurrenceRef(34)}));
  requireDirectOwner(FabricInventoryOwnerRef::of(
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(34)}));
  requireDirectOwner(FabricInventoryOwnerRef::of(
      FabricMemoryServiceRef::system(SystemMemoryServiceRef(35))));
  requireDirectOwner(FabricInventoryOwnerRef::of(SystemServiceEndpointRef(36)));
  requireDirectOwner(
      FabricInventoryOwnerRef::of(SystemServiceTransformRef(37)));
  requireDirectOwner(systemOwner);
  requireDirectOwner(FabricInventoryOwnerRef::of(
      FabricTransferPatternRef{SystemTransportResourceRef(30), 1}));
  requireDirectOwner(FabricInventoryOwnerRef::of(HardwareDomainRef(38)));
  requireDirectOwner(FabricInventoryOwnerRef::of(ExternalBoundaryRef(39)));

  const SpatialCoreInternalOccurrenceRef internalOwner =
      moduleInternalOwner(test, 12);
  requireTaggedUnion<FabricPhysicalOccurrenceOwnerRef>(test, internalOwner, 1);
  requireRejected(test, FabricPhysicalOccurrenceOwnerRef::create(moduleOwner));
  requireRejected(test,
                  FabricPhysicalOccurrenceOwnerRef::create(templateOwner));
  requireRejected(
      test, FabricPhysicalOccurrenceOwnerRef::create(
                FabricInventoryOwnerRef::of(FabricModuleTemplateRef(40))));
  requireRejected(
      test,
      FabricPhysicalOccurrenceOwnerRef::create(FabricInventoryOwnerRef::of(
          FabricMemoryServiceRef::local(FabricMemoryOccurrenceRef(41)))));

  const SpatialCoreInternalOccurrenceRef internalField =
      moduleInternalField(test, 12);
  requireRejected(test,
                  FabricPhysicalOccurrenceOwnerRef::create(internalField));
  std::vector<std::uint8_t> invalidInternalOwner = {0, 0, 0, 1};
  const std::vector<std::uint8_t> internalFieldBytes =
      canonicalFabricBytes(internalField);
  invalidInternalOwner.insert(invalidInternalOwner.end(),
                              internalFieldBytes.begin(),
                              internalFieldBytes.end());
  requireRejected(test, decodeFabricRef<FabricPhysicalOccurrenceOwnerRef>(
                            invalidInternalOwner));
  requireRejected(test, parseFabricRef<FabricPhysicalOccurrenceOwnerRef>(
                            printFabricRef(internalField)));

  const FabricSemanticConfigFieldRef directField{
      FabricConfigurationOwnerRef(systemOwner), 5};
  const FabricSemanticConfigFieldRef localField{
      FabricConfigurationOwnerRef(moduleOwner), 5};
  requireTaggedUnion<FabricPhysicalConfigurationFieldRef>(test, directField, 0);
  requireTaggedUnion<FabricPhysicalConfigurationFieldRef>(test, internalField,
                                                          1);
  requireRejected(test,
                  FabricPhysicalConfigurationFieldRef::create(localField));
  requireRejected(test,
                  FabricPhysicalConfigurationFieldRef::create(internalOwner));

  std::vector<std::uint8_t> invalidInternalField = {0, 0, 0, 1};
  const std::vector<std::uint8_t> internalOwnerBytes =
      canonicalFabricBytes(internalOwner);
  invalidInternalField.insert(invalidInternalField.end(),
                              internalOwnerBytes.begin(),
                              internalOwnerBytes.end());
  requireRejected(test, decodeFabricRef<FabricPhysicalConfigurationFieldRef>(
                            invalidInternalField));
}

void hardwareDomainMembersUseOneSystemWire() {
  const llvm::StringRef test = __func__;
  const FabricInventoryOwnerRef directOwner =
      FabricInventoryOwnerRef::of(AccCoreOccurrenceRef(40));
  const SpatialCoreDomainSlotOccurrenceRef slot{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(40)},
      FabricClockResetKind::Reset, 2};
  requireTaggedUnion<FabricHardwareDomainMemberRef>(test, directOwner, 0);
  requireTaggedUnion<FabricHardwareDomainMemberRef>(test, slot, 1);

  const FabricInventoryOwnerRef moduleOwner =
      FabricInventoryOwnerRef::of(FabricPeOccurrenceRef(41));
  requireRejected(test, FabricHardwareDomainMemberRef::create(moduleOwner));
  requireRejected(test, parseFabricRef<FabricHardwareDomainMemberRef>(
                            printFabricRef(moduleOwner)));
}

void configurationSlotsPreserveResidencyAndOccurrence() {
  const llvm::StringRef test = __func__;
  const FabricInventoryOwnerRef moduleOwner =
      FabricInventoryOwnerRef::of(FabricFuOccurrenceNodeRef{
          FabricFuNodeKind::Op, FabricFuOccurrenceRef(70), 0});
  const FabricSemanticConfigFieldRef field{
      FabricConfigurationOwnerRef(moduleOwner), 0};
  const FabricConfigurationSlotRef staticSlot{
      field, FabricStaticConfigurationResidency{}};
  const FabricConfigurationSlotRef firstContextSlot{
      field, InstructionContextRef{FabricPeOccurrenceRef(71), 0}};
  const FabricConfigurationSlotRef secondContextSlot{
      field, InstructionContextRef{FabricPeOccurrenceRef(71), 1}};

  require(test,
          canonicalFabricBytes(staticSlot) !=
                  canonicalFabricBytes(firstContextSlot) &&
              canonicalFabricBytes(firstContextSlot) !=
                  canonicalFabricBytes(secondContextSlot),
          "configuration slot residency aliased a distinct storage slot");
  require(test,
          take(test, decodeFabricRef<FabricConfigurationSlotRef>(
                         canonicalFabricBytes(staticSlot))) == staticSlot &&
              take(test, decodeFabricRef<FabricConfigurationSlotRef>(
                             canonicalFabricBytes(firstContextSlot))) ==
                  firstContextSlot,
          "configuration slot codec changed residency");

  const SpatialCoreInternalConfigurationSlotRef firstPhysical{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(72)}, firstContextSlot};
  const SpatialCoreInternalConfigurationSlotRef secondPhysical{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(73)}, firstContextSlot};
  const FabricPhysicalConfigurationSlotRef first =
      take(test, FabricPhysicalConfigurationSlotRef::create(firstPhysical));
  const FabricPhysicalConfigurationSlotRef second =
      take(test, FabricPhysicalConfigurationSlotRef::create(secondPhysical));
  require(test, canonicalFabricBytes(first) != canonicalFabricBytes(second),
          "physical configuration slot lost its SpatialCore occurrence");
  require(test,
          take(test, decodeFabricRef<FabricPhysicalConfigurationSlotRef>(
                         canonicalFabricBytes(first))) == first,
          "physical configuration slot codec changed its payload");
  requireRejected(test, FabricPhysicalConfigurationSlotRef::create(staticSlot));
}

void clockResetDirectOwnerIsOneValidatedRefinement() {
  const llvm::StringRef test = __func__;
  const auto requireAdmitted = [&](const FabricInventoryOwnerRef &owner) {
    const FabricClockResetDirectOwnerRef refined =
        take(test, FabricClockResetDirectOwnerRef::create(owner));
    require(test, refined.underlying() == owner,
            "Clock/Reset refinement changed its inventory owner");
    require(test,
            canonicalFabricBytes(refined) == canonicalFabricBytes(owner) &&
                printFabricRef(refined) == printFabricRef(owner),
            "Clock/Reset refinement added a persistent role tag");
    require(test,
            take(test, decodeFabricRef<FabricClockResetDirectOwnerRef>(
                           canonicalFabricBytes(refined))) == refined &&
                take(test, parseFabricRef<FabricClockResetDirectOwnerRef>(
                               printFabricRef(refined))) == refined,
            "Clock/Reset refinement codec changed its owner");
  };

  requireAdmitted(FabricInventoryOwnerRef::of(HostCoreOccurrenceRef(50)));
  requireAdmitted(FabricInventoryOwnerRef::of(
      InstructionCoreContextRef{AccCoreOccurrenceRef(51)}));
  requireAdmitted(FabricInventoryOwnerRef::of(
      FabricMemoryServiceRef::system(SystemMemoryServiceRef(52))));
  requireAdmitted(FabricInventoryOwnerRef::of(SystemServiceEndpointRef(53)));
  requireAdmitted(FabricInventoryOwnerRef::of(SystemServiceTransformRef(54)));
  requireAdmitted(FabricInventoryOwnerRef::of(SystemTransportResourceRef(55)));
  requireAdmitted(FabricInventoryOwnerRef::of(ExternalBoundaryRef(56)));

  const FabricInventoryOwnerRef accCore =
      FabricInventoryOwnerRef::of(AccCoreOccurrenceRef(60));
  const FabricInventoryOwnerRef spatialCore = FabricInventoryOwnerRef::of(
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(60)});
  const FabricInventoryOwnerRef domain =
      FabricInventoryOwnerRef::of(HardwareDomainRef(61));
  const FabricInventoryOwnerRef transfer = FabricInventoryOwnerRef::of(
      FabricTransferPatternRef{SystemTransportResourceRef(62), 0});
  const FabricInventoryOwnerRef localMemory = FabricInventoryOwnerRef::of(
      FabricMemoryServiceRef::local(FabricMemoryOccurrenceRef(63)));
  requireRejected(test, FabricClockResetDirectOwnerRef::create(accCore));
  requireRejected(test, FabricClockResetDirectOwnerRef::create(spatialCore));
  requireRejected(test, FabricClockResetDirectOwnerRef::create(domain));
  requireRejected(test, FabricClockResetDirectOwnerRef::create(transfer));
  requireRejected(test, FabricClockResetDirectOwnerRef::create(localMemory));
  requireRejected(test, decodeFabricRef<FabricClockResetDirectOwnerRef>(
                            canonicalFabricBytes(accCore)));
  requireRejected(test, parseFabricRef<FabricClockResetDirectOwnerRef>(
                            printFabricRef(spatialCore)));
}

void decoderWorkspaceDefaultsRemainStructurallyValid() {
  const llvm::StringRef test = __func__;
  const SpatialCorePhysicalDomainTargetRef spatialTarget;
  require(test,
          take(test, SpatialCorePhysicalDomainTargetRef::create(
                         std::get<FabricTransportEndpointRef>(
                             spatialTarget.payload()))) == spatialTarget,
          "default spatial target bypassed occurrence admission");

  const FabricPhysicalOccurrenceOwnerRef physicalOwner;
  require(test,
          take(test, FabricPhysicalOccurrenceOwnerRef::create(
                         std::get<FabricInventoryOwnerRef>(
                             physicalOwner.payload()))) == physicalOwner,
          "default physical owner bypassed direct-System admission");

  const FabricPhysicalConfigurationFieldRef physicalField;
  require(test,
          take(test, FabricPhysicalConfigurationFieldRef::create(
                         std::get<FabricSemanticConfigFieldRef>(
                             physicalField.payload()))) == physicalField,
          "default physical field bypassed direct-System admission");

  const FabricHardwareDomainMemberRef domainMember;
  require(test,
          take(test, FabricHardwareDomainMemberRef::create(
                         std::get<FabricInventoryOwnerRef>(
                             domainMember.payload()))) == domainMember,
          "default domain member bypassed direct-System admission");

  const FabricClockResetDirectOwnerRef clockResetOwner;
  require(test,
          take(test, FabricClockResetDirectOwnerRef::create(
                         clockResetOwner.underlying())) == clockResetOwner,
          "default Clock/Reset owner bypassed role admission");
}

} // namespace

int main() {
  spatialCoreDomainTargetsAreOccurrenceExact();
  physicalOwnerAndConfigurationFieldRolesStayDisjoint();
  configurationSlotsPreserveResidencyAndOccurrence();
  hardwareDomainMembersUseOneSystemWire();
  clockResetDirectOwnerIsOneValidatedRefinement();
  decoderWorkspaceDefaultsRemainStructurallyValid();
  llvm::outs() << "fabric system physical identity ok\n";
  return EXIT_SUCCESS;
}
