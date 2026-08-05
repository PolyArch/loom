#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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

void requireKind(llvm::StringRef test, llvm::Error error,
                 FabricRefErrorKind expected) {
  if (!error)
    fail(test, "accepted an invalid persistent reference");
  FabricRefErrorKind actual = takeFabricRefErrorKind(std::move(error));
  require(test, actual == expected,
          "persistent reference failure kind changed");
}

template <typename Ref>
void requireCanonical(llvm::StringRef test, llvm::StringRef spelling) {
  Ref parsed = take(test, parseFabricRef<Ref>(spelling));
  require(test, printFabricRef(parsed) == spelling,
          "canonical text roundtrip changed spelling");
}

template <typename Ref>
void requireParseKind(llvm::StringRef test, llvm::StringRef spelling,
                      FabricRefErrorKind expected) {
  llvm::Expected<Ref> parsed = parseFabricRef<Ref>(spelling);
  if (parsed)
    fail(test, "accepted invalid spelling '" + spelling.str() + "'");
  requireKind(test, parsed.takeError(), expected);
}

template <typename T>
void requireRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted malformed canonical bytes");
  llvm::consumeError(value.takeError());
}

template <typename T, typename = void>
struct IsTransportEndpointOwner : std::false_type {};

template <typename T>
struct IsTransportEndpointOwner<
    T, std::void_t<decltype(FabricTransportEndpointOwnerRef::of(
           std::declval<T>()))>> : std::true_type {};

template <typename T, typename = void>
struct IsMemoryEndpointOwner : std::false_type {};

template <typename T>
struct IsMemoryEndpointOwner<
    T,
    std::void_t<decltype(FabricMemoryEndpointOwnerRef::of(std::declval<T>()))>>
    : std::true_type {};

void typedFamiliesRemainDistinct() {
  const llvm::StringRef test = __func__;
  static_assert(!std::is_same_v<FabricFuTemplateRef, FabricFuOccurrenceRef>);
  static_assert(
      !std::is_convertible_v<FabricFuTemplateRef, FabricFuOccurrenceRef>);
  static_assert(
      !std::is_same_v<FabricTransportEndpointRef, FabricMemoryEndpointRef>);
  static_assert(!std::is_convertible_v<FabricTransportEndpointRef,
                                       FabricMemoryEndpointRef>);
  static_assert(
      !std::is_same_v<FabricPhysicalTraversalRef, FabricResourceStateRef>);
  static_assert(!std::is_same_v<FabricMemoryEngineTemplateRef,
                                FabricMemoryOccurrenceRef>);
  static_assert(!std::is_convertible_v<FabricMemoryEngineTemplateRef,
                                       FabricMemoryOccurrenceRef>);

  require(test,
          canonicalFabricBytes(FabricFuTemplateRef(11)) !=
              canonicalFabricBytes(FabricFuOccurrenceRef(11)),
          "equal ordinals under distinct entity kinds shared identity");

  const FabricSwitchOccurrenceRef sw(21);
  const FabricTransportEndpointRef source{
      FabricTransportEndpointOwnerRef::of(sw), 2};
  const FabricTransportEndpointRef destination{
      FabricTransportEndpointOwnerRef::of(sw), 3};
  const FabricPhysicalTraversalRef switchTraversal =
      FabricPhysicalTraversalRef::switchTraversal(sw, 0, 1);
  const FabricPhysicalTraversalRef connection =
      FabricPhysicalTraversalRef::pointConnection(source, destination);
  const FabricResourceStateRef state{
      FabricResourceStateOwnerRef(FabricInventoryOwnerRef::of(sw)), 1};
  require(test,
          canonicalFabricBytes(switchTraversal) !=
                  canonicalFabricBytes(connection) &&
              canonicalFabricBytes(switchTraversal) !=
                  canonicalFabricBytes(state),
          "traversal, connection, and resource state shared identity");

  const SpatialCoreOccurrenceRef spatial{AccCoreOccurrenceRef(41)};
  const FabricInventoryOwnerRef expected = FabricInventoryOwnerRef::of(spatial);
  require(test,
          projectFabricInventoryOwner(
              FabricTransportEndpointOwnerRef::of(spatial)) == expected &&
              projectFabricInventoryOwner(
                  FabricMemoryEndpointOwnerRef::of(spatial)) == expected,
          "endpoint owner projection changed the spatial-core owner");
}

void memoryEngineTemplateReferencesRoundTrip() {
  const llvm::StringRef test = __func__;
  const FabricMemoryEngineTemplateRef engine(61);
  const FabricMemoryEngineTemplateOperationPortRef port{engine, 2};
  const FabricMemoryEngineTemplateCapabilityAlternativeRef alternative{port, 3};
  const FabricMemoryEngineTemplateEndpointRef source{engine, 4};
  const FabricMemoryEngineTemplateEndpointRef sink{engine, 5};
  const FabricMemoryEngineTemplateInternalConnectionRef connection{
      engine, source, sink};

  const auto roundTrip = [&](const auto &reference) {
    using Ref = std::decay_t<decltype(reference)>;
    const std::vector<std::uint8_t> bytes = canonicalFabricBytes(reference);
    require(test, take(test, decodeFabricRef<Ref>(bytes)) == reference,
            "memory engine template reference byte roundtrip changed value");
    const std::string text = printFabricRef(reference);
    require(test, take(test, parseFabricRef<Ref>(text)) == reference,
            "memory engine template reference text roundtrip changed value");
  };

  roundTrip(engine);
  roundTrip(port);
  roundTrip(alternative);
  roundTrip(source);
  roundTrip(sink);
  roundTrip(connection);
  require(
      test,
      canonicalFabricBytes(connection) !=
          canonicalFabricBytes(FabricMemoryEngineTemplateInternalConnectionRef{
              engine, sink, source}),
      "memory engine template connection lost direction");
}

void systemServiceEndpointOwnsOperationServicePlanes() {
  const llvm::StringRef test = __func__;
  static_assert(IsTransportEndpointOwner<SystemServiceEndpointRef>::value);
  static_assert(IsMemoryEndpointOwner<SystemServiceEndpointRef>::value);
  static_assert(!IsTransportEndpointOwner<AccCoreOccurrenceRef>::value);
  static_assert(!IsTransportEndpointOwner<ExternalBoundaryRef>::value);
  static_assert(!IsMemoryEndpointOwner<AccCoreOccurrenceRef>::value);
  static_assert(!IsMemoryEndpointOwner<SystemMemoryServiceRef>::value);
  static_assert(!IsMemoryEndpointOwner<SystemServiceTransformRef>::value);
  static_assert(!IsMemoryEndpointOwner<ExternalBoundaryRef>::value);

  std::vector<std::uint8_t> transport = canonicalFabricBytes(
      FabricTransportEndpointOwnerRef::of(SystemServiceEndpointRef(30)));
  require(test,
          transport.size() >= 4 && transport[0] == 0 && transport[1] == 0 &&
              transport[2] == 0 && transport[3] == 8,
          "System service transport owner changed its stable discriminant");
  transport[3] = 7;
  llvm::Expected<FabricTransportEndpointOwnerRef> retiredTransport =
      decodeFabricRef<FabricTransportEndpointOwnerRef>(transport);
  if (retiredTransport)
    fail(test, "accepted retired direct AccCore transport ownership");
  requireKind(test, retiredTransport.takeError(),
              FabricRefErrorKind::MalformedSyntax);

  std::vector<std::uint8_t> memory = canonicalFabricBytes(
      FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(30)));
  require(test,
          memory.size() >= 4 && memory[0] == 0 && memory[1] == 0 &&
              memory[2] == 0 && memory[3] == 4,
          "System service memory owner changed its stable discriminant");
  memory[3] = 2;
  llvm::Expected<FabricMemoryEndpointOwnerRef> retiredMemory =
      decodeFabricRef<FabricMemoryEndpointOwnerRef>(memory);
  if (retiredMemory)
    fail(test, "accepted retired direct AccCore memory ownership");
  requireKind(test, retiredMemory.takeError(),
              FabricRefErrorKind::MalformedSyntax);
}

void refinementsAddNoIdentity() {
  const llvm::StringRef test = __func__;
  static_assert(!std::is_same_v<ManagerEndpointRef, SubordinateEndpointRef>);
  static_assert(
      !std::is_convertible_v<FabricMemoryEndpointRef, ManagerEndpointRef>);

  const FabricMemoryServiceRef service =
      FabricMemoryServiceRef::local(FabricMemoryOccurrenceRef(31));
  const LocalMemoryServiceRef local(service);
  require(test,
          canonicalFabricBytes(local) == canonicalFabricBytes(service) &&
              printFabricRef(local) == printFabricRef(service),
          "a typed refinement introduced a second identity");

  const HardwareDomainRef clockDomain(52);
  const HardwareDomainRef resetDomain(53);
  require(test,
          canonicalFabricBytes(ClockDomainRef(clockDomain)) ==
                  canonicalFabricBytes(clockDomain) &&
              canonicalFabricBytes(ResetDomainRef(resetDomain)) ==
                  canonicalFabricBytes(resetDomain),
          "hardware-domain refinements changed owner bytes");
}

void moduleDomainSlotsHaveOneCanonicalIdentity() {
  const llvm::StringRef test = __func__;
  static_assert(!std::is_same_v<FabricModuleDomainSlotRef,
                                SpatialCoreDomainSlotOccurrenceRef>);
  static_assert(!std::is_convertible_v<FabricModuleDomainSlotRef,
                                       SpatialCoreDomainSlotOccurrenceRef>);

  const FabricModuleDomainSlotRef moduleClock{FabricModuleTemplateRef(7),
                                              FabricClockResetKind::Clock, 2};
  const FabricModuleDomainSlotRef moduleReset{FabricModuleTemplateRef(7),
                                              FabricClockResetKind::Reset, 2};
  const SpatialCoreDomainSlotOccurrenceRef occurrenceClock{
      SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(7)},
      FabricClockResetKind::Clock, 2};

  require(test,
          printFabricRef(moduleClock) ==
              "fabric.module_domain_slot<fabric.module_template<7>, clock, 2>",
          "Module domain slot canonical spelling changed");
  require(test,
          printFabricRef(occurrenceClock) ==
              "fabric.spatial_core_domain_slot_occurrence<"
              "fabric.spatial_core_occurrence<"
              "fabric.acc_core_occurrence<7>>, clock, 2>",
          "spatial-core slot occurrence canonical spelling changed");

  const auto moduleBytes = canonicalFabricBytes(moduleClock);
  const auto resetBytes = canonicalFabricBytes(moduleReset);
  const auto occurrenceBytes = canonicalFabricBytes(occurrenceClock);
  require(test, moduleBytes.size() == 24 && occurrenceBytes.size() == 24,
          "domain slot encoding duplicated structural owner facts");
  require(test,
          moduleBytes[12] == 0 && moduleBytes[13] == 0 &&
              moduleBytes[14] == 0 && moduleBytes[15] == 0 &&
              resetBytes[12] == 0 && resetBytes[13] == 0 &&
              resetBytes[14] == 0 && resetBytes[15] == 1,
          "Clock/Reset slot discriminants changed");
  require(test, moduleBytes != occurrenceBytes,
          "Module and occurrence slots shared canonical identity");
  require(test,
          occurrenceBytes !=
              canonicalFabricBytes(SpatialCoreDomainSlotOccurrenceRef{
                  SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(8)},
                  FabricClockResetKind::Clock, 2}),
          "slot occurrence identity lost its owning occurrence");
  require(test,
          take(test, decodeFabricRef<FabricModuleDomainSlotRef>(moduleBytes)) ==
                  moduleClock &&
              take(test, decodeFabricRef<SpatialCoreDomainSlotOccurrenceRef>(
                             occurrenceBytes)) == occurrenceClock,
          "domain slot canonical byte roundtrip changed value");
  require(test,
          take(test, parseFabricRef<FabricModuleDomainSlotRef>(
                         printFabricRef(moduleReset))) == moduleReset,
          "Reset slot canonical text roundtrip changed value");

  std::vector<std::uint8_t> invalidKind = moduleBytes;
  invalidKind[15] = 2;
  requireRejected(test,
                  decodeFabricRef<FabricModuleDomainSlotRef>(invalidKind));
  requireParseKind<FabricModuleDomainSlotRef>(
      test, "fabric.module_domain_slot<fabric.module_template<7>, power, 2>",
      FabricRefErrorKind::MalformedSyntax);
  requireParseKind<SpatialCoreDomainSlotOccurrenceRef>(
      test, "fabric.module_domain_slot<fabric.module_template<7>, clock, 2>",
      FabricRefErrorKind::MalformedSyntax);
}

void strictTextLanguage() {
  const llvm::StringRef test = __func__;
  requireCanonical<FabricFuOccurrenceRef>(test, "fabric.fu_occurrence<11>");
  requireCanonical<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<switch, fabric.switch_occurrence<21>, 0, 1>");
  requireCanonical<FabricMemoryServiceRegionRef>(
      test, "fabric.memory_service_region<fabric.memory_service<local, "
            "fabric.memory_occurrence<31>>, 1>");

  requireParseKind<FabricMemoryEndpointRef>(
      test, "fabric.memory_endpoint<fabric.switch_occurrence<21>, 0>",
      FabricRefErrorKind::PlaneMisuse);
  requireParseKind<FabricTransportEndpointRef>(
      test, "fabric.transport_endpoint<fabric.hardware_domain<51>, 0>",
      FabricRefErrorKind::InvalidOwnerFamily);
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu<11>",
                                          FabricRefErrorKind::DeprecatedAlias);
  requireParseKind<FabricFuOccurrenceRef>(test,
                                          "fabric.fu_occurrence<11>.port[3]",
                                          FabricRefErrorKind::DeprecatedAlias);
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu_template<11>",
                                          FabricRefErrorKind::WrongEntityKind);
  requireParseKind<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<switch, fabric.switch_occurrence<21>, 0>",
      FabricRefErrorKind::MalformedSyntax);
  requireParseKind<FabricFuOccurrenceRef>(
      test, "fabric.fu_occurrence<18446744073709551616>",
      FabricRefErrorKind::MalformedSyntax);
}

void canonicalByteRoundTrip() {
  const llvm::StringRef test = __func__;
  const FabricPhysicalTraversalRef traversal =
      FabricPhysicalTraversalRef::switchTraversal(FabricSwitchOccurrenceRef(21),
                                                  0, 1);
  const std::string text = printFabricRef(traversal);
  const FabricPhysicalTraversalRef parsed =
      take(test, parseFabricRef<FabricPhysicalTraversalRef>(text));
  const std::vector<std::uint8_t> bytes = canonicalFabricBytes(parsed);
  const FabricPhysicalTraversalRef decoded =
      take(test, decodeFabricRef<FabricPhysicalTraversalRef>(bytes));
  require(test, decoded == traversal && printFabricRef(decoded) == text,
          "text and byte codecs did not recover the same reference");

  std::vector<std::uint8_t> extended = bytes;
  extended.push_back(0);
  requireRejected(test, decodeFabricRef<FabricPhysicalTraversalRef>(extended));
  std::vector<std::uint8_t> truncated = bytes;
  truncated.pop_back();
  requireRejected(test, decodeFabricRef<FabricPhysicalTraversalRef>(truncated));

  std::vector<std::uint8_t> unknown =
      canonicalFabricBytes(FabricFuOccurrenceRef(12));
  unknown[3] = 0xff;
  llvm::Expected<FabricFuOccurrenceRef> result =
      decodeFabricRef<FabricFuOccurrenceRef>(unknown);
  if (result)
    fail(test, "accepted an unknown entity-kind discriminant");
  requireKind(test, result.takeError(), FabricRefErrorKind::MalformedSyntax);
}

} // namespace

int main() {
  typedFamiliesRemainDistinct();
  memoryEngineTemplateReferencesRoundTrip();
  systemServiceEndpointOwnsOperationServicePlanes();
  refinementsAddNoIdentity();
  moduleDomainSlotsHaveOneCanonicalIdentity();
  strictTextLanguage();
  canonicalByteRoundTrip();
  llvm::outs() << "fabric persistent references ok\n";
  return EXIT_SUCCESS;
}
