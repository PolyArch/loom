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
  systemServiceEndpointOwnsOperationServicePlanes();
  refinementsAddNoIdentity();
  strictTextLanguage();
  canonicalByteRoundTrip();
  llvm::outs() << "fabric persistent references ok\n";
  return EXIT_SUCCESS;
}
